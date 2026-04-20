import SwiftUI
import SwiftData
import AppKit
import OSLog
import AppIntents
import FluidAudio
import Security

@main
struct VoiceInkApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    let container: ModelContainer
    let containerInitializationFailed: Bool

    @StateObject private var engine: VoiceInkEngine
    @StateObject private var whisperModelManager: WhisperModelManager
    @StateObject private var parakeetModelManager: ParakeetModelManager
    @StateObject private var transcriptionModelManager: TranscriptionModelManager
    @StateObject private var recorderUIManager: RecorderUIManager
    @StateObject private var hotkeyManager: HotkeyManager
    @StateObject private var menuBarManager: MenuBarManager
    @AppStorage("hasCompletedOnboarding") private var hasCompletedOnboarding = false
    @AppStorage("enableAnnouncements") private var enableAnnouncements = true
    @State private var showMenuBarIcon = true

    // Audio cleanup manager for automatic deletion of old audio files
    private let audioCleanupManager = AudioCleanupManager.shared

    // Transcription auto-cleanup service for zero data retention
    private let transcriptionAutoCleanupService = TranscriptionAutoCleanupService.shared

    // Model prewarm service for optimizing model on wake from sleep
    @StateObject private var prewarmService: ModelPrewarmService

    init() {
        // Signpost the full sync init so cold-launch time can be captured in
        // Instruments (Points of Interest instrument → com.fightingentropy.voiceink.launch).
        let launchSignpost = OSSignposter(subsystem: "com.fightingentropy.voiceink.launch", category: .pointsOfInterest)
        let initState = launchSignpost.beginInterval("app-init")
        defer { launchSignpost.endInterval("app-init", initState) }

        let launchClock = ContinuousClock()
        let initStartedAt = launchClock.now

        AppDefaults.registerDefaults()

        let logger = Logger(subsystem: "com.fightingentropy.voiceink", category: "Initialization")
        do {
            try AppStoragePaths.migrateLegacyApplicationSupportIfNeeded()
        } catch {
            logger.error("❌ Failed to migrate legacy application support data: \(error.localizedDescription, privacy: .public)")
        }
        let schema = Schema([
            Transcription.self,
            WordReplacement.self
        ])
        var initializationFailed = false

        // Attempt 1: Try persistent storage
        if let persistentContainer = Self.createPersistentContainer(schema: schema, logger: logger) {
            container = persistentContainer
        }
        // Attempt 2: Try in-memory storage
        else if let memoryContainer = Self.createInMemoryContainer(schema: schema, logger: logger) {
            container = memoryContainer

            logger.warning("Using in-memory storage as fallback. Data will not persist between sessions.")

            // Show alert to user about storage issue
            DispatchQueue.main.async {
                let alert = NSAlert()
                alert.messageText = "Storage Warning"
                alert.informativeText = "VoiceInk couldn't access its storage location. Your transcriptions will not be saved between sessions."
                alert.alertStyle = .warning
                alert.addButton(withTitle: "OK")
                alert.runModal()
            }
        }
        // All attempts failed
        else {
            logger.critical("ModelContainer initialization failed")
            initializationFailed = true

            // Create minimal in-memory container to satisfy initialization
            let config = ModelConfiguration(schema: schema, isStoredInMemoryOnly: true)
            container = (try? ModelContainer(for: schema, configurations: [config])) ?? {
                preconditionFailure("Unable to create ModelContainer. SwiftData is unavailable.")
            }()
        }

        containerInitializationFailed = initializationFailed

        // 1. Create model managers
        let whisperModelManager = WhisperModelManager()
        let parakeetModelManager = ParakeetModelManager()
        let transcriptionModelManager = TranscriptionModelManager(
            whisperModelManager: whisperModelManager,
            parakeetModelManager: parakeetModelManager
        )

        // 2. Create UI manager
        let recorderUIManager = RecorderUIManager()

        // 3. Create engine
        let engine = VoiceInkEngine(
            modelContext: container.mainContext,
            whisperModelManager: whisperModelManager,
            transcriptionModelManager: transcriptionModelManager
        )

        // 4. Configure circular deps
        recorderUIManager.configure(engine: engine, recorder: engine.recorder)
        engine.recorderUIManager = recorderUIManager

        // 5. Publish managers now; populate their caches asynchronously after the
        // window appears so disk enumeration doesn't block first paint.
        _whisperModelManager = StateObject(wrappedValue: whisperModelManager)
        _parakeetModelManager = StateObject(wrappedValue: parakeetModelManager)
        _transcriptionModelManager = StateObject(wrappedValue: transcriptionModelManager)
        _recorderUIManager = StateObject(wrappedValue: recorderUIManager)
        _engine = StateObject(wrappedValue: engine)

        // 7. Create other services that depend on engine
        let hotkeyManager = HotkeyManager(engine: engine, recorderUIManager: recorderUIManager)
        _hotkeyManager = StateObject(wrappedValue: hotkeyManager)

        let menuBarManager = MenuBarManager()
        _menuBarManager = StateObject(wrappedValue: menuBarManager)
        menuBarManager.configure(modelContainer: container, engine: engine)

        let prewarmService = ModelPrewarmService(engine: engine)
        _prewarmService = StateObject(wrappedValue: prewarmService)

        appDelegate.menuBarManager = menuBarManager

        // Ensure no lingering recording state from previous runs
        Task {
            await recorderUIManager.resetOnLaunch()
        }

        // Populate model caches after init returns so directory enumeration and
        // UserDefaults lookups don't stall first paint on cold launch.
        Task { @MainActor in
            whisperModelManager.createModelsDirectoryIfNeeded()
            whisperModelManager.loadAvailableModels()
            transcriptionModelManager.refreshAllAvailableModels()
            transcriptionModelManager.loadCurrentTranscriptionModel()
        }

        AppShortcuts.updateAppShortcutParameters()

        let benchmarkModelContext = container.mainContext
        Task { @MainActor in
            await RecentBenchmarkCorpusService.shared.bootstrapIfNeeded(modelContext: benchmarkModelContext)
        }

        // Start cleanup service for the app's lifetime, not tied to window lifecycle
        TranscriptionAutoCleanupService.shared.startMonitoring(modelContext: container.mainContext)

        let initElapsed = launchClock.now - initStartedAt
        let elapsedComps = initElapsed.components
        let initElapsedMs =
            Double(elapsedComps.seconds) * 1_000 +
            Double(elapsedComps.attoseconds) / 1_000_000_000_000_000
        logger.notice("🚀 VoiceInkApp.init completed in \(String(format: "%.1f", initElapsedMs), privacy: .public) ms")
    }

    // MARK: - Container Creation Helpers

    private static func createPersistentContainer(schema: Schema, logger: Logger) -> ModelContainer? {
        do {
            // Create app-specific Application Support directory URL
            let appSupportURL = AppStoragePaths.applicationSupportDirectory

            // Create the directory if it doesn't exist
            try? AppStoragePaths.createDirectoryIfNeeded(at: appSupportURL)

            // Define storage locations
            let defaultStoreURL = AppStoragePaths.defaultStoreURL
            let dictionaryStoreURL = AppStoragePaths.dictionaryStoreURL

            // Transcript configuration
            let transcriptSchema = Schema([Transcription.self])
            let transcriptConfig = ModelConfiguration(
                "default",
                schema: transcriptSchema,
                url: defaultStoreURL,
                cloudKitDatabase: .none
            )

            // Dictionary configuration
            let dictionarySchema = Schema([WordReplacement.self])
            let dictionaryCloudKit = dictionaryCloudKitDatabase(logger: logger)
            let dictionaryConfig = ModelConfiguration(
                "dictionary",
                schema: dictionarySchema,
                url: dictionaryStoreURL,
                cloudKitDatabase: dictionaryCloudKit
            )

            // Initialize container
            return try ModelContainer(
                for: schema,
                configurations: transcriptConfig, dictionaryConfig
            )
        } catch {
            logger.error("❌ Failed to create persistent ModelContainer: \(error.localizedDescription, privacy: .public)")
            return nil
        }
    }

    private static func createInMemoryContainer(schema: Schema, logger: Logger) -> ModelContainer? {
        do {
            // Transcript configuration
            let transcriptSchema = Schema([Transcription.self])
            let transcriptConfig = ModelConfiguration(
                "default",
                schema: transcriptSchema,
                isStoredInMemoryOnly: true
            )

            // Dictionary configuration
            let dictionarySchema = Schema([WordReplacement.self])
            let dictionaryConfig = ModelConfiguration(
                "dictionary",
                schema: dictionarySchema,
                isStoredInMemoryOnly: true
            )

            return try ModelContainer(for: schema, configurations: transcriptConfig, dictionaryConfig)
        } catch {
            logger.error("❌ Failed to create in-memory ModelContainer: \(error.localizedDescription, privacy: .public)")
            return nil
        }
    }

    private static func dictionaryCloudKitDatabase(logger: Logger) -> ModelConfiguration.CloudKitDatabase {
        #if LOCAL_BUILD || OPEN_SOURCE_DISTRIBUTION
        return .none
        #else
        let containerIdentifier = "iCloud.com.fightingentropy.VoiceInk"

        guard runtimeCloudKitIsAvailable(requiredContainerIdentifier: containerIdentifier, logger: logger) else {
            return .none
        }

        return .private(containerIdentifier)
        #endif
    }

    private static func runtimeCloudKitIsAvailable(requiredContainerIdentifier: String, logger: Logger) -> Bool {
        guard let task = SecTaskCreateFromSelf(nil) else {
            logger.warning("CloudKit disabled because the current code signature could not be inspected.")
            return false
        }

        let services = SecTaskCopyValueForEntitlement(task, "com.apple.developer.icloud-services" as CFString, nil) as? [String]
        let containers = SecTaskCopyValueForEntitlement(task, "com.apple.developer.icloud-container-identifiers" as CFString, nil) as? [String]

        guard services?.contains("CloudKit") == true,
              containers?.contains(requiredContainerIdentifier) == true else {
            logger.notice("CloudKit disabled because the runtime signature does not expose the required iCloud entitlements.")
            return false
        }

        guard let teamIdentifier = signingTeamIdentifier(),
              !teamIdentifier.isEmpty else {
            logger.notice("CloudKit disabled because the runtime signature has no team identifier.")
            return false
        }

        return true
    }

    private static func signingTeamIdentifier() -> String? {
        guard let executableURL = Bundle.main.executableURL else {
            return nil
        }

        var staticCode: SecStaticCode?
        guard SecStaticCodeCreateWithPath(executableURL as CFURL, [], &staticCode) == errSecSuccess,
              let staticCode else {
            return nil
        }

        var signingInfo: CFDictionary?
        guard SecCodeCopySigningInformation(staticCode, SecCSFlags(rawValue: kSecCSSigningInformation), &signingInfo) == errSecSuccess,
              let info = signingInfo as? [String: Any] else {
            return nil
        }

        return info[kSecCodeInfoTeamIdentifier as String] as? String
    }

    var body: some Scene {
        WindowGroup {
            if hasCompletedOnboarding {
                ContentView()
                    .environmentObject(engine)
                    .environmentObject(whisperModelManager)
                    .environmentObject(parakeetModelManager)
                    .environmentObject(transcriptionModelManager)
                    .environmentObject(recorderUIManager)
                    .environmentObject(hotkeyManager)
                    .environmentObject(menuBarManager)
                    .modelContainer(container)
                    .onAppear {
                        // Check if container initialization failed
                        if containerInitializationFailed {
                            let alert = NSAlert()
                            alert.messageText = "Critical Storage Error"
                            alert.informativeText = "VoiceInk cannot initialize its storage system. The app cannot continue.\n\nPlease try reinstalling the app or contact support if the issue persists."
                            alert.alertStyle = .critical
                            alert.addButton(withTitle: "Quit")
                            alert.runModal()

                            NSApplication.shared.terminate(nil)
                            return
                        }

                        // Migrate dictionary data from UserDefaults to SwiftData (one-time operation)
                        DictionaryMigrationService.shared.migrateIfNeeded(context: container.mainContext)

                        if enableAnnouncements {
                            AnnouncementsService.shared.start()
                        }

                        // Start the automatic audio cleanup process only if transcript cleanup is not enabled
                        if !UserDefaults.standard.bool(forKey: "IsTranscriptionCleanupEnabled") {
                            audioCleanupManager.startAutomaticCleanup(modelContext: container.mainContext)
                        }

                        // Process any pending open-file request now that the main ContentView is ready.
                        if let pendingURL = appDelegate.pendingOpenFileURL {
                            NotificationCenter.default.post(name: .navigateToDestination, object: nil, userInfo: ["destination": "Transcribe Audio"])
                            DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                                NotificationCenter.default.post(name: .openFileForTranscription, object: nil, userInfo: ["url": pendingURL])
                            }
                            appDelegate.pendingOpenFileURL = nil
                        }
                    }
                    .background(WindowAccessor { window in
                        WindowManager.shared.configureWindow(window)
                    })
                    .onDisappear {
                        AnnouncementsService.shared.stop()
                        prewarmService.handleWindowDidDisappear()

                        // Stop the automatic audio cleanup process
                        audioCleanupManager.stopAutomaticCleanup()
                    }
            } else {
                OnboardingView(hasCompletedOnboarding: $hasCompletedOnboarding)
                    .environmentObject(hotkeyManager)
                    .environmentObject(engine)
                    .environmentObject(whisperModelManager)
                    .environmentObject(parakeetModelManager)
                    .environmentObject(transcriptionModelManager)
                    .environmentObject(recorderUIManager)
                    .frame(minWidth: 880, minHeight: 780)
                    .background(WindowAccessor { window in
                        if window.identifier == nil || window.identifier != NSUserInterfaceItemIdentifier("com.fightingentropy.voiceink.onboardingWindow") {
                            WindowManager.shared.configureOnboardingPanel(window)
                        }
                    })
            }
        }
        .windowStyle(.hiddenTitleBar)
        .defaultSize(width: 950, height: 730)
        .windowResizability(.contentSize)
        .commands {
            CommandGroup(replacing: .newItem) { }
        }

        MenuBarExtra(isInserted: $showMenuBarIcon) {
            MenuBarView()
                .environmentObject(engine)
                .environmentObject(whisperModelManager)
                .environmentObject(parakeetModelManager)
                .environmentObject(transcriptionModelManager)
                .environmentObject(recorderUIManager)
                .environmentObject(hotkeyManager)
                .environmentObject(menuBarManager)
        } label: {
            let image: NSImage = {
                let ratio = $0.size.height / $0.size.width
                $0.size.height = 22
                $0.size.width = 22 / ratio
                return $0
            }(NSImage(named: "menuBarIcon")!)

            Image(nsImage: image)
        }
        .menuBarExtraStyle(.menu)

        #if DEBUG
        WindowGroup("Debug") {
            Button("Toggle Menu Bar Only") {
                menuBarManager.isMenuBarOnly.toggle()
            }
        }
        #endif
    }
}

struct WindowAccessor: NSViewRepresentable {
    let callback: (NSWindow) -> Void

    func makeNSView(context: Context) -> NSView {
        let view = NSView()
        DispatchQueue.main.async {
            if let window = view.window {
                callback(window)
            }
        }
        return view
    }

    func updateNSView(_ nsView: NSView, context: Context) {}
}
