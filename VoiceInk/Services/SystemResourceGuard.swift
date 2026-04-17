import Foundation
import os

/// Lightweight advisory that surfaces the Mac's current thermal / power
/// state so heavyweight local transcription models can warn the user (or
/// route to a cheaper model) before we wedge the ANE / GPU under load.
///
/// This is an *advisory* — callers decide whether to proceed. We deliberately
/// keep this synchronous and allocation-free on the hot path.
enum SystemResourceGuard {
    enum ThermalSeverity: Int, Sendable, Comparable {
        case nominal = 0
        case fair = 1
        case serious = 2
        case critical = 3

        static func < (lhs: ThermalSeverity, rhs: ThermalSeverity) -> Bool {
            lhs.rawValue < rhs.rawValue
        }

        init(_ state: ProcessInfo.ThermalState) {
            switch state {
            case .nominal: self = .nominal
            case .fair: self = .fair
            case .serious: self = .serious
            case .critical: self = .critical
            @unknown default: self = .nominal
            }
        }

        var userFacingLabel: String {
            switch self {
            case .nominal: return "Nominal"
            case .fair: return "Fair"
            case .serious: return "Serious"
            case .critical: return "Critical"
            }
        }
    }

    /// Describes how aggressively a given transcription pipeline taxes the
    /// thermal envelope of an Apple Silicon Mac under sustained dictation.
    enum Workload: Sendable {
        /// ANE-dominated, modest footprint (Parakeet, Apple native Speech).
        case light
        /// Mixed ANE + GPU (WhisperKit large-v3 turbo full precision).
        case moderate
        /// GPU-dominant MLX inference (Voxtral realtime, Cohere Transcribe).
        case heavy
    }

    struct Advisory: Sendable {
        let severity: ThermalSeverity
        let lowPowerMode: Bool
        let shouldWarn: Bool
        let recommendedFallback: Bool
        let message: String?

        /// Present-tense sentence suitable for inline log / notification.
        var logDescription: String {
            var parts = ["thermal=\(severity.userFacingLabel)"]
            if lowPowerMode { parts.append("low-power-mode=on") }
            return parts.joined(separator: " ")
        }
    }

    /// Queries the current state and returns an advisory for the given
    /// workload class. Does *not* block and is safe to call from any queue.
    static func evaluate(workload: Workload = .moderate) -> Advisory {
        let severity = ThermalSeverity(ProcessInfo.processInfo.thermalState)
        let lowPowerMode = ProcessInfo.processInfo.isLowPowerModeEnabled

        // Matrix: how each workload tolerates rising thermal pressure.
        // light    → only warn on .critical
        // moderate → warn on .serious, fallback on .critical
        // heavy    → warn on .fair,    fallback on .serious+
        let warnThreshold: ThermalSeverity
        let fallbackThreshold: ThermalSeverity
        switch workload {
        case .light:
            warnThreshold = .critical
            fallbackThreshold = .critical
        case .moderate:
            warnThreshold = .serious
            fallbackThreshold = .critical
        case .heavy:
            warnThreshold = .fair
            fallbackThreshold = .serious
        }

        let shouldWarn = severity >= warnThreshold || lowPowerMode
        let recommendedFallback = severity >= fallbackThreshold

        let message: String?
        if recommendedFallback {
            message = "Your Mac is thermally stressed (\(severity.userFacingLabel)). Consider switching to Parakeet or Apple Speech for this session — they run cooler."
        } else if severity >= warnThreshold {
            message = "Mac thermal state is \(severity.userFacingLabel). Transcription latency may be elevated."
        } else if lowPowerMode {
            message = "Low Power Mode is on. Transcription will be slower."
        } else {
            message = nil
        }

        return Advisory(
            severity: severity,
            lowPowerMode: lowPowerMode,
            shouldWarn: shouldWarn,
            recommendedFallback: recommendedFallback,
            message: message
        )
    }

    /// Returns true when we should proactively release large on-device
    /// transcription models after an idle period. We only recommend auto-
    /// unload on Macs with <= 16 GB of physical RAM — on larger machines
    /// keeping the model resident wins on latency and doesn't meaningfully
    /// pressure memory.
    static func shouldAutoUnloadIdleModels() -> Bool {
        let physicalMemoryBytes = ProcessInfo.processInfo.physicalMemory
        let sixteenGiB: UInt64 = 16 * 1024 * 1024 * 1024
        return physicalMemoryBytes <= sixteenGiB
    }

    /// How long to wait after the last transcription before we consider
    /// local models eligible for auto-unload. 10 minutes balances "user
    /// stepped away for a meeting" against "user is in flow, reuse model".
    static let idleUnloadThreshold: TimeInterval = 10 * 60

    /// Convenience for the VoiceInk engine: classifies a transcription
    /// provider into a workload class.
    static func workload(for provider: ModelProvider) -> Workload {
        switch provider {
        case .nativeApple, .parakeet:
            return .light
        case .local:
            return .moderate
        case .localVoxtral, .cohereTranscribe:
            return .heavy
        case .elevenLabs, .custom:
            // Network-bound: negligible local thermal impact.
            return .light
        }
    }
}
