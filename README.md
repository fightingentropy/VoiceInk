# VoiceInk

<div align="center">
  <img src="VoiceInk/Assets.xcassets/AppIcon.appiconset/256-mac.png" width="180" height="180" />
  <p><strong>Fast, local-first voice-to-text for macOS.</strong></p>
  <p>Press a hotkey, speak, get text at the cursor. Transcription runs on-device; optional cloud providers, AI enhancement, and context-aware Power Modes are all opt-in.</p>

  [![License](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
  ![Platform](https://img.shields.io/badge/platform-macOS%2014.4%2B-brightgreen)
  ![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-required%20for%20local%20MLX-orange)
</div>

---

## Overview

VoiceInk is a native macOS menu-bar app that captures your voice, transcribes it with a local model (or a cloud API if you configure one), optionally rewrites it with an LLM, and pastes the final text at your cursor. It targets the workflow where dictation replaces typing inside any app — editors, terminals, chat windows, email, browsers, etc.

This fork is a personal, non-commercial build:

- **No license/trial/paywall** — all purchase, activation, and Polar integration code has been removed.
- **No auto-updater** — Sparkle is completely removed. Builds are produced by running `make local`.
- **No telemetry or remote calls** other than the cloud transcription / LLM providers you explicitly configure.

If you want a signed, auto-updating distribution channel, that infrastructure is no longer present in this repo (see [What's different in this fork](#whats-different-in-this-fork)).

---

## Features

- 🎙️ **Local transcription** — WhisperKit (Whisper Large v3 Turbo), Voxtral Realtime (MLX), Parakeet v2/v3 (FluidAudio), Apple Speech, and Cohere Transcribe (MLX). All run on-device on Apple Silicon.
- ☁️ **Optional cloud models** — ElevenLabs Scribe and any OpenAI-compatible transcription endpoint.
- 🧠 **AI enhancement (optional)** — Post-transcription rewrites via OpenAI / Anthropic / local LLMs via [LLMkit](https://github.com/Beingpax/LLMkit), with your own prompt templates.
- ⚡ **Power Mode** — Per-app / per-URL configurations auto-apply when the frontmost window matches, with custom system prompts and an "auto-Enter" option for chat UIs.
- 🔥 **Prewarm & warm retention** — Local models are preloaded on launch and kept in memory for a configurable idle window (5 min – "until quit") so the first transcription is instant.
- 📝 **Personal dictionary & word replacement** — Custom vocabulary and text substitutions, with case-insensitive matching and CJK/Thai-aware boundary handling. Stored in SwiftData.
- 🎯 **Global hotkeys** — Customizable push-to-talk or toggle shortcuts via [KeyboardShortcuts](https://github.com/sindresorhus/KeyboardShortcuts), including modifier-only hotkeys (e.g. hold Right Option).
- 🧪 **Benchmark suite** — Run WER/CER benchmarks against a synthetic corpus or your own recent recordings; exports JSON + Markdown reports.
- 🗂️ **History & retention** — Searchable transcript history, automatic audio/transcription cleanup with configurable retention.
- 🤐 **Privacy-first** — Local models never transmit audio. Cloud providers are only hit when you enable them.

---

## Requirements

- macOS **14.4** or later
- **Apple Silicon** (M-series) for local MLX paths (Voxtral, Cohere). Intel Macs will work with Whisper / Parakeet / Apple Speech / cloud providers only, though performance will suffer.
- Xcode 16+ to build from source (latest stable recommended)
- Microphone + Accessibility permission (required to insert text at the cursor)
- Optional: Screen Recording permission (used only if you opt in to context-aware AI enhancement)

---

## Install

### Build and install locally (recommended for this fork)

```bash
git clone https://github.com/fightingentropy/VoiceInk.git
cd VoiceInk
scripts/create_local_codesigning_identity.sh   # first time only
make local
```

`make local` builds a Release app with a stable local codesigning identity named `VoiceInk`, then rsync-installs it to `/Applications/VoiceInk.app` and launches it. The identity is self-signed in your login keychain so macOS treats re-installs as the same app (preserving microphone / accessibility grants).

See [BUILDING.md](BUILDING.md) for more details and a plain `xcodebuild` path.

### Makefile targets

| Command | What it does |
| --- | --- |
| `make build` | Debug build in Xcode's default location |
| `make local` | Release build → sign → install to `/Applications/VoiceInk.app` → launch |
| `make run` | Open `/Applications/VoiceInk.app` (or the Debug build) |
| `make dev` | `make build` then `make run` |
| `make resolve-packages` | Resolve Swift packages only |
| `make bump-version` | Increment marketing/build versions in the pbxproj |
| `make clean` | Remove `.codex-build/` |
| `make healthcheck` | Verify `xcodebuild`, `swift`, `rsync` are installed |

---

## Quick Start

1. Launch VoiceInk. It lives in the menu bar (look for the waveform glyph).
2. Walk through onboarding: grant Microphone and Accessibility, download at least one local model (Whisper Large v3 Turbo is the recommended starting point).
3. Set a recording hotkey (e.g. hold Right Option).
4. Focus any text field, hold the hotkey, speak, release. Text appears at your cursor.
5. Explore:
   - **Metrics** tab — transcription stats
   - **History** tab — searchable past transcripts
   - **Models** tab — download / switch transcription models, run benchmarks
   - **AI Enhancement** — plug in an LLM provider and create prompts
   - **Power Mode** — per-app overrides
   - **Dictionary** — custom vocabulary + word replacements

---

## Architecture

### High-level flow

```
Hotkey  →  Recorder  →  Transcription Pipeline  ──▶  Paste at cursor
                              │
                              ├── Transcribe (local or cloud)
                              ├── Output filter (strip artifacts)
                              ├── Text formatter (capitalization, punctuation)
                              ├── Word replacement (dictionary substitutions)
                              ├── Prompt detection (auto-enable AI mode)
                              ├── AI enhancement (LLM rewrite, optional)
                              └── Persist (SwiftData + audio file)
```

The orchestrator is [`TranscriptionPipeline`](VoiceInk/Whisper/TranscriptionPipeline.swift). It runs after the recorder finishes capturing audio (or, for streaming models, consumes a live session). Each stage is optional and gated by a UserDefaults flag.

### Project layout

```
VoiceInk/
├── VoiceInk.swift              # @main entry point, App + Scene graph
├── AppDelegate.swift           # Activation policy, re-open, termination guards
├── Recorder.swift              # Controls recording lifecycle
├── CoreAudioRecorder.swift     # Low-level CoreAudio capture
├── HotkeyManager.swift         # Global hotkey registration
├── MenuBarManager.swift        # MenuBarExtra controller
├── CursorPaster.swift          # Inserts text via AppleScript / CGEvent
├── SoundManager.swift          # Plays start / stop tones
├── ClipboardManager.swift      # Snapshot + restore clipboard around pastes
├── WindowManager.swift         # Main window + settings panels
├── Whisper/
│   ├── TranscriptionPipeline.swift   # Stage-by-stage post-recording pipeline
│   ├── VoiceInkEngine.swift          # High-level recording orchestrator
│   ├── WhisperKitRuntime.swift       # WhisperKit bindings
│   ├── WhisperModelManager.swift     # Download + manage Whisper assets
│   ├── TranscriptionModelManager.swift
│   └── RecorderUIManager.swift       # Mini-recorder UI state
├── Services/
│   ├── TranscriptionServiceRegistry.swift
│   ├── LocalTranscriptionService.swift
│   ├── AIEnhancement/        # LLM rewrite + prompt system
│   ├── CloudTranscription/   # ElevenLabs + OpenAI-compatible
│   ├── CohereTranscribe/     # Native MLX Cohere runtime
│   ├── Voxtral/              # Native MLX Voxtral runtime
│   ├── StreamingTranscription/
│   ├── Benchmark/            # WER/CER suite
│   ├── WordReplacementService.swift
│   ├── ModelPrewarmService.swift
│   ├── TranscriptionAutoCleanupService.swift
│   ├── SystemInfoService.swift
│   └── …
├── PowerMode/                # Context-aware config system
├── Models/                   # SwiftData models + presets
├── Notifications/            # NotificationCenter names
├── Resources/Sounds/         # recstart.mp3, recstop.mp3, esc.wav
├── Views/                    # All SwiftUI UI
└── Assets.xcassets/          # App icon + menu bar template icon
```

### Entry point & lifecycle

[`VoiceInk.swift`](VoiceInk/VoiceInk.swift) performs a dependency-ordered cold start:

1. Register `AppDefaults` and migrate legacy storage via `AppStoragePaths`.
2. Build the SwiftData `ModelContainer` (two stores: transcripts + dictionary). Falls back to in-memory if the on-disk store can't be opened.
3. Instantiate services in order: `AIService` → `AIEnhancementService` → model managers → `RecorderUIManager` → `VoiceInkEngine` → `HotkeyManager` → `MenuBarManager` → `ModelPrewarmService`.
4. Wire circular dependencies (engine ↔ recorder UI, menu bar ↔ engine).
5. Publish everything as `@StateObject` and inject into the SwiftUI environment.
6. Kick off async boot tasks: model cache load, benchmark corpus bootstrap, auto-cleanup, audio trim.

The scene graph is a hidden-titlebar `WindowGroup` + `MenuBarExtra` + a DEBUG-only utility window.

### Transcription pipeline stages

Order matches [`TranscriptionPipeline.run()`](VoiceInk/Whisper/TranscriptionPipeline.swift):

1. **Transcribe** — `TranscriptionSession.transcribe(audioURL:)` (streaming models) or `TranscriptionServiceRegistry.transcribe(audioURL:model:)` (batch).
2. **Output filter** — `TranscriptionOutputFilter.filter` strips whisper-specific artifacts like `[BLANK_AUDIO]`.
3. **Text formatter** — `WhisperTextFormatter.format` applies capitalization and punctuation cleanup (gated on `IsTextFormattingEnabled`).
4. **Word replacement** — `WordReplacementService.applyReplacements` swaps tokens using your dictionary.
5. **Prompt detection** — `PromptDetectionService` inspects the text and can auto-enable AI enhancement with a specific prompt.
6. **AI enhancement** — `AIEnhancementService.enhance` rewrites via your chosen LLM if `isEnhancementEnabled` and the service is configured. Failures fall back to the un-enhanced text.
7. **Paste** — `CursorPaster.pasteAtCursor` inserts the text, optionally appending a trailing space and pressing Enter (when the active Power Mode has auto-send enabled).
8. **Persist** — Background `Task.detached` writes a `Transcription` record (with metadata: durations, models used, prompt name, power mode, audio URL) to SwiftData and captures it into the benchmark corpus if that's enabled.

### Local runtimes

| Family | Location | Notes |
| --- | --- | --- |
| **WhisperKit** — Whisper Large v3 Turbo | [`Whisper/WhisperKitRuntime.swift`](VoiceInk/Whisper/WhisperKitRuntime.swift) | Core ML on Apple Silicon, model downloaded to App Support. |
| **Voxtral Realtime** | [`Services/Voxtral/`](VoiceInk/Services/Voxtral) | Native MLX + Tekken tokenizer. Streaming-only (mini-realtime). |
| **Parakeet v2 / v3** | [`Services/ParakeetTranscriptionService.swift`](VoiceInk/Services/ParakeetTranscriptionService.swift) + FluidAudio | Local Core ML; batch + streaming. |
| **Apple Speech** | [`Services/NativeAppleTranscriptionService.swift`](VoiceInk/Services/NativeAppleTranscriptionService.swift) | Uses the system speech framework. |
| **Cohere Transcribe** | [`Services/CohereTranscribe/`](VoiceInk/Services/CohereTranscribe) | Native MLX path. Defaults to 4-bit quantization. Live-recorder only — imported audio still routes to Whisper / Parakeet / Apple Speech / cloud. |

All local models download on demand. Assets live under `~/Library/Application Support/VoiceInk/models/` (or `~/Library/Containers/…` for sandboxed builds).

### Cloud providers

Implemented in [`Services/CloudTranscription/`](VoiceInk/Services/CloudTranscription):

- **ElevenLabs Scribe v2** — Streaming transcription via server-sent events.
- **OpenAI-compatible** — Any endpoint that exposes `/audio/transcriptions` with a `whisper-1`-style contract. Configure base URL + API key under Settings → Models → Custom.

When a cloud model is selected, audio leaves your device. Nothing is sent otherwise.

### AI enhancement

[`AIEnhancementService`](VoiceInk/Services/AIEnhancement/AIEnhancementService.swift) wraps LLMkit and exposes:

- A list of custom prompts (stored as JSON in UserDefaults).
- Optional context capture: clipboard and screen (screen requires explicit user opt-in and the Screen Recording permission).
- A 1-second rate limit between requests.
- `lastSystemMessageSent` / `lastUserMessageSent` captured for the transcript record — so you can always audit exactly what the LLM saw.

Providers supported via LLMkit include OpenAI, Anthropic, Cohere, Gemini, Groq, xAI, Ollama, and other OpenAI-compatible endpoints.

### Power Mode

[`PowerMode/`](VoiceInk/PowerMode) lets you define configurations that activate when:

- A target application is frontmost, **or**
- (Browsers only) A target URL is open in Safari / Chrome / Arc via [`BrowserURLService`](VoiceInk/PowerMode/BrowserURLService.swift).

Each config can override:

- AI enhancement on/off + prompt
- "Auto-send Enter" after paste (useful for chat apps)
- Custom system prompt
- Custom emoji / name for the menu bar indicator

[`PowerModeManager`](VoiceInk/PowerMode/PowerModeManager.swift) picks the most specific match; [`PowerModeSessionManager`](VoiceInk/PowerMode/PowerModeSessionManager.swift) applies and restores settings around a single transcription.

### Prewarm & warm retention

[`ModelPrewarmService`](VoiceInk/Services/ModelPrewarmService.swift) preloads the active local model by running inference on a short bundled clip (`Resources/Sounds/esc.wav`). Triggers:

- 3s after cold launch
- 3s after wake from sleep
- 500ms after a model change (debounced)
- Always-on for Cohere (its startup cost is highest)

Retention policy (Settings → Models → "Keep model warm"):

- 5 min / 15 min / 30 min / 1 hour / Until quit

Unload is rescheduled on every transcription and skipped while recording is active.

### Hotkeys

[`HotkeyManager`](VoiceInk/HotkeyManager.swift) registers:

- `toggleMiniRecorder`, `toggleMiniRecorder2` — two independent recorder hotkeys
- `pasteLastTranscription`, `pasteLastEnhancement` — re-insert last output
- `retryLastTranscription` — re-run the pipeline on the last audio file
- `openHistoryWindow`

Hotkey modes:

- **Push-to-talk** (hold)
- **Toggle** (press to start, press to stop)
- **Modifier-only shortcuts** — Right Option, Left Option, Right Control, Left Control, Fn, Right Command, Right Shift, or custom combo
- **Middle-click to toggle** (optional, with activation delay)
- 0.5s keyboard cooldown to debounce fast presses

### Dictionary & word replacement

[`WordReplacementService`](VoiceInk/Services/WordReplacementService.swift):

- Singleton, fetches enabled `WordReplacement` rows from SwiftData each call.
- Accepts comma-separated source variants per rule.
- Case-insensitive.
- Uses regex word boundaries for spaced languages, falls back to substring replacement for CJK / Thai / Hiragana / Katakana (no word-boundary concept).
- Applied post-formatting, pre-enhancement.

Related: [`Models/VocabularyWord.swift`](VoiceInk/Models/VocabularyWord.swift), [`Services/DictionaryImportExportService.swift`](VoiceInk/Services/DictionaryImportExportService.swift) (CSV import/export), and [`Services/CustomVocabularyService.swift`](VoiceInk/Services/CustomVocabularyService.swift) (injects custom terms into model prompts).

### Persistence

[`Models/Transcription.swift`](VoiceInk/Models/Transcription.swift) is the main SwiftData `@Model`:

- Two `ModelConfiguration`s: one for transcripts, one for dictionary/vocabulary data. Keeping them separate lets the dictionary sync to iCloud (on signed builds only) while transcripts stay local.
- Persistent store lives under `~/Library/Application Support/VoiceInk/`.
- Graceful in-memory fallback if the on-disk store fails to open.

Auto-cleanup:

- [`TranscriptionAutoCleanupService`](VoiceInk/Services/TranscriptionAutoCleanupService.swift) — soft-deletes transcripts per retention (30/7/1 days or "never"), with one backup before deletion.
- [`AudioCleanupManager`](VoiceInk/Services/AudioCleanupManager.swift) — removes old audio files so the store doesn't grow unbounded.

### Benchmark suite

[`Services/Benchmark/`](VoiceInk/Services/Benchmark) provides an in-app suite (Settings → Models → Benchmarks):

- Runs against on-device model families (Whisper / Voxtral / Parakeet / Apple Speech / Cohere). Cloud models are excluded because network latency distorts the signal.
- Two corpus sources:
  - **Standard corpus** — fixed synthetic phrases, generated once and reused for consistent cross-model comparison.
  - **Recent recordings** — your last N completed recordings, captured into an app-managed benchmark directory. The "accuracy" here is agreement with the saved transcript, not neutral ground truth.
- Reports are written as JSON + Markdown under the app's benchmark directory.

### Menu bar icon & sounds

Taken from SuperWhisper's assets for visual/audio consistency:

- `VoiceInk/Assets.xcassets/menuBarIcon.imageset/` — 22×22 / 44×44 / 66×66 monochrome template PNGs derived from SuperWhisper's `Icon` asset. `template-rendering-intent: template` makes macOS auto-tint for light/dark menu bars.
- `VoiceInk/Resources/Sounds/recstart.mp3` + `recstop.mp3` — transcoded from SuperWhisper's `Start1.m4a` / `Stop1.m4a` with ffmpeg (`-codec:a libmp3lame -qscale 2`).

---

## Performance optimizations

Notable changes that are already in the codebase (see `git log --oneline` for commits):

- **Native MLX Cohere path** — Replaced the old Python/PyTorch bridge with a pure Swift MLX runtime. Reduced startup, eliminated the Hugging Face token requirement, and made 4-bit the default.
- **Quantize-safe module registration** — Cohere `Linear` and `Embedding` fields are annotated `@ModuleInfo` so MLX's `quantize` pass can walk the module tree correctly. `lm_head` and dtype probes were re-routed through quantize-safe paths.
- **Correct MLX-native tensor layouts** — Subsampling convolutions now use MLX's `(out, kernel, in)` order (not PyTorch's `(out, in, kernel)`), fixing shape mismatches during the encoder's first conv stack.
- **Download progress via KVO** — `URLSessionDownloadTask` progress is tracked via Foundation's `Progress` KVO so the UI shows accurate per-file percentages for multi-asset MLX bundles (Cohere, Voxtral).
- **Keep-alive during downloads** — A process-level activity token prevents App Nap / sudden termination while a model is downloading.
- **Startup-cost trimming** — Removed redundant meter polling, skipped unnecessary PCM decode on the hot path, and tightened model-load logging.
- **Prewarm with ESC-50 clip** — First-transcription latency dropped by preloading the model on launch and wake.
- **4-bit quantized Cohere default** — Lower memory and faster first token on 8 GB / 16 GB Macs.
- **Pinned runtime versions** — FluidAudio and WhisperKit pinned to known-good revisions to avoid regression drift.
- **Benchmark corpus cap** — Recent recordings corpus is capped at 20 entries to keep runs fast and reproducible.

---

## Dependencies

Resolved via Swift Package Manager (`VoiceInk.xcodeproj/project.xcworkspace/xcshareddata/swiftpm/Package.resolved`):

| Package | Purpose |
| --- | --- |
| [WhisperKit](https://github.com/argmaxinc/WhisperKit) | Native Whisper + Core ML pipeline |
| [mlx-swift](https://github.com/ml-explore/mlx-swift) | MLX GPU runtime (Cohere, Voxtral) |
| [FluidAudio](https://github.com/FluidInference/FluidAudio) | Parakeet v2 / v3 backend |
| [KeyboardShortcuts](https://github.com/sindresorhus/KeyboardShortcuts) | User-customizable global hotkeys |
| [LaunchAtLogin-Modern](https://github.com/sindresorhus/LaunchAtLogin-Modern) | Start-on-login |
| [MediaRemoteAdapter](https://github.com/ejbills/mediaremote-adapter) | Pause/resume media around recording |
| [SelectedTextKit](https://github.com/tisfeng/SelectedTextKit) | Capture selected text for context |
| [LLMkit](https://github.com/Beingpax/LLMkit) | LLM provider abstraction (OpenAI, Anthropic, Cohere, Gemini, Groq, xAI, Ollama, …) |
| [swift-atomics](https://github.com/apple/swift-atomics) | Lock-free counters in hot paths |
| [swift-transformers](https://github.com/huggingface/swift-transformers) | Tokenization for MLX models |

Removed from this fork: **Sparkle** (auto-updater).

---

## Privacy & data flow

- **Local models** send nothing over the network after their initial asset download.
- **Cloud transcription** sends audio to ElevenLabs or your configured OpenAI-compatible endpoint — only when a cloud model is the active transcription model.
- **AI enhancement** sends the transcript text (plus optional clipboard/screen context that you explicitly enable) to whichever LLM provider you configure in LLMkit.
- **Screen context** is opt-in and requires the Screen Recording permission. It is used only when AI enhancement is enabled with `useScreenCaptureContext` on.
- **Storage** is local SwiftData + audio files under `~/Library/Application Support/VoiceInk/`. The dictionary store is replicated to iCloud on signed builds only (disabled for local / `LOCAL_BUILD`).
- **No analytics, telemetry, crash reporting, or update pings** are sent. There is no backend.

---

## What's different in this fork

Removed from upstream:

- **License / trial / paywall system** — `LicenseViewModel`, `LicenseManager`, `PolarService`, `LicenseView`, `LicenseManagementView`, `TrialMessageView`, `DashboardPromotionsSection` all deleted; Polar.sh integration gone; `licenseStatusChanged` notification gone.
- **Auto-updater** — `Sparkle` SwiftPM package, `UpdaterViewModel`, `CheckForUpdatesView`, the `SUFeedURL` / `SUEnableAutomaticChecks` / `SUEnableInstallerLauncherService` / `SUPublicEDKey` keys, the "Check for updates" menu item, and `scripts/build_sparkle_release.sh` / `.github/workflows/sparkle-release.yml` are all gone.
- **Help & Resources dashboard section** — removed from the Metrics view.

Replaced:

- **Menu bar icon** — swapped to a template-rendered icon derived from SuperWhisper.
- **Start / stop tones** — swapped to SuperWhisper's recording sounds.

Net effect: the app launches, runs, transcribes, and persists with zero network traffic unless you explicitly configure a cloud model or LLM.

---

## Development

### Swift compilation flags

- `LOCAL_BUILD` — set by `LocalBuild.xcconfig` during `make local`. Disables CloudKit sync (no iCloud entitlement in `VoiceInk.local.entitlements`).

### Local codesigning identity

`make local` signs with a stable self-signed identity called `VoiceInk`. Create it once:

```bash
scripts/create_local_codesigning_identity.sh
```

Using the same identity across rebuilds means macOS treats each install as the same app and preserves microphone / accessibility permissions.

### Entitlements

- [`VoiceInk/VoiceInk.entitlements`](VoiceInk/VoiceInk.entitlements) — full entitlements (CloudKit, push notifications) for signed-distribution builds.
- [`VoiceInk/VoiceInk.local.entitlements`](VoiceInk/VoiceInk.local.entitlements) — minimal entitlements for local builds (no CloudKit, no provisioning profile required).

### Debugging

- Logs go to `os.Logger` under the `com.fightingentropy.voiceink` subsystem. Stream them from `Console.app` or with:
  ```bash
  log stream --predicate 'subsystem == "com.fightingentropy.voiceink"' --level debug
  ```
- The DEBUG build shows a second window with a "Use Menu Bar Only" toggle.

### Tests

- `VoiceInkTests/` contains correctness tests for local runtimes (`LocalTranscriptionHotPathTests.swift`, `CohereNativeSmokeTests.swift`, `VoxtralNativeStreamingSmokeTests.swift`) and benchmark metrics (`BenchmarkMetricsTests.swift`, `LocalOnDeviceBenchmarkTests.swift`).
- Run from Xcode (`⌘U`) or `xcodebuild test -project VoiceInk.xcodeproj -scheme VoiceInk`.

---

## Storage paths

| What | Where |
| --- | --- |
| SwiftData transcripts | `~/Library/Application Support/VoiceInk/default.store` |
| SwiftData dictionary | `~/Library/Application Support/VoiceInk/dictionary.store` |
| Downloaded model assets | `~/Library/Application Support/VoiceInk/models/` |
| Recorded audio files | `~/Library/Application Support/VoiceInk/recordings/` |
| Benchmark corpus | `~/Library/Application Support/VoiceInk/benchmarks/` |
| Local build output | `.codex-build/local-install/deriveddata/` |

---

## Contributing

This repo is a personal fork, but PRs that improve correctness, performance, or documentation are welcome. Please:

1. Open an issue first to discuss scope.
2. Follow the existing Swift style (default Xcode format, no extra docstrings unless non-obvious).
3. Include tests for new runtime / pipeline logic.
4. Keep feature additions opt-in and local-first.

See [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

---

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).

---

## Acknowledgments

- [WhisperKit](https://github.com/argmaxinc/WhisperKit) — Native Whisper + Core ML for Apple Silicon
- [FluidAudio](https://github.com/FluidInference/FluidAudio) — Parakeet integration
- [mlx-swift](https://github.com/ml-explore/mlx-swift) — GPU runtime for Cohere / Voxtral
- [LLMkit](https://github.com/Beingpax/LLMkit) — LLM provider abstraction
- [KeyboardShortcuts](https://github.com/sindresorhus/KeyboardShortcuts), [LaunchAtLogin-Modern](https://github.com/sindresorhus/LaunchAtLogin-Modern), [MediaRemoteAdapter](https://github.com/ejbills/mediaremote-adapter), [SelectedTextKit](https://github.com/tisfeng/SelectedTextKit), [swift-atomics](https://github.com/apple/swift-atomics)
- [SuperWhisper](https://superwhisper.com) — Menu bar icon and recording sound design
- The original VoiceInk project, which this fork is based on
