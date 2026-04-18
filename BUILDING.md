# Building VoiceInk

This guide covers local development builds and self-signed `/Applications` installs. There is **no signed-distribution or auto-update path** in this fork — Sparkle has been removed.

## Prerequisites

- macOS **14.4** or later (Apple Silicon strongly recommended for local MLX runtimes)
- **Xcode 16+** (latest stable is safest)
- Swift + `xcodebuild` command-line tools (`xcode-select --install`)
- `rsync` (ships with macOS)

VoiceInk pulls all of its runtimes (WhisperKit, FluidAudio/Parakeet, mlx-swift for Voxtral/Cohere) as Swift packages. There is no separate external build step.

## Quick start

```bash
git clone https://github.com/fightingentropy/VoiceInk.git
cd VoiceInk

# First-time only: create the self-signed local codesigning identity
scripts/create_local_codesigning_identity.sh

# Debug build for development (opens in Xcode's default DerivedData location)
make build

# Release build → install → launch at /Applications/VoiceInk.app
make local
```

## Makefile targets

| Command | What it does |
| --- | --- |
| `make check` / `make healthcheck` | Verify `xcodebuild`, `swift`, `rsync` are installed |
| `make resolve-packages` | Resolve Swift package dependencies |
| `make build` | Debug build via `xcodebuild` |
| `make local` | Release build → sign with local `VoiceInk` identity → rsync to `/Applications/VoiceInk.app` → launch |
| `make run` | Open the installed app, or the Debug build if no install exists |
| `make dev` | `make build && make run` |
| `make bump-version` | Increment marketing/build version numbers in `project.pbxproj` |
| `make clean` | Remove `.codex-build/` |
| `make help` | Print all targets |

## How `make local` works

`make local` uses:

- [`LocalBuild.xcconfig`](LocalBuild.xcconfig) — Overrides signing to use the self-signed local identity, sets manual signing, clears any `DEVELOPMENT_TEAM` / provisioning-profile requirements, and defines the `LOCAL_BUILD` Swift compilation flag.
- [`VoiceInk/VoiceInk.local.entitlements`](VoiceInk/VoiceInk.local.entitlements) — Minimal entitlements (no iCloud, no push notifications) so the build does not require an Apple Developer provisioning profile.
- A stable bundle identifier + a stable install path (`/Applications/VoiceInk.app`) so macOS treats rebuilds as the same app and preserves mic / accessibility grants.
- `rsync -aE --delete` — Replaces the installed bundle atomically and strips stale files.
- `xattr -cr` on the installed bundle — Removes the quarantine flag Gatekeeper sometimes sets.

If the `VoiceInk` codesigning identity isn't present, `make local` fails early with a list of available identities.

### Create the local codesigning identity (first time)

```bash
scripts/create_local_codesigning_identity.sh
```

That script generates a self-signed certificate named `VoiceInk` in your login keychain and marks it as trusted for code signing. You only need to do this once per machine.

### Compilation flags

- `LOCAL_BUILD` — Set by `LocalBuild.xcconfig`. Swift code can use `#if LOCAL_BUILD` for local-only paths (e.g. skipping CloudKit init).

## Manual / Xcode path

If you'd rather work from Xcode or call `xcodebuild` yourself:

```bash
git clone https://github.com/fightingentropy/VoiceInk.git
cd VoiceInk

# Resolve SPM dependencies
xcodebuild -project VoiceInk.xcodeproj -resolvePackageDependencies

# Open in Xcode
open VoiceInk.xcodeproj

# Or Debug-build from the CLI
xcodebuild -project VoiceInk.xcodeproj -scheme VoiceInk -configuration Debug build
```

If you want to produce your own signed build with an Apple Developer account, you'll need to re-enable iCloud / Push Notifications capabilities on the target and supply your team + provisioning profile. No automation is provided for that path in this fork.

## Model runtime notes

- Whisper assets use `WhisperKit + Core ML`. Downloaded to `~/Library/Application Support/VoiceInk/models/`.
- Parakeet uses FluidAudio + Core ML.
- Voxtral and Cohere use the native MLX path (`mlx-swift`). Both download assets on demand; progress is surfaced in the Models UI.
- Apple Speech uses the system Speech framework and needs no download.

All model downloads are local; nothing is telemetered.

## Development setup

1. Open `VoiceInk.xcodeproj` in Xcode.
2. Let Xcode resolve packages (or run `xcodebuild -resolvePackageDependencies`).
3. Build once before testing the model downloaders — SPM package resources need to be in place.
4. For microphone / accessibility testing, use the `/Applications` install (`make local`) rather than the DerivedData Debug build so macOS preserves permissions.
5. `log stream --predicate 'subsystem == "com.fightingentropy.voiceink"' --level debug` tails logs.

## Tests

```bash
xcodebuild test -project VoiceInk.xcodeproj -scheme VoiceInk -destination 'platform=macOS'
```

Or `⌘U` in Xcode. See [`VoiceInkTests/`](VoiceInkTests/) for the available targets — local transcription hot-path tests, benchmark metrics tests, and Cohere/Voxtral smoke tests.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `make local` fails with "Missing local code signing identity" | Run `scripts/create_local_codesigning_identity.sh`, then retry. |
| App launches but gets killed with a `CODESIGNING` crash log | Re-run `make local` — the bundle needs to be re-signed as a whole after any modification inside `/Applications/VoiceInk.app`. |
| Build error: "requires a provisioning profile with the iCloud and Push Notifications features" | You're building Release without `LocalBuild.xcconfig`. Use `make local` or pass `-xcconfig LocalBuild.xcconfig` to `xcodebuild`. |
| Stale local install build | `make clean && make local` |
| Xcode package resolution hangs | `rm -rf ~/Library/Developer/Xcode/DerivedData/VoiceInk-*`, then `make resolve-packages`. |

For bugs unrelated to the build, open an issue on GitHub.
