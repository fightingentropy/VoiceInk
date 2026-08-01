# Building VoiceInk

This guide covers local development builds, `/Applications` installs, tests, and the fail-closed Developer ID distribution path. Sparkle is not part of this fork.

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
cp Signing.example.mk Signing.local.mk
# Set LOCAL_CODESIGN_IDENTITY in Signing.local.mk to the identity you created.

# Debug build for development (opens in Xcode's default DerivedData location)
make build

# Release build → install → launch at /Applications/VoiceInk.app
make local
```

## Makefile targets

| Command | What it does |
| --- | --- |
| `make check` | Run the mandatory deterministic unit and streaming lifecycle suites |
| `make healthcheck` | Verify `xcodebuild`, `swift`, `rsync` are installed |
| `make resolve-packages` | Resolve Swift package dependencies |
| `make build` | Debug build via `xcodebuild` |
| `make local` | Release build → sign with local `VoiceInk` identity → rsync to `/Applications/VoiceInk.app` → launch |
| `make dist` | Require Developer ID + notary configuration, build, sign, notarize, staple, assess, and archive |
| `make run` | Open the installed app, or the Debug build if no install exists |
| `make dev` | `make build && make run` |
| `make bump-version` | Increment marketing/build version numbers in `project.pbxproj` |
| `make clean` | Remove `.codex-build/` |
| `make help` | Print all targets |

## How `make local` works

`make local` uses:

- [`LocalBuild.xcconfig`](LocalBuild.xcconfig) — Sets manual signing, clears team/provisioning-profile requirements, and defines the `LOCAL_BUILD` Swift compilation flag. It contains no account-specific identity.
- `Signing.local.mk` — An ignored, machine-local copy of [`Signing.example.mk`](Signing.example.mk) that supplies `LOCAL_CODESIGN_IDENTITY`.
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

Then configure the ignored local file:

```bash
cp Signing.example.mk Signing.local.mk
```

Set `LOCAL_CODESIGN_IDENTITY` in that file to the exact certificate name reported by `security find-identity -v -p codesigning`. Personal signing identifiers never need to be committed.

## Developer ID distribution

Distribution settings must come from ignored `Signing.local.mk` values or CI environment variables:

- `DIST_CODESIGN_IDENTITY`
- `DIST_TEAM_ID`
- `NOTARY_PROFILE`

Create the named notary profile separately with `xcrun notarytool store-credentials`, then run `make dist`. The target validates the identity and notary profile before building. It publishes no archive unless notarization, stapling, signature verification, and Gatekeeper assessment all succeed; a failure removes the incomplete distribution output.

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

If you want to produce a signed distribution with an Apple Developer account, supply the release settings through the ignored local file or CI environment. Any additional capabilities still require matching entitlements and provisioning in your own developer account.

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
| `make local` reports that `LOCAL_CODESIGN_IDENTITY` is required | Copy `Signing.example.mk` to `Signing.local.mk`, set the exact local identity, then retry. |
| `make dist` stops before building | Configure all three distribution variables and a working notary keychain profile; the target intentionally fails closed. |
| App launches but gets killed with a `CODESIGNING` crash log | Re-run `make local` — the bundle needs to be re-signed as a whole after any modification inside `/Applications/VoiceInk.app`. |
| Build error: "requires a provisioning profile with the iCloud and Push Notifications features" | You're building Release without `LocalBuild.xcconfig`. Use `make local` or pass `-xcconfig LocalBuild.xcconfig` to `xcodebuild`. |
| Stale local install build | `make clean && make local` |
| Xcode package resolution hangs | `rm -rf ~/Library/Developer/Xcode/DerivedData/VoiceInk-*`, then `make resolve-packages`. |

For bugs unrelated to the build, open an issue on GitHub.
