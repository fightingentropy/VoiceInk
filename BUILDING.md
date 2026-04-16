# Building VoiceInk

This guide covers local development and local signed installs of VoiceInk.

## Prerequisites

Before you begin, ensure you have:
- macOS 14.4 or later
- Xcode (latest version recommended)
- Swift command line tools

VoiceInk resolves its dependencies through Swift Package Manager; speech runtimes (WhisperKit, FluidAudio/Parakeet, Voxtral/Cohere MLX) are vendored as SPM packages under `Dependencies/` and build as part of the Xcode project. There is no separate external speech-runtime build step.

## Quick Start With Makefile

```bash
git clone https://github.com/fightingentropy/VoiceInk.git
cd VoiceInk

# Build the app in Debug
make build

# Or build a locally signed Release app and install it to /Applications
make local
```

### Available Makefile Commands

- `make check` or `make healthcheck` - Verify required tools are installed
- `make resolve-packages` - Resolve Swift package dependencies
- `make build` - Build the VoiceInk Xcode project in Debug
- `make local` - Build a local signed Release app and install it to `/Applications/VoiceInk.app`
- `make run` - Launch the built app
- `make dev` - Build and run
- `make all` - Run the default build flow
- `make clean` - Remove local build artifacts
- `make help` - Show all available commands

## Local Signed Install

If you do not have an Apple Developer certificate, use `make local`:

```bash
git clone https://github.com/fightingentropy/VoiceInk.git
cd VoiceInk
make local
open /Applications/VoiceInk.app
```

This builds VoiceInk with a self-signed local `VoiceInk` identity using `LocalBuild.xcconfig`, then replaces `/Applications/VoiceInk.app` with a clean deletion-aware sync.

If you do not already have the local signing identity, create it with:

```bash
scripts/create_local_codesigning_identity.sh
```

That creates a self-signed local code-signing identity named `VoiceInk` in your login keychain.

### How It Works

The `make local` flow uses:
- `LocalBuild.xcconfig` to override signing and entitlements settings
- `VoiceInk.local.entitlements` for the local install configuration
- the `LOCAL_BUILD` Swift compilation flag for local-only code paths
- a stable bundle identifier and stable install path for better macOS permission continuity

## Manual Build Process

If you prefer Xcode directly:

1. Clone the repository:

```bash
git clone https://github.com/fightingentropy/VoiceInk.git
cd VoiceInk
```

2. Resolve Swift packages:

```bash
xcodebuild -project VoiceInk.xcodeproj -resolvePackageDependencies
```

3. Open `VoiceInk.xcodeproj` in Xcode or build from the command line:

```bash
xcodebuild -project VoiceInk.xcodeproj -scheme VoiceInk -configuration Debug build
```

## Model Runtime Notes

- The built-in Whisper preset now uses native `WhisperKit + Core ML`.
- Parakeet uses local Core ML via FluidAudio.
- Voxtral and Cohere use native MLX paths.
- Downloadable model assets are fetched by the app at runtime and stored under VoiceInk's Application Support directory.

## Development Setup

1. **Xcode Configuration**
   - Use a current Xcode release
   - Install command line tools if needed

2. **Dependencies**
   - Run `xcodebuild -resolvePackageDependencies` or let Xcode resolve packages automatically
   - Build once before testing local model downloads

3. **Testing**
   - Run the test suite before making changes
   - Ensure all tests pass after your modifications

## Troubleshooting

If you encounter build issues:

1. Clean the build folder in Xcode
2. Resolve packages again with `xcodebuild -resolvePackageDependencies`
3. Check Xcode and macOS versions
4. Delete `.codex-build/local-install` if a local install build is stale

For more help, check the [issues](https://github.com/fightingentropy/VoiceInk/issues) section or create a new issue.

## Signing And Updates

For a signed distribution build under your own ownership:

1. Open the project in Xcode and select your Apple Developer team in `Signing & Capabilities`.
2. Keep or change the bundle identifiers now set under `com.fightingentropy.*` if you want a different reverse-DNS namespace.
3. Create your own Sparkle EdDSA key pair, then add your public key to `VoiceInk/Info.plist`.
4. Publish your signed DMG and update `appcast.xml` before re-enabling automatic update checks.
5. Push `announcements.json` from your repo if you want in-app announcement banners.

For the automated GitHub Actions release path, see [RELEASING.md](RELEASING.md).
