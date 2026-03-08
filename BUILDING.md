# Building VoiceInk

This guide provides detailed instructions for building VoiceInk from source.

## Prerequisites

Before you begin, ensure you have:
- macOS 14.4 or later
- Xcode (latest version recommended)
- Swift (latest version recommended)
- Git (for cloning repositories)

## Quick Start with Makefile (Recommended)

The easiest way to build VoiceInk is using the included Makefile, which automates the entire build process including building and linking the whisper framework.

### Simple Build Commands

```bash
# Clone the repository
git clone https://github.com/fightingentropy/VoiceInk.git
cd VoiceInk

# Build everything (recommended for first-time setup)
make all

# Or for development (build and run)
make dev
```

### Available Makefile Commands

- `make check` or `make healthcheck` - Verify all required tools are installed
- `make whisper` - Clone and build whisper.cpp XCFramework automatically
- `make setup` - Prepare the whisper framework for linking
- `make build` - Build the VoiceInk Xcode project
- `make local` - Build for local use (no Apple Developer certificate needed)
- `make run` - Launch the built VoiceInk app
- `make dev` - Build and run (ideal for development workflow)
- `make all` - Complete build process (default)
- `make clean` - Remove build artifacts and dependencies
- `make help` - Show all available commands

### How the Makefile Helps

The Makefile automatically:
1. **Manages Dependencies**: Creates a dedicated `~/VoiceInk-Dependencies` directory for all external frameworks
2. **Builds Whisper Framework**: Clones whisper.cpp and builds the XCFramework with the correct configuration
3. **Handles Framework Linking**: Sets up the whisper.xcframework in the proper location for Xcode to find
4. **Verifies Prerequisites**: Checks that git, xcodebuild, and swift are installed before building
5. **Streamlines Development**: Provides convenient shortcuts for common development tasks

This approach ensures consistent builds across different machines and eliminates manual framework setup errors.

---

## Building for Local Use (No Apple Developer Certificate)

If you don't have an Apple Developer certificate, use `make local`:

```bash
git clone https://github.com/fightingentropy/VoiceInk.git
cd VoiceInk
make local
open /Applications/VoiceInk.app
```

This builds VoiceInk with a self-signed local `VoiceInk` identity using a separate build configuration (`LocalBuild.xcconfig`), bumps the app version, and installs a clean copy to `/Applications/VoiceInk.app`.

If you do not already have the local signing identity, create it with:

```bash
scripts/create_local_codesigning_identity.sh
```

That creates a self-signed local code-signing identity named `VoiceInk` in your login keychain.

### How It Works

The `make local` command uses:
- `LocalBuild.xcconfig` to override signing and entitlements settings
- `VoiceInk.local.entitlements` (stripped-down, no CloudKit/keychain groups)
- `LOCAL_BUILD` Swift compilation flag for conditional code paths
- the `VoiceInk` local signing identity so rebuilt `/Applications/VoiceInk.app` installs keep a stable macOS app identity

### Local Signing Reference

Use this pattern when you want stable local installs on your own Mac without introducing Apple Developer distribution certificates:

1. Create a self-signed code-signing identity in your login keychain, for example `VoiceInk`.
2. Keep the installed app path stable, for example `/Applications/VoiceInk.app`.
3. Keep the bundle identifier stable.
4. Point your local build configuration at that identity instead of `-` ad-hoc signing.
5. Reinstall by replacing the app in place instead of launching from a new timestamped build folder.

That combination gives macOS the best chance of preserving Accessibility, Microphone, and Automation permissions across rebuilds.

Your normal `make all` / `make build` commands are completely unaffected.

---

## Manual Build Process (Alternative)

If you prefer to build manually or need more control over the build process, follow these steps:

### Building whisper.cpp Framework

1. Clone and build whisper.cpp:
```bash
git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper.cpp
./build-xcframework.sh
```
This will create the XCFramework at `build-apple/whisper.xcframework`.

### Building VoiceInk

1. Clone the VoiceInk repository:
```bash
git clone https://github.com/fightingentropy/VoiceInk.git
cd VoiceInk
```

2. Add the whisper.xcframework to your project:
   - Drag and drop `../whisper.cpp/build-apple/whisper.xcframework` into the project navigator, or
   - Add it manually in the "Frameworks, Libraries, and Embedded Content" section of project settings

3. Build and Run
   - Build the project using Cmd+B or Product > Build
   - Run the project using Cmd+R or Product > Run

## Development Setup

1. **Xcode Configuration**
   - Ensure you have the latest Xcode version
   - Install any required Xcode Command Line Tools

2. **Dependencies**
   - The project uses [whisper.cpp](https://github.com/ggerganov/whisper.cpp) for transcription
   - Ensure the whisper.xcframework is properly linked in your Xcode project
   - Test the whisper.cpp installation independently before proceeding

3. **Building for Development**
   - Use the Debug configuration for development
   - Enable relevant debugging options in Xcode

4. **Testing**
   - Run the test suite before making changes
   - Ensure all tests pass after your modifications

## Troubleshooting

If you encounter any build issues:
1. Clean the build folder (Cmd+Shift+K)
2. Clean the build cache (Cmd+Shift+K twice)
3. Check Xcode and macOS versions
4. Verify all dependencies are properly installed
5. Make sure whisper.xcframework is properly built and linked

For more help, please check the [issues](https://github.com/fightingentropy/VoiceInk/issues) section or create a new issue. 

## Signing And Updates

For a signed distribution build under your own ownership:

1. Open the project in Xcode and select your Apple Developer team in `Signing & Capabilities`.
2. Keep or change the bundle identifiers now set under `com.fightingentropy.*` if you want a different reverse-DNS namespace.
3. Create your own Sparkle EdDSA key pair, then add your public key to `VoiceInk/Info.plist`.
4. Publish your signed DMG and update `appcast.xml` before re-enabling automatic update checks.
5. Push `announcements.json` from your repo if you want in-app announcement banners.

For the automated GitHub Actions release path, see [RELEASING.md](RELEASING.md).
