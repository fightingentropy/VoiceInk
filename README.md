# README

<div align="center">
  <img src="VoiceInk/Assets.xcassets/AppIcon.appiconset/256-mac.png" width="180" height="180" />
  <h1>VoiceInk</h1>
  <p>Fast voice-to-text for macOS with local transcription, personal vocabulary, and configurable recording workflows.</p>

  [![License](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
  ![Platform](https://img.shields.io/badge/platform-macOS%2014.0%2B-brightgreen)
  [![GitHub release (latest by date)](https://img.shields.io/github/v/release/fightingentropy/VoiceInk)](https://github.com/fightingentropy/VoiceInk/releases)
  ![GitHub all releases](https://img.shields.io/github/downloads/fightingentropy/VoiceInk/total)
  ![GitHub stars](https://img.shields.io/github/stars/fightingentropy/VoiceInk?style=social)
  <p>
    <a href="https://github.com/fightingentropy/VoiceInk">Repository</a> •
    <a href="https://github.com/fightingentropy/VoiceInk/releases">Releases</a> •
    <a href="https://github.com/fightingentropy/VoiceInk/issues">Issues</a>
  </p>

  <a href="https://github.com/fightingentropy/VoiceInk">
    <img src="https://img.shields.io/badge/View%20Repository-GitHub-blue?style=for-the-badge&logo=github" alt="View VoiceInk repository" width="250"/>
  </a>
</div>

---

VoiceInk is a native macOS application that transcribes speech to text with a focus on speed, privacy, and local-first workflows.

![VoiceInk Mac App](https://github.com/user-attachments/assets/12367379-83e7-48a6-b52c-4488a6a04bba)

This repository is set up for self-hosted development and distribution. If you plan to ship signed builds or Sparkle updates, use your own Apple Developer team, bundle identifiers, signing keys, and release infrastructure.

## Features

- 🎙️ **Accurate Transcription**: Local AI models that transcribe your voice to text with 99% accuracy, almost instantly
- 🔒 **Privacy First**: 100% offline processing ensures your data never leaves your device
- 🔥 **Model Prewarm & Warm Retention**: Prewarm local models on launch and wake, then keep them resident for a configurable idle window
- ⚡ **Power Mode**: Intelligent app detection automatically applies your perfect pre-configured settings based on the app/ URL you're on
- 🧠 **Context Aware**: Smart AI that understands your screen content and adapts to the context
- 🎯 **Global Shortcuts**: Configurable keyboard shortcuts for quick recording and push-to-talk functionality
- 📝 **Personal Dictionary**: Train the AI to understand your unique terminology with custom words, industry terms, and smart text replacements
- 🔄 **Smart Modes**: Instantly switch between AI-powered modes optimized for different writing styles and contexts
- 🤖 **AI Assistant**: Built-in voice assistant mode for a quick chatGPT like conversational assistant

## Get Started

### Download
Download the latest signed build from [GitHub Releases](https://github.com/fightingentropy/VoiceInk/releases), or build locally from source using the instructions in [BUILDING.md](BUILDING.md).

#### Homebrew
Alternatively, you can install VoiceInk via `brew`:

```shell
brew install --cask voiceink
```

### Build from Source
Build VoiceInk locally by following [BUILDING.md](BUILDING.md). The project includes a `LocalBuild.xcconfig` path for stable `/Applications` installs signed with a local `VoiceInk` identity, plus the normal Xcode configuration for shipped distribution builds.

## Requirements

- macOS 14.4 or later

## Model Runtime Matrix

VoiceInk supports a mix of native local runtimes and cloud providers. They are not all implemented the same way.

| Model family | Runs where | Runtime | Notes |
| --- | --- | --- | --- |
| Whisper local models | On-device | Native `whisper.cpp` runtime | Uses the bundled `whisper` C runtime. Supported models can also download an optional Core ML encoder on Apple Silicon. |
| Voxtral Realtime (Local MLX) | On-device | Native MLX | Apple Silicon path for low-latency realtime transcription. |
| Parakeet V2 / V3 | On-device | Native Core ML via FluidAudio | Local runtime with on-device batch and streaming support. |
| Apple Speech | On-device | Native Apple Speech framework | Uses Apple's Speech APIs when built and run with the required macOS/SDK support. |
| Cohere Transcribe (Local MLX) | On-device | Native MLX | Local Apple Silicon path for high-accuracy batch transcription. |
| ElevenLabs Scribe / custom API models | Cloud | Remote API | Audio leaves the device and is transcribed by the configured provider. |

### Cohere Transcribe Note

`Cohere Transcribe (Local MLX)` now runs through the app's native MLX path on Apple Silicon. It downloads MLX model assets directly and no longer depends on the older Python/PyTorch `mps` worker or a saved Hugging Face access token inside VoiceInk.

## Documentation

- [Building from Source](BUILDING.md) - Detailed instructions for building the project
- [Releasing](RELEASING.md) - Signing, Sparkle, and GitHub release automation
- [Contributing Guidelines](CONTRIBUTING.md) - How to contribute to VoiceInk
- [Code of Conduct](CODE_OF_CONDUCT.md) - Our community standards

## Ownership Notes

- This repository is configured for `fightingentropy/VoiceInk`.
- Local builds still disable Sparkle update checks with `LOCAL_BUILD`.
- Local installs are rebuilt into `/Applications/VoiceInk.app` and signed with the self-signed local `VoiceInk` identity for better macOS permission continuity.
- Announcements are read from `announcements.json` in this repository.

## Contributing

We welcome contributions! However, please note that all contributions should align with the project's goals and vision. Before starting work on any feature or fix:

1. Read our [Contributing Guidelines](CONTRIBUTING.md)
2. Open an issue to discuss your proposed changes
3. Wait for maintainer feedback

For build instructions, see our [Building Guide](BUILDING.md).

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Support

If you encounter any issues or have questions, please:
1. Check the existing issues in the GitHub repository
2. Create a new issue if your problem isn't already reported
3. Provide as much detail as possible about your environment and the problem

## Acknowledgments

### Core Technology
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) - High-performance inference of OpenAI's Whisper model
- [FluidAudio](https://github.com/FluidInference/FluidAudio) - Used for Parakeet model implementation

### Essential Dependencies
- [Sparkle](https://github.com/sparkle-project/Sparkle) - Keeping VoiceInk up to date
- [KeyboardShortcuts](https://github.com/sindresorhus/KeyboardShortcuts) - User-customizable keyboard shortcuts
- [LaunchAtLogin](https://github.com/sindresorhus/LaunchAtLogin) - Launch at login functionality
- [MediaRemoteAdapter](https://github.com/ejbills/mediaremote-adapter) - Media playback control during recording
- [Zip](https://github.com/marmelroy/Zip) - File compression and decompression utilities
- [SelectedTextKit](https://github.com/tisfeng/SelectedTextKit) - A modern macOS library for getting selected text
- [Swift Atomics](https://github.com/apple/swift-atomics) - Low-level atomic operations for thread-safe concurrent programming
