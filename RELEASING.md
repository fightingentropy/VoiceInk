# Releasing VoiceInk

This repository includes a Sparkle release workflow in `.github/workflows/sparkle-release.yml`.

## One-time setup

1. Enable GitHub Pages for the repository and let GitHub Actions deploy it.
2. Generate a Sparkle Ed25519 key locally with Sparkle's `generate_keys` tool.
3. Export the Sparkle private key and save it as the repository secret `SPARKLE_PRIVATE_KEY`.
4. Import your macOS signing certificate into GitHub Actions using:
   - `APPLE_DEVELOPER_ID_APPLICATION_P12_BASE64`
   - `APPLE_DEVELOPER_ID_APPLICATION_P12_PASSWORD`
5. Set `APPLE_CODESIGN_IDENTITY` if the certificate identity name is not the default `Developer ID Application`.
6. Optionally set `APPLE_TEAM_ID` if your signing setup requires an explicit Apple team during archive.
7. Optionally configure notarization with:
   - `APPLE_NOTARY_KEY_ID`
   - `APPLE_NOTARY_ISSUER_ID`
   - `APPLE_NOTARY_API_PRIVATE_KEY_BASE64`

## Publishing an update

1. Make sure `VoiceInk.xcodeproj/project.pbxproj` already contains the version you want to ship.
2. Push a tag like `v1.70.2`.
3. The workflow will:
   - create the GitHub release if it does not exist
   - build a signed release app
   - notarize and staple it if notary secrets are configured
   - package `VoiceInk.app` into a Sparkle `.zip`
   - upload the archive to the GitHub release
   - generate `appcast.xml`
   - deploy `appcast.xml` to GitHub Pages

## Notes

- The appcast feed URL is injected at build time, so the release build uses the current repository owner/name automatically.
- The automated release script defaults to `VoiceInk/VoiceInk.local.entitlements` so open-source builds do not require iCloud or Push Notifications provisioning.
- For local release testing on a Mac without a real Apple Developer certificate, run `scripts/create_local_codesigning_identity.sh`. The release script will prefer a local `VoiceInk` identity automatically when `APPLE_CODESIGN_IDENTITY` is not set.
- `make local` now uses that same `VoiceInk` identity for `/Applications/VoiceInk.app`, so local development installs and local release testing share the same signing reference.
- If you later want a fully provisioned Apple-team build, set `VOICEINK_RELEASE_ENTITLEMENTS` to `VoiceInk/VoiceInk.entitlements` in the workflow environment and provide the matching signing setup.
- Local `make local` builds still do not participate in Sparkle updates because they compile with `LOCAL_BUILD`.
- Sparkle update prompts require a shipped build with a higher version than the installed app and a published appcast entry.
