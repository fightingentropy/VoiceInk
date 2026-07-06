# VoiceInk build helpers
LOCAL_BUILD_ROOT := $(CURDIR)/.codex-build/local-install
LOCAL_DERIVED_DATA := $(LOCAL_BUILD_ROOT)/deriveddata
LOCAL_SPM_DIR := $(LOCAL_BUILD_ROOT)/spm
LOCAL_BUILD_CONFIGURATION := Release
INSTALL_APP_PATH := /Applications/VoiceInk.app
LOCAL_CODESIGN_IDENTITY ?= Apple Development: erlin.hx@icloud.com (476S656MHV)

# Distribution (Developer ID + notarization)
DIST_BUILD_ROOT := $(CURDIR)/.codex-build/dist
DIST_DERIVED_DATA := $(DIST_BUILD_ROOT)/deriveddata
DIST_SPM_DIR := $(DIST_BUILD_ROOT)/spm
DIST_OUTPUT_DIR := $(DIST_BUILD_ROOT)/out
DIST_CODESIGN_IDENTITY ?= Developer ID Application: Erlin Hoxha (T29NU9NCA2)
DIST_TEAM_ID ?= T29NU9NCA2
NOTARY_PROFILE ?= VoiceInkNotary

.PHONY: all clean build local install-local bump-version check healthcheck help dev run resolve-packages dist

all: check build

dev: build run

bump-version:
	@./scripts/bump_version.sh $(CURDIR)

check:
	@echo "Checking prerequisites..."
	@command -v xcodebuild >/dev/null 2>&1 || { echo "xcodebuild is not installed (need Xcode)"; exit 1; }
	@command -v swift >/dev/null 2>&1 || { echo "swift is not installed"; exit 1; }
	@command -v rsync >/dev/null 2>&1 || { echo "rsync is not installed"; exit 1; }
	@echo "Prerequisites OK"

healthcheck: check

resolve-packages:
	xcodebuild -project VoiceInk.xcodeproj -resolvePackageDependencies

build: check resolve-packages
	xcodebuild -project VoiceInk.xcodeproj -scheme VoiceInk -configuration Debug build

install-local: bump-version check resolve-packages
	@echo "Building VoiceInk for local use and installing to $(INSTALL_APP_PATH)..."
	@security find-identity -v -p codesigning | grep -F '"$(LOCAL_CODESIGN_IDENTITY)"' >/dev/null || { \
		echo "Missing local code signing identity: $(LOCAL_CODESIGN_IDENTITY)"; \
		echo "Available identities:"; \
		security find-identity -v -p codesigning; \
		exit 1; \
	}
	@rm -rf "$(LOCAL_BUILD_ROOT)"
	@mkdir -p "$(LOCAL_BUILD_ROOT)"
	xcodebuild -project VoiceInk.xcodeproj -scheme VoiceInk -configuration $(LOCAL_BUILD_CONFIGURATION) \
		-xcconfig LocalBuild.xcconfig \
		-derivedDataPath "$(LOCAL_DERIVED_DATA)" \
		-clonedSourcePackagesDirPath "$(LOCAL_SPM_DIR)" \
		CODE_SIGN_IDENTITY="$(LOCAL_CODESIGN_IDENTITY)" \
		CODE_SIGNING_REQUIRED=NO \
		CODE_SIGNING_ALLOWED=YES \
		DEVELOPMENT_TEAM="" \
		CODE_SIGN_ENTITLEMENTS=$(CURDIR)/VoiceInk/VoiceInk.local.entitlements \
		SWIFT_ACTIVE_COMPILATION_CONDITIONS='$$(inherited) LOCAL_BUILD' \
		build
	@APP_PATH="$(LOCAL_DERIVED_DATA)/Build/Products/$(LOCAL_BUILD_CONFIGURATION)/VoiceInk.app" && \
	if [ -d "$$APP_PATH" ]; then \
		echo "Installing clean build to $(INSTALL_APP_PATH)..."; \
		pkill -x VoiceInk >/dev/null 2>&1 || true; \
		mkdir -p "$(INSTALL_APP_PATH)"; \
		rsync -aE --delete "$$APP_PATH"/ "$(INSTALL_APP_PATH)"/; \
		xattr -cr "$(INSTALL_APP_PATH)"; \
		rm -rf "$$HOME/Downloads/VoiceInk.app"; \
		open -na "$(INSTALL_APP_PATH)"; \
		echo ""; \
		echo "Build complete! App installed to: $(INSTALL_APP_PATH)"; \
		echo "Run with: open $(INSTALL_APP_PATH)"; \
		echo ""; \
		echo "Limitations of local builds:"; \
		echo "  - No iCloud dictionary sync"; \
		echo "  - No automatic in-app update notifications without a published Sparkle feed"; \
	else \
		echo "Error: Could not find built VoiceInk.app in $(LOCAL_DERIVED_DATA)."; \
		exit 1; \
	fi

local: install-local

run:
	@if [ -d "$(INSTALL_APP_PATH)" ]; then \
		echo "Opening $(INSTALL_APP_PATH)..."; \
		open "$(INSTALL_APP_PATH)"; \
	else \
		echo "Looking for VoiceInk.app in $(LOCAL_DERIVED_DATA)..."; \
		APP_PATH="$(LOCAL_DERIVED_DATA)/Build/Products/Debug/VoiceInk.app" && \
		if [ -d "$$APP_PATH" ]; then \
			echo "Found app at: $$APP_PATH"; \
			open "$$APP_PATH"; \
		else \
			echo "VoiceInk.app not found. Please run 'make build' or 'make local' first."; \
			exit 1; \
		fi; \
	fi

dist: check resolve-packages
	@echo "Building VoiceInk for distribution (Developer ID + hardened runtime)..."
	@security find-identity -v -p codesigning | grep -F '"$(DIST_CODESIGN_IDENTITY)"' >/dev/null || { \
		echo "Missing distribution signing identity: $(DIST_CODESIGN_IDENTITY)"; \
		echo "Available identities:"; \
		security find-identity -v -p codesigning; \
		exit 1; \
	}
	@rm -rf "$(DIST_BUILD_ROOT)"
	@mkdir -p "$(DIST_OUTPUT_DIR)"
	xcodebuild -project VoiceInk.xcodeproj -scheme VoiceInk -configuration Release \
		-derivedDataPath "$(DIST_DERIVED_DATA)" \
		-clonedSourcePackagesDirPath "$(DIST_SPM_DIR)" \
		CODE_SIGN_IDENTITY="$(DIST_CODESIGN_IDENTITY)" \
		CODE_SIGN_STYLE=Manual \
		CODE_SIGNING_REQUIRED=YES \
		CODE_SIGNING_ALLOWED=YES \
		DEVELOPMENT_TEAM="$(DIST_TEAM_ID)" \
		OTHER_CODE_SIGN_FLAGS="--timestamp" \
		CODE_SIGN_INJECT_BASE_ENTITLEMENTS=NO \
		CODE_SIGN_ENTITLEMENTS=$(CURDIR)/VoiceInk/VoiceInk.dist.entitlements \
		SWIFT_ACTIVE_COMPILATION_CONDITIONS='$$(inherited) OPEN_SOURCE_DISTRIBUTION' \
		build
	@APP_PATH="$(DIST_DERIVED_DATA)/Build/Products/Release/VoiceInk.app" && \
	if [ ! -d "$$APP_PATH" ]; then echo "Error: built app not found at $$APP_PATH"; exit 1; fi && \
	echo "Verifying signature..." && \
	codesign --verify --deep --strict "$$APP_PATH" && \
	codesign -d --verbose=2 "$$APP_PATH" 2>&1 | grep -q "flags=.*runtime" || { echo "Error: hardened runtime flag missing"; exit 1; }; \
	APP_PATH="$(DIST_DERIVED_DATA)/Build/Products/Release/VoiceInk.app" && \
	VERSION=$$(/usr/libexec/PlistBuddy -c 'Print CFBundleShortVersionString' "$$APP_PATH/Contents/Info.plist") && \
	ZIP_PATH="$(DIST_OUTPUT_DIR)/VoiceInk-$$VERSION.zip" && \
	ditto -c -k --keepParent "$$APP_PATH" "$$ZIP_PATH" && \
	echo "Signed archive: $$ZIP_PATH" && \
	if xcrun notarytool history --keychain-profile "$(NOTARY_PROFILE)" >/dev/null 2>&1; then \
		echo "Submitting to Apple notary service (waits for verdict)..."; \
		xcrun notarytool submit "$$ZIP_PATH" --keychain-profile "$(NOTARY_PROFILE)" --wait || exit 1; \
		echo "Stapling ticket..."; \
		xcrun stapler staple "$$APP_PATH" && \
		ditto -c -k --keepParent "$$APP_PATH" "$$ZIP_PATH" && \
		spctl --assess --type execute --verbose=2 "$$APP_PATH" && \
		echo "" && \
		echo "Distribution build ready: $$ZIP_PATH (notarized + stapled)"; \
	else \
		echo ""; \
		echo "Notary credentials not found — produced a signed (NOT notarized) archive."; \
		echo "One-time setup (needs an app-specific password from account.apple.com):"; \
		echo "  xcrun notarytool store-credentials $(NOTARY_PROFILE) --apple-id erlin.hx@icloud.com --team-id $(DIST_TEAM_ID)"; \
		echo "Then re-run: make dist"; \
	fi

clean:
	@echo "Cleaning build artifacts..."
	@rm -rf "$(LOCAL_BUILD_ROOT)"
	@rm -rf "$(DIST_BUILD_ROOT)"
	@echo "Clean complete"

help:
	@echo "Available targets:"
	@echo "  check/healthcheck  Check if required CLI tools are installed"
	@echo "  resolve-packages   Resolve Swift package dependencies"
	@echo "  build              Build the VoiceInk Xcode project"
	@echo "  bump-version       Increment the app marketing/build versions"
	@echo "  install-local      Build for local use and install a clean app to /Applications"
	@echo "  dist               Developer ID build: sign, notarize (if credentials stored), staple, zip"
	@echo "  local              Alias for install-local"
	@echo "  run                Launch the built VoiceInk app"
	@echo "  dev                Build and run the app (for development)"
	@echo "  all                Run full build process (default)"
	@echo "  clean              Remove build artifacts"
	@echo "  help               Show this help message"
