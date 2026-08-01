# VoiceInk build helpers
-include Signing.local.mk

LOCAL_BUILD_ROOT := $(CURDIR)/.codex-build/local-install
LOCAL_DERIVED_DATA := $(LOCAL_BUILD_ROOT)/deriveddata
LOCAL_SPM_DIR := $(LOCAL_BUILD_ROOT)/spm
LOCAL_BUILD_CONFIGURATION := Release
INSTALL_APP_PATH := /Applications/VoiceInk.app
LOCAL_CODESIGN_IDENTITY ?=
CHECK_DERIVED_DATA ?= $(CURDIR)/.codex-build/check

# Distribution (Developer ID + notarization)
DIST_BUILD_ROOT := $(CURDIR)/.codex-build/dist
DIST_DERIVED_DATA := $(DIST_BUILD_ROOT)/deriveddata
DIST_SPM_DIR := $(DIST_BUILD_ROOT)/spm
DIST_OUTPUT_DIR := $(DIST_BUILD_ROOT)/out
DIST_CODESIGN_IDENTITY ?=
DIST_TEAM_ID ?=
NOTARY_PROFILE ?=

.PHONY: all clean build local install-local bump-version prerequisites test check healthcheck help dev run resolve-packages dist require-local-signing require-dist-signing

all: check build

dev: build run

bump-version:
	@./scripts/bump_version.sh $(CURDIR)

prerequisites:
	@echo "Checking prerequisites..."
	@command -v xcodebuild >/dev/null 2>&1 || { echo "xcodebuild is not installed (need Xcode)"; exit 1; }
	@command -v swift >/dev/null 2>&1 || { echo "swift is not installed"; exit 1; }
	@command -v rsync >/dev/null 2>&1 || { echo "rsync is not installed"; exit 1; }
	@echo "Prerequisites OK"

test: prerequisites resolve-packages
	xcodebuild -project VoiceInk.xcodeproj -scheme VoiceInk \
		-destination 'platform=macOS' \
		-derivedDataPath "$(CHECK_DERIVED_DATA)" \
		CODE_SIGNING_ALLOWED=NO \
		test \
		-only-testing:VoiceInkTests/BenchmarkMetricsTests \
		-only-testing:VoiceInkTests/CohereNativeFoundationTests \
		-only-testing:VoiceInkTests/DictationFormattingTests \
		-only-testing:VoiceInkTests/EphemeralTranscriptionPolicyTests \
		-only-testing:VoiceInkTests/LocalTranscriptionHotPathTests \
		-only-testing:VoiceInkTests/OpenAITranscriptionTests \
		-only-testing:VoiceInkTests/StreamingTranscriptionServiceTests \
		-only-testing:VoiceInkTests/TranscriptionModelManagerTests \
		-only-testing:VoiceInkTests/VoiceInkTests

check: test

healthcheck: prerequisites

resolve-packages:
	xcodebuild -project VoiceInk.xcodeproj -resolvePackageDependencies

build: prerequisites resolve-packages
	xcodebuild -project VoiceInk.xcodeproj -scheme VoiceInk -configuration Debug build

require-local-signing:
	@test -n "$(LOCAL_CODESIGN_IDENTITY)" || { \
		echo "LOCAL_CODESIGN_IDENTITY is required. Copy Signing.example.mk to Signing.local.mk and configure it locally."; \
		exit 1; \
	}
	@security find-identity -v -p codesigning | grep -F '"$(LOCAL_CODESIGN_IDENTITY)"' >/dev/null || { \
		echo "The configured LOCAL_CODESIGN_IDENTITY is not available in the current keychain."; \
		exit 1; \
	}

install-local: require-local-signing bump-version prerequisites resolve-packages
	@echo "Building VoiceInk for local use and installing to $(INSTALL_APP_PATH)..."
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
		echo "  - No automatic in-app update checks"; \
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

require-dist-signing:
	@test -n "$(DIST_CODESIGN_IDENTITY)" || { \
		echo "DIST_CODESIGN_IDENTITY is required through Signing.local.mk or the CI environment."; \
		exit 1; \
	}
	@test -n "$(DIST_TEAM_ID)" || { \
		echo "DIST_TEAM_ID is required through Signing.local.mk or the CI environment."; \
		exit 1; \
	}
	@test -n "$(NOTARY_PROFILE)" || { \
		echo "NOTARY_PROFILE is required through Signing.local.mk or the CI environment."; \
		exit 1; \
	}
	@command -v codesign >/dev/null 2>&1 || { echo "codesign is unavailable."; exit 1; }
	@command -v ditto >/dev/null 2>&1 || { echo "ditto is unavailable."; exit 1; }
	@command -v plutil >/dev/null 2>&1 || { echo "plutil is unavailable."; exit 1; }
	@command -v spctl >/dev/null 2>&1 || { echo "spctl is unavailable."; exit 1; }
	@security find-identity -v -p codesigning | grep -F '"$(DIST_CODESIGN_IDENTITY)"' >/dev/null || { \
		echo "The configured DIST_CODESIGN_IDENTITY is not available in the current keychain."; \
		exit 1; \
	}
	@xcrun --find notarytool >/dev/null 2>&1 || { echo "xcrun notarytool is unavailable."; exit 1; }
	@xcrun --find stapler >/dev/null 2>&1 || { echo "xcrun stapler is unavailable."; exit 1; }
	@xcrun notarytool history --keychain-profile "$(NOTARY_PROFILE)" >/dev/null 2>&1 || { \
		echo "The configured NOTARY_PROFILE is unavailable or cannot authenticate."; \
		exit 1; \
	}

dist: require-dist-signing prerequisites resolve-packages
	@echo "Building VoiceInk for distribution (Developer ID + hardened runtime)..."
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
	@set -eu; \
	APP_PATH="$(DIST_DERIVED_DATA)/Build/Products/Release/VoiceInk.app"; \
	if [ ! -d "$$APP_PATH" ]; then echo "Error: built app not found at $$APP_PATH"; exit 1; fi; \
	VERSION=$$(/usr/libexec/PlistBuddy -c 'Print CFBundleShortVersionString' "$$APP_PATH/Contents/Info.plist"); \
	SUBMISSION_ZIP="$(DIST_BUILD_ROOT)/VoiceInk-$$VERSION.notary-submission.zip"; \
	NOTARY_RESULT="$(DIST_BUILD_ROOT)/notary-result.json"; \
	ZIP_PATH="$(DIST_OUTPUT_DIR)/VoiceInk-$$VERSION.zip"; \
	cleanup_failed_distribution() { rm -f "$$SUBMISSION_ZIP" "$$NOTARY_RESULT" "$$ZIP_PATH"; rm -rf "$$APP_PATH"; }; \
	trap cleanup_failed_distribution 0 HUP INT TERM; \
	echo "Verifying Developer ID signature and hardened runtime..."; \
	codesign --verify --deep --strict "$$APP_PATH"; \
	codesign -d --verbose=2 "$$APP_PATH" 2>&1 | grep -q "flags=.*runtime" || { echo "Error: hardened runtime flag missing"; exit 1; }; \
	ditto -c -k --sequesterRsrc --keepParent "$$APP_PATH" "$$SUBMISSION_ZIP"; \
	echo "Submitting to Apple notary service (waits for verdict)..."; \
	xcrun notarytool submit "$$SUBMISSION_ZIP" --keychain-profile "$(NOTARY_PROFILE)" --wait --output-format json > "$$NOTARY_RESULT"; \
	NOTARY_STATUS=$$(plutil -extract status raw -o - "$$NOTARY_RESULT"); \
	if [ "$$NOTARY_STATUS" != "Accepted" ]; then echo "Notarization was not accepted (status: $$NOTARY_STATUS)."; exit 1; fi; \
	echo "Stapling and validating the notarization ticket..."; \
	xcrun stapler staple "$$APP_PATH"; \
	xcrun stapler validate "$$APP_PATH"; \
	codesign --verify --deep --strict "$$APP_PATH"; \
	spctl --assess --type execute --verbose=2 "$$APP_PATH"; \
	ditto -c -k --sequesterRsrc --keepParent "$$APP_PATH" "$$ZIP_PATH"; \
	rm -f "$$SUBMISSION_ZIP" "$$NOTARY_RESULT"; \
	trap - 0 HUP INT TERM; \
	echo "Distribution build ready: $$ZIP_PATH (signed, notarized, and stapled)"

clean:
	@echo "Cleaning build artifacts..."
	@rm -rf "$(LOCAL_BUILD_ROOT)"
	@rm -rf "$(DIST_BUILD_ROOT)"
	@echo "Clean complete"

help:
	@echo "Available targets:"
	@echo "  prerequisites      Check if required CLI tools are installed"
	@echo "  test               Run deterministic unit and streaming lifecycle tests"
	@echo "  check              Run the mandatory local/CI test gate"
	@echo "  healthcheck        Alias for prerequisites"
	@echo "  resolve-packages   Resolve Swift package dependencies"
	@echo "  build              Build the VoiceInk Xcode project"
	@echo "  bump-version       Increment the app marketing/build versions"
	@echo "  install-local      Build for local use and install a clean app to /Applications"
	@echo "  dist               Developer ID build: require signing + notarization, staple, verify, zip"
	@echo "  local              Alias for install-local"
	@echo "  run                Launch the built VoiceInk app"
	@echo "  dev                Build and run the app (for development)"
	@echo "  all                Run full build process (default)"
	@echo "  clean              Remove build artifacts"
	@echo "  help               Show this help message"
