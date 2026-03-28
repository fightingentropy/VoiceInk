# VoiceInk build helpers
LOCAL_BUILD_ROOT := $(CURDIR)/.codex-build/local-install
LOCAL_DERIVED_DATA := $(LOCAL_BUILD_ROOT)/deriveddata
LOCAL_SPM_DIR := $(LOCAL_BUILD_ROOT)/spm
LOCAL_BUILD_CONFIGURATION := Release
INSTALL_APP_PATH := /Applications/VoiceInk.app
LOCAL_CODESIGN_IDENTITY ?= VoiceInk

.PHONY: all clean build local install-local bump-version check healthcheck help dev run resolve-packages

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

clean:
	@echo "Cleaning build artifacts..."
	@rm -rf "$(LOCAL_BUILD_ROOT)"
	@echo "Clean complete"

help:
	@echo "Available targets:"
	@echo "  check/healthcheck  Check if required CLI tools are installed"
	@echo "  resolve-packages   Resolve Swift package dependencies"
	@echo "  build              Build the VoiceInk Xcode project"
	@echo "  bump-version       Increment the app marketing/build versions"
	@echo "  install-local      Build for local use and install a clean app to /Applications"
	@echo "  local              Alias for install-local"
	@echo "  run                Launch the built VoiceInk app"
	@echo "  dev                Build and run the app (for development)"
	@echo "  all                Run full build process (default)"
	@echo "  clean              Remove build artifacts"
	@echo "  help               Show this help message"
