# Keep third-party build dependencies inside the project instead of the user's home folder
DEPS_DIR := $(CURDIR)/Dependencies
WHISPER_CPP_DIR := $(DEPS_DIR)/whisper.cpp
FRAMEWORK_PATH := $(WHISPER_CPP_DIR)/build-apple/whisper.xcframework
LOCAL_BUILD_ROOT := $(CURDIR)/.codex-build/local-install
LOCAL_DERIVED_DATA := $(LOCAL_BUILD_ROOT)/deriveddata
LOCAL_SPM_DIR := $(LOCAL_BUILD_ROOT)/spm
LOCAL_BUILD_CONFIGURATION := Release
INSTALL_APP_PATH := /Applications/VoiceInk.app
LOCAL_CODESIGN_IDENTITY ?= VoiceInk

.PHONY: all clean whisper setup build local install-local bump-version check healthcheck help dev run

# Default target
all: check build

# Development workflow
dev: build run

bump-version:
	@./scripts/bump_version.sh $(CURDIR)

# Prerequisites
check:
	@echo "Checking prerequisites..."
	@command -v git >/dev/null 2>&1 || { echo "git is not installed"; exit 1; }
	@command -v xcodebuild >/dev/null 2>&1 || { echo "xcodebuild is not installed (need Xcode)"; exit 1; }
	@command -v swift >/dev/null 2>&1 || { echo "swift is not installed"; exit 1; }
	@echo "Prerequisites OK"

healthcheck: check

# Build process
whisper:
	@mkdir -p $(DEPS_DIR)
	@if [ ! -d "$(FRAMEWORK_PATH)" ]; then \
		echo "Building whisper.xcframework in $(DEPS_DIR)..."; \
		if [ ! -d "$(WHISPER_CPP_DIR)" ]; then \
			git clone https://github.com/ggerganov/whisper.cpp.git $(WHISPER_CPP_DIR); \
		else \
			(cd $(WHISPER_CPP_DIR) && git pull); \
		fi; \
		cd $(WHISPER_CPP_DIR) && ./build-xcframework.sh; \
	else \
		echo "whisper.xcframework already built in $(DEPS_DIR), skipping build"; \
	fi

setup: whisper
	@echo "Whisper framework is ready at $(FRAMEWORK_PATH)"
	@echo "Please ensure your Xcode project references the framework from this new location."

build: setup
	xcodebuild -project VoiceInk.xcodeproj -scheme VoiceInk -configuration Debug build

# Build for local use without Apple Developer certificate
install-local: bump-version check setup
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
		rm -rf "$(INSTALL_APP_PATH)"; \
		ditto "$$APP_PATH" "$(INSTALL_APP_PATH)"; \
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

# Run application
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

# Cleanup
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(DEPS_DIR)
	@echo "Clean complete"

# Help
help:
	@echo "Available targets:"
	@echo "  check/healthcheck  Check if required CLI tools are installed"
	@echo "  whisper            Clone and build whisper.cpp XCFramework"
	@echo "  setup              Copy whisper XCFramework to VoiceInk project"
	@echo "  build              Build the VoiceInk Xcode project with local Sign to Run Locally signing"
	@echo "  bump-version       Increment the app marketing/build versions"
	@echo "  install-local      Build for local use and install a clean app to /Applications"
	@echo "  local              Alias for install-local"
	@echo "  run                Launch the built VoiceInk app"
	@echo "  dev                Build and run the app (for development)"
	@echo "  all                Run full build process (default)"
	@echo "  clean              Remove build artifacts"
	@echo "  help               Show this help message"
