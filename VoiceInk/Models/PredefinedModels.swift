import Foundation
 
 enum PredefinedModels {
    static func getLanguageDictionary(isMultilingual: Bool, provider: ModelProvider = .local) -> [String: String] {
        if !isMultilingual {
            return ["en": "English"]
        } else {
            // For Apple Native models, return only supported languages in simple format
            if provider == .nativeApple {
                let appleSupportedCodes = ["ar", "de", "en", "es", "fr", "it", "ja", "ko", "pt", "yue", "zh"]
                return allLanguages.filter { appleSupportedCodes.contains($0.key) }
            }
            if provider == .cohereTranscribe {
                return cohereTranscribeLanguages
            }
            return allLanguages
        }
    }
    
    // Apple Native Speech specific languages with proper BCP-47 format
    // Based on actual supported locales from SpeechTranscriber.supportedLocales
    static let appleNativeLanguages = [
        // English variants
        "en-US": "English (United States)",
        "en-GB": "English (United Kingdom)",
        "en-CA": "English (Canada)",
        "en-AU": "English (Australia)",
        "en-IN": "English (India)",
        "en-IE": "English (Ireland)",
        "en-NZ": "English (New Zealand)",
        "en-ZA": "English (South Africa)",
        "en-SA": "English (Saudi Arabia)",
        "en-AE": "English (UAE)",
        "en-SG": "English (Singapore)",
        "en-PH": "English (Philippines)",
        "en-ID": "English (Indonesia)",
        
        // Spanish variants
        "es-ES": "Spanish (Spain)",
        "es-MX": "Spanish (Mexico)",
        "es-US": "Spanish (United States)",
        "es-CO": "Spanish (Colombia)",
        "es-CL": "Spanish (Chile)",
        "es-419": "Spanish (Latin America)",
        
        // French variants
        "fr-FR": "French (France)",
        "fr-CA": "French (Canada)",
        "fr-BE": "French (Belgium)",
        "fr-CH": "French (Switzerland)",
        
        // German variants
        "de-DE": "German (Germany)",
        "de-AT": "German (Austria)",
        "de-CH": "German (Switzerland)",
        
        // Chinese variants
        "zh-CN": "Chinese Simplified (China)",
        "zh-TW": "Chinese Traditional (Taiwan)",
        "zh-HK": "Chinese Traditional (Hong Kong)",
        
        // Other Asian languages
        "ja-JP": "Japanese (Japan)",
        "ko-KR": "Korean (South Korea)",
        "yue-CN": "Cantonese (China)",
        
        // Portuguese variants
        "pt-BR": "Portuguese (Brazil)",
        "pt-PT": "Portuguese (Portugal)",
        
        // Italian variants
        "it-IT": "Italian (Italy)",
        "it-CH": "Italian (Switzerland)",
        
        // Arabic
        "ar-SA": "Arabic (Saudi Arabia)"
    ]

    static let cohereTranscribeLanguages = [
        "ar": "Arabic",
        "de": "German",
        "el": "Greek",
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "it": "Italian",
        "ja": "Japanese",
        "ko": "Korean",
        "nl": "Dutch",
        "pl": "Polish",
        "pt": "Portuguese",
        "vi": "Vietnamese",
        "zh": "Chinese"
    ]
    
    static var models: [any TranscriptionModel] {
        return predefinedModels + CustomModelManager.shared.customModels
    }
    
    private static let predefinedModels: [any TranscriptionModel] = [
        // Native Apple Model
        NativeAppleModel(
            name: "apple-speech",
            displayName: "Apple Speech",
            description: "Uses the native Apple Speech framework for transcription. Requires macOS 26",
            isMultilingualModel: true,
            supportedLanguages: getLanguageDictionary(isMultilingual: true, provider: .nativeApple)
        ),
        
        // Parakeet Models
        ParakeetModel(
            name: "parakeet-tdt-0.6b-v2",
            displayName: "Parakeet V2",
            description: "NVIDIA's Parakeet V2 model optimized for lightning-fast English-only transcription",
            size: "474 MB",
            speed: 0.99,
            accuracy: 0.94,
            ramUsage: 0.8,
            supportedLanguages: getLanguageDictionary(isMultilingual: false, provider: .parakeet)
        ),
        ParakeetModel(
            name: "parakeet-tdt-0.6b-v3",
            displayName: "Parakeet V3",
            description: "NVIDIA's Parakeet V3 model with multilingual support across English and 25 European languages",
            size: "494 MB",
            speed: 0.99,
            accuracy: 0.94,
            ramUsage: 0.8,
            supportedLanguages: getLanguageDictionary(isMultilingual: true, provider: .parakeet)
        ),
        LocalModel(
            name: "whisper-large-v3-turbo",
            displayName: "Whisper Large v3 Turbo",
            size: "2.8 GB",
            supportedLanguages: getLanguageDictionary(isMultilingual: true, provider: .local),
            description: "Fast local Whisper preset using WhisperKit and Core ML on Apple Silicon.",
            speed: 0.96,
            accuracy: 0.95,
            ramUsage: 2.4,
            whisperKitVariant: "openai_whisper-large-v3_turbo"
        ),

        LocalVoxtralModel(
            name: "voxtral-mini-realtime-local",
            displayName: "Voxtral Realtime (Local MLX)",
            size: "2.5 GB",
            description: "Runs locally on Apple Silicon with native MLX inference for low-latency realtime transcription",
            speed: 0.99,
            accuracy: 0.97,
            isMultilingualModel: true,
            supportedLanguages: getLanguageDictionary(isMultilingual: true, provider: .localVoxtral)
        ),
        LocalCohereTranscribeModel(
            name: "cohere-transcribe-03-2026-local",
            displayName: "Cohere Transcribe (Local MLX)",
            size: "4.1 GB",
            description: "Runs locally on Apple Silicon with native MLX inference for high-accuracy live recorder transcription.",
            speed: 0.94,
            accuracy: 0.99,
            isMultilingualModel: true,
            supportedLanguages: getLanguageDictionary(isMultilingual: true, provider: .cohereTranscribe)
        ),
        // Cloud Models
       CloudModel(
           name: "scribe_v2",
           displayName: "Scribe V2 Realtime (ElevenLabs)",
           description: "ElevenLabs' Scribe V2 Realtime model for the most accurate transcription",
           provider: .elevenLabs,
           speed: 0.99,
           accuracy: 0.98,
           isMultilingual: true,
           supportedLanguages: getLanguageDictionary(isMultilingual: true, provider: .elevenLabs)
       ),
     ]
 
     static let allLanguages = [
         "auto": "Auto-detect",
         "af": "Afrikaans",
         "am": "Amharic",
         "ar": "Arabic",
         "as": "Assamese",
         "az": "Azerbaijani",
         "ba": "Bashkir",
         "be": "Belarusian",
         "bg": "Bulgarian",
         "bn": "Bengali",
         "bo": "Tibetan",
         "br": "Breton",
         "bs": "Bosnian",
         "ca": "Catalan",
         "cs": "Czech",
         "cy": "Welsh",
         "da": "Danish",
         "de": "German",
         "el": "Greek",
         "en": "English",
         "es": "Spanish",
         "et": "Estonian",
         "eu": "Basque",
         "fa": "Persian",
         "fi": "Finnish",
         "fo": "Faroese",
         "fr": "French",
         "gl": "Galician",
         "gu": "Gujarati",
         "ha": "Hausa",
         "haw": "Hawaiian",
         "he": "Hebrew",
         "hi": "Hindi",
         "hr": "Croatian",
         "ht": "Haitian Creole",
         "hu": "Hungarian",
         "hy": "Armenian",
         "id": "Indonesian",
         "is": "Icelandic",
         "it": "Italian",
         "ja": "Japanese",
         "jw": "Javanese",
         "ka": "Georgian",
         "kk": "Kazakh",
         "km": "Khmer",
         "kn": "Kannada",
         "ko": "Korean",
         "la": "Latin",
         "lb": "Luxembourgish",
         "ln": "Lingala",
         "lo": "Lao",
         "lt": "Lithuanian",
         "lv": "Latvian",
         "mg": "Malagasy",
         "mi": "Maori",
         "mk": "Macedonian",
         "ml": "Malayalam",
         "mn": "Mongolian",
         "mr": "Marathi",
         "ms": "Malay",
         "mt": "Maltese",
         "my": "Myanmar",
         "ne": "Nepali",
         "nl": "Dutch",
         "nn": "Norwegian Nynorsk",
         "no": "Norwegian",
         "oc": "Occitan",
         "pa": "Punjabi",
         "pl": "Polish",
         "ps": "Pashto",
         "pt": "Portuguese",
         "ro": "Romanian",
         "ru": "Russian",
         "sa": "Sanskrit",
         "sd": "Sindhi",
         "si": "Sinhala",
         "sk": "Slovak",
         "sl": "Slovenian",
         "sn": "Shona",
         "so": "Somali",
         "sq": "Albanian",
         "sr": "Serbian",
         "su": "Sundanese",
         "sv": "Swedish",
         "sw": "Swahili",
         "ta": "Tamil",
         "te": "Telugu",
         "tg": "Tajik",
         "th": "Thai",
         "tk": "Turkmen",
         "tl": "Tagalog",
         "tr": "Turkish",
         "tt": "Tatar",
         "uk": "Ukrainian",
         "ur": "Urdu",
         "uz": "Uzbek",
         "vi": "Vietnamese",
         "yi": "Yiddish",
         "yo": "Yoruba",
         "yue": "Cantonese",
         "zh": "Chinese",
     ]
 }
