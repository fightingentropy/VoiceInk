import Foundation

struct VoxtralTekkenSpec: Decodable, Sendable {
    struct Config: Decodable, Sendable {
        let pattern: String
        let numVocabTokens: Int
        let defaultVocabSize: Int
        let defaultNumSpecialTokens: Int
        let version: String

        private enum CodingKeys: String, CodingKey {
            case pattern
            case numVocabTokens = "num_vocab_tokens"
            case defaultVocabSize = "default_vocab_size"
            case defaultNumSpecialTokens = "default_num_special_tokens"
            case version
        }
    }

    struct Token: Decodable, Sendable {
        let rank: Int
        let tokenBytes: String?
        let tokenString: String?
        let isControl: Bool?

        private enum CodingKeys: String, CodingKey {
            case rank
            case tokenBytes = "token_bytes"
            case tokenString = "token_str"
            case isControl = "is_control"
        }
    }

    struct Audio: Decodable, Sendable {
        let samplingRate: Int
        let frameRate: Double
        let transcriptionDelayMS: Int
        let streamingLookAheadMS: Double
        let streamingLookBackMS: Double
        let streamingLeftPadTokens: Int
        let transcriptionFormat: String

        private enum CodingKeys: String, CodingKey {
            case samplingRate = "sampling_rate"
            case frameRate = "frame_rate"
            case transcriptionDelayMS = "transcription_delay_ms"
            case streamingLookAheadMS = "streaming_look_ahead_ms"
            case streamingLookBackMS = "streaming_look_back_ms"
            case streamingLeftPadTokens = "streaming_n_left_pad_tokens"
            case transcriptionFormat = "transcription_format"
        }
    }

    let config: Config
    let vocab: [Token]
    let specialTokens: [Token]
    let audio: Audio

    private enum CodingKeys: String, CodingKey {
        case config
        case vocab
        case specialTokens = "special_tokens"
        case audio
    }
}

struct VoxtralStreamingPrompt: Sendable {
    let tokenIDs: [Int]
    let delayTokenCount: Int
}

struct VoxtralTekkenTokenizer: Sendable {
    private let spec: VoxtralTekkenSpec
    private let vocabulary: [VoxtralTekkenSpec.Token]
    private let specialTokenCount: Int
    private let specialTokensByID: [Int: VoxtralTekkenSpec.Token]
    private let specialTokenIDsByName: [String: Int]

    init(directoryURL: URL) throws {
        let specURL = directoryURL.appendingPathComponent("tekken.json")
        let data = try Data(contentsOf: specURL)
        let spec = try JSONDecoder().decode(VoxtralTekkenSpec.self, from: data)
        self.spec = spec
        self.specialTokenCount = spec.config.defaultNumSpecialTokens

        let innerVocabularyCount = max(0, spec.config.defaultVocabSize - specialTokenCount)
        self.vocabulary = Array(spec.vocab.prefix(innerVocabularyCount))

        var specialTokensByID: [Int: VoxtralTekkenSpec.Token] = [:]
        var specialTokenIDsByName: [String: Int] = [:]
        for token in spec.specialTokens {
            specialTokensByID[token.rank] = token
            if let tokenName = token.tokenString {
                specialTokenIDsByName[tokenName] = token.rank
            }
        }
        self.specialTokensByID = specialTokensByID
        self.specialTokenIDsByName = specialTokenIDsByName
    }

    var audioConfiguration: VoxtralTekkenSpec.Audio {
        spec.audio
    }

    var bosID: Int {
        specialTokenIDsByName["<s>"] ?? 1
    }

    var eosID: Int {
        specialTokenIDsByName["</s>"] ?? 2
    }

    func specialTokenID(named name: String) -> Int? {
        specialTokenIDsByName[name]
    }

    func buildStreamingPrompt(nLeftPadTokens: Int? = nil, delayTokens: Int = 6) throws -> VoxtralStreamingPrompt {
        guard let streamingPadID = specialTokenID(named: "[STREAMING_PAD]") else {
            throw VoxtralTekkenTokenizerError.missingStreamingPadToken
        }

        let leftPadCount = nLeftPadTokens ?? spec.audio.streamingLeftPadTokens
        let prefixLength = leftPadCount + delayTokens
        let tokenIDs = [bosID] + Array(repeating: streamingPadID, count: prefixLength)
        return VoxtralStreamingPrompt(tokenIDs: tokenIDs, delayTokenCount: delayTokens)
    }

    func decode(_ tokenIDs: [Int], ignoreSpecialTokens: Bool = true) -> String {
        var data = Data()
        var fragments: [String] = []

        for tokenID in tokenIDs {
            if tokenID < specialTokenCount {
                if !ignoreSpecialTokens, let token = specialTokensByID[tokenID] {
                    fragments.append(token.tokenString ?? "")
                }
                continue
            }

            let vocabularyIndex = tokenID - specialTokenCount
            guard vocabulary.indices.contains(vocabularyIndex) else {
                continue
            }

            let token = vocabulary[vocabularyIndex]
            if let tokenData = tokenPayload(for: token) {
                data.append(tokenData)
            } else if let tokenString = token.tokenString {
                fragments.append(tokenString)
            }
        }

        if !data.isEmpty {
            fragments.insert(String(decoding: data, as: UTF8.self), at: 0)
        }

        return fragments.joined()
    }

    private func tokenPayload(for token: VoxtralTekkenSpec.Token) -> Data? {
        if let tokenBytes = token.tokenBytes, let decoded = Data(base64Encoded: tokenBytes) {
            return decoded
        }

        guard let tokenString = token.tokenString, !tokenString.isEmpty else {
            return nil
        }

        return tokenString.data(using: .utf8)
    }
}

enum VoxtralTekkenTokenizerError: LocalizedError {
    case missingStreamingPadToken

    var errorDescription: String? {
        switch self {
        case .missingStreamingPadToken:
            return "The Voxtral tokenizer is missing the [STREAMING_PAD] token."
        }
    }
}
