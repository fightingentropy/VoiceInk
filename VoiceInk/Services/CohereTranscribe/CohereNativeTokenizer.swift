import Foundation

private enum CohereSentencePiecePieceType: Int {
    case normal = 1
    case unknown = 2
    case control = 3
    case userDefined = 4
    case unused = 5
    case byte = 6
}

private enum CohereSentencePieceModelError: Error, CustomStringConvertible {
    case malformedVarint
    case truncatedField
    case unsupportedWireType(UInt64)
    case missingVocabulary

    var description: String {
        switch self {
        case .malformedVarint:
            "Malformed SentencePiece protobuf varint"
        case .truncatedField:
            "Truncated SentencePiece protobuf field"
        case .unsupportedWireType(let wireType):
            "Unsupported SentencePiece protobuf wire type: \(wireType)"
        case .missingVocabulary:
            "SentencePiece model did not contain any vocabulary pieces"
        }
    }
}

private struct CohereSentencePieceProtobufReader {
    let data: Data
    var index: Data.Index

    init(_ data: Data) {
        self.data = data
        self.index = data.startIndex
    }

    var isAtEnd: Bool { index >= data.endIndex }

    mutating func readVarint() throws -> UInt64 {
        var value: UInt64 = 0
        var shift: UInt64 = 0

        while index < data.endIndex, shift < 64 {
            let byte = data[index]
            index = data.index(after: index)
            value |= UInt64(byte & 0x7f) << shift
            if byte & 0x80 == 0 {
                return value
            }
            shift += 7
        }

        throw CohereSentencePieceModelError.malformedVarint
    }

    mutating func readLengthDelimited() throws -> Data {
        let length = Int(try readVarint())
        guard let end = data.index(index, offsetBy: length, limitedBy: data.endIndex) else {
            throw CohereSentencePieceModelError.truncatedField
        }
        let slice = data[index..<end]
        index = end
        return Data(slice)
    }

    mutating func readFixed32() throws -> UInt32 {
        guard let end = data.index(index, offsetBy: 4, limitedBy: data.endIndex) else {
            throw CohereSentencePieceModelError.truncatedField
        }

        let slice = data[index..<end]
        index = end
        return slice.enumerated().reduce(UInt32(0)) { partial, entry in
            partial | (UInt32(entry.element) << (entry.offset * 8))
        }
    }

    mutating func skipField(wireType: UInt64) throws {
        switch wireType {
        case 0:
            _ = try readVarint()
        case 1:
            guard let end = data.index(index, offsetBy: 8, limitedBy: data.endIndex) else {
                throw CohereSentencePieceModelError.truncatedField
            }
            index = end
        case 2:
            _ = try readLengthDelimited()
        case 5:
            _ = try readFixed32()
        default:
            throw CohereSentencePieceModelError.unsupportedWireType(wireType)
        }
    }
}

private struct CohereSentencePieceToken {
    let token: String
    let score: Float
    let type: CohereSentencePiecePieceType
}

private struct CohereSentencePieceModelParser {
    static func parsePieces(from data: Data) throws -> (pieces: [CohereSentencePieceToken], unknownTokenId: Int) {
        var reader = CohereSentencePieceProtobufReader(data)
        var pieces: [CohereSentencePieceToken] = []
        var unknownTokenId: Int?

        while !reader.isAtEnd {
            let key = try reader.readVarint()
            let fieldNumber = Int(key >> 3)
            let wireType = key & 0x7

            if fieldNumber == 1, wireType == 2 {
                let pieceData = try reader.readLengthDelimited()
                if let piece = try parsePiece(from: pieceData) {
                    if piece.type == .unknown, unknownTokenId == nil {
                        unknownTokenId = pieces.count
                    }
                    pieces.append(piece)
                }
            } else {
                try reader.skipField(wireType: wireType)
            }
        }

        guard !pieces.isEmpty else {
            throw CohereSentencePieceModelError.missingVocabulary
        }

        let resolvedUnknownId = unknownTokenId
            ?? pieces.firstIndex(where: { $0.token == "<unk>" })
            ?? 0
        return (pieces, resolvedUnknownId)
    }

    private static func parsePiece(from data: Data) throws -> CohereSentencePieceToken? {
        var reader = CohereSentencePieceProtobufReader(data)
        var token: String?
        var score: Float = 0
        var type: CohereSentencePiecePieceType = .normal

        while !reader.isAtEnd {
            let key = try reader.readVarint()
            let fieldNumber = Int(key >> 3)
            let wireType = key & 0x7

            switch (fieldNumber, wireType) {
            case (1, 2):
                let tokenData = try reader.readLengthDelimited()
                token = String(decoding: tokenData, as: UTF8.self)
            case (2, 5):
                score = Float(bitPattern: try reader.readFixed32())
            case (3, 0):
                let rawType = Int(try reader.readVarint())
                type = CohereSentencePiecePieceType(rawValue: rawType) ?? .normal
            default:
                try reader.skipField(wireType: wireType)
            }
        }

        guard let token else { return nil }
        return CohereSentencePieceToken(token: token, score: score, type: type)
    }
}

private final class CohereTokenLatticeNode {
    let tokenId: Int
    let startOffset: Int
    let length: Int
    let score: Float

    var prev: CohereTokenLatticeNode?
    var backtraceScore: Float = 0

    init(
        tokenId: Int,
        startOffset: Int,
        length: Int,
        score: Float,
        prev: CohereTokenLatticeNode? = nil,
        backtraceScore: Float = 0
    ) {
        self.tokenId = tokenId
        self.startOffset = startOffset
        self.length = length
        self.score = score
        self.prev = prev
        self.backtraceScore = backtraceScore
    }

    func clone() -> CohereTokenLatticeNode {
        CohereTokenLatticeNode(
            tokenId: tokenId,
            startOffset: startOffset,
            length: length,
            score: score,
            prev: prev,
            backtraceScore: backtraceScore
        )
    }
}

private struct CohereTokenLattice {
    let sentence: String
    let bosTokenId: Int
    let eosTokenId: Int

    var nodes: [CohereTokenLatticeNode] = []
    var beginNodes: [[CohereTokenLatticeNode]]
    var endNodes: [[CohereTokenLatticeNode]]

    var count: Int { sentence.count }

    init(sentence: String, bosTokenId: Int, eosTokenId: Int) {
        self.sentence = sentence
        self.bosTokenId = bosTokenId
        self.eosTokenId = eosTokenId

        beginNodes = Array(repeating: [], count: sentence.count + 1)
        endNodes = Array(repeating: [], count: sentence.count + 1)

        let bos = CohereTokenLatticeNode(tokenId: bosTokenId, startOffset: 0, length: 0, score: 0)
        let eos = CohereTokenLatticeNode(tokenId: eosTokenId, startOffset: sentence.count, length: 0, score: 0)

        nodes.append(bos)
        nodes.append(eos)

        beginNodes[sentence.count].append(eos)
        endNodes[0].append(bos)
    }

    mutating func insert(startOffset: Int, length: Int, score: Float, tokenId: Int) {
        let node = CohereTokenLatticeNode(tokenId: tokenId, startOffset: startOffset, length: length, score: score)
        beginNodes[startOffset].append(node)
        endNodes[startOffset + length].append(node)
        nodes.append(node)
    }

    func piece(_ node: CohereTokenLatticeNode) -> any StringProtocol {
        let start = sentence.index(sentence.startIndex, offsetBy: node.startOffset)
        let end = sentence.index(start, offsetBy: node.length)
        return sentence[start..<end]
    }

    func viterbi() -> [CohereTokenLatticeNode] {
        for offset in 0...count {
            guard beginNodes[offset].count > 0 else { return [] }

            for rnode in beginNodes[offset] {
                rnode.prev = nil
                var bestScore: Float = 0
                var bestNode: CohereTokenLatticeNode?
                for lnode in endNodes[offset] {
                    let score = lnode.backtraceScore + rnode.score
                    if bestNode == nil || score > bestScore {
                        bestNode = lnode.clone()
                        bestScore = score
                    }
                }

                if let bestNode {
                    rnode.prev = bestNode
                    rnode.backtraceScore = bestScore
                }
            }
        }

        let root = beginNodes[count][0]
        guard let prev = root.prev else { return [] }

        var result: [CohereTokenLatticeNode] = []
        var node = prev
        while node.prev != nil {
            result.append(node.clone())
            node = node.prev!
        }
        return result.reversed()
    }
}

private final class CohereTrieNode {
    var children: [Character: CohereTrieNode] = [:]
    var isEnd = false
}

private final class CohereTrie: @unchecked Sendable {
    private let root = CohereTrieNode()

    func append(contentsOf tokens: [String]) {
        for token in tokens {
            insert(token)
        }
    }

    private func insert(_ token: String) {
        var node = root
        for character in token {
            if node.children[character] == nil {
                node.children[character] = CohereTrieNode()
            }
            node = node.children[character]!
        }
        node.isEnd = true
    }

    func commonPrefixSearch(_ substring: Substring) -> [String] {
        var results: [String] = []
        var node = root
        var current = ""

        for character in substring {
            guard let next = node.children[character] else { break }
            current.append(character)
            node = next
            if node.isEnd {
                results.append(current)
            }
        }

        return results
    }
}

struct CohereNativeTokenizer: Sendable {
    private let vocabulary: [CohereSentencePieceToken]
    private let unknownTokenId: Int
    private let unknownTokenScore: Float
    private let tokensToIds: [String: Int]
    private let trie: CohereTrie

    init(sentencePieceModelData: Data) throws {
        let parsed = try CohereSentencePieceModelParser.parsePieces(from: sentencePieceModelData)
        self.vocabulary = parsed.pieces
        self.unknownTokenId = parsed.unknownTokenId
        let minimumScore = vocabulary.reduce(Float.greatestFiniteMagnitude) { min($0, $1.score) }
        self.unknownTokenScore = minimumScore - 10

        var mapping: [String: Int] = [:]
        mapping.reserveCapacity(vocabulary.count)
        for (index, token) in vocabulary.enumerated() {
            mapping[token.token] = index
        }
        self.tokensToIds = mapping

        let trie = CohereTrie()
        trie.append(contentsOf: vocabulary.map(\.token))
        self.trie = trie
    }

    init(modelURL: URL) throws {
        try self.init(sentencePieceModelData: Data(contentsOf: modelURL))
    }

    var vocabularySize: Int {
        vocabulary.count
    }

    func tokenID(for token: String) -> Int? {
        tokensToIds[token]
    }

    func encode(_ text: String) -> [Int] {
        let preprocessed = applyMetaspace(to: text)
        var lattice = CohereTokenLattice(
            sentence: preprocessed,
            bosTokenId: unknownTokenId,
            eosTokenId: unknownTokenId
        )

        let sentence = lattice.sentence
        var beginPosition = 0
        while beginPosition < sentence.count {
            let tokenLength = 1
            var hasSingleNode = false
            let beginIndex = sentence.index(sentence.startIndex, offsetBy: beginPosition)

            for token in trie.commonPrefixSearch(sentence[beginIndex...]) {
                guard let tokenId = tokensToIds[token] else { continue }
                lattice.insert(
                    startOffset: beginPosition,
                    length: token.count,
                    score: vocabulary[tokenId].score,
                    tokenId: tokenId
                )
                if !hasSingleNode, token.count == tokenLength {
                    hasSingleNode = true
                }
            }

            if !hasSingleNode {
                lattice.insert(
                    startOffset: beginPosition,
                    length: tokenLength,
                    score: unknownTokenScore,
                    tokenId: unknownTokenId
                )
            }

            beginPosition += tokenLength
        }

        let path = lattice.viterbi()
        var ids: [Int] = []
        ids.reserveCapacity(path.count)

        for node in path {
            if node.tokenId == unknownTokenId {
                let piece = lattice.piece(node)
                for byte in piece.utf8 {
                    ids.append(byteFallbackMap[byte] ?? unknownTokenId)
                }
            } else {
                ids.append(node.tokenId)
            }
        }

        return ids
    }

    func decode(_ ids: [Int]) -> String {
        var bytes: [UInt8] = []
        var pieces: [String] = []

        for id in ids {
            guard id >= 0, id < vocabulary.count else { continue }
            let token = vocabulary[id]
            if token.type == .control || token.type == .unused {
                continue
            }

            let rawToken = token.token
            if rawToken.hasPrefix("<0x"), rawToken.hasSuffix(">"), rawToken.count == 6 {
                let hex = rawToken.dropFirst(3).dropLast(1)
                if let byte = UInt8(hex, radix: 16) {
                    bytes.append(byte)
                }
                continue
            }

            if !bytes.isEmpty {
                if let decodedBytes = String(bytes: bytes, encoding: .utf8) {
                    pieces.append(decodedBytes)
                }
                bytes.removeAll()
            }

            pieces.append(rawToken)
        }

        if !bytes.isEmpty, let decodedBytes = String(bytes: bytes, encoding: .utf8) {
            pieces.append(decodedBytes)
        }

        let joined = pieces.joined()
        return joined.replacingOccurrences(of: "▁", with: " ")
            .trimmingCharacters(in: .whitespaces)
    }

    private var byteFallbackMap: [UInt8: Int] {
        var map: [UInt8: Int] = [:]
        map.reserveCapacity(vocabulary.count)
        for (index, token) in vocabulary.enumerated() {
            let piece = token.token
            if piece.hasPrefix("<0x"), piece.hasSuffix(">"), piece.count == 6 {
                let hex = piece.dropFirst(3).dropLast(1)
                if let byte = UInt8(hex, radix: 16) {
                    map[byte] = index
                }
            }
        }
        return map
    }

    private func applyMetaspace(to text: String) -> String {
        let replaced = text.replacingOccurrences(of: " ", with: "▁")
        return "▁" + replaced
    }
}
