
import Foundation
import AppKit
import SwiftData
import os

class VoiceInkCSVExportService {
    private let logger = Logger(subsystem: "com.fightingentropy.voiceink", category: "VoiceInkCSVExportService")

    @MainActor
    func exportTranscriptionsToCSV(transcriptions: [Transcription]) {
        let csvString = generateCSV(for: transcriptions)

        let savePanel = NSSavePanel()
        savePanel.allowedContentTypes = [.commaSeparatedText]
        savePanel.nameFieldStringValue = "VoiceInk-transcription.csv"

        savePanel.begin { [logger] result in
            if result == .OK, let url = savePanel.url {
                do {
                    try csvString.write(to: url, atomically: true, encoding: .utf8)
                } catch {
                    logger.error("Error writing CSV file: \(error.localizedDescription, privacy: .public)")
                }
            }
        }
    }
    
    private func generateCSV(for transcriptions: [Transcription]) -> String {
        var csvString = "Transcript,Transcription Model,Transcription Time,Timestamp,Duration\n"

        for transcription in transcriptions {
            let text = escapeCSVString(transcription.text)
            let transcriptionModel = escapeCSVString(transcription.transcriptionModelName ?? "")
            let transcriptionTime = transcription.transcriptionDuration ?? 0
            let timestamp = transcription.timestamp.ISO8601Format()
            let duration = transcription.duration

            let row = "\(text),\(transcriptionModel),\(transcriptionTime),\(timestamp),\(duration)\n"
            csvString.append(row)
        }

        return csvString
    }

    private func escapeCSVString(_ string: String) -> String {
        let escapedString = string.replacingOccurrences(of: "\"", with: "\"\"")
        if escapedString.contains(",") || escapedString.contains("\n") {
            return "\"\(escapedString)\""
        }
        return escapedString
    }
}
