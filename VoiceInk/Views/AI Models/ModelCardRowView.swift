import SwiftUI
import AppKit

struct ModelCardRowView: View {
    let model: any TranscriptionModel
    let parakeetModelManager: ParakeetModelManager
    let transcriptionModelManager: TranscriptionModelManager
    let isDownloaded: Bool
    let isCurrent: Bool
    let downloadProgress: [String: Double]
    let modelURL: URL?

    // Actions
    var deleteAction: () -> Void
    var setDefaultAction: () -> Void
    var downloadAction: () -> Void
    var editAction: ((CustomCloudModel) -> Void)?
    var body: some View {
        Group {
            switch model.provider {
            case .local:
                if let localModel = model as? LocalModel {
                    LocalModelCardView(
                        model: localModel,
                        isDownloaded: isDownloaded,
                        isCurrent: isCurrent,
                        downloadProgress: downloadProgress,
                        modelURL: modelURL,
                        deleteAction: deleteAction,
                        setDefaultAction: setDefaultAction,
                        downloadAction: downloadAction
                    )
                }
            case .parakeet:
                if let parakeetModel = model as? ParakeetModel {
                    ParakeetModelCardRowView(
                        model: parakeetModel,
                        parakeetModelManager: parakeetModelManager,
                        transcriptionModelManager: transcriptionModelManager
                    )
                }
            case .nativeApple:
                if let nativeAppleModel = model as? NativeAppleModel {
                    NativeAppleModelCardView(
                        model: nativeAppleModel,
                        isCurrent: isCurrent,
                        setDefaultAction: setDefaultAction
                    )
                }
            case .localVoxtral:
                if let localVoxtralModel = model as? LocalVoxtralModel {
                    LocalVoxtralModelCardView(
                        model: localVoxtralModel,
                        isCurrent: isCurrent,
                        deleteAction: deleteAction,
                        setDefaultAction: setDefaultAction
                    )
                }
            case .cohereTranscribe:
                if let cohereModel = model as? LocalCohereTranscribeModel {
                    LocalCohereTranscribeModelCardView(
                        model: cohereModel,
                        isCurrent: isCurrent,
                        deleteAction: deleteAction,
                        setDefaultAction: setDefaultAction
                    )
                }
            case .elevenLabs:
                if let cloudModel = model as? CloudModel {
                    CloudModelCardView(
                        model: cloudModel,
                        isCurrent: isCurrent,
                        setDefaultAction: setDefaultAction
                    )
                }
            case .custom:
                if let customModel = model as? CustomCloudModel {
                    CustomModelCardView(
                        model: customModel,
                        isCurrent: isCurrent,
                        setDefaultAction: setDefaultAction,
                        deleteAction: deleteAction,
                        editAction: editAction ?? { _ in }
                    )
                }
            }
        }
    }
}
