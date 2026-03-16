import Foundation

enum LocalVoxtralConfiguration {
    static let endpointKey = "LocalVoxtralEndpoint"
    static let modelNameKey = "LocalVoxtralModelName"
    static let autoManageKey = "LocalVoxtralAutoManageServer"

    static let defaultEndpoint = "ws://127.0.0.1:8000/v1/realtime"
    static let defaultModelName = "T0mSIlver/Voxtral-Mini-4B-Realtime-2602-MLX-4bit"

    static var endpointString: String {
        let stored = UserDefaults.standard.string(forKey: endpointKey)?.trimmingCharacters(in: .whitespacesAndNewlines)
        return stored?.isEmpty == false ? stored! : defaultEndpoint
    }

    static var endpointURL: URL? {
        URL(string: endpointString)
    }

    static var modelName: String {
        let stored = UserDefaults.standard.string(forKey: modelNameKey)?.trimmingCharacters(in: .whitespacesAndNewlines)
        return stored?.isEmpty == false ? stored! : defaultModelName
    }

    static var autoManageServer: Bool {
        UserDefaults.standard.object(forKey: autoManageKey) as? Bool ?? true
    }

    static func launchCommand(endpointString: String = LocalVoxtralConfiguration.endpointString, modelName: String = LocalVoxtralConfiguration.modelName) -> String {
        let binding = serverBinding(from: endpointString) ?? (host: "127.0.0.1", port: 8000)
        return #"uvx --from "git+https://github.com/T0mSIlver/voxmlx.git[server]" voxmlx-serve --host \#(binding.host) --port \#(binding.port) --model \#(modelName)"#
    }

    static func healthCheckURL(from realtimeEndpoint: String) -> URL? {
        guard let url = URL(string: realtimeEndpoint),
              var components = URLComponents(url: url, resolvingAgainstBaseURL: false)
        else {
            return nil
        }

        switch components.scheme?.lowercased() {
        case "ws":
            components.scheme = "http"
        case "wss":
            components.scheme = "https"
        case "http", "https":
            break
        default:
            return nil
        }

        components.path = "/health"
        components.query = nil
        components.fragment = nil
        return components.url
    }

    static func serverBinding(from realtimeEndpoint: String) -> (host: String, port: Int)? {
        guard let url = URL(string: realtimeEndpoint),
              let host = url.host?.trimmingCharacters(in: .whitespacesAndNewlines),
              !host.isEmpty
        else {
            return nil
        }

        let normalizedHost = host.lowercased()
        guard ["127.0.0.1", "localhost", "::1"].contains(normalizedHost) else {
            return nil
        }

        let port = url.port ?? (url.scheme?.lowercased() == "wss" ? 443 : 8000)
        return (host, port)
    }
}
