// Copyright © 2026 Apple Inc.

import Foundation

/// Parser for the ATEM function-call protocol emitted by Meta Muse models.
public struct ATEMToolCallParser: ToolCallParser, Sendable {
    public let startTag: String? = "to=self<|message|>"
    public let endTag: String? = "</atem:function_calls>"

    public init() {}

    public func parse(content: String, tools _: [[String: any Sendable]]?) -> ToolCall? {
        parseInvocations(content).first
    }

    public func parseEOS(
        _ toolCallBuffer: String,
        tools _: [[String: any Sendable]]?
    ) -> [ToolCall] {
        parseInvocations(toolCallBuffer)
    }

    private func parseInvocations(_ content: String) -> [ToolCall] {
        guard
            let invokeRegex = try? NSRegularExpression(
                pattern: #"<atem:invoke\b[^>]*?\bname=\"([^\"]+)\">(.*?)</atem:invoke>"#,
                options: [.dotMatchesLineSeparators]
            ),
            let parameterRegex = try? NSRegularExpression(
                pattern:
                    #"<atem:parameter\b[^>]*?\bname=\"([^\"]+)\"[^>]*?>(.*?)</atem:parameter>"#,
                options: [.dotMatchesLineSeparators]
            )
        else { return [] }

        let contentRange = NSRange(content.startIndex..., in: content)
        return invokeRegex.matches(in: content, range: contentRange).compactMap { match in
            guard let nameRange = Range(match.range(at: 1), in: content),
                let bodyRange = Range(match.range(at: 2), in: content)
            else { return nil }

            let body = String(content[bodyRange])
            var arguments: [String: any Sendable] = [:]
            let bodySearchRange = NSRange(body.startIndex..., in: body)
            for parameter in parameterRegex.matches(in: body, range: bodySearchRange) {
                guard let parameterNameRange = Range(parameter.range(at: 1), in: body),
                    let valueRange = Range(parameter.range(at: 2), in: body)
                else { continue }
                let value = String(body[valueRange])
                let decoded = value.data(using: .utf8).flatMap {
                    try? JSONDecoder().decode(JSONValue.self, from: $0)
                }
                arguments[String(body[parameterNameRange])] = decoded?.sendableValue ?? value
            }

            return ToolCall(
                function: .init(name: String(content[nameRange]), arguments: arguments))
        }
    }
}
