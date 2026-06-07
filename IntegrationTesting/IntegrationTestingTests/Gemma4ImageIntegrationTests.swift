// Copyright © 2026 Apple Inc.
//
// Real end-to-end Gemma 4 still-IMAGE inference — a regression guard that the
// audio/video work didn't disturb the vision path (which routes through the same
// getInputEmbeddings multimodal scatter). Downloads gemma-4-e4b-it-4bit and asks
// it to describe a committed photo of a red Citroën 2CV in front of foliage.
//
// Run:
//   xcodebuild test -project IntegrationTesting.xcodeproj \
//     -scheme IntegrationTesting -destination 'platform=macOS' \
//     -only-testing:IntegrationTestingTests/Gemma4ImageIntegrationTests

import Foundation
import HuggingFace
import IntegrationTestHelpers
import MLXHuggingFace
import MLXLMCommon
import Testing
import Tokenizers

private let imageModels = IntegrationTestModels(
    downloader: #hubDownloader(),
    tokenizerLoader: #huggingFaceTokenizerLoader()
)

@Suite(.serialized)
struct Gemma4ImageIntegrationTests {

    private static let imageURL = URL(
        fileURLWithPath:
            "/Users/timapple/Documents/Guest/mlx-swift-lm/Tests/MLXLMTests/Resources/gemma_image_test.jpg"
    )

    @Test func gemma4_e4b_describesImage() async throws {
        let container = try await imageModels.vlmContainer(
            for: ModelConfiguration(id: "mlx-community/gemma-4-e4b-it-4bit"))
        let session = ChatSession(
            container, generateParameters: GenerateParameters(maxTokens: 120, temperature: 0))

        let answer = try await session.respond(
            to: "Describe this image in one or two sentences.",
            images: [.url(Self.imageURL)], videos: [], audios: [])

        print("🖼️ Gemma 4 image description:\n\(answer)")
        let lower = answer.lowercased()
        #expect(!lower.contains("<pad>"), "image path regressed to a <pad> wall")
        // The photo is a red car in front of green foliage; require a real
        // visual cue so a generic reply can't pass.
        let cues = ["car", "vehicle", "automobile", "red", "citro", "tree", "green", "parked"]
        #expect(
            cues.contains(where: { lower.contains($0) }),
            "image description lacks any visual cue: \(answer)")
    }
}
