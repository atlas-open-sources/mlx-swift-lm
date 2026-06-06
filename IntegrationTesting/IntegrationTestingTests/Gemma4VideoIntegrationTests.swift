// Copyright © 2026 Apple Inc.
//
// Real end-to-end Gemma 4 VIDEO inference. Downloads gemma-4-e4b-it-4bit and
// asks it to describe a real video clip, exercising the PR #256 video tower
// end to end (frame sampling via MediaProcessing → vision tower → text).
//
// Run:
//   xcodebuild test -project IntegrationTesting.xcodeproj \
//     -scheme IntegrationTesting -destination 'platform=macOS' \
//     -only-testing:IntegrationTestingTests/Gemma4VideoIntegrationTests

import Foundation
import HuggingFace
import IntegrationTestHelpers
import MLXHuggingFace
import MLXLMCommon
import Testing
import Tokenizers

private let models = IntegrationTestModels(
    downloader: #hubDownloader(),
    tokenizerLoader: #huggingFaceTokenizerLoader()
)

@Suite(.serialized)
struct Gemma4VideoIntegrationTests {

    // The repo already ships a small real clip for VLM tests.
    private static let videoURL = URL(
        fileURLWithPath:
            "/Users/timapple/Documents/Guest/mlx-swift-lm/Tests/MLXLMTests/Resources/1080p_30.mov"
    )

    @Test func gemma4_e4b_describesVideo() async throws {
        let container = try await models.vlmContainer(
            for: ModelConfiguration(id: "mlx-community/gemma-4-e4b-it-4bit")
        )

        let session = ChatSession(
            container,
            generateParameters: GenerateParameters(maxTokens: 120, temperature: 0)
        )

        let answer = try await session.respond(
            to: "Describe what happens in this video in one or two sentences.",
            images: [],
            videos: [.url(Self.videoURL)],
            audios: []
        )

        print("🎬 Gemma 4 video description:\n\(answer)")
        #expect(!answer.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
    }
}
