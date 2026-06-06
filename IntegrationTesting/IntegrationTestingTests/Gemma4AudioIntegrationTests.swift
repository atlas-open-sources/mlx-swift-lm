// Copyright © 2026 Apple Inc.
//
// Real end-to-end Gemma 4 AUDIO inference. Downloads gemma-4-e4b-it-4bit (an
// audio-capable variant) and asks it to transcribe a real speech clip,
// exercising the PR #192 Conformer audio tower end to end (AVAssetReader PCM →
// mel feature extractor → audio tower → text).
//
// The speech clip is generated offline with macOS `say` and committed at
// Tests/MLXLMTests/Resources/gemma_speech_test.aiff:
//   "The quick brown fox jumps over the lazy dog near the river bank."
//
// Run:
//   xcodebuild test -project IntegrationTesting.xcodeproj \
//     -scheme IntegrationTesting -destination 'platform=macOS' \
//     -only-testing:IntegrationTestingTests/Gemma4AudioIntegrationTests

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
struct Gemma4AudioIntegrationTests {

    private static let audioURL = URL(
        fileURLWithPath:
            "/Users/timapple/Documents/Guest/mlx-swift-lm/Tests/MLXLMTests/Resources/gemma_speech_test.aiff"
    )

    @Test func gemma4_e4b_transcribesAudio() async throws {
        let container = try await models.vlmContainer(
            for: ModelConfiguration(id: "mlx-community/gemma-4-e4b-it-4bit")
        )

        let session = ChatSession(
            container,
            generateParameters: GenerateParameters(maxTokens: 120, temperature: 0)
        )

        let answer = try await session.respond(
            to: "Transcribe the speech in this audio clip.",
            images: [],
            videos: [],
            audios: [.url(Self.audioURL)]
        )

        print("🎙️ Gemma 4 audio transcription:\n\(answer)")
        #expect(!answer.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
    }
}
