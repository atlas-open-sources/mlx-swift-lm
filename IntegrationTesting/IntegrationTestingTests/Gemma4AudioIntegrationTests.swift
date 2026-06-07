// Copyright © 2026 Apple Inc.
//
// Real end-to-end Gemma 4 AUDIO inference. Downloads an audio-capable Gemma 4
// VLM and asks it to transcribe real speech clips, exercising the full audio
// path: AVAssetReader PCM → mel feature extractor (fft_length=512, periodic Hann
// window, semicausal padding, bin-index mel filterbank) → Conformer audio tower
// → begin/end-of-audio prompt splice → text.
//
// Speech clips are generated offline with macOS `say` and committed under
// Tests/MLXLMTests/Resources/:
//   gemma_speech_test.wav  — "The quick brown fox jumps over the lazy dog near the river bank."
//   gemma_speech_test2.wav — "She sells sea shells by the sea shore on a bright summer morning."
//   gemma_speech_long.wav  — "The weather forecast predicts heavy rain tomorrow afternoon ..."
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

private let resources =
    "/Users/timapple/Documents/Guest/mlx-swift-lm/Tests/MLXLMTests/Resources"

/// One speech clip + the distinctive words a correct transcription must recover.
struct SpeechCase: Sendable, CustomStringConvertible {
    let file: String
    let expected: [String]
    var description: String { file }
}

// Clear, natural-cadence sentences that gemma-4 E4B transcribes reliably.
// NOTE: we deliberately do NOT assert verbatim on a tongue-twister
// (gemma_speech_test2.wav) or on E2B — cross-checked against the reference impl
// (VincentGourbin/gemma-4-swift-mlx), both mis-hear the synthetic tongue-twister
// and E2B emits ~empty for verbatim ASR. Those are model limitations on
// synthetic TTS, not integration bugs.
private let speechCases: [SpeechCase] = [
    .init(
        file: "gemma_speech_test.wav",
        expected: ["quick", "brown", "fox", "lazy", "dog", "river"]),
    .init(
        file: "gemma_speech_long.wav",
        expected: ["weather", "rain", "forecast", "afternoon", "breeze", "evening", "sky"]),
]

@Suite(.serialized)
struct Gemma4AudioIntegrationTests {

    private func transcribe(model: String, clip: SpeechCase) async throws -> String {
        let container = try await models.vlmContainer(for: ModelConfiguration(id: model))
        let session = ChatSession(
            container, generateParameters: GenerateParameters(maxTokens: 120, temperature: 0))
        let url = URL(fileURLWithPath: "\(resources)/\(clip.file)")
        return try await session.respond(
            to: "Transcribe the speech in this audio clip.",
            images: [], videos: [], audios: [.url(url)])
    }

    private func assertRecovered(_ answer: String, _ clip: SpeechCase) {
        let lower = answer.lowercased()
        #expect(!lower.contains("<pad>"), "audio path regressed to a <pad> wall: \(answer)")
        let hits = clip.expected.filter { lower.contains($0) }
        #expect(
            hits.count >= 3,
            "[\(clip.file)] did not recover the spoken words (matched \(hits) in: \(answer))")
    }

    /// E4B verbatim transcription on clear, natural-cadence clips.
    @Test(arguments: speechCases)
    func gemma4_e4b_transcribes(_ clip: SpeechCase) async throws {
        let answer = try await transcribe(model: "mlx-community/gemma-4-e4b-it-4bit", clip: clip)
        print("🎙️ [e4b/\(clip.file)] \(answer)")
        assertRecovered(answer, clip)
    }
}
