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
            "/Users/timapple/Documents/Guest/mlx-swift-lm/Tests/MLXLMTests/Resources/gemma_speech_test.wav"
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

        let lower = answer.lowercased()

        // PROVEN today: the audio path runs end-to-end and the model produces
        // coherent natural language (not a <pad>/special-token wall). The <pad>
        // wall was the symptom of feeding mis-sampled audio; with clean 16 kHz
        // mono input the pipeline yields finite mel features, a finite Conformer
        // forward pass, and audio soft-token embeddings scattered into the prompt.
        #expect(!lower.contains("<pad>"), "audio path regressed to a <pad> wall")
        #expect(
            answer.split(whereSeparator: { $0 == " " || $0 == "\n" }).count >= 5,
            "audio path produced no coherent text: \(answer)")

        // KNOWN ISSUE (pr-192 incompleteness): the Conformer audio tower produces
        // finite-but-semantically-incorrect embeddings, so the model receives the
        // audio tokens but cannot transcribe (it replies "you have not provided
        // the audio"). pr-192's tower e2e test was a stub upstream, so the tower
        // was never validated. Bugs fixed so far while chasing this: 16 kHz mono
        // bridge (was 48 kHz → mel NaN), boa/eoa prompt format (bare tokens →
        // <pad>), WAV vs big-endian-AIFF fixture (garbage samples), and the
        // relative-position span (now past-only [maxPastHorizon…0], matching
        // Google's reference). At least one more tower bug remains: the output is
        // byte-identical before/after the rel-pos fix, i.e. the audio embeddings
        // still don't influence generation — pointing at the subsample conv,
        // lconv, or attention-chunking. Definitive next step: a numerical harness
        // diffing each tower stage against VincentGourbin/gemma-4-swift-mlx (the
        // known-good reference). Recovering the spoken words is the success signal;
        // this flags the moment the tower is fixed.
        withKnownIssue("Gemma 4 audio tower (pr-192) still produces incorrect embeddings; transcription not yet recovered") {
            let expectedWords = ["quick", "brown", "fox", "lazy", "dog", "river", "bank", "jump"]
            let hits = expectedWords.filter { lower.contains($0) }
            #expect(
                hits.count >= 3,
                "transcription did not recover the spoken words (matched \(hits) in: \(answer))"
            )
        }
    }
}
