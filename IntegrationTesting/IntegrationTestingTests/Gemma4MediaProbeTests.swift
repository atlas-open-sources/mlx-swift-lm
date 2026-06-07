// Copyright © 2026 Apple Inc.
//
// Ad-hoc probe: run ANY model on ANY image/video/audio with ANY prompt, driven
// by environment variables — no recompile per run. Opt-in via GEMMA_PROBE=1.
//
// Examples:
//   # Describe a user-supplied video on the default model
//   GEMMA_PROBE=1 GEMMA_VIDEO=/tmp/myclip.mov \
//   xcodebuild test -project IntegrationTesting.xcodeproj -scheme IntegrationTesting \
//     -destination 'platform=macOS' \
//     -only-testing:IntegrationTestingTests/Gemma4MediaProbeTests
//
//   # Non-quant model on an audio clip
//   GEMMA_PROBE=1 GEMMA_MODEL=mlx-community/gemma-4-e4b-it-bf16 \
//     GEMMA_AUDIO=.../gemma_speech_test.wav GEMMA_PROMPT="Transcribe this." \
//     xcodebuild test ...

import Foundation
import HuggingFace
import IntegrationTestHelpers
import MLXHuggingFace
import MLXLMCommon
import Testing
import Tokenizers

private let probeModels = IntegrationTestModels(
    downloader: #hubDownloader(),
    tokenizerLoader: #huggingFaceTokenizerLoader()
)

@Suite(.serialized)
struct Gemma4MediaProbeTests {

    @Test func probe() async throws {
        let env = ProcessInfo.processInfo.environment
        try #require(env["GEMMA_PROBE"] == "1", "Set GEMMA_PROBE=1 to run the media probe.")

        let model = env["GEMMA_MODEL"] ?? "mlx-community/gemma-4-e4b-it-4bit"
        let prompt = env["GEMMA_PROMPT"] ?? "Describe what you perceive in this media in detail."
        let maxTokens = Int(env["GEMMA_MAX_TOKENS"] ?? "200") ?? 200

        let images = (env["GEMMA_IMAGE"].map { [UserInput.Image.url(URL(fileURLWithPath: $0))] }) ?? []
        let videos = (env["GEMMA_VIDEO"].map { [UserInput.Video.url(URL(fileURLWithPath: $0))] }) ?? []
        let audios = (env["GEMMA_AUDIO"].map { [UserInput.Audio.url(URL(fileURLWithPath: $0))] }) ?? []

        print("🔎 PROBE model=\(model)")
        print("🔎 prompt=\(prompt)")
        print("🔎 image=\(env["GEMMA_IMAGE"] ?? "-") video=\(env["GEMMA_VIDEO"] ?? "-") audio=\(env["GEMMA_AUDIO"] ?? "-")")

        let container = try await probeModels.vlmContainer(for: ModelConfiguration(id: model))
        let session = ChatSession(
            container, generateParameters: GenerateParameters(maxTokens: maxTokens, temperature: 0))

        let answer = try await session.respond(
            to: prompt, images: images, videos: videos, audios: audios)

        print("🔎 ====== GEMMA OUTPUT ======")
        print(answer)
        print("🔎 ===========================")
        #expect(!answer.isEmpty)
    }
}
