// Copyright © 2026 Apple Inc.

import Foundation
@testable import MLXLLM
import MLXLMCommon
import Testing
@testable import MLXVLM

@Suite(.serialized)
struct Gemma4QATLoadTests {

    @Test(
        "Gemma4 QAT checkpoints load cached weights",
        .enabled(if: ProcessInfo.processInfo.environment["MLX_GEMMA4_QAT_TESTS"] == "1")
    )
    func qatCheckpointsLoadCachedWeights() throws {
        for checkpoint in ["gemma-4-E2B-it-qat-4bit", "gemma-4-E4B-it-qat-4bit"] {
            let url = FileManager.default.homeDirectoryForCurrentUser
                .appending(path: "Library/Caches/models/mlx-community")
                .appending(path: checkpoint)
            #expect(FileManager.default.fileExists(atPath: url.appending(path: "config.json").path))
                let configData = try Data(contentsOf: url.appending(path: "config.json"))
                let vlmConfig = try JSONDecoder.json5().decode(
                    MLXVLM.Gemma4Configuration.self, from: configData)
                let llmConfig = try JSONDecoder.json5().decode(
                    MLXLLM.Gemma4Configuration.self, from: configData)
                let baseConfig = try JSONDecoder.json5().decode(
                    BaseConfiguration.self, from: configData)
                let vlmModel = MLXVLM.Gemma4(vlmConfig)
                try loadWeights(
                    modelDirectory: url, model: vlmModel,
                    perLayerQuantization: baseConfig.perLayerQuantization)

                let llmModel = MLXLLM.Gemma4Model(llmConfig)
                try loadWeights(
                    modelDirectory: url, model: llmModel,
                    perLayerQuantization: baseConfig.perLayerQuantization)
            }
        }
    }
