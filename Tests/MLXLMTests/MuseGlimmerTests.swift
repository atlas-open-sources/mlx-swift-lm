// Copyright © 2026 Apple Inc.

import CoreImage
import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXVLM

struct MuseGlimmerTests {
    private func configuration() throws -> MuseGlimmerConfiguration {
        try JSONDecoder().decode(
            MuseGlimmerConfiguration.self,
            from: Data(
                """
                {
                  "model_type": "muse_glimmer",
                  "image_token_id": 7,
                  "video_token_id": 6,
                  "out_hidden_size": 32,
                  "projector_hidden_size": 16,
                  "text_config": {
                    "model_type": "muse_glimmer_text",
                    "vocab_size": 64,
                    "hidden_size": 16,
                    "intermediate_size": 32,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "head_dim": 4,
                    "max_position_embeddings": 128,
                    "rms_norm_eps": 0.00001,
                    "post_norm_eps": 0.00000001,
                    "attention_bias": false,
                    "sliding_window": 8,
                    "rope_parameters": {"rope_theta": 10000},
                    "layer_types": ["sliding_attention", "full_attention"],
                    "layer_rope_theta": [10000, 0],
                    "qk_scale_factor": 3.87,
                    "output_multiplier": 0.19611613513818404,
                    "final_logit_softcapping": 20,
                    "tie_word_embeddings": false
                  },
                  "vision_config": {
                    "model_type": "muse_glimmer_vision",
                    "patch_size": 2,
                    "patch_temporal": 2,
                    "merge_size": 2,
                    "pos_emb_height": 4,
                    "pos_emb_width": 4,
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "num_attention_heads": 2,
                    "num_hidden_layers": 2,
                    "layer_norm_eps": 0.00001,
                    "rope_parameters": {"rope_theta": 10000},
                    "layer_types": ["window_attention", "full_attention"]
                  }
                }
                """.utf8))
    }

    @Test("Muse Glimmer tiny text and image forward passes")
    func tinyForwardPasses() throws {
        let model = MuseGlimmer(try configuration())

        let textLogits = model(MLXArray([1, 2, 3]).reshaped(1, 3), cache: nil)
        eval(textLogits)
        #expect(textLogits.shape == [1, 3, 64])
        #expect(all(isFinite(textLogits)).item(Bool.self))

        let imageInput = LMInput(
            text: .init(tokens: MLXArray([1, 7, 2]).reshaped(1, 3)),
            image: .init(
                pixels: MLXArray.zeros([4, 24]),
                frames: [THW(1, 2, 2)]))
        let result = try model.prepare(
            imageInput, cache: model.newCache(parameters: nil), state: nil,
            windowSize: nil)
        guard case .logits(let imageOutput) = result else {
            Issue.record("Muse Glimmer image prepare should return logits")
            return
        }
        eval(imageOutput.logits)
        #expect(imageOutput.logits.shape == [1, 3, 64])
        #expect(all(isFinite(imageOutput.logits)).item(Bool.self))
    }

    @Test("Muse Glimmer uses mixed caches and native conventions")
    func cacheAndConventions() throws {
        let model = MuseGlimmer(try configuration())
        let caches = model.newCache(parameters: nil)

        #expect(caches[0] is RotatingKVCache)
        #expect(caches[1] is KVCacheSimple)
        #expect(model.toolCallFormat == .atem)
        #expect(model.reasoningConfig?.startDelimiter == "to=self<|message|>")
        #expect(model.reasoningConfig?.endDelimiter == "<|eom|>")
        #expect(ToolCallFormat.infer(from: "muse_glimmer") == .atem)
        #expect(
            ReasoningConfig.infer(from: "muse_glimmer")
                == ReasoningConfig(
                    startDelimiter: "to=self<|message|>", endDelimiter: "<|eom|>",
                    promptStrategy: .none, isSpecialToken: true))
    }

    @Test("Muse Glimmer processor decodes nested metadata and resizes on merged patches")
    func processorConfiguration() throws {
        let data = Data(
            """
            {"processor_class":"MuseGlimmerProcessor","image_processor":{
              "image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],
              "max_image_tokens":4096,"merge_size":2,"patch_size":14,
              "temporal_patch_size":2}}
            """.utf8)
        let config = try JSONDecoder().decode(MuseGlimmerProcessorConfiguration.self, from: data)
        #expect(config.patchSize == 14)
        #expect(config.temporalPatchSize == 2)
        #expect(config.mergeSize == 2)
        #expect(
            MuseGlimmerProcessor.smartResize(
                height: 28, width: 56, factor: 28, maximumTokens: 4096)
                == (28, 56))
    }

    @Test("Muse Glimmer image markers do not inherit tokenizer BOS tokens")
    func processorImageMarkersExcludeSpecialTokens() async throws {
        let data = Data(
            """
            {"processor_class":"MuseGlimmerProcessor","image_processor":{
              "image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],
              "max_image_tokens":4,"merge_size":2,"patch_size":14,
              "temporal_patch_size":2}}
            """.utf8)
        let config = try JSONDecoder().decode(MuseGlimmerProcessorConfiguration.self, from: data)
        let processor = MuseGlimmerProcessor(config, tokenizer: MuseGlimmerTestTokenizer())
        let image = CIImage(color: .white).cropped(
            to: CGRect(x: 0, y: 0, width: 28, height: 28))

        let input = try await processor.prepare(
            input: UserInput(prompt: "describe", images: [.ciImage(image)]))

        #expect(input.text.tokens.asArray(Int32.self) == [1, 8, 7, 9, 2])
        let frame = try #require(input.image?.frames?.first)
        #expect(frame.t == 1)
        #expect(frame.h == 2)
        #expect(frame.w == 2)
    }

    @Test("Muse Glimmer checkpoint prefixes map to Swift modules")
    func sanitization() throws {
        let model = MuseGlimmer(try configuration())
        let weights = model.sanitize(weights: [
            "model.language_model.layers.0.self_attn.q_proj.weight": .zeros([16, 16]),
            "model.vision_tower.ln_pre.weight": .ones([8]),
            "lm_head.weight": .zeros([64, 16]),
            "model.language_model.layers.0.self_attn.rotary_emb.inv_freq": .zeros([2]),
        ])

        #expect(weights["language_model.model.layers.0.self_attn.q_proj.weight"] != nil)
        #expect(weights["vision_tower.ln_pre.weight"] != nil)
        #expect(weights["language_model.lm_head.weight"] != nil)
        #expect(!weights.keys.contains { $0.contains("rotary_emb.inv_freq") })
    }
}

private struct MuseGlimmerTestTokenizer: Tokenizer {
    let bosToken: String? = "<bos>"
    let eosToken: String? = nil
    let unknownToken: String? = nil

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        let tokens: [Int]
        switch text {
        case "<|patch|>":
            tokens = [7]
        case "<|image_start|><|patch|><|image_end|>":
            tokens = [8, 7, 9]
        default:
            tokens = [42]
        }
        return addSpecialTokens ? [99] + tokens : tokens
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map(String.init).joined(separator: " ")
    }

    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { String(id) }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        [1, 7, 2]
    }
}
