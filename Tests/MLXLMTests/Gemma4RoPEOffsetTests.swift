import Foundation
import MLX
import Testing

@testable import MLXLMCommon
@testable import MLXVLM

private final class Gemma4BatchOffsetProbeCache: BaseKVCache {
    override var ropeOffset: RoPEOffset { .batch(MLXArray([5, 2])) }

    override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        (keys, values)
    }
}

@Test("Gemma4 VLM preserves per-row RoPE offsets")
func gemma4VLMPreservesPerRowRoPEOffsets() throws {
    let attention = Gemma4TextAttention(config: try gemma4ProbeConfiguration(), layerIdx: 0)
    let cache: any KVCache = Gemma4BatchOffsetProbeCache()
    let input = MLXArray(Array(repeating: Float(1), count: 8)).reshaped([2, 1, 4])

    let (_, _, propagatedOffset) = attention(input, cache: cache)

    let offsets = try #require(batchOffsets(propagatedOffset))
    #expect(offsets.asArray(Int32.self) == [5, 2])
}

private func batchOffsets(_ offset: RoPEOffset?) -> MLXArray? {
    guard case .batch(let offsets)? = offset else { return nil }
    return offsets
}

private func batchOffsets(_: Int) -> MLXArray? {
    nil
}

private func gemma4ProbeConfiguration() throws -> Gemma4TextConfiguration {
    let json = """
        {
          "model_type": "gemma4_text",
          "hidden_size": 4,
          "num_hidden_layers": 1,
          "num_attention_heads": 2,
          "num_key_value_heads": 1,
          "head_dim": 2,
          "global_head_dim": 2,
          "vocab_size": 10,
          "num_kv_shared_layers": 0,
          "hidden_size_per_layer_input": 0,
          "sliding_window": 4,
          "sliding_window_pattern": 1,
          "max_position_embeddings": 16,
          "rms_norm_eps": 1e-6,
          "rope_traditional": false,
          "use_double_wide_mlp": false,
          "enable_moe_block": false,
          "attention_k_eq_v": true,
          "intermediate_size": 8,
          "layer_types": ["full_attention"],
          "rope_parameters": {},
          "tie_word_embeddings": true
        }
        """
    return try JSONDecoder().decode(
        Gemma4TextConfiguration.self, from: Data(json.utf8))
}
