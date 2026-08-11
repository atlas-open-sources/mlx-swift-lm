// Copyright © 2026 Apple Inc.

// Port of https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/muse_glimmer

import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXNN

private enum MuseGlimmerError: LocalizedError {
    case featureTokenMismatch(expected: Int, actual: Int)

    var errorDescription: String? {
        switch self {
        case .featureTokenMismatch(let expected, let actual):
            "Muse Glimmer received \(expected) media tokens but produced \(actual) features."
        }
    }
}

// MARK: - Configuration

public struct MuseGlimmerConfiguration: Codable, Sendable {
    public struct RopeParameters: Codable, Sendable {
        private let _theta: Float?
        public var theta: Float { _theta ?? 10_000 }

        enum CodingKeys: String, CodingKey { case _theta = "rope_theta" }
    }

    public struct TextConfiguration: Codable, Sendable {
        public let modelType: String
        public let vocabularySize: Int
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let hiddenLayers: Int
        public let attentionHeads: Int
        public let kvHeads: Int
        public let headDim: Int
        public let maxPositionEmbeddings: Int
        public let rmsNormEps: Float
        public let postNormEps: Float
        public let attentionBias: Bool
        public let slidingWindow: Int
        public let ropeParameters: RopeParameters
        public let layerTypes: [String]
        public let layerRopeTheta: [Float]
        public let qkScaleFactor: Float
        public let outputMultiplier: Float
        public let finalLogitSoftcapping: Float
        public let tieWordEmbeddings: Bool

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case vocabularySize = "vocab_size"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case hiddenLayers = "num_hidden_layers"
            case attentionHeads = "num_attention_heads"
            case kvHeads = "num_key_value_heads"
            case headDim = "head_dim"
            case maxPositionEmbeddings = "max_position_embeddings"
            case rmsNormEps = "rms_norm_eps"
            case postNormEps = "post_norm_eps"
            case attentionBias = "attention_bias"
            case slidingWindow = "sliding_window"
            case ropeParameters = "rope_parameters"
            case layerTypes = "layer_types"
            case layerRopeTheta = "layer_rope_theta"
            case qkScaleFactor = "qk_scale_factor"
            case outputMultiplier = "output_multiplier"
            case finalLogitSoftcapping = "final_logit_softcapping"
            case tieWordEmbeddings = "tie_word_embeddings"
        }
    }

    public struct VisionConfiguration: Codable, Sendable {
        public let modelType: String
        public let patchSize: Int
        public let patchTemporal: Int
        public let mergeSize: Int
        public let positionEmbeddingHeight: Int
        public let positionEmbeddingWidth: Int
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let attentionHeads: Int
        public let hiddenLayers: Int
        public let layerNormEps: Float
        public let ropeParameters: RopeParameters
        public let layerTypes: [String]

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case patchSize = "patch_size"
            case patchTemporal = "patch_temporal"
            case mergeSize = "merge_size"
            case positionEmbeddingHeight = "pos_emb_height"
            case positionEmbeddingWidth = "pos_emb_width"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case attentionHeads = "num_attention_heads"
            case hiddenLayers = "num_hidden_layers"
            case layerNormEps = "layer_norm_eps"
            case ropeParameters = "rope_parameters"
            case layerTypes = "layer_types"
        }
    }

    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let modelType: String
    public let imageTokenID: Int
    public let videoTokenID: Int
    public let outputHiddenSize: Int
    public let projectorHiddenSize: Int
    public let quantization: BaseConfiguration.Quantization?

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case modelType = "model_type"
        case imageTokenID = "image_token_id"
        case videoTokenID = "video_token_id"
        case outputHiddenSize = "out_hidden_size"
        case projectorHiddenSize = "projector_hidden_size"
        case quantization
    }
}

public struct MuseGlimmerProcessorConfiguration: Codable, Sendable {
    private struct ImageProcessor: Codable, Sendable {
        let imageMean: [CGFloat]?
        let imageStd: [CGFloat]?
        let maximumImageTokens: Int?
        let mergeSize: Int?
        let patchSize: Int?
        let temporalPatchSize: Int?

        enum CodingKeys: String, CodingKey {
            case imageMean = "image_mean"
            case imageStd = "image_std"
            case maximumImageTokens = "max_image_tokens"
            case mergeSize = "merge_size"
            case patchSize = "patch_size"
            case temporalPatchSize = "temporal_patch_size"
        }
    }

    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let maximumImageTokens: Int
    public let mergeSize: Int
    public let patchSize: Int
    public let temporalPatchSize: Int

    private enum CodingKeys: String, CodingKey { case imageProcessor = "image_processor" }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let image = try container.decodeIfPresent(ImageProcessor.self, forKey: .imageProcessor)
        imageMean = image?.imageMean ?? [0.5, 0.5, 0.5]
        imageStd = image?.imageStd ?? [0.5, 0.5, 0.5]
        maximumImageTokens = image?.maximumImageTokens ?? 4096
        mergeSize = image?.mergeSize ?? 2
        patchSize = image?.patchSize ?? 14
        temporalPatchSize = image?.temporalPatchSize ?? 2
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(
            ImageProcessor(
                imageMean: imageMean, imageStd: imageStd,
                maximumImageTokens: maximumImageTokens, mergeSize: mergeSize,
                patchSize: patchSize, temporalPatchSize: temporalPatchSize),
            forKey: .imageProcessor)
    }
}

// MARK: - Processor

public struct MuseGlimmerProcessor: UserInputProcessor {
    private let config: MuseGlimmerProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: MuseGlimmerProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    static func smartResize(
        height: Int, width: Int, factor: Int, maximumTokens: Int
    ) -> (height: Int, width: Int) {
        var idealHeight = Double(height) / Double(factor)
        var idealWidth = Double(width) / Double(factor)
        let ratio = idealHeight > 0 ? idealWidth / idealHeight : 1
        if idealHeight * idealWidth > Double(maximumTokens) {
            idealHeight = sqrt(Double(maximumTokens) / ratio)
            idealWidth = idealHeight * ratio
        }

        let heightCandidates = [Int(floor(idealHeight)), Int(ceil(idealHeight))]
        let widthCandidates = [Int(floor(idealWidth)), Int(ceil(idealWidth))]
        let candidates = heightCandidates.flatMap { h in
            widthCandidates.compactMap { w -> (Int, Int)? in
                h >= 1 && w >= 1 && h * w <= maximumTokens ? (h, w) : nil
            }
        }
        let selected =
            candidates.min { lhs, rhs in
                abs(Double(lhs.0) / Double(lhs.1) - Double(height) / Double(width))
                    < abs(Double(rhs.0) / Double(rhs.1) - Double(height) / Double(width))
            } ?? (max(1, Int(idealHeight.rounded())), max(1, Int(idealWidth.rounded())))
        return (selected.0 * factor, selected.1 * factor)
    }

    private func preprocess(_ image: CIImage, processing: UserInput.Processing?) throws -> (
        MLXArray, THW
    ) {
        let source = MediaProcessing.apply(image, processing: processing)
        let extent = source.extent.size
        guard extent.width > 0, extent.height > 0 else {
            throw VLMError.imageProcessingFailure("Image has no pixels")
        }
        let factor = config.patchSize * config.mergeSize
        let size = Self.smartResize(
            height: Int(extent.height), width: Int(extent.width), factor: factor,
            maximumTokens: config.maximumImageTokens)
        let target = CGSize(width: size.width, height: size.height)
        let resized = MediaProcessing.resampleLanczos(
            MediaProcessing.inSRGBToneCurveSpace(source), to: target)
        let normalized = MediaProcessing.normalize(
            resized,
            mean: (config.imageMean[0], config.imageMean[1], config.imageMean[2]),
            std: (config.imageStd[0], config.imageStd[1], config.imageStd[2]))
        var pixels = MediaProcessing.asMLXArray(normalized)
        let channels = pixels.dim(1)
        let gridHeight = size.height / config.patchSize
        let gridWidth = size.width / config.patchSize
        pixels = pixels.reshaped(
            channels, gridHeight, config.patchSize, gridWidth, config.patchSize)
        pixels = pixels.transposed(1, 3, 0, 2, 4)
        pixels = expandedDimensions(pixels, axis: 2)
        pixels = tiled(pixels, repetitions: [1, 1, config.temporalPatchSize, 1, 1, 1])
        pixels = pixels.reshaped(
            gridHeight * gridWidth,
            config.temporalPatchSize * channels * config.patchSize * config.patchSize)
        return (pixels, THW(1, gridHeight, gridWidth))
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        guard input.videos.isEmpty else {
            throw VLMError.processing("Muse Glimmer video input is not supported yet")
        }

        let messages = Qwen3VLMessageGenerator().generate(from: input)
        var promptTokens = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools, additionalContext: input.additionalContext)

        guard !input.images.isEmpty else {
            let prompt = MLXArray(promptTokens).expandedDimensions(axis: 0)
            return LMInput(text: .init(tokens: prompt, mask: ones(like: prompt).asType(.int8)))
        }

        let processed = try input.images.map {
            try preprocess($0.asCIImage(), processing: input.processing)
        }
        let placeholder = tokenizer.encode(text: "<|patch|>")
        let placeholderRanges = promptTokens.ranges(of: placeholder)
        guard placeholderRanges.count == processed.count else {
            throw VLMError.processing(
                "Muse Glimmer chat template produced \(placeholderRanges.count) image markers for \(processed.count) images"
            )
        }

        var expandedTokens: [Int] = []
        var cursor = promptTokens.startIndex
        for (range, item) in zip(placeholderRanges, processed) {
            expandedTokens.append(contentsOf: promptTokens[cursor ..< range.lowerBound])
            let patchCount = item.1.product / (config.mergeSize * config.mergeSize)
            let replacement =
                "<|image_start|>"
                + Array(repeating: "<|patch|>", count: patchCount).joined()
                + "<|image_end|>"
            expandedTokens.append(contentsOf: tokenizer.encode(text: replacement))
            cursor = range.upperBound
        }
        expandedTokens.append(contentsOf: promptTokens[cursor...])
        promptTokens = expandedTokens

        let prompt = MLXArray(promptTokens).expandedDimensions(axis: 0)
        return LMInput(
            text: .init(tokens: prompt, mask: ones(like: prompt).asType(.int8)),
            image: .init(
                pixels: concatenated(processed.map(\.0)), frames: processed.map(\.1)))
    }
}

// MARK: - Text model

private final class MuseRMSNormNoScale: Module, UnaryLayer {
    let eps: Float
    init(eps: Float) { self.eps = eps }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let dtype = x.dtype
        let x32 = x.asType(.float32)
        return (x32 * rsqrt(MLX.mean(square(x32), axis: -1, keepDims: true) + eps))
            .asType(dtype)
    }
}

private final class MuseCenteredRMSNorm: Module, UnaryLayer {
    @ParameterInfo var weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float) {
        _weight.wrappedValue = MLXArray.zeros([dimensions])
        self.eps = eps
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let dtype = x.dtype
        let x32 = x.asType(.float32)
        let variance = MLX.mean(square(x32), axis: -1, keepDims: true)
        return (x32 * rsqrt(variance + eps) * (1 + weight.asType(.float32))).asType(dtype)
    }
}

private final class MuseMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(_ config: MuseGlimmerConfiguration.TextConfiguration) {
        _gate.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _up.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _down.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { down(silu(gate(x)) * up(x)) }
}

private final class MuseAttention: Module {
    @ModuleInfo(key: "q_proj") var query: Linear
    @ModuleInfo(key: "k_proj") var key: Linear
    @ModuleInfo(key: "v_proj") var value: Linear
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "o_proj") var output: Linear
    @ModuleInfo(key: "qk_norm") var qkNorm: MuseRMSNormNoScale

    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float
    let qkScaleFactor: Float
    let rope: RoPELayer?

    init(_ config: MuseGlimmerConfiguration.TextConfiguration, layer: Int) {
        heads = config.attentionHeads
        kvHeads = config.kvHeads
        headDim = config.headDim
        scale = pow(Float(headDim), -0.5)
        qkScaleFactor = config.qkScaleFactor
        _query.wrappedValue = Linear(config.hiddenSize, heads * headDim, bias: config.attentionBias)
        _key.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: config.attentionBias)
        _value.wrappedValue = Linear(
            config.hiddenSize, kvHeads * headDim, bias: config.attentionBias)
        _gate.wrappedValue = Linear(config.hiddenSize, heads * headDim, bias: false)
        _output.wrappedValue = Linear(
            heads * headDim, config.hiddenSize, bias: config.attentionBias)
        _qkNorm.wrappedValue = MuseRMSNormNoScale(eps: config.rmsNormEps)
        let theta = config.layerRopeTheta[layer]
        rope =
            theta == 0
            ? nil
            : initializeRope(
                dims: headDim, base: theta, traditional: false, scalingConfig: nil,
                maxPositionEmbeddings: config.maxPositionEmbeddings)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)
        var queries = query(x).reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)
        var keys = key(x).reshaped(batch, length, kvHeads, headDim).transposed(0, 2, 1, 3)
        let values = value(x).reshaped(batch, length, kvHeads, headDim).transposed(0, 2, 1, 3)
        queries = qkNorm(queries) * qkScaleFactor
        keys = qkNorm(keys)
        if let rope {
            queries = rope(queries, offset: cache?.offset ?? 0)
            keys = rope(keys, offset: cache?.offset ?? 0)
        }
        var attended = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values, cache: cache, scale: scale, mask: mask)
        attended = attended.transposed(0, 2, 1, 3).reshaped(batch, length, -1)
        return output(attended * sigmoid(gate(x)))
    }
}

private final class MuseDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: MuseAttention
    @ModuleInfo var mlp: MuseMLP
    @ModuleInfo(key: "input_layernorm") var inputNorm: MuseCenteredRMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: MuseCenteredRMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedForwardNorm: MuseCenteredRMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedForwardNorm: MuseCenteredRMSNorm
    let isSliding: Bool

    init(_ config: MuseGlimmerConfiguration.TextConfiguration, layer: Int) {
        _attention.wrappedValue = MuseAttention(config, layer: layer)
        _mlp.wrappedValue = MuseMLP(config)
        _inputNorm.wrappedValue = MuseCenteredRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionNorm.wrappedValue = MuseCenteredRMSNorm(
            dimensions: config.hiddenSize, eps: config.postNormEps)
        _preFeedForwardNorm.wrappedValue = MuseCenteredRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postFeedForwardNorm.wrappedValue = MuseCenteredRMSNorm(
            dimensions: config.hiddenSize, eps: config.postNormEps)
        isSliding = config.layerTypes[layer] == "sliding_attention"
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let afterAttention =
            x + postAttentionNorm(attention(inputNorm(x), mask: mask, cache: cache))
        return afterAttention + postFeedForwardNorm(mlp(preFeedForwardNorm(afterAttention)))
    }
}

private final class MuseTextModel: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "embed_norm") var embedNorm: MuseRMSNormNoScale
    @ModuleInfo var layers: [MuseDecoderLayer]
    @ModuleInfo var norm: RMSNorm
    let slidingWindow: Int

    init(_ config: MuseGlimmerConfiguration.TextConfiguration) {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        _embedNorm.wrappedValue = MuseRMSNormNoScale(eps: config.rmsNormEps)
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map {
            MuseDecoderLayer(config, layer: $0)
        }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        slidingWindow = config.slidingWindow
    }

    func callAsFunction(
        _ input: MLXArray?, inputEmbeddings: MLXArray?, cache: [KVCache]?
    ) -> MLXArray {
        var hidden = inputEmbeddings ?? embedNorm(embedTokens(input!))
        let optionalCache = cache?.map { Optional($0) }
        let fullIndex = layers.firstIndex { !$0.isSliding } ?? 0
        let slidingIndex = layers.firstIndex { $0.isSliding }
        let fullMask = createAttentionMask(h: hidden, cache: optionalCache?[fullIndex])
        let slidingMask =
            slidingIndex.map {
                createAttentionMask(h: hidden, cache: optionalCache?[$0], windowSize: slidingWindow)
            } ?? .none
        for (index, layer) in layers.enumerated() {
            hidden = layer(
                hidden, mask: layer.isSliding ? slidingMask : fullMask,
                cache: optionalCache?[index])
        }
        return norm(hidden)
    }
}

private final class MuseLanguageModel: Module {
    @ModuleInfo var model: MuseTextModel
    @ModuleInfo(key: "lm_head") var head: Linear
    let config: MuseGlimmerConfiguration.TextConfiguration

    init(_ config: MuseGlimmerConfiguration.TextConfiguration) {
        self.config = config
        _model.wrappedValue = MuseTextModel(config)
        _head.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
    }

    func callAsFunction(
        _ input: MLXArray?, inputEmbeddings: MLXArray? = nil, cache: [KVCache]? = nil
    ) -> LMOutput {
        let hidden = model(input, inputEmbeddings: inputEmbeddings, cache: cache)
        var logits = head(hidden) * config.outputMultiplier
        logits = tanh(logits / config.finalLogitSoftcapping) * config.finalLogitSoftcapping
        return LMOutput(logits: logits)
    }
}

// MARK: - Vision model

private final class MuseVisionPatchEmbedder: Module {
    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Linear
    @ModuleInfo(key: "position_embedding_table") var positionEmbedding: Embedding
    let side: Int

    init(_ config: MuseGlimmerConfiguration.VisionConfiguration) {
        let patchDimensions = config.patchTemporal * 3 * config.patchSize * config.patchSize
        _patchEmbedding.wrappedValue = Linear(patchDimensions, config.hiddenSize, bias: false)
        _positionEmbedding.wrappedValue = Embedding(
            embeddingCount: config.positionEmbeddingHeight * config.positionEmbeddingWidth,
            dimensions: config.hiddenSize)
        side = config.positionEmbeddingHeight
    }

    private func positions(_ grids: [THW]) -> MLXArray {
        var allIndices = Array(repeating: [Int32](), count: 4)
        var allWeights = Array(repeating: [Float](), count: 4)
        for grid in grids {
            for _ in 0 ..< grid.t {
                for row in 0 ..< grid.h {
                    let h = (Float(row) + 0.5) * Float(side) / Float(grid.h) - 0.5
                    let h0Raw = Int(floor(h))
                    let h1Raw = h0Raw + 1
                    let dh = h - Float(h0Raw)
                    for column in 0 ..< grid.w {
                        let w = (Float(column) + 0.5) * Float(side) / Float(grid.w) - 0.5
                        let w0Raw = Int(floor(w))
                        let w1Raw = w0Raw + 1
                        let dw = w - Float(w0Raw)
                        let h0 = min(max(h0Raw, 0), side - 1)
                        let h1 = min(max(h1Raw, 0), side - 1)
                        let w0 = min(max(w0Raw, 0), side - 1)
                        let w1 = min(max(w1Raw, 0), side - 1)
                        let validH0 = h0Raw >= 0 && h0Raw < side
                        let validH1 = h1Raw >= 0 && h1Raw < side
                        let validW0 = w0Raw >= 0 && w0Raw < side
                        let validW1 = w1Raw >= 0 && w1Raw < side
                        allIndices[0].append(Int32(h0 * side + w0))
                        allIndices[1].append(Int32(h0 * side + w1))
                        allIndices[2].append(Int32(h1 * side + w0))
                        allIndices[3].append(Int32(h1 * side + w1))
                        allWeights[0].append(validH0 && validW0 ? (1 - dh) * (1 - dw) : 0)
                        allWeights[1].append(validH0 && validW1 ? (1 - dh) * dw : 0)
                        allWeights[2].append(validH1 && validW0 ? dh * (1 - dw) : 0)
                        allWeights[3].append(validH1 && validW1 ? dh * dw : 0)
                    }
                }
            }
        }
        var result = MLXArray.zeros(
            [allIndices[0].count, positionEmbedding.weight.dim(1)],
            dtype: positionEmbedding.weight.dtype)
        for corner in 0 ..< 4 {
            let embeddings = positionEmbedding(MLXArray(allIndices[corner]))
            result = result + embeddings * MLXArray(allWeights[corner]).expandedDimensions(axis: -1)
        }
        return result
    }

    func callAsFunction(_ pixels: MLXArray, grids: [THW]) -> MLXArray {
        let embeddings = patchEmbedding(pixels)
        return embeddings + positions(grids).asType(embeddings.dtype)
    }
}

private final class MuseVisionAttention: Module {
    @ModuleInfo(key: "q_proj") var query: Linear
    @ModuleInfo(key: "k_proj") var key: Linear
    @ModuleInfo(key: "v_proj") var value: Linear
    @ModuleInfo var proj: Linear
    let heads: Int
    let headDim: Int
    let scale: Float

    init(_ config: MuseGlimmerConfiguration.VisionConfiguration) {
        heads = config.attentionHeads
        headDim = config.hiddenSize / config.attentionHeads
        scale = pow(Float(headDim), -0.5)
        _query.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
        _key.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
        _value.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
        _proj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
    }

    private func rotateHalf(_ x: MLXArray) -> MLXArray {
        let half = x.dim(-1) / 2
        return concatenated([-x[.ellipsis, half...], x[.ellipsis, ..<half]], axis: -1)
    }

    private func applyRotary(
        queries: MLXArray, keys: MLXArray, cosine: MLXArray, sine: MLXArray
    ) -> (MLXArray, MLXArray) {
        let cos32 = cosine.expandedDimensions(axis: 0).expandedDimensions(axis: 2).asType(.float32)
        let sin32 = sine.expandedDimensions(axis: 0).expandedDimensions(axis: 2).asType(.float32)
        let queryType = queries.dtype
        let keyType = keys.dtype
        let queries32 = queries.asType(.float32)
        let keys32 = keys.asType(.float32)
        return (
            (queries32 * cos32 + rotateHalf(queries32) * sin32).asType(queryType),
            (keys32 * cos32 + rotateHalf(keys32) * sin32).asType(keyType)
        )
    }

    private func fusedAttention(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray) -> MLXArray {
        let target = [64, 80, 128].first(where: { headDim <= $0 }) ?? headDim
        guard target != headDim else {
            return MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: scale, mask: .none)
        }
        let widths: [IntOrPair] = [0, 0, 0, .init((0, target - headDim))]
        return MLXFast.scaledDotProductAttention(
            queries: padded(q, widths: widths), keys: padded(k, widths: widths),
            values: padded(v, widths: widths), scale: scale, mask: .none
        )[.ellipsis, ..<headDim]
    }

    func callAsFunction(
        _ hidden: MLXArray, splitPoints: [Int], cosine: MLXArray, sine: MLXArray
    ) -> MLXArray {
        let length = hidden.dim(0)
        var q = query(hidden).reshaped(1, length, heads, headDim)
        var k = key(hidden).reshaped(1, length, heads, headDim)
        let v = value(hidden).reshaped(1, length, heads, headDim)
        (q, k) = applyRotary(queries: q, keys: k, cosine: cosine, sine: sine)
        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        let values = v.transposed(0, 2, 1, 3)
        let output: MLXArray
        if splitPoints.isEmpty {
            output = fusedAttention(q, k, values)
        } else {
            let queryParts = q.split(indices: splitPoints, axis: 2)
            let keyParts = k.split(indices: splitPoints, axis: 2)
            let valueParts = values.split(indices: splitPoints, axis: 2)
            output = concatenated(
                zip(zip(queryParts, keyParts), valueParts).map {
                    fusedAttention($0.0.0, $0.0.1, $0.1)
                }, axis: 2)
        }
        return proj(output.transposed(0, 2, 1, 3).reshaped(length, -1))
    }
}

private final class MuseVisionMLP: Module, UnaryLayer {
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear
    @ModuleInfo var activation: GELU
    init(_ config: MuseGlimmerConfiguration.VisionConfiguration) {
        _fc1.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: true)
        _fc2.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: true)
        _activation.wrappedValue = GELU(approximation: .precise)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(activation(fc1(x))) }
}

private final class MuseVisionBlock: Module {
    @ModuleInfo var norm1: LayerNorm
    @ModuleInfo var norm2: LayerNorm
    @ModuleInfo var attn: MuseVisionAttention
    @ModuleInfo var mlp: MuseVisionMLP
    init(_ config: MuseGlimmerConfiguration.VisionConfiguration) {
        _norm1.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        _norm2.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        _attn.wrappedValue = MuseVisionAttention(config)
        _mlp.wrappedValue = MuseVisionMLP(config)
    }
    func callAsFunction(
        _ x: MLXArray, splitPoints: [Int], cosine: MLXArray, sine: MLXArray
    ) -> MLXArray {
        let attended = x + attn(norm1(x), splitPoints: splitPoints, cosine: cosine, sine: sine)
        return attended + mlp(norm2(attended))
    }
}

private final class MuseVisionModel: Module {
    @ModuleInfo(key: "patch_embedder") var patchEmbedder: MuseVisionPatchEmbedder
    @ModuleInfo(key: "ln_pre") var preNorm: LayerNorm
    @ModuleInfo var layers: [MuseVisionBlock]
    @ModuleInfo(key: "ln_post") var postNorm: LayerNorm
    let config: MuseGlimmerConfiguration.VisionConfiguration

    init(_ config: MuseGlimmerConfiguration.VisionConfiguration) {
        self.config = config
        _patchEmbedder.wrappedValue = MuseVisionPatchEmbedder(config)
        _preNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in MuseVisionBlock(config) }
        _postNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
    }

    private func fullSplitPoints(_ grids: [THW]) -> [Int] {
        var offset = 0
        var result: [Int] = []
        for grid in grids {
            for _ in 0 ..< grid.t {
                offset += grid.h * grid.w
                result.append(offset)
            }
        }
        return Array(result.dropLast())
    }

    private func windowLayout(_ grids: [THW]) -> (indices: [Int32]?, splitPoints: [Int]) {
        let side = config.positionEmbeddingHeight
        var indices: [Int32] = []
        var cumulative = [0]
        var offset = 0
        for grid in grids {
            for frame in 0 ..< grid.t {
                var row = 0
                while row < grid.h {
                    var column = 0
                    while column < grid.w {
                        var count = 0
                        for localRow in 0 ..< side where row + localRow < grid.h {
                            for localColumn in 0 ..< side where column + localColumn < grid.w {
                                indices.append(
                                    Int32(
                                        offset + frame * grid.h * grid.w + (row + localRow) * grid.w
                                            + column + localColumn))
                                count += 1
                            }
                        }
                        cumulative.append(cumulative.last! + count)
                        column += side
                    }
                    row += side
                }
            }
            offset += grid.product
        }
        let identity = indices.enumerated().allSatisfy { Int($0.element) == $0.offset }
        return (identity ? nil : indices, Array(cumulative.dropFirst().dropLast()))
    }

    private func positionIDs(_ grids: [THW]) -> MLXArray {
        var result: [Int32] = []
        for grid in grids {
            for _ in 0 ..< grid.t {
                for row in 0 ..< grid.h {
                    for column in 0 ..< grid.w {
                        result.append(Int32(column + 1))
                        result.append(Int32(row + 1))
                    }
                }
            }
        }
        return MLXArray(result).reshaped(-1, 2)
    }

    private func rotary(_ positionIDs: MLXArray) -> (MLXArray, MLXArray) {
        let spatialDimensions = config.hiddenSize / config.attentionHeads / 2
        let indices = MLXArray(stride(from: 0, to: spatialDimensions, by: 2)).asType(.float32)
        let inverse = 1 / pow(config.ropeParameters.theta, indices / Float(spatialDimensions))
        let width = positionIDs[0..., 0].asType(.float32).expandedDimensions(axis: 1) * inverse
        let height = positionIDs[0..., 1].asType(.float32).expandedDimensions(axis: 1) * inverse
        let frequencies = concatenated([width, height, width, height], axis: -1)
        return (cos(frequencies), sin(frequencies))
    }

    private func pixelShuffle(_ hidden: MLXArray, grids: [THW]) -> MLXArray {
        var outputs: [MLXArray] = []
        var offset = 0
        let merge = config.mergeSize
        let dimensions = hidden.dim(-1)
        for grid in grids {
            let count = grid.product
            var chunk = hidden[offset ..< offset + count, 0...]
            chunk = chunk.reshaped(grid.t, grid.h, grid.w, dimensions)
            chunk = chunk.reshaped(
                grid.t, grid.h / merge, merge, grid.w / merge, merge, dimensions)
            chunk = chunk.transposed(0, 1, 3, 5, 2, 4)
            outputs.append(chunk.reshaped(-1, dimensions * merge * merge))
            offset += count
        }
        return concatenated(outputs)
    }

    func callAsFunction(_ pixels: MLXArray, grids: [THW]) -> MLXArray {
        let fullSplits = fullSplitPoints(grids)
        let window = windowLayout(grids)
        var hidden = preNorm(patchEmbedder(pixels, grids: grids))
        var positions = positionIDs(grids)
        if let indices = window.indices {
            let array = MLXArray(indices)
            hidden = hidden[array]
            positions = positions[array]
        }
        let (cosine, sine) = rotary(positions)
        for (index, layer) in layers.enumerated() {
            hidden = layer(
                hidden,
                splitPoints: config.layerTypes[index] == "full_attention"
                    ? fullSplits : window.splitPoints,
                cosine: cosine, sine: sine)
        }
        if let indices = window.indices {
            hidden = hidden[argSort(MLXArray(indices))]
        }
        return pixelShuffle(postNorm(hidden), grids: grids)
    }
}

private final class MuseVisionAdapter: Module, UnaryLayer {
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear
    @ModuleInfo var activation: GELU
    init(_ config: MuseGlimmerConfiguration) {
        _fc1.wrappedValue = Linear(config.outputHiddenSize, config.projectorHiddenSize, bias: false)
        _fc2.wrappedValue = Linear(
            config.projectorHiddenSize, config.projectorHiddenSize, bias: false)
        _activation.wrappedValue = GELU(approximation: .precise)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { activation(fc2(activation(fc1(x)))) }
}

// MARK: - Muse Glimmer

public final class MuseGlimmer: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "language_model") private var languageModel: MuseLanguageModel
    @ModuleInfo(key: "vision_tower") private var visionTower: MuseVisionModel
    @ModuleInfo(key: "vision_adapter") private var visionAdapter: MuseVisionAdapter
    @ModuleInfo(key: "vision_projection") private var visionProjection: Linear
    @ModuleInfo(key: "perception_emb_norm") private var perceptionNorm: MuseRMSNormNoScale
    public let config: MuseGlimmerConfiguration

    public var kvHeads: [Int] {
        Array(
            repeating: config.textConfiguration.kvHeads,
            count: config.textConfiguration.hiddenLayers)
    }
    public var loraLayers: [Module] { languageModel.model.layers }
    public var toolCallFormat: ToolCallFormat? { .atem }
    public var reasoningConfig: ReasoningConfig? {
        ReasoningConfig(
            startDelimiter: "to=self<|message|>", endDelimiter: "<|eom|>",
            promptStrategy: .none, isSpecialToken: true)
    }

    public init(_ config: MuseGlimmerConfiguration) {
        self.config = config
        _languageModel.wrappedValue = MuseLanguageModel(config.textConfiguration)
        _visionTower.wrappedValue = MuseVisionModel(config.visionConfiguration)
        _visionAdapter.wrappedValue = MuseVisionAdapter(config)
        _visionProjection.wrappedValue = Linear(
            config.projectorHiddenSize, config.textConfiguration.hiddenSize, bias: false)
        _perceptionNorm.wrappedValue = MuseRMSNormNoScale(eps: config.textConfiguration.rmsNormEps)
    }

    public func newCache(parameters _: GenerateParameters?) -> [KVCache] {
        languageModel.model.layers.map {
            $0.isSliding
                ? RotatingKVCache(maxSize: config.textConfiguration.slidingWindow, keep: 0)
                : KVCacheSimple()
        }
    }

    private func inputEmbeddings(_ input: LMInput) throws -> MLXArray {
        let inputIDs = input.text.tokens
        var embeddings = languageModel.model.embedNorm(languageModel.model.embedTokens(inputIDs))
        guard let image = input.image, let frames = image.frames else { return embeddings }
        let dtype = visionTower.patchEmbedder.patchEmbedding.weight.dtype
        var features = visionTower(image.pixels.asType(dtype), grids: frames)
        features = perceptionNorm(visionProjection(visionAdapter(features))).asType(
            embeddings.dtype)
        let tokenIDs = inputIDs.asArray(Int.self)
        let mediaIndices = tokenIDs.enumerated().compactMap { index, token in
            token == config.imageTokenID || token == config.videoTokenID ? index : nil
        }
        guard mediaIndices.count == features.dim(0) else {
            throw MuseGlimmerError.featureTokenMismatch(
                expected: mediaIndices.count, actual: features.dim(0))
        }
        let hidden = embeddings.dim(-1)
        embeddings = embeddings.reshaped(-1, hidden)
        embeddings[MLXArray(mediaIndices.map(Int32.init)), 0...] = features
        return embeddings.reshaped(inputIDs.dim(0), inputIDs.dim(1), hidden)
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state _: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        let embeddings = try inputEmbeddings(input)
        let total = embeddings.dim(1)
        let processed = try prefill.forEachChunk(total: total) { range in
            _ = languageModel(nil, inputEmbeddings: embeddings[0..., range, 0...], cache: cache)
            asyncEval(cache)
        }
        if processed > 0 { eval(cache) }
        let output = languageModel(
            nil, inputEmbeddings: embeddings[0..., processed..., 0...], cache: cache)
        prefill.progress?(total, total)
        return .logits(output)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache).logits
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        sanitized.reserveCapacity(weights.count)
        for (originalKey, value) in weights {
            guard !originalKey.contains("rotary_emb.inv_freq") else { continue }
            let key: String
            if originalKey.hasPrefix("model.language_model.") {
                key = originalKey.replacingOccurrences(
                    of: "model.language_model.", with: "language_model.model.",
                    options: [.anchored])
            } else if originalKey.hasPrefix("model.") {
                key = String(originalKey.dropFirst("model.".count))
            } else if originalKey.hasPrefix("lm_head.") {
                key = "language_model." + originalKey
            } else {
                key = originalKey
            }
            sanitized[key] = value
        }
        return sanitized
    }
}
