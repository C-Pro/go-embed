package spagoref

import (
	"fmt"
	"math"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"go-embed/pkg/safetensors"
	"go-embed/pkg/tokenizer"
)

const (
	HiddenSize       = 384
	IntermediateSize = 1536
	NumLayers        = 12
	NumHeads         = 12
	HeadDim          = 32
	MaxPositions     = 512
	VocabSize        = 250037
	LayerNormEps     = 1e-12
)

// LayerWeights holds the weights for a single transformer encoder layer.
type LayerWeights struct {
	QueryWeight mat.Matrix
	QueryBias   mat.Matrix
	KeyWeight   mat.Matrix
	KeyBias     mat.Matrix
	ValueWeight mat.Matrix
	ValueBias   mat.Matrix
	OutWeight   mat.Matrix
	OutBias     mat.Matrix
	AttnNormW   mat.Matrix
	AttnNormB   mat.Matrix

	FFN1Weight mat.Matrix
	FFN1Bias   mat.Matrix
	FFN2Weight mat.Matrix
	FFN2Bias   mat.Matrix
	FFNNormW   mat.Matrix
	FFNNormB   mat.Matrix
}

// SpagoModel holds all model weights loaded as Spago matrices.
type SpagoModel struct {
	WordEmbeddings      mat.Matrix
	PositionEmbeddings  mat.Matrix
	TokenTypeEmbeddings mat.Matrix
	EmbeddingsNormW     mat.Matrix
	EmbeddingsNormB     mat.Matrix

	Layers [NumLayers]LayerWeights
	Tok    *tokenizer.Tokenizer
}

// LoadModel loads the model from safetensors and tokenizer.json files.
func LoadModel(modelPath, tokenizerPath string) (*SpagoModel, error) {
	tok, err := tokenizer.LoadFromFile(tokenizerPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load tokenizer: %w", err)
	}

	st, err := safetensors.Open(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open safetensors: %w", err)
	}
	defer st.Close()

	m := &SpagoModel{Tok: tok}

	// 1. Embeddings
	wEmb, err := st.ReadTensorF32("embeddings.word_embeddings.weight")
	if err != nil {
		return nil, err
	}
	m.WordEmbeddings = mat.NewDense[float32](mat.WithShape(VocabSize, HiddenSize), mat.WithBacking(wEmb))

	pEmb, err := st.ReadTensorF32("embeddings.position_embeddings.weight")
	if err != nil {
		return nil, err
	}
	m.PositionEmbeddings = mat.NewDense[float32](mat.WithShape(MaxPositions, HiddenSize), mat.WithBacking(pEmb))

	tEmb, err := st.ReadTensorF32("embeddings.token_type_embeddings.weight")
	if err != nil {
		return nil, err
	}
	m.TokenTypeEmbeddings = mat.NewDense[float32](mat.WithShape(2, HiddenSize), mat.WithBacking(tEmb))

	lnW, err := st.ReadTensorF32("embeddings.LayerNorm.weight")
	if err != nil {
		return nil, err
	}
	lnB, err := st.ReadTensorF32("embeddings.LayerNorm.bias")
	if err != nil {
		return nil, err
	}
	m.EmbeddingsNormW = mat.NewDense[float32](mat.WithShape(HiddenSize, 1), mat.WithBacking(lnW))
	m.EmbeddingsNormB = mat.NewDense[float32](mat.WithShape(HiddenSize, 1), mat.WithBacking(lnB))

	// 2. Encoder Layers
	for i := 0; i < NumLayers; i++ {
		prefix := fmt.Sprintf("encoder.layer.%d.", i)

		// Self Attention Query/Key/Value/Output
		qw, err := st.ReadTensorF32(prefix + "attention.self.query.weight")
		if err != nil {
			return nil, err
		}
		qb, err := st.ReadTensorF32(prefix + "attention.self.query.bias")
		if err != nil {
			return nil, err
		}
		kw, err := st.ReadTensorF32(prefix + "attention.self.key.weight")
		if err != nil {
			return nil, err
		}
		kb, err := st.ReadTensorF32(prefix + "attention.self.key.bias")
		if err != nil {
			return nil, err
		}
		vw, err := st.ReadTensorF32(prefix + "attention.self.value.weight")
		if err != nil {
			return nil, err
		}
		vb, err := st.ReadTensorF32(prefix + "attention.self.value.bias")
		if err != nil {
			return nil, err
		}
		ow, err := st.ReadTensorF32(prefix + "attention.output.dense.weight")
		if err != nil {
			return nil, err
		}
		ob, err := st.ReadTensorF32(prefix + "attention.output.dense.bias")
		if err != nil {
			return nil, err
		}
		anw, err := st.ReadTensorF32(prefix + "attention.output.LayerNorm.weight")
		if err != nil {
			return nil, err
		}
		anb, err := st.ReadTensorF32(prefix + "attention.output.LayerNorm.bias")
		if err != nil {
			return nil, err
		}

		// Feed Forward
		ffn1w, err := st.ReadTensorF32(prefix + "intermediate.dense.weight")
		if err != nil {
			return nil, err
		}
		ffn1b, err := st.ReadTensorF32(prefix + "intermediate.dense.bias")
		if err != nil {
			return nil, err
		}
		ffn2w, err := st.ReadTensorF32(prefix + "output.dense.weight")
		if err != nil {
			return nil, err
		}
		ffn2b, err := st.ReadTensorF32(prefix + "output.dense.bias")
		if err != nil {
			return nil, err
		}
		ffnw, err := st.ReadTensorF32(prefix + "output.LayerNorm.weight")
		if err != nil {
			return nil, err
		}
		ffnb, err := st.ReadTensorF32(prefix + "output.LayerNorm.bias")
		if err != nil {
			return nil, err
		}

		m.Layers[i] = LayerWeights{
			QueryWeight: mat.NewDense[float32](mat.WithShape(HiddenSize, HiddenSize), mat.WithBacking(qw)),
			QueryBias:   mat.NewDense[float32](mat.WithShape(HiddenSize, 1), mat.WithBacking(qb)),
			KeyWeight:   mat.NewDense[float32](mat.WithShape(HiddenSize, HiddenSize), mat.WithBacking(kw)),
			KeyBias:     mat.NewDense[float32](mat.WithShape(HiddenSize, 1), mat.WithBacking(kb)),
			ValueWeight: mat.NewDense[float32](mat.WithShape(HiddenSize, HiddenSize), mat.WithBacking(vw)),
			ValueBias:   mat.NewDense[float32](mat.WithShape(HiddenSize, 1), mat.WithBacking(vb)),
			OutWeight:   mat.NewDense[float32](mat.WithShape(HiddenSize, HiddenSize), mat.WithBacking(ow)),
			OutBias:     mat.NewDense[float32](mat.WithShape(HiddenSize, 1), mat.WithBacking(ob)),
			AttnNormW:   mat.NewDense[float32](mat.WithShape(HiddenSize, 1), mat.WithBacking(anw)),
			AttnNormB:   mat.NewDense[float32](mat.WithShape(HiddenSize, 1), mat.WithBacking(anb)),

			FFN1Weight: mat.NewDense[float32](mat.WithShape(IntermediateSize, HiddenSize), mat.WithBacking(ffn1w)),
			FFN1Bias:   mat.NewDense[float32](mat.WithShape(IntermediateSize, 1), mat.WithBacking(ffn1b)),
			FFN2Weight: mat.NewDense[float32](mat.WithShape(HiddenSize, IntermediateSize), mat.WithBacking(ffn2w)),
			FFN2Bias:   mat.NewDense[float32](mat.WithShape(HiddenSize, 1), mat.WithBacking(ffn2b)),
			FFNNormW:   mat.NewDense[float32](mat.WithShape(HiddenSize, 1), mat.WithBacking(ffnw)),
			FFNNormB:   mat.NewDense[float32](mat.WithShape(HiddenSize, 1), mat.WithBacking(ffnb)),
		}
	}

	return m, nil
}

// ForwardLayerNorm computes LayerNorm(x) with gamma and bias weights and epsilon.
func forwardLayerNorm(x mat.Tensor, weight, bias mat.Matrix, eps float64) mat.Tensor {
	mean := ag.ReduceMean(x)
	diff := ag.SubScalar(x, mean)
	sqDiff := ag.Square(diff)
	variance := ag.ReduceMean(sqDiff)
	epsScalar := mat.Scalar[float32](float32(eps))
	std := ag.Sqrt(ag.Add(variance, epsScalar))
	norm := ag.DivScalar(diff, std)
	return ag.Add(ag.Prod(norm, weight), bias)
}

// EncodeText executes the full forward pipeline using Spago computational graph.
func (m *SpagoModel) EncodeText(text string) ([]float32, error) {
	inputIDs, _ := m.Tok.Encode(text, MaxPositions)
	return m.EncodeTokenIDs(inputIDs)
}

// EncodeTokenIDs runs the transformer encoder, mean pooling, and L2 normalization.
func (m *SpagoModel) EncodeTokenIDs(inputIDs []int) ([]float32, error) {
	seqLen := len(inputIDs)
	if seqLen == 0 {
		return nil, fmt.Errorf("empty inputIDs")
	}

	// 1. Embedding lookup for each token
	embeddedTokens := make([]mat.Tensor, seqLen)
	for t, id := range inputIDs {
		wordEmb := ag.T(ag.RowView(m.WordEmbeddings, id))
		posEmb := ag.T(ag.RowView(m.PositionEmbeddings, t))
		typeEmb := ag.T(ag.RowView(m.TokenTypeEmbeddings, 0))

		sumEmb := ag.Add(ag.Add(wordEmb, posEmb), typeEmb)
		normEmb := forwardLayerNorm(sumEmb, m.EmbeddingsNormW, m.EmbeddingsNormB, LayerNormEps)
		embeddedTokens[t] = normEmb
	}

	x := embeddedTokens // []mat.Tensor of length seqLen

	// 2. Transformer layers (12 layers)
	for l := 0; l < NumLayers; l++ {
		layer := &m.Layers[l]

		// Multi-head self attention
		// Compute Q, K, V for each position
		q := make([]mat.Tensor, seqLen)
		k := make([]mat.Tensor, seqLen)
		v := make([]mat.Tensor, seqLen)

		for t := 0; t < seqLen; t++ {
			q[t] = ag.Affine(layer.QueryBias, layer.QueryWeight, x[t])
			k[t] = ag.Affine(layer.KeyBias, layer.KeyWeight, x[t])
			v[t] = ag.Affine(layer.ValueBias, layer.ValueWeight, x[t])
		}

		// Split into 12 heads and compute attention
		// Scale factor 1.0 / sqrt(32)
		scaleScalar := mat.Scalar[float32](float32(1.0 / math.Sqrt(float64(HeadDim))))
		attnOut := make([]mat.Tensor, seqLen)

		// For each position t
		for t := 0; t < seqLen; t++ {
			headOutputs := make([]mat.Tensor, NumHeads)
			for h := 0; h < NumHeads; h++ {
				hStart := h * HeadDim
				hEnd := hStart + HeadDim

				qh := ag.Slice(q[t], hStart, 0, hEnd, 1)

				// Scores for all key positions s
				scores := make([]mat.Tensor, seqLen)
				for s := 0; s < seqLen; s++ {
					kh := ag.Slice(k[s], hStart, 0, hEnd, 1)
					dot := ag.Dot(qh, kh)
					scores[s] = ag.ProdScalar(dot, scaleScalar)
				}

				// Softmax over scores
				stackedScores := ag.Stack(scores...)
				probs := ag.Softmax(stackedScores)

				// Weighted sum of values
				weightedSum := make([]mat.Tensor, seqLen)
				for s := 0; s < seqLen; s++ {
					vh := ag.Slice(v[s], hStart, 0, hEnd, 1)
					probS := ag.Slice(probs, s, 0, s+1, 1)
					weightedSum[s] = ag.ProdScalar(vh, probS)
				}
				headOutputs[h] = ag.Sum(weightedSum...)
			}

			// Concatenate all 12 heads: [HiddenSize]
			concatHeads := ag.Concat(headOutputs...)
			projOut := ag.Affine(layer.OutBias, layer.OutWeight, concatHeads)

			// Residual + Post-LN
			res := ag.Add(x[t], projOut)
			attnOut[t] = forwardLayerNorm(res, layer.AttnNormW, layer.AttnNormB, LayerNormEps)
		}

		// Feed-Forward Network + Residual + Post-LN
		layerOut := make([]mat.Tensor, seqLen)
		for t := 0; t < seqLen; t++ {
			ffn1 := ag.Affine(layer.FFN1Bias, layer.FFN1Weight, attnOut[t])
			act := ag.GELU(ffn1)
			ffn2 := ag.Affine(layer.FFN2Bias, layer.FFN2Weight, act)

			res := ag.Add(attnOut[t], ffn2)
			layerOut[t] = forwardLayerNorm(res, layer.FFNNormW, layer.FFNNormB, LayerNormEps)
		}

		x = layerOut
	}

	// 3. Mean Pooling across all token positions
	pooled := ag.Mean(x)

	// 4. L2 Normalization: v / sqrt(sum(v_i^2) + eps)
	vData := mat.Data[float32](pooled.Value())
	var sumSq float64
	for _, val := range vData {
		vF := float64(val)
		sumSq += vF * vF
	}
	norm := float32(math.Sqrt(math.Max(sumSq, 1e-12)))

	res := make([]float32, HiddenSize)
	for i, val := range vData {
		res[i] = val / norm
	}

	return res, nil
}
