package engine

import (
	"fmt"

	"go-embed/pkg/safetensors"
	"go-embed/pkg/tokenizer"
)

const (
	HiddenSize       = 384
	IntermediateSize = 1536
	NumLayers        = 12
	NumHeads         = 12
	HeadDim          = 32
	MaxSeqLen        = 512
	VocabSize        = 250037
	LayerNormEps     = 1e-12
)

// Layer holds all parameter weights and biases for a single transformer layer.
// All matrices are stored in row-major order:
// - Query, Key, Value, Out: [384, 384] (each row has 384 elements)
// - FFN1: [1536, 384] (each row has 384 elements)
// - FFN2: [384, 1536] (each row has 1536 elements)
type Layer struct {
	QueryWeight []float32
	QueryBias   []float32
	KeyWeight   []float32
	KeyBias     []float32
	ValueWeight []float32
	ValueBias   []float32
	OutWeight   []float32
	OutBias     []float32
	AttnNormW   []float32
	AttnNormB   []float32

	FFN1Weight []float32
	FFN1Bias   []float32
	FFN2Weight []float32
	FFN2Bias   []float32
	FFNNormW   []float32
	FFNNormB   []float32
}

// Model represents the self-contained transformer embedding model.
type Model struct {
	WordEmbeddings      []float32 // [250037 * 384]
	PositionEmbeddings  []float32 // [512 * 384]
	TokenTypeEmbeddings []float32 // [2 * 384]
	EmbeddingsNormW     []float32 // [384]
	EmbeddingsNormB     []float32 // [384]

	Layers [NumLayers]Layer
	Tok    *tokenizer.Tokenizer
}

// LoadModel loads the model directly from a safetensors file and tokenizer.json.
func LoadModel(modelPath, tokenizerPath string) (*Model, error) {
	tok, err := tokenizer.LoadFromFile(tokenizerPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load tokenizer: %w", err)
	}

	st, err := safetensors.Open(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open safetensors file: %w", err)
	}
	defer st.Close()

	m := &Model{Tok: tok}

	// 1. Embeddings
	var readErr error
	if m.WordEmbeddings, readErr = st.ReadTensorF32("embeddings.word_embeddings.weight"); readErr != nil {
		return nil, readErr
	}
	if m.PositionEmbeddings, readErr = st.ReadTensorF32("embeddings.position_embeddings.weight"); readErr != nil {
		return nil, readErr
	}
	if m.TokenTypeEmbeddings, readErr = st.ReadTensorF32("embeddings.token_type_embeddings.weight"); readErr != nil {
		return nil, readErr
	}
	if m.EmbeddingsNormW, readErr = st.ReadTensorF32("embeddings.LayerNorm.weight"); readErr != nil {
		return nil, readErr
	}
	if m.EmbeddingsNormB, readErr = st.ReadTensorF32("embeddings.LayerNorm.bias"); readErr != nil {
		return nil, readErr
	}

	// 2. 12 Transformer Layers
	for i := 0; i < NumLayers; i++ {
		prefix := fmt.Sprintf("encoder.layer.%d.", i)
		l := &m.Layers[i]

		if l.QueryWeight, readErr = st.ReadTensorF32(prefix + "attention.self.query.weight"); readErr != nil {
			return nil, readErr
		}
		if l.QueryBias, readErr = st.ReadTensorF32(prefix + "attention.self.query.bias"); readErr != nil {
			return nil, readErr
		}
		if l.KeyWeight, readErr = st.ReadTensorF32(prefix + "attention.self.key.weight"); readErr != nil {
			return nil, readErr
		}
		if l.KeyBias, readErr = st.ReadTensorF32(prefix + "attention.self.key.bias"); readErr != nil {
			return nil, readErr
		}
		if l.ValueWeight, readErr = st.ReadTensorF32(prefix + "attention.self.value.weight"); readErr != nil {
			return nil, readErr
		}
		if l.ValueBias, readErr = st.ReadTensorF32(prefix + "attention.self.value.bias"); readErr != nil {
			return nil, readErr
		}
		if l.OutWeight, readErr = st.ReadTensorF32(prefix + "attention.output.dense.weight"); readErr != nil {
			return nil, readErr
		}
		if l.OutBias, readErr = st.ReadTensorF32(prefix + "attention.output.dense.bias"); readErr != nil {
			return nil, readErr
		}
		if l.AttnNormW, readErr = st.ReadTensorF32(prefix + "attention.output.LayerNorm.weight"); readErr != nil {
			return nil, readErr
		}
		if l.AttnNormB, readErr = st.ReadTensorF32(prefix + "attention.output.LayerNorm.bias"); readErr != nil {
			return nil, readErr
		}

		if l.FFN1Weight, readErr = st.ReadTensorF32(prefix + "intermediate.dense.weight"); readErr != nil {
			return nil, readErr
		}
		if l.FFN1Bias, readErr = st.ReadTensorF32(prefix + "intermediate.dense.bias"); readErr != nil {
			return nil, readErr
		}
		if l.FFN2Weight, readErr = st.ReadTensorF32(prefix + "output.dense.weight"); readErr != nil {
			return nil, readErr
		}
		if l.FFN2Bias, readErr = st.ReadTensorF32(prefix + "output.dense.bias"); readErr != nil {
			return nil, readErr
		}
		if l.FFNNormW, readErr = st.ReadTensorF32(prefix + "output.LayerNorm.weight"); readErr != nil {
			return nil, readErr
		}
		if l.FFNNormB, readErr = st.ReadTensorF32(prefix + "output.LayerNorm.bias"); readErr != nil {
			return nil, readErr
		}
	}

	return m, nil
}
