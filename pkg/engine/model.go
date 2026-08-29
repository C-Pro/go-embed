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
// It supports FP32 full precision, BF16 16-bit float, and INT8 8-bit dynamic quantization.
type Model struct {
	Precision   PrecisionMode
	IsQuantized bool

	// FP32 Parameters
	WordEmbeddings      []float32 // [250037 * 384]
	PositionEmbeddings  []float32 // [512 * 384]
	TokenTypeEmbeddings []float32 // [2 * 384]
	EmbeddingsNormW     []float32 // [384]
	EmbeddingsNormB     []float32 // [384]
	Layers              [NumLayers]Layer

	// BF16 Parameters
	BF16WordEmbeddings []uint16 // [250037 * 384]
	BF16Layers         [NumLayers]BF16Layer

	// INT8 Quantized Parameters
	QWordEmbeddings QuantizedWordEmbeddings
	QLayers         [NumLayers]QuantizedLayer

	Tok *tokenizer.Tokenizer
}

// LoadModel loads the model directly from a safetensors file and tokenizer.json in FP32 precision.
func LoadModel(modelPath, tokenizerPath string) (*Model, error) {
	return LoadModelWithPrecision(modelPath, tokenizerPath, PrecisionFP32)
}

// LoadBF16Model loads the model directly and converts all linear layers and embeddings to BFloat16.
func LoadBF16Model(modelPath, tokenizerPath string) (*Model, error) {
	return LoadModelWithPrecision(modelPath, tokenizerPath, PrecisionBF16)
}

// LoadQuantizedModel loads the model directly and converts all linear layers and embeddings to INT8.
func LoadQuantizedModel(modelPath, tokenizerPath string) (*Model, error) {
	return LoadModelWithPrecision(modelPath, tokenizerPath, PrecisionINT8)
}

// LoadModelWithOptions loads model weights with optional INT8 quantization (backwards compatible).
func LoadModelWithOptions(modelPath, tokenizerPath string, quantize bool) (*Model, error) {
	if quantize {
		return LoadModelWithPrecision(modelPath, tokenizerPath, PrecisionINT8)
	}
	return LoadModelWithPrecision(modelPath, tokenizerPath, PrecisionFP32)
}

// VocabSize returns the vocabulary size of the loaded model.
func (m *Model) VocabSize() int {
	if len(m.WordEmbeddings) > 0 {
		return len(m.WordEmbeddings) / HiddenSize
	}
	if len(m.BF16WordEmbeddings) > 0 {
		return len(m.BF16WordEmbeddings) / HiddenSize
	}
	if len(m.QWordEmbeddings.Scale) > 0 {
		return len(m.QWordEmbeddings.Scale)
	}
	return VocabSize
}

func readTensorF32Flex(st *safetensors.Safetensors, name string) ([]float32, error) {
	if t, err := st.ReadTensorF32(name); err == nil {
		return t, nil
	}
	prefixes := []string{"roberta.", "bert.", "transformer.", "model."}
	for _, p := range prefixes {
		if t, err := st.ReadTensorF32(p + name); err == nil {
			return t, nil
		}
	}
	return nil, fmt.Errorf("tensor %q (and standard prefixes) not found in safetensors", name)
}

// LoadModelWithPrecision loads model weights with the specified precision mode.
func LoadModelWithPrecision(modelPath, tokenizerPath string, prec PrecisionMode) (*Model, error) {
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
	if m.WordEmbeddings, readErr = readTensorF32Flex(st, "embeddings.word_embeddings.weight"); readErr != nil {
		return nil, readErr
	}
	if m.PositionEmbeddings, readErr = readTensorF32Flex(st, "embeddings.position_embeddings.weight"); readErr != nil {
		return nil, readErr
	}
	if m.TokenTypeEmbeddings, readErr = readTensorF32Flex(st, "embeddings.token_type_embeddings.weight"); readErr != nil {
		// Optional in some architectures
		m.TokenTypeEmbeddings = make([]float32, HiddenSize)
	}
	if m.EmbeddingsNormW, readErr = readTensorF32Flex(st, "embeddings.LayerNorm.weight"); readErr != nil {
		return nil, readErr
	}
	if m.EmbeddingsNormB, readErr = readTensorF32Flex(st, "embeddings.LayerNorm.bias"); readErr != nil {
		return nil, readErr
	}

	// 2. 12 Transformer Layers
	for i := 0; i < NumLayers; i++ {
		prefix := fmt.Sprintf("encoder.layer.%d.", i)
		l := &m.Layers[i]

		if l.QueryWeight, readErr = readTensorF32Flex(st, prefix+"attention.self.query.weight"); readErr != nil {
			return nil, readErr
		}
		if l.QueryBias, readErr = readTensorF32Flex(st, prefix+"attention.self.query.bias"); readErr != nil {
			return nil, readErr
		}
		if l.KeyWeight, readErr = readTensorF32Flex(st, prefix+"attention.self.key.weight"); readErr != nil {
			return nil, readErr
		}
		if l.KeyBias, readErr = readTensorF32Flex(st, prefix+"attention.self.key.bias"); readErr != nil {
			return nil, readErr
		}
		if l.ValueWeight, readErr = readTensorF32Flex(st, prefix+"attention.self.value.weight"); readErr != nil {
			return nil, readErr
		}
		if l.ValueBias, readErr = readTensorF32Flex(st, prefix+"attention.self.value.bias"); readErr != nil {
			return nil, readErr
		}
		if l.OutWeight, readErr = readTensorF32Flex(st, prefix+"attention.output.dense.weight"); readErr != nil {
			return nil, readErr
		}
		if l.OutBias, readErr = readTensorF32Flex(st, prefix+"attention.output.dense.bias"); readErr != nil {
			return nil, readErr
		}
		if l.AttnNormW, readErr = readTensorF32Flex(st, prefix+"attention.output.LayerNorm.weight"); readErr != nil {
			return nil, readErr
		}
		if l.AttnNormB, readErr = readTensorF32Flex(st, prefix+"attention.output.LayerNorm.bias"); readErr != nil {
			return nil, readErr
		}

		if l.FFN1Weight, readErr = readTensorF32Flex(st, prefix+"intermediate.dense.weight"); readErr != nil {
			return nil, readErr
		}
		if l.FFN1Bias, readErr = readTensorF32Flex(st, prefix+"intermediate.dense.bias"); readErr != nil {
			return nil, readErr
		}
		if l.FFN2Weight, readErr = readTensorF32Flex(st, prefix+"output.dense.weight"); readErr != nil {
			return nil, readErr
		}
		if l.FFN2Bias, readErr = readTensorF32Flex(st, prefix+"output.dense.bias"); readErr != nil {
			return nil, readErr
		}
		if l.FFNNormW, readErr = readTensorF32Flex(st, prefix+"output.LayerNorm.weight"); readErr != nil {
			return nil, readErr
		}
		if l.FFNNormB, readErr = readTensorF32Flex(st, prefix+"output.LayerNorm.bias"); readErr != nil {
			return nil, readErr
		}
	}

	switch prec {
	case PrecisionBF16:
		return ConvertToBF16Model(m), nil
	case PrecisionINT8:
		return QuantizeModel(m), nil
	default:
		m.Precision = PrecisionFP32
		return m, nil
	}
}
