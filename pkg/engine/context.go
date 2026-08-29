package engine

import (
	"fmt"
	"strings"
	"sync"

	"go-embed/pkg/tokenizer"
)

const (
	scaleFactor = float32(0.17677669529663687) // 1.0 / sqrt(32)
)

// InferenceContext encapsulates all pre-allocated scratchpad buffers
// needed to execute forward inference passes with 0 heap allocations.
type InferenceContext struct {
	Model   *Model
	UseSIMD bool

	RuneBuf       []rune
	InputIDs      []int
	AttentionMask []int8
	DPBuf         []tokenizer.DPState

	HiddenStates []float32 // [512 * 384]
	Residual     []float32 // [512 * 384]
	Q            []float32 // [512 * 384]
	K            []float32 // [512 * 384]
	V            []float32 // [512 * 384]
	AttnContext  []float32 // [512 * 384]
	AttnOut      []float32 // [512 * 384]
	Scores       []float32 // [512]
	FFNMid       []float32 // [512 * 1536]
	FFNOut       []float32 // [512 * 384]
	Pooled       []float32 // [384]
	Output       []float32 // [384]
}

// NewContext creates a new InferenceContext with all buffers pre-allocated.
func NewContext(m *Model) *InferenceContext {
	return &InferenceContext{
		Model:         m,
		UseSIMD:       HasSIMD,
		RuneBuf:       make([]rune, 0, 1024),
		InputIDs:      make([]int, 0, MaxSeqLen),
		AttentionMask: make([]int8, 0, MaxSeqLen),
		DPBuf:         make([]tokenizer.DPState, 0, 2048),
		HiddenStates:  make([]float32, MaxSeqLen*HiddenSize),
		Residual:      make([]float32, MaxSeqLen*HiddenSize),
		Q:             make([]float32, MaxSeqLen*HiddenSize),
		K:             make([]float32, MaxSeqLen*HiddenSize),
		V:             make([]float32, MaxSeqLen*HiddenSize),
		AttnContext:   make([]float32, MaxSeqLen*HiddenSize),
		AttnOut:       make([]float32, MaxSeqLen*HiddenSize),
		Scores:        make([]float32, MaxSeqLen),
		FFNMid:        make([]float32, MaxSeqLen*IntermediateSize),
		FFNOut:        make([]float32, MaxSeqLen*HiddenSize),
		Pooled:        make([]float32, HiddenSize),
		Output:        make([]float32, HiddenSize),
	}
}

// ContextPool manages reusable InferenceContext instances for high concurrency.
type ContextPool struct {
	pool sync.Pool
}

// NewContextPool creates a new pool of inference contexts for the given model.
func NewContextPool(m *Model) *ContextPool {
	return &ContextPool{
		pool: sync.Pool{
			New: func() any {
				return NewContext(m)
			},
		},
	}
}

// Get borrows an InferenceContext from the pool.
func (cp *ContextPool) Get() *InferenceContext {
	return cp.pool.Get().(*InferenceContext)
}

// Put returns an InferenceContext back to the pool.
func (cp *ContextPool) Put(ctx *InferenceContext) {
	cp.pool.Put(ctx)
}

// Embed generates an L2-normalized 384-dimensional embedding for raw text.
// If out is provided (must have cap >= 384), it writes directly into out without allocation.
// If out is nil, it uses ctx.Output.
func (ctx *InferenceContext) Embed(text string, out []float32) ([]float32, error) {
	if out == nil {
		out = ctx.Output
	} else if len(out) < HiddenSize {
		out = out[:HiddenSize]
	}

	// Zero-allocation tokenization
	ctx.RuneBuf, ctx.InputIDs, ctx.AttentionMask = ctx.Model.Tok.EncodeIntoZeroAlloc(
		text,
		ctx.RuneBuf,
		ctx.InputIDs,
		ctx.AttentionMask,
		ctx.DPBuf,
		MaxSeqLen,
	)

	return ctx.Forward(len(ctx.InputIDs), out), nil
}

// EmbedQuery embeds text with 'query: ' prefix.
func (ctx *InferenceContext) EmbedQuery(text string, out []float32) ([]float32, error) {
	if !strings.HasPrefix(text, "query: ") {
		text = "query: " + text
	}
	return ctx.Embed(text, out)
}

// EmbedPassage embeds text with 'passage: ' prefix.
func (ctx *InferenceContext) EmbedPassage(text string, out []float32) ([]float32, error) {
	if !strings.HasPrefix(text, "passage: ") {
		text = "passage: " + text
	}
	return ctx.Embed(text, out)
}

// EmbedTokenIDs runs the forward pass given pre-computed token IDs and attention mask.
func (ctx *InferenceContext) EmbedTokenIDs(inputIDs []int, attnMask []int8, out []float32) ([]float32, error) {
	if len(inputIDs) == 0 {
		return nil, fmt.Errorf("empty inputIDs")
	}
	if len(inputIDs) > MaxSeqLen {
		inputIDs = inputIDs[:MaxSeqLen]
		inputIDs[MaxSeqLen-1] = tokenizer.EOS_ID
	}

	ctx.InputIDs = append(ctx.InputIDs[:0], inputIDs...)
	if attnMask != nil {
		ctx.AttentionMask = append(ctx.AttentionMask[:0], attnMask[:len(inputIDs)]...)
	} else {
		ctx.AttentionMask = ctx.AttentionMask[:0]
		for i := 0; i < len(inputIDs); i++ {
			ctx.AttentionMask = append(ctx.AttentionMask, 1)
		}
	}

	if out == nil {
		out = ctx.Output
	} else if len(out) < HiddenSize {
		out = out[:HiddenSize]
	}

	return ctx.Forward(len(inputIDs), out), nil
}

// Forward executes the transformer encoder, mean pooling, and L2 normalization.
func (ctx *InferenceContext) Forward(seqLen int, out []float32) []float32 {
	m := ctx.Model

	matMul := MatVecMulAddScalar
	layerNorm := LayerNormScalar
	gelu := GELUScalar
	l2Norm := L2NormalizeScalar

	if ctx.UseSIMD && HasSIMD {
		matMul = MatVecMulAddSIMD
		layerNorm = LayerNormSIMD
		gelu = GELUSIMD
		l2Norm = L2NormalizeSIMD
	}

	// 1. Embeddings lookup & LayerNorm
	for t := 0; t < seqLen; t++ {
		id := ctx.InputIDs[t]
		if id < 0 || id >= VocabSize {
			id = tokenizer.UNK_ID
		}

		wOffset := id * HiddenSize
		pOffset := t * HiddenSize
		hOffset := t * HiddenSize

		wEmb := m.WordEmbeddings[wOffset : wOffset+HiddenSize]
		pEmb := m.PositionEmbeddings[pOffset : pOffset+HiddenSize]
		tEmb := m.TokenTypeEmbeddings[:HiddenSize]
		sumSlice := ctx.Residual[hOffset : hOffset+HiddenSize]

		for d := 0; d < HiddenSize; d++ {
			sumSlice[d] = wEmb[d] + pEmb[d] + tEmb[d]
		}

		layerNorm(
			sumSlice,
			m.EmbeddingsNormW,
			m.EmbeddingsNormB,
			ctx.HiddenStates[hOffset:hOffset+HiddenSize],
			HiddenSize,
			LayerNormEps,
		)
	}

	// 2. 12 Transformer Layers
	for l := 0; l < NumLayers; l++ {
		layer := &m.Layers[l]

		// Multi-head self attention projections Q, K, V
		for t := 0; t < seqLen; t++ {
			tOffset := t * HiddenSize
			xt := ctx.HiddenStates[tOffset : tOffset+HiddenSize]

			matMul(xt, layer.QueryWeight, layer.QueryBias, ctx.Q[tOffset:tOffset+HiddenSize], HiddenSize, HiddenSize)
			matMul(xt, layer.KeyWeight, layer.KeyBias, ctx.K[tOffset:tOffset+HiddenSize], HiddenSize, HiddenSize)
			matMul(xt, layer.ValueWeight, layer.ValueBias, ctx.V[tOffset:tOffset+HiddenSize], HiddenSize, HiddenSize)
		}

		// Scaled dot-product attention per head
		for h := 0; h < NumHeads; h++ {
			hOffset := h * HeadDim

			for t := 0; t < seqLen; t++ {
				qt := ctx.Q[t*HiddenSize+hOffset : t*HiddenSize+hOffset+HeadDim]

				// Compute scores against all positions s
				for s := 0; s < seqLen; s++ {
					ks := ctx.K[s*HiddenSize+hOffset : s*HiddenSize+hOffset+HeadDim]
					var dot float32
					for d := 0; d < HeadDim; d++ {
						dot += qt[d] * ks[d]
					}
					ctx.Scores[s] = dot * scaleFactor
				}

				// Softmax over sequence scores
				SoftmaxScalar(ctx.Scores[:seqLen], ctx.Scores[:seqLen], seqLen)

				// Weighted sum of values
				for d := 0; d < HeadDim; d++ {
					var sumVal float32
					for s := 0; s < seqLen; s++ {
						sumVal += ctx.Scores[s] * ctx.V[s*HiddenSize+hOffset+d]
					}
					ctx.AttnContext[t*HiddenSize+hOffset+d] = sumVal
				}
			}
		}

		// Attention output projection, residual connection, and LayerNorm
		for t := 0; t < seqLen; t++ {
			tOffset := t * HiddenSize
			cSlice := ctx.AttnContext[tOffset : tOffset+HiddenSize]
			matMul(cSlice, layer.OutWeight, layer.OutBias, ctx.AttnOut[tOffset:tOffset+HiddenSize], HiddenSize, HiddenSize)

			resSlice := ctx.Residual[tOffset : tOffset+HiddenSize]
			hSlice := ctx.HiddenStates[tOffset : tOffset+HiddenSize]
			attnOutSlice := ctx.AttnOut[tOffset : tOffset+HiddenSize]

			for d := 0; d < HiddenSize; d++ {
				resSlice[d] = hSlice[d] + attnOutSlice[d]
			}

			layerNorm(resSlice, layer.AttnNormW, layer.AttnNormB, hSlice, HiddenSize, LayerNormEps)
		}

		// Feed-Forward Network: Linear (384 -> 1536) -> GELU -> Linear (1536 -> 384) -> Residual -> LayerNorm
		for t := 0; t < seqLen; t++ {
			tOffsetH := t * HiddenSize
			tOffsetFFN := t * IntermediateSize

			xt := ctx.HiddenStates[tOffsetH : tOffsetH+HiddenSize]
			ffnMid := ctx.FFNMid[tOffsetFFN : tOffsetFFN+IntermediateSize]
			ffnOut := ctx.FFNOut[tOffsetH : tOffsetH+HiddenSize]

			matMul(xt, layer.FFN1Weight, layer.FFN1Bias, ffnMid, HiddenSize, IntermediateSize)
			gelu(ffnMid, ffnMid, IntermediateSize)
			matMul(ffnMid, layer.FFN2Weight, layer.FFN2Bias, ffnOut, IntermediateSize, HiddenSize)

			resSlice := ctx.Residual[tOffsetH : tOffsetH+HiddenSize]
			for d := 0; d < HiddenSize; d++ {
				resSlice[d] = xt[d] + ffnOut[d]
			}

			layerNorm(resSlice, layer.FFNNormW, layer.FFNNormB, xt, HiddenSize, LayerNormEps)
		}
	}

	// 3. Mean Pooling
	MeanPoolScalar(ctx.HiddenStates, ctx.AttentionMask, ctx.Pooled, seqLen, HiddenSize)

	// 4. L2 Normalization
	l2Norm(ctx.Pooled, out, HiddenSize)

	return out
}

// CosineSimilarity computes cosine similarity between two 384-dimensional embeddings.
func CosineSimilarity(a, b []float32) float32 {
	if HasSIMD {
		return CosineSimilaritySIMD(a, b, HiddenSize)
	}
	return CosineSimilarityScalar(a, b, HiddenSize)
}
