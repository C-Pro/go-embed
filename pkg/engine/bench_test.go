package engine_test

import (
	"fmt"
	"path/filepath"
	"strings"
	"testing"

	"go-embed/pkg/engine"
	"go-embed/pkg/spagoref"
)

func makeDummyText(targetLen int) string {
	word := "consensus "
	repeat := (targetLen / 2) + 1
	res := strings.Repeat(word, repeat)
	return "passage: " + res
}

// BenchmarkComparative benchmarks Spago Baseline vs Pure Go Scalar vs Pure Go SIMD.
func BenchmarkComparative(b *testing.B) {
	modelPath := filepath.Join("..", "..", "models", "intfloat", "multilingual-e5-small", "model.safetensors")
	tokPath := filepath.Join("..", "..", "models", "intfloat", "multilingual-e5-small", "tokenizer.json")

	eng, err := engine.New(modelPath, tokPath)
	if err != nil {
		b.Fatalf("Failed to load engine: %v", err)
	}

	seqLengths := []int{32, 128, 256, 512}

	for _, l := range seqLengths {
		txt := makeDummyText(l)
		toks, _ := eng.Model().Tok.Encode(txt, l)
		actualLen := len(toks)

		// 1. Spago Baseline (only L <= 128 due to extreme memory overhead of dynamic computation graphs)
		if actualLen <= 128 {
			b.Run(fmt.Sprintf("Spago/L=%d", actualLen), func(b *testing.B) {
				spagoModel, err := spagoref.LoadModel(modelPath, tokPath)
				if err != nil {
					b.Fatalf("Failed to load spago model: %v", err)
				}
				b.ResetTimer()
				b.ReportAllocs()

				for i := 0; i < b.N; i++ {
					if _, err := spagoModel.EncodeTokenIDs(toks); err != nil {
						b.Fatalf("Spago encode failed: %v", err)
					}
				}
			})
		}

		// 2. Pure Go Scalar Engine (Zero-Allocation Context)
		b.Run(fmt.Sprintf("ScalarEngine/L=%d", actualLen), func(b *testing.B) {
			ctx := engine.NewContext(eng.Model())
			ctx.UseSIMD = false
			out := make([]float32, engine.HiddenSize)
			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				if _, err := ctx.EmbedTokenIDs(toks, nil, out); err != nil {
					b.Fatalf("Scalar encode failed: %v", err)
				}
			}
		})

		// 3. Pure Go SIMD Engine (Zero-Allocation Context)
		b.Run(fmt.Sprintf("SIMDEngine/L=%d", actualLen), func(b *testing.B) {
			ctx := engine.NewContext(eng.Model())
			ctx.UseSIMD = true
			out := make([]float32, engine.HiddenSize)
			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				if _, err := ctx.EmbedTokenIDs(toks, nil, out); err != nil {
					b.Fatalf("SIMD encode failed: %v", err)
				}
			}
		})
	}
}

// BenchmarkBatch measures multi-threaded batch embedding throughput using ContextPool.
func BenchmarkBatch(b *testing.B) {
	modelPath := filepath.Join("..", "..", "models", "intfloat", "multilingual-e5-small", "model.safetensors")
	tokPath := filepath.Join("..", "..", "models", "intfloat", "multilingual-e5-small", "tokenizer.json")

	eng, err := engine.New(modelPath, tokPath)
	if err != nil {
		b.Fatalf("Failed to load engine: %v", err)
	}

	batchSizes := []int{1, 4, 8, 16}
	sampleText := "query: how to implement consensus in distributed systems?"

	for _, bs := range batchSizes {
		texts := make([]string, bs)
		for i := 0; i < bs; i++ {
			texts[i] = sampleText
		}

		b.Run(fmt.Sprintf("BatchSize=%d", bs), func(b *testing.B) {
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				embs, err := eng.EmbedBatch(texts)
				if err != nil || len(embs) != bs {
					b.Fatalf("EmbedBatch failed: %v", err)
				}
			}
		})
	}
}

// BenchmarkShortQuery measures end-to-end embedding generation latency for a typical short query.
func BenchmarkShortQuery(b *testing.B) {
	modelPath := filepath.Join("..", "..", "models", "intfloat", "multilingual-e5-small", "model.safetensors")
	tokPath := filepath.Join("..", "..", "models", "intfloat", "multilingual-e5-small", "tokenizer.json")

	eng, err := engine.New(modelPath, tokPath)
	if err != nil {
		b.Fatalf("Failed to load engine: %v", err)
	}

	query := "query: how to implement consensus in distributed systems?"
	ctx := engine.NewContext(eng.Model())
	out := make([]float32, engine.HiddenSize)

	b.Run("Scalar/EndToEnd", func(b *testing.B) {
		ctx.UseSIMD = false
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			if _, err := ctx.Embed(query, out); err != nil {
				b.Fatalf("Embed failed: %v", err)
			}
		}
	})

	b.Run("SIMD/EndToEnd", func(b *testing.B) {
		ctx.UseSIMD = true
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			if _, err := ctx.Embed(query, out); err != nil {
				b.Fatalf("Embed failed: %v", err)
			}
		}
	})
}
