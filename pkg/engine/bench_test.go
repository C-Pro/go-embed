package engine_test

import (
	"fmt"
	"path/filepath"
	"strings"
	"testing"

	"go-embed/pkg/engine"
)

func makeDummyText(targetLen int) string {
	word := "consensus "
	repeat := (targetLen / 2) + 1
	res := strings.Repeat(word, repeat)
	return "passage: " + res
}

// BenchmarkComparative benchmarks Pure Go Scalar vs Pure Go SIMD across FP32, BF16, and INT8.
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

		// 1. Pure Go Scalar Engine (Zero-Allocation Context)
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

		// 4. Pure Go INT8 Scalar Engine
		b.Run(fmt.Sprintf("INT8Scalar/L=%d", actualLen), func(b *testing.B) {
			qEng, err := engine.NewQuantized(modelPath, tokPath)
			if err != nil {
				b.Fatalf("Failed to load quantized engine: %v", err)
			}
			ctx := engine.NewContext(qEng.Model())
			ctx.UseSIMD = false
			out := make([]float32, engine.HiddenSize)
			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				if _, err := ctx.EmbedTokenIDs(toks, nil, out); err != nil {
					b.Fatalf("INT8 Scalar encode failed: %v", err)
				}
			}
		})

		// 5. Pure Go INT8 SIMD Engine
		b.Run(fmt.Sprintf("INT8SIMD/L=%d", actualLen), func(b *testing.B) {
			qEng, err := engine.NewQuantized(modelPath, tokPath)
			if err != nil {
				b.Fatalf("Failed to load quantized engine: %v", err)
			}
			ctx := engine.NewContext(qEng.Model())
			ctx.UseSIMD = true
			out := make([]float32, engine.HiddenSize)
			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				if _, err := ctx.EmbedTokenIDs(toks, nil, out); err != nil {
					b.Fatalf("INT8 SIMD encode failed: %v", err)
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

	// BF16 Benchmarks
	bf16Eng, err := engine.NewBF16(modelPath, tokPath)
	if err != nil {
		b.Fatalf("Failed to load BF16 engine: %v", err)
	}
	bf16Ctx := engine.NewContext(bf16Eng.Model())

	b.Run("BF16/Scalar/EndToEnd", func(b *testing.B) {
		bf16Ctx.UseSIMD = false
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			if _, err := bf16Ctx.Embed(query, out); err != nil {
				b.Fatalf("Embed failed: %v", err)
			}
		}
	})

	b.Run("BF16/SIMD/EndToEnd", func(b *testing.B) {
		bf16Ctx.UseSIMD = true
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			if _, err := bf16Ctx.Embed(query, out); err != nil {
				b.Fatalf("Embed failed: %v", err)
			}
		}
	})

	// INT8 Quantized Benchmarks
	qEng, err := engine.NewQuantized(modelPath, tokPath)
	if err != nil {
		b.Fatalf("Failed to load quantized engine: %v", err)
	}
	qCtx := engine.NewContext(qEng.Model())

	b.Run("INT8/Scalar/EndToEnd", func(b *testing.B) {
		qCtx.UseSIMD = false
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			if _, err := qCtx.Embed(query, out); err != nil {
				b.Fatalf("Embed failed: %v", err)
			}
		}
	})

	b.Run("INT8/SIMD/EndToEnd", func(b *testing.B) {
		qCtx.UseSIMD = true
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			if _, err := qCtx.Embed(query, out); err != nil {
				b.Fatalf("Embed failed: %v", err)
			}
		}
	})
}

func BenchmarkModelComparison(b *testing.B) {
	e5Dir := filepath.Join("..", "..", "models", "intfloat", "multilingual-e5-small")
	minilmDir := filepath.Join("..", "..", "models", "sentence-transformers", "paraphrase-multilingual-MiniLM-L12-v2")

	e5Eng, err := engine.NewEngine(engine.WithDataDir(e5Dir))
	if err != nil {
		b.Fatalf("Failed to load E5 engine: %v", err)
	}
	minilmEng, err := engine.NewEngine(engine.WithDataDir(minilmDir))
	if err != nil {
		b.Fatalf("Failed to load MiniLM engine: %v", err)
	}

	e5Ctx := engine.NewContext(e5Eng.Model())
	minilmCtx := engine.NewContext(minilmEng.Model())
	out := make([]float32, engine.HiddenSize)
	text := "How do you implement consensus in a distributed system?"

	b.Run("Multilingual-E5-Small/FP32-SIMD", func(b *testing.B) {
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			if _, err := e5Ctx.Embed(text, out); err != nil {
				b.Fatalf("Embed failed: %v", err)
			}
		}
	})

	b.Run("MiniLM-L12-v2/FP32-SIMD", func(b *testing.B) {
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			if _, err := minilmCtx.Embed(text, out); err != nil {
				b.Fatalf("Embed failed: %v", err)
			}
		}
	})
}
