package engine_test

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"go-embed/pkg/engine"
)

type GoldenEntry struct {
	Text            string    `json:"text"`
	InputIDs        []int     `json:"input_ids"`
	AttentionMask   []int8    `json:"attention_mask"`
	Tokens          []string  `json:"tokens"`
	SeqLen          int       `json:"seq_len"`
	PooledEmbedding []float32 `json:"pooled_embedding"`
	Embedding       []float32 `json:"embedding"`
}

func loadTestModel(t *testing.T) *engine.Engine {
	t.Helper()
	modelPath := filepath.Join("..", "..", "models", "intfloat", "multilingual-e5-small", "model.safetensors")
	tokPath := filepath.Join("..", "..", "models", "intfloat", "multilingual-e5-small", "tokenizer.json")

	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skipf("model file %s not found", modelPath)
	}

	eng, err := engine.New(modelPath, tokPath)
	if err != nil {
		t.Fatalf("Failed to initialize engine: %v", err)
	}
	return eng
}

func loadGolden(t *testing.T) []GoldenEntry {
	t.Helper()
	goldenPath := filepath.Join("..", "..", "testdata", "golden.json")
	data, err := os.ReadFile(goldenPath)
	if err != nil {
		t.Fatalf("Failed to read golden.json: %v", err)
	}

	var entries []GoldenEntry
	if err := json.Unmarshal(data, &entries); err != nil {
		t.Fatalf("Failed to parse golden.json: %v", err)
	}
	return entries
}

func TestGoldenParity(t *testing.T) {
	eng := loadTestModel(t)
	entries := loadGolden(t)
	ctx := engine.NewContext(eng.Model())

	for i, entry := range entries {
		t.Run(entry.Text, func(t *testing.T) {
			out := make([]float32, engine.HiddenSize)
			_, err := ctx.Embed(entry.Text, out)
			if err != nil {
				t.Fatalf("Case #%d failed: %v", i, err)
			}

			cosSim := engine.CosineSimilarity(out, entry.Embedding)
			var maxDelta float32
			for d := 0; d < engine.HiddenSize; d++ {
				diff := float32(math.Abs(float64(out[d] - entry.Embedding[d])))
				if diff > maxDelta {
					maxDelta = diff
				}
			}

			t.Logf("Case #%d: seqLen=%d, CosineSim=%.7f, MaxDelta=%.2e", i, entry.SeqLen, cosSim, maxDelta)

			if cosSim < 0.9999 {
				t.Errorf("Case #%d (%q) cosine similarity too low: %.7f (expected >= 0.9999)", i, entry.Text, cosSim)
			}
			if maxDelta > 1e-4 {
				t.Errorf("Case #%d (%q) max delta too large: %.2e (expected <= 1e-4)", i, entry.Text, maxDelta)
			}
		})
	}
}

func TestScalarVsSIMDParity(t *testing.T) {
	eng := loadTestModel(t)
	entries := loadGolden(t)

	ctxScalar := engine.NewContext(eng.Model())
	ctxScalar.UseSIMD = false

	ctxSIMD := engine.NewContext(eng.Model())
	ctxSIMD.UseSIMD = true

	for i, entry := range entries {
		outScalar := make([]float32, engine.HiddenSize)
		outSIMD := make([]float32, engine.HiddenSize)

		if _, err := ctxScalar.Embed(entry.Text, outScalar); err != nil {
			t.Fatalf("Scalar embed failed: %v", err)
		}
		if _, err := ctxSIMD.Embed(entry.Text, outSIMD); err != nil {
			t.Fatalf("SIMD embed failed: %v", err)
		}

		cosSim := engine.CosineSimilarity(outScalar, outSIMD)
		var maxDelta float32
		for d := 0; d < engine.HiddenSize; d++ {
			diff := float32(math.Abs(float64(outScalar[d] - outSIMD[d])))
			if diff > maxDelta {
				maxDelta = diff
			}
		}

		if cosSim < 0.99999 {
			t.Errorf("Case #%d Scalar vs SIMD Cosine similarity too low: %.7f", i, cosSim)
		}
		if maxDelta > 1e-5 {
			t.Errorf("Case #%d Scalar vs SIMD Max delta too large: %.2e", i, maxDelta)
		}
	}
}

func TestZeroAllocations(t *testing.T) {
	eng := loadTestModel(t)
	ctx := engine.NewContext(eng.Model())
	out := make([]float32, engine.HiddenSize)
	query := "query: how to implement consensus in distributed systems?"

	// Warm up
	if _, err := ctx.Embed(query, out); err != nil {
		t.Fatalf("Warmup failed: %v", err)
	}

	// Test zero allocations
	allocs := testing.AllocsPerRun(100, func() {
		if _, err := ctx.Embed(query, out); err != nil {
			t.Fatalf("Embed failed: %v", err)
		}
	})

	t.Logf("Steady-state allocations per run: %.1f", allocs)
	if allocs != 0 {
		t.Errorf("Expected 0 allocations per run, got %.1f", allocs)
	}
}
