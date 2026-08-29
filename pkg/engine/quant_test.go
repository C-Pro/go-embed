package engine_test

import (
	"math"
	"path/filepath"
	"testing"

	"go-embed/pkg/engine"
)

func TestQuantizeMatrix(t *testing.T) {
	rows, cols := 4, 8
	weights := []float32{
		0.5, -0.2, 0.1, -0.8, 0.0, 0.3, -0.4, 0.7,
		-0.9, 0.4, -0.1, 0.6, -0.5, 0.2, -0.3, 0.8,
		0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8,
		-0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8,
	}

	qW, scales := engine.QuantizeMatrix(weights, rows, cols)
	if len(qW) != rows*cols || len(scales) != rows {
		t.Fatalf("Unexpected slice lengths: qW=%d, scales=%d", len(qW), len(scales))
	}

	// Verify dequantization error is minimal (< 1/127 of max amplitude)
	for r := 0; r < rows; r++ {
		scale := scales[r]
		for c := 0; c < cols; c++ {
			orig := weights[r*cols+c]
			dequant := float32(qW[r*cols+c]) * scale
			diff := float32(math.Abs(float64(orig - dequant)))
			if diff > scale {
				t.Errorf("Dequantization error at [%d, %d] too high: orig=%.4f, dequant=%.4f, diff=%.4f, scale=%.4f",
					r, c, orig, dequant, diff, scale)
			}
		}
	}
}

func TestQuantizedParityAgainstFP32(t *testing.T) {
	modelPath := filepath.Join("..", "..", "models", "intfloat", "multilingual-e5-small", "model.safetensors")
	tokPath := filepath.Join("..", "..", "models", "intfloat", "multilingual-e5-small", "tokenizer.json")

	fp32Eng, err := engine.New(modelPath, tokPath)
	if err != nil {
		t.Fatalf("Failed to load FP32 engine: %v", err)
	}

	int8Eng, err := engine.NewQuantized(modelPath, tokPath)
	if err != nil {
		t.Fatalf("Failed to load INT8 engine: %v", err)
	}

	if !int8Eng.IsQuantized() {
		t.Fatal("Expected INT8 engine to be quantized")
	}

	golden := loadGolden(t)

	for i, tc := range golden {
		t.Run(tc.Text, func(t *testing.T) {
			fp32Emb, err := fp32Eng.Embed(tc.Text)
			if err != nil {
				t.Fatalf("FP32 embed failed: %v", err)
			}

			int8Emb, err := int8Eng.Embed(tc.Text)
			if err != nil {
				t.Fatalf("INT8 embed failed: %v", err)
			}

			sim := engine.CosineSimilarity(fp32Emb, int8Emb)
			simGolden := engine.CosineSimilarity(int8Emb, tc.Embedding)

			t.Logf("Case #%d: Sim(INT8, FP32)=%.6f, Sim(INT8, Golden)=%.6f", i, sim, simGolden)

			if sim < 0.990 {
				t.Errorf("INT8 vs FP32 similarity too low: %.6f (expected >= 0.990)", sim)
			}
			if simGolden < 0.990 {
				t.Errorf("INT8 vs Golden similarity too low: %.6f (expected >= 0.990)", simGolden)
			}
		})
	}
}

func TestQuantizedZeroAllocations(t *testing.T) {
	modelPath := filepath.Join("..", "..", "models", "intfloat", "multilingual-e5-small", "model.safetensors")
	tokPath := filepath.Join("..", "..", "models", "intfloat", "multilingual-e5-small", "tokenizer.json")

	eng, err := engine.NewQuantized(modelPath, tokPath)
	if err != nil {
		t.Fatalf("Failed to load INT8 engine: %v", err)
	}

	ctx := engine.NewContext(eng.Model())
	out := make([]float32, engine.HiddenSize)
	query := "query: how to implement consensus in distributed systems?"

	// Warm up
	if _, err := ctx.Embed(query, out); err != nil {
		t.Fatalf("Warmup failed: %v", err)
	}

	allocs := testing.AllocsPerRun(100, func() {
		if _, err := ctx.Embed(query, out); err != nil {
			t.Fatalf("Embed error: %v", err)
		}
	})

	t.Logf("INT8 Steady-state allocations per run: %.1f", allocs)
	if allocs != 0 {
		t.Errorf("Expected 0 allocations per run in INT8, got %.1f", allocs)
	}
}
