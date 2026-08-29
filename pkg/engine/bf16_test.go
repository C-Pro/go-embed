package engine_test

import (
	"math"
	"testing"

	"go-embed/pkg/engine"
)

func TestBFloat16Conversion(t *testing.T) {
	testFloats := []float32{
		0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 3.14159265, -2.7182818,
		1e-4, -1e-4, 1e4, -1e4, 0.00390625,
	}

	for _, f := range testFloats {
		bf := engine.Float32ToBFloat16(f)
		reconstructed := engine.BFloat16ToFloat32(bf)

		if f == 0.0 {
			if reconstructed != 0.0 {
				t.Errorf("Zero conversion failed: got %v", reconstructed)
			}
			continue
		}

		relErr := math.Abs(float64(f-reconstructed)) / math.Abs(float64(f))
		// BF16 has 7 mantissa bits, precision is ~ 1/256 ≈ 0.0039 (0.39%)
		if relErr > 0.005 {
			t.Errorf("Relative error too high for %v: got %v (relErr=%v)", f, reconstructed, relErr)
		}
	}
}

func TestBF16ParityAgainstFP32(t *testing.T) {
	fp32Eng := loadTestModel(t)
	bf16Eng := loadTestBF16Model(t)

	if bf16Eng.Precision() != engine.PrecisionBF16 {
		t.Fatalf("Expected BF16 precision, got %v", bf16Eng.Precision())
	}

	golden := loadGolden(t)

	for i, tc := range golden {
		t.Run(tc.Text, func(t *testing.T) {
			fp32Emb, err := fp32Eng.Embed(tc.Text)
			if err != nil {
				t.Fatalf("FP32 embed failed: %v", err)
			}

			bf16Emb, err := bf16Eng.Embed(tc.Text)
			if err != nil {
				t.Fatalf("BF16 embed failed: %v", err)
			}

			sim := engine.CosineSimilarity(fp32Emb, bf16Emb)
			simGolden := engine.CosineSimilarity(bf16Emb, tc.Embedding)

			t.Logf("Case #%d: Sim(BF16, FP32)=%.6f, Sim(BF16, Golden)=%.6f", i, sim, simGolden)

			// BF16 must achieve >= 0.9995 cosine similarity with full precision
			if sim < 0.9995 {
				t.Errorf("BF16 vs FP32 similarity too low: %.6f (expected >= 0.9995)", sim)
			}
			if simGolden < 0.9995 {
				t.Errorf("BF16 vs Golden similarity too low: %.6f (expected >= 0.9995)", simGolden)
			}
		})
	}
}

func TestBF16ZeroAllocations(t *testing.T) {
	eng := loadTestBF16Model(t)

	ctx := engine.NewContext(eng.Model())
	out := make([]float32, engine.HiddenSize)
	query := "query: how to implement consensus in distributed systems?"

	// Warm up
	if _, err := ctx.Embed(query, out); err != nil {
		t.Fatalf("Warmup failed: %v", err)
	}

	allocIters := 100
	if isCI() {
		allocIters = 2
	}

	allocs := testing.AllocsPerRun(allocIters, func() {
		if _, err := ctx.Embed(query, out); err != nil {
			t.Fatalf("Embed error: %v", err)
		}
	})

	t.Logf("BF16 Steady-state allocations per run: %.1f", allocs)
	if allocs != 0 {
		t.Errorf("Expected 0 allocations per run in BF16, got %.1f", allocs)
	}
}
