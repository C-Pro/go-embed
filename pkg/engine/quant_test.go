package engine_test

import (
	"math"
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
	fp32Eng := loadTestModel(t)
	int8Eng := loadTestINT8Model(t)

	if !int8Eng.IsQuantized() {
		t.Fatal("Expected INT8 engine to be quantized")
	}

	golden := loadGolden(t)

	for i, tc := range golden {
		t.Run(tc.Text, func(t *testing.T) {
			fp32Embs, err := fp32Eng.Embed(tc.Text)
			if err != nil {
				t.Fatalf("FP32 embed failed: %v", err)
			}

			int8Embs, err := int8Eng.Embed(tc.Text)
			if err != nil {
				t.Fatalf("INT8 embed failed: %v", err)
			}

			sim := engine.CosineSimilarity(fp32Embs[0], int8Embs[0])
			simGolden := engine.CosineSimilarity(int8Embs[0], tc.Embedding)

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
