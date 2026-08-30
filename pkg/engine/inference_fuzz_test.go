package engine_test

import (
	"flag"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/C-Pro/go-embed/pkg/engine"
)

func skipFuzzInCI(f *testing.F) {
	if (os.Getenv("CI") != "" || os.Getenv("GITHUB_ACTIONS") != "") && isFuzzing() {
		f.Skip("Skipping continuous fuzzing runs in CI environment")
	}
}

func isFuzzing() bool {
	fl := flag.Lookup("test.fuzz")
	return fl != nil && fl.Value.String() != ""
}

func FuzzInferenceEmbed(f *testing.F) {
	skipFuzzInCI(f)

	f.Add("query: how to implement consensus in distributed systems?")
	f.Add("passage: Authentischer italienischer Tiramisu")
	f.Add("Привет мир! 12345 🚀")
	f.Add("")
	f.Add("   \t\n   ")
	f.Add("a")
	f.Add("？？！！：：；；，，（（））【【】】“”‘’")
	f.Add("\x00\x01\x02\xff\xfe\xfd")

	f.Fuzz(func(t *testing.T, text string) {
		if (os.Getenv("CI") != "" || os.Getenv("GITHUB_ACTIONS") != "") && !strings.Contains(t.Name(), "/seed#") {
			t.Skip("skipping continuous fuzz generation in CI")
		}
		engFP32 := loadTestModel(t)
		if engFP32 == nil {
			return
		}

		var engines []*engine.Engine
		if isCI() {
			engines = []*engine.Engine{engFP32}
		} else {
			engBF16 := loadTestBF16Model(t)
			engINT8 := loadTestINT8Model(t)
			engines = []*engine.Engine{engFP32, engBF16, engINT8}
		}

		for _, eng := range engines {
			if eng == nil {
				continue
			}
			embs, err := eng.Embed(text)
			if err == nil {
				for _, vec := range embs {
					if len(vec) != engine.HiddenSize {
						t.Fatalf("expected vector dim %d, got %d", engine.HiddenSize, len(vec))
					}
				}
			}

			qEmbs, err := eng.EmbedQuery(text)
			if err == nil {
				for _, vec := range qEmbs {
					if len(vec) != engine.HiddenSize {
						t.Fatalf("expected vector dim %d, got %d", engine.HiddenSize, len(vec))
					}
				}
			}

			pEmbs, err := eng.EmbedPassage(text)
			if err == nil {
				for _, vec := range pEmbs {
					if len(vec) != engine.HiddenSize {
						t.Fatalf("expected vector dim %d, got %d", engine.HiddenSize, len(vec))
					}
				}
			}

			batch := []string{text, text + " suffix"}
			bEmbs, err := eng.EmbedBatch(batch)
			if err == nil && len(bEmbs) != len(batch) {
				t.Fatalf("expected %d batch embeddings, got %d", len(batch), len(bEmbs))
			}
		}
	})
}

func FuzzInferenceEmbedTokenIDs(f *testing.F) {
	skipFuzzInCI(f)

	// Seed with valid and edge case token ID sequences
	f.Add([]byte{0, 0, 0, 0, 2, 0, 0, 0}, []byte{1, 1})
	f.Add([]byte{0, 0, 0, 0, 100, 0, 0, 0, 2, 0, 0, 0}, []byte{1, 1, 1})
	f.Add([]byte{}, []byte{})
	f.Add([]byte{0xff, 0xff, 0xff, 0xff}, []byte{0})

	f.Fuzz(func(t *testing.T, idBytes, maskBytes []byte) {
		if len(idBytes) > 2048 {
			idBytes = idBytes[:2048]
		}
		if len(maskBytes) > 512 {
			maskBytes = maskBytes[:512]
		}
		// Convert idBytes to []int
		nIDs := len(idBytes) / 4
		ids := make([]int, nIDs)
		for i := 0; i < nIDs; i++ {
			ids[i] = int(int32(uint32(idBytes[i*4]) | uint32(idBytes[i*4+1])<<8 | uint32(idBytes[i*4+2])<<16 | uint32(idBytes[i*4+3])<<24))
		}

		mask := make([]int8, len(maskBytes))
		for i, b := range maskBytes {
			mask[i] = int8(b)
		}

		eng := loadTestModel(t)
		if eng == nil {
			return
		}
		ctx := engine.NewContext(eng.Model())

		_, _ = ctx.EmbedTokenIDs(ids, mask)
	})
}

func FuzzInferenceSimilarity(f *testing.F) {
	skipFuzzInCI(f)

	f.Add("query: consensus in distributed systems", "passage: Paxos and Raft are consensus protocols")
	f.Add("hello world", "")
	f.Add("", "")
	f.Add("Cat on couch", "K8s deployment")

	f.Fuzz(func(t *testing.T, textA, textB string) {
		eng := loadTestModel(t)
		if eng == nil {
			return
		}
		sim, err := eng.Similarity(textA, textB)
		if err != nil {
			return
		}
		if math.IsNaN(float64(sim)) || math.IsInf(float64(sim), 0) {
			t.Fatalf("Similarity returned non-finite value: %v", sim)
		}
		if sim < -1.0001 || sim > 1.0001 {
			t.Fatalf("Similarity out of [-1, 1] range: %v", sim)
		}
	})
}

func FuzzLowLevelOps(f *testing.F) {
	skipFuzzInCI(f)

	f.Add(float32(1.0), float32(2.0), float32(0.5), int(16), int(16))
	f.Add(float32(0.0), float32(-1.0), float32(100.0), int(384), int(384))
	f.Add(float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1)), int(8), int(8))

	f.Fuzz(func(t *testing.T, v1, v2, v3 float32, inDim, outDim int) {
		if inDim < 0 || inDim > 1024 || outDim < 0 || outDim > 1024 {
			return
		}

		x := make([]float32, inDim)
		for i := range x {
			x[i] = v1
		}
		weightF32 := make([]float32, inDim*outDim)
		for i := range weightF32 {
			weightF32[i] = v2
		}
		weightI8 := make([]int8, inDim*outDim)
		weightBF16 := make([]uint16, inDim*outDim)
		for i := range weightBF16 {
			weightBF16[i] = engine.Float32ToBFloat16(v2)
		}
		scale := make([]float32, outDim)
		bias := make([]float32, outDim)
		for i := range bias {
			scale[i] = v3
			bias[i] = v3
		}
		out := make([]float32, outDim)

		// Test scalar ops
		engine.MatVecMulAddScalar(x, weightF32, bias, out, inDim, outDim)
		engine.MatVecMulAddINT8Scalar(x, weightI8, scale, bias, out, inDim, outDim)
		engine.MatVecMulAddBF16Scalar(x, weightBF16, bias, out, inDim, outDim)
		engine.LayerNormScalar(x, x, bias, out, inDim, 1e-12)
		engine.GELUScalar(x, out, inDim)
		engine.GELUApproxScalar(x, out, inDim)
		engine.SoftmaxScalar(x, out, inDim)
		engine.L2NormalizeScalar(x, out, inDim)
		_ = engine.CosineSimilarityScalar(x, x, inDim)

		// Test SIMD ops
		engine.MatVecMulAddSIMD(x, weightF32, bias, out, inDim, outDim)
		engine.MatVecMulAddINT8SIMD(x, weightI8, scale, bias, out, inDim, outDim)
		engine.MatVecMulAddBF16SIMD(x, weightBF16, bias, out, inDim, outDim)
		engine.LayerNormSIMD(x, x, bias, out, inDim, 1e-12)
		engine.GELUSIMD(x, out, inDim)
		engine.GELUApproxSIMD(x, out, inDim)
		engine.L2NormalizeSIMD(x, out, inDim)
		_ = engine.CosineSimilaritySIMD(x, x, inDim)

		// MeanPool
		mask := make([]int8, inDim)
		hiddenStates := make([]float32, inDim*outDim)
		pooled := make([]float32, outDim)
		engine.MeanPoolScalar(hiddenStates, mask, pooled, inDim, outDim)
	})
}
