package engine_test

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/C-Pro/go-embed/pkg/engine"
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

var (
	sharedFP32Once sync.Once
	sharedFP32Eng  *engine.Engine
	sharedFP32Err  error

	sharedBF16Once sync.Once
	sharedBF16Eng  *engine.Engine

	sharedINT8Once sync.Once
	sharedINT8Eng  *engine.Engine
)

func loadTestModel(t *testing.T) *engine.Engine {
	t.Helper()
	sharedFP32Once.Do(func() {
		modelPath := filepath.Join("..", "..", "models", "intfloat", "multilingual-e5-small", "model.safetensors")
		tokPath := filepath.Join("..", "..", "models", "intfloat", "multilingual-e5-small", "tokenizer.json")

		if _, err := os.Stat(modelPath); os.IsNotExist(err) {
			sharedFP32Err = err
			return
		}

		sharedFP32Eng, sharedFP32Err = engine.New(modelPath, tokPath)
	})

	if sharedFP32Err != nil {
		t.Skipf("model not available: %v", sharedFP32Err)
	}
	return sharedFP32Eng
}

func loadTestBF16Model(t *testing.T) *engine.Engine {
	t.Helper()
	loadTestModel(t)
	sharedBF16Once.Do(func() {
		modelPath := filepath.Join("..", "..", "models", "intfloat", "multilingual-e5-small", "model.safetensors")
		tokPath := filepath.Join("..", "..", "models", "intfloat", "multilingual-e5-small", "tokenizer.json")
		var err error
		sharedBF16Eng, err = engine.NewBF16(modelPath, tokPath)
		if err != nil {
			sharedFP32Err = err
		}
	})
	if sharedBF16Eng == nil {
		t.Skipf("BF16 model not available: %v", sharedFP32Err)
	}
	return sharedBF16Eng
}

func loadTestINT8Model(t *testing.T) *engine.Engine {
	t.Helper()
	loadTestModel(t)
	sharedINT8Once.Do(func() {
		modelPath := filepath.Join("..", "..", "models", "intfloat", "multilingual-e5-small", "model.safetensors")
		tokPath := filepath.Join("..", "..", "models", "intfloat", "multilingual-e5-small", "tokenizer.json")
		var err error
		sharedINT8Eng, err = engine.NewQuantized(modelPath, tokPath)
		if err != nil {
			sharedFP32Err = err
		}
	})
	if sharedINT8Eng == nil {
		t.Skipf("INT8 model not available: %v", sharedFP32Err)
	}
	return sharedINT8Eng
}

func isCI() bool {
	return os.Getenv("CI") != "" || testing.Short()
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

	// In CI, skip long 512-token sequences and test a fast representative subset
	if isCI() {
		var fastEntries []GoldenEntry
		for _, e := range entries {
			if e.SeqLen < 64 {
				fastEntries = append(fastEntries, e)
			}
			if len(fastEntries) >= 2 {
				break
			}
		}
		return fastEntries
	}
	return entries
}

func TestGoldenParity(t *testing.T) {
	eng := loadTestModel(t)
	entries := loadGolden(t)
	ctx := engine.NewContext(eng.Model())

	for i, entry := range entries {
		t.Run(entry.Text, func(t *testing.T) {
			embs, err := ctx.Embed(entry.Text)
			if err != nil {
				t.Fatalf("Case #%d failed: %v", i, err)
			}
			if len(embs) == 0 {
				t.Fatalf("Case #%d returned 0 embeddings", i)
			}
			out := embs[0]

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
		embsScalar, err := ctxScalar.Embed(entry.Text)
		if err != nil {
			t.Fatalf("Scalar embed failed: %v", err)
		}
		embsSIMD, err := ctxSIMD.Embed(entry.Text)
		if err != nil {
			t.Fatalf("SIMD embed failed: %v", err)
		}

		outScalar := embsScalar[0]
		outSIMD := embsSIMD[0]

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

func TestSlidingWindowChunking(t *testing.T) {
	eng := loadTestModel(t)

	longParagraph := "Consensus algorithms such as Raft and Paxos are fundamental protocols in distributed systems engineering. " +
		"They allow a collection of machines to work as a coherent group that can survive the failures of some of its members. " +
		"In a typical deployment, nodes elect a leader responsible for managing log replication across followers. "
	var longTextBuilder strings.Builder
	for i := 0; i < 4; i++ {
		longTextBuilder.WriteString(longParagraph)
	}
	longText := "passage: " + longTextBuilder.String()

	// Test 1: Explicit sliding window with WindowSize=64, Overlap=32
	ctxSmallWindow := engine.NewContextWithOptions(eng.Model(), 64, 32, "", "")
	embsSmall, err := ctxSmallWindow.Embed(longText)
	if err != nil {
		t.Fatalf("Embed on small window failed: %v", err)
	}
	if len(embsSmall) < 2 {
		t.Fatalf("Expected multiple window chunks with WindowSize=64, got %d", len(embsSmall))
	}
	t.Logf("Small window (64/32) generated %d chunks", len(embsSmall))

	for i, emb := range embsSmall {
		if len(emb) != engine.HiddenSize {
			t.Errorf("Chunk %d has dimension %d (expected %d)", i, len(emb), engine.HiddenSize)
		}
		var normSq float32
		for _, v := range emb {
			normSq += v * v
		}
		norm := float32(math.Sqrt(float64(normSq)))
		if math.Abs(float64(norm-1.0)) > 1e-4 {
			t.Errorf("Chunk %d norm is %.5f (expected 1.0)", i, norm)
		}
	}

	// Test 2: Full 512-token sliding window if not CI
	if !isCI() {
		var hugeBuilder strings.Builder
		for i := 0; i < 20; i++ {
			hugeBuilder.WriteString(longParagraph)
		}
		hugeText := "passage: " + hugeBuilder.String()
		embsDefault, err := eng.Embed(hugeText)
		if err != nil {
			t.Fatalf("Embed on huge text failed: %v", err)
		}
		if len(embsDefault) < 2 {
			t.Fatalf("Expected multiple window chunks for >512 tokens text, got %d", len(embsDefault))
		}
		t.Logf("Default window (512/256) generated %d chunks", len(embsDefault))
	}
}
