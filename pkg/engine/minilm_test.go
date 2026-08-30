package engine_test

import (
	"path/filepath"
	"testing"

	"github.com/C-Pro/go-embed/pkg/engine"
)

func TestParaphraseMultilingualMiniLM(t *testing.T) {
	if isCI() {
		t.Skip("Skipping secondary model download and test in CI environment")
	}

	modelName := "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
	dataDir := filepath.Join("..", "..", "models", modelName)

	t.Logf("Testing model: %s", modelName)

	eng, err := engine.NewEngine(
		engine.WithModelName(modelName),
		engine.WithDataDir(dataDir),
		engine.WithPrecision(engine.PrecisionFP32),
	)
	if err != nil {
		t.Fatalf("Failed to initialize engine for %s: %v", modelName, err)
	}

	t.Logf("Model loaded successfully! Vocab Size: %d, Hidden Size: %d, Layers: %d",
		eng.Model().VocabSize(), engine.HiddenSize, engine.NumLayers)

	// Paraphrase pairs across languages (paraphrase models do not need "query: " or "passage: " prefixes)
	testPairs := []struct {
		desc   string
		text1  string
		text2  string
		isPair bool
	}{
		{
			desc:   "English Paraphrase",
			text1:  "How do you implement consensus in a distributed system?",
			text2:  "What is the way to achieve consensus across distributed nodes?",
			isPair: true,
		},
		{
			desc:   "Cross-lingual English to German Paraphrase",
			text1:  "The cat is sleeping peacefully on the sofa.",
			text2:  "Die Katze schläft friedlich auf dem Sofa.",
			isPair: true,
		},
		{
			desc:   "Cross-lingual English to Russian Paraphrase",
			text1:  "Machine learning is a subset of artificial intelligence.",
			text2:  "Машинное обучение — это подраздел искусственного интеллекта.",
			isPair: true,
		},
		{
			desc:   "Cross-lingual English to Chinese Paraphrase",
			text1:  "Artificial intelligence is transforming software engineering.",
			text2:  "人工智能正在改变软件工程。",
			isPair: true,
		},
		{
			desc:   "Semantic Dissimilarity Test",
			text1:  "How to implement consensus in a distributed system?",
			text2:  "Authentic Italian recipe for tiramisu dessert with mascarpone.",
			isPair: false,
		},
	}

	for _, tc := range testPairs {
		t.Run(tc.desc, func(t *testing.T) {
			emb1, err := eng.Embed(tc.text1)
			if err != nil {
				t.Fatalf("Embed text1 failed: %v", err)
			}
			emb2, err := eng.Embed(tc.text2)
			if err != nil {
				t.Fatalf("Embed text2 failed: %v", err)
			}

			sim := engine.CosineSimilarity(emb1[0], emb2[0])
			t.Logf("[%s] Similarity: %.4f\n  T1: %q\n  T2: %q", tc.desc, sim, tc.text1, tc.text2)

			if tc.isPair {
				if sim < 0.75 {
					t.Errorf("Expected high similarity (>= 0.75) for paraphrase pair, got %.4f", sim)
				}
			} else {
				if sim > 0.50 {
					t.Errorf("Expected low similarity (<= 0.50) for unrelated pair, got %.4f", sim)
				}
			}
		})
	}
}

func TestParaphraseMultilingualMiniLM_BF16(t *testing.T) {
	if isCI() {
		t.Skip("Skipping secondary model download and test in CI environment")
	}

	modelName := "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
	dataDir := filepath.Join("..", "..", "models", modelName)

	bf16Eng, err := engine.NewEngine(
		engine.WithModelName(modelName),
		engine.WithDataDir(dataDir),
		engine.WithBF16(),
	)
	if err != nil {
		t.Fatalf("Failed to initialize BF16 engine: %v", err)
	}

	text1 := "How do you implement consensus in a distributed system?"
	text2 := "What is the way to achieve consensus across distributed nodes?"

	emb1, err := bf16Eng.Embed(text1)
	if err != nil {
		t.Fatalf("Embed failed: %v", err)
	}
	emb2, err := bf16Eng.Embed(text2)
	if err != nil {
		t.Fatalf("Embed failed: %v", err)
	}

	sim := engine.CosineSimilarity(emb1[0], emb2[0])
	t.Logf("BF16 Paraphrase Similarity: %.4f", sim)
	if sim < 0.75 {
		t.Errorf("Expected high similarity in BF16, got %.4f", sim)
	}
}
