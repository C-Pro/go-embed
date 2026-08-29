package spagoref_test

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"go-embed/pkg/spagoref"
)

type GoldenEntry struct {
	Text      string    `json:"text"`
	InputIDs  []int     `json:"input_ids"`
	Embedding []float32 `json:"embedding"`
}

func cosineSimilarity(a, b []float32) float32 {
	var dot, nA, nB float32
	for i := range a {
		dot += a[i] * b[i]
		nA += a[i] * a[i]
		nB += b[i] * b[i]
	}
	if nA == 0 || nB == 0 {
		return 0
	}
	return dot / (float32(math.Sqrt(float64(nA))) * float32(math.Sqrt(float64(nB))))
}

func TestSpagoReferenceGolden(t *testing.T) {
	modelPath := filepath.Join("..", "..", "models", "intfloat", "multilingual-e5-small", "model.safetensors")
	tokPath := filepath.Join("..", "..", "models", "intfloat", "multilingual-e5-small", "tokenizer.json")
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("model not found")
	}

	model, err := spagoref.LoadModel(modelPath, tokPath)
	if err != nil {
		t.Fatalf("Failed to load spago model: %v", err)
	}

	goldenPath := filepath.Join("..", "..", "testdata", "golden.json")
	data, err := os.ReadFile(goldenPath)
	if err != nil {
		t.Fatalf("Failed to read golden.json: %v", err)
	}

	var entries []GoldenEntry
	if err := json.Unmarshal(data, &entries); err != nil {
		t.Fatalf("Failed to parse golden.json: %v", err)
	}

	// Test first few entries for quick verification
	testCases := entries
	if len(testCases) > 3 {
		testCases = testCases[:3]
	}

	for i, entry := range testCases {
		vec, err := model.EncodeText(entry.Text)
		if err != nil {
			t.Fatalf("EncodeText failed for case %d: %v", i, err)
		}

		cosSim := cosineSimilarity(vec, entry.Embedding)
		t.Logf("Case #%d: text=%q, Cosine Similarity with Golden=%0.6f", i, entry.Text, cosSim)

		if cosSim < 0.999 {
			t.Errorf("Case #%d Cosine similarity too low: %f (expected >= 0.999)", i, cosSim)
		}
	}
}
