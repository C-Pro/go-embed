package engine_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/C-Pro/go-embed/pkg/engine"
)

func FuzzModelLoad(f *testing.F) {
	skipFuzzInCI(f)

	// Seed with empty, truncated, or random bytes for model and tokenizer
	f.Add([]byte{}, []byte{})
	f.Add([]byte("corrupted model data"), []byte(`{}`))
	f.Add([]byte{0, 0, 0, 0, 0, 0, 0, 0}, []byte(`{"model":{"type":"Unigram","vocab":[]}}`))

	f.Fuzz(func(t *testing.T, modelData, tokData []byte) {
		tmpDir := t.TempDir()
		modelPath := filepath.Join(tmpDir, "model.safetensors")
		tokPath := filepath.Join(tmpDir, "tokenizer.json")

		if err := os.WriteFile(modelPath, modelData, 0644); err != nil {
			return
		}
		if err := os.WriteFile(tokPath, tokData, 0644); err != nil {
			return
		}

		// Test loading across precisions
		for _, prec := range []engine.PrecisionMode{engine.PrecisionFP32, engine.PrecisionBF16, engine.PrecisionINT8} {
			m, err := engine.LoadModelWithPrecision(modelPath, tokPath, prec)
			if err == nil && m != nil {
				_ = m.VocabSize()
				_ = m.Validate()
				_ = m.Close()
			}
		}
	})
}
