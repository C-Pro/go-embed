package safetensors_test

import (
	"os"
	"path/filepath"
	"testing"

	"go-embed/pkg/safetensors"
)

func TestSafetensorsOpenAndRead(t *testing.T) {
	modelPath := filepath.Join("..", "..", "models", "intfloat", "multilingual-e5-small", "model.safetensors")
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("model file does not exist")
	}

	st, err := safetensors.Open(modelPath)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer st.Close()

	if len(st.Tensors) == 0 {
		t.Fatalf("Expected tensors in safetensors, got 0")
	}

	t.Logf("Found %d tensors in %s", len(st.Tensors), modelPath)

	// Check embeddings
	emb, err := st.ReadTensorF32("embeddings.word_embeddings.weight")
	if err != nil {
		t.Fatalf("ReadTensorF32 word_embeddings failed: %v", err)
	}
	expectedLen := 250037 * 384
	if len(emb) != expectedLen {
		t.Fatalf("Expected word_embeddings length %d, got %d", expectedLen, len(emb))
	}
	t.Logf("word_embeddings first 5 values: %v", emb[:5])
}
