package engine_test

import (
	"path/filepath"
	"testing"

	"go-embed/pkg/engine"
)

func TestNewEngineFunctionalOptions(t *testing.T) {
	dataDir := filepath.Join("..", "..", "models", "intfloat", "multilingual-e5-small")

	// 1. NewEngine with WithDataDir
	eng, err := engine.NewEngine(
		engine.WithDataDir(dataDir),
		engine.WithSilentDownload(true),
	)
	if err != nil {
		t.Fatalf("Failed to initialize engine with WithDataDir: %v", err)
	}

	if eng.Precision() != engine.PrecisionFP32 {
		t.Errorf("Expected FP32 precision by default, got %v", eng.Precision())
	}

	emb, err := eng.EmbedQuery("test query")
	if err != nil || len(emb) != engine.HiddenSize {
		t.Fatalf("EmbedQuery failed: %v", err)
	}

	// 2. NewEngine with WithBF16
	bf16Eng, err := engine.NewEngine(
		engine.WithDataDir(dataDir),
		engine.WithBF16(),
		engine.WithSilentDownload(true),
	)
	if err != nil {
		t.Fatalf("Failed to initialize engine with WithBF16: %v", err)
	}

	if bf16Eng.Precision() != engine.PrecisionBF16 {
		t.Errorf("Expected BF16 precision, got %v", bf16Eng.Precision())
	}

	// 3. NewEngine with WithINT8
	int8Eng, err := engine.NewEngine(
		engine.WithDataDir(dataDir),
		engine.WithINT8(),
		engine.WithSilentDownload(true),
	)
	if err != nil {
		t.Fatalf("Failed to initialize engine with WithINT8: %v", err)
	}

	if int8Eng.Precision() != engine.PrecisionINT8 {
		t.Errorf("Expected INT8 precision, got %v", int8Eng.Precision())
	}
}
