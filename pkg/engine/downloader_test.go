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
	if err != nil || len(emb) == 0 || len(emb[0]) != engine.HiddenSize {
		t.Fatalf("EmbedQuery failed: err=%v, embs=%d", err, len(emb))
	}

	// 2. NewEngine with WithBF16
	if isCI() {
		return // Skip redundant full re-loads in CI
	}

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

func TestPrefixDetectionAndOptions(t *testing.T) {
	// 1. Automatic detection on e5 model
	qPreE5, pPreE5 := engine.DetectModelPrefixes("", "intfloat/multilingual-e5-small", true)
	if qPreE5 != "query: " || pPreE5 != "passage: " {
		t.Errorf("Expected e5 prefixes ('query: ', 'passage: '), got (%q, %q)", qPreE5, pPreE5)
	}

	// 2. Automatic detection on MiniLM / sentence-transformers model
	qPreMiniLM, pPreMiniLM := engine.DetectModelPrefixes("", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", true)
	if qPreMiniLM != "" || pPreMiniLM != "" {
		t.Errorf("Expected empty prefixes for MiniLM, got (%q, %q)", qPreMiniLM, pPreMiniLM)
	}

	// 3. Automatic detection on BGE model
	qPreBGE, pPreBGE := engine.DetectModelPrefixes("", "BAAI/bge-small-en-v1.5", true)
	if qPreBGE != "Represent this sentence for searching relevant passages: " || pPreBGE != "" {
		t.Errorf("Unexpected BGE prefixes: (%q, %q)", qPreBGE, pPreBGE)
	}

	// 4. Test engine with WithPrefixes option
	dataDir := filepath.Join("..", "..", "models", "intfloat", "multilingual-e5-small")
	engCustom, err := engine.NewEngine(
		engine.WithDataDir(dataDir),
		engine.WithPrefixes("custom_q: ", "custom_p: "),
		engine.WithSilentDownload(true),
	)
	if err != nil {
		t.Fatalf("Failed to initialize engine with WithPrefixes: %v", err)
	}
	if engCustom.QueryPrefix() != "custom_q: " || engCustom.PassagePrefix() != "custom_p: " {
		t.Errorf("Expected custom prefixes, got (%q, %q)", engCustom.QueryPrefix(), engCustom.PassagePrefix())
	}

	// 5. Test engine with WithNoPrefixes option
	engNoPrefix, err := engine.NewEngine(
		engine.WithDataDir(dataDir),
		engine.WithNoPrefixes(),
		engine.WithSilentDownload(true),
	)
	if err != nil {
		t.Fatalf("Failed to initialize engine with WithNoPrefixes: %v", err)
	}
	if engNoPrefix.QueryPrefix() != "" || engNoPrefix.PassagePrefix() != "" {
		t.Errorf("Expected empty prefixes with WithNoPrefixes, got (%q, %q)", engNoPrefix.QueryPrefix(), engNoPrefix.PassagePrefix())
	}
}
