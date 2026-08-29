package tokenizer_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"go-embed/pkg/tokenizer"
)

type GoldenEntry struct {
	Text          string    `json:"text"`
	InputIDs      []int     `json:"input_ids"`
	AttentionMask []int8    `json:"attention_mask"`
	Tokens        []string  `json:"tokens"`
	SeqLen        int       `json:"seq_len"`
	Embedding     []float32 `json:"embedding"`
}

func TestTokenizerParity(t *testing.T) {
	tokPath := filepath.Join("..", "..", "models", "intfloat", "multilingual-e5-small", "tokenizer.json")
	if _, err := os.Stat(tokPath); os.IsNotExist(err) {
		t.Skip("tokenizer.json not found")
	}

	tok, err := tokenizer.LoadFromFile(tokPath)
	if err != nil {
		t.Fatalf("Failed to load tokenizer: %v", err)
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

	for i, entry := range entries {
		ids, mask := tok.Encode(entry.Text, 512)

		if !reflect.DeepEqual(ids, entry.InputIDs) {
			t.Errorf("Case #%d (%q) input_ids mismatch:\n  got:  %v\n  want: %v", i, entry.Text, ids, entry.InputIDs)
		}
		if !reflect.DeepEqual(mask, entry.AttentionMask) {
			t.Errorf("Case #%d (%q) attention_mask mismatch:\n  got:  %v\n  want: %v", i, entry.Text, mask, entry.AttentionMask)
		}
	}
}
