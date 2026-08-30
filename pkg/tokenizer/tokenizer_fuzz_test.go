package tokenizer_test

import (
	"flag"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/C-Pro/go-embed/pkg/tokenizer"
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

var (
	sharedTokOnce sync.Once
	sharedTok     *tokenizer.Tokenizer
)

func getSharedTokenizer(t testing.TB) *tokenizer.Tokenizer {
	sharedTokOnce.Do(func() {
		tokPath := filepath.Join("..", "..", "models", "intfloat", "multilingual-e5-small", "tokenizer.json")
		var err error
		sharedTok, err = tokenizer.LoadFromFile(tokPath)
		if err != nil {
			sharedTok = nil
		}
	})
	if sharedTok == nil {
		t.Skip("shared tokenizer not available")
	}
	return sharedTok
}

func FuzzTokenizerLoad(f *testing.F) {
	skipFuzzInCI(f)

	// Seed with empty, minimal valid and invalid json
	f.Add([]byte{})
	f.Add([]byte(`{}`))
	f.Add([]byte(`{"model":{"type":"Unigram","vocab":[["<s>",0.0],["</s>",0.0],["<unk>",0.0]]}}`))
	f.Add([]byte(`{"model":{"type":"Unigram","vocab":[["a",-1.0],["b",-2.0]]},"added_tokens":[{"id":10,"content":"[CLS]"}]}`))

	f.Fuzz(func(t *testing.T, data []byte) {
		tok, err := tokenizer.LoadFromBytes(data)
		if err != nil {
			return
		}
		_ = tok.VocabSize()
		_ = tok.Normalize("sample text")
		_, _ = tok.Encode("sample text", 128)
		_ = tok.Decode([]int{0, 1, 2, 3, 4, 100})
	})
}

func FuzzTokenizerEncode(f *testing.F) {
	skipFuzzInCI(f)

	f.Add("query: how to implement consensus in distributed systems?", 512)
	f.Add("passage: Authentischer italienischer Tiramisu", 128)
	f.Add("Привет мир! 12345 🚀", 64)
	f.Add("", 0)
	f.Add("   \t\n   ", 1)
	f.Add("a", -5)
	f.Add("？？！！：：；；，，（（））【【】】“”‘’", 512)
	f.Add("\x00\x01\x02\xff\xfe\xfd", 256)

	f.Fuzz(func(t *testing.T, text string, maxLen int) {
		tok := getSharedTokenizer(t)

		_ = tok.Normalize(text)

		raw := tok.EncodeRaw(text)
		_ = tok.Decode(raw)

		ids, mask := tok.Encode(text, maxLen)
		if len(ids) != len(mask) {
			t.Fatalf("ids len (%d) != mask len (%d)", len(ids), len(mask))
		}
		_ = tok.Decode(ids)

		qIDs, qMask := tok.EncodeQuery(text, maxLen)
		if len(qIDs) != len(qMask) {
			t.Fatalf("qIDs len (%d) != qMask len (%d)", len(qIDs), len(qMask))
		}

		pIDs, pMask := tok.EncodePassage(text, maxLen)
		if len(pIDs) != len(pMask) {
			t.Fatalf("pIDs len (%d) != pMask len (%d)", len(pIDs), len(pMask))
		}

		// Test zero-alloc variant with reusable buffers
		runeBuf := make([]rune, 0, 1024)
		inputIDs := make([]int, 0, 1024)
		attnMask := make([]int8, 0, 1024)
		dpBuf := make([]tokenizer.DPState, 0, 2048)

		_, inputIDs, attnMask = tok.EncodeIntoZeroAlloc(text, runeBuf, inputIDs, attnMask, dpBuf, maxLen)
		if len(inputIDs) != len(attnMask) {
			t.Fatalf("zero-alloc ids len (%d) != mask len (%d)", len(inputIDs), len(attnMask))
		}
	})
}
