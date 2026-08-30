package embed_test

import (
	"path/filepath"
	"testing"

	"github.com/C-Pro/go-embed"
)

func TestTopLevelEngineAPI(t *testing.T) {
	dataDir := filepath.Join("models", "intfloat", "multilingual-e5-small")

	// 1. Initialize Engine using top-level NewEngine and Option wrappers
	eng, err := embed.NewEngine(
		embed.WithDataDir(dataDir),
		embed.WithPrefixes("query: ", "passage: "),
		embed.WithChunking(512, 256),
	)
	if err != nil {
		t.Fatalf("embed.NewEngine failed: %v", err)
	}
	defer eng.Close()

	if eng.Precision() != embed.PrecisionFP32 {
		t.Fatalf("expected precision %v, got %v", embed.PrecisionFP32, eng.Precision())
	}

	if eng.QueryPrefix() != "query: " {
		t.Fatalf("expected query prefix 'query: ', got %q", eng.QueryPrefix())
	}
	if eng.PassagePrefix() != "passage: " {
		t.Fatalf("expected passage prefix 'passage: ', got %q", eng.PassagePrefix())
	}

	// 2. Test EmbedQuery and EmbedPassage
	query := "how to implement consensus in distributed systems?"
	qEmbs, err := eng.EmbedQuery(query)
	if err != nil {
		t.Fatalf("EmbedQuery failed: %v", err)
	}
	if len(qEmbs) == 0 || len(qEmbs[0]) != embed.HiddenSize {
		t.Fatalf("unexpected embedding dimensions: %v", len(qEmbs))
	}

	passage := "Consensus algorithms like Raft and Paxos ensure consistency across nodes."
	pEmbs, err := eng.EmbedPassage(passage)
	if err != nil {
		t.Fatalf("EmbedPassage failed: %v", err)
	}

	// 3. Test top-level CosineSimilarity function
	sim := embed.CosineSimilarity(qEmbs[0], pEmbs[0])
	if sim < 0.85 {
		t.Errorf("expected high similarity, got %f", sim)
	}

	// 4. Test Engine.Similarity method
	directSim, err := eng.Similarity("query: "+query, "passage: "+passage)
	if err != nil {
		t.Fatalf("Engine.Similarity failed: %v", err)
	}
	if directSim < 0.85 {
		t.Errorf("expected high direct similarity, got %f", directSim)
	}
}

func TestTopLevelBF16AndQuantized(t *testing.T) {
	dataDir := filepath.Join("models", "intfloat", "multilingual-e5-small")

	// BF16 Option
	bf16Eng, err := embed.NewEngine(
		embed.WithDataDir(dataDir),
		embed.WithBF16(),
	)
	if err != nil {
		t.Fatalf("NewEngine with WithBF16 failed: %v", err)
	}
	defer bf16Eng.Close()

	if bf16Eng.Precision() != embed.PrecisionBF16 {
		t.Fatalf("expected BF16 precision, got %v", bf16Eng.Precision())
	}

	// INT8 Option
	int8Eng, err := embed.NewEngine(
		embed.WithDataDir(dataDir),
		embed.WithINT8(),
	)
	if err != nil {
		t.Fatalf("NewEngine with WithINT8 failed: %v", err)
	}
	defer int8Eng.Close()

	if int8Eng.Precision() != embed.PrecisionINT8 {
		t.Fatalf("expected INT8 precision, got %v", int8Eng.Precision())
	}
	if !int8Eng.IsQuantized() {
		t.Fatalf("expected IsQuantized() == true")
	}
}

func TestTopLevelContextPool(t *testing.T) {
	dataDir := filepath.Join("models", "intfloat", "multilingual-e5-small")
	eng, err := embed.NewEngine(embed.WithDataDir(dataDir))
	if err != nil {
		t.Fatalf("NewEngine failed: %v", err)
	}
	defer eng.Close()

	pool := embed.NewContextPool(eng.Model(), embed.DefaultWindowSize, embed.DefaultOverlap, "", "")
	ctx := pool.Get()
	defer pool.Put(ctx)

	embs, err := ctx.Embed("Test context pool execution")
	if err != nil {
		t.Fatalf("ctx.Embed failed: %v", err)
	}
	if len(embs) == 0 || len(embs[0]) != embed.HiddenSize {
		t.Fatalf("unexpected context pool embedding dimensions")
	}
}
