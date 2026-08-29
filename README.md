# go-embed

A high-performance, pure Go (CGO-free), CPU-only inference library for tokenization and text embedding generation. 

`go-embed` natively executes [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small) (XLM-RoBERTa architecture: 12 layers, hidden dimension 384, 12 attention heads, 1536 intermediate size, ~250k vocabulary) with exact numerical parity to PyTorch/Hugging Face Transformers.

> [!NOTE]
> This library is based on and inspired by [`github.com/nlpodyssey/spago`](https://github.com/nlpodyssey/spago), re-engineered into a lightweight, standalone, zero-allocation runtime optimized for CPU inference with Go 1.27 hardware vector acceleration (SIMD).

---

## Key Features

- **Pure Go / CGO-Free:** Zero external C/C++ libraries, BLAS/LAPACK binaries, or Python runtimes required. Cross-compiles to any Go-supported platform out of the box.
- **Zero Heap Allocations in Steady-State:** Pre-allocated scratchpad buffers (`InferenceContext`) achieve **`0 B/op` and `0 allocs/op`** during inference.
- **Hardware Acceleration via Go 1.27 SIMD:** Leverages Go 1.27's standard `simd.Float32s` with 8-way unrolled fused multiply-add (`MulAdd`), vectorized LayerNorm, fast GELU, and cosine similarity. Includes portable scalar fallback when compiled without SIMD flags.
- **Exact Numerical Parity:** Validated against PyTorch reference outputs with Cosine Similarity $\ge 0.999999$ across short queries, full sentences, code snippets, empty strings, and 512-token max-length sequences.
- **Cross-Lingual Support:** Supports 100+ languages (English, Russian, Indonesian, German, Chinese, etc.) with semantic consistency $\ge 0.80$ across language pairs.
- **Built-in Tokenizer & Safetensors Loader:** Includes a pure Go SentencePiece/Unigram tokenizer (with NFKC normalization and Viterbi DP search) and `.safetensors` weight loader with zero external dependencies.

---

## Performance & Benchmarks

*Benchmarked on AMD Ryzen 7 PRO 8840U (16 threads), Go 1.27, Linux amd64.*

| Sequence Length | Spago Reference Baseline | `go-embed` (Scalar Engine) | `go-embed` (SIMD Engine) | Speedup vs Baseline | Allocations |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **$L=32$** (Short Query) | 405 ms | 351 ms | **90 ms** | **4.5× faster** | **0 allocs / 0 B** |
| **$L=128$** (Medium Passage) | 8,200 ms | 1,528 ms | **581 ms** | **14.1× faster** | **0 allocs / 0 B** |
| **$L=256$** (Long Passage) | 45,303 ms | 3,327 ms | **1,638 ms** | **27.6× faster** | **0 allocs / 0 B** |
| **$L=512$** (Max Sequence) | ~3.5 min | 8,143 ms | **5,147 ms** | **>40× faster** | **0 allocs / 0 B** |
| **Concurrent Batch ($B=16$)** | N/A | N/A | **150 ms total (~9.3 ms/query)** | **High Throughput** | Minimal pool overhead |

---

## Installation & Requirements

- **Go 1.27+** (recommended for hardware SIMD acceleration) or **Go 1.21+** (for scalar fallback).

To install the module:

```bash
go get go-embed
```

---

## Quick Start

### 1. Download Model Weights

Download the `multilingual-e5-small` model weights and tokenizer:

```bash
mkdir -p models/intfloat/multilingual-e5-small
cd models/intfloat/multilingual-e5-small
wget https://huggingface.co/intfloat/multilingual-e5-small/resolve/main/model.safetensors
wget https://huggingface.co/intfloat/multilingual-e5-small/resolve/main/tokenizer.json
wget https://huggingface.co/intfloat/multilingual-e5-small/resolve/main/config.json
cd ../../..
```

### 2. High-Level Engine API

```go
package main

import (
	"fmt"
	"log"

	"go-embed/pkg/engine"
)

func main() {
	// Initialize engine (loads model.safetensors & tokenizer.json)
	eng, err := engine.New(
		"models/intfloat/multilingual-e5-small/model.safetensors",
		"models/intfloat/multilingual-e5-small/tokenizer.json",
	)
	if err != nil {
		log.Fatalf("Failed to initialize engine: %v", err)
	}

	// E5 models use "query: " and "passage: " prefixes for asymmetric retrieval
	query := "how to implement consensus in distributed systems?"
	relPassage := "Consensus algorithms like Raft and Paxos ensure consistency across nodes."
	irrelPassage := "Authentic Italian tiramisu recipe with mascarpone and espresso."

	// Generate 384-dimensional L2-normalized embeddings
	qEmb, _ := eng.EmbedQuery(query)
	pRelEmb, _ := eng.EmbedPassage(relPassage)
	pIrrelEmb, _ := eng.EmbedPassage(irrelPassage)

	// Compute Cosine Similarity
	simRel := engine.CosineSimilarity(qEmb, pRelEmb)
	simIrrel := engine.CosineSimilarity(qEmb, pIrrelEmb)

	fmt.Printf("Query: %s\n", query)
	fmt.Printf("Similarity to relevant passage:   %.4f\n", simRel)   // ~0.87
	fmt.Printf("Similarity to irrelevant passage: %.4f\n", simIrrel) // ~0.73
}
```

---

### 3. Zero-Allocation Batching & Context Pool

For high-throughput web servers or pipeline processing, reuse pre-allocated buffers with `ContextPool`:

```go
package main

import (
	"fmt"
	"go-embed/pkg/engine"
)

func main() {
	eng, _ := engine.New(
		"models/intfloat/multilingual-e5-small/model.safetensors",
		"models/intfloat/multilingual-e5-small/tokenizer.json",
	)

	// ContextPool manages reusable memory scratchpads for goroutines
	pool := engine.NewContextPool(eng.Model())
	ctx := pool.Get()
	defer pool.Put(ctx)

	// Pre-allocated destination slice (384 float32s)
	outBuf := make([]float32, engine.HiddenSize)

	// Generates embedding into outBuf with 0 heap allocations
	emb, err := ctx.Embed("query: low latency Go inference", outBuf)
	if err != nil {
		panic(err)
	}

	fmt.Printf("First 3 dimensions: [%.4f, %.4f, %.4f]\n", emb[0], emb[1], emb[2])
}
```

---

### 4. Parallel Batch Embedding

Process multiple queries concurrently across CPU cores:

```go
texts := []string{
	"query: Kubernetes cluster architecture",
	"query: Distributed key-value store in Go",
	"query: Lock-free ring buffer implementation",
}

// Concurrently computes embeddings across available CPU threads
embeddings, err := eng.EmbedBatch(texts)
if err != nil {
	log.Fatal(err)
}
fmt.Printf("Computed %d embeddings\n", len(embeddings))
```

---

## Interactive CLI Tool

`go-embed` includes an interactive CLI utility for generating embeddings and computing semantic similarities:

```bash
# Interactive REPL
GOEXPERIMENT=simd go run cmd/embed/main.go -mode=cli

# Direct Similarity Comparison
GOEXPERIMENT=simd go run cmd/embed/main.go -mode=sim \
  -t1="how to implement consensus in distributed systems?" \
  -t2="Consensus algorithms like Raft and Paxos ensure state machine consistency."

# Single Query Embedding
GOEXPERIMENT=simd go run cmd/embed/main.go -mode=query -t1="machine learning in Go"
```

---

## Running Tests & Verification

```bash
# Run all unit tests, golden parity verification, and cross-lingual sanity tests
GOEXPERIMENT=simd go test -v ./...

# Run zero-allocation assertion (verifies 0.0 allocs/op)
GOEXPERIMENT=simd go test -v -run TestZeroAllocations ./pkg/engine

# Run comparative benchmarks
GOEXPERIMENT=simd go test -run=^$ -bench=. -benchmem ./pkg/engine
```

---

## Architecture & Project Structure

```
.
├── cmd/
│   └── embed/              # Interactive CLI application
├── pkg/
│   ├── engine/             # Core inference engine, Post-LN transformer layers & scratchpad pool
│   │   ├── ops_simd.go     # Go 1.27 hardware SIMD vector kernels (FMA, LayerNorm, GELU)
│   │   ├── ops_scalar.go   # Unrolled portable scalar math kernels
│   │   ├── context.go      # Pre-allocated zero-allocation scratchpad and sync.Pool
│   │   ├── model.go        # Contiguous model parameter layouts
│   │   └── engine.go       # High-level thread-safe embedding API
│   ├── safetensors/        # Pure Go .safetensors binary reader and parser
│   ├── tokenizer/          # Pure Go SentencePiece/Unigram tokenizer & Viterbi search
│   └── spagoref/           # Spago reference baseline harness for parity testing
└── testdata/
    └── golden.json         # 26 PyTorch/Transformers ground truth vectors
```

---

## Acknowledgments

- [`github.com/nlpodyssey/spago`](https://github.com/nlpodyssey/spago) for the original pure-Go machine learning framework and foundation.
- [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small) for the multilingual embedding model weights.

## License

MIT License.
