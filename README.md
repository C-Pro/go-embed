# go-embed

A high-performance, pure Go (CGO-free), CPU-only inference library for tokenization and text embedding generation. 

`go-embed` natively executes [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small) (XLM-RoBERTa architecture: 12 layers, hidden dimension 384, 12 attention heads, 1536 intermediate size, ~250k vocabulary) with exact numerical parity to PyTorch/Hugging Face Transformers.

> [!NOTE]
> This library is based on and inspired by [`github.com/nlpodyssey/spago`](https://github.com/nlpodyssey/spago), re-engineered into a lightweight, standalone, zero-allocation runtime optimized for CPU inference with Go 1.27 hardware vector acceleration (SIMD).

---

## Key Features

- **Pure Go / CGO-Free:** Zero external C/C++ libraries, BLAS/LAPACK binaries, or Python runtimes required. Cross-compiles to any Go-supported platform out of the box.
- **Multiple Precision Modes:**
  - **FP32 Full Precision (Default):** Maximum compute performance (`~54 ms` short query latency, 449 MB RAM).
  - **BFloat16 (BF16):** Native 16-bit float weights reduce RAM to **225 MB (2× reduction)** with $>99.999\%$ fidelity and zero scaling overhead.
  - **INT8 Dynamic Quantization (W8A32):** Dynamic 8-bit integer weights reduce RAM to **~125 MB (3.6× reduction)** with $>99.97\%$ fidelity.
- **Zero Heap Allocations in Steady-State:** Pre-allocated scratchpad buffers (`InferenceContext`) achieve **`0 B/op` and `0 allocs/op`** during inference across all precision modes.
- **Hardware Acceleration via Go 1.27 SIMD:** Leverages Go 1.27's standard `simd.Float32s` with 8-way unrolled fused multiply-add (`MulAdd`), vectorized LayerNorm, fast GELU, and cosine similarity. Includes portable scalar fallback when compiled without SIMD flags.
- **Exact Numerical Parity:** Validated against PyTorch reference outputs with Cosine Similarity $\ge 0.999999$ across short queries, full sentences, code snippets, empty strings, and 512-token max-length sequences.
- **Cross-Lingual Support:** Supports 100+ languages (English, Russian, Indonesian, German, Chinese, etc.) with semantic consistency $\ge 0.80$ across language pairs.
- **Built-in Tokenizer & Safetensors Loader:** Includes a pure Go SentencePiece/Unigram tokenizer (with NFKC normalization and Viterbi DP search) and `.safetensors` weight loader with zero external dependencies.

---

## Precision Modes & Benchmarks

*Benchmarked on AMD Ryzen 7 PRO 8840U (16 threads), Go 1.27, Linux amd64.*

| Sequence Length | Precision | RAM Footprint | Single-Query Latency | Allocations | Fidelity vs PyTorch |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Short Query ($L=32$)** | **FP32** | 449 MB | **`54 ms`** | **`0 B / 0 allocs`** | 100% (Baseline) |
| **Short Query ($L=32$)** | **BF16** | **225 MB** *(2×)* | **`128 ms`** | **`0 B / 0 allocs`** | **`0.999996`** |
| **Short Query ($L=32$)** | **INT8** | **125 MB** *(3.6×)* | `326 ms` | **`0 B / 0 allocs`** | `0.999787` |
| **Long Sequence ($L=512$)** | **FP32** | 449 MB | **`5.57 s`** | **`0 B / 0 allocs`** | 100% (Baseline) |
| **Long Sequence ($L=512$)** | **BF16** | **225 MB** *(2×)* | **`5.89 s`** | **`0 B / 0 allocs`** | **`0.999996`** |
| **Long Sequence ($L=512$)** | **INT8** | **125 MB** *(3.6×)* | `15.50 s` | **`0 B / 0 allocs`** | `0.999787` |
| *Spago Baseline ($L=512$)* | *FP32* | *>30 GB (OOM risk)* | *~3.5+ min* | *100M+ allocs* | Baseline |

---

## Installation & Requirements

- **Go 1.27+** (recommended for hardware SIMD acceleration) or **Go 1.21+** (for scalar fallback).

To install the module:

```bash
go get go-embed
```

---

## Quick Start

### 1. Minimal Setup (Zero Manual Downloads)

`NewEngine()` automatically checks the local models directory and downloads `multilingual-e5-small` weights from Hugging Face if not already present:

```go
package main

import (
	"fmt"
	"log"

	"go-embed/pkg/engine"
)

func main() {
	// Initialize engine: automatically downloads model if missing, uses FP32 by default
	eng, err := engine.NewEngine()
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

### 2. Ergonomic Functional Options

Customize data directories, precision modes, or custom Hugging Face model repositories:

```go
// Override data directory where to store/look up models
eng, err := engine.NewEngine(
    engine.WithDataDir("./my_models"),
)

// BFloat16 mode (cuts RAM in half to 225 MB; trades off slight compute speed for memory)
eng, err := engine.NewEngine(
    engine.WithDataDir("./my_models"),
    engine.WithBF16(),
)

// INT8 dynamic quantization mode (125 MB RAM for edge/memory-constrained environments)
eng, err := engine.NewEngine(
    engine.WithDataDir("./my_models"),
    engine.WithINT8(),
)

// Override Hugging Face repository name
eng, err := engine.NewEngine(
    engine.WithModelName("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
)
```

---

## Supported Models & Architectures

`go-embed` natively executes any transformer model based on the **XLM-RoBERTa / RoBERTa (12-layer, 384-dim)** architecture with SentencePiece tokenization and `.safetensors` weights.

### Architecture Specifications

| Parameter | Supported Specification |
| :--- | :--- |
| **Architecture Family** | XLM-RoBERTa / RoBERTa / BERT-compatible Encoder |
| **Hidden Dimension ($D$)** | `384` |
| **Transformer Layers ($L$)** | `12` |
| **Attention Heads ($H$)** | `12` ($d_{\text{head}} = 32$) |
| **Intermediate FFN Size ($D_{\text{ffn}}$)** | `1536` |
| **Layer Normalization** | Post-LN ($\epsilon = 10^{-5}$) |
| **Activation Function** | GELU |
| **Tokenizer Algorithm** | SentencePiece (Unigram) with NFKC normalization |
| **Max Sequence Length** | Up to 512 tokens |
| **File Formats** | `model.safetensors` + `tokenizer.json` |

---

### Verified Models (Out-of-the-Box)

The following models are verified and automatically downloaded from Hugging Face when specified via `WithModelName(...)`:

#### 1. [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small) (Default)
- **Best for:** Semantic search, document retrieval, and question answering across 100+ languages.
- **Prefix convention:** Requires `query: ` and `passage: ` prefixes for asymmetric retrieval.
- **Example:**
  ```go
  eng, err := engine.NewEngine(
      engine.WithModelName("intfloat/multilingual-e5-small"),
  )
  qEmb, _ := eng.EmbedQuery("how to implement consensus in distributed systems?")
  pEmb, _ := eng.EmbedPassage("Consensus algorithms like Raft and Paxos ensure consistency.")
  ```

#### 2. [`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- **Best for:** Semantic similarity, sentence paraphrasing, deduplication, and cross-lingual clustering across 50+ languages.
- **Prefix convention:** Direct text input (no prefix required).
- **Example:**
  ```go
  eng, err := engine.NewEngine(
      engine.WithModelName("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
      engine.WithBF16(), // Cuts RAM in half (225 MB)
  )
  emb1, _ := eng.Embed("The cat is sleeping peacefully on the sofa.")
  emb2, _ := eng.Embed("Die Katze schläft friedlich auf dem Sofa.")
  similarity := engine.CosineSimilarity(emb1, emb2) // ~0.9846
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

### 5. Precision Modes & Trade-offs (FP32 vs BF16 vs INT8)

`go-embed` provides three precision modes to balance memory footprint and latency:

* **FP32 Full Precision (`449 MB RAM`, default):** Fastest compute speed for short queries (`54 ms`) with 100% baseline numerical precision.
* **BFloat16 (`225 MB RAM`, `WithBF16()`):** **Trades off slight CPU instruction speed for a 50% memory reduction.** Because weights are stored as `uint16` and unpacked into 32-bit SIMD registers during computation, short query latency is `128 ms`, but for long sequences ($L=512$) the latency difference is only ~5% (`5.89 s` vs `5.57 s`) due to better CPU cache locality.
* **INT8 Dynamic Quantization (`125 MB RAM`, `WithINT8()`):** Maximum memory reduction (3.6× smaller) for resource-constrained edge environments.

```go
// BFloat16 mode (2x smaller RAM, >99.999% fidelity)
bf16Eng, err := engine.NewEngine(engine.WithBF16())

// INT8 mode (3.6x smaller RAM)
int8Eng, err := engine.NewEngine(engine.WithINT8())
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
