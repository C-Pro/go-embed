package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"go-embed/pkg/engine"
)

func main() {
	modelDir := flag.String("model-dir", "", "Path to model directory containing model.safetensors and tokenizer.json")
	modelName := flag.String("model-name", "", "Hugging Face model repository name (e.g. 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')")
	mode := flag.String("mode", "cli", "Mode of operation: 'cli' (interactive), 'query' (single query embedding), or 'sim' (cosine similarity between two texts)")
	text1 := flag.String("t1", "", "First text for similarity comparison or single text for query")
	text2 := flag.String("t2", "", "Second text for similarity comparison")
	isQuery := flag.Bool("query-prefix", true, "Automatically prepend 'query: ' if no prefix is present")
	isPassage := flag.Bool("passage-prefix", false, "Prepend 'passage: ' prefix")
	precFlag := flag.String("prec", "fp32", "Precision mode: 'fp32' (full precision), 'bf16' (bfloat16 2x smaller), or 'int8' (int8 3.6x smaller)")
	quant := flag.Bool("quant", false, "Shortcut for -prec=int8")
	bf16 := flag.Bool("bf16", false, "Shortcut for -prec=bf16")
	flag.Parse()

	prec := engine.PrecisionFP32
	if *quant || strings.ToLower(*precFlag) == "int8" {
		prec = engine.PrecisionINT8
	} else if *bf16 || strings.ToLower(*precFlag) == "bf16" {
		prec = engine.PrecisionBF16
	}

	opts := []engine.Option{engine.WithPrecision(prec)}
	if *modelName != "" {
		opts = append(opts, engine.WithModelName(*modelName))
	}
	if *modelDir != "" {
		opts = append(opts, engine.WithDataDir(*modelDir))
	}

	targetDesc := *modelName
	if targetDesc == "" {
		if *modelDir != "" {
			targetDesc = *modelDir
		} else {
			targetDesc = engine.DefaultModelName
		}
	}

	fmt.Printf("Initializing pure Go embedding engine (Precision: %s, Model: %s)...\n", prec, targetDesc)
	start := time.Now()
	eng, err := engine.NewEngine(opts...)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to load model: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Model ready in %v (SIMD Accelerated: %v, Precision: %s, Hidden Size: %d, Layers: %d)\n\n",
		time.Since(start), engine.HasSIMD, eng.Precision(), engine.HiddenSize, engine.NumLayers)

	prepareText := func(t string) string {
		if strings.HasPrefix(t, "query: ") || strings.HasPrefix(t, "passage: ") {
			return t
		}
		if *isPassage {
			return "passage: " + t
		}
		if *isQuery {
			return "query: " + t
		}
		return t
	}

	switch *mode {
	case "query":
		if *text1 == "" {
			fmt.Fprintln(os.Stderr, "Error: -t1 is required in query mode")
			os.Exit(1)
		}
		input := prepareText(*text1)
		t0 := time.Now()
		embs, err := eng.Embed(input)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Embed error: %v\n", err)
			os.Exit(1)
		}
		elapsed := time.Since(t0)
		fmt.Printf("Input: %q\n", input)
		fmt.Printf("Generated %d vector(s) (%d-dim) in %v:\n", len(embs), len(embs[0]), elapsed)
		for i, emb := range embs {
			if len(embs) > 1 {
				fmt.Printf("Chunk [%d/%d]:\n", i+1, len(embs))
			}
			fmt.Printf("[%.5f, %.5f, %.5f, ..., %.5f, %.5f, %.5f]\n",
				emb[0], emb[1], emb[2], emb[len(emb)-3], emb[len(emb)-2], emb[len(emb)-1])
		}

	case "sim":
		if *text1 == "" || *text2 == "" {
			fmt.Fprintln(os.Stderr, "Error: -t1 and -t2 are required in sim mode")
			os.Exit(1)
		}
		in1 := prepareText(*text1)
		in2 := *text2
		if !strings.HasPrefix(in2, "query: ") && !strings.HasPrefix(in2, "passage: ") {
			in2 = "passage: " + in2
		}

		t0 := time.Now()
		sim, err := eng.Similarity(in1, in2)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Similarity error: %v\n", err)
			os.Exit(1)
		}
		elapsed := time.Since(t0)

		fmt.Printf("Text 1: %q\n", in1)
		fmt.Printf("Text 2: %q\n", in2)
		fmt.Printf("Cosine Similarity: %.4f (computed in %v)\n", sim, elapsed)

	case "cli":
		fmt.Println("=== Interactive Text Embedding & Semantic Search CLI ===")
		fmt.Println("Enter text below to generate embeddings, or type 'sim <text1> | <text2>' for cosine similarity.")
		fmt.Printf("Type 'exit' or 'quit' to quit.\n\n")

		scanner := bufio.NewScanner(os.Stdin)
		for {
			fmt.Print("go-embed> ")
			if !scanner.Scan() {
				break
			}
			line := strings.TrimSpace(scanner.Text())
			if line == "" {
				continue
			}
			if line == "exit" || line == "quit" {
				break
			}

			if strings.HasPrefix(line, "sim ") {
				rest := strings.TrimPrefix(line, "sim ")
				parts := strings.Split(rest, "|")
				if len(parts) != 2 {
					fmt.Println("Usage: sim <query> | <passage>")
					continue
				}
				q := prepareText(strings.TrimSpace(parts[0]))
				p := strings.TrimSpace(parts[1])
				if !strings.HasPrefix(p, "query: ") && !strings.HasPrefix(p, "passage: ") {
					p = "passage: " + p
				}

				t0 := time.Now()
				sim, err := eng.Similarity(q, p)
				if err != nil {
					fmt.Printf("Error: %v\n", err)
					continue
				}
				fmt.Printf("Cosine Similarity: %.4f (latency: %v)\n", sim, time.Since(t0))
				continue
			}

			input := prepareText(line)
			t0 := time.Now()
			embs, err := eng.Embed(input)
			if err != nil {
				fmt.Printf("Error: %v\n", err)
				continue
			}
			elapsed := time.Since(t0)
			fmt.Printf("Generated %d vector(s) (%d-dim, latency %v):\n", len(embs), len(embs[0]), elapsed)
			for i, emb := range embs {
				if len(embs) > 1 {
					fmt.Printf("Chunk [%d/%d]: ", i+1, len(embs))
				}
				fmt.Printf("[%.4f, %.4f, %.4f, ..., %.4f, %.4f, %.4f]\n",
					emb[0], emb[1], emb[2], emb[len(emb)-3], emb[len(emb)-2], emb[len(emb)-1])
			}
			fmt.Println()
		}
	}
}
