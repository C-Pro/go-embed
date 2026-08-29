package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"go-embed/pkg/engine"
)

func main() {
	modelDir := flag.String("model-dir", "models/intfloat/multilingual-e5-small", "Path to model directory containing model.safetensors and tokenizer.json")
	mode := flag.String("mode", "cli", "Mode of operation: 'cli' (interactive), 'query' (single query embedding), or 'sim' (cosine similarity between two texts)")
	text1 := flag.String("t1", "", "First text for similarity comparison or single text for query")
	text2 := flag.String("t2", "", "Second text for similarity comparison")
	isQuery := flag.Bool("query-prefix", true, "Automatically prepend 'query: ' if no prefix is present")
	isPassage := flag.Bool("passage-prefix", false, "Prepend 'passage: ' prefix")
	flag.Parse()

	safetensorsPath := filepath.Join(*modelDir, "model.safetensors")
	tokPath := filepath.Join(*modelDir, "tokenizer.json")

	if _, err := os.Stat(safetensorsPath); os.IsNotExist(err) {
		fmt.Fprintf(os.Stderr, "Error: model weights not found at %s\n", safetensorsPath)
		os.Exit(1)
	}

	fmt.Printf("Loading pure Go E5 engine from %s...\n", *modelDir)
	start := time.Now()
	eng, err := engine.New(safetensorsPath, tokPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to load model: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Model loaded in %v (SIMD Accelerated: %v, Hidden Size: %d, Layers: %d)\n\n",
		time.Since(start), engine.HasSIMD, engine.HiddenSize, engine.NumLayers)

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
		emb, err := eng.Embed(input)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Embed error: %v\n", err)
			os.Exit(1)
		}
		elapsed := time.Since(t0)
		fmt.Printf("Input: %q\n", input)
		fmt.Printf("Generated %d-dim vector in %v:\n", len(emb), elapsed)
		fmt.Printf("[%.5f, %.5f, %.5f, ..., %.5f, %.5f, %.5f]\n",
			emb[0], emb[1], emb[2], emb[len(emb)-3], emb[len(emb)-2], emb[len(emb)-1])

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
		emb1, err1 := eng.Embed(in1)
		emb2, err2 := eng.Embed(in2)
		if err1 != nil || err2 != nil {
			fmt.Fprintf(os.Stderr, "Embed error: %v / %v\n", err1, err2)
			os.Exit(1)
		}
		sim := engine.CosineSimilarity(emb1, emb2)
		elapsed := time.Since(t0)

		fmt.Printf("Text 1: %q\n", in1)
		fmt.Printf("Text 2: %q\n", in2)
		fmt.Printf("Cosine Similarity: %.4f (computed in %v)\n", sim, elapsed)

	case "cli":
		fmt.Println("=== Interactive Text Embedding & Semantic Search CLI ===")
		fmt.Println("Enter text below to generate embeddings, or type 'sim <text1> | <text2>' for cosine similarity.")
		fmt.Println("Type 'exit' or 'quit' to quit.\n")

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
				eq, err1 := eng.Embed(q)
				ep, err2 := eng.Embed(p)
				if err1 != nil || err2 != nil {
					fmt.Printf("Error: %v / %v\n", err1, err2)
					continue
				}
				sim := engine.CosineSimilarity(eq, ep)
				fmt.Printf("Cosine Similarity: %.4f (latency: %v)\n", sim, time.Since(t0))
				continue
			}

			input := prepareText(line)
			t0 := time.Now()
			emb, err := eng.Embed(input)
			if err != nil {
				fmt.Printf("Error: %v\n", err)
				continue
			}
			elapsed := time.Since(t0)
			fmt.Printf("Embedding (%d dims, latency %v):\n[%.4f, %.4f, %.4f, ..., %.4f, %.4f, %.4f]\n\n",
				len(emb), elapsed, emb[0], emb[1], emb[2], emb[len(emb)-3], emb[len(emb)-2], emb[len(emb)-1])
		}
	}
}
