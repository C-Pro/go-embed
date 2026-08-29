package engine

import (
	"fmt"
	"sync"
)

// Engine is the thread-safe, production-ready inference engine for text embedding generation.
type Engine struct {
	model *Model
	pool  *ContextPool
}

// New creates a new Engine instance by loading model weights and tokenizer from disk.
func New(modelPath, tokenizerPath string) (*Engine, error) {
	m, err := LoadModel(modelPath, tokenizerPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load model: %w", err)
	}

	return &Engine{
		model: m,
		pool:  NewContextPool(m),
	}, nil
}

// Model returns the underlying Model.
func (e *Engine) Model() *Model {
	return e.model
}

// Embed generates an L2-normalized 384-dimensional embedding for the input text.
func (e *Engine) Embed(text string) ([]float32, error) {
	ctx := e.pool.Get()
	defer e.pool.Put(ctx)

	out := make([]float32, HiddenSize)
	return ctx.Embed(text, out)
}

// EmbedQuery generates an embedding for a query with standard 'query: ' prefix.
func (e *Engine) EmbedQuery(text string) ([]float32, error) {
	ctx := e.pool.Get()
	defer e.pool.Put(ctx)

	out := make([]float32, HiddenSize)
	return ctx.EmbedQuery(text, out)
}

// EmbedPassage generates an embedding for a passage with standard 'passage: ' prefix.
func (e *Engine) EmbedPassage(text string) ([]float32, error) {
	ctx := e.pool.Get()
	defer e.pool.Put(ctx)

	out := make([]float32, HiddenSize)
	return ctx.EmbedPassage(text, out)
}

// EmbedBatch generates embeddings concurrently across multiple texts.
func (e *Engine) EmbedBatch(texts []string) ([][]float32, error) {
	results := make([][]float32, len(texts))
	var wg sync.WaitGroup
	errCh := make(chan error, len(texts))

	for i, text := range texts {
		wg.Add(1)
		go func(idx int, txt string) {
			defer wg.Done()
			ctx := e.pool.Get()
			defer e.pool.Put(ctx)

			out := make([]float32, HiddenSize)
			if _, err := ctx.Embed(txt, out); err != nil {
				errCh <- fmt.Errorf("batch item %d failed: %w", idx, err)
				return
			}
			results[idx] = out
		}(i, text)
	}

	wg.Wait()
	close(errCh)

	if len(errCh) > 0 {
		return nil, <-errCh
	}
	return results, nil
}

// Similarity calculates the cosine similarity between two text strings.
func (e *Engine) Similarity(textA, textB string) (float32, error) {
	embA, err := e.Embed(textA)
	if err != nil {
		return 0, err
	}
	embB, err := e.Embed(textB)
	if err != nil {
		return 0, err
	}
	return CosineSimilarity(embA, embB), nil
}
