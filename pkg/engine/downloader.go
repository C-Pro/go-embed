package engine

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

const (
	DefaultModelName = "intfloat/multilingual-e5-small"
	HuggingFaceBase  = "https://huggingface.co"
)

// EnsureModelFiles checks if the required model weights (model.safetensors and tokenizer.json)
// exist in targetDir. If they do not exist, it automatically downloads them from Hugging Face.
func EnsureModelFiles(dataDir, modelName string, silent bool) (safetensorsPath, tokenizerPath string, err error) {
	if modelName == "" {
		modelName = DefaultModelName
	}
	if dataDir == "" {
		dataDir = filepath.Join("models", modelName)
	}

	if err := os.MkdirAll(dataDir, 0755); err != nil {
		return "", "", fmt.Errorf("failed to create model directory %s: %w", dataDir, err)
	}

	safetensorsPath = filepath.Join(dataDir, "model.safetensors")
	tokenizerPath = filepath.Join(dataDir, "tokenizer.json")

	filesToDownload := []string{"tokenizer.json", "model.safetensors"}

	for _, filename := range filesToDownload {
		targetPath := filepath.Join(dataDir, filename)
		if fi, err := os.Stat(targetPath); err == nil && fi.Size() > 0 {
			continue // Already exists
		}

		url := fmt.Sprintf("%s/%s/resolve/main/%s", HuggingFaceBase, modelName, filename)
		if !silent {
			fmt.Printf("go-embed: downloading %s from %s...\n", filename, url)
		}

		if err := downloadFile(url, targetPath, silent); err != nil {
			return "", "", fmt.Errorf("failed to download %s: %w", filename, err)
		}
	}

	return safetensorsPath, tokenizerPath, nil
}

// downloadFile streams an HTTP file to a temporary file and atomically renames it upon completion.
func downloadFile(url, destPath string, silent bool) error {
	client := &http.Client{
		Timeout: 30 * time.Minute, // Allow sufficient time for larger model files on slow connections
	}

	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("User-Agent", "go-embed/1.0 (pure Go embedding engine)")

	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("HTTP request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("bad HTTP status: %s", resp.Status)
	}

	tmpPath := destPath + ".tmp"
	outFile, err := os.OpenFile(tmpPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
	if err != nil {
		return fmt.Errorf("failed to create temp file %s: %w", tmpPath, err)
	}

	var reader io.Reader = resp.Body
	if !silent && resp.ContentLength > 0 {
		reader = &progressReader{
			Reader: resp.Body,
			Total:  resp.ContentLength,
			Name:   filepath.Base(destPath),
		}
	}

	if _, err := io.Copy(outFile, reader); err != nil {
		outFile.Close()
		os.Remove(tmpPath)
		return fmt.Errorf("failed during streaming download: %w", err)
	}

	if err := outFile.Close(); err != nil {
		os.Remove(tmpPath)
		return fmt.Errorf("failed to close file: %w", err)
	}

	if err := os.Rename(tmpPath, destPath); err != nil {
		os.Remove(tmpPath)
		return fmt.Errorf("failed to finalize downloaded file: %w", err)
	}

	if !silent {
		fmt.Printf("go-embed: successfully downloaded %s\n", filepath.Base(destPath))
	}

	return nil
}

type progressReader struct {
	io.Reader
	Total   int64
	Current int64
	Name    string
	lastLog time.Time
}

func (pr *progressReader) Read(p []byte) (int, error) {
	n, err := pr.Reader.Read(p)
	pr.Current += int64(n)

	now := time.Now()
	if now.Sub(pr.lastLog) > 2*time.Second || (err == io.EOF && pr.Current == pr.Total) {
		pr.lastLog = now
		pct := float64(pr.Current) / float64(pr.Total) * 100.0
		mbCurrent := float64(pr.Current) / (1024 * 1024)
		mbTotal := float64(pr.Total) / (1024 * 1024)
		fmt.Printf("go-embed: [%s] %.1f%% (%.1f / %.1f MB)\n", pr.Name, pct, mbCurrent, mbTotal)
	}

	return n, err
}
