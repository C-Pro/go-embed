package safetensors_test

import (
	"flag"
	"os"
	"path/filepath"
	"testing"

	"github.com/C-Pro/go-embed/pkg/safetensors"
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

func FuzzSafetensorsOpen(f *testing.F) {
	skipFuzzInCI(f)

	// Seed with a minimal valid safetensors header
	seedTensors := map[string]safetensors.TensorData{
		"weight": safetensors.NewTensorF32([]int{2, 3}, []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}),
		"bias":   safetensors.NewTensorF32([]int{2}, []float32{0.1, 0.2}),
	}
	tmpSeed := filepath.Join(f.TempDir(), "seed.safetensors")
	if err := safetensors.WriteFile(tmpSeed, seedTensors); err == nil {
		if data, err := os.ReadFile(tmpSeed); err == nil {
			f.Add(data)
		}
	}

	// Also seed with some invalid/empty payloads
	f.Add([]byte{})
	f.Add([]byte{0, 0, 0, 0, 0, 0, 0, 0})
	f.Add([]byte{8, 0, 0, 0, 0, 0, 0, 0, '{', '}'})
	f.Add([]byte{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff})

	f.Fuzz(func(t *testing.T, data []byte) {
		tmpPath := filepath.Join(t.TempDir(), "fuzz.safetensors")
		if err := os.WriteFile(tmpPath, data, 0644); err != nil {
			return
		}
		defer os.Remove(tmpPath)

		st, err := safetensors.Open(tmpPath)
		if err != nil {
			// Returning an error on malformed data is the expected outcome
			return
		}
		defer st.Close()

		// Attempt reading tensor views across all reported tensors
		for name := range st.Tensors {
			_, _ = st.TensorF32View(name)
			_, _ = st.TensorBF16View(name)
			_, _ = st.TensorI8View(name)
			dest := make([]float32, 100)
			_ = st.ReadTensorF32Into(name, dest)
		}

		// Also query nonexistent names
		_, _ = st.TensorF32View("nonexistent")
		_, _ = st.TensorBF16View("nonexistent")
		_, _ = st.TensorI8View("nonexistent")
	})
}
