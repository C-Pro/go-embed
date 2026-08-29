package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
)

// TensorInfo contains metadata for a single tensor in a Safetensors file.
type TensorInfo struct {
	DType       string   `json:"dtype"`
	Shape       []int    `json:"shape"`
	DataOffsets [2]int64 `json:"data_offsets"`
}

// Safetensors represents an opened Safetensors file.
type Safetensors struct {
	file       *os.File
	headerSize int64
	Tensors    map[string]TensorInfo
	dataOffset int64
}

// Open opens and parses a Safetensors file header.
func Open(path string) (*Safetensors, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open safetensors file: %w", err)
	}

	var headerLen uint64
	if err := binary.Read(f, binary.LittleEndian, &headerLen); err != nil {
		f.Close()
		return nil, fmt.Errorf("failed to read header size: %w", err)
	}

	headerBytes := make([]byte, headerLen)
	if _, err := io.ReadFull(f, headerBytes); err != nil {
		f.Close()
		return nil, fmt.Errorf("failed to read header JSON: %w", err)
	}

	var rawMap map[string]json.RawMessage
	if err := json.Unmarshal(headerBytes, &rawMap); err != nil {
		f.Close()
		return nil, fmt.Errorf("failed to unmarshal header JSON: %w", err)
	}

	tensors := make(map[string]TensorInfo)
	for k, v := range rawMap {
		if k == "__metadata__" {
			continue
		}
		var info TensorInfo
		if err := json.Unmarshal(v, &info); err != nil {
			f.Close()
			return nil, fmt.Errorf("failed to parse tensor info for %s: %w", k, err)
		}
		tensors[k] = info
	}

	dataOffset := int64(8 + headerLen)
	return &Safetensors{
		file:       f,
		headerSize: int64(headerLen),
		Tensors:    tensors,
		dataOffset: dataOffset,
	}, nil
}

// Close closes the underlying file.
func (s *Safetensors) Close() error {
	if s.file != nil {
		return s.file.Close()
	}
	return nil
}

// ReadTensorF32 reads a float32 tensor by name into a newly allocated slice.
func (s *Safetensors) ReadTensorF32(name string) ([]float32, error) {
	info, ok := s.Tensors[name]
	if !ok {
		return nil, fmt.Errorf("tensor %q not found in safetensors", name)
	}
	if info.DType != "F32" {
		return nil, fmt.Errorf("tensor %q has dtype %s, expected F32", name, info.DType)
	}

	numElements := 1
	for _, dim := range info.Shape {
		numElements *= dim
	}

	out := make([]float32, numElements)
	if err := s.ReadTensorF32Into(name, out); err != nil {
		return nil, err
	}
	return out, nil
}

// ReadTensorF32Into reads a float32 tensor directly into an existing slice.
func (s *Safetensors) ReadTensorF32Into(name string, dest []float32) error {
	info, ok := s.Tensors[name]
	if !ok {
		return fmt.Errorf("tensor %q not found in safetensors", name)
	}
	if info.DType != "F32" {
		return fmt.Errorf("tensor %q has dtype %s, expected F32", name, info.DType)
	}

	numElements := 1
	for _, dim := range info.Shape {
		numElements *= dim
	}
	if len(dest) < numElements {
		return fmt.Errorf("destination slice len (%d) < tensor elements (%d)", len(dest), numElements)
	}

	fileOffset := s.dataOffset + info.DataOffsets[0]
	byteLen := info.DataOffsets[1] - info.DataOffsets[0]
	if int64(numElements*4) != byteLen {
		return fmt.Errorf("tensor %q size mismatch: shape %v requires %d bytes, offsets span %d bytes", name, info.Shape, numElements*4, byteLen)
	}

	byteBuf := make([]byte, byteLen)
	if _, err := s.file.ReadAt(byteBuf, fileOffset); err != nil {
		return fmt.Errorf("failed to read tensor bytes: %w", err)
	}

	for i := 0; i < numElements; i++ {
		bits := binary.LittleEndian.Uint32(byteBuf[i*4 : (i+1)*4])
		dest[i] = math.Float32frombits(bits)
	}
	return nil
}
