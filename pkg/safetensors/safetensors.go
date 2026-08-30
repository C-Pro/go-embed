package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"unsafe"
)

// TensorInfo contains metadata for a single tensor in a Safetensors file.
type TensorInfo struct {
	DType       string   `json:"dtype"`
	Shape       []int    `json:"shape"`
	DataOffsets [2]int64 `json:"data_offsets"`
}

// Safetensors represents an opened, memory-mapped Safetensors file.
type Safetensors struct {
	mmap       MmapFile
	headerSize int64
	Tensors    map[string]TensorInfo
	dataOffset int64
}

// Open opens and memory-maps a Safetensors file.
func Open(path string) (*Safetensors, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open safetensors file: %w", err)
	}

	mmapFile, err := openMmap(f)
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("failed to memory-map safetensors file: %w", err)
	}

	allBytes := mmapFile.Bytes()
	if len(allBytes) < 8 {
		mmapFile.Close()
		return nil, fmt.Errorf("file too small for safetensors header")
	}

	headerLen := binary.LittleEndian.Uint64(allBytes[0:8])
	if int64(8+headerLen) > int64(len(allBytes)) {
		mmapFile.Close()
		return nil, fmt.Errorf("header length %d exceeds file size %d", headerLen, len(allBytes))
	}

	headerBytes := allBytes[8 : 8+headerLen]

	var rawMap map[string]json.RawMessage
	if err := json.Unmarshal(headerBytes, &rawMap); err != nil {
		mmapFile.Close()
		return nil, fmt.Errorf("failed to unmarshal header JSON: %w", err)
	}

	tensors := make(map[string]TensorInfo, len(rawMap))
	for k, v := range rawMap {
		if k == "__metadata__" {
			continue
		}
		var info TensorInfo
		if err := json.Unmarshal(v, &info); err != nil {
			mmapFile.Close()
			return nil, fmt.Errorf("failed to parse tensor info for %s: %w", k, err)
		}
		tensors[k] = info
	}

	dataOffset := int64(8 + headerLen)
	return &Safetensors{
		mmap:       mmapFile,
		headerSize: int64(headerLen),
		Tensors:    tensors,
		dataOffset: dataOffset,
	}, nil
}

// Close unmaps memory and closes the underlying file descriptor.
func (s *Safetensors) Close() error {
	if s.mmap != nil {
		err := s.mmap.Close()
		s.mmap = nil
		return err
	}
	return nil
}

// TensorF32View returns a float32 slice backed directly by mmap memory if 4-byte aligned,
// or a newly copied slice if unaligned.
func (s *Safetensors) TensorF32View(name string) ([]float32, error) {
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

	start := s.dataOffset + info.DataOffsets[0]
	end := s.dataOffset + info.DataOffsets[1]
	allBytes := s.mmap.Bytes()

	if start < 0 || end > int64(len(allBytes)) || start > end {
		return nil, fmt.Errorf("tensor %q offsets [%d, %d] out of bounds (file size %d)", name, start, end, len(allBytes))
	}

	raw := allBytes[start:end]
	if len(raw) != numElements*4 {
		return nil, fmt.Errorf("tensor %q byte length %d != expected %d", name, len(raw), numElements*4)
	}

	if uintptr(unsafe.Pointer(&raw[0]))%4 == 0 {
		return unsafe.Slice((*float32)(unsafe.Pointer(&raw[0])), numElements), nil
	}

	// Unaligned fallback: copy
	out := make([]float32, numElements)
	for i := 0; i < numElements; i++ {
		bits := binary.LittleEndian.Uint32(raw[i*4 : (i+1)*4])
		out[i] = math.Float32frombits(bits)
	}
	return out, nil
}

// TensorBF16View returns a uint16 BFloat16 slice backed directly by mmap memory if 2-byte aligned,
// or a newly copied slice if unaligned.
func (s *Safetensors) TensorBF16View(name string) ([]uint16, error) {
	info, ok := s.Tensors[name]
	if !ok {
		return nil, fmt.Errorf("tensor %q not found in safetensors", name)
	}
	if info.DType != "BF16" {
		return nil, fmt.Errorf("tensor %q has dtype %s, expected BF16", name, info.DType)
	}

	numElements := 1
	for _, dim := range info.Shape {
		numElements *= dim
	}

	start := s.dataOffset + info.DataOffsets[0]
	end := s.dataOffset + info.DataOffsets[1]
	allBytes := s.mmap.Bytes()

	if start < 0 || end > int64(len(allBytes)) || start > end {
		return nil, fmt.Errorf("tensor %q offsets [%d, %d] out of bounds (file size %d)", name, start, end, len(allBytes))
	}

	raw := allBytes[start:end]
	if len(raw) != numElements*2 {
		return nil, fmt.Errorf("tensor %q byte length %d != expected %d", name, len(raw), numElements*2)
	}

	if uintptr(unsafe.Pointer(&raw[0]))%2 == 0 {
		return unsafe.Slice((*uint16)(unsafe.Pointer(&raw[0])), numElements), nil
	}

	out := make([]uint16, numElements)
	for i := 0; i < numElements; i++ {
		out[i] = binary.LittleEndian.Uint16(raw[i*2 : (i+1)*2])
	}
	return out, nil
}

// TensorI8View returns an int8 slice backed directly by mmap memory.
func (s *Safetensors) TensorI8View(name string) ([]int8, error) {
	info, ok := s.Tensors[name]
	if !ok {
		return nil, fmt.Errorf("tensor %q not found in safetensors", name)
	}
	if info.DType != "I8" && info.DType != "INT8" {
		return nil, fmt.Errorf("tensor %q has dtype %s, expected I8", name, info.DType)
	}

	numElements := 1
	for _, dim := range info.Shape {
		numElements *= dim
	}

	start := s.dataOffset + info.DataOffsets[0]
	end := s.dataOffset + info.DataOffsets[1]
	allBytes := s.mmap.Bytes()

	if start < 0 || end > int64(len(allBytes)) || start > end {
		return nil, fmt.Errorf("tensor %q offsets [%d, %d] out of bounds (file size %d)", name, start, end, len(allBytes))
	}

	raw := allBytes[start:end]
	if len(raw) != numElements {
		return nil, fmt.Errorf("tensor %q byte length %d != expected %d", name, len(raw), numElements)
	}

	return unsafe.Slice((*int8)(unsafe.Pointer(&raw[0])), numElements), nil
}

// ReadTensorF32 reads a float32 tensor by name.
func (s *Safetensors) ReadTensorF32(name string) ([]float32, error) {
	return s.TensorF32View(name)
}

// ReadTensorF32Into reads a float32 tensor directly into an existing slice.
func (s *Safetensors) ReadTensorF32Into(name string, dest []float32) error {
	view, err := s.TensorF32View(name)
	if err != nil {
		return err
	}
	if len(dest) < len(view) {
		return fmt.Errorf("destination slice len (%d) < tensor elements (%d)", len(dest), len(view))
	}
	copy(dest, view)
	return nil
}
