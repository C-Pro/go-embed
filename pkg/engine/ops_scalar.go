package engine

import (
	"math"
)

// MatVecMulAddScalar computes out = x * W^T + bias
// x: [inDim], W: [outDim, inDim] (row-major), bias: [outDim], out: [outDim]
func MatVecMulAddScalar(x []float32, weight []float32, bias []float32, out []float32, inDim, outDim int) {
	for j := 0; j < outDim; j++ {
		wRow := weight[j*inDim : (j+1)*inDim]
		var dot float32
		if bias != nil {
			dot = bias[j]
		}

		// Loop unrolling 8x for instruction pipeline efficiency
		k := 0
		n8 := inDim - (inDim % 8)
		var d0, d1, d2, d3, d4, d5, d6, d7 float32
		for ; k < n8; k += 8 {
			d0 += x[k] * wRow[k]
			d1 += x[k+1] * wRow[k+1]
			d2 += x[k+2] * wRow[k+2]
			d3 += x[k+3] * wRow[k+3]
			d4 += x[k+4] * wRow[k+4]
			d5 += x[k+5] * wRow[k+5]
			d6 += x[k+6] * wRow[k+6]
			d7 += x[k+7] * wRow[k+7]
		}
		dot += (d0 + d1) + (d2 + d3) + (d4 + d5) + (d6 + d7)
		for ; k < inDim; k++ {
			dot += x[k] * wRow[k]
		}
		out[j] = dot
	}
}

// MatVecMulAddINT8Scalar computes out = (x * W_int8^T) * scale + bias
// x: [inDim], weight: [outDim, inDim] (int8 row-major), scale: [outDim], bias: [outDim], out: [outDim]
func MatVecMulAddINT8Scalar(x []float32, weight []int8, scale []float32, bias []float32, out []float32, inDim, outDim int) {
	j := 0
	for ; j <= outDim-4; j += 4 {
		wRow0 := weight[j*inDim : (j+1)*inDim]
		wRow1 := weight[(j+1)*inDim : (j+2)*inDim]
		wRow2 := weight[(j+2)*inDim : (j+3)*inDim]
		wRow3 := weight[(j+3)*inDim : (j+4)*inDim]

		var dot0, dot1, dot2, dot3 float32
		k := 0
		n8 := inDim - (inDim % 8)
		for ; k < n8; k += 8 {
			xk0 := x[k]
			xk1 := x[k+1]
			xk2 := x[k+2]
			xk3 := x[k+3]
			xk4 := x[k+4]
			xk5 := x[k+5]
			xk6 := x[k+6]
			xk7 := x[k+7]

			dot0 += xk0*float32(wRow0[k]) + xk1*float32(wRow0[k+1]) + xk2*float32(wRow0[k+2]) + xk3*float32(wRow0[k+3]) +
				xk4*float32(wRow0[k+4]) + xk5*float32(wRow0[k+5]) + xk6*float32(wRow0[k+6]) + xk7*float32(wRow0[k+7])

			dot1 += xk0*float32(wRow1[k]) + xk1*float32(wRow1[k+1]) + xk2*float32(wRow1[k+2]) + xk3*float32(wRow1[k+3]) +
				xk4*float32(wRow1[k+4]) + xk5*float32(wRow1[k+5]) + xk6*float32(wRow1[k+6]) + xk7*float32(wRow1[k+7])

			dot2 += xk0*float32(wRow2[k]) + xk1*float32(wRow2[k+1]) + xk2*float32(wRow2[k+2]) + xk3*float32(wRow2[k+3]) +
				xk4*float32(wRow2[k+4]) + xk5*float32(wRow2[k+5]) + xk6*float32(wRow2[k+6]) + xk7*float32(wRow2[k+7])

			dot3 += xk0*float32(wRow3[k]) + xk1*float32(wRow3[k+1]) + xk2*float32(wRow3[k+2]) + xk3*float32(wRow3[k+3]) +
				xk4*float32(wRow3[k+4]) + xk5*float32(wRow3[k+5]) + xk6*float32(wRow3[k+6]) + xk7*float32(wRow3[k+7])
		}
		for ; k < inDim; k++ {
			xk := x[k]
			dot0 += xk * float32(wRow0[k])
			dot1 += xk * float32(wRow1[k])
			dot2 += xk * float32(wRow2[k])
			dot3 += xk * float32(wRow3[k])
		}

		res0 := dot0 * scale[j]
		res1 := dot1 * scale[j+1]
		res2 := dot2 * scale[j+2]
		res3 := dot3 * scale[j+3]
		if bias != nil {
			res0 += bias[j]
			res1 += bias[j+1]
			res2 += bias[j+2]
			res3 += bias[j+3]
		}
		out[j] = res0
		out[j+1] = res1
		out[j+2] = res2
		out[j+3] = res3
	}

	for ; j < outDim; j++ {
		wRow := weight[j*inDim : (j+1)*inDim]
		var dot float32
		for k := 0; k < inDim; k++ {
			dot += x[k] * float32(wRow[k])
		}
		res := dot * scale[j]
		if bias != nil {
			res += bias[j]
		}
		out[j] = res
	}
}

// MatVecMulAddBF16Scalar computes out = x * W_bf16^T + bias
// x: [inDim], weight: [outDim, inDim] (BFloat16 uint16 row-major), bias: [outDim], out: [outDim]
func MatVecMulAddBF16Scalar(x []float32, weight []uint16, bias []float32, out []float32, inDim, outDim int) {
	j := 0
	for ; j <= outDim-4; j += 4 {
		wRow0 := weight[j*inDim : (j+1)*inDim]
		wRow1 := weight[(j+1)*inDim : (j+2)*inDim]
		wRow2 := weight[(j+2)*inDim : (j+3)*inDim]
		wRow3 := weight[(j+3)*inDim : (j+4)*inDim]

		var dot0, dot1, dot2, dot3 float32
		if bias != nil {
			dot0 = bias[j]
			dot1 = bias[j+1]
			dot2 = bias[j+2]
			dot3 = bias[j+3]
		}

		k := 0
		n8 := inDim - (inDim % 8)
		for ; k < n8; k += 8 {
			xk0 := x[k]
			xk1 := x[k+1]
			xk2 := x[k+2]
			xk3 := x[k+3]
			xk4 := x[k+4]
			xk5 := x[k+5]
			xk6 := x[k+6]
			xk7 := x[k+7]

			dot0 += xk0*BFloat16ToFloat32(wRow0[k]) + xk1*BFloat16ToFloat32(wRow0[k+1]) + xk2*BFloat16ToFloat32(wRow0[k+2]) + xk3*BFloat16ToFloat32(wRow0[k+3]) +
				xk4*BFloat16ToFloat32(wRow0[k+4]) + xk5*BFloat16ToFloat32(wRow0[k+5]) + xk6*BFloat16ToFloat32(wRow0[k+6]) + xk7*BFloat16ToFloat32(wRow0[k+7])

			dot1 += xk0*BFloat16ToFloat32(wRow1[k]) + xk1*BFloat16ToFloat32(wRow1[k+1]) + xk2*BFloat16ToFloat32(wRow1[k+2]) + xk3*BFloat16ToFloat32(wRow1[k+3]) +
				xk4*BFloat16ToFloat32(wRow1[k+4]) + xk5*BFloat16ToFloat32(wRow1[k+5]) + xk6*BFloat16ToFloat32(wRow1[k+6]) + xk7*BFloat16ToFloat32(wRow1[k+7])

			dot2 += xk0*BFloat16ToFloat32(wRow2[k]) + xk1*BFloat16ToFloat32(wRow2[k+1]) + xk2*BFloat16ToFloat32(wRow2[k+2]) + xk3*BFloat16ToFloat32(wRow2[k+3]) +
				xk4*BFloat16ToFloat32(wRow2[k+4]) + xk5*BFloat16ToFloat32(wRow2[k+5]) + xk6*BFloat16ToFloat32(wRow2[k+6]) + xk7*BFloat16ToFloat32(wRow2[k+7])

			dot3 += xk0*BFloat16ToFloat32(wRow3[k]) + xk1*BFloat16ToFloat32(wRow3[k+1]) + xk2*BFloat16ToFloat32(wRow3[k+2]) + xk3*BFloat16ToFloat32(wRow3[k+3]) +
				xk4*BFloat16ToFloat32(wRow3[k+4]) + xk5*BFloat16ToFloat32(wRow3[k+5]) + xk6*BFloat16ToFloat32(wRow3[k+6]) + xk7*BFloat16ToFloat32(wRow3[k+7])
		}
		for ; k < inDim; k++ {
			xk := x[k]
			dot0 += xk * BFloat16ToFloat32(wRow0[k])
			dot1 += xk * BFloat16ToFloat32(wRow1[k])
			dot2 += xk * BFloat16ToFloat32(wRow2[k])
			dot3 += xk * BFloat16ToFloat32(wRow3[k])
		}

		out[j] = dot0
		out[j+1] = dot1
		out[j+2] = dot2
		out[j+3] = dot3
	}

	for ; j < outDim; j++ {
		wRow := weight[j*inDim : (j+1)*inDim]
		var dot float32
		if bias != nil {
			dot = bias[j]
		}
		for k := 0; k < inDim; k++ {
			dot += x[k] * BFloat16ToFloat32(wRow[k])
		}
		out[j] = dot
	}
}

// LayerNormScalar computes out = (x - mean) / sqrt(variance + eps) * weight + bias
func LayerNormScalar(x []float32, weight, bias, out []float32, dim int, eps float32) {
	var sum float32
	for i := 0; i < dim; i++ {
		sum += x[i]
	}
	mean := sum / float32(dim)

	var sumSqDiff float32
	for i := 0; i < dim; i++ {
		diff := x[i] - mean
		sumSqDiff += diff * diff
	}
	variance := sumSqDiff / float32(dim)
	invStd := float32(1.0 / math.Sqrt(float64(variance+eps)))

	for i := 0; i < dim; i++ {
		out[i] = (x[i]-mean)*invStd*weight[i] + bias[i]
	}
}

const (
	sqrt2OverPi = float32(0.7978845608028654) // sqrt(2/pi)
	geluCoeff   = float32(0.044715)
	invSqrt2    = float32(0.7071067811865475)
)

// GELUScalar computes elementwise GELU activation on x and stores in out.
// Exact erf: x * 0.5 * (1 + erf(x / sqrt(2)))
func GELUScalar(x, out []float32, n int) {
	for i := 0; i < n; i++ {
		val := x[i]
		out[i] = float32(float64(val) * 0.5 * (1.0 + math.Erf(float64(val*invSqrt2))))
	}
}

// GELUApproxScalar computes fast polynomial tanh approximation:
// 0.5x * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))
func GELUApproxScalar(x, out []float32, n int) {
	for i := 0; i < n; i++ {
		val := x[i]
		inner := sqrt2OverPi * (val + geluCoeff*val*val*val)
		out[i] = float32(0.5 * float64(val) * (1.0 + math.Tanh(float64(inner))))
	}
}

// SoftmaxScalar computes inplace or out softmax over a 1D slice of length n.
func SoftmaxScalar(scores, out []float32, n int) {
	maxVal := scores[0]
	for i := 1; i < n; i++ {
		if scores[i] > maxVal {
			maxVal = scores[i]
		}
	}

	var sumExp float32
	for i := 0; i < n; i++ {
		expVal := float32(math.Exp(float64(scores[i] - maxVal)))
		out[i] = expVal
		sumExp += expVal
	}

	invSum := 1.0 / sumExp
	for i := 0; i < n; i++ {
		out[i] *= invSum
	}
}

// MeanPoolScalar averages token representations for non-padded tokens.
func MeanPoolScalar(hiddenStates []float32, mask []int8, out []float32, seqLen, dim int) {
	for i := 0; i < dim; i++ {
		out[i] = 0
	}

	var validCount float32
	for t := 0; t < seqLen; t++ {
		if mask != nil && mask[t] == 0 {
			continue
		}
		validCount++
		tOffset := t * dim
		for i := 0; i < dim; i++ {
			out[i] += hiddenStates[tOffset+i]
		}
	}

	if validCount > 0 {
		invCount := 1.0 / validCount
		for i := 0; i < dim; i++ {
			out[i] *= invCount
		}
	}
}

// L2NormalizeScalar normalizes vector to unit length.
func L2NormalizeScalar(v, out []float32, dim int) {
	var sumSq float32
	for i := 0; i < dim; i++ {
		sumSq += v[i] * v[i]
	}
	norm := float32(math.Sqrt(math.Max(float64(sumSq), 1e-12)))
	invNorm := 1.0 / norm
	for i := 0; i < dim; i++ {
		out[i] = v[i] * invNorm
	}
}

// CosineSimilarityScalar computes cosine similarity between two vectors.
func CosineSimilarityScalar(a, b []float32, dim int) float32 {
	var dot, nA, nB float32
	for i := 0; i < dim; i++ {
		dot += a[i] * b[i]
		nA += a[i] * a[i]
		nB += b[i] * b[i]
	}
	if nA == 0 || nB == 0 {
		return 0
	}
	return dot / float32(math.Sqrt(float64(nA))*math.Sqrt(float64(nB)))
}
