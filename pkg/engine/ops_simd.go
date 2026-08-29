//go:build goexperiment.simd

package engine

import (
	"math"
	"simd"
)

// HasSIMD indicates SIMD hardware acceleration is active.
const HasSIMD = true

// reduceSumFloat32s sums all elements in a SIMD vector.
func reduceSumFloat32s(v simd.Float32s) float32 {
	l := v.Len()
	var buf [64]float32
	v.Store(buf[:l])
	var sum float32
	for i := 0; i < l; i++ {
		sum += buf[i]
	}
	return sum
}

// MatVecMulAddSIMD computes out = x * W^T + bias using SIMD vector FMA operations.
// It unrolls 4 rows at a time to keep x in CPU registers and hide FMA latency across 4 independent vector accumulators.
func MatVecMulAddSIMD(x []float32, weight []float32, bias []float32, out []float32, inDim, outDim int) {
	vecLen := simd.BroadcastFloat32s(0).Len()
	vecLimit := inDim - (inDim % vecLen)

	j := 0
	for ; j <= outDim-8; j += 8 {
		wRow0 := weight[j*inDim : (j+1)*inDim]
		wRow1 := weight[(j+1)*inDim : (j+2)*inDim]
		wRow2 := weight[(j+2)*inDim : (j+3)*inDim]
		wRow3 := weight[(j+3)*inDim : (j+4)*inDim]
		wRow4 := weight[(j+4)*inDim : (j+5)*inDim]
		wRow5 := weight[(j+5)*inDim : (j+6)*inDim]
		wRow6 := weight[(j+6)*inDim : (j+7)*inDim]
		wRow7 := weight[(j+7)*inDim : (j+8)*inDim]

		acc0 := simd.BroadcastFloat32s(0)
		acc1 := simd.BroadcastFloat32s(0)
		acc2 := simd.BroadcastFloat32s(0)
		acc3 := simd.BroadcastFloat32s(0)
		acc4 := simd.BroadcastFloat32s(0)
		acc5 := simd.BroadcastFloat32s(0)
		acc6 := simd.BroadcastFloat32s(0)
		acc7 := simd.BroadcastFloat32s(0)

		k := 0
		for ; k < vecLimit; k += vecLen {
			vx := simd.LoadFloat32s(x[k:])
			vw0 := simd.LoadFloat32s(wRow0[k:])
			vw1 := simd.LoadFloat32s(wRow1[k:])
			vw2 := simd.LoadFloat32s(wRow2[k:])
			vw3 := simd.LoadFloat32s(wRow3[k:])
			vw4 := simd.LoadFloat32s(wRow4[k:])
			vw5 := simd.LoadFloat32s(wRow5[k:])
			vw6 := simd.LoadFloat32s(wRow6[k:])
			vw7 := simd.LoadFloat32s(wRow7[k:])

			acc0 = vx.MulAdd(vw0, acc0)
			acc1 = vx.MulAdd(vw1, acc1)
			acc2 = vx.MulAdd(vw2, acc2)
			acc3 = vx.MulAdd(vw3, acc3)
			acc4 = vx.MulAdd(vw4, acc4)
			acc5 = vx.MulAdd(vw5, acc5)
			acc6 = vx.MulAdd(vw6, acc6)
			acc7 = vx.MulAdd(vw7, acc7)
		}

		dot0 := reduceSumFloat32s(acc0)
		dot1 := reduceSumFloat32s(acc1)
		dot2 := reduceSumFloat32s(acc2)
		dot3 := reduceSumFloat32s(acc3)
		dot4 := reduceSumFloat32s(acc4)
		dot5 := reduceSumFloat32s(acc5)
		dot6 := reduceSumFloat32s(acc6)
		dot7 := reduceSumFloat32s(acc7)

		for ; k < inDim; k++ {
			xk := x[k]
			dot0 += xk * wRow0[k]
			dot1 += xk * wRow1[k]
			dot2 += xk * wRow2[k]
			dot3 += xk * wRow3[k]
			dot4 += xk * wRow4[k]
			dot5 += xk * wRow5[k]
			dot6 += xk * wRow6[k]
			dot7 += xk * wRow7[k]
		}

		if bias != nil {
			dot0 += bias[j]
			dot1 += bias[j+1]
			dot2 += bias[j+2]
			dot3 += bias[j+3]
			dot4 += bias[j+4]
			dot5 += bias[j+5]
			dot6 += bias[j+6]
			dot7 += bias[j+7]
		}

		out[j] = dot0
		out[j+1] = dot1
		out[j+2] = dot2
		out[j+3] = dot3
		out[j+4] = dot4
		out[j+5] = dot5
		out[j+6] = dot6
		out[j+7] = dot7
	}

	// Remaining rows
	for ; j < outDim; j++ {
		wRow := weight[j*inDim : (j+1)*inDim]
		var dot float32
		if bias != nil {
			dot = bias[j]
		}

		acc0 := simd.BroadcastFloat32s(0)
		k := 0
		for ; k < vecLimit; k += vecLen {
			vx := simd.LoadFloat32s(x[k:])
			vw := simd.LoadFloat32s(wRow[k:])
			acc0 = vx.MulAdd(vw, acc0)
		}

		dot += reduceSumFloat32s(acc0)

		for ; k < inDim; k++ {
			dot += x[k] * wRow[k]
		}

		out[j] = dot
	}
}

// MatVecMulAddINT8SIMD computes out = (x * W_int8^T) * scale + bias using SIMD vector FMA operations.
func MatVecMulAddINT8SIMD(x []float32, weight []int8, scale []float32, bias []float32, out []float32, inDim, outDim int) {
	vecLen := simd.BroadcastFloat32s(0).Len()
	vecLimit := inDim - (inDim % vecLen)

	var wBuf0, wBuf1, wBuf2, wBuf3 [64]float32

	j := 0
	for ; j <= outDim-4; j += 4 {
		wRow0 := weight[j*inDim : (j+1)*inDim]
		wRow1 := weight[(j+1)*inDim : (j+2)*inDim]
		wRow2 := weight[(j+2)*inDim : (j+3)*inDim]
		wRow3 := weight[(j+3)*inDim : (j+4)*inDim]

		acc0 := simd.BroadcastFloat32s(0)
		acc1 := simd.BroadcastFloat32s(0)
		acc2 := simd.BroadcastFloat32s(0)
		acc3 := simd.BroadcastFloat32s(0)

		k := 0
		for ; k < vecLimit; k += vecLen {
			vx := simd.LoadFloat32s(x[k:])

			for idx := 0; idx < vecLen; idx++ {
				wBuf0[idx] = float32(wRow0[k+idx])
				wBuf1[idx] = float32(wRow1[k+idx])
				wBuf2[idx] = float32(wRow2[k+idx])
				wBuf3[idx] = float32(wRow3[k+idx])
			}

			vw0 := simd.LoadFloat32s(wBuf0[:vecLen])
			vw1 := simd.LoadFloat32s(wBuf1[:vecLen])
			vw2 := simd.LoadFloat32s(wBuf2[:vecLen])
			vw3 := simd.LoadFloat32s(wBuf3[:vecLen])

			acc0 = vx.MulAdd(vw0, acc0)
			acc1 = vx.MulAdd(vw1, acc1)
			acc2 = vx.MulAdd(vw2, acc2)
			acc3 = vx.MulAdd(vw3, acc3)
		}

		dot0 := reduceSumFloat32s(acc0)
		dot1 := reduceSumFloat32s(acc1)
		dot2 := reduceSumFloat32s(acc2)
		dot3 := reduceSumFloat32s(acc3)

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

	// Remaining rows
	for ; j < outDim; j++ {
		wRow := weight[j*inDim : (j+1)*inDim]
		acc0 := simd.BroadcastFloat32s(0)
		k := 0
		for ; k < vecLimit; k += vecLen {
			vx := simd.LoadFloat32s(x[k:])
			for idx := 0; idx < vecLen; idx++ {
				wBuf0[idx] = float32(wRow[k+idx])
			}
			vw := simd.LoadFloat32s(wBuf0[:vecLen])
			acc0 = vx.MulAdd(vw, acc0)
		}

		dot := reduceSumFloat32s(acc0)
		for ; k < inDim; k++ {
			dot += x[k] * float32(wRow[k])
		}

		res := dot * scale[j]
		if bias != nil {
			res += bias[j]
		}
		out[j] = res
	}
}

// MatVecMulAddBF16SIMD computes out = x * W_bf16^T + bias using SIMD vector FMA operations.
func MatVecMulAddBF16SIMD(x []float32, weight []uint16, bias []float32, out []float32, inDim, outDim int) {
	vecLen := simd.BroadcastFloat32s(0).Len()
	vecLimit := inDim - (inDim % vecLen)

	var wBuf0, wBuf1, wBuf2, wBuf3 [64]float32

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

		acc0 := simd.BroadcastFloat32s(0)
		acc1 := simd.BroadcastFloat32s(0)
		acc2 := simd.BroadcastFloat32s(0)
		acc3 := simd.BroadcastFloat32s(0)

		k := 0
		for ; k < vecLimit; k += vecLen {
			vx := simd.LoadFloat32s(x[k:])

			for idx := 0; idx < vecLen; idx++ {
				wBuf0[idx] = BFloat16ToFloat32(wRow0[k+idx])
				wBuf1[idx] = BFloat16ToFloat32(wRow1[k+idx])
				wBuf2[idx] = BFloat16ToFloat32(wRow2[k+idx])
				wBuf3[idx] = BFloat16ToFloat32(wRow3[k+idx])
			}

			vw0 := simd.LoadFloat32s(wBuf0[:vecLen])
			vw1 := simd.LoadFloat32s(wBuf1[:vecLen])
			vw2 := simd.LoadFloat32s(wBuf2[:vecLen])
			vw3 := simd.LoadFloat32s(wBuf3[:vecLen])

			acc0 = vx.MulAdd(vw0, acc0)
			acc1 = vx.MulAdd(vw1, acc1)
			acc2 = vx.MulAdd(vw2, acc2)
			acc3 = vx.MulAdd(vw3, acc3)
		}

		dot0 += reduceSumFloat32s(acc0)
		dot1 += reduceSumFloat32s(acc1)
		dot2 += reduceSumFloat32s(acc2)
		dot3 += reduceSumFloat32s(acc3)

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

	// Remaining rows
	for ; j < outDim; j++ {
		wRow := weight[j*inDim : (j+1)*inDim]
		var dot float32
		if bias != nil {
			dot = bias[j]
		}
		acc0 := simd.BroadcastFloat32s(0)
		k := 0
		for ; k < vecLimit; k += vecLen {
			vx := simd.LoadFloat32s(x[k:])
			for idx := 0; idx < vecLen; idx++ {
				wBuf0[idx] = BFloat16ToFloat32(wRow[k+idx])
			}
			vw := simd.LoadFloat32s(wBuf0[:vecLen])
			acc0 = vx.MulAdd(vw, acc0)
		}

		dot += reduceSumFloat32s(acc0)
		for ; k < inDim; k++ {
			dot += x[k] * BFloat16ToFloat32(wRow[k])
		}

		out[j] = dot
	}
}

// LayerNormSIMD computes LayerNorm with SIMD vectorization.
func LayerNormSIMD(x []float32, weight, bias, out []float32, dim int, eps float32) {
	vecLen := simd.BroadcastFloat32s(0).Len()
	vecLimit := dim - (dim % vecLen)

	// 1. Mean
	accMean := simd.BroadcastFloat32s(0)
	var tailMean float32
	k := 0
	for ; k < vecLimit; k += vecLen {
		vx := simd.LoadFloat32s(x[k:])
		accMean = accMean.Add(vx)
	}
	for ; k < dim; k++ {
		tailMean += x[k]
	}
	mean := (reduceSumFloat32s(accMean) + tailMean) / float32(dim)

	// 2. Variance
	vMean := simd.BroadcastFloat32s(mean)
	accVar := simd.BroadcastFloat32s(0)
	var tailVar float32
	k = 0
	for ; k < vecLimit; k += vecLen {
		vx := simd.LoadFloat32s(x[k:])
		diff := vx.Sub(vMean)
		accVar = diff.MulAdd(diff, accVar)
	}
	for ; k < dim; k++ {
		d := x[k] - mean
		tailVar += d * d
	}
	variance := (reduceSumFloat32s(accVar) + tailVar) / float32(dim)
	invStd := float32(1.0 / math.Sqrt(float64(variance+eps)))

	// 3. Normalize, scale, and shift
	vInvStd := simd.BroadcastFloat32s(invStd)
	k = 0
	for ; k < vecLimit; k += vecLen {
		vx := simd.LoadFloat32s(x[k:])
		vw := simd.LoadFloat32s(weight[k:])
		vb := simd.LoadFloat32s(bias[k:])

		vDiff := vx.Sub(vMean)
		vNorm := vDiff.Mul(vInvStd)
		vOut := vNorm.MulAdd(vw, vb)
		vOut.Store(out[k:])
	}
	for ; k < dim; k++ {
		out[k] = (x[k]-mean)*invStd*weight[k] + bias[k]
	}
}

// GELUSIMD computes GELU activation using exact erf with SIMD acceleration where applicable.
func GELUSIMD(x, out []float32, n int) {
	// Standard exact erf mapping
	GELUScalar(x, out, n)
}

// GELUApproxSIMD computes fast polynomial approximation with SIMD.
func GELUApproxSIMD(x, out []float32, n int) {
	vecLen := simd.BroadcastFloat32s(0).Len()
	vecLimit := n - (n % vecLen)

	v05 := simd.BroadcastFloat32s(0.5)
	vOne := simd.BroadcastFloat32s(1.0)
	vCoeff := simd.BroadcastFloat32s(geluCoeff)
	vSqrt2Pi := simd.BroadcastFloat32s(sqrt2OverPi)

	var buf [64]float32
	k := 0
	for ; k < vecLimit; k += vecLen {
		vx := simd.LoadFloat32s(x[k:])
		// x^3 = x * x * x
		vx2 := vx.Mul(vx)
		vx3 := vx2.Mul(vx)
		// inner = sqrt(2/pi) * (x + 0.044715 * x^3)
		inner := vSqrt2Pi.Mul(vx.MulAdd(vCoeff, vx3))

		// Apply tanh to inner elements
		inner.Store(buf[:vecLen])
		for i := 0; i < vecLen; i++ {
			buf[i] = float32(math.Tanh(float64(buf[i])))
		}
		vTanh := simd.LoadFloat32s(buf[:vecLen])
		// out = 0.5 * x * (1.0 + tanh)
		vOut := v05.Mul(vx).Mul(vOne.Add(vTanh))
		vOut.Store(out[k:])
	}

	for ; k < n; k++ {
		val := x[k]
		inner := sqrt2OverPi * (val + geluCoeff*val*val*val)
		out[k] = float32(0.5 * float64(val) * (1.0 + math.Tanh(float64(inner))))
	}
}

// L2NormalizeSIMD normalizes a vector with SIMD.
func L2NormalizeSIMD(v, out []float32, dim int) {
	vecLen := simd.BroadcastFloat32s(0).Len()
	vecLimit := dim - (dim % vecLen)

	acc := simd.BroadcastFloat32s(0)
	var tailSum float32
	k := 0
	for ; k < vecLimit; k += vecLen {
		vv := simd.LoadFloat32s(v[k:])
		acc = vv.MulAdd(vv, acc)
	}
	for ; k < dim; k++ {
		tailSum += v[k] * v[k]
	}

	sumSq := reduceSumFloat32s(acc) + tailSum
	norm := float32(math.Sqrt(math.Max(float64(sumSq), 1e-12)))
	invNorm := float32(1.0) / norm
	vInvNorm := simd.BroadcastFloat32s(invNorm)

	k = 0
	for ; k < vecLimit; k += vecLen {
		vv := simd.LoadFloat32s(v[k:])
		vv.Mul(vInvNorm).Store(out[k:])
	}
	for ; k < dim; k++ {
		out[k] = v[k] * invNorm
	}
}

// CosineSimilaritySIMD computes cosine similarity between two vectors using SIMD dot products.
func CosineSimilaritySIMD(a, b []float32, dim int) float32 {
	vecLen := simd.BroadcastFloat32s(0).Len()
	vecLimit := dim - (dim % vecLen)

	accDot := simd.BroadcastFloat32s(0)
	accNA := simd.BroadcastFloat32s(0)
	accNB := simd.BroadcastFloat32s(0)
	var tailDot, tailNA, tailNB float32

	k := 0
	for ; k < vecLimit; k += vecLen {
		va := simd.LoadFloat32s(a[k:])
		vb := simd.LoadFloat32s(b[k:])

		accDot = va.MulAdd(vb, accDot)
		accNA = va.MulAdd(va, accNA)
		accNB = vb.MulAdd(vb, accNB)
	}
	for ; k < dim; k++ {
		tailDot += a[k] * b[k]
		tailNA += a[k] * a[k]
		tailNB += b[k] * b[k]
	}

	dot := reduceSumFloat32s(accDot) + tailDot
	nA := reduceSumFloat32s(accNA) + tailNA
	nB := reduceSumFloat32s(accNB) + tailNB

	if nA == 0 || nB == 0 {
		return 0
	}
	return dot / float32(math.Sqrt(float64(nA))*math.Sqrt(float64(nB)))
}
