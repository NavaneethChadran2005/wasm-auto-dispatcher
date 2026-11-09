/* Freestanding WebAssembly kernels (no libc, no Emscripten glue).
 * Pointers are byte offsets into the linear memory that JavaScript owns.
 * The same source is built three ways (see Makefile): scalar, SIMD128,
 * and SIMD128 + shared memory (used by the Web Worker thread pool).
 * IMPORTANT: keep these functions free of stack use (no address-taken locals,
 * no calls that spill), because all worker threads share one linear memory.
 */
#include <stdint.h>
#ifdef __wasm_simd128__
#include <wasm_simd128.h>
#endif

#define EXPORT __attribute__((visibility("default")))

/* C[r0:r1, :] = A[r0:r1, :] x B, all n x n row-major float32.
 * i-k-j loop order: the inner loop streams rows of B and C (cache friendly,
 * vectorisable). Same algorithm as the JS and WebGPU versions. */
EXPORT void matmul_rows(const float *A, const float *B, float *C, int n, int r0, int r1) {
  for (int i = r0; i < r1; i++) {
    float *c = C + (long)i * n;
    for (int j = 0; j < n; j++) c[j] = 0.0f;
    for (int k = 0; k < n; k++) {
      const float a = A[(long)i * n + k];
      const float *b = B + (long)k * n;
      int j = 0;
#ifdef __wasm_simd128__
      const v128_t va = wasm_f32x4_splat(a);
      for (; j + 4 <= n; j += 4) {
        v128_t vc = wasm_v128_load(c + j);
        v128_t vb = wasm_v128_load(b + j);
        wasm_v128_store(c + j, wasm_f32x4_add(vc, wasm_f32x4_mul(va, vb)));
      }
#endif
      for (; j < n; j++) c[j] += a * b[j];
    }
  }
}

/* y[i0:i1] = a * x[i0:i1] + y[i0:i1]   (memory-bound kernel) */
EXPORT void saxpy_range(float a, const float *x, float *y, int i0, int i1) {
  int i = i0;
#ifdef __wasm_simd128__
  const v128_t va = wasm_f32x4_splat(a);
  for (; i + 4 <= i1; i += 4) {
    v128_t vx = wasm_v128_load(x + i);
    v128_t vy = wasm_v128_load(y + i);
    wasm_v128_store(y + i, wasm_f32x4_add(wasm_f32x4_mul(va, vx), vy));
  }
#endif
  for (; i < i1; i++) y[i] = a * x[i] + y[i];
}
