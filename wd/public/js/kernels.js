// Kernel specifications: sizes, work/bytes models, cost-model features, inputs.
// Pure data + functions, no DOM, so the Node analysis scripts can import it.
import { rng } from "./util.js";

export const KERNELS = {
  matmul: {
    label: "Matrix multiplication (n x n, float32)",
    sizeLabel: "n",
    defaultSizes: [64, 96, 128, 192, 256, 512, 1024],
    probeLadder: [64, 96, 128, 192, 256, 384, 512, 768, 1024],   // calibration climbs this ladder
    probeSizes: [64, 96, 128, 192],                               // minimum probe set
    flops: (n) => 2 * n * n * n,
    work: (n) => n * n * n,
    bytes: (n) => 3 * n * n * 4,
    // cost-model features: [constant, compute term (n^3), data-movement term (n^2)]
    features: (n) => [1, n * n * n, n * n],
    makeInputs(n, seed = 1) {
      const r = rng(seed), A = new Float32Array(n * n), B = new Float32Array(n * n);
      for (let i = 0; i < A.length; i++) { A[i] = r(); B[i] = r(); }
      return { A, B };
    },
  },
  saxpy: {
    label: "SAXPY  y = a*x + y  (memory-bound, float32)",
    sizeLabel: "N",
    defaultSizes: [2 ** 12, 2 ** 14, 2 ** 16, 2 ** 18, 2 ** 20, 2 ** 22, 2 ** 24],
    probeLadder: [2 ** 12, 2 ** 14, 2 ** 16, 2 ** 18, 2 ** 20, 2 ** 21, 2 ** 22],
    probeSizes: [2 ** 12, 2 ** 14, 2 ** 16, 2 ** 18],
    flops: (N) => 2 * N,
    work: (N) => N,
    bytes: (N) => 12 * N,
    features: (N) => [1, N],                  // bytes moved is proportional to N
    makeInputs(N, seed = 2) {
      const r = rng(seed), x = new Float32Array(N), y = new Float32Array(N);
      for (let i = 0; i < N; i++) { x[i] = r(); y[i] = r(); }
      return { a: 1.5, x, y };
    },
  },
};
