// B0: tuned JavaScript baseline (flattened Float32Array, i-k-j loop order).
import { now } from "../util.js";

export class JsBackend {
  name = "js";
  async init() { return { load_ms: 0, setup_ms: 0 }; }
  async call(kernel, N, inp) {
    const t0 = now();
    let out;
    if (kernel === "matmul") {
      const n = N, { A, B } = inp;
      out = new Float32Array(n * n);
      for (let i = 0; i < n; i++) {
        const ci = i * n;
        for (let k = 0; k < n; k++) {
          const a = A[ci + k], bk = k * n;
          for (let j = 0; j < n; j++) out[ci + j] += a * B[bk + j];
        }
      }
    } else if (kernel === "saxpy") {
      const { a, x, y } = inp;
      out = new Float32Array(N);
      for (let i = 0; i < N; i++) out[i] = a * x[i] + y[i];
    } else throw new Error("unknown kernel " + kernel);
    return { out, phases: { in_ms: 0, exec_ms: now() - t0, out_ms: 0 } };
  }
  dispose() {}
}
