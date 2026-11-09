// Generates SYNTHETIC datasets (made-up cost parameters + noise) in the same JSON schema as the
// browser harness. Purpose: test the evaluation pipeline before you have real measurements.
// These numbers are NOT measurements and must never appear in the report as results.
import fs from "node:fs";
import { rng } from "../public/js/util.js";
import { KERNELS } from "../public/js/kernels.js";

const DEVICES = {
  A: { // "desktop": many cores, discrete GPU
    cores: 12,
    inits: { js: [0, 0], wasm: [3, 0.3], simd: [3, 0.3], threads: [3, 25], webgpu: [80, 40] },
    cold:  { js: 5, wasm: 0.5, simd: 0.5, threads: 2, webgpu: 30 },
    matmul: { js: [0.02, 1.4e-6, 0], wasm: [0.05, 0.95e-6, 1.2e-6], simd: [0.05, 0.30e-6, 1.2e-6], threads: [0.8, 0.045e-6, 1.2e-6], webgpu: [4, 0.002e-6, 6e-6] },
    saxpy:  { js: [0.01, 1.1e-6], wasm: [0.03, 1.0e-6], simd: [0.03, 0.8e-6], threads: [0.8, 0.55e-6], webgpu: [4, 1.6e-6] },
  },
  B: { // "laptop": 4 cores, integrated GPU
    cores: 4,
    inits: { js: [0, 0], wasm: [4, 0.4], simd: [4, 0.4], threads: [4, 15], webgpu: [120, 60] },
    cold:  { js: 6, wasm: 0.6, simd: 0.6, threads: 3, webgpu: 45 },
    matmul: { js: [0.03, 1.8e-6, 0], wasm: [0.06, 1.2e-6, 1.6e-6], simd: [0.06, 0.45e-6, 1.6e-6], threads: [1.2, 0.13e-6, 1.6e-6], webgpu: [6, 0.02e-6, 1.5e-5] },
    saxpy:  { js: [0.02, 1.5e-6], wasm: [0.04, 1.4e-6], simd: [0.04, 1.1e-6], threads: [1.2, 0.9e-6], webgpu: [6, 2.4e-6] },
  },
};
const SIZES = { matmul: [64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048], saxpy: KERNELS.saxpy.defaultSizes };

const r = rng(7);
const noise = (s = 0.03) => Math.exp(s * Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r()));

for (const [dev, P] of Object.entries(DEVICES)) {
  const ds = { schema: 1, synthetic: true, createdAt: "synthetic", env: { ua: `SYNTHETIC device ${dev}`, cores: P.cores },
    config: { synthetic: true },
    inits: Object.fromEntries(Object.entries(P.inits).map(([k, [l, s]]) => [k, { load_ms: l, setup_ms: s }])), results: [] };
  for (const kernel of ["matmul", "saxpy"]) {
    for (const N of SIZES[kernel]) {
      for (const b of Object.keys(P.inits)) {
        const [a, be, g = 0] = P[kernel][b];
        const t = (a + be * KERNELS[kernel].work(N) + (kernel === "matmul" ? g * N * N : 0)) * noise();
        ds.results.push({ kernel, N, backend: b, first_ms: t + P.cold[b] * noise(0.1), median_ms: t, reps: 30, correct: true, rel_error: 1e-7 });
      }
    }
  }
  fs.writeFileSync(new URL(`./sample-data/synthetic_device_${dev}.json`, import.meta.url), JSON.stringify(ds));
  console.log(`wrote synthetic_device_${dev}.json (${ds.results.length} rows)`);
}
