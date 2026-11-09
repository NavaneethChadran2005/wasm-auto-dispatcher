// Benchmark harness: phase-separated timing, randomised back-end order, correctness check,
// median / IQR / bootstrap CI, adaptive repetition count.
import { KERNELS } from "./kernels.js";
import { median, iqr, bootstrapCI } from "./stats.js";
import { now, shuffle, sleep0, relMaxError } from "./util.js";

export const TOLERANCE = 1e-4;      // max|out-ref| / max|ref|

/**
 * @param backends Map<name, backend>
 * @param plan     { matmul: [sizes], saxpy: [sizes] }
 * @param opts     { maxReps=30, minReps=5, warmup=2, budgetMs=4000, skipAboveMs=20000, onProgress, features }
 */
export async function runBenchmark(backends, plan, opts = {}) {
  const { maxReps = 30, minReps = 5, warmup = 2, budgetMs = 4000, skipAboveMs = 20000,
          onProgress = () => {}, features = {}, inits: preInits = null } = opts;   // preInits: back-ends already initialised
  const dataset = { schema: 1, createdAt: new Date().toISOString(), env: features,
    config: { maxReps, minReps, warmup, budgetMs, skipAboveMs, tolerance: TOLERANCE }, inits: {}, results: [] };
  const names = [...backends.keys()];
  const dead = new Set();

  if (preInits) dataset.inits = preInits;
  else for (const name of names) {
    try { dataset.inits[name] = await backends.get(name).init(); onProgress(`init ${name}: ${JSON.stringify(dataset.inits[name])}`); }
    catch (e) { dead.add(name); onProgress(`init ${name} FAILED: ${e}`); }
  }
  const reference = backends.get("wasm") ?? backends.get("js");

  for (const [kernel, sizes] of Object.entries(plan)) {
    const spec = KERNELS[kernel];
    for (const N of sizes) {
      const inputs = spec.makeInputs(N);
      const ref = (await reference.call(kernel, N, inputs)).out;        // untimed reference result
      for (const name of shuffle(names.filter((n) => !dead.has(n)))) {
        const b = backends.get(name), row = { kernel, N, backend: name };
        try {
          let t = now();
          const r0 = await b.call(kernel, N, inputs);
          row.first_ms = now() - t;
          row.rel_error = relMaxError(r0.out, ref);
          row.correct = row.rel_error < TOLERANCE;
          if (row.first_ms > skipAboveMs) { row.skipped = "first call exceeded skipAboveMs"; row.median_ms = row.first_ms; row.reps = 1; }
          else {
            const reps = Math.max(minReps, Math.min(maxReps, Math.ceil(budgetMs / Math.max(row.first_ms, 0.05))));
            const wu = row.first_ms > 2000 ? 0 : warmup;
            for (let i = 0; i < wu; i++) await b.call(kernel, N, inputs);
            const ts = [], ph = { in_ms: [], exec_ms: [], out_ms: [] };
            for (let i = 0; i < reps; i++) {
              t = now(); const r = await b.call(kernel, N, inputs); ts.push(now() - t);
              for (const k of Object.keys(ph)) ph[k].push(r.phases[k]);
              if (i % 4 === 3) await sleep0();                              // let the UI breathe
            }
            const [lo, hi] = bootstrapCI(ts);
            Object.assign(row, { reps, median_ms: median(ts), iqr_ms: iqr(ts), ci95_ms: [lo, hi], min_ms: Math.min(...ts),
              gflops: spec.flops(N) / (median(ts) * 1e6),
              phases_ms: Object.fromEntries(Object.entries(ph).map(([k, v]) => [k, median(v)])) });
          }
        } catch (e) { row.error = String(e); }
        dataset.results.push(row);
        onProgress(`${kernel} N=${N} ${name}: ${row.error ? "ERROR " + row.error : row.median_ms.toFixed(2) + " ms" + (row.correct === false ? "  !! INCORRECT" : "")}`);
        await sleep0();
      }
    }
  }
  return dataset;
}
