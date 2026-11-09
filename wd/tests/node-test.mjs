// Runs in Node 20+: correctness of JS / scalar / SIMD / threaded kernels, cost model, dispatcher.
// (WebGPU needs a browser and is not exercised here.)
import { detectFeatures } from "../public/js/features.js";
import { createBackends, Dispatcher } from "../public/js/dispatcher.js";
import { KERNELS } from "../public/js/kernels.js";
import { relMaxError } from "../public/js/util.js";
import { fitCostModel, predictCall } from "../public/js/costmodel.js";

let failed = 0;
const ok = (cond, msg) => { console.log((cond ? "PASS " : "FAIL ") + msg); if (!cond) failed++; };

const features = await detectFeatures();
console.log("features:", JSON.stringify({ simd: features.simd, threads: features.threads, webgpu: features.webgpu, cores: features.cores }));
ok(features.simd, "SIMD detected");
// Force-enable threads even on a 1-core CI box so the worker pool is always tested.
const backends = createBackends({ ...features, threads: true }, { workers: 4 });
for (const b of backends.values()) await b.init();
console.log("backends:", [...backends.keys()].join(", "));

// 1. correctness vs JS reference, including sizes that are not multiples of 4 (tail loops)
for (const [kernel, sizes] of [["matmul", [1, 5, 37, 100]], ["saxpy", [1, 3, 1001, 65537]]]) {
  for (const N of sizes) {
    const inp = KERNELS[kernel].makeInputs(N);
    const ref = (await backends.get("js").call(kernel, N, inp)).out;
    for (const [name, b] of backends) {
      const out = (await b.call(kernel, N, inp)).out;
      const e = relMaxError(out, ref);
      ok(e < 1e-4 && out.length === ref.length, `${kernel} N=${N} ${name} (rel err ${e.toExponential(1)})`);
    }
  }
}

// 2. cost-model fit recovers known coefficients from noiseless data
const f = (n) => [1, n ** 3, n ** 2], truth = [2, 3e-6, 4e-5];
const pts = [64, 96, 128, 192].map((N) => ({ N, t: f(N).reduce((s, v, i) => s + v * truth[i], 0) }));
const model = fitCostModel(pts, f);
const pred = predictCall(model, 1024, f), want = f(1024).reduce((s, v, i) => s + v * truth[i], 0);
ok(Math.abs(pred - want) / want < 1e-3, `cost model extrapolates (pred ${pred.toFixed(1)} vs ${want.toFixed(1)})`);

// 3. dispatcher: calibrate, rank, run, fallback
const d = new Dispatcher(backends, features, { log: () => {} });
const c = await d.calibrate({ useCache: false, repeats: 2 });
console.log(`calibration overhead: ${c.overhead_ms.toFixed(0)} ms`);
const r1 = d.predict("matmul", 64, 1), r2 = d.predict("matmul", 512, 100);
console.log("matmul n=64  m=1  ->", r1.map((x) => `${x.name}:${x.est_ms.toFixed(1)}`).join("  "));
console.log("matmul n=512 m=100->", r2.map((x) => `${x.name}:${x.est_ms.toFixed(0)}`).join("  "));
ok(r1.length === backends.size, "ranking contains all available back-ends");
const inp = KERNELS.matmul.makeInputs(128);
const run = await d.run("matmul", 128, inp, { m: 10 });
ok(relMaxError(run.out, (await backends.get("js").call("matmul", 128, inp)).out) < 1e-4, `dispatcher run chose '${run.backend}' and result is correct`);
// forced failure -> fallback
const broken = { init: async () => ({ load_ms: 0, setup_ms: 0 }), call: async () => { throw new Error("boom"); }, dispose() {} };
d.backends.set("simd", broken); d.inited.add("simd");
const rf = await d.run("matmul", 128, inp, { m: 10000 });
ok(rf.backend !== "simd" || rf.fallbacks.length === 0, "fallback works when the chosen back-end throws");
d.dispose();
console.log(failed ? `\n${failed} test(s) FAILED` : "\nall tests passed");
process.exit(failed ? 1 : 0);
