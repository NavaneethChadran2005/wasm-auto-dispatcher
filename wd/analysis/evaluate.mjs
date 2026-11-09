#!/usr/bin/env node
// Offline evaluation of the selection models against measured datasets (RQ3 / experiment E2-E3).
//   node analysis/evaluate.mjs dataA.json dataB.json [more.json ...]
//
// For every device, kernel, size N and expected call count m, the ground truth latency of a back-end is
//     L = load + setup + first_call + (m-1) * median_call          (measured)
// The ORACLE picks min L. Regret = L(chosen) / L(oracle).
//   M0  static table      : thresholds taken from a DIFFERENT device (the "one reference machine" approach)
//   M1  fitted model      : cost model fitted on all OTHER sizes of the same device (leave-one-size-out)
//   M2  calibrated model  : cost model fitted ONLY on a time-targeted probe ladder (what the dispatcher does on-device;
//                           probes are capped below the evaluated size N, i.e. an idealised, slightly optimistic variant)
//   best-fixed            : the single back-end with the lowest mean regret (no adaptivity)
// Only sizes larger than the probe sizes are evaluated, so M2 is always extrapolating.
import fs from "node:fs";
import path from "node:path";
import { KERNELS } from "../public/js/kernels.js";
import { fitCostModel, predictCall, CHAIN, PROBE_POLICY, shouldStopProbing } from "../public/js/costmodel.js";

const files = process.argv.slice(2);
if (!files.length) { console.error("usage: node analysis/evaluate.mjs <dataset.json> [...]"); process.exit(1); }
const M_VALUES = [1, 10, 100, 1000];

// ---- load -------------------------------------------------------------------------------------------------
const devices = files.map((f) => {
  const d = JSON.parse(fs.readFileSync(f, "utf8")), tab = {};
  for (const r of d.results) {
    if (r.error || r.correct === false || r.median_ms == null) continue;
    ((tab[r.kernel] ??= {})[r.N] ??= {})[r.backend] = { first: r.first_ms, med: r.median_ms };
  }
  return { name: path.basename(f, ".json"), inits: d.inits, tab, synthetic: !!d.synthetic };
});
if (devices.some((d) => d.synthetic)) console.log("NOTE: synthetic data detected - results below test the pipeline only.\n");
if (devices.length < 2) console.log("NOTE: only one dataset given; M0 will use the same device as reference (optimistic).\n");

const oneoff = (dev, b) => (dev.inits[b]?.load_ms ?? 0) + (dev.inits[b]?.setup_ms ?? 0);
const truth = (dev, kernel, N, b, m) => { const c = dev.tab[kernel][N][b]; return oneoff(dev, b) + c.first + (m - 1) * c.med; };
const sizesOf = (dev, kernel) => Object.keys(dev.tab[kernel] ?? {}).map(Number).sort((a, b) => a - b);
const argmin = (obj) => Object.entries(obj).sort((a, b) => a[1] - b[1])[0][0];

function fitAll(dev, kernel, trainSizes, backends) {
  const spec = KERNELS[kernel], models = {}, cold = {};
  for (const b of backends) {
    const pts = trainSizes.filter((N) => dev.tab[kernel][N]?.[b]).map((N) => ({ N, t: dev.tab[kernel][N][b].med }));
    if (pts.length < 2) continue;
    models[b] = fitCostModel(pts, spec.features);
    const c = dev.tab[kernel][pts[0].N][b];
    cold[b] = Math.max(0, c.first - c.med);
  }
  return { models, cold };
}
function fitProbed(dev, kernel, N, backends) {          // simulate the dispatcher's calibration for target size N
  const spec = KERNELS[kernel], models = {}, cold = {};
  for (const b of backends) {
    const pts = []; let spent = 0;
    for (const P of spec.probeLadder.filter((x) => x < N && dev.tab[kernel][x]?.[b])) {
      const c = dev.tab[kernel][P][b];
      spent += c.first + PROBE_POLICY.repeats * c.med;
      if (!pts.length) cold[b] = Math.max(0, c.first - c.med);
      pts.push({ N: P, t: c.med });
      if (shouldStopProbing(pts.length, c.med, spent)) break;
    }
    if (pts.length >= 2) models[b] = fitCostModel(pts, spec.features);
  }
  return { models, cold };
}
function pickModel(dev, kernel, N, m, { models, cold }) {
  const f = KERNELS[kernel].features, est = {};
  for (const b of Object.keys(models)) if (dev.tab[kernel][N][b]) est[b] = oneoff(dev, b) + cold[b] + m * predictCall(models[b], N, f);
  return argmin(est);
}
function pickStatic(ref, dev, kernel, N, m) {                  // nearest reference size -> reference oracle
  const rs = sizesOf(ref, kernel);
  const near = rs.reduce((a, b) => (Math.abs(Math.log(b / N)) < Math.abs(Math.log(a / N)) ? b : a));
  const want = argmin(Object.fromEntries(Object.keys(ref.tab[kernel][near]).map((b) => [b, truth(ref, kernel, near, b, m)])));
  const avail = Object.keys(dev.tab[kernel][N]);
  for (let i = Math.max(0, CHAIN.indexOf(want)); i < CHAIN.length; i++) if (avail.includes(CHAIN[i])) return CHAIN[i];
  return avail[0];
}

// ---- evaluate ---------------------------------------------------------------------------------------------
const regrets = {};            // regrets[model] = [ {r, kernel, m} ]
const push = (model, r, kernel, m, dev) => (regrets[model] ??= []).push({ r, kernel, m, dev });
const fixed = {};              // fixed[backend] = [regret...]

for (const dev of devices) {
  const ref = devices.find((d) => d !== dev) ?? dev;
  for (const kernel of Object.keys(KERNELS)) {
    if (!dev.tab[kernel]) continue;
    const spec = KERNELS[kernel], sizes = sizesOf(dev, kernel);
    const probes = spec.probeSizes.filter((N) => sizes.includes(N));
    const evalSizes = sizes.filter((N) => N > Math.max(...probes));
    if (probes.length < 2 || !evalSizes.length) continue;
    const backends = [...new Set(sizes.flatMap((N) => Object.keys(dev.tab[kernel][N])))];
    for (const N of evalSizes) {
      const m1 = fitAll(dev, kernel, sizes.filter((s) => s !== N), backends);
      const m2 = fitProbed(dev, kernel, N, backends);
      for (const m of M_VALUES) {
        const L = Object.fromEntries(Object.keys(dev.tab[kernel][N]).map((b) => [b, truth(dev, kernel, N, b, m)]));
        const best = Math.min(...Object.values(L));
        push("M0 static table", L[pickStatic(ref, dev, kernel, N, m)] / best, kernel, m, dev.name);
        push("M1 fitted (leave-one-size-out)", L[pickModel(dev, kernel, N, m, m1)] / best, kernel, m, dev.name);
        push("M2 on-device calibration", L[pickModel(dev, kernel, N, m, m2)] / best, kernel, m, dev.name);
        for (const b of Object.keys(L)) ((fixed[`${dev.name}/${kernel}/${b}`] ??= [])).push(L[b] / best);
      }
    }
  }
}
// best single fixed back-end per device+kernel (lowest mean regret), pooled
const bestFixed = {};
for (const [key, arr] of Object.entries(fixed)) { const [dv, k] = key.split("/"); const id = dv + "/" + k; const mean = arr.reduce((a, b) => a + b, 0) / arr.length; if (!bestFixed[id] || mean < bestFixed[id].mean) bestFixed[id] = { mean, arr }; }
regrets["best single fixed back-end"] = Object.values(bestFixed).flatMap((x) => x.arr.map((r) => ({ r })));

const q = (a, p) => { const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(p * s.length))]; };
const fmt = (x) => x.toFixed(3).padStart(7);
console.log("model".padEnd(34) + " cases  median    mean     p90     max   optimal%");
for (const [model, arr] of Object.entries(regrets)) {
  const r = arr.map((x) => x.r);
  console.log(model.padEnd(34) + String(r.length).padStart(5) + fmt(q(r, 0.5)) + fmt(r.reduce((a, b) => a + b, 0) / r.length) + fmt(q(r, 0.9)) + fmt(Math.max(...r)) + (100 * r.filter((x) => x < 1.0001).length / r.length).toFixed(1).padStart(9));
}
console.log("\nregret = latency of chosen back-end / latency of oracle (1.000 = perfect).  m = expected calls: " + M_VALUES.join(", "));
console.log("\nmean regret by m (M2 on-device calibration):");
for (const m of M_VALUES) { const r = regrets["M2 on-device calibration"].filter((x) => x.m === m).map((x) => x.r); console.log("  m=" + String(m).padEnd(5) + (r.reduce((a, b) => a + b, 0) / r.length).toFixed(3)); }
