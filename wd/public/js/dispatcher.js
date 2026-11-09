// The adaptive dispatcher (M2): feature probe -> on-device calibration -> cost model -> selection -> fallback.
import { KERNELS } from "./kernels.js";
import { detectFeatures } from "./features.js";
import { fitCostModel, rankBackends, staticChoice, CHAIN, DEFAULT_STATIC_TABLE, PROBE_POLICY, shouldStopProbing } from "./costmodel.js";
import { median } from "./stats.js";
import { now } from "./util.js";
import { JsBackend } from "./backends/js.js";
import { WasmBackend } from "./backends/wasm.js";
import { WasmThreadsBackend } from "./backends/wasm_mt.js";
import { WebGpuBackend } from "./backends/webgpu.js";

export function createBackends(features, { workers } = {}) {
  const b = new Map();
  b.set("js", new JsBackend());
  if (features.wasm) b.set("wasm", new WasmBackend("wasm", "kernels_scalar.wasm"));
  if (features.wasm && features.simd) b.set("simd", new WasmBackend("simd", "kernels_simd.wasm"));
  if (features.wasm && features.simd && features.threads) b.set("threads", new WasmThreadsBackend({ workers }));
  if (features.webgpu) b.set("webgpu", new WebGpuBackend());
  return b;
}

export class Dispatcher {
  constructor(backends, features, { policy = "model", log = () => {}, staticTable = DEFAULT_STATIC_TABLE } = {}) {
    this.backends = backends; this.features = features; this.policy = policy; this.log = log;
    this.staticTable = staticTable;
    this.models = {};      // models[kernel][backend] = {coef}
    this.oneoff = {};      // oneoff[backend] = load_ms + setup_ms
    this.cold = {};        // cold[kernel][backend] = first-call penalty (ms)
    this.inited = new Set(); this.disabled = new Set();
    this.calibrated = false;
  }
  static async create(opts = {}) {
    const features = await detectFeatures();
    return new Dispatcher(createBackends(features, opts), features, opts);
  }

  get available() { return CHAIN.filter((n) => this.backends.has(n) && !this.disabled.has(n)); }
  signature() {
    const f = this.features;
    return ["wd1", f.ua, f.cores, f.simd, f.threads, f.webgpu, f.gpu?.description ?? ""].join("|");
  }

  async _init(name) {
    if (this.inited.has(name)) return;
    const r = await this.backends.get(name).init();
    this.oneoff[name] = r.load_ms + r.setup_ms;
    this.inited.add(name);
    this.log(`init ${name}: load ${r.load_ms.toFixed(1)} ms, setup ${r.setup_ms.toFixed(1)} ms`);
  }

  /** Short on-device probe of every available back-end at a few small sizes. */
  async calibrate({ kernels = Object.keys(KERNELS), repeats = PROBE_POLICY.repeats, useCache = true, policy = { ...PROBE_POLICY, repeats } } = {}) {
    const t0 = now(), key = "wd-calib:" + this.signature();
    if (useCache) {
      try {
        const c = JSON.parse(globalThis.localStorage?.getItem(key) ?? "null");
        if (c) { Object.assign(this, { models: c.models, oneoff: c.oneoff, cold: c.cold, calibrated: true });
          this.log("calibration loaded from cache"); return { fromCache: true, overhead_ms: 0 }; }
      } catch {}
    }
    for (const name of this.available) {
      try { await this._init(name); } catch (e) { this.disabled.add(name); this.log(`init ${name} failed: ${e}`); }
    }
    for (const kernel of kernels) {
      const spec = KERNELS[kernel];
      this.models[kernel] = {}; this.cold[kernel] = {};
      for (const name of this.available) {
        const b = this.backends.get(name), pts = [];
        let spent = 0;
        try {
          for (let i = 0; i < spec.probeLadder.length; i++) {
            const N = spec.probeLadder[i], inp = spec.makeInputs(N);
            let t = now(); await b.call(kernel, N, inp); const first = now() - t; spent += first;
            const ts = [];
            for (let r = 0; r < repeats; r++) { t = now(); await b.call(kernel, N, inp); const d = now() - t; ts.push(d); spent += d; }
            const med = median(ts);
            if (i === 0) this.cold[kernel][name] = Math.max(0, first - med);
            pts.push({ N, t: med });
            if (shouldStopProbing(pts.length, med, spent, policy)) break;
          }
          this.models[kernel][name] = fitCostModel(pts, spec.features);
          this.log(`  ${kernel}/${name}: probed ${pts.length} sizes up to N=${pts[pts.length - 1].N} (${spent.toFixed(0)} ms)`);
        } catch (e) { this.disabled.add(name); this.log(`calibration of ${name} failed: ${e}`); }
      }
    }
    this.calibrated = true;
    try { globalThis.localStorage?.setItem(key, JSON.stringify({ models: this.models, oneoff: this.oneoff, cold: this.cold })); } catch {}
    const overhead_ms = now() - t0;
    this.log(`calibration done in ${overhead_ms.toFixed(0)} ms`);
    return { fromCache: false, overhead_ms };
  }

  /** Ranked estimates [{name, est_ms}] for m calls at size N. */
  predict(kernel, N, m = 1) {
    if (!this.calibrated) throw new Error("call calibrate() first (or use policy 'static')");
    const oneoff = {};
    for (const n of this.available) oneoff[n] = this.inited.has(n) ? 0 : (this.oneoff[n] ?? 0);
    return rankBackends({ names: this.available, models: this.models[kernel] ?? {}, oneoff,
      cold: this.cold[kernel] ?? {}, N, m, featFn: KERNELS[kernel].features });
  }

  _ranking(kernel, N, m) {
    if (this.policy === "static") {
      const first = staticChoice(this.staticTable[kernel], N, this.available);
      return [{ name: first }, ...this.available.filter((n) => n !== first).map((name) => ({ name }))];
    }
    return this.predict(kernel, N, m);
  }

  /** Run a kernel on the predicted-best back-end; fall back down the ranking on failure. */
  async run(kernel, N, inputs, { m = 1 } = {}) {
    const ranking = this._ranking(kernel, N, m), fallbacks = [];
    for (const cand of ranking) {
      try {
        await this._init(cand.name);
        const t = now(), r = await this.backends.get(cand.name).call(kernel, N, inputs);
        return { ...r, backend: cand.name, total_ms: now() - t, ranking, fallbacks };
      } catch (e) {
        fallbacks.push({ name: cand.name, error: String(e) });
        this.disabled.add(cand.name);
        this.log(`${cand.name} failed (${e}); falling back`);
      }
    }
    throw new Error("all back-ends failed");
  }

  dispose() { for (const b of this.backends.values()) b.dispose?.(); }
}
