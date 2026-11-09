// B3: SIMD128 + threads. A pool of workers shares one WebAssembly.Memory (SharedArrayBuffer).
// Needs cross-origin isolation in browsers (COOP/COEP headers, see server.js).
import { now, readBytes } from "../util.js";

const BASE = 65536, PAGE = 65536;

async function spawn(url) {
  if (typeof Worker !== "undefined") {
    const w = new Worker(url, { type: "module" });
    return { post: (m) => w.postMessage(m), on: (f) => { w.onmessage = (e) => f(e.data); }, kill: () => w.terminate() };
  }
  const { Worker: NodeWorker } = await import("node:worker_threads");
  const { fileURLToPath } = await import("node:url");
  const w = new NodeWorker(fileURLToPath(url));
  return { post: (m) => w.postMessage(m), on: (f) => w.on("message", f), kill: () => w.terminate() };
}

export class WasmThreadsBackend {
  name = "threads";
  constructor({ workers } = {}) {
    this.nWorkers = workers ?? Math.max(2, Math.min(globalThis.navigator?.hardwareConcurrency ?? 4, 8));
    this.url = new URL("../../wasm/kernels_simd_mt.wasm", import.meta.url);
    this.workerUrl = new URL("./wasm_worker.js", import.meta.url);
    this.pool = []; this.pending = new Map(); this.nextId = 1;
  }
  async init() {
    this.dispose();                      // idempotent: drop any previous pool/memory
    this.pending.clear();
    let t = now();
    const module = await WebAssembly.compile(await readBytes(this.url));
    const load_ms = now() - t;
    t = now();
    this.memory = new WebAssembly.Memory({ initial: 16, maximum: 16384, shared: true });
    await Promise.all(Array.from({ length: this.nWorkers }, async () => {
      const w = await spawn(this.workerUrl);
      await new Promise((resolve, reject) => {
        w.on((m) => {
          if (m.type === "ready") resolve();
          else if (m.type === "done" || m.type === "error") {
            const p = this.pending.get(m.id); this.pending.delete(m.id);
            m.type === "done" ? p.resolve() : p.reject(new Error(m.error));
          }
        });
        w.post({ type: "init", module, memory: this.memory });
        setTimeout(() => reject(new Error("worker init timeout")), 15000);
      });
      this.pool.push(w);
    }));
    return { load_ms, setup_ms: now() - t };
  }
  _ensure(bytes) {
    const need = Math.ceil((BASE + bytes) / PAGE), have = this.memory.buffer.byteLength / PAGE;
    if (need > have) this.memory.grow(need - have);
    return new Float32Array(this.memory.buffer);
  }
  _runAll(fn, argsFor, total) {
    const k = this.pool.length, chunk = Math.ceil(total / k), jobs = [];
    this.pool.forEach((w, i) => {
      const lo = i * chunk, hi = Math.min(total, lo + chunk);
      if (lo >= hi) return;
      const id = this.nextId++;
      jobs.push(new Promise((resolve, reject) => this.pending.set(id, { resolve, reject })));
      w.post({ type: "run", id, fn, args: argsFor(lo, hi) });
    });
    return Promise.all(jobs);
  }
  async call(kernel, N, inp) {
    const ph = { in_ms: 0, exec_ms: 0, out_ms: 0 };
    let t = now(), out;
    if (kernel === "matmul") {
      const n = N, bytes = n * n * 4, f32 = this._ensure(3 * bytes);
      const A = BASE, B = A + bytes, C = B + bytes;
      f32.set(inp.A, A / 4); f32.set(inp.B, B / 4);
      ph.in_ms = now() - t; t = now();
      await this._runAll("matmul_rows", (lo, hi) => [A, B, C, n, lo, hi], n);
      ph.exec_ms = now() - t; t = now();
      out = f32.slice(C / 4, C / 4 + n * n);
    } else if (kernel === "saxpy") {
      const bytes = N * 4, f32 = this._ensure(2 * bytes);
      const X = BASE, Y = X + bytes;
      f32.set(inp.x, X / 4); f32.set(inp.y, Y / 4);
      ph.in_ms = now() - t; t = now();
      await this._runAll("saxpy_range", (lo, hi) => [inp.a, X, Y, lo, hi], N);
      ph.exec_ms = now() - t; t = now();
      out = f32.slice(Y / 4, Y / 4 + N);
    } else throw new Error("unknown kernel " + kernel);
    ph.out_ms = now() - t;
    return { out, phases: ph };
  }
  dispose() { this.pool.forEach((w) => w.kill()); this.pool = []; }
}
