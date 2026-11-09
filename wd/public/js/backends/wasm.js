// B1 (scalar) and B2 (SIMD128): single-threaded WebAssembly built from src/kernels.c.
// JS owns the linear memory; kernels receive byte offsets.
import { now, readBytes } from "../util.js";

const BASE = 65536;                 // keep the first page free (stack/zero page)
const PAGE = 65536;

export class WasmBackend {
  constructor(name, file) {
    this.name = name;
    this.url = new URL(`../../wasm/${file}`, import.meta.url);
  }
  async init() {
    let t = now();
    const bytes = await readBytes(this.url);
    const module = await WebAssembly.compile(bytes);
    const load_ms = now() - t;
    t = now();
    this.memory = new WebAssembly.Memory({ initial: 16, maximum: 16384 });
    this.ex = (await WebAssembly.instantiate(module, { env: { memory: this.memory } })).exports;
    return { load_ms, setup_ms: now() - t };
  }
  _ensure(bytes) {
    const need = Math.ceil((BASE + bytes) / PAGE);
    const have = this.memory.buffer.byteLength / PAGE;
    if (need > have) this.memory.grow(need - have);
    return new Float32Array(this.memory.buffer);
  }
  async call(kernel, N, inp) {
    const ph = { in_ms: 0, exec_ms: 0, out_ms: 0 };
    let t = now(), out;
    if (kernel === "matmul") {
      const n = N, bytes = n * n * 4;
      const f32 = this._ensure(3 * bytes);
      const A = BASE, B = A + bytes, C = B + bytes;
      f32.set(inp.A, A / 4); f32.set(inp.B, B / 4);
      ph.in_ms = now() - (t); t = now();
      this.ex.matmul_rows(A, B, C, n, 0, n);
      ph.exec_ms = now() - t; t = now();
      out = f32.slice(C / 4, C / 4 + n * n);
    } else if (kernel === "saxpy") {
      const bytes = N * 4;
      const f32 = this._ensure(2 * bytes);
      const X = BASE, Y = X + bytes;
      f32.set(inp.x, X / 4); f32.set(inp.y, Y / 4);
      ph.in_ms = now() - t; t = now();
      this.ex.saxpy_range(inp.a, X, Y, 0, N);
      ph.exec_ms = now() - t; t = now();
      out = f32.slice(Y / 4, Y / 4 + N);
    } else throw new Error("unknown kernel " + kernel);
    ph.out_ms = now() - t;
    return { out, phases: ph };
  }
  dispose() {}
}
