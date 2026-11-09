// B4: WebGPU compute (WGSL). float32 only (WGSL has no f64).
// Phases are separated with onSubmittedWorkDone(): upload | dispatch | readback.
import { now } from "../util.js";

const MATMUL_WGSL = /* wgsl */ `
struct Params { n: u32 };
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;
const TS: u32 = 16u;
var<workgroup> tA: array<f32, 256>;
var<workgroup> tB: array<f32, 256>;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let n = p.n;
  let row = gid.y;
  let col = gid.x;
  var acc = 0.0;
  let tiles = (n + TS - 1u) / TS;
  for (var t = 0u; t < tiles; t = t + 1u) {
    let aCol = t * TS + lid.x;
    let bRow = t * TS + lid.y;
    tA[lid.y * TS + lid.x] = select(0.0, A[row * n + aCol], row < n && aCol < n);
    tB[lid.y * TS + lid.x] = select(0.0, B[bRow * n + col], bRow < n && col < n);
    workgroupBarrier();
    for (var k = 0u; k < TS; k = k + 1u) { acc = acc + tA[lid.y * TS + k] * tB[k * TS + lid.x]; }
    workgroupBarrier();
  }
  if (row < n && col < n) { C[row * n + col] = acc; }
}`;

const SAXPY_WGSL = /* wgsl */ `
struct P { a: f32, n: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x + gid.y * 16776960u;   // 65535 workgroups * 256 threads per row of workgroups
  if (i < p.n) { y[i] = p.a * x[i] + y[i]; }
}`;

const U = () => ({ STORAGE: GPUBufferUsage.STORAGE, DST: GPUBufferUsage.COPY_DST, SRC: GPUBufferUsage.COPY_SRC,
                   UNI: GPUBufferUsage.UNIFORM, MAPR: GPUBufferUsage.MAP_READ });

export class WebGpuBackend {
  name = "webgpu";
  cache = new Map();   // kernel -> {N, buffers...}  (only the latest size per kernel is kept)

  async init() {
    const t0 = now();
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("no WebGPU adapter");
    this.device = await adapter.requestDevice();
    const load_ms = now() - t0, t1 = now();
    const mk = (code) => this.device.createComputePipeline({
      layout: "auto", compute: { module: this.device.createShaderModule({ code }), entryPoint: "main" } });
    this.pipe = { matmul: mk(MATMUL_WGSL), saxpy: mk(SAXPY_WGSL) };
    // tiny warm-up dispatch so driver-level compilation is part of setup, not of the first real call
    const b = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE });
    await this.device.queue.onSubmittedWorkDone(); b.destroy();
    return { load_ms, setup_ms: now() - t1 };
  }

  _buf(size, usage) { return this.device.createBuffer({ size: Math.max(16, Math.ceil(size / 4) * 4), usage }); }

  _state(kernel, N) {
    const c = this.cache.get(kernel);
    if (c && c.N === N) return c;
    if (c) c.buffers.forEach((b) => b.destroy());
    const u = U(), d = this.device, pipe = this.pipe[kernel];
    let s;
    if (kernel === "matmul") {
      const bytes = N * N * 4;
      const A = this._buf(bytes, u.STORAGE | u.DST), B = this._buf(bytes, u.STORAGE | u.DST);
      const C = this._buf(bytes, u.STORAGE | u.SRC), P = this._buf(16, u.UNI | u.DST);
      const staging = this._buf(bytes, u.MAPR | u.DST);
      d.queue.writeBuffer(P, 0, new Uint32Array([N, 0, 0, 0]));
      const bind = d.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [A, B, C, P].map((b, i) => ({ binding: i, resource: { buffer: b } })) });
      s = { N, A, B, C, P, staging, bind, outBuf: C, bytes, buffers: [A, B, C, P, staging] };
    } else {
      const bytes = N * 4;
      const X = this._buf(bytes, u.STORAGE | u.DST), Y = this._buf(bytes, u.STORAGE | u.DST | u.SRC);
      const P = this._buf(16, u.UNI | u.DST), staging = this._buf(bytes, u.MAPR | u.DST);
      const bind = d.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [X, Y, P].map((b, i) => ({ binding: i, resource: { buffer: b } })) });
      s = { N, X, Y, P, staging, bind, outBuf: Y, bytes, buffers: [X, Y, P, staging] };
    }
    this.cache.set(kernel, s);
    return s;
  }

  async call(kernel, N, inp) {
    const d = this.device, q = d.queue, ph = {};
    let t = now();
    const s = this._state(kernel, N);                     // first call at a size includes allocation
    if (kernel === "matmul") { q.writeBuffer(s.A, 0, inp.A); q.writeBuffer(s.B, 0, inp.B); }
    else {
      q.writeBuffer(s.X, 0, inp.x); q.writeBuffer(s.Y, 0, inp.y);
      const dv = new DataView(new ArrayBuffer(16)); dv.setFloat32(0, inp.a, true); dv.setUint32(4, N, true);
      q.writeBuffer(s.P, 0, dv.buffer);
    }
    await q.onSubmittedWorkDone();
    ph.in_ms = now() - t; t = now();

    const enc = d.createCommandEncoder(), pass = enc.beginComputePass();
    pass.setPipeline(this.pipe[kernel]); pass.setBindGroup(0, s.bind);
    if (kernel === "matmul") pass.dispatchWorkgroups(Math.ceil(N / 16), Math.ceil(N / 16));
    else {
      const groups = Math.ceil(N / 256), wx = Math.min(groups, 65535), wy = Math.ceil(groups / wx);
      pass.dispatchWorkgroups(wx, wy);
    }
    pass.end();
    q.submit([enc.finish()]);
    await q.onSubmittedWorkDone();
    ph.exec_ms = now() - t; t = now();

    const enc2 = d.createCommandEncoder();
    enc2.copyBufferToBuffer(s.outBuf, 0, s.staging, 0, s.bytes);
    q.submit([enc2.finish()]);
    await s.staging.mapAsync(GPUMapMode.READ, 0, s.bytes);
    const out = new Float32Array(s.staging.getMappedRange(0, s.bytes).slice(0));
    s.staging.unmap();
    ph.out_ms = now() - t;
    return { out, phases: ph };
  }

  dispose() { this.cache.forEach((s) => s.buffers.forEach((b) => b.destroy())); this.cache.clear(); this.device?.destroy(); }
}
