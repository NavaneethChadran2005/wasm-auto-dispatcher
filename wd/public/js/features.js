// Runtime capability probe. Used by the dispatcher and recorded with every dataset.
const SIMD_PROBE = new Uint8Array([0,97,115,109,1,0,0,0,1,5,1,96,0,1,123,3,2,1,0,10,10,1,8,0,65,0,253,15,253,98,11]);

export async function detectFeatures() {
  const nav = globalThis.navigator ?? {};
  const f = {
    ua: nav.userAgent ?? "node",
    cores: nav.hardwareConcurrency ?? 4,
    deviceMemoryGB: nav.deviceMemory ?? null,
    secureContext: globalThis.isSecureContext ?? true,          // false on plain http://<lan-ip>: WebGPU + threads are disabled
    crossOriginIsolated: globalThis.crossOriginIsolated ?? true, // Node has no such restriction
    wasm: typeof WebAssembly === "object",
    simd: false,
    threads: false,
    webgpu: false,
    gpu: null,
  };
  try { f.simd = WebAssembly.validate(SIMD_PROBE); } catch {}
  try {
    // Shared memory + Atomics + isolation + a way to start workers.
    new WebAssembly.Memory({ initial: 1, maximum: 1, shared: true });
    f.threads = typeof SharedArrayBuffer !== "undefined" && typeof Atomics !== "undefined" &&
      f.crossOriginIsolated && f.cores > 1;
  } catch {}
  try {
    if (nav.gpu) {
      const adapter = await nav.gpu.requestAdapter();
      if (adapter) {
        f.webgpu = true;
        const info = adapter.info ?? (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : {});
        f.gpu = { vendor: info.vendor ?? "", architecture: info.architecture ?? "", description: info.description ?? "" };
      }
    }
  } catch {}
  return f;
}
