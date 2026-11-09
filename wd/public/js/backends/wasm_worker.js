// Worker for B3 (SIMD + threads). Each worker instantiates the SAME module against the
// SAME shared memory, then runs row/element ranges on request.
let ex = null;
let post, listen;
if (typeof self !== "undefined" && typeof self.postMessage === "function" && typeof window === "undefined") {
  post = (m) => self.postMessage(m);
  listen = (f) => { self.onmessage = (e) => f(e.data); };
} else {
  const { parentPort } = await import("node:worker_threads");   // Node (used by tests)
  post = (m) => parentPort.postMessage(m);
  listen = (f) => parentPort.on("message", f);
}

listen(async (msg) => {
  try {
    if (msg.type === "init") {
      ex = (await WebAssembly.instantiate(msg.module, { env: { memory: msg.memory } })).exports;
      post({ type: "ready" });
    } else if (msg.type === "run") {
      ex[msg.fn](...msg.args);
      post({ type: "done", id: msg.id });
    }
  } catch (err) {
    post({ type: "error", id: msg.id, error: String(err) });
  }
});
