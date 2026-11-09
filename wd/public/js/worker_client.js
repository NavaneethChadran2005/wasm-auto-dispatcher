// Remote worker: runs in a browser on the device under test (e.g. a Raspberry Pi) or in Node (tests).
// Protocol (plain HTTP long-polling, works through an SSH tunnel, no extra dependencies):
//   POST /api/register {features, inits}  -> {workerId}
//   GET  /api/next?worker=ID              -> {type:'job'|'wait'|'done'|'stale', ...}   (held open up to ~15 s)
//   POST /api/result {workerId, jobId, rows | error}
import { detectFeatures } from "./features.js";
import { createBackends } from "./dispatcher.js";
import { runBenchmark } from "./bench.js";

export async function runWorker({ baseUrl = "", log = console.log, workers } = {}) {
  const api = async (path, body) => {
    const r = await fetch(baseUrl + path, body ? { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) } : undefined);
    if (!r.ok) throw new Error(`${path} -> HTTP ${r.status}`);
    return r.json();
  };

  const features = await detectFeatures();
  const backends = createBackends(features, { workers });
  const inits = {};
  for (const [name, b] of [...backends]) {
    try { inits[name] = await b.init(); log(`init ${name}: load ${inits[name].load_ms.toFixed(1)} ms, setup ${inits[name].setup_ms.toFixed(1)} ms`); }
    catch (e) { backends.delete(name); log(`init ${name} FAILED, skipping: ${e}`); }
  }
  const { workerId } = await api("/api/register", { features, inits });
  log(`registered as worker ${workerId}; back-ends: ${[...backends.keys()].join(", ")}`);

  let done = 0;
  try {
    for (;;) {
      let msg;
      try { msg = await api(`/api/next?worker=${workerId}`); }
      catch (e) { log(`poll failed (${e}); retrying in 2 s`); await new Promise((r) => setTimeout(r, 2000)); continue; }
      if (msg.type === "done" || msg.type === "stale") { log(msg.type === "done" ? "all jobs finished" : "replaced by a newer worker"); break; }
      if (msg.type !== "job") continue;
      const { job, opts } = msg;
      log(`job ${job.id}: ${job.kernel} N=${job.N}`);
      try {
        const ds = await runBenchmark(backends, { [job.kernel]: [job.N] }, { ...opts, features, inits });
        await api("/api/result", { workerId, jobId: job.id, rows: ds.results });
        done++;
      } catch (e) {
        await api("/api/result", { workerId, jobId: job.id, error: String(e) });
      }
    }
  } finally {
    for (const b of backends.values()) b.dispose?.();
  }
  return done;
}
