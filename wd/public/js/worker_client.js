// Auto-benchmark client. Runs in the browser of ANY device that opens the controller's URL (or in Node for tests).
// Protocol (plain HTTP long-polling, no extra dependencies):
//   POST /api/register {features, inits, label, clientId, timerResolutionMs} -> {workerId, total}
//   GET  /api/next?worker=ID    -> {type:'job'|'wait'|'done'|'stale', ...}   (held open up to ~15 s)
//   POST /api/result {workerId, jobId, rows | error}
//   GET  /api/report?worker=ID  -> {text, done, total, finished}
import { detectFeatures } from "./features.js";
import { createBackends } from "./dispatcher.js";
import { runBenchmark } from "./bench.js";
import { measureTimerResolution } from "./util.js";

function clientId() {
  try {
    let id = globalThis.localStorage?.getItem("wd-client-id");
    if (!id) { id = Math.random().toString(36).slice(2, 10); globalThis.localStorage?.setItem("wd-client-id", id); }
    return id;
  } catch { return Math.random().toString(36).slice(2, 10); }
}

/**
 * @returns {Promise<{jobsDone:number, report:string, features:object}>}
 * options: baseUrl, log, label, workers (thread count), onProgress({done,total,text}), featuresOverride (tests only)
 */
export async function runWorker({ baseUrl = "", log = console.log, label = "", workers, onProgress = () => {}, featuresOverride = {} } = {}) {
  const api = async (path, body) => {
    const r = await fetch(baseUrl + path, body ? { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) } : undefined);
    if (!r.ok) throw new Error(`${path} -> HTTP ${r.status}`);
    return r.json();
  };

  const features = Object.assign(await detectFeatures(), featuresOverride);
  const timerResolutionMs = measureTimerResolution();
  const backends = createBackends(features, { workers });
  const inits = {};
  for (const [name, b] of [...backends]) {
    try { inits[name] = await b.init(); log(`init ${name}: load ${inits[name].load_ms.toFixed(1)} ms, setup ${inits[name].setup_ms.toFixed(1)} ms`); }
    catch (e) { backends.delete(name); log(`init ${name} FAILED, skipping: ${e}`); }
  }
  const { workerId, total } = await api("/api/register", { features, inits, label, clientId: clientId(), timerResolutionMs });
  log(`registered (${total} payloads); back-ends: ${[...backends.keys()].join(", ")}`);
  onProgress({ done: 0, total, text: "registered" });

  let jobsDone = 0;
  try {
    for (;;) {
      let msg;
      try { msg = await api(`/api/next?worker=${workerId}`); }
      catch (e) { log(`poll failed (${e}); retrying in 2 s`); await new Promise((r) => setTimeout(r, 2000)); continue; }
      if (msg.type === "done" || msg.type === "stale") break;
      if (msg.type !== "job") continue;
      const { job, opts } = msg;
      const text = `payload ${job.id}/${total}: ${job.kernel} N=${job.N}`;
      log(text); onProgress({ done: jobsDone, total, text });
      try {
        const ds = await runBenchmark(backends, { [job.kernel]: [job.N] }, { ...opts, features, inits });
        await api("/api/result", { workerId, jobId: job.id, rows: ds.results });
      } catch (e) {
        await api("/api/result", { workerId, jobId: job.id, error: String(e) });
      }
      jobsDone++;
    }
  } finally {
    for (const b of backends.values()) b.dispose?.();
  }
  let report = "";
  try { report = (await api(`/api/report?worker=${workerId}`)).text; } catch {}
  onProgress({ done: jobsDone, total, text: "finished" });
  return { jobsDone, report, features };
}
