#!/usr/bin/env node
// Controller: serves the pages, queues small payloads, hands them to ONE remote worker (e.g. a Pi's browser),
// collects the timings of every back-end per payload, and writes the best back-end per payload to a text file.
//
//   node controller.js [--label pi4] [--out results] [--m 10] [--jobs jobs.json] [--quick]
//                      [--reps 15] [--budget 1500] [--exit-when-done] [--wait 600] [--port 8080]
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import { staticHandler } from "./server.js";

function parseArgs(a) {
  const o = {};
  for (let i = 0; i < a.length; i++) {
    if (!a[i].startsWith("--")) continue;
    const key = a[i].slice(2), next = a[i + 1];
    if (next !== undefined && !next.startsWith("--")) { o[key] = next; i++; } else o[key] = true;
  }
  return o;
}
const args = parseArgs(process.argv.slice(2));
const cfg = {
  port: Number(args.port ?? 8080), out: path.resolve(String(args.out ?? "results")), label: String(args.label ?? "device"),
  m: Number(args.m ?? 10), maxReps: Number(args.reps ?? 15), budgetMs: Number(args.budget ?? 1500),
  exitWhenDone: Boolean(args["exit-when-done"]), waitSec: Number(args.wait ?? 600), jobTimeoutSec: Number(args["job-timeout"] ?? 180),
};
const SMALL = [["matmul", [64, 128, 192, 256, 384, 512]], ["saxpy", [2 ** 14, 2 ** 16, 2 ** 18, 2 ** 20, 2 ** 22]]];
const QUICK = [["matmul", [64, 256]], ["saxpy", [2 ** 16, 2 ** 20]]];
const jobSpecs = args.jobs ? JSON.parse(fs.readFileSync(String(args.jobs), "utf8"))
  : (args.quick ? QUICK : SMALL).flatMap(([kernel, ns]) => ns.map((N) => ({ kernel, N })));

const state = { jobs: jobSpecs.map((j, i) => ({ id: i + 1, kernel: j.kernel, N: j.N, m: j.m ?? cfg.m, status: "queued", attempts: 0 })),
  worker: null, finished: false, startedAt: new Date() };
const waiting = [];                                    // parked long-poll requests
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);

// ---------- ranking & output -------------------------------------------------------------------------------
const num = (x, d = 3) => (x == null ? "-" : x.toFixed(d));
function summarise(job) {
  const inits = state.worker?.inits ?? {};
  const oneoff = (b) => (inits[b]?.load_ms ?? 0) + (inits[b]?.setup_ms ?? 0);
  const ok = job.rows.filter((r) => !r.error && r.correct !== false && r.median_ms != null);
  if (!ok.length) return null;
  const steady = [...ok].sort((a, b) => a.median_ms - b.median_ms);
  const total = ok.map((r) => ({ r, t: oneoff(r.backend) + r.first_ms + (job.m - 1) * r.median_ms })).sort((a, b) => a.t - b.t);
  return { steady, total, bad: job.rows.filter((r) => r.error || r.correct === false) };
}
function renderBest() {
  const w = state.worker, L = [];
  L.push("# Best back-end per payload (measured remotely)");
  L.push(`# device label : ${cfg.label}`);
  if (w) L.push(`# browser      : ${w.features.ua}`, `# cores/features: ${w.features.cores} cores, simd=${w.features.simd}, threads=${w.features.threads}, webgpu=${w.features.webgpu}, isolated=${w.features.crossOriginIsolated}`);
  L.push(`# generated    : ${new Date().toISOString()}`);
  L.push(`# steady = lowest median time per call;  total(m=${cfg.m}) = load + setup + first call + (m-1) x median`, "#");
  const head = ["kernel", "N", "best_steady", "median_ms", "GFLOP/s", `best_total(m)`, "total_ms", "all medians (ms)"];
  const rows = [head];
  let top = null;
  for (const j of state.jobs) {
    if (j.status === "failed") { rows.push([j.kernel, String(j.N), "FAILED", "", "", "", "", j.error ?? ""]); continue; }
    if (!j.summary) continue;
    const s = j.summary, b = s.steady[0], t = s.total[0];
    rows.push([j.kernel, String(j.N), b.backend, num(b.median_ms), num(b.gflops, 2), `${t.r.backend} (m=${j.m})`, num(t.t, 2),
      s.steady.map((r) => `${r.backend}=${num(r.median_ms)}`).join(" ")]);
    if (!top || b.gflops > top.gflops) top = { ...b, kernel: j.kernel, N: j.N };
  }
  const widths = head.map((_, c) => Math.max(...rows.map((r) => r[c].length)));
  for (const r of rows) L.push(r.map((x, c) => (c === rows[0].length - 1 ? x : x.padEnd(widths[c]))).join("  "));
  if (top) L.push("", `# highest throughput overall: ${top.kernel} N=${top.N} on ${top.backend} at ${num(top.gflops, 2)} GFLOP/s (${num(top.median_ms)} ms)`);
  const pending = state.jobs.filter((j) => j.status === "queued" || j.status === "running").length;
  if (pending) L.push(`# (${pending} payload(s) still pending)`);
  return L.join("\n") + "\n";
}
function writeOutputs() {
  fs.mkdirSync(cfg.out, { recursive: true });
  fs.writeFileSync(path.join(cfg.out, "best.txt"), renderBest());
  const rows = state.jobs.flatMap((j) => j.rows ?? []);
  const stamp = state.startedAt.toISOString().slice(0, 16).replace(/[-:T]/g, "");
  const ds = { schema: 1, createdAt: new Date().toISOString(), label: cfg.label, env: state.worker?.features ?? {},
    config: { remote: true, maxReps: cfg.maxReps, budgetMs: cfg.budgetMs }, inits: state.worker?.inits ?? {}, results: rows };
  fs.writeFileSync(path.join(cfg.out, `dataset-${cfg.label}-${stamp}.json`), JSON.stringify(ds, null, 1));   // same schema as the UI download
}

// ---------- job scheduling ---------------------------------------------------------------------------------
const settled = () => state.jobs.every((j) => j.status === "done" || j.status === "failed");
function requeueOrFail(job, why) {
  job.error = why;
  if (job.attempts < 2) { job.status = "queued"; log(`job ${job.id} requeued (${why})`); } else { job.status = "failed"; log(`job ${job.id} FAILED (${why})`); }
}
function finish() {
  if (state.finished) return;
  state.finished = true; writeOutputs();
  log(`all payloads processed -> ${path.join(cfg.out, "best.txt")}`);
  console.log("\n" + renderBest());
  for (const w of waiting.splice(0)) { clearTimeout(w.timer); send(w.res, { type: "done" }); }
  if (cfg.exitWhenDone) setTimeout(() => process.exit(0), 2500);
}
function pump() {
  if (settled()) finish();
  while (waiting.length) {
    if (state.finished) { const w = waiting.shift(); clearTimeout(w.timer); send(w.res, { type: "done" }); continue; }
    const job = state.jobs.find((j) => j.status === "queued");
    if (!job) break;
    const w = waiting.shift(); clearTimeout(w.timer);
    job.status = "running"; job.attempts++; job.assignedAt = Date.now();
    log(`-> job ${job.id}/${state.jobs.length}: ${job.kernel} N=${job.N}`);
    send(w.res, { type: "job", job: { id: job.id, kernel: job.kernel, N: job.N, m: job.m }, opts: { maxReps: cfg.maxReps, budgetMs: cfg.budgetMs } });
  }
}
setInterval(() => {
  for (const j of state.jobs) if (j.status === "running" && Date.now() - j.assignedAt > cfg.jobTimeoutSec * 1000) requeueOrFail(j, "timeout");
  pump();
}, 5000).unref();

// ---------- HTTP API ---------------------------------------------------------------------------------------
function send(res, obj, code = 200) { res.writeHead(code, { "Content-Type": "application/json", "Cache-Control": "no-store" }); res.end(JSON.stringify(obj)); }
const readBody = (req) => new Promise((resolve, reject) => { let b = ""; req.on("data", (c) => (b += c)); req.on("end", () => { try { resolve(b ? JSON.parse(b) : {}); } catch (e) { reject(e); } }); });

async function api(req, res) {
  const url = new URL(req.url, "http://x");
  try {
    if (req.method === "POST" && url.pathname === "/api/register") {
      const body = await readBody(req);
      for (const j of state.jobs) if (j.status === "running") j.status = "queued";
      state.worker = { id: Math.random().toString(36).slice(2, 8), features: body.features ?? {}, inits: body.inits ?? {} };
      log(`worker ${state.worker.id} registered: ${Object.keys(state.worker.inits).join(", ")} | ${state.worker.features.cores} cores | ${state.worker.features.ua}`);
      send(res, { workerId: state.worker.id }); pump(); return;
    }
    if (req.method === "GET" && url.pathname === "/api/next") {
      if (!state.worker || url.searchParams.get("worker") !== state.worker.id) return send(res, { type: "stale" });
      const entry = { res, timer: setTimeout(() => { waiting.splice(waiting.indexOf(entry), 1); send(res, { type: "wait" }); }, 15000) };
      req.on("close", () => { clearTimeout(entry.timer); const i = waiting.indexOf(entry); if (i >= 0) waiting.splice(i, 1); });
      waiting.push(entry); pump(); return;
    }
    if (req.method === "POST" && url.pathname === "/api/result") {
      const body = await readBody(req), job = state.jobs.find((j) => j.id === body.jobId);
      if (!job || job.status !== "running" || body.workerId !== state.worker?.id) return send(res, { ok: false });
      if (body.error) requeueOrFail(job, body.error);
      else {
        job.rows = body.rows; job.summary = summarise(job); job.status = job.summary ? "done" : "failed";
        const b = job.summary?.steady[0];
        log(`<- job ${job.id} ${job.kernel} N=${job.N}: best ${b ? `${b.backend} ${b.median_ms.toFixed(3)} ms` : "none valid"}` + (job.summary?.bad.length ? `  (${job.summary.bad.length} back-end error/incorrect)` : ""));
      }
      writeOutputs(); send(res, { ok: true }); pump(); return;
    }
    send(res, { error: "not found" }, 404);
  } catch (e) { send(res, { error: String(e) }, 400); }
}

http.createServer((req, res) => (req.url.startsWith("/api/") ? api(req, res) : staticHandler(req, res))).listen(cfg.port, "0.0.0.0", () => {
  log(`controller on http://localhost:${cfg.port}   payloads: ${state.jobs.length}   output: ${cfg.out}`);
  log(`waiting for a worker: open http://localhost:${cfg.port}/worker.html in a browser on the device under test`);
  for (const list of Object.values(os.networkInterfaces())) for (const i of list ?? []) if (i.family === "IPv4" && !i.internal) log(`  (network address ${i.address}:${cfg.port} is NOT a secure context; prefer the SSH tunnel, see README)`);
});
if (cfg.exitWhenDone) setTimeout(() => { if (!state.worker) { log(`no worker connected within ${cfg.waitSec} s, exiting`); process.exit(2); } }, cfg.waitSec * 1000).unref();
process.on("SIGINT", () => { log("interrupted, saving partial results"); writeOutputs(); process.exit(130); });
