#!/usr/bin/env node
// Multi-device controller.
// Any device that opens  http://<host-ip>:<port>/  automatically receives a series of small payloads, runs every
// back-end its browser supports on each one, and sends the timings back. The controller writes ONE REPORT PER DEVICE
// (reports/<device>_<ip>_<time>.txt + a dataset .json) and appends a line to reports/summary.txt.
//
//   node controller.js [--port 8080] [--out reports] [--m 10] [--quick] [--jobs jobs.json]
//                      [--reps 15] [--budget 1500] [--job-timeout 180] [--https cert.pem key.pem] [--exit-after N]
import http from "node:http";
import https from "node:https";
import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import { staticHandler } from "./server.js";

// ---------- configuration ----------------------------------------------------------------------------------
function parseArgs(a) {
  const o = {};
  for (let i = 0; i < a.length; i++) {
    if (!a[i].startsWith("--")) continue;
    const key = a[i].slice(2), next = a[i + 1];
    if (next !== undefined && !next.startsWith("--")) { o[key] = next; i++; } else o[key] = true;
  }
  return o;
}
const argv = process.argv.slice(2), args = parseArgs(argv);
const cfg = {
  port: Number(args.port ?? 8080), out: path.resolve(String(args.out ?? "reports")),
  m: Number(args.m ?? 10), maxReps: Number(args.reps ?? 15), budgetMs: Number(args.budget ?? 1500),
  jobTimeoutSec: Number(args["job-timeout"] ?? 180), exitAfter: Number(args["exit-after"] ?? 0),
};
const SMALL = [["matmul", [128, 192, 256, 384, 512, 768]], ["saxpy", [2 ** 18, 2 ** 20, 2 ** 22]]];   // big enough to rise above coarse browser timers
const QUICK = [["matmul", [128, 256]], ["saxpy", [2 ** 18, 2 ** 20]]];
const jobSpecs = args.jobs ? JSON.parse(fs.readFileSync(String(args.jobs), "utf8"))
  : (args.quick ? QUICK : SMALL).flatMap(([kernel, ns]) => ns.map((N) => ({ kernel, N })));

const sessions = new Map();          // one session per browser visit
let finishedCount = 0;
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);

// ---------- small helpers ----------------------------------------------------------------------------------
const num = (x, d = 3) => (x == null || Number.isNaN(x) ? "-" : x.toFixed(d));
const slug = (s) => String(s).replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^-+|-+$/g, "");
function parseUA(ua = "") {
  const pick = (re) => (ua.match(re) || [])[1] ?? "";
  let browser = "browser";
  if (/Edg\//.test(ua)) browser = "Edge " + pick(/Edg\/(\d+)/);
  else if (/Firefox\//.test(ua)) browser = "Firefox " + pick(/Firefox\/(\d+)/);
  else if (/CriOS\//.test(ua)) browser = "Chrome " + pick(/CriOS\/(\d+)/);
  else if (/Chrome\//.test(ua)) browser = "Chrome " + pick(/Chrome\/(\d+)/);
  else if (/Safari\//.test(ua)) browser = "Safari " + pick(/Version\/([\d.]+)/);
  else if (/node/i.test(ua)) browser = "node";
  let osName = "unknown OS";
  if (/Android/.test(ua)) osName = "Android"; else if (/iPhone|iPad/.test(ua)) osName = "iOS";
  else if (/Windows/.test(ua)) osName = "Windows"; else if (/Mac OS X/.test(ua)) osName = "macOS";
  else if (/CrOS/.test(ua)) osName = "ChromeOS"; else if (/Linux/.test(ua)) osName = "Linux";
  return { browser: browser.trim(), os: osName };
}
const reliableMin = (s) => Math.max(0.2, 10 * s.timerRes);     // below this a median is within timer noise

// ---------- per-payload ranking ----------------------------------------------------------------------------
function summarise(s, job) {
  const oneoff = (b) => (s.inits[b]?.load_ms ?? 0) + (s.inits[b]?.setup_ms ?? 0);
  const ok = job.rows.filter((r) => !r.error && r.correct !== false && r.median_ms != null);
  if (!ok.length) return null;
  const steady = [...ok].sort((a, b) => a.median_ms - b.median_ms);
  const total = (m) => ok.map((r) => ({ r, t: oneoff(r.backend) + r.first_ms + (m - 1) * r.median_ms })).sort((a, b) => a.t - b.t);
  return { steady, totalM: total(job.m), total1: total(1), bad: job.rows.filter((r) => r.error || r.correct === false),
           reliable: steady[0].median_ms >= reliableMin(s) };
}
function ranges(items) {            // [{N, name}] (sorted by N) -> "N=64-96: simd; N=128-512: threads"
  const out = [];
  for (const it of items) {
    const last = out[out.length - 1];
    if (last && last.name === it.name) last.hi = it.N; else out.push({ name: it.name, lo: it.N, hi: it.N });
  }
  return out.map((r) => `${r.lo === r.hi ? `N=${r.lo}` : `N=${r.lo}-${r.hi}`}: ${r.name}`).join(";  ");
}

// ---------- report -----------------------------------------------------------------------------------------
function renderReport(s) {
  const f = s.features, L = [];
  const done = s.jobs.filter((j) => j.summary), failed = s.jobs.filter((j) => j.status === "failed").length;
  const skipped = s.jobs.filter((j) => j.status === "skipped").length;
  const yn = (b) => (b ? "yes" : "NO");
  L.push("=".repeat(78), ` BACK-END REPORT: ${s.name}`, "=".repeat(78));
  L.push(` device       : ${s.name}   (ip ${s.ip}${s.label ? `, label "${s.label}"` : ""}, client ${s.clientId ?? "?"})`);
  L.push(` browser UA   : ${f.ua}`);
  L.push(` hardware     : ${f.cores} logical cores${f.deviceMemoryGB ? `, ~${f.deviceMemoryGB} GB RAM` : ""}${f.gpu?.description || f.gpu?.vendor ? `, GPU ${f.gpu.description || f.gpu.vendor}` : ""}`);
  L.push(` capabilities : SIMD ${yn(f.simd)} | threads ${yn(f.threads)} | WebGPU ${yn(f.webgpu)} | secure context ${yn(f.secureContext)} | isolated ${yn(f.crossOriginIsolated)}`);
  L.push(` timer tick   : ${num(s.timerRes, 3)} ms   (medians below ${num(reliableMin(s), 2)} ms are marked '?' = unreliable)`);
  L.push(` back-ends run: ${Object.keys(s.inits).join(", ")}`);
  L.push(` payloads     : ${done.length} measured, ${failed} failed, ${skipped} skipped   | started ${s.startedAt.toISOString()}${s.finished ? "" : "   (IN PROGRESS)"}`);
  if (s.note) L.push(` NOTE         : ${s.note}`);

  const warn = [];
  if (!f.secureContext) warn.push("Opened over plain HTTP from a network address, so the browser disabled WebGPU and threads and coarsened its timers. Only JS / WASM / SIMD were tested. For threads + WebGPU, serve over trusted HTTPS (node controller.js --https cert key) or open the page via localhost (SSH / adb tunnel).");
  else if (!f.threads) warn.push(`Threads not available (${f.cores > 1 ? "page not cross-origin isolated" : "single core"}).`);
  if (f.secureContext && !f.webgpu) warn.push("WebGPU is not available in this browser/device.");
  if (!f.simd) warn.push("WebAssembly SIMD is not supported here.");
  if (warn.length) { L.push("", " LIMITATIONS"); warn.forEach((w) => L.push("  - " + w)); }

  L.push("", ` BEST BACK-END PER PAYLOAD   (steady = lowest median ms per call; total = load+setup+first call+(m-1)*median)`);
  const head = ["kernel", "N", "best_steady", "median_ms", "GFLOP/s", "best_total(m)", "total_ms", "all medians (ms)"];
  const rows = [head];
  for (const j of s.jobs) {
    if (j.status === "failed" || j.status === "skipped") { rows.push([j.kernel, String(j.N), j.status.toUpperCase(), "", "", "", "", j.error ?? ""]); continue; }
    if (!j.summary) continue;
    const sm = j.summary, b = sm.steady[0], t = sm.totalM[0], mark = sm.reliable ? "" : "?";
    rows.push([j.kernel, String(j.N), b.backend + mark, num(b.median_ms), num(b.gflops, 2), `${t.r.backend}${mark} (m=${j.m})`, num(t.t, 2),
      sm.steady.map((r) => `${r.backend}=${num(r.median_ms)}`).join(" ")]);
  }
  const w = head.map((_, c) => Math.max(...rows.map((r) => r[c].length)));
  rows.forEach((r) => L.push("  " + r.map((x, c) => (c === head.length - 1 ? x : x.padEnd(w[c]))).join("  ")));

  L.push("", " RECOMMENDATION FOR THIS DEVICE");
  for (const kernel of [...new Set(s.jobs.map((j) => j.kernel))]) {
    const js = s.jobs.filter((j) => j.kernel === kernel && j.summary && j.summary.reliable).sort((a, b) => a.N - b.N);
    if (!js.length) { L.push(`  ${kernel}: no reliable measurements (all below timer resolution)`); continue; }
    L.push(`  ${kernel}`,
      `    repeated calls : ${ranges(js.map((j) => ({ N: j.N, name: j.summary.steady[0].backend })))}`,
      `    single call    : ${ranges(js.map((j) => ({ N: j.N, name: j.summary.total1[0].r.backend })))}   (includes start-up cost)`);
  }
  const untested = ["threads", "webgpu"].filter((b) => !(b in s.inits));
  if (untested.length) L.push("", ` NOT TESTED ON THIS DEVICE: ${untested.join(", ")}  (see limitations)`);
  L.push("", " '?' = measurement within timer noise; treat as unreliable. Rankings use medians of repeated calls; spreads are in the dataset .json.");
  return L.join("\n") + "\n";
}

function writeOutputs(s) {
  fs.mkdirSync(cfg.out, { recursive: true });
  fs.writeFileSync(path.join(cfg.out, s.base + ".txt"), renderReport(s));
  const ds = { schema: 1, createdAt: new Date().toISOString(), label: s.name, env: { ...s.features, ip: s.ip },
    config: { remote: true, timerResolutionMs: s.timerRes, maxReps: cfg.maxReps, budgetMs: cfg.budgetMs },
    inits: s.inits, results: s.jobs.flatMap((j) => j.rows ?? []) };
  fs.writeFileSync(path.join(cfg.out, s.base + ".json"), JSON.stringify(ds, null, 1));   // same schema as the UI download
}

// ---------- sessions & scheduling --------------------------------------------------------------------------
const settled = (s) => s.jobs.every((j) => ["done", "failed", "skipped"].includes(j.status));
const send = (res, obj, code = 200) => { res.writeHead(code, { "Content-Type": "application/json", "Cache-Control": "no-store" }); res.end(JSON.stringify(obj)); };

function register(body, req) {
  const features = body.features ?? {}, ip = (req.socket.remoteAddress ?? "?").replace(/^::ffff:/, "");
  const label = String(body.label ?? "").slice(0, 40), { browser, os: osName } = parseUA(features.ua);
  const name = label || `${browser} on ${osName}`;
  const stamp = new Date().toISOString().slice(0, 16).replace(/[-:T]/g, "");
  const s = { id: Math.random().toString(36).slice(2, 10), ip, label, clientId: body.clientId, features, inits: body.inits ?? {},
    timerRes: Number(body.timerResolutionMs) || 0.1, name, startedAt: new Date(), finished: false, note: "", waiting: null,
    base: slug(`${name}_${ip}_${stamp}`),
    jobs: jobSpecs.map((j, i) => ({ id: i + 1, kernel: j.kernel, N: j.N, m: j.m ?? cfg.m, status: "queued" })) };
  sessions.set(s.id, s);
  log(`+ device connected: ${name} | ${ip} | ${features.cores} cores | simd=${features.simd} threads=${features.threads} webgpu=${features.webgpu} secure=${features.secureContext} | tick ${num(s.timerRes, 3)} ms`);
  writeOutputs(s);
  return s;
}
function finalize(s) {
  if (s.finished) return;
  s.finished = true; writeOutputs(s);
  const best = {};
  for (const kernel of new Set(s.jobs.map((j) => j.kernel))) {
    const last = s.jobs.filter((j) => j.kernel === kernel && j.summary).sort((a, b) => b.N - a.N)[0];
    if (last) best[kernel] = `${last.summary.steady[0].backend}@N=${last.N}`;
  }
  fs.appendFileSync(path.join(cfg.out, "summary.txt"),
    `${new Date().toISOString()}  ${s.name.padEnd(24)} ${s.ip.padEnd(15)} ${Object.entries(best).map(([k, v]) => `${k}:${v}`).join("  ")}  ${s.note ? "[" + s.note + "] " : ""}-> ${s.base}.txt\n`);
  log(`= report written: ${path.join(cfg.out, s.base + ".txt")}`);
  if (s.waiting) { clearTimeout(s.waiting.timer); send(s.waiting.res, { type: "done" }); s.waiting = null; }
  if (cfg.exitAfter && ++finishedCount >= cfg.exitAfter) setTimeout(() => process.exit(0), 2500);
}
function abandon(s, why) {
  for (const j of s.jobs) if (j.status === "running") { j.status = "failed"; j.error = why; } else if (j.status === "queued") j.status = "skipped";
  s.note = `incomplete: ${why}`; log(`! ${s.name} (${s.ip}): ${why}; saving partial report`); finalize(s);
}
function pump(s) {
  if (settled(s)) finalize(s);
  if (!s.waiting || s.finished) return;
  const job = s.jobs.find((j) => j.status === "queued");
  if (!job) return;
  const w = s.waiting; s.waiting = null; clearTimeout(w.timer);
  job.status = "running"; job.assignedAt = Date.now();
  log(`-> ${s.name}: payload ${job.id}/${s.jobs.length} ${job.kernel} N=${job.N}`);
  send(w.res, { type: "job", job: { id: job.id, kernel: job.kernel, N: job.N, m: job.m }, opts: { maxReps: cfg.maxReps, budgetMs: cfg.budgetMs } });
}
setInterval(() => {
  for (const s of sessions.values()) {
    if (s.finished) continue;
    const stuck = s.jobs.find((j) => j.status === "running" && Date.now() - j.assignedAt > cfg.jobTimeoutSec * 1000);
    if (stuck) abandon(s, `no answer for payload ${stuck.id} within ${cfg.jobTimeoutSec}s (tab closed or device too slow)`);
  }
}, 5000).unref();

// ---------- HTTP API ---------------------------------------------------------------------------------------
const readBody = (req) => new Promise((resolve, reject) => { let b = ""; req.on("data", (c) => (b += c)); req.on("end", () => { try { resolve(b ? JSON.parse(b) : {}); } catch (e) { reject(e); } }); });

async function api(req, res) {
  const url = new URL(req.url, "http://x"), sid = url.searchParams.get("worker");
  try {
    if (req.method === "POST" && url.pathname === "/api/register") {
      const s = register(await readBody(req), req);
      return send(res, { workerId: s.id, total: s.jobs.length });
    }
    const s = sessions.get(sid);
    if (req.method === "GET" && url.pathname === "/api/next") {
      if (!s || s.finished) return send(res, { type: s ? "done" : "stale" });
      if (s.waiting) { clearTimeout(s.waiting.timer); send(s.waiting.res, { type: "wait" }); }
      const entry = { res, timer: setTimeout(() => { if (s.waiting === entry) s.waiting = null; send(res, { type: "wait" }); }, 15000) };
      req.on("close", () => { clearTimeout(entry.timer); if (s.waiting === entry) s.waiting = null; });
      s.waiting = entry; return pump(s);
    }
    if (req.method === "POST" && url.pathname === "/api/result") {
      const body = await readBody(req), rs = sessions.get(body.workerId), job = rs?.jobs.find((j) => j.id === body.jobId);
      if (!job || job.status !== "running") return send(res, { ok: false });
      const s = rs;
      if (body.error) { job.status = "failed"; job.error = body.error; log(`<- ${s.name}: payload ${job.id} FAILED (${body.error})`); }
      else {
        job.rows = body.rows; job.summary = summarise(s, job); job.status = job.summary ? "done" : "failed";
        const b = job.summary?.steady[0];
        log(`<- ${s.name}: payload ${job.id} ${job.kernel} N=${job.N}: best ${b ? `${b.backend} ${b.median_ms.toFixed(3)} ms${job.summary.reliable ? "" : " (?)"}` : "none valid"}`);
      }
      writeOutputs(s); send(res, { ok: true }); return pump(s);
    }
    if (req.method === "GET" && url.pathname === "/api/report") {
      if (!s) return send(res, { error: "unknown worker" }, 404);
      return send(res, { text: renderReport(s), done: s.jobs.filter((j) => j.status === "done").length, total: s.jobs.length, finished: s.finished });
    }
    send(res, { error: "not found" }, 404);
  } catch (e) { send(res, { error: String(e) }, 400); }
}

function handler(req, res) {
  const p = new URL(req.url, "http://x").pathname;
  if (p.startsWith("/api/")) return api(req, res);
  if (p === "/") req.url = "/worker.html" + (req.url.includes("?") ? req.url.slice(req.url.indexOf("?")) : "");   // "/" = auto-start page
  return staticHandler(req, res);
}

const hi = argv.indexOf("--https");
const server = hi >= 0
  ? https.createServer({ cert: fs.readFileSync(argv[hi + 1]), key: fs.readFileSync(argv[hi + 2]) }, handler)
  : http.createServer(handler);
const scheme = hi >= 0 ? "https" : "http";
server.listen(cfg.port, "0.0.0.0", () => {
  log(`controller up: ${jobSpecs.length} payloads per device, reports in ${cfg.out}`);
  log(`this machine : ${scheme}://localhost:${cfg.port}/`);
  for (const list of Object.values(os.networkInterfaces()))
    for (const i of list ?? []) if (i.family === "IPv4" && !i.internal) log(`other devices: ${scheme}://${i.address}:${cfg.port}/   (open this in their browser; add ?label=name to name the device)`);
  log("waiting for devices... (Ctrl+C to stop; the manual benchmark UI is at /ui.html)");
});
process.on("SIGINT", () => { for (const s of sessions.values()) if (!s.finished) { s.note ||= "controller stopped"; writeOutputs(s); } process.exit(130); });
