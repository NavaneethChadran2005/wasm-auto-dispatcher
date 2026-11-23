import { detectFeatures } from "./features.js";
import { createBackends, Dispatcher } from "./dispatcher.js";
import { runBenchmark } from "./bench.js";
import { KERNELS } from "./kernels.js";

const $ = (id) => document.getElementById(id);
const log = (m) => { const el = $("log"); el.textContent += m + "\n"; el.scrollTop = el.scrollHeight; };
let features, backends, dispatcher, dataset;

function parseSizes(txt) { return txt.split(/[\s,]+/).filter(Boolean).map(Number).filter((n) => n > 0); }

async function boot() {
  features = await detectFeatures();
  const yes = (b) => `<span class="${b ? "ok" : "no"}">${b ? "yes" : "no"}</span>`;
  $("env").innerHTML = `CPU cores: <b>${features.cores}</b> &middot; SIMD128: ${yes(features.simd)} &middot; ` +
    `cross-origin isolated: ${yes(features.crossOriginIsolated)} &middot; threads: ${yes(features.threads)} &middot; ` +
    `WebGPU: ${yes(features.webgpu)}${features.gpu ? ` (${features.gpu.vendor} ${features.gpu.architecture})` : ""}` +
    `<div class="mute">${features.ua}</div>` +
    (!features.crossOriginIsolated ? `<div class="no">Not cross-origin isolated: serve with <code>node server.js</code> (localhost or HTTPS).</div>` : "");
  backends = createBackends(features);
  dispatcher = new Dispatcher(backends, features, { log });
  $("backendBoxes").innerHTML = [...backends.keys()].map((n) => `<label><input type="checkbox" class="bk" value="${n}" checked> ${n}</label>`).join("");
  for (const k of Object.keys(KERNELS)) {
    $("sz-" + k).value = KERNELS[k].defaultSizes.join(", ");
    $("dKernel").insertAdjacentHTML("beforeend", `<option value="${k}">${k}</option>`);
  }
  for (const id of ["runBench", "calib"]) $(id).disabled = false;
  startBenchmark();
}

async function startBenchmark() {
  const chosen = new Map([...document.querySelectorAll(".bk:checked")].map((c) => [c.value, backends.get(c.value)]));
  const plan = Object.fromEntries(Object.keys(KERNELS).map((k) => [k, parseSizes($("sz-" + k).value)]).filter(([, v]) => v.length));
  $("runBench").disabled = true; $("benchState").textContent = "running… keep this tab in the foreground";
  const tb = $("results").tBodies[0]; tb.innerHTML = ""; $("results").hidden = false;
  try {
    dataset = await runBenchmark(chosen, plan, {
      features, maxReps: +$("maxReps").value, budgetMs: +$("budget").value,
      onProgress: (m) => {
        log(m);
      },
    });
    for (const r of dataset.results) {
      tb.insertAdjacentHTML("beforeend", `<tr><td>${r.kernel}</td><td>${r.N}</td><td>${r.backend}</td>` +
        (r.error ? `<td colspan="6" class="no">${r.error}</td>` :
        `<td>${r.first_ms.toFixed(2)}</td><td>${r.median_ms.toFixed(3)}</td><td>${r.ci95_ms ? r.ci95_ms.map((x) => x.toFixed(2)).join("–") : "–"}</td>` +
        `<td>${r.gflops ? r.gflops.toFixed(2) : "–"}</td><td>${r.reps}</td><td class="${r.correct ? "ok" : "no"}">${r.correct ? "✓" : "✗"}</td>`) + "</tr>");
    }
    $("download").disabled = false; $("benchState").textContent = "done";
  } catch (e) { log("benchmark error: " + e); $("benchState").textContent = "failed (see log)"; }
  $("runBench").disabled = false;
}

$("runBench").onclick = startBenchmark;

$("download").onclick = () => {
  const name = `wasm-bench-${(features.gpu?.vendor || "cpu")}-${new Date().toISOString().slice(0, 16).replace(/[:T]/g, "")}.json`;
  const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(new Blob([JSON.stringify(dataset, null, 1)], { type: "application/json" })), download: name });
  a.click(); URL.revokeObjectURL(a.href);
};

$("calib").onclick = async () => {
  $("calib").disabled = true; $("dOut").textContent = "calibrating…";
  const c = await dispatcher.calibrate({ useCache: false });
  $("dOut").textContent = `calibrated in ${c.overhead_ms.toFixed(0)} ms (available: ${dispatcher.available.join(", ")})`;
  $("runD").disabled = false; $("calib").disabled = false;
};

$("runD").onclick = async () => {
  const kernel = $("dKernel").value, N = +$("dN").value, m = +$("dM").value;
  dispatcher.policy = $("dPolicy").value;
  const inp = KERNELS[kernel].makeInputs(N);
  try {
    const r = await dispatcher.run(kernel, N, inp, { m });
    const rank = r.ranking.map((x) => `${x.name}${x.est_ms !== undefined ? ` (${x.est_ms.toFixed(1)} ms)` : ""}`).join(" &gt; ");
    $("dOut").innerHTML = `chose <b>${r.backend}</b> &middot; actual call ${r.total_ms.toFixed(2)} ms ` +
      `(in ${r.phases.in_ms.toFixed(2)} / exec ${r.phases.exec_ms.toFixed(2)} / out ${r.phases.out_ms.toFixed(2)})` +
      `<div class="mute">predicted ranking for m=${m}: ${rank}</div>` +
      (r.fallbacks.length ? `<div class="no">fallbacks: ${r.fallbacks.map((f) => f.name + ": " + f.error).join("; ")}</div>` : "");
  } catch (e) { $("dOut").textContent = "error: " + e; }
};

boot().catch((e) => log("boot error: " + e));
