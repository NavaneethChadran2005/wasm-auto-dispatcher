// End-to-end test of the multi-device automation on one machine:
// starts controller.js, connects TWO simulated devices at the same time, checks reports, summary and dataset files.
import { spawn } from "node:child_process";
import fs from "node:fs";
import { runWorker } from "../public/js/worker_client.js";

const out = fs.mkdtempSync("/tmp/wd-e2e-"), port = 8132;
const ctl = spawn("node", ["controller.js", "--port", port, "--out", out, "--jobs", "tests/e2e-jobs.json",
  "--reps", "5", "--budget", "200", "--exit-after", "2"], { stdio: ["ignore", "pipe", "inherit"] });
ctl.stdout.on("data", () => {});
const exited = new Promise((r) => ctl.on("exit", r));
await new Promise((r) => setTimeout(r, 800));

const base = `http://localhost:${port}`;
// the page that devices open must be the auto-start worker page
const root = await (await fetch(base + "/")).text();
const hdr = (await fetch(base + "/")).headers.get("cross-origin-opener-policy");

const [a, b] = await Promise.all([
  runWorker({ baseUrl: base, label: "deviceA", workers: 2, log: () => {} }),
  // device B simulates a phone opened over plain http: no secure context, coarse timer
  runWorker({ baseUrl: base, label: "deviceB-plainhttp", workers: 2, log: () => {}, featuresOverride: { secureContext: false, crossOriginIsolated: false } }),
]);
const code = await Promise.race([exited, new Promise((r) => setTimeout(() => r("timeout"), 15000))]);

let failed = 0;
const ok = (c, m) => { console.log((c ? "PASS " : "FAIL ") + m); if (!c) failed++; };
ok(root.includes("runWorker") && hdr === "same-origin", "GET / serves the auto-start page with COOP/COEP headers");
ok(a.jobsDone === 4 && b.jobsDone === 4, `both devices completed 4 payloads (${a.jobsDone}, ${b.jobsDone})`);
ok(code === 0, `controller exited after 2 reports (${code})`);
const files = fs.readdirSync(out);
const txts = files.filter((f) => f.endsWith(".txt") && f !== "summary.txt"), jsons = files.filter((f) => f.endsWith(".json"));
ok(txts.length === 2 && jsons.length === 2, `one report + one dataset per device (${files.join(", ")})`);
const repA = fs.readFileSync(`${out}/${txts.find((f) => f.startsWith("deviceA"))}`, "utf8");
const repB = fs.readFileSync(`${out}/${txts.find((f) => f.startsWith("deviceB"))}`, "utf8");
console.log("\n" + repA);
ok(/BEST BACK-END PER PAYLOAD/.test(repA) && /RECOMMENDATION FOR THIS DEVICE/.test(repA) && /matmul\s+96\s+\w+/.test(repA), "report A has per-payload table and recommendation");
ok(!/LIMITATIONS[\s\S]*plain HTTP/.test(repA), "report A has no plain-HTTP warning");
ok(/plain HTTP/.test(repB), "report B warns that plain HTTP limited the test");
ok(a.report.includes("BEST BACK-END") && b.report.includes("plain HTTP"), "devices received their own report text back");
const summary = fs.readFileSync(`${out}/summary.txt`, "utf8").trim().split("\n");
ok(summary.length === 2 && summary.some((l) => l.includes("deviceA")) && summary.some((l) => l.includes("deviceB")), "summary.txt has one line per device");
const ds = JSON.parse(fs.readFileSync(`${out}/${jsons[0]}`, "utf8"));
ok(ds.results.length >= 12 && Object.keys(ds.inits).length >= 3 && ds.results.every((r) => r.correct === true), `dataset JSON valid (${ds.results.length} rows, all correct)`);
process.exit(failed ? 1 : 0);
