// End-to-end test of the remote-run automation inside one machine:
// starts controller.js, connects a Node worker over HTTP, checks the output files.
import { spawn } from "node:child_process";
import fs from "node:fs";
import { runWorker } from "../public/js/worker_client.js";

const out = fs.mkdtempSync("/tmp/wd-e2e-"), port = 8131;
const ctl = spawn("node", ["controller.js", "--port", port, "--out", out, "--label", "e2e", "--jobs", "tests/e2e-jobs.json",
  "--reps", "5", "--budget", "200", "--exit-when-done"], { stdio: ["ignore", "pipe", "inherit"] });
let ctlOut = ""; ctl.stdout.on("data", (d) => (ctlOut += d));
const exited = new Promise((r) => ctl.on("exit", r));
await new Promise((r) => setTimeout(r, 800));

const n = await runWorker({ baseUrl: `http://localhost:${port}`, log: (m) => console.log("  [worker]", m), workers: 2 });
const code = await Promise.race([exited, new Promise((r) => setTimeout(() => r("timeout"), 15000))]);

let failed = 0;
const ok = (c, m) => { console.log((c ? "PASS " : "FAIL ") + m); if (!c) failed++; };
ok(n === 4, `worker completed 4 jobs (got ${n})`);
ok(code === 0, `controller exited cleanly (${code})`);
const best = fs.readFileSync(`${out}/best.txt`, "utf8");
console.log("\n" + best);
ok(/matmul\s+96\s+\w+/.test(best) && /saxpy\s+65536\s+\w+/.test(best), "best.txt has a best back-end line per payload");
ok(!best.includes("FAILED") && !best.includes("pending"), "no failed or pending payloads");
const dsFile = fs.readdirSync(out).find((f) => f.startsWith("dataset-e2e"));
const ds = JSON.parse(fs.readFileSync(`${out}/${dsFile}`, "utf8"));
ok(ds.results.length >= 4 * 3 && Object.keys(ds.inits).length >= 3, `dataset JSON has ${ds.results.length} rows, inits for ${Object.keys(ds.inits).join(",")}`);
ok(ds.results.every((r) => r.correct === true), "all results numerically correct");
process.exit(failed ? 1 : 0);
