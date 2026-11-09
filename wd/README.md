# Cost-model-driven back-end dispatcher (CS4098E)

Decides, **per call and per device**, whether a compute kernel should run on
**JavaScript, scalar WebAssembly, SIMD WebAssembly, threaded SIMD WebAssembly, or WebGPU**,
using a cost model fitted from a short on-device calibration.

```
latency(back-end) = load + setup + cold + m * T_call(N)        m = expected number of calls
T_call(N) = c0 + c1*work(N) + c2*bytes(N)                       fitted per back-end & kernel
```
Success measure: **regret = latency(chosen) / latency(best in hindsight)**.

## What is in here

| Path | Purpose |
|---|---|
| `src/kernels.c` | C kernels (matmul, saxpy), freestanding, no Emscripten glue. Built 3 ways: scalar / SIMD128 / SIMD128+shared memory |
| `build.mjs`, `Makefile` | Build the `.wasm` files (needs `clang` + `wasm-ld`). Prebuilt files are already in `public/wasm/` |
| `server.js` | Static server that sends COOP/COEP headers (required for threads / `SharedArrayBuffer`) |
| `public/js/backends/` | The five back-ends behind one interface: `init()` and `call(kernel, N, inputs)` |
| `public/js/costmodel.js` | Weighted non-negative least-squares fit, ranking, static baseline, probe policy |
| `public/js/dispatcher.js` | Feature probe, calibration, prediction, selection, fallback chain |
| `public/js/bench.js`, `index.html`, `ui.js` | Benchmark harness and UI (phase timings, median/IQR/CI, correctness check) |
| `analysis/evaluate.mjs` | Offline comparison of M0 / M1 / M2 / best-fixed against the oracle (leave-one-device-out for M0) |
| `analysis/sample-data/` | **Synthetic** datasets, only to test the pipeline. Never cite them as results |
| `controller.js`, `scripts/run-on-pi.sh`, `public/worker.html` | Automated remote runs (see section below) |
| `tests/node-test.mjs` | Correctness (JS vs scalar vs SIMD vs threads), cost model, dispatcher, fallback |

## Run it

```bash
node build.mjs          # optional: only if you change src/kernels.c
node tests/node-test.mjs
node server.js          # open http://localhost:8080 in Chrome (add `--https cert key` for HTTPS)
```
Browser page: (1) check the environment line (SIMD / isolated / threads / WebGPU all "yes"),
(2) **Run benchmark**, then **Download JSON**, (3) **Calibrate** and **Run via dispatcher**.

Evaluate collected data (one JSON per device, 2 or more recommended):
```bash
node analysis/evaluate.mjs data/desktop.json data/laptop.json data/phone.json
```

### Testing on a phone (no cable)
WebGPU and cross-origin isolation need a **secure context**; `http://<pc-ip>:8080` is *not* one. Four wireless options,
in order of recommendation. After connecting, confirm the page's environment line shows "cross-origin isolated: yes".
Campus Wi-Fi often blocks device-to-device traffic ("client isolation"); if A, B or D cannot connect, put the PC on the
phone's hotspot, or use C.

**A. Wireless ADB (Android 11+). Same as the cable method, keeps `localhost`.**
1. Phone and PC on the same network. Phone: Settings > Developer options > *Wireless debugging* > on > *Pair device with pairing code*.
2. PC: `adb pair <ip>:<pairing-port>` and enter the code, then `adb connect <ip>:<port>` (the port shown on the Wireless debugging screen).
3. `adb reverse tcp:8080 tcp:8080`, then open `http://localhost:8080` on the phone.

**B. Chrome flag (fastest setup, Android).** On the phone open `chrome://flags/#unsafely-treat-insecure-origin-as-secure`,
enable it, add `http://<pc-ip>:8080` (the "network" address `node server.js` prints), relaunch Chrome, open that address.

**C. HTTPS tunnel (works on any network, also iPhone).** `cloudflared tunnel --url http://localhost:8080` prints an
`https://<random>.trycloudflare.com` address to open on the phone. Caveat: `load_ms` then includes internet round-trips,
so for phone data compare `setup_ms`, first-call and steady-state times, and treat `load_ms` as inflated (note this in the report).

**D. Trusted local HTTPS (permanent).** Install [mkcert](https://github.com/FiloSottile/mkcert), then
`mkcert -install && mkcert <pc-ip> localhost` and `node server.js 8443 --https <pc-ip>+1.pem <pc-ip>+1-key.pem`.
Send `rootCA.pem` (folder from `mkcert -CAROOT`) to the phone by email or cloud drive and install it
(Android: Settings > Security > Encryption & credentials > Install a certificate > CA certificate), then open `https://<pc-ip>:8443`.

Whatever you use: plug the phone in or note its battery state, disable battery saver, keep the screen on and the tab in the
foreground, and let the phone cool between runs, since throttling changes the results. WebGPU on Android and Firefox depends
on the browser version, so check the environment line before relying on it.

## Automated remote runs (e.g. a Raspberry Pi) -> `best.txt`

Your laptop (WSL) runs a **controller** that queues small payloads. A browser on the Pi runs a **worker page**
that pulls one payload at a time (HTTP long-polling), runs *every available back-end* on it, and posts the timings
back. The controller picks the best back-end per payload and writes `results/best.txt`
plus `results/dataset-<label>-<time>.json` (same schema as the UI download, so `analysis/evaluate.mjs` accepts it).

**Why an SSH reverse tunnel.** Opening `http://<laptop-ip>:8080` on the Pi is not a secure context, so threads
(and WebGPU) are disabled and `performance.now()` is coarsened to 0.1 ms. Tunnelling makes the Pi see the controller
as `localhost`, which *is* a secure context. It also avoids WSL2 networking: the connection is outbound from WSL.

**One command (headless browser on the Pi, needs `chromium`, `chromium-browser` or `firefox` installed there):**
```bash
scripts/run-on-pi.sh pi@<pi-ip> pi4            # add --quick for a short run, --m 100 to rank for 100 repeated calls
```
**Manual version (if you prefer to watch it in the Pi's own browser):**
```bash
# WSL terminal 1
node controller.js --label pi4
# WSL terminal 2
ssh -R 8080:localhost:8080 pi@<pi-ip>
# on the Pi: open  http://localhost:8080/worker.html   (must say localhost, not the laptop's IP)
```
Options: `--out DIR` (e.g. `/mnt/c/Users/<you>/Documents/wasm-results` to land on the Windows side), `--m N` (expected calls
for the "best_total" column), `--jobs file.json` (custom payloads, e.g. `[{"kernel":"matmul","N":256,"m":50}]`),
`--reps`/`--budget` (precision vs. time), `--exit-when-done`.

`best.txt` has, per payload, the back-end with the lowest median time per call (*steady*) and the one with the lowest
expected total for `m` calls (*total* = load + setup + first call + (m-1) x median), plus all medians.
Caveats: back-ends are initialised once per session, so "first call" later in a run is warmer than a fresh page load;
use the UI's full benchmark when you need true cold-start numbers. One worker at a time; a newer worker replaces the old one.
`node tests/e2e-test.mjs` tests the whole loop on one machine.

## Design decisions worth knowing (put these in the report)

* **Plain clang instead of Emscripten.** The kernels are tiny and freestanding, so there is no glue code and
  load time is just fetch + compile. Threads use our own Web Worker pool over one shared `WebAssembly.Memory`,
  which works because the kernels never touch the wasm shadow stack (checked with `llvm-objdump`: zero `global.get/set`).
  Report the flags: `-O3`, `-msimd128`, `-matomics -mbulk-memory`.
* **Same algorithm everywhere** (i-k-j matmul, float32). WGSL has no f64, so everything is float32.
* **A "call" includes data movement** (copy-in, compute, copy-out), because an application hands over JS arrays and wants
  JS arrays back. Phases are logged separately (`phases_ms`).
* **Time-targeted probing.** Small fixed probes cannot see the compute cost of fast back-ends (fixed overhead hides it),
  which gave mean regret ~1.7 on synthetic data. Probing each back-end up the size ladder until one call takes ~25 ms
  (`PROBE_POLICY`) fixed that (~1.00 on synthetic data). Expect real data to be harder: cache effects, GPU clocks, thermal throttling.
* **Reference result** for the correctness check is the scalar Wasm output (tolerance 1e-4 relative to max|ref|);
  scalar Wasm itself is checked against JS in `tests/node-test.mjs`.

## What has and has not been verified

Verified in Node 22 (Linux, 1 core): wasm builds, JS/scalar/SIMD/threaded results agree (including odd sizes), cost-model
fit, calibration + ranking + fallback, the evaluation script (on synthetic data), server headers.

**Not verified here (needs your machine):** the WebGPU back-end and the browser UI have not been run (no browser or GPU in
the build environment); the threaded path was tested with Node worker threads, not browser Web Workers. Expect to fix small
things on first run. Calibration took ~2 s on the slow single-core test box, above the 200 ms target in the proposal; tune
`PROBE_POLICY` (smaller `targetMs` / `maxProbeMs`) and note the overhead as a result. Calibration results are cached in `localStorage`.

## Known gaps / next steps

1. Add the third kernel (2-D convolution) in `kernels.js`, `kernels.c`, `webgpu.js`, `backends/*.js`.
2. Run benchmarks inside a Web Worker and record main-thread long tasks (the design's responsiveness metric).
3. GPU watchdog: chunk long dispatches on mobile (WgPy reports stalls there).
4. Calibration is per kernel family; consider caching per size bucket and re-calibrating on large timing drift.
5. Stretch goal (RQ4): pipeline planner. Dynamic program over stages with a transfer cost when switching back-ends.
6. Add Firefox/Safari runs, and repeat a subset at the end of the project to control for browser-version drift.

## Adding a kernel
1. `public/js/kernels.js`: add an entry (`defaultSizes`, `probeLadder`, `features`, `flops`, `makeInputs`).
2. `src/kernels.c`: add an exported range function; add it to `LDCOMMON`/`link` exports in `Makefile` and `build.mjs`.
3. Implement it in `js.js`, `wasm.js`, `wasm_mt.js` (split by range) and `webgpu.js` (WGSL + pipeline).
4. Add it to `tests/node-test.mjs` and rerun.
