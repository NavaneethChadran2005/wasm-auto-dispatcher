// Cost model:  L(B) = load + setup + cold + m * T_call(N),   T_call(N) = c . features(N)
// Coefficients c are fitted by weighted, non-negative least squares (relative error).
// Pure functions: used by the browser dispatcher AND the Node evaluation script.

/** Dispatch order when something is unavailable or fails (fastest-for-big-jobs first). */
export const CHAIN = ["webgpu", "threads", "simd", "wasm", "js"];

function solve(A, b) {                      // Gaussian elimination with partial pivoting
  const n = b.length, M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    if (Math.abs(M[p][c]) < 1e-18) return null;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / M[c][c];
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r][n];
    for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k];
    x[r] = s / M[r][r];
  }
  return x;
}

/** points: [{N, t}] (t = median per-call latency in ms). Returns {coef}. */
export function fitCostModel(points, featFn) {
  const X = points.map((p) => featFn(p.N)), y = points.map((p) => p.t);
  const w = y.map((t) => 1 / Math.max(t, 1e-3));               // relative-error weights
  const P = X[0].length;
  const scale = Array.from({ length: P }, (_, j) => Math.max(...X.map((r) => Math.abs(r[j]))) || 1);
  let active = [...Array(P).keys()];
  const coef = new Array(P).fill(0);
  while (active.length) {
    const m = active.length;
    const AtA = Array.from({ length: m }, () => new Array(m).fill(0)), Atb = new Array(m).fill(0);
    for (let r = 0; r < X.length; r++) {
      for (let a = 0; a < m; a++) {
        const xa = X[r][active[a]] / scale[active[a]];
        Atb[a] += xa * y[r] * w[r] * w[r];
        for (let b = 0; b < m; b++) AtA[a][b] += xa * (X[r][active[b]] / scale[active[b]]) * w[r] * w[r];
      }
    }
    for (let a = 0; a < m; a++) AtA[a][a] += 1e-12;            // tiny ridge for stability
    const sol = solve(AtA, Atb);
    if (!sol) { active = active.slice(0, -1); continue; }
    let worst = -1, worstVal = 0;
    sol.forEach((v, i) => { if (v < worstVal) { worstVal = v; worst = i; } });
    if (worst < 0) { sol.forEach((v, i) => (coef[active[i]] = v / scale[active[i]])); break; }
    active.splice(worst, 1);                                   // drop most negative term, refit
  }
  if (!coef.some((c) => c > 0)) coef[0] = y.reduce((a, b) => a + b, 0) / y.length;
  return { coef };
}

export const predictCall = (model, N, featFn) =>
  Math.max(0, featFn(N).reduce((s, f, i) => s + f * model.coef[i], 0));

/** Estimated end-to-end latency for m calls on one back-end. */
export const estimateTotal = ({ oneoff_ms, cold_ms, model, N, m, featFn }) =>
  oneoff_ms + cold_ms + m * predictCall(model, N, featFn);

/**
 * Rank back-ends by estimated total latency (ascending).
 * models[name] = {coef}; oneoff[name] = load+setup (0 if already initialised);
 * cold[name] = first-call penalty.
 */
export function rankBackends({ names, models, oneoff, cold, N, m, featFn }) {
  return names
    .filter((n) => models[n])
    .map((n) => ({ name: n, est_ms: estimateTotal({ oneoff_ms: oneoff[n] ?? 0, cold_ms: cold[n] ?? 0, model: models[n], N, m, featFn }) }))
    .sort((a, b) => a.est_ms - b.est_ms);
}

/** M0 baseline: static threshold table [[maxN, backend], ...] with availability fallback. */
export function staticChoice(table, N, available) {
  const want = (table.find(([maxN]) => N <= maxN) ?? table[table.length - 1])[1];
  const start = CHAIN.indexOf(want);
  for (let i = Math.max(0, start); i < CHAIN.length; i++) if (available.includes(CHAIN[i])) return CHAIN[i];
  return "js";
}

/** Hand-set defaults (replace with thresholds measured on your reference machine). */
export const DEFAULT_STATIC_TABLE = {
  matmul: [[96, "js"], [192, "simd"], [Infinity, "webgpu"]],
  saxpy: [[2 ** 16, "js"], [Infinity, "simd"]],
};

/**
 * Time-targeted probing. Climb the size ladder until one call takes >= targetMs (so the compute term is
 * visible above fixed overheads) but never spend more than maxProbeMs on one back-end/kernel.
 * Small probes alone cannot identify the compute cost of a fast back-end (e.g. WebGPU), which makes
 * extrapolation to large sizes unreliable - this rule fixes that.
 */
export const PROBE_POLICY = { targetMs: 25, minPoints: 4, maxProbeMs: 400, repeats: 3 };
export function shouldStopProbing(nPoints, lastMedianMs, spentMs, policy = PROBE_POLICY) {
  if (nPoints < policy.minPoints) return false;
  return lastMedianMs >= policy.targetMs || spentMs >= policy.maxProbeMs;
}
