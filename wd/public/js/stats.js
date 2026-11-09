export function quantile(a, q) {
  const s = Float64Array.from(a).sort();
  const pos = (s.length - 1) * q, lo = Math.floor(pos), hi = Math.ceil(pos);
  return s[lo] + (s[hi] - s[lo]) * (pos - lo);
}
export const median = (a) => quantile(a, 0.5);
export const iqr = (a) => quantile(a, 0.75) - quantile(a, 0.25);
export const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;

/** 95% bootstrap confidence interval for the median. */
export function bootstrapCI(a, B = 1000, alpha = 0.05, rand = Math.random) {
  if (a.length < 3) return [Math.min(...a), Math.max(...a)];
  const meds = new Float64Array(B);
  const tmp = new Float64Array(a.length);
  for (let b = 0; b < B; b++) {
    for (let i = 0; i < a.length; i++) tmp[i] = a[Math.floor(rand() * a.length)];
    meds[b] = median(tmp);
  }
  return [quantile(meds, alpha / 2), quantile(meds, 1 - alpha / 2)];
}
