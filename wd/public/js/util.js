// Small helpers shared by browser and Node (tests / analysis).
export const now = () => performance.now();
// Yield to the event loop WITHOUT setTimeout (browsers throttle timers to ~1/s in background tabs).
export const sleep0 = () => new Promise((resolve) => {
  const c = new MessageChannel();
  c.port1.onmessage = () => { c.port1.close(); resolve(); };
  c.port2.postMessage(0);
});

/** Read a file as bytes in either environment (http(s) URL in browser, file: URL in Node). */
export async function readBytes(url) {
  if (url.protocol === "file:") {
    const fs = await import("node:fs/promises");
    const { fileURLToPath } = await import("node:url");
    return new Uint8Array(await fs.readFile(fileURLToPath(url)));
  }
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url} -> ${r.status}`);
  return new Uint8Array(await r.arrayBuffer());
}

/** Deterministic PRNG (mulberry32) so every back-end gets identical inputs. */
export function rng(seed = 1) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function shuffle(arr, rand = Math.random) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

/** max|a-b| / max|b|  (relative to the largest reference value). */
export function relMaxError(out, ref) {
  let maxDiff = 0, maxRef = 0;
  for (let i = 0; i < ref.length; i++) {
    const d = Math.abs(out[i] - ref[i]);
    if (d > maxDiff || Number.isNaN(d)) maxDiff = Number.isNaN(d) ? Infinity : d;
    const r = Math.abs(ref[i]);
    if (r > maxRef) maxRef = r;
  }
  return maxDiff / (maxRef || 1);
}
