// Static dev server with the headers that WebAssembly threads (SharedArrayBuffer) require.
//   node server.js [port]                         http://localhost:8080 (localhost counts as a secure context)
//   node server.js [port] --https cert.pem key.pem   HTTPS, e.g. for opening it from a phone over Wi-Fi
import http from "node:http";
import https from "node:https";
import os from "node:os";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), "public");
const args = process.argv.slice(2);
const port = Number(args.find((a) => /^\d+$/.test(a)) ?? 8080);
const httpsAt = args.indexOf("--https");
const types = { ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
  ".wasm": "application/wasm", ".json": "application/json", ".css": "text/css" };

export async function staticHandler(req, res) {
  const headers = {
    "Cross-Origin-Opener-Policy": "same-origin",     // together these two make the page
    "Cross-Origin-Embedder-Policy": "require-corp",  // cross-origin isolated => threads + fine timers
    "Cache-Control": "no-store",
  };
  try {
    let p = decodeURIComponent(new URL(req.url, "http://x").pathname);
    if (p.endsWith("/")) p += "index.html";
    const file = path.normalize(path.join(root, p));
    if (!file.startsWith(root)) { res.writeHead(403, headers); return res.end("forbidden"); }
    const body = await fs.readFile(file);
    res.writeHead(200, { ...headers, "Content-Type": types[path.extname(file)] ?? "application/octet-stream" });
    res.end(body);
  } catch {
    res.writeHead(404, headers); res.end("not found");
  }
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  const isHttps = httpsAt >= 0;
  const server = isHttps
    ? https.createServer({ cert: await fs.readFile(args[httpsAt + 1]), key: await fs.readFile(args[httpsAt + 2]) }, staticHandler)
    : http.createServer(staticHandler);

  server.listen(port, "0.0.0.0", () => {
    const scheme = isHttps ? "https" : "http";
    console.log(`serving ${root}`);
    console.log(`  this PC : ${scheme}://localhost:${port}`);
    for (const list of Object.values(os.networkInterfaces()))
      for (const i of list ?? []) if (i.family === "IPv4" && !i.internal) console.log(`  network : ${scheme}://${i.address}:${port}${isHttps ? "" : "   (NOT a secure context; see README)"}`);
  });

}
