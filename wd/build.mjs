// Cross-platform build of the three WebAssembly modules (same as the Makefile).
// Needs clang with the wasm32 target + wasm-ld.  Windows: install LLVM (llvm.org releases).
// macOS: `brew install llvm lld` and run with CLANG=$(brew --prefix llvm)/bin/clang
// Linux: `sudo apt install clang lld`
import { spawnSync } from "node:child_process";
import fs from "node:fs";

const clang = process.env.CLANG ?? "clang";
const common = ["--target=wasm32", "-O3", "-nostdlib", "-ffreestanding", "-fno-builtin", "-Wall"];
const link = ["-Wl,--no-entry", "-Wl,--export=matmul_rows", "-Wl,--export=saxpy_range", "-Wl,--import-memory",
  "-Wl,--stack-first", "-Wl,-z,stack-size=4096", "-Wl,--initial-memory=1048576", "-Wl,--max-memory=1073741824"];
const builds = [
  ["kernels_scalar.wasm", []],
  ["kernels_simd.wasm", ["-msimd128"]],
  ["kernels_simd_mt.wasm", ["-msimd128", "-matomics", "-mbulk-memory", "-Wl,--shared-memory"]],
];
fs.mkdirSync("public/wasm", { recursive: true });
for (const [out, flags] of builds) {
  const args = [...common, ...flags, "src/kernels.c", "-o", `public/wasm/${out}`, ...link];
  const r = spawnSync(clang, args, { stdio: "inherit" });
  if (r.status !== 0) { console.error(`\nbuild of ${out} failed (is clang with wasm32 + wasm-ld installed?)`); process.exit(1); }
  console.log(`built public/wasm/${out} (${fs.statSync(`public/wasm/${out}`).size} bytes)`);
}
