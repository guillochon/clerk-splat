// Convert a Brush .ply to GaussianSplats3D's compressed .ksplat (smaller, faster to load).
//   npm install @mkkellogg/gaussian-splats-3d@0.4.7 three@0.160.0
//   node viewer/make-ksplat.mjs in.ply out.ksplat [compressionLevel=1] [shDegree=1]
// compressionLevel 1 = 16-bit positions/scales/colours; 2 also packs SH to 8 bits.
// Splats with alpha < 5/255 are dropped, same as the viewer's splatAlphaRemovalThreshold.
import { readFileSync, writeFileSync } from "node:fs";
globalThis.window ??= globalThis;   // the library calls window.setTimeout
const GS = await import("@mkkellogg/gaussian-splats-3d");

const [input, output, level = "1", sh = "1"] = process.argv.slice(2);
if (!input || !output) {
  console.error("usage: node make-ksplat.mjs in.ply out.ksplat [compressionLevel] [shDegree]");
  process.exit(1);
}
const ply = readFileSync(input);
const data = ply.buffer.slice(ply.byteOffset, ply.byteOffset + ply.byteLength);
const splatBuffer = await GS.PlyLoader.loadFromFileData(data, 5, +level, true, +sh);
writeFileSync(output, Buffer.from(splatBuffer.bufferData));
console.log(`${input} (${(ply.byteLength / 1e6).toFixed(1)} MB) -> ${output} ` +
            `(${(splatBuffer.bufferData.byteLength / 1e6).toFixed(1)} MB, ${splatBuffer.getSplatCount()} splats)`);
