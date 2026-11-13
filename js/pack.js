import * as THREE from "three";
import { MeshoptDecoder } from "https://cdn.jsdelivr.net/npm/meshoptimizer@1.3.0/meshopt_decoder.mjs";
const id = (k) => String(k).padStart(2, "0");
const cache = new Map();
async function gunzip(buf) {
  const u = new Uint8Array(buf);
  if (u[0] !== 31 || u[1] !== 139) return buf;
  return new Response(new Blob([buf]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
}
function loadPack(k) {
  if (!cache.has(k)) cache.set(k, fetch(`data/traj/${id(k)}.bin.gz`).then((r) => {
    if (!r.ok) throw new Error(`trajectory ${id(k)}: HTTP ${r.status}`);
    return r.arrayBuffer();
  }).then(gunzip).then(decode));
  return cache.get(k);
}
async function decode(buf) {
  await MeshoptDecoder.ready;
  const hl = new DataView(buf).getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hl)));
  const n = header.parts.length, B = {};
  for (const [key, kind, off, len, count] of header.buffers) {
    const src = new Uint8Array(buf, 4 + hl + off, len);
    if (kind === "vertex") {
      const o = new Uint8Array(count * 8);
      MeshoptDecoder.decodeVertexBuffer(o, count, 8, src);
      B[key] = new Int16Array(o.buffer);
    } else if (kind === "index") {
      const o = new Uint8Array(count * 4);
      MeshoptDecoder.decodeIndexBuffer(o, count, 4, src);
      B[key] = new Uint32Array(o.buffer);
    } else if (kind === "sequence") {
      const o = new Uint8Array(count * 4);
      MeshoptDecoder.decodeIndexSequence(o, count, 4, src);
      B[key] = new Uint32Array(o.buffer);
    } else {
      const a = new Int16Array(src.slice().buffer), w = n * (kind === "delta3" ? 3 : 4);
      for (let i = w; i < a.length; i++) a[i] += a[i - w];
      B[key] = a;
    }
  }
  const meshes = {};
  for (const [part, codes] of Object.entries(header.meshes)) {
    const v = B[`${part}/v`], pos = new Float32Array(v.length / 4 * 3);
    for (let i = 0; i < pos.length; i++) pos[i] = v[4 * Math.floor(i / 3) + i % 3] / 32;
    const indexed = (key) => new THREE.BufferGeometry().setAttribute("position", new THREE.BufferAttribute(pos, 3)).setIndex(new THREE.BufferAttribute(B[key], 1));
    const prims = codes.map((code, k) => {
      const g = indexed(`${part}/t${k}`), geometry = g.toNonIndexed();
      geometry.computeVertexNormals();
      g.dispose();
      return { geometry, code };
    });
    meshes[part] = { prims, edges: [indexed(`${part}/e`)] };
  }
  const LDQ = new THREE.Quaternion(1, 0, 0, 0);
  const frames = (key, steps) => Array.from({ length: steps }, (_, s) => Array.from({ length: n }, (_2, k) => {
    const X = B[`${key}/x`], Q = B[`${key}/q`], a = 3 * (s * n + k), b = 4 * (s * n + k);
    return { p: new THREE.Vector3(X[a], -X[a + 1], -X[a + 2]).multiplyScalar(header.tstep), q: new THREE.Quaternion(Q[b], Q[b + 1], Q[b + 2], Q[b + 3]).normalize().premultiply(LDQ) };
  }));
  return { header, meshes, X: frames("X", header.nsteps[0]), X1: frames("X1", header.nsteps[1]), S: header.has_snap ? frames("S", 1)[0] : null };
}
export {
  id,
  loadPack
};
