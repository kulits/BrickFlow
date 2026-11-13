import * as THREE from 'three';
import { Viewer, makePart, setPose, LD, occluderIdMaterial } from './viewer.js';
import { manipulate } from './manipulate.js';
import { guided } from './guided.js';
import { lerpPose } from './energy.js';

const IDENT = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], UP = new THREE.Vector3(0, 1, 0), FLIP = new THREE.Vector3(1, -1, -1);
const lin = (c) => new THREE.Color().setRGB(...c, THREE.LinearSRGBColorSpace);
const glsl = (c) => `vec3(${c.map((x) => x.toFixed(4)).join(', ')})`;

export async function pairDemo(el) {
  const S = await (await fetch('data/pair_static.json')).json(), v = new Viewer(el.querySelector('.stage'), { inkWidth: 1.5, fov: 10 });
  const occluder = (mesh) => { mesh.userData.occluder = true; v.root.add(mesh); return mesh; };

  const lower = await makePart('3001', 71, { cad: true });
  const clay = new THREE.MeshStandardMaterial({ color: lin(S.clay_rgb), roughness: 0.55, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1, side: THREE.DoubleSide });
  lower.children.forEach((o) => (o.material = clay));
  setPose(lower, IDENT);
  v.root.add(lower);

  const { x0, x1, z0, z1 } = S.paint, y = S.sheet_y, corners = [[x0, z0], [x1, z0], [x1, z1], [x0, z1]];
  const quad = new THREE.BufferGeometry().setIndex([0, 2, 1, 0, 3, 2]);
  quad.setAttribute('position', new THREE.Float32BufferAttribute(corners.flatMap(([x, z]) => [x, -y, -z]), 3));
  quad.setAttribute('uv', new THREE.Float32BufferAttribute(corners.flatMap(([x, z]) => [(x - x0) / (x1 - x0), 1 - (z - z0) / (z1 - z0)]), 2));
  quad.computeVertexNormals();
  const field = await new THREE.TextureLoader().loadAsync('assets/pair_field.avif');
  field.colorSpace = THREE.SRGBColorSpace;
  field.anisotropy = 8;
  const sheet = occluder(new THREE.Mesh(quad, new THREE.MeshStandardMaterial({ map: field, transparent: true, roughness: 0.8, alphaTest: 0.02, side: THREE.DoubleSide, depthWrite: false })));
  sheet.renderOrder = -1;
  sheet.userData.idMaterial = occluderIdMaterial(field, 0.5);
  const core = new THREE.Mesh(quad, new THREE.MeshBasicMaterial({ map: field, alphaTest: 0.6, colorWrite: false, side: THREE.DoubleSide }));
  core.userData.noInk = true;
  v.root.add(core);

  const ink = new THREE.MeshStandardMaterial({ color: lin(S.ink_rgb), roughness: 0.4 }), dome = new THREE.SphereGeometry(S.bead_r, 24, 12, 0, 2 * Math.PI, 0, Math.PI / 2);
  for (const b of S.beads) occluder(new THREE.Mesh(dome, ink)).position.set(b[0], -y, -b[2]);
  for (const a of S.arcs) occluder(new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(a.map((p) => new THREE.Vector3(p[0], -y, -p[2]))), 96, S.arc_r, 10, false), ink));

  const upper = await makePart('3003', 15, { cad: true });
  const uD = { value: new THREE.Matrix4() }, uFar = { value: 1 }, uRange = { value: new THREE.Vector3() }, { stops, scale, gamma } = S.dpaint;
  const ramp = stops.slice(1).map(([at, c], i) => `c = mix(c, ${glsl(c)}, smoothstep(${stops[i][0].toFixed(3)}, ${at.toFixed(3)}, u));`).join('\n');
  const paint = new THREE.MeshPhysicalMaterial({ roughness: 0.15, clearcoat: 0.6, transparent: true, opacity: 0.9, side: THREE.DoubleSide });
  paint.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, { uD, uFar, uRange });
    sh.vertexShader = sh.vertexShader.replace('#include <common>', '#include <common>\nvarying vec3 vLocal;').replace('#include <begin_vertex>', '#include <begin_vertex>\nvLocal = position;');
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vLocal;\nuniform mat4 uD;\nuniform float uFar;\nuniform vec3 uRange;\nvec3 rampC;')
      .replace('vec4 diffuseColor = vec4( diffuse, opacity );', `
        float len = length((uD * vec4(vLocal, 1.0)).xyz), uAbs = pow(clamp(len / ${scale.toFixed(2)}, 0.0, 1.0), ${gamma.toFixed(3)});
        float uRel = clamp((len - uRange.x) / max(uRange.y - uRange.x, 1e-3), 0.0, 1.0);
        float u = uFar > 0.5 ? 1.0 : mix(uAbs, uRel, uRange.z);
        vec3 c = ${glsl(stops[0][1])};
        ${ramp}
        rampC = c;
        vec4 diffuseColor = vec4(0.25 * c, opacity);`)
      .replace('#include <emissivemap_fragment>', '#include <emissivemap_fragment>\ntotalEmissiveRadiance += 0.8 * rampC;');
  };
  upper.children.forEach((o) => (o.material = paint));
  v.root.add(upper);

  const samples = [[-10, -10], [-10, 10], [10, -10], [10, 10]].map(([x, z]) => [x, -4.3, z]).concat([[0, 0], [0, -10], [0, 10], [-10, 0], [10, 0]].map(([x, z]) => [x, -0.3, z]));
  for (const t of [-12, 0, 12]) for (const h of [6, 18]) samples.push([t, h, -20.3], [t, h, 20.3], [-20.3, h, t], [20.3, h, t]);
  const needles = occluder(new THREE.InstancedMesh(new THREE.ConeGeometry(0.38, 1, 8, 1).translate(0, 0.5, 0), new THREE.MeshBasicMaterial({ color: 0x1a1a1e }), samples.length));
  needles.frustumCulled = false;

  const start = Float64Array.from(S.pose2x2), M = new THREE.Matrix4(), q = new THREE.Quaternion();
  let pose = start, target = null, man = null;
  function refresh() {
    setPose(upper, pose);
    man?.follow();
    const P = new THREE.Matrix4().set(...pose), T = new THREE.Matrix4().set(...(target || pose));
    uD.value.set(...Array.from(target || pose, (x, i) => x - pose[i]));
    uFar.value = target ? 0 : 1;
    let lo = Infinity, hi = 0;
    samples.forEach((s, k) => {
      const here = new THREE.Vector3(...s).applyMatrix4(P), d = new THREE.Vector3(...s).applyMatrix4(T).sub(here).multiply(FLIP), len = d.length();
      lo = Math.min(lo, len); hi = Math.max(hi, len);
      needles.setMatrixAt(k, len < 0.4 ? M.makeScale(0, 0, 0) : M.compose(here.applyMatrix4(LD), q.setFromUnitVectors(UP, d.normalize()), new THREE.Vector3(1, Math.min(5.5, 0.22 * len + 0.8), 1)));
    });
    needles.instanceMatrix.needsUpdate = true;
    uRange.value.set(lo, hi, 0.85 * Math.min(1, (hi - lo) / 6));
    v.request();
  }
  refresh();
  v.frame(new THREE.Box3(new THREE.Vector3(-56, -26, -38), new THREE.Vector3(56, 36, 38)), new THREE.Vector3(0.6, 0.53, -0.6), 0.74);

  const guide = el.querySelector('[data-act="guide"]');
  const g = await guided(el, {
    payload: () => [{ name: '3001', pose: IDENT }, { name: '3003', pose: Array.from(pose) }], frame: 0, speed: 4, already: 'already seated',
    status: (st, ok) => {
      guide.disabled = !ok && !!st.isolated;
      return ok ? 'seated' : st.isolated ? 'too far from any seat (drag it closer or press Reset)' : st.stalled ? 'stuck before seating (drag it or press Reset)' : 'not seated';
    },
    onEval: (m) => { target = m.target; refresh(); },
    onPath: (a, f, b) => {
      pose = lerpPose(a.poses[1], b.poses[1], f, g.com('3003'));
      target = a.target && b.target ? a.target.map((x, e) => x + f * (b.target[e] - x)) : (f < 0.5 ? a : b).target;
      refresh();
    },
    onGuide: () => man.detach(),
  });
  man = manipulate(v, [{ group: upper, com: g.com('3003'), get: () => pose, set: (P) => { pose = P; } }], {
    onStart: g.clear, onChange: () => { refresh(); g.evaluate(); }, onEnd: g.evaluate,
    bound: { center: new THREE.Vector3(...g.com('3001')).applyMatrix4(LD), radius: 40 },
  });
  el.querySelector('[data-act="reset"]').addEventListener('click', () => { g.clear(); man.detach(); pose = start; v.resetView(); refresh(); g.evaluate(); });
}
