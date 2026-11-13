import * as THREE from 'three';
import { Viewer, makePart, setPose, glassMaterial, eyeMaterial } from './viewer.js';
import { manipulate } from './manipulate.js';
import { guided } from './guided.js';
import { lerpPose, twist, comWorld } from './energy.js';

const BODY = { 3021: [0, 4, 0] }, HUB = [false, true, true, false, true, false], SHAKE_DEG = 12, SHAKE_LDU = 4.5;
const WIRE = {
  charged: { r: [0.78, 0.78], c: [2.45, 2.05, 0.85], e: [2.1, 1.45, 0.3], glow: [1, 0.62, 0.12], g: [1, 1] },
  uncharged: { r: [0.3, 0.4], c: [2, 0.85, 0.42], e: [1.3, 0.32, 0.12], glow: [1, 0.32, 0.1], g: [0.1, 0.25] },
  inactive: { r: [0.26, 0.34], c: [1.8, 0.75, 0.38], e: [1.15, 0.28, 0.1], glow: [1, 0.32, 0.1], g: [0.06, 0.18] },
};
const UP = new THREE.Vector3(0, 1, 0), V3 = (x) => new THREE.Vector3(...x), M4 = (P) => new THREE.Matrix4().set(...P);
const gauss = () => Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random());
const memo = (make) => { const made = new Map(); return (...key) => made.get(key.join()) ?? made.set(key.join(), make(...key)).get(key.join()); };

const coreMaterial = memo((c, e) => new THREE.ShaderMaterial({
  uniforms: { cc: { value: V3(c) }, ce: { value: V3(e) } },
  vertexShader: 'varying vec3 vN, vP; void main(){ vec4 wp = modelMatrix * vec4(position, 1.0); vP = wp.xyz; vN = normalize(mat3(modelMatrix) * normal); gl_Position = projectionMatrix * viewMatrix * wp; }',
  fragmentShader: 'precision highp float; varying vec3 vN, vP; uniform vec3 cc, ce; void main(){ float t = abs(dot(normalize(vN), normalize(cameraPosition - vP))); gl_FragColor = vec4(mix(ce, cc, smoothstep(0.15, 0.95, t)), 1.0); }',
}));
const glowMaterial = memo((c, a) => new THREE.MeshBasicMaterial({ color: new THREE.Color().setRGB(...c.map((x) => x * a), THREE.LinearSRGBColorSpace), toneMapped: false }));

function simplify(pts) {
  const q = pts.map(V3).filter((x, i, all) => !i || x.distanceTo(all[i - 1]) > 1e-3);
  return q.filter((x, j) => {
    if (!j || j === q.length - 1) return true;
    const u = x.clone().sub(q[j - 1]), w = q[j + 1].clone().sub(x);
    return u.dot(w) < 0 || u.normalize().cross(w.normalize()).length() > 1e-3;
  }).map((x) => x.toArray());
}

export async function networkDemo(el, { states }) {
  const v = new Viewer(el.querySelector('.stage'), { inkWidth: 1.4, fov: 10, glowSigma: 1.3, glowGain: 0.9, inkDepth: false });
  const act = (k) => el.querySelector(`[data-act="${k}"]`), glass = glassMaterial({ t0: 0.87, edge: 0.3, kdiff: 0.05, tint: [0.9, 0.95, 1.06] });
  const start = states.paper.parts.map((p) => Float64Array.from(p.pose)), parts = [];
  for (const p of states.gt.parts) {
    const group = await makePart(p.name, p.code, { cad: true });
    for (const o of [...group.children]) {
      o.material = glass;
      o.castShadow = true;
      if (p.name !== '3003pe3') continue;
      const eye = new THREE.Mesh(o.geometry, eyeMaterial('#3b3c3f', '#f3f3f3')), shade = new THREE.Mesh(o.geometry, eyeMaterial('#000000', '#000000'));
      eye.renderOrder = 10;
      group.add(eye);
      Object.assign(shade.material, { transparent: false, depthWrite: true, polygonOffset: false });
      Object.assign(shade, { matrixAutoUpdate: false, frustumCulled: false, onBeforeRender: () => shade.matrixWorld.copy(eye.matrixWorld) });
      v.glowScene.add(shade);
    }
    v.root.add(group);
    parts.push({ name: p.name, pose: Float64Array.from(p.pose), group });
  }

  let man = null, pristine = true;
  const sync = () => { parts.forEach((p) => setPose(p.group, p.pose)); man?.follow(); v.request(); };
  const place = (poses) => { parts.forEach((p, k) => (p.pose = poses[k])); sync(); };
  sync();
  const box = new THREE.Box3().setFromObject(v.root), floor = box.min.y - 0.05;
  place(start);
  v.frame(box.union(new THREE.Box3().setFromObject(v.root)).expandByScalar(6), new THREE.Vector3(-0.41, 0.242, -0.879), 0.92);

  v.renderer.shadowMap.enabled = true;
  v.renderer.shadowMap.type = THREE.VSMShadowMap;
  const sun = new THREE.DirectionalLight(0xffffff, 1e-4);
  sun.position.set(465, 650, -169).normalize().multiplyScalar(300);
  sun.castShadow = true;
  Object.assign(sun.shadow.camera, { left: -160, right: 160, top: 160, bottom: -160, near: 1, far: 700 });
  Object.assign(sun.shadow, { radius: 36, blurSamples: 24, bias: -5e-4 });
  sun.shadow.mapSize.set(1024, 1024);
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(6000, 6000).rotateX(-Math.PI / 2), new THREE.ShadowMaterial({ opacity: 0.09 }));
  ground.position.y = floor;
  ground.receiveShadow = true;
  ground.userData.noInk = true;
  v.scene.add(sun, sun.target, ground);

  const wires = new THREE.Group(), glow = new THREE.Group(), tube = new THREE.CylinderGeometry(1, 1, 1, 16, 1, true), ball = new THREE.SphereGeometry(1, 16, 12);
  v.scene.add(wires);
  v.glowScene.add(glow);
  function route(k, x) {
    const P = M4(parts[k].pose), l = V3(x).applyMatrix4(P.clone().invert()).toArray(), hub = BODY[parts[k].name] || [0, 12, 0];
    const local = HUB[k] ? [l, [l[0], hub[1], l[2]], [hub[0], hub[1], l[2]], hub] : [l, [l[0], l[1] + Math.sign(hub[1] - l[1]) * Math.min(3, Math.abs(hub[1] - l[1])), l[2]]];
    return local.map((p) => V3(p).applyMatrix4(P).toArray());
  }
  function addTube(pts, style, u) {
    const P = pts.map((x) => new THREE.Vector3(x[0], -x[1], -x[2])), r = style.r[0] + u * (style.r[1] - style.r[0]);
    const mats = [[wires, coreMaterial(style.c, style.e)], [glow, glowMaterial(style.glow, style.g[0] + u * (style.g[1] - style.g[0]))]];
    const add = (geometry, at, turn, size) => mats.forEach(([group, material]) => {
      const m = new THREE.Mesh(geometry, material);
      m.matrixAutoUpdate = false;
      m.matrix.compose(at, turn, size);
      m.userData.noInk = true;
      group.add(m);
    });
    P.forEach((a, k) => {
      add(ball, a, new THREE.Quaternion(), new THREE.Vector3(r, r, r));
      const d = P[k + 1]?.clone().sub(a), L = d?.length();
      if (L > 1e-3) add(tube, a.clone().add(P[k + 1]).multiplyScalar(0.5), new THREE.Quaternion().setFromUnitVectors(UP, d.normalize()), new THREE.Vector3(r, L, r));
    });
  }
  function drawWires(seats) {
    wires.clear();
    glow.clear();
    v.request();
    if (!act('net').checked) return;
    const weak = 1 / (16 * parts.length), exact = (s) => s.c <= 1e-6 && s.w >= 1, root = parts.map((_, k) => k), find = (k) => (root[k] === k ? k : (root[k] = find(root[k])));
    for (const s of seats) if (exact(s)) root[find(s.i)] = find(s.j);
    for (const s of seats) {
      if (exact(s)) { addTube(simplify([...route(s.i, s.pa).reverse(), ...route(s.j, s.pb)]), WIRE.charged, 1); continue; }
      const u = Math.min(1, Math.log(s.g / weak) / Math.log(0.3 / weak));
      if (s.w <= 0 || !(u > 0) || (find(s.i) === find(s.j) && s.c > 4)) continue;
      addTube([s.pa, s.pb], WIRE.uncharged, u);
      addTube(simplify(route(s.i, s.pa)), WIRE.inactive, u);
      addTube(simplify(route(s.j, s.pb)), WIRE.inactive, u);
    }
  }
  function carry(seats, from, to) {
    const T = from.map((P, k) => M4(to[k]).multiply(M4(P).invert())), move = (x, k) => V3(x).applyMatrix4(T[k]).toArray();
    return seats.map((s) => ({ ...s, pa: move(s.pa, s.i), pb: move(s.pb, s.j) }));
  }

  let seats = [];
  const g = await guided(el, {
    payload: () => parts.map((p) => ({ name: p.name, pose: Array.from(p.pose) })), speed: 2.5, already: 'already connected',
    status: (st, ok) => (ok ? 'connected'
      : st.isolated === 1 ? 'a brick is too far from the rest (drag it closer or press Reset)'
      : st.isolated ? `${st.isolated} bricks are too far from the rest (drag them closer or press Reset)`
      : st.stalled ? 'stuck before connecting (drag a brick or press Reset)'
      : pristine ? 'not connected (the noisy estimate)' : 'not connected'),
    onEval: (m) => drawWires(seats = m.seats),
    onPath: (a, f, b) => {
      const near = f < 0.5 ? a : b, now = a.poses.map((A, k) => lerpPose(A, b.poses[k], f, g.com(parts[k].name)));
      place(now);
      drawWires(seats = carry(near.seats, near.poses, now));
    },
    onGuide: () => { man.detach(); pristine = false; },
  });
  man = manipulate(v, parts.map((p) => ({ group: p.group, com: g.com(p.name), get: () => p.pose, set: (P) => { p.pose = P; } })), {
    slide: 'view', onStart: () => { g.clear(); pristine = false; }, onChange: () => { sync(); g.evaluate(); }, onEnd: g.evaluate,
  });

  act('shake').addEventListener('click', () => {
    g.clear();
    pristine = false;
    place(parts.map((p) => {
      const axis = V3([gauss(), gauss(), gauss()]).normalize().multiplyScalar(gauss() * SHAKE_DEG * Math.PI / 180).toArray();
      return twist(p.pose, comWorld(p.pose, [0, ...g.com(p.name)]), axis, [SHAKE_LDU * gauss(), 0.7 * SHAKE_LDU * gauss(), SHAKE_LDU * gauss()]);
    }));
    g.evaluate();
  });
  act('reset').addEventListener('click', () => { g.clear(); man.detach(); pristine = true; place(start); v.resetView(); g.evaluate(); });
  act('net').addEventListener('change', () => drawWires(seats));
}
