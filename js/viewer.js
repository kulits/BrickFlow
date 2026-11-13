import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { toCreasedNormals } from 'three/addons/utils/BufferGeometryUtils.js';
import { MeshoptDecoder } from 'https://cdn.jsdelivr.net/npm/meshoptimizer@1.3.0/meshopt_decoder.mjs';

export const LD = new THREE.Matrix4().makeScale(1, -1, -1);
const colors = await (await fetch('data/ldcolors.json')).json();
const memo = (make) => { const made = new Map(); return (key) => made.get(key) ?? made.set(key, make(key)).get(key); };

const surfaceMaterial = memo((code) => {
  const [hex, alpha = 255] = [colors[code] ?? '#888888'].flat();
  return new THREE.MeshStandardMaterial({ color: hex, roughness: 0.55, transparent: alpha < 255, opacity: alpha / 255, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1, side: THREE.DoubleSide });
});
const edgeMaterial = new THREE.LineBasicMaterial({ color: 0x0a0a0a });

const loader = new GLTFLoader().setMeshoptDecoder(MeshoptDecoder);
const cadPart = memo((name) => loader.loadAsync(`parts_cad/${name}.glb`).then((gltf) => {
  const prims = [];
  gltf.scene.traverse((o) => {
    if (!o.isMesh) return;
    const p = o.geometry.attributes.position;
    o.geometry.setAttribute('position', new THREE.Float32BufferAttribute(Float32Array.from({ length: 3 * p.count }, (_, i) => p.getComponent(Math.floor(i / 3), i % 3)), 3));
    const g = o.geometry.toNonIndexed().applyMatrix4(o.matrixWorld);
    g.deleteAttribute('normal');
    prims.push({ geometry: toCreasedNormals(g, Math.PI / 4), code: +(/color_(\d+)/.exec(o.material.name)?.[1] ?? 16) });
  });
  return { prims };
}));

export function makePartFrom({ prims, edges = [] }, code, name) {
  const group = new THREE.Group();
  group.matrixAutoUpdate = false;
  group.userData.name = name;
  prims.forEach((p) => group.add(new THREE.Mesh(p.geometry, surfaceMaterial(p.code === 16 ? code : p.code))));
  edges.forEach((e) => group.add(new THREE.LineSegments(e, edgeMaterial)));
  return group;
}
export const makePart = async (name, code) => makePartFrom(await cadPart(name === '3003pe3' ? '3003' : name), code, name);
export function setPose(group, pose) {
  group.matrix.multiplyMatrices(LD, new THREE.Matrix4().set(...pose));
  group.matrixWorldNeedsUpdate = true;
}

const QUAD_VS = 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }';
const INK_FS = `precision highp float;
varying vec2 vUv;
uniform sampler2D tColor, tId, tDepth, tGlow;
uniform vec2 texel;
uniform float radius, cNear, cFar, inkDepth, glowGain;
const vec3 OCC = vec3(200.0 / 255.0), INK = vec3(0.00425);
float LZ(float d){ return cNear * cFar / (cFar - d * (cFar - cNear)); }
float D(vec2 uv){ return texture2D(tDepth, uv).x; }
vec3 I(vec2 uv){ return texture2D(tId, uv).rgb; }
bool isOcc(vec3 c){ return distance(c, OCC) < 0.002; }
float edgeAt(vec2 p, vec2 s){
  vec3 a = I(p), b = I(p + s);
  bool oa = isOcc(a), ob = isOcc(b);
  if (oa || ob) {
    vec3 ink = oa ? b : a;
    if ((oa && ob) || dot(ink, ink) == 0.0) return 0.0;
    return LZ(oa ? D(p) : D(p + s)) - LZ(oa ? D(p + s) : D(p)) > 0.6 ? 1.0 : 0.0;
  }
  bool ia = dot(a, a) > 0.0, ib = dot(b, b) > 0.0;
  if (!ia && !ib) return 0.0;
  if (ia != ib || distance(a, b) > 0.002) return 1.0;
  if (inkDepth < 0.5) return 0.0;
  float d0 = D(p - s), d1 = D(p), d2 = D(p + s), d3 = D(p + 2.0 * s);
  return abs(d2 - d1) > 4.0 * max(abs(d1 - d0), abs(d3 - d2)) + 2e-6 ? 1.0 : 0.0;
}
void main(){
  vec3 col = texture2D(tColor, vUv).rgb + glowGain * texture2D(tGlow, vUv).rgb;
  float e = 0.0;
  int R = int(ceil(radius));
  for (int dy = -3; dy <= 3; dy++) for (int dx = -3; dx <= 3; dx++) {
    if (abs(dx) > R || abs(dy) > R) continue;
    vec2 o = vec2(float(dx), float(dy));
    float w = clamp(radius + 0.5 - length(o + 0.5), 0.0, 1.0);
    if (w <= 0.0) continue;
    vec2 p = vUv + o * texel;
    e = max(e, w * max(edgeAt(p, vec2(texel.x, 0.0)), edgeAt(p, vec2(0.0, texel.y))));
  }
  gl_FragColor = vec4(mix(col, INK, e), 1.0);
  #include <colorspace_fragment>
}`;
const BLUR_FS = `precision highp float; varying vec2 vUv; uniform sampler2D tSrc; uniform vec2 dir; uniform float sigma;
void main(){
  float st = max(1.0, 3.0 * sigma / 8.0), wsum = 0.0; vec3 acc = vec3(0.0);
  for (int i = -8; i <= 8; i++) { float x = float(i) * st, w = exp(-0.5 * x * x / (sigma * sigma)); acc += w * texture2D(tSrc, vUv + dir * x).rgb; wsum += w; }
  gl_FragColor = vec4(acc / wsum, 1.0);
}`;
const pass = (fragmentShader, uniforms) => {
  const material = new THREE.ShaderMaterial({ vertexShader: QUAD_VS, fragmentShader, uniforms, depthTest: false, depthWrite: false });
  return { u: material.uniforms, scene: new THREE.Scene().add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material)) };
};
const idMaterial = memo((k) => new THREE.MeshBasicMaterial({ color: new THREE.Color().setRGB((k >> 16 & 255) / 255, (k >> 8 & 255) / 255, (k & 255) / 255, THREE.LinearSRGBColorSpace), side: THREE.DoubleSide }));

export class Viewer {
  constructor(container, { fov = 30, inkWidth = 1.25, inkDepth = true, glowSigma = 1.2, glowGain = 1 } = {}) {
    Object.assign(this, { container, inkWidth, glowSigma, glowGain, touchGrab: () => false, onFrame: [], needs: true, visible: true });
    const r = this.renderer = new THREE.WebGLRenderer({ powerPreference: 'high-performance' }), canvas = r.domElement;
    r.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
    r.setClearColor(0xffffff, 1);
    Object.assign(canvas.style, { display: 'block', width: '100%', height: '100%' });
    container.appendChild(canvas);

    const touches = new Map(), capture = { capture: true };
    container.addEventListener('pointerdown', (e) => e.pointerType === 'touch' && touches.set(e.pointerId, { x: e.clientX, y: e.clientY }), capture);
    container.addEventListener('pointermove', (e) => {
      const t = touches.get(e.pointerId);
      if (!t || touches.size !== 1 || this.touchGrab()) return;
      const dx = e.clientX - t.x, dy = e.clientY - t.y;
      if (!t.mode && Math.hypot(dx, dy) >= 8) t.mode = Math.abs(dx) > Math.abs(dy) ? 'orbit' : 'scroll';
      if (t.mode !== 'orbit') e.stopPropagation();
    }, capture);
    for (const k of ['pointerup', 'pointercancel']) container.addEventListener(k, (e) => touches.delete(e.pointerId), capture);
    canvas.addEventListener('touchstart', (e) => (e.touches.length > 1 || this.touchGrab()) && e.preventDefault(), { passive: false });
    container.addEventListener('wheel', (e) => e.ctrlKey || e.metaKey || e.stopPropagation(), capture);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xffffff);
    this.root = new THREE.Group();
    this.overlayScene = new THREE.Scene();
    this.glowScene = new THREE.Scene();
    const key = new THREE.DirectionalLight(0xffffff, 1.8), fill = new THREE.DirectionalLight(0xffffff, 0.5);
    key.position.set(-0.6, 1, 0.8);
    fill.position.set(0.8, 0.3, -0.5);
    this.scene.add(this.root, new THREE.HemisphereLight(0xffffff, 0xc4c0ba, 1.9), key, fill);

    this.camera = new THREE.PerspectiveCamera(fov, 1, 1, 20000);
    this.controls = Object.assign(new OrbitControls(this.camera, canvas), { enableDamping: true, dampingFactor: 0.12, enablePan: false, touches: { ONE: THREE.TOUCH.ROTATE, TWO: THREE.TOUCH.DOLLY_ROTATE } });
    this.controls.addEventListener('change', () => this.request());

    const target = (o) => new THREE.WebGLRenderTarget(1, 1, o), half = { type: THREE.HalfFloatType, colorSpace: THREE.LinearSRGBColorSpace };
    this.rtColor = target({ ...half, samples: 4 });
    this.rtId = target({ minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter, depthTexture: new THREE.DepthTexture(1, 1, THREE.FloatType) });
    this.rtGlow = [target({ ...half, samples: 4 }), target({ ...half, depthBuffer: false }), target({ ...half, depthBuffer: false })];
    this.ink = pass(INK_FS, { tColor: { value: this.rtColor.texture }, tId: { value: this.rtId.texture }, tDepth: { value: this.rtId.depthTexture }, tGlow: { value: null }, glowGain: { value: 0 },
      texel: { value: new THREE.Vector2() }, radius: { value: 1 }, cNear: { value: 1 }, cFar: { value: 1000 }, inkDepth: { value: +inkDepth } });
    this.blur = pass(BLUR_FS, { tSrc: { value: null }, dir: { value: new THREE.Vector2() }, sigma: { value: 1 } });
    this.quadCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

    new ResizeObserver(() => this.resize()).observe(container);
    new IntersectionObserver(([e]) => { this.visible = e.isIntersecting; this.request(); }).observe(container);
    this.resize();
    const loop = () => {
      requestAnimationFrame(loop);
      if (!this.visible) return;
      const busy = this.onFrame.reduce((b, f) => f() || b, this.controls.update());
      if (busy || this.needs) { this.needs = false; this.render(); }
    };
    loop();
  }
  request() { this.needs = true; }
  resize() {
    const w = Math.max(1, this.container.clientWidth), h = Math.max(1, this.container.clientHeight), pr = this.renderer.getPixelRatio();
    this.renderer.setSize(w, h, false);
    for (const t of [this.rtColor, this.rtId]) t.setSize(w * pr, h * pr);
    for (const t of this.rtGlow) t.setSize(Math.max(1, Math.round(w * pr / 2)), Math.max(1, Math.round(h * pr / 2)));
    this.ink.u.texel.value.set(1 / (w * pr), 1 / (h * pr));
    this.ink.u.radius.value = this.inkWidth * pr * 0.5 + 0.25;
    const changed = Math.abs(w / h - this.camera.aspect) > 1e-3;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    if (changed && this.home) this.place(this.home.box, this.camera.position.clone().sub(this.controls.target), this.home.pad);
    this.request();
  }
  render() {
    const r = this.renderer, restore = [], ids = new Map();
    r.setRenderTarget(this.rtColor);
    r.render(this.scene, this.camera);
    this.root.traverse((o) => o.isGroup && o.userData.name !== undefined && ids.set(o, ids.size + 1));
    this.scene.traverse((o) => {
      if (o.isLine || o.isPoints || o.userData.noInk) { if (o.visible) { o.visible = false; restore.push(() => (o.visible = true)); } return; }
      if (!o.isMesh) return;
      let p = o.parent;
      while (p && !ids.has(p)) p = p.parent;
      const shaded = o.material;
      restore.push(() => (o.material = shaded));
      o.material = o.userData.idMaterial || idMaterial(o.userData.occluder ? 0xc8c8c8 : ids.get(p) ?? 0);
    });
    const background = this.scene.background;
    this.scene.background = null;
    r.setClearColor(0x000000, 1);
    r.setRenderTarget(this.rtId);
    r.render(this.scene, this.camera);
    restore.forEach((undo) => undo());
    this.scene.background = background;
    this.glow();
    r.setClearColor(0xffffff, 1);
    Object.assign(this.ink.u.cNear, { value: this.camera.near });
    Object.assign(this.ink.u.cFar, { value: this.camera.far });
    r.setRenderTarget(null);
    r.render(this.ink.scene, this.quadCam);
    r.autoClear = false;
    r.clearDepth();
    r.render(this.overlayScene, this.camera);
    r.autoClear = true;
  }
  glow() {
    const r = this.renderer, [src, a, b] = this.rtGlow, u = this.blur.u;
    this.ink.u.glowGain.value = 0;
    if (!this.glowScene.children.some((o) => o.visible && (o.isMesh || o.children.length))) return;
    r.setRenderTarget(src);
    r.render(this.glowScene, this.camera);
    u.sigma.value = Math.max(0.75, this.glowSigma * a.height / (2 * this.camera.position.distanceTo(this.controls.target) * Math.tan(this.camera.fov * Math.PI / 360)));
    for (const [from, to, dx, dy] of [[src, a, 1, 0], [a, b, 0, 1]]) {
      u.tSrc.value = from.texture;
      u.dir.value.set(dx / from.width, dy / from.height);
      r.setRenderTarget(to);
      r.render(this.blur.scene, this.quadCam);
    }
    this.ink.u.tGlow.value = b.texture;
    this.ink.u.glowGain.value = this.glowGain;
  }
  frame(box, dir = new THREE.Vector3(0.55, 0.5, 0.75), pad = 1.15) {
    this.home = { box: box.clone(), dir: dir.clone().normalize(), pad };
    this.resetView();
  }
  resetView() { if (this.home) this.place(this.home.box, this.home.dir.clone(), this.home.pad); }
  place(box, dir, pad) {
    const c = box.getCenter(new THREE.Vector3()), w = dir.clone().normalize(), up0 = Math.abs(w.y) > 0.99 ? new THREE.Vector3(0, 0, 1) : new THREE.Vector3(0, 1, 0);
    const right = new THREE.Vector3().crossVectors(up0, w).normalize(), up = new THREE.Vector3().crossVectors(w, right);
    const tv = Math.tan(this.camera.fov * Math.PI / 360), th = tv * this.camera.aspect;
    const fit = Math.max(...[0, 1, 2, 3, 4, 5, 6, 7].map((k) => {
      const q = new THREE.Vector3(k & 1 ? box.max.x : box.min.x, k & 2 ? box.max.y : box.min.y, k & 4 ? box.max.z : box.min.z).sub(c);
      return q.dot(w) + Math.max(Math.abs(q.dot(right)) / th, Math.abs(q.dot(up)) / tv);
    }));
    const d = Math.max(box.getBoundingSphere(new THREE.Sphere()).radius * pad / Math.sin(this.camera.fov * Math.PI / 360) / Math.min(1, this.camera.aspect), 1.02 * fit);
    this.controls.target.copy(c);
    this.camera.position.copy(c).addScaledVector(w, d);
    Object.assign(this.camera, { near: d / 50, far: d * 20 });
    this.camera.updateProjectionMatrix();
    this.controls.update();
    this.request();
  }
}

export const occluderIdMaterial = (map, cut) => new THREE.ShaderMaterial({
  side: THREE.DoubleSide, uniforms: { map: { value: map }, cut: { value: cut } },
  vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
  fragmentShader: 'uniform sampler2D map; uniform float cut; varying vec2 vUv; void main(){ if (texture2D(map, vUv).a < cut) discard; gl_FragColor = vec4(vec3(200.0 / 255.0), 1.0); }',
});

const WORLD_VS = 'varying vec3 vL, vN, vP; void main(){ vL = position; vec4 wp = modelMatrix * vec4(position, 1.0); vP = wp.xyz; vN = normalize(mat3(modelMatrix) * normal); gl_Position = projectionMatrix * viewMatrix * wp; }';

export const glassMaterial = ({ t0 = 0.8, edge = 0.55, ks = 0.5, shin = 90, kenv = 0.6, kdiff = 0.035, keyDir = [0.35, 1, -0.45], tint = [0.97, 0.98, 1] } = {}) => new THREE.ShaderMaterial({
  vertexShader: WORLD_VS,
  fragmentShader: `precision highp float;
    varying vec3 vN, vP;
    uniform float t0, edge, ks, shin, kenv, kdiff;
    uniform vec3 keyDir, tint;
    void main(){
      vec3 V = normalize(cameraPosition - vP), N = normalize(vN);
      if (dot(N, V) < 0.0) N = -N;
      float ndv = clamp(dot(N, V), 0.0, 1.0), F = 0.04 + 0.96 * pow(1.0 - ndv, 5.0);
      float T = t0 * (1.0 - edge * pow(1.0 - ndv, 2.0)) * (1.0 - F);
      vec3 L = normalize(keyDir), H = normalize(L + V), R = reflect(-V, N);
      float S = ks * pow(max(dot(N, H), 0.0), shin) + kenv * F * smoothstep(-0.2, 0.9, R.y) + kdiff * (0.35 + 0.65 * max(dot(N, L), 0.0));
      gl_FragColor = vec4(S * tint, T);
    }`,
  uniforms: { t0: { value: t0 }, edge: { value: edge }, ks: { value: ks }, shin: { value: shin }, kenv: { value: kenv }, kdiff: { value: kdiff }, keyDir: { value: new THREE.Vector3(...keyDir) }, tint: { value: new THREE.Vector3(...tint) } },
  transparent: true, depthWrite: false, side: THREE.DoubleSide,
  blending: THREE.CustomBlending, blendEquation: THREE.AddEquation, blendSrc: THREE.OneFactor, blendDst: THREE.SrcAlphaFactor, blendSrcAlpha: THREE.ZeroFactor, blendDstAlpha: THREE.OneFactor,
});

export const eyeMaterial = (black, white) => new THREE.ShaderMaterial({
  vertexShader: WORLD_VS,
  fragmentShader: `precision highp float;
    varying vec3 vL, vN, vP;
    uniform vec3 black, white;
    void main(){
      vec2 p = vL.xy;
      float l = 0.0;
      if (distance(p, vec2(6.2, 13.1)) < 7.6) l = 1.0;
      if (distance(p, vec2(6.2, 13.1)) < 6.7165) l = 2.0;
      if (distance(p, vec2(6.2, 15.0)) < 4.997) l = 1.0;
      if (distance(p, vec2(6.2, 12.9005)) < 1.3395) l = 2.0;
      float facing = dot(normalize(vN), normalize(cameraPosition - vP));
      if (abs(vL.z) < 18.5 || l < 0.5 || facing < 0.0) discard;
      gl_FragColor = vec4((l < 1.5 ? black : white) * (0.8 + 0.2 * facing), 1.0);
    }`,
  uniforms: { black: { value: new THREE.Color(black) }, white: { value: new THREE.Color(white) } },
  transparent: true, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2,
});
