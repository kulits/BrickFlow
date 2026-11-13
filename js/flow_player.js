import * as THREE from 'three';
import { Viewer, makePartFrom } from './viewer.js';
import { loadPack, id } from './pack.js';

const SNAP_T = 1.06, DURATION = 6, HOLD = 2.2, START = 4, UP = new THREE.Vector3(0, 1, 0);
const flip = ([x, y, z]) => new THREE.Vector3(x, -y, -z);

export async function flowPlayer(el, { index }) {
  const $ = (q) => el.querySelector(q), v = new Viewer($('.stage'), { inkWidth: 1.05 });
  const slider = $('[data-t]'), playButton = $('[data-act="play"]'), final = $('[data-act="x1"]'), picker = $('[data-picker]');
  let cur = null, playing = true, t = 0, last = 0, hold = 0, loads = 0;

  function apply() {
    const frames = final.checked ? cur.X1 : cur.X, tt = final.checked ? Math.min(t, cur.t[frames.length - 1]) : t;
    let i = 0;
    while (i < cur.t.length - 2 && cur.t[i + 1] <= tt) i++;
    const j = Math.min(i + 1, frames.length - 1), a = frames[Math.min(i, frames.length - 1)], b = frames[j];
    const f = cur.t[j] > cur.t[i] ? Math.min(1, Math.max(0, (tt - cur.t[i]) / (cur.t[j] - cur.t[i]))) : 0;
    cur.groups.forEach((g, k) => { g.position.lerpVectors(a[k].p, b[k].p, f); g.quaternion.slerpQuaternions(a[k].q, b[k].q, f); });
    slider.value = Math.round(t / cur.end * 1000);
    $('[data-tlabel]').textContent = t <= 1 ? t.toFixed(2) : t >= SNAP_T ? '1 (snapped)' : '1 (snapping)';
    v.request();
  }
  const setPlaying = (on) => { playing = on; playButton.textContent = on ? 'Pause' : 'Play'; };

  async function load(k) {
    const turn = ++loads, pack = await loadPack(k), { parts, view, figure_rot = 0 } = pack.header;
    if (turn !== loads) return;
    v.root.clear();
    v.root.quaternion.identity();
    const groups = parts.map(([part, code]) => Object.assign(makePartFrom(pack.meshes[part], code, part), { matrixAutoUpdate: true }));
    v.root.add(...groups);
    cur = { groups, X: pack.S ? [...pack.X, pack.S] : pack.X, X1: pack.X1, t: pack.S ? [...pack.header.t, SNAP_T] : pack.header.t, end: pack.S ? SNAP_T : 1 };
    const dir = flip(view.dir).normalize(), up = flip(view.up).addScaledVector(dir, -flip(view.up).dot(dir)).normalize();
    const right = new THREE.Vector3().crossVectors(up, dir).normalize(), angle = figure_rot * Math.PI / 180, imageUp = up.multiplyScalar(Math.cos(angle)).addScaledVector(right, Math.sin(angle));
    v.root.quaternion.setFromUnitVectors(imageUp, UP);
    t = cur.end;
    apply();
    v.root.updateMatrixWorld(true);
    v.frame(new THREE.Box3().setFromObject(v.root), dir.applyQuaternion(v.root.quaternion), 0.92);
    $('[data-photo]').src = `assets/thumb/${id(k)}.avif`;
    $('[data-cap]').textContent = `${index[k].photo ? 'Photograph' : 'Rendered human-designed model'} · ${index[k].n} parts`;
    picker.querySelectorAll('.thumb').forEach((b, i) => b.classList.toggle('sel', i === k));
    t = hold = 0;
    setPlaying(true);
  }

  v.onFrame.push(() => {
    const now = performance.now() / 1000, dt = Math.min(0.1, now - (last || now));
    last = now;
    if (!cur || !playing) return false;
    if (t < cur.end) t = Math.min(cur.end, t + dt / (t < 1 ? DURATION : 1.2 * DURATION));
    else if ((hold += dt) > HOLD) t = hold = 0;
    apply();
    return true;
  });
  slider.addEventListener('input', () => { setPlaying(false); t = slider.value / 1000 * cur.end; apply(); });
  playButton.addEventListener('click', () => { setPlaying(!playing); if (playing && t >= cur.end) t = hold = 0; });
  final.addEventListener('change', apply);

  index.forEach((_, k) => {
    const b = picker.appendChild(Object.assign(document.createElement('button'), { className: 'thumb', innerHTML: `<img src="assets/thumb/${id(k)}.avif" alt="" loading="lazy"/>` }));
    b.addEventListener('click', () => load(k));
  });
  const fade = () => picker.classList.toggle('more', picker.scrollLeft + picker.clientWidth < picker.scrollWidth - 4);
  picker.addEventListener('scroll', fade, { passive: true });
  new ResizeObserver(fade).observe(picker);
  await load(Math.min(START, index.length - 1));
  (async () => { for (const k of index.keys()) await loadPack(k).catch(() => {}); })();
}
