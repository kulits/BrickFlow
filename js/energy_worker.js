import * as en from './energy.js';

let atlas;

function pull(S) {
  const near = S.reduce((b, s) => (!b || s.c < b.c ? s : b), null);
  if (!near) return null;
  const same = S.filter((s) => s.i === near.i && s.j === near.j && s.w > 0), top = Math.max(...same.map((s) => s.g));
  const wts = same.map((s) => (s.g / top) ** 4), W = wts.reduce((x, y) => x + y, 0);
  return Array.from({ length: 16 }, (_, e) => same.reduce((t, s, k) => t + wts[k] / W * s.target[e], 0));
}

const report = (n, S, E) => ({
  E,
  isolated: n - new Set(S.filter((s) => s.w > 0).flatMap((s) => [s.i, s.j])).size,
  seats: S.map(({ i, j, c, g, w, pa, pb }) => ({ i, j, c, g, w, pa, pb })),
  target: pull(S),
});

self.onmessage = async ({ data: m }) => {
  if (m.atlas) {
    atlas = en.prepareAtlas(m.atlas);
    return self.postMessage({ ready: true });
  }
  const parts = m.parts.map((p) => ({ name: p.name, pose: Float64Array.from(p.pose) })), n = parts.length;
  if (m.flow === undefined) {
    const S = en.seats(atlas, parts);
    return self.postMessage(report(n, S, en.energy(atlas, parts, S).E));
  }
  const states = [];
  for (const st of en.guideFlow(atlas, parts)) {
    const undo = m.frame === undefined ? null : en.mul4(parts[m.frame].pose, en.inv4rigid(st.parts[m.frame].pose));
    states.push({ ...report(n, st.S, st.E), poses: st.parts.map((p) => Array.from(undo ? en.mul4(undo, p.pose) : p.pose)), tau: st.tau, stalled: !!st.stalled });
  }
  self.postMessage({ flow: m.flow, states });
};
