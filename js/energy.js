const RHO = 1e-8, TAU = 1e-8, INNER = 12, RADIUS = 16, TOL = 0.0025;

function mm3(A, B) {
  const C = new Array(9);
  for (let i = 0; i < 9; i += 3) for (let j = 0; j < 3; j++) C[i + j] = A[i] * B[j] + A[i + 1] * B[3 + j] + A[i + 2] * B[6 + j];
  return C;
}
const mT3 = (A) => [A[0], A[3], A[6], A[1], A[4], A[7], A[2], A[5], A[8]];
const mv3 = (A, v) => [A[0] * v[0] + A[1] * v[1] + A[2] * v[2], A[3] * v[0] + A[4] * v[1] + A[5] * v[2], A[6] * v[0] + A[7] * v[1] + A[8] * v[2]];
const add3 = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub3 = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross3 = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const trace = (A) => A[0] + A[4] + A[8];
const rot3 = (P) => [P[0], P[1], P[2], P[4], P[5], P[6], P[8], P[9], P[10]];
const tr3 = (P) => [P[3], P[7], P[11]];
const rigid = (R, t) => Float64Array.from([R[0], R[1], R[2], t[0], R[3], R[4], R[5], t[1], R[6], R[7], R[8], t[2], 0, 0, 0, 1]);
const point = (P, x) => add3(mv3(rot3(P), x), tr3(P));

export function mul4(A, B) {
  const C = new Float64Array(16);
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) for (let k = 0; k < 4; k++) C[4 * i + j] += A[4 * i + k] * B[4 * k + j];
  return C;
}
export const inv4rigid = (A) => { const R = mT3(rot3(A)); return rigid(R, mv3(R, tr3(A)).map((x) => -x)); };
export const comWorld = (P, mom) => point(P, mom.slice(1, 4));

function axisAngle(k, th) {
  const c = Math.cos(th), s = Math.sin(th), C = 1 - c;
  return [c + k[0] * k[0] * C, k[0] * k[1] * C - k[2] * s, k[0] * k[2] * C + k[1] * s,
    k[1] * k[0] * C + k[2] * s, c + k[1] * k[1] * C, k[1] * k[2] * C - k[0] * s,
    k[2] * k[0] * C - k[1] * s, k[2] * k[1] * C + k[0] * s, c + k[2] * k[2] * C];
}
const quatR = ([w, x, y, z]) => [w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y),
  2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x),
  2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z];

export function twist(P, x, w, v) {
  const th = Math.hypot(...w), R = th > 1e-15 ? axisAngle(w.map((c) => c / th), th) : [1, 0, 0, 0, 1, 0, 0, 0, 1];
  return mul4(rigid(R, add3(sub3(x, mv3(R, x)), v)), P);
}

export function lerpPose(A, B, f, com = [0, 0, 0]) {
  const RA = rot3(A), D = mm3(mT3(RA), rot3(B)), th = Math.acos(Math.min(1, Math.max(-1, (trace(D) - 1) / 2)));
  const k = [D[7] - D[5], D[2] - D[6], D[3] - D[1]].map((x) => x / (2 * Math.sin(th)));
  const R = th > 1e-12 ? mm3(RA, axisAngle(k, f * th)) : RA, ca = point(A, com), cb = point(B, com);
  return rigid(R, sub3(ca.map((x, i) => x + f * (cb[i] - x)), mv3(R, com)));
}

function topEigen4(N) {
  const a = N.slice(), v = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  const turn = (m, i, j, c, s) => { const x = m[i], y = m[j]; m[i] = c * x - s * y; m[j] = s * x + c * y; };
  for (let sweep = 0; sweep < 30; sweep++) {
    let off = 0;
    for (let p = 0; p < 4; p++) for (let r = p + 1; r < 4; r++) off += a[4 * p + r] ** 2;
    if (off < 1e-30) break;
    for (let p = 0; p < 4; p++) for (let r = p + 1; r < 4; r++) {
      const apr = a[4 * p + r];
      if (Math.abs(apr) < 1e-300) continue;
      const theta = (a[4 * r + r] - a[4 * p + p]) / (2 * apr), t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < 4; k++) turn(a, 4 * k + p, 4 * k + r, c, s);
      for (let k = 0; k < 4; k++) turn(a, 4 * p + k, 4 * r + k, c, s);
      for (let k = 0; k < 4; k++) turn(v, 4 * k + p, 4 * k + r, c, s);
    }
  }
  let best = 0;
  for (let i = 1; i < 4; i++) if (a[5 * i] > a[5 * best]) best = i;
  return [v[best], v[4 + best], v[8 + best], v[12 + best]];
}

function bestRotation([Sxx, Sxy, Sxz, Syx, Syy, Syz, Szx, Szy, Szz]) {
  const n00 = Sxx + Syy + Szz, n01 = Syz - Szy, n02 = Szx - Sxz, n03 = Sxy - Syx;
  const n11 = Sxx - Syy - Szz, n12 = Sxy + Syx, n13 = Szx + Sxz, n22 = -Sxx + Syy - Szz, n23 = Syz + Szy, n33 = -Sxx - Syy + Szz;
  const a01 = n01 * n01, a02 = n02 * n02, a03 = n03 * n03, a12 = n12 * n12, a13 = n13 * n13, a23 = n23 * n23;
  const c2 = n00 * n11 + n00 * n22 + n00 * n33 + n11 * n22 + n11 * n33 + n22 * n33 - (a01 + a02 + a03 + a12 + a13 + a23);
  const c1 = -(n00 * n11 * n22 + n00 * n11 * n33 + n00 * n22 * n33 + n11 * n22 * n33) + n00 * (a12 + a13 + a23) + n11 * (a02 + a03 + a23) + n22 * (a01 + a03 + a13) + n33 * (a01 + a02 + a12) - 2 * (n01 * n12 * n02 + n01 * n13 * n03 + n02 * n23 * n03 + n12 * n23 * n13);
  const m0 = n22 * n33 - a23, m1 = n12 * n33 - n13 * n23, m2 = n12 * n23 - n13 * n22, m3 = n02 * n33 - n03 * n23, m4 = n02 * n23 - n03 * n22, m5 = n02 * n13 - n03 * n12;
  const c0 = n00 * (n11 * m0 - n12 * m1 + n13 * m2) - n01 * (n01 * m0 - n12 * m3 + n13 * m4) + n02 * (n01 * m1 - n11 * m3 + n13 * m5) - n03 * (n01 * m2 - n11 * m4 + n12 * m5);
  let l = Math.max(n00 + Math.abs(n01) + Math.abs(n02) + Math.abs(n03), n11 + Math.abs(n01) + Math.abs(n12) + Math.abs(n13), n22 + Math.abs(n02) + Math.abs(n12) + Math.abs(n23), n33 + Math.abs(n03) + Math.abs(n13) + Math.abs(n23));
  for (let it = 0; it < 60; it++) {
    const l2 = l * l, p = l2 * l2 + c2 * l2 + c1 * l + c0, dp = 4 * l2 * l + 2 * c2 * l + c1;
    if (!(dp > 0)) break;
    l -= p / dp;
    if (Math.abs(p / dp) <= 1e-13 * Math.max(1, Math.abs(l))) break;
  }
  const A00 = n00 - l, A11 = n11 - l, A22 = n22 - l, A33 = n33 - l, A01 = n01, A02 = n02, A03 = n03, A12 = n12, A13 = n13, A23 = n23;
  const s01 = A02 * A13 - A12 * A03, s02 = A02 * A23 - A22 * A03, s03 = A02 * A33 - A23 * A03, s12 = A12 * A23 - A22 * A13, s13 = A12 * A33 - A23 * A13, s23 = A22 * A33 - A23 * A23;
  const t01 = A00 * A11 - A01 * A01, t02 = A00 * A12 - A02 * A01, t03 = A00 * A13 - A03 * A01, t12 = A01 * A12 - A02 * A11, t13 = A01 * A13 - A03 * A11, t23 = A02 * A13 - A03 * A12;
  const rows = [
    [A11 * s23 - A12 * s13 + A13 * s12, -(A01 * s23 - A12 * s03 + A13 * s02), A01 * s13 - A11 * s03 + A13 * s01, -(A01 * s12 - A11 * s02 + A12 * s01)],
    [-(A01 * s23 - A02 * s13 + A03 * s12), A00 * s23 - A02 * s03 + A03 * s02, -(A00 * s13 - A01 * s03 + A03 * s01), A00 * s12 - A01 * s02 + A02 * s01],
    [A13 * t23 - A23 * t13 + A33 * t12, -(A03 * t23 - A23 * t03 + A33 * t02), A03 * t13 - A13 * t03 + A33 * t01, -(A03 * t12 - A13 * t02 + A23 * t01)],
    [-(A12 * t23 - A22 * t13 + A23 * t12), A02 * t23 - A22 * t03 + A23 * t02, -(A02 * t13 - A12 * t03 + A23 * t01), A02 * t12 - A12 * t02 + A22 * t01],
  ];
  const norm2 = rows.map((v) => v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]), b = norm2.indexOf(Math.max(...norm2));
  if (!(norm2[b] > 1e-300)) return quatR(topEigen4([n00, n01, n02, n03, n01, n11, n12, n13, n02, n12, n22, n23, n03, n13, n23, n33]));
  return quatR(rows[b].map((x) => x / Math.sqrt(norm2[b])));
}

function seatError(ma, mb, source, target, grad = false) {
  const mA = ma[0], mB = mb[0], M = mA + mB, mu = mA * mB / M, sa = ma.slice(4, 13), sb = mb.slice(4, 13), ca = ma.slice(1, 4), cb = mb.slice(1, 4);
  const R0 = rot3(source), R1 = rot3(target), p = point(source, cb), h = sub3(ca, p), g = sub3(ca, point(target, cb));
  const K = mm3(mm3(R1, sb), mT3(R0));
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) K[3 * i + j] = mA * sa[3 * i + j] + mB * K[3 * i + j] + mu * g[i] * h[j];
  const E = bestRotation(K);
  const da = E.map((x, e) => x - (e % 4 === 0)), db = mm3(E, R1).map((x, e) => x - R0[e]), r = sub3(mv3(E, g), h);
  const spread = (D, S) => trace(mm3(mm3(D, S), mT3(D))), d2 = Math.max(0, (mA * spread(da, sa) + mB * spread(db, sb) + mu * dot3(r, r)) / M);
  if (!grad) return d2;
  const A = mm3(mm3(R0, sb), mT3(db)), va = [A[5] - A[7], A[6] - A[2], A[1] - A[3]], pxr = cross3(p, r);
  return [d2, ...va.map((x, i) => (-2 * mB * x + 2 * mu * pxr[i]) / M), ...r.map((x) => 2 * mu * x / M)];
}

const taper = (gap) => { const s = (RADIUS - gap) / (RADIUS - INNER); return gap <= INNER ? 1 : gap >= RADIUS ? 0 : s * s * s * (6 * s * s - 15 * s + 10); };
const taperDeriv = (gap) => { const s = (RADIUS - gap) / (RADIUS - INNER); return gap <= INNER || gap >= RADIUS ? 0 : -30 * s * s * (s - 1) * (s - 1) / (RADIUS - INNER); };
const H = (x) => (x > 0 ? x * x * x / (x * x + 1) : 0);
const Hp = (x) => (x > 0 ? (x * x * x * x + 3 * x * x) / ((x * x + 1) * (x * x + 1)) : 0);
const qPenalty = (c) => TAU * H((c - TAU) / TAU);

export function prepareAtlas(raw) {
  const parts = {}, regions = {};
  for (const [name, p] of Object.entries(raw.parts)) {
    const frames = Object.fromEntries(Object.entries(p.frames).map(([c, m]) => [c, Float64Array.from(m.flat())]));
    parts[name] = { moments: p.moments, frames, finv: Object.fromEntries(Object.entries(frames).map(([c, F]) => [c, inv4rigid(F)])), order: p.id };
  }
  for (const [key, list] of Object.entries(raw.regions)) regions[key] = list.map(([ca, cb, , reversed, cells]) => ({ ca: String(ca), cb: String(cb), reversed, cells: mergeSeam(cells.map(([lo, hi]) => [lo[0], hi[0]])) }));
  return { parts, regions, alias: raw.alias || {} };
}

function mergeSeam(cells) {
  const low = cells.findIndex((c) => c[0] <= 0 && c[1] < 360), high = cells.findIndex((c) => c[1] >= 360 && c[0] > 0);
  if (low < 0 || high < 0 || low === high) return cells;
  return [...cells.filter((_, i) => i !== low && i !== high), [cells[high][0], 360 + cells[low][1]]];
}

const bases = new WeakMap();
function placementBasis(Fa, FbInv) {
  if (!bases.has(Fa)) bases.set(Fa, new WeakMap());
  const byB = bases.get(Fa);
  if (!byB.has(FbInv)) byB.set(FbInv, [[5, 15], [0, 10], [2, -8]].map(([plus, other]) => {
    const X = new Float64Array(16);
    X[plus] = 1; X[Math.abs(other)] = Math.sign(other);
    return mul4(mul4(Fa, X), FbInv);
  }));
  return byB.get(FbInv);
}
function placement([MY, MC, MS], radians, out = new Float64Array(16)) {
  const c = Math.cos(radians), s = Math.sin(radians);
  for (let i = 0; i < 16; i++) out[i] = MY[i] + c * MC[i] + s * MS[i];
  return out;
}

function seatDistance(ma, mb, source, basis, reversed, cells, q0 = null) {
  const k = (reversed ? -1 : 1) * Math.PI / 180, tgt = new Float64Array(16), f = (q) => seatError(ma, mb, source, placement(basis, k * q, tgt));
  const cand = [];
  if (q0 !== null) {
    const cell = cells.find(([lo, hi]) => q0 >= lo - 1e-9 && q0 <= hi + 1e-9);
    if (cell) { const x = Math.min(cell[1], Math.max(cell[0], q0)); cand.push([f(x), x, ...cell, Math.min(12, cell[1] - cell[0])]); }
  }
  if (!cand.length) for (const [lo, hi] of cells) {
    const n = Math.max(2, Math.ceil((hi - lo) / 12) + 1), st = (hi - lo) / (n - 1);
    for (let i = 0; i < n; i++) cand.push([f(lo + st * i), lo + st * i, lo, hi, st]);
  }
  cand.sort((p, q) => p[0] - q[0]);
  let best = { d2: cand[0][0], q: cand[0][1] };
  for (const [v0, x0, lo, hi, st] of cand.slice(0, 2)) {
    let x = x0, fx = v0, a = Math.max(lo, x0 - st), b = Math.min(hi, x0 + st);
    for (let it = 0; it < 12; it++) {
      const e = 1e-3, xp = Math.min(b, x + e), xm = Math.max(a, x - e), fp = f(xp), fm = f(xm), hp = xp - x, hm = x - xm;
      if (hp + hm <= 0) break;
      const g = (fp - fm) / (hp + hm), curv = hp > 0 && hm > 0 ? (fp * hm + fm * hp - fx * (hp + hm)) / (0.5 * hp * hm * (hp + hm)) : 0;
      let step = curv > 1e-12 ? -g / curv : -Math.sign(g) * st * 0.25;
      const clamp = () => Math.min(b, Math.max(a, x + step));
      let xn = clamp(), fn = f(xn);
      while (fn > fx && Math.abs(xn - x) > 1e-9) { step *= 0.5; xn = clamp(); fn = f(xn); }
      if (fn > fx) break;
      const moved = Math.abs(xn - x);
      if (xn > x) a = x; else b = x;
      x = xn; fx = fn;
      if (moved < 1e-7) break;
    }
    if (fx < best.d2) best = { d2: fx, q: x };
  }
  return best;
}

export function seats(atlas, parts, warm = null) {
  const out = [], canon = (name) => atlas.alias[name] || name;
  for (let i = 0; i < parts.length; i++) for (let j = i + 1; j < parts.length; j++) {
    const swap = atlas.parts[canon(parts[i].name)].order > atlas.parts[canon(parts[j].name)].order, a = swap ? j : i, b = swap ? i : j;
    const na = canon(parts[a].name), nb = canon(parts[b].name), A = atlas.parts[na], B = atlas.parts[nb], Pa = parts[a].pose, Pb = parts[b].pose;
    const source = mul4(inv4rigid(Pa), Pb);
    for (const r of atlas.regions[`${na}|${nb}`] || []) {
      const Fa = A.frames[r.ca], pa = point(Pa, tr3(Fa)), pb = point(Pb, tr3(B.frames[r.cb])), gap = Math.hypot(...sub3(pa, pb));
      if (gap >= RADIUS) continue;
      const key = `${a}|${b}|${r.ca}|${r.cb}`, basis = placementBasis(Fa, B.finv[r.cb]);
      const { d2, q } = seatDistance(A.moments, B.moments, source, basis, r.reversed, r.cells, warm && warm.has(key) ? warm.get(key) : null);
      const w = taper(gap);
      out.push({ i: a, j: b, key, c: d2, w, g: w / (RHO + qPenalty(d2)), pa, pb, q, ma: A.moments, mb: B.moments, target: placement(basis, (r.reversed ? -q : q) * Math.PI / 180) });
    }
  }
  return out;
}

function network(S, n) {
  const P4 = new Float64Array(n * n), G = new Float64Array(n * n), m = n - 1, L = new Float64Array(m * m), Inv = new Float64Array(m * m);
  for (const s of S) { P4[n * s.i + s.j] += s.g ** 4; P4[n * s.j + s.i] += s.g ** 4; }
  for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) if (a !== b) G[n * a + b] = Math.pow(P4[n * a + b], 0.25) + 1 / (16 * n);
  for (let i = 0; i < m; i++) {
    for (let k = 0; k < n; k++) if (k !== i) L[m * i + i] += G[n * i + k];
    for (let j = 0; j < m; j++) if (j !== i) L[m * i + j] = -G[n * i + j];
    Inv[m * i + i] = 1;
  }
  for (let c = 0; c < m; c++) {
    let p = c;
    for (let r = c + 1; r < m; r++) if (Math.abs(L[m * r + c]) > Math.abs(L[m * p + c])) p = r;
    if (p !== c) for (const X of [L, Inv]) for (let k = 0; k < m; k++) [X[m * c + k], X[m * p + k]] = [X[m * p + k], X[m * c + k]];
    const piv = L[m * c + c];
    for (const X of [L, Inv]) for (let k = 0; k < m; k++) X[m * c + k] /= piv;
    for (let r = 0; r < m; r++) {
      const f = L[m * r + c];
      if (r === c || !f) continue;
      for (const X of [L, Inv]) for (let k = 0; k < m; k++) X[m * r + k] -= f * X[m * c + k];
    }
  }
  return { P4, G, Z: (i, j) => (i === m || j === m ? 0 : Inv[m * i + j]) };
}

export function energy(atlas, parts, S = seats(atlas, parts), grad = false) {
  const n = parts.length, moms = parts.map((p) => atlas.parts[atlas.alias[p.name] || p.name].moments), grads = parts.map(() => [0, 0, 0, 0, 0, 0]);
  if (n < 2) return { E: 0, seats: S, G: new Float64Array(n * n), grads, moms };
  const { P4, G, Z } = network(S, n), T = (n - 1) * RHO, dEdG = new Float64Array(n * n), D = new Float64Array(n);
  let E = 0;
  for (let a = 0; a < n; a++) for (let b = a + 1; b < n; b++) {
    const x = Z(a, a) + Z(b, b) - 2 * Z(a, b) - T;
    E += T * H(x / T) / (n - 1);
    if (!grad || x <= 0) continue;
    for (let i = 0; i < n; i++) D[i] = Z(i, a) - Z(i, b);
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) dEdG[n * i + j] -= Hp(x / T) / (n - 1) * (D[i] - D[j]) ** 2;
  }
  if (!grad) return { E, seats: S, G };
  const com = parts.map((p, k) => comWorld(p.pose, moms[k]));
  for (const s of S) {
    const lo = Math.min(s.i, s.j), hi = Math.max(s.i, s.j), p4 = P4[n * lo + hi], dEdg = p4 > 0 ? dEdG[n * lo + hi] * s.g ** 3 / Math.pow(p4, 0.75) : 0;
    if (s.w <= 0 || !dEdg) continue;
    const den = RHO + qPenalty(s.c), gap = Math.hypot(...sub3(s.pb, s.pa)), u = gap > 1e-12 ? sub3(s.pb, s.pa).map((x) => x / gap) : [0, 0, 0];
    const dgdc = -s.w * Hp((s.c - TAU) / TAU) / (den * den), dgdgap = taperDeriv(gap) / den;
    const Pa = parts[s.i].pose, Ra = rot3(Pa), ta = tr3(Pa);
    const dc = dgdc ? seatError(s.ma, s.mb, mul4(inv4rigid(Pa), parts[s.j].pose), s.target, true) : [0, 0, 0, 0, 0, 0, 0];
    const gw = mv3(Ra, dc.slice(1, 4)), gv = mv3(Ra, dc.slice(4, 7));
    for (const [k, sign, x] of [[s.j, 1, s.pb], [s.i, -1, s.pa]]) {
      const cw = add3(gw, cross3(sub3(ta, com[k]), gv)), rxu = cross3(sub3(x, com[k]), u);
      for (let i = 0; i < 3; i++) {
        grads[k][i] += dEdg * sign * (dgdc * cw[i] + dgdgap * rxu[i]);
        grads[k][3 + i] += dEdg * sign * (dgdc * gv[i] + dgdgap * u[i]);
      }
    }
  }
  return { E, seats: S, G, grads, moms };
}

function inertia(P, mom) {
  const R = rot3(P), S = mm3(mm3(R, mom.slice(4, 13)), mT3(R)), t = trace(S);
  return S.map((x, e) => (e % 4 === 0 ? t - x : -x));
}
function solve3(J, b) {
  const col = (k) => [J[k], J[3 + k], J[6 + k]], det = (x, y, z) => dot3(x, cross3(y, z)), d = det(col(0), col(1), col(2));
  return [det(b, col(1), col(2)) / d, det(col(0), b, col(2)) / d, det(col(0), col(1), b) / d];
}
const quadJ = (J, w) => dot3(w, mv3(J, w));
const combine = (terms, n) => Array.from({ length: n }, (_, k) => Object.fromEntries(['w', 'v'].map((key) => [key, [0, 1, 2].map((i) => terms.reduce((s, [c, K]) => s + c * K[k][key][i], 0))])));
const advance = (parts, moms, K) => parts.map((p, k) => ({ ...p, pose: twist(p.pose, comWorld(p.pose, moms[k]), K[k].w, K[k].v) }));

export function* guideFlow(atlas, parts) {
  const n = parts.length;
  let y = parts, J, moms, Mt;
  const size2 = (u, k) => dot3(u.v, u.v) + quadJ(J[k], u.w);
  const field = (at, warm) => {
    const S = seats(atlas, at, warm), r = energy(atlas, at, S, true);
    moms ??= r.moms; Mt ??= moms.reduce((t, m) => t + m[0], 0); J ??= at.map((p, k) => inertia(p.pose, moms[k]));
    const vel = r.grads.map((g, k) => ({ w: solve3(inertia(at[k].pose, moms[k]), g.slice(0, 3).map((x) => -x / moms[k][0])), v: g.slice(3).map((x) => -x / moms[k][0]) }));
    const sp = Math.sqrt(vel.reduce((t, u, k) => t + moms[k][0] * size2(u, k), 0) / Mt);
    return { E: r.E, S, sp, vel: sp > 0 ? combine([[1 / sp, vel]], n) : vel };
  };
  let F = field(y), h = 0.5, tau = 0, step = 0;
  const hist = [F.E];
  yield { parts: y, E: F.E, S: F.S, tau, step };
  while (F.E >= 1e-10 && step < 6000 && tau < 400 && F.sp >= 1e-14) {
    const hh = Math.min(h, Math.max(1e-6, F.E / Math.max(Mt * F.sp, 1e-300)), 2), k1 = F.vel, warm = new Map(F.S.map((t) => [t.key, t.q]));
    const k2 = field(advance(y, moms, combine([[hh / 2, k1]], n)), warm).vel, k3 = field(advance(y, moms, combine([[0.75 * hh, k2]], n)), warm).vel;
    const y1 = advance(y, moms, combine([[2 * hh / 9, k1], [hh / 3, k2], [4 * hh / 9, k3]], n)), F4 = field(y1);
    const err = Math.max(...combine([[hh * (2 / 9 - 7 / 24), k1], [hh * (1 / 3 - 1 / 4), k2], [hh * (4 / 9 - 1 / 3), k3], [-hh / 8, F4.vel]], n).map((u, k) => Math.sqrt(size2(u, k))));
    const scale = 0.9 * Math.cbrt(TOL / Math.max(err, 1e-300));
    if (!(err <= TOL && F4.E <= F.E * (1 + 1e-9)) && hh > 1e-6) { h = hh * Math.max(0.1, Math.min(0.5, scale)); continue; }
    y = y1; F = F4; tau += hh; step++;
    J = y.map((p, k) => inertia(p.pose, moms[k]));
    hist.push(F.E);
    const stalled = hist.length > 20 && hist[hist.length - 21] - F.E < 1e-9 * F.E;
    yield { parts: y, E: F.E, S: F.S, tau, step, stalled };
    if (stalled) break;
    h = hh * Math.min(4, Math.max(0.2, scale));
  }
}
