//! Connector-seat snap.

use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;
use std::io;

use procbrick::parts::{self, near, Pairing, AXLE_LENGTH, AXLE_SWEEP, REACH};
use procbrick::store::bad;
use procbrick::trees;

use crate::atlas::{components, Atlas, Cell, Connector, Moments, Region, AXLE, BALL, FIXED};
use crate::geometry::{actual, fit, into_cell, joint, origin, placement, pose, rot, rotation, vee, M3, M4, V3};
use crate::seat;

const CAP_T: f64 = 2.0; // per-part move cap: rms displacement of its mass, LDU
const CAP_R: f64 = 10.0 * PI / 180.0; // and rotation, radians (10 deg)
const TAUS: [f64; 2] = [1.001, 1.1]; // collision allowances, ascending, LDU
const RADIUS: f64 = 12.0; // connector-origin gap of a seat hypothesis, LDU
const AT_RAW: f64 = 1e-6; // D^2 below which a pair's raw pose is its seat, LDU^2
const CONS_T: f64 = 1e-3; // loop consistency inside a cluster: LDU
const CONS_R: f64 = 1e-4; // and radians
const COINCIDENT: f64 = 0.05; // LDU
const MAX_REPAIRS: usize = 64;

/// Penetration depths (LDU) of part pairs (i, j) at poses (P_i, P_j).
pub type Depths<'a> = dyn FnMut(&[(usize, usize, M4, M4)]) -> Vec<f64> + 'a;

/// World AABB of part `id`'s collider at pose P: its box's center moved, +- |R| times the half extents.
fn aabb(atlas: &Atlas, id: u32, p: &M4) -> Option<[V3; 2]> {
    let [lo, hi] = atlas.bounds(id)?;
    let r = rotation(p);
    let center = r * (lo + hi) * 0.5 + origin(p);
    let half = r.abs() * (hi - lo) * 0.5;
    Some([center - half, center + half])
}

/// Connector of part `part` placed in the world.
struct WorldConnector<'a> {
    part: usize,
    conn: &'a Connector,
    at: parts::Placed,
}

/// Whether two collider boxes [lo, hi] (None: no collider) overlap.
fn overlap(a: Option<[V3; 2]>, b: Option<[V3; 2]>) -> bool {
    let apart = |a: [V3; 2], b: [V3; 2]| (0..3).any(|k| a[0][k] > b[1][k] || a[1][k] < b[0][k]);
    matches!((a, b), (Some(a), Some(b)) if !apart(a, b))
}

/// [R^-1 | -R^-1 t] of a pose [R | t]: the general inverse, as poses composed with the rigid one drift from orthonormal
/// over chained merges. The snap's poses are products of rigid ones, checked at its entry.
fn invert(p: &M4) -> M4 {
    let r = rotation(p).try_inverse().expect("rigid poses have invertible rotations");
    pose(&r, &-(r * origin(p)))
}

/// Depth cache key of a relative pose: rotation to 1e-9, translation to 1e-7 LDU.
fn pose_key(rel: &M4) -> [i64; 12] {
    std::array::from_fn(|k| {
        let x = if k < 9 { rel[(k % 3, k / 3)] * 1e9 } else { rel[(k - 9, 3)] * 1e7 };
        x.round() as i64
    })
}

/// Part pairs (i < j) of n parts.
fn pairs(n: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..n).flat_map(move |i| (i + 1..n).map(move |j| (i, j)))
}

/// BrickNet's frame of a part at LDraw pose P: rounded to 12 digits and made orthonormal.
fn canonical(p: &M4) -> M4 {
    let written = |x: f64| format!("{x:.11e}").parse::<f64>().expect("a formatted float parses");
    let p = M4::from_fn(|i, j| if i < 3 { written(p[(i, j)]) } else { p[(i, j)] });
    let svd = rotation(&p).svd(true, true);
    let polar = svd.u.expect("computed") * svd.v_t.expect("computed");
    pose(&polar, &origin(&p))
}

/// Connectors of part `part` (type `id`) at LDraw pose P, placed at its BrickNet frame.
fn place<'a>(atlas: &'a Atlas, part: usize, id: u32, p: &M4) -> impl Iterator<Item = WorldConnector<'a>> + 'a {
    let frame = canonical(p);
    atlas.connectors(id).iter().map(move |conn| {
        let w = frame * conn.frame;
        let at = parts::Placed::new(trees::pose(&std::array::from_fn(|k| w[(k / 4, k % 4)])), conn.length);
        WorldConnector { part, conn, at }
    })
}

/// BrickNet's rule for connectors a, b of different parts (a first): None, or Some(fixed connection).
fn rule(atlas: &Atlas, a: &WorldConnector, b: &WorldConnector) -> Option<bool> {
    let pairing = atlas.pairing(a.conn, b.conn)?;
    pairing.joins(&a.at, &b.at).then_some(matches!(pairing, Pairing::Fixed(_)))
}

/// Connected pairs of placed connectors of different parts: (part a, part b, fixed). Two sweeps along the axis the
/// connector origins spread most, `near` the prefilter: all pairs but axle ones (axles: the connectors with a length)
/// within REACH, then axle pairs within the reach of the longest axles.
fn connections(atlas: &Atlas, placed: &[WorldConnector]) -> Vec<(usize, usize, bool)> {
    let at = |i: usize| placed[i].at.pose.translation;
    let origins = || placed.iter().map(|c| c.at.pose.translation);
    let (lo, hi) = (origins().reduce(|a, b| a.min(b)), origins().reduce(|a, b| a.max(b)));
    let axis = lo.zip(hi).map_or(0, |(lo, hi)| V3::from((hi - lo).to_array()).imax());
    let is_axle = |c: &WorldConnector| c.conn.length > 0.0;
    let mut out = Vec::new();
    let mut sweep = |mut by: Vec<usize>, width: f64, axles: bool| {
        by.sort_by(|&i, &j| at(i)[axis].total_cmp(&at(j)[axis]));
        for (s, &i) in by.iter().enumerate() {
            for &j in &by[s + 1..] {
                if at(j)[axis] - at(i)[axis] > width {
                    break;
                }
                let (a, b) = (&placed[i.min(j)], &placed[i.max(j)]);
                if a.part == b.part || (is_axle(a) && is_axle(b)) != axles || !near(&a.at, &b.at) {
                    continue;
                }
                if let Some(fixed) = rule(atlas, a, b) {
                    out.push((a.part, b.part, fixed));
                }
            }
        }
    };
    sweep((0..placed.len()).collect(), REACH, false);
    let axles: Vec<usize> = (0..placed.len()).filter(|&i| is_axle(&placed[i])).collect();
    let longest = axles.iter().map(|&i| placed[i].conn.length.max(AXLE_LENGTH)).fold(0.0, f64::max);
    sweep(axles, 2.0 * longest + AXLE_SWEEP, true);
    out
}

/// Angle between the rotations of A and B in radians, accurate near 0.
fn angle(a: &M4, b: &M4) -> f64 {
    let r = rotation(a).transpose() * rotation(b);
    (vee(&r).norm() * 0.5).atan2((r.trace() - 1.0) * 0.5)
}

fn close(a: &M4, b: &M4, t: f64, r: f64) -> bool {
    (origin(a) - origin(b)).norm() <= t && angle(a, b) <= r
}

/// Rms displacement of a body's mass between poses A and B: sqrt(|dR c + dt|^2 + tr(dR S dR^T)), LDU.
fn moved(a: &M4, b: &M4, m: &Moments) -> f64 {
    let dr = rotation(a) - rotation(b);
    ((dr * m.c + origin(a) - origin(b)).norm_squared() + (dr * m.s).dot(&dr)).sqrt()
}

/// Placements of b in a's frame from seat s of region r in `cell`: T*, then the actual joint in the cell.
fn seats(r: &Region, cell: &Cell, s: &seat::Seat, rel: &M4) -> Vec<M4> {
    let mut out = vec![if s.d2 < AT_RAW { *rel } else { placement(r, &s.q) }];
    match r.family {
        FIXED => {
            // the actual turn about the connector axis, where the y axes are near parallel
            let j = joint(r, rel);
            if j[(1, 1)] > 0.999 {
                let turn = rot(1, j[(0, 2)].atan2(j[(0, 0)]).to_degrees());
                out.push(r.fa * pose(&turn, &V3::zeros()) * invert(&r.fb));
            }
        }
        BALL => {
            let (_, q) = into_cell(BALL, &actual(r, rel), cell);
            out.push(placement(r, &q));
        }
        _ => {
            // the actual yaw, and its nearest multiple of 90 deg
            let candidates = actual(r, rel);
            let rounded = candidates.map(|c| [c[0], 90.0 * (c[1] / 90.0).round(), c[2]]);
            for x in [candidates, rounded] {
                let (_, q) = into_cell(r.family, &x, cell);
                out.push(placement(r, &q));
            }
        }
    }
    out
}

/// Solved hypothesis of a link's pair: its region, placements of b in a's frame, and D^2.
struct Solved<'a> {
    region: &'a Region,
    placements: Vec<M4>,
    d2: f64,
}

/// Option of a link: v's pose in u's frame, its count k, rms displacement, and whether it is the raw pose.
#[derive(Clone, Copy)]
struct Opt {
    t: M4,
    k: usize,
    dist: f64,
    raw: bool,
    feasible: Option<bool>,
}

/// Linked part pair u < v and its options.
struct Link {
    u: usize,
    v: usize,
    options: Vec<Opt>,
}

/// Greedy state: clusters, their members and gauges (None: at the raw poses), and links taken.
#[derive(Clone)]
struct State {
    cluster: Vec<usize>,
    x: Vec<M4>,
    members: Vec<Vec<usize>>,
    gauge: Vec<Option<M4>>,
    taken: Vec<bool>,
}

/// One assembly to snap: its links and raw fixed links, the raw depth of each pair whose colliders overlap at the raw
/// poses (measured on first use), the depths found so far, the options checked for their pair alone, the
/// constraints, and the greedy pass.
struct Assembly<'a> {
    atlas: &'a Atlas,
    ids: &'a [u32],
    raw: &'a [M4],
    moments: Vec<&'a Moments>,
    links: Vec<Link>,
    linked: HashSet<(usize, usize)>,
    fixed: HashSet<(usize, usize)>,
    raw_depth: HashMap<(usize, usize), Option<f64>>,
    cache: HashMap<(usize, usize, [i64; 12]), f64>,
}

impl Assembly<'_> {
    fn init(&self) -> State {
        let n = self.ids.len();
        State {
            cluster: (0..n).collect(),
            x: self.raw.to_vec(),
            members: (0..n).map(|i| vec![i]).collect(),
            gauge: vec![None; n],
            taken: vec![false; self.links.len()],
        }
    }

    /// Whether parts i, j at poses pi, pj connect, and whether by a fixed connection.
    fn link(&self, i: usize, j: usize, pi: &M4, pj: &M4) -> (bool, bool) {
        let placed: Vec<WorldConnector> =
            place(self.atlas, i, self.ids[i], pi).chain(place(self.atlas, j, self.ids[j], pj)).collect();
        let c = connections(self.atlas, &placed);
        (!c.is_empty(), c.iter().any(|e| e.2))
    }

    /// Depths of pairs (i, j) at poses (P_i, P_j), cached by relative pose; the missing ones are queried in one call.
    fn measure(&mut self, at: &[(usize, usize, M4, M4)], depths: &mut Depths) -> Vec<f64> {
        let keys: Vec<_> = at.iter().map(|(i, j, pi, pj)| (*i, *j, pose_key(&(invert(pi) * pj)))).collect();
        let need: Vec<usize> = (0..at.len()).filter(|&k| !self.cache.contains_key(&keys[k])).collect();
        if !need.is_empty() {
            let found = depths(&need.iter().map(|&k| at[k]).collect::<Vec<_>>());
            for (&k, d) in need.iter().zip(found) {
                self.cache.insert(keys[k], d);
            }
        }
        keys.iter().map(|key| self.cache[key]).collect()
    }

    /// Pairs (i < j) at poses q that collide anew. A pair at its raw relative pose keeps its raw depth, so only fixed
    /// pairs there are measured. A raw depth is measured where it may show a pair safe, or once the depth at q crosses
    /// an allowance.
    fn collisions(&mut self, pairs: &[(usize, usize)], q: &[M4], depths: &mut Depths) -> Vec<(usize, usize)> {
        // (i, j, how far j's box moved in i's frame) of the pairs whose collider boxes overlap, to measure
        let near = |&&(i, j): &&(usize, usize)| {
            overlap(aabb(self.atlas, self.ids[i], &q[i]), aabb(self.atlas, self.ids[j], &q[j]))
        };
        let mut moved = Vec::new();
        for &(i, j) in pairs.iter().filter(near) {
            let rel = invert(&q[i]) * q[j];
            let raw_rel = invert(&self.raw[i]) * self.raw[j];
            if self.fixed.contains(&(i, j)) || pose_key(&rel) != pose_key(&raw_rel) {
                moved.push((i, j, self.shift(j, &(rel - raw_rel))));
            }
        }
        // a raw depth may show safe only a pair whose box moved less than the smallest allowance
        let small: Vec<(usize, usize)> = moved.iter().filter(|p| p.2 <= TAUS[0] - 1e-9).map(|p| (p.0, p.1)).collect();
        self.measure_raw(&small, depths);
        let suspects: Vec<_> = moved.into_iter().filter(|&(i, j, shift)| !self.allowed(i, j, shift, None)).collect();
        let at: Vec<_> = suspects.iter().map(|&(i, j, _)| (i, j, q[i], q[j])).collect();
        let found = self.measure(&at, depths);
        let deep: Vec<(usize, usize)> =
            suspects.iter().zip(&found).filter(|&(_, &d)| d > TAUS[0]).map(|(&(i, j, _), _)| (i, j)).collect();
        self.measure_raw(&deep, depths);
        let collides = |(&(i, j, shift), &d): (&(usize, usize, f64), &f64)| {
            !self.allowed(i, j, shift, Some(d)) && !self.link(i, j, &q[i], &q[j]).1
        };
        suspects.iter().zip(&found).filter(|&p| collides(p)).map(|(&(i, j, _), _)| (i, j)).collect()
    }

    /// How far a corner of part j's collider box moves under `delta`, a change of its pose in another part's frame
    /// (LDU).
    fn shift(&self, j: usize, delta: &M4) -> f64 {
        let [lo, hi] = self.atlas.bounds(self.ids[j]).expect("near parts have colliders");
        let corner = |c: usize| V3::from_fn(|k, _| if c >> k & 1 == 0 { lo[k] } else { hi[k] }).push(1.0);
        (0..8).map(|c| (delta * corner(c)).norm()).fold(0.0, f64::max)
    }

    /// Measure the raw depths not yet measured of pairs whose colliders overlap at the raw poses.
    fn measure_raw(&mut self, pairs: &[(usize, usize)], depths: &mut Depths) {
        let unmeasured = pairs.iter().filter(|p| self.raw_depth.get(p) == Some(&None));
        let at: Vec<_> = unmeasured.map(|&(i, j)| (i, j, self.raw[i], self.raw[j])).collect();
        for (&(i, j, ..), d) in at.iter().zip(self.measure(&at, depths)) {
            self.raw_depth.insert((i, j), Some(d));
        }
    }

    /// Whether pair (i, j) stays within each allowance its raw depth was within (a fixed pair: every allowance), by its
    /// depth d at q (None: not measured) or as j's box moved at most `shift` from the raw poses, which bounds how far a
    /// depth moves. A raw depth not measured counts as 0.
    fn allowed(&self, i: usize, j: usize, shift: f64, d: Option<f64>) -> bool {
        let true_raw = self.raw_depth.get(&(i, j)).copied().flatten().unwrap_or(0.0);
        let raw = if self.fixed.contains(&(i, j)) { 0.0 } else { true_raw };
        let kept = |tau: f64| raw > tau || true_raw + shift <= tau - 1e-9 || d.is_some_and(|d| d <= tau);
        TAUS.iter().all(|&tau| kept(tau))
    }

    /// Rigid G minimizing sum m_i w_i MSD(G x_i, raw_i) over a cluster within the cap, or None.
    fn gauge(&self, members: &[usize], x: &[M4]) -> Option<M4> {
        let mut w = vec![1.0; members.len()];
        let centroid = |p: &M4, i: usize| (p * self.moments[i].c.push(1.0)).xyz();
        for _ in 0..24 {
            let mass: Vec<f64> = members.iter().zip(&w).map(|(&i, w)| self.moments[i].mass * w).collect();
            let total = mass.iter().sum::<f64>();
            let center =
                |p: &[M4]| members.iter().zip(&mass).map(|(&i, m)| centroid(&p[i], i) * *m).sum::<V3>() / total;
            let cx = center(x);
            let cy = center(self.raw);
            let mut h = M3::zeros();
            for (&i, m) in members.iter().zip(&mass) {
                let z = centroid(&x[i], i) - cx;
                let y = centroid(&self.raw[i], i) - cy;
                let spread = rotation(&x[i]) * self.moments[i].s * rotation(&self.raw[i]).transpose();
                h += (z * y.transpose() + spread) * *m;
            }
            let r = fit(&h);
            let g = pose(&r, &(cy - r * cx));
            let mut over = false;
            for (k, &i) in members.iter().enumerate() {
                let p = g * x[i];
                let e = (moved(&p, &self.raw[i], self.moments[i]) / CAP_T).max(angle(&p, &self.raw[i]) / CAP_R);
                if e > 1.0 - 1e-9 {
                    w[k] *= 2.0 * e * e;
                    over = true;
                }
            }
            if !over {
                return Some(g);
            }
        }
        None
    }

    /// s with option oi of link li taken, or None if it is inconsistent or breaks a constraint. Two raw clusters merge
    /// in place by a raw option; otherwise v's cluster moves onto the option.
    fn apply(&mut self, s: &State, li: usize, oi: usize, depths: &mut Depths) -> Option<State> {
        let u = self.links[li].u;
        let v = self.links[li].v;
        let o = self.links[li].options[oi];
        let cu = s.cluster[u];
        let cv = s.cluster[v];
        if cu == cv && !close(&(invert(&s.x[u]) * s.x[v]), &o.t, CONS_T, CONS_R) {
            return None;
        }
        let mut t = s.clone();
        t.taken[li] = true;
        if cu == cv {
            return Some(t);
        }
        let in_place = s.gauge[cu].is_none() && s.gauge[cv].is_none() && o.raw;
        let m = if in_place { M4::identity() } else { s.x[u] * o.t * invert(&s.x[v]) };
        for &w in &s.members[cv] {
            t.x[w] = m * s.x[w];
            t.cluster[w] = cu;
        }
        t.members[cu].extend_from_slice(&s.members[cv]);
        t.members[cv].clear();
        if in_place {
            return Some(t);
        }
        let g = self.gauge(&t.members[cu], &t.x)?;
        t.gauge[cu] = Some(g);
        let q: Vec<M4> = t.x.iter().map(|x| g * x).collect();
        let pair = |a: usize| s.members[cv].iter().map(move |&b| (a.min(b), a.max(b)));
        let across: Vec<(usize, usize)> = s.members[cu].iter().flat_map(|&a| pair(a)).collect();
        if across.iter().any(|&(i, j)| self.linked.contains(&(i, j)) && !self.link(i, j, &q[i], &q[j]).0) {
            return None;
        }
        self.collisions(&across, &q, depths).is_empty().then_some(t)
    }

    /// Whether option oi of link li applies to its pair alone (checked once); otherwise it applies in no cluster.
    fn feasible(&mut self, init: &State, li: usize, oi: usize, depths: &mut Depths) -> bool {
        if let Some(ok) = self.links[li].options[oi].feasible {
            return ok;
        }
        let ok = self.apply(init, li, oi, depths).is_some();
        self.links[li].options[oi].feasible = Some(ok);
        ok
    }

    fn pose(&self, s: &State, i: usize) -> M4 {
        s.gauge[s.cluster[i]].map_or(self.raw[i], |g| g * s.x[i])
    }

    /// Part pairs that lost their raw link or collide anew at the poses of s.
    fn violations(&mut self, s: &State, depths: &mut Depths) -> Vec<(usize, usize)> {
        let n = self.ids.len();
        let q: Vec<M4> = (0..n).map(|i| self.pose(s, i)).collect();
        let moved: Vec<bool> = (0..n).map(|i| s.gauge[s.cluster[i]].is_some()).collect();
        let lost_link = |&(u, v): &(usize, usize)| (moved[u] || moved[v]) && !self.link(u, v, &q[u], &q[v]).0;
        let lost: Vec<(usize, usize)> = self.links.iter().map(|l| (l.u, l.v)).filter(lost_link).collect();
        if !lost.is_empty() {
            return lost;
        }
        let moving: Vec<(usize, usize)> = pairs(n).filter(|&(i, j)| moved[i] || moved[j]).collect();
        self.collisions(&moving, &q, depths)
    }

    /// State from the initial one where each link of `order` not forbidden takes its first feasible option that
    /// applies.
    fn greedy(&mut self, order: &[usize], forbid: &HashSet<usize>, depths: &mut Depths) -> State {
        let init = self.init();
        let mut s = init.clone();
        for &li in order.iter().filter(|li| !forbid.contains(li)) {
            for oi in 0..self.links[li].options.len() {
                if !self.feasible(&init, li, oi, depths) {
                    continue;
                }
                if let Some(t) = self.apply(&s, li, oi, depths) {
                    s = t;
                    break;
                }
            }
        }
        s
    }

    /// Greedy pass, repeated with the links of violating pairs forbidden.
    fn repair(&mut self, order: &[usize], depths: &mut Depths) -> State {
        let mut forbid = HashSet::new();
        let mut repairs = 0;
        let mut s = self.greedy(order, &forbid, depths);
        loop {
            let bad = self.violations(&s, depths);
            if bad.is_empty() {
                return s;
            }
            repairs += 1;
            if repairs > MAX_REPAIRS {
                return self.init();
            }
            let parts: HashSet<usize> = bad.iter().flat_map(|&(a, b)| [a, b]).collect();
            let taken_at = |&li: &usize| {
                let link = &self.links[li];
                s.taken[li] && (parts.contains(&link.u) || parts.contains(&link.v))
            };
            let before = forbid.len();
            forbid.extend(order.iter().copied().filter(taken_at));
            if forbid.len() == before {
                let clusters: HashSet<usize> = parts.iter().map(|&i| s.cluster[i]).collect();
                forbid.extend(order.iter().copied().filter(|&li| clusters.contains(&s.cluster[self.links[li].u])));
            }
            s = self.greedy(order, &forbid, depths);
        }
    }
}

/// Snap of parts `ids` at rigid raw poses, `depths` measuring penetration; an error for a part without mass
/// properties.
pub fn snap(atlas: &Atlas, ids: &[u32], raw: &[M4], depths: &mut Depths<'_>) -> io::Result<Vec<M4>> {
    let n = ids.len();
    let moments = |&t: &u32| atlas.moments(t).ok_or_else(|| bad(&format!("part {t} has no mass properties")));
    let moments: Vec<&Moments> = ids.iter().map(moments).collect::<io::Result<_>>()?;
    let placed: Vec<WorldConnector> = (0..n).flat_map(|i| place(atlas, i, ids[i], &raw[i])).collect();
    let edges = connections(atlas, &placed);
    let pair = |&(a, b, _): &(usize, usize, bool)| (a.min(b), a.max(b));
    let linked: HashSet<_> = edges.iter().map(pair).collect();
    let links = links(atlas, ids, raw, &linked, &moments);
    let boxes: Vec<_> = (0..n).map(|i| aabb(atlas, ids[i], &raw[i])).collect();
    let mut assembly = Assembly {
        atlas,
        ids,
        raw,
        moments,
        links,
        linked,
        fixed: edges.iter().filter(|e| e.2).map(pair).collect(),
        raw_depth: pairs(n).filter(|&(i, j)| overlap(boxes[i], boxes[j])).map(|p| (p, None)).collect(),
        cache: HashMap::new(),
    };
    let init = assembly.init();
    for li in 0..assembly.links.len() {
        let count = assembly.links[li].options.len();
        let first = (0..count).find(|&oi| assembly.feasible(&init, li, oi, depths)).unwrap_or(count);
        assembly.links[li].options.drain(..first);
    }
    let order = link_order(&assembly, &edges);
    let state = assembly.repair(&order, depths);
    Ok((0..n).map(|i| assembly.pose(&state, i)).collect())
}

/// Links (u, v) in order with their options, from the hypotheses of their pairs solved at the raw poses.
fn links(atlas: &Atlas, ids: &[u32], raw: &[M4], linked: &HashSet<(usize, usize)>, moments: &[&Moments]) -> Vec<Link> {
    let (pairs, hyps) = seat::hypotheses(atlas, ids, raw, RADIUS, |i, j| linked.contains(&(i, j)));
    let mut solved: HashMap<(usize, usize), Vec<Solved>> = HashMap::new();
    for h in &hyps {
        let p = &pairs[h.pair];
        let region = &p.regions[h.region];
        let s = seat::start(atlas, &pairs, h).search(false, f64::INFINITY);
        let placements = seats(region, &atlas.body(region).cells[s.cell], &s, &p.rel);
        solved.entry((p.a.min(p.b), p.a.max(p.b))).or_default().push(Solved { region, placements, d2: s.d2 });
    }
    let mut order: Vec<(usize, usize)> = linked.iter().copied().collect();
    order.sort_unstable();
    let link = |(u, v): (usize, usize)| {
        let hyps = solved.get(&(u, v)).map_or(&[][..], |h| h);
        let rel = invert(&raw[u]) * raw[v];
        Link { u, v, options: options(hyps, ids[u] <= ids[v], &rel, moments[v]) }
    };
    order.into_iter().map(link).collect()
}

/// Greedy order of the links with a feasible option, each ranked by its first one: per BrickNet component,
/// largest first (ties by lowest part), breadth first from the heaviest part.
fn link_order(assembly: &Assembly, edges: &[(usize, usize, bool)]) -> Vec<usize> {
    let n = assembly.ids.len();
    let links = &assembly.links;
    let mut components = components(n, edges.iter().map(|&(a, b, _)| (a, b)));
    components.sort_by_key(|c| std::cmp::Reverse(c.len()));
    let rank = |li: usize| links[li].options.first().map_or((0, f64::INFINITY), |o| (o.k, o.dist));
    let by_rank = |&a: &usize, &b: &usize| {
        let (ka, da) = rank(a);
        let (kb, db) = rank(b);
        kb.cmp(&ka).then(da.total_cmp(&db)).then(a.cmp(&b))
    };
    let mass = |i: usize| assembly.moments[i].mass;
    let mut used = vec![false; links.len()];
    let mut seen = vec![false; n];
    let mut order = Vec::new();
    for c in &components {
        let heaviest = |&a: &&usize, &b: &&usize| mass(*a).total_cmp(&mass(*b)).then(b.cmp(a));
        let root = *c.iter().max_by(heaviest).expect("components are not empty");
        seen[root] = true;
        let mut frontier = vec![root];
        while !frontier.is_empty() {
            let at_frontier = |li: &usize| frontier.iter().any(|&x| x == links[*li].u || x == links[*li].v);
            let mut level: Vec<usize> = (0..links.len()).filter(|&li| !used[li]).filter(at_frontier).collect();
            level.sort_by(by_rank);
            frontier.clear();
            for li in level {
                used[li] = true;
                order.push(li);
                for w in [links[li].u, links[li].v] {
                    if !seen[w] {
                        seen[w] = true;
                        frontier.push(w);
                    }
                }
            }
        }
    }
    order.retain(|&li| !links[li].options.is_empty());
    order
}

/// Distinct connector pairs of a part pair's hypotheses whose frames coincide when b sits at T in a's frame.
fn coincident(hyps: &[Solved], t: &M4) -> usize {
    let mut pairs = HashSet::new();
    for h in hyps {
        let r = h.region;
        let d = invert(&r.fa) * t * r.fb;
        let off = if r.family == AXLE { d[(0, 3)].hypot(d[(2, 3)]) } else { origin(&d).norm() };
        if off < COINCIDENT && d[(1, 1)].abs() > 1.0 - 1e-6 {
            pairs.insert((r.ca, r.cb));
        }
    }
    pairs.len()
}

/// Options of link (u, v) from its pair's hypotheses: poses of v in u's frame; rel: v's raw pose there.
fn options(hyps: &[Solved], a_is_u: bool, rel: &M4, mv: &Moments) -> Vec<Opt> {
    let mut opts: Vec<Opt> = Vec::new();
    for h in hyps {
        for (si, t) in h.placements.iter().enumerate() {
            let tuv = if a_is_u { *t } else { invert(t) };
            let raw = si == 0 && h.d2 < AT_RAW;
            let dist = moved(&tuv, rel, mv);
            if !raw && (dist > 2.0 * CAP_T + 1e-9 || angle(&tuv, rel) > 2.0 * CAP_R + 1e-9) {
                continue;
            }
            match opts.iter_mut().find(|o| close(&o.t, &tuv, 1e-6, 1e-8)) {
                Some(o) if raw && !o.raw => {
                    o.t = tuv;
                    o.dist = 0.0;
                    o.raw = true;
                }
                Some(_) => {}
                None => opts.push(Opt {
                    t: if raw { *rel } else { tuv },
                    k: coincident(hyps, t),
                    dist: if raw { 0.0 } else { dist },
                    raw,
                    feasible: None,
                }),
            }
        }
    }
    if opts.iter().any(|o| o.raw) {
        opts.retain(|o| o.raw);
    }
    opts.retain(|o| o.k > 0);
    opts.sort_by(|a, b| b.k.cmp(&a.k).then(a.dist.total_cmp(&b.dist)));
    opts
}
