//! Training couplings and mass movement.

use std::cell::RefCell;
use std::io::Result;
use std::path::Path;

use nalgebra::{Quaternion, SymmetricEigen};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

use crate::certify;
use crate::geometry::{fit, origin, quat, rotation, M3, M4, Q, V3};
use procbrick::coding::Records;
use procbrick::store::bad;
use procbrick::trees::{Coding, PARTS};

pub const ROW: usize = PARTS + 2; // tokens per training row

/// Training row: types, the two samples' part counts, the time, and per token the 13 outputs of a `Sample`.
pub struct Row {
    pub types: [i64; ROW],
    pub lengths: [i64; 2],
    pub t: f32,
    pub tokens: Vec<[f32; 13]>,
}

/// Coupled sample: its time, the per-slot outputs, and the part in each slot.
pub struct Sample {
    pub t: f64,
    pub out: Vec<[f32; 13]>,
    pub order: Vec<usize>,
}

/// Centers of mass (times `scale`) and rotations of parts at poses at their LDraw origins.
pub fn frames(parts: &[&Part], poses: &[M4], scale: f64) -> (Vec<V3>, Vec<Q>) {
    let x = parts.iter().zip(poses).map(|(part, p)| (rotation(p) * part.com + origin(p)) * scale).collect();
    let q = poses.iter().map(|p| Q::new_normalize(quat(&rotation(p)).into_inner())).collect();
    (x, q)
}

/// Random stream of (seed, position, stream).
fn rng(seed: u64, position: u64, stream: u64) -> SmallRng {
    SmallRng::seed_from_u64(seed.wrapping_mul(0x9E3779B97F4A7C15) ^ position.wrapping_mul(0xD1B54A32D192ED03) ^ stream)
}

/// q or -q, whichever is closer to r.
fn toward(q: Q, r: &Q) -> Q {
    let sign = if q.coords.dot(&r.coords) < 0.0 { -1.0 } else { 1.0 };
    Q::new_unchecked(q.into_inner() * sign)
}

/// Part type in scaled units: mass, center of mass, second moment S and inertia J = tr(S) I - S (both about it), its
/// symmetries, and how far they move its inertia. The best twist about a continuous symmetry axis is found in
/// closed form.
pub struct Part {
    mass: f64,
    com: V3,
    s: M3,
    inertia: M3,
    orbit: Vec<Q>,
    axis: Option<V3>,
    mats: Vec<M3>,
    eta: f64,
}

impl Part {
    pub fn new(mass: f64, com: V3, second: M3, orbit: &[Q], axis: V3, scale: f64) -> Self {
        let s = second * scale.powi(5);
        let axis = (axis.norm_squared() > 0.5).then_some(axis);
        let mut kept: Vec<Q> = Vec::new();
        for g in orbit {
            // g is in the coset of a kept h when h^-1 g turns about the axis
            let same_coset = |a: V3| kept.iter().any(|h| (h.inverse() * g).imag().cross(&a).norm_squared() < 1e-12);
            if !axis.is_some_and(same_coset) {
                kept.push(*g);
            }
        }
        // tr(A g R_theta S) = tr(A Y) + cos theta tr(A (g S - Y)) + sin theta tr(A g [a]x S), Y = g a a^T S (axis a)
        let mut mats = Vec::new();
        for g in &kept {
            let r = g.to_rotation_matrix().into_inner();
            match axis {
                None => mats.push(r * s),
                Some(a) => {
                    let y = r * a * a.transpose() * s;
                    mats.extend([y, r * s - y, r * a.cross_matrix() * s]);
                }
            }
        }
        // how far the symmetries move the inertia J = tr(S) I - S (Frobenius): over the orbit, and for a continuous
        // axis twice the part of S not symmetric about it
        let inertia = M3::identity() * s.trace() - s;
        let moved = |g: &Q| {
            let r = g.to_rotation_matrix().into_inner();
            (r * inertia * r.transpose() - inertia).norm()
        };
        let mut eta = orbit.iter().map(moved).fold(0.0, f64::max);
        if let Some(a) = axis {
            let along = a.dot(&(s * a));
            let symmetric =
                a * a.transpose() * along + (M3::identity() - a * a.transpose()) * ((s.trace() - along) / 2.0);
            eta += 2.0 * (s - symmetric).norm();
        }
        Part { mass: mass * scale.powi(3), com, s, inertia, orbit: kept, axis, mats, eta }
    }

    /// Symmetry g maximizing max_theta tr(A g R_theta S), and that maximum, given tr(A M) of each of `mats`.
    fn pick(&self, tr: impl Fn(usize) -> f64) -> (usize, f64) {
        let value = |c: usize| match self.axis {
            Some(_) => {
                let (x, z) = (tr(3 * c + 1), tr(3 * c + 2));
                tr(3 * c) + (x * x + z * z).sqrt()
            }
            None => tr(c),
        };
        (0..self.orbit.len()).map(|c| (c, value(c))).fold((0, f64::NEG_INFINITY), |b, a| if a.1 > b.1 { a } else { b })
    }

    /// Symmetric equivalent q g_c R_theta of q closest to a slot rotation (matrix r0), as the sign of it closest to
    /// q0; c is `known` if given (a part without a continuous axis), 0 for a part without symmetries.
    fn turn(&self, q: Q, r0: &M3, q0: &Q, known: Option<usize>) -> Q {
        let (c, theta) = match (known, self.axis) {
            (Some(c), None) => (c, 0.0),
            (None, None) if self.orbit.len() == 1 => (0, 0.0),
            _ => {
                let a = r0.transpose() * q.to_rotation_matrix().into_inner();
                let (c, _) = self.pick(|m| a.tr_dot(&self.mats[m]));
                let tr = |j: usize| a.tr_dot(&self.mats[3 * c + j]);
                (c, self.axis.map_or(0.0, |_| tr(2).atan2(tr(1))))
            }
        };
        let twist = self.axis.map_or(Q::identity(), |axis| Q::from_scaled_axis(axis * theta));
        toward(q * self.orbit[c] * twist, q0)
    }

    /// Squared displacements (translation, rotation) from a slot at (x0, r0) to (x, q): m |x0 - x|^2 and
    /// tr((R - R0) S (R - R0)^T).
    fn cost(&self, x0: &V3, r0: &M3, x: &V3, q: &Q) -> [f64; 2] {
        let d = q.to_rotation_matrix().into_inner() - r0;
        [self.mass * (x0 - x).norm_squared(), (d * self.s).component_mul(&d).sum()]
    }

    /// Squared displacement from (x0, q0) to the symmetric equivalent of (x, q) nearest it.
    pub fn distance2(&self, x0: &V3, q0: &Q, x: &V3, q: &Q) -> f64 {
        let r0 = q0.to_rotation_matrix().into_inner();
        let [t, r] = self.cost(x0, &r0, x, &self.turn(*q, &r0, q0, None));
        t + r
    }
}

/// Reset `b` to n copies of `value`.
fn reset<T: Clone>(b: &mut Vec<T>, n: usize, value: T) {
    b.clear();
    b.resize(n, value);
}

/// Assignment buffers, reused: the n x n cost (row-major) and, at warm prices, per row its cheapest column and least
/// reduced cost; the column prices, path lengths, the column of each row, the row of each column, the previous row on
/// each path, and the columns not yet and already reached.
#[derive(Default)]
struct Lap {
    cost: Vec<f64>,
    least: Vec<(usize, f64)>,
    v: Vec<f64>,
    d: Vec<f64>,
    col: Vec<usize>,
    row: Vec<usize>,
    pred: Vec<usize>,
    todo: Vec<usize>,
    done: Vec<usize>,
}

impl Lap {
    /// Least assignment of the n x n cost by shortest augmenting paths, the column of each row into `col`. Column
    /// prices start from `price` if it holds n of them (warm, `least` at them), else from the column minima, and are
    /// written back. Once the dual bound at the starting prices (a lower bound on the least assignment) reaches
    /// `stop`, that bound instead.
    fn assign(&mut self, n: usize, price: &mut Vec<f64>, stop: f64) -> f64 {
        let Lap { cost, least, v, d, col, row, pred, todo, done } = self;
        let c = |i: usize, j: usize| cost[i * n + j];
        let warm = price.len() == n;
        v.clear();
        if warm {
            v.extend_from_slice(price);
        } else {
            v.extend((0..n).map(|j| (0..n).map(|i| c(i, j)).fold(f64::INFINITY, f64::min)));
        }
        // a row whose cheapest column is free takes it; the dual: sum over rows of the least c - v, plus sum v
        let free = usize::MAX;
        reset(col, n, free);
        reset(row, n, free);
        let mut dual: f64 = v.iter().sum();
        for (i, column) in col.iter_mut().enumerate() {
            let (j, least) = if warm {
                least[i]
            } else {
                (0..n).map(|j| (j, c(i, j) - v[j])).fold((0, f64::INFINITY), |b, a| if a.1 < b.1 { a } else { b })
            };
            dual += least;
            if row[j] == free {
                (*column, row[j]) = (j, i);
            }
        }
        if dual >= stop {
            return dual;
        }
        // every other row by the shortest path of reduced costs to a free column (Dijkstra), then the prices of the
        // columns reached on the way lowered by their distance short of the path's
        for f in 0..n {
            if col[f] != free {
                continue;
            }
            d.clear();
            d.extend((0..n).map(|j| c(f, j) - v[j]));
            reset(pred, n, f);
            todo.clear();
            todo.extend(0..n);
            done.clear();
            let (end, length) = loop {
                let (k, _) = todo
                    .iter()
                    .enumerate()
                    .fold((0, f64::INFINITY), |b, (k, &j)| if d[j] < b.1 { (k, d[j]) } else { b });
                let j = todo.swap_remove(k);
                if row[j] == free {
                    break (j, d[j]);
                }
                done.push(j);
                let i = row[j];
                let base = c(i, j) - v[j] - d[j];
                for &k in todo.iter() {
                    let h = c(i, k) - v[k] - base;
                    if h < d[k] {
                        (d[k], pred[k]) = (h, i);
                    }
                }
            };
            for &j in done.iter() {
                v[j] += d[j] - length;
            }
            let mut j = end;
            loop {
                let i = pred[j];
                row[j] = i;
                (col[i], j) = (j, col[i]);
                if i == f {
                    break;
                }
            }
        }
        price.clone_from(v);
        col.iter().enumerate().map(|(i, &j)| c(i, j)).sum()
    }
}

/// Problem buffers, reused: per slot the traces of each class's symmetry matrices, per slot and class the best
/// matrix, and per column the rotation term; the assignment, and those of the groups whose assignment waits, with
/// their indices and duals; and per slot its part's known symmetry.
#[derive(Default)]
struct Work {
    traces: Vec<f64>,
    best: Vec<usize>,
    rotations: Vec<f64>,
    lap: Lap,
    kept: Vec<Lap>,
    deferred: Vec<(usize, f64)>,
    known: Vec<Option<usize>>,
}

/// Center of mass of parts at x, in turn.
fn centroid<'v>(parts: &[&Part], x: impl IntoIterator<Item = &'v V3>) -> V3 {
    let total: f64 = parts.iter().map(|p| p.mass).sum();
    parts.iter().zip(x).map(|(p, x)| x * p.mass).sum::<V3>() / total
}

/// One side of a problem (slots or parts): per part its position and rotation (also as a matrix), and their center
/// of mass.
#[derive(Clone)]
struct Side<'a> {
    x: &'a [V3],
    q: &'a [Q],
    r: Vec<M3>,
    c: V3,
}

impl<'a> Side<'a> {
    fn new(parts: &[&Part], x: &'a [V3], q: &'a [Q]) -> Self {
        Side { x, q, r: q.iter().map(|q| q.to_rotation_matrix().into_inner()).collect(), c: centroid(parts, x) }
    }

    /// Distances of parts g from the center, sorted.
    fn distances(&self, g: &[usize]) -> Vec<f64> {
        let mut r: Vec<f64> = g.iter().map(|&i| (self.x[i] - self.c).norm()).collect();
        r.sort_by(f64::total_cmp);
        r
    }

    /// Inertia tensor of the parts about the center.
    fn inertia(&self, parts: &[&Part]) -> M3 {
        let mut inertia = M3::zeros();
        for ((part, x), r) in parts.iter().zip(self.x).zip(&self.r) {
            let d = x - self.c;
            inertia +=
                (M3::identity() * d.norm_squared() - d * d.transpose()) * part.mass + r * part.inertia * r.transpose();
        }
        inertia
    }

    /// Principal axes of the parts' positions by decreasing mass-weighted variance.
    fn axes(&self, parts: &[&Part]) -> M3 {
        let cov = parts.iter().zip(self.x).map(|(p, x)| (x - self.c) * (x - self.c).transpose() * p.mass).sum::<M3>();
        let eig = SymmetricEigen::new(cov);
        let mut idx = [0, 1, 2];
        idx.sort_by(|&a, &b| eig.eigenvalues[b].total_cmp(&eig.eigenvalues[a]));
        M3::from_columns(&idx.map(|i| eig.eigenvectors.column(i).into_owned()))
    }
}

/// Uniform random rotation from four standard normals.
fn haar(normal: &mut impl FnMut() -> f64) -> Q {
    Q::from_quaternion(Quaternion::new(normal(), normal(), normal(), normal()))
}

/// Group of two or more interchangeable parts (one key): its parts by symmetry class, and the products and lower bound
/// its assignment uses.
struct Group {
    parts: Vec<usize>,
    runs: Vec<usize>,
    products: Vec<f64>,
    places: Vec<f64>,
    fixed: f64,
    floor: f64,
}

/// Parts of keys of their own, whose cost at rotation R is `fixed` - 2 <R, `linear`> - 2 sum over `others` of the
/// most of <R, U> over its matrices U (a part of one matrix and no continuous axis is in `linear`): per part
/// m (|x0 - c0|^2 + |x1 - c1|^2) + 2 tr S, m (x0 - c0) (x1 - c1)^T, and per symmetry matrix M, U = R0 M^T R1^T (so
/// that tr(R0^T R R1 M) = <R, U>).
#[derive(Default)]
struct Singles {
    fixed: f64,
    linear: M3,
    others: Vec<(usize, Vec<M3>)>,
}

/// m (|x0 - c0|^2 + |x1 - c1|^2) of part i: its slot's and its own squared distance from their centroids.
fn offsets(part: &Part, slots: &Side, moved: &Side, i: usize) -> f64 {
    part.mass * ((slots.x[i] - slots.c).norm_squared() + (moved.x[i] - moved.c).norm_squared())
}

/// Per part of a group: its class in the group, whether that class's products are its own, and its slack.
struct Classes {
    class: Vec<usize>,
    own: Vec<bool>,
    slack: Vec<f64>,
}

impl Group {
    /// Group of parts g of one key (two or more), sorted here by class; with `share` (a tolerance relative to tr S) a
    /// part whose products are each within it of one of an earlier class's shares that class, its class, ownership,
    /// and slack into `classes`.
    fn new(
        part: &Part,
        g: &mut [usize],
        slots: &Side,
        moved: &Side,
        share: Option<f64>,
        classes: &mut Classes,
    ) -> Self {
        // a part's traces exceed those of a class whose products are each within e of one of its own by at most e
        // (nuclear norm, at most sqrt 3 times Frobenius): it shares them, its cost lowered by its slack 2 e
        let width = part.mats.len();
        let tol = share.unwrap_or(0.0) * part.s.trace();
        let gap = |a: &M3, b: &[M3]| b.iter().map(|b| 3f64.sqrt() * (a - b).norm()).fold(f64::INFINITY, f64::min);
        // each class's products in turn, and after a 0 each class's count
        let mut runs = Vec::with_capacity(g.len() + 1);
        runs.push(0);
        let mut matrices: Vec<M3> = Vec::with_capacity(g.len() * width);
        for &p in g.iter() {
            let start = matrices.len();
            matrices.extend(part.mats.iter().map(|m| moved.r[p] * m));
            let (earlier, mine) = matrices.split_at(start);
            let near = |class: &[M3]| {
                let most = mine.iter().map(|m| gap(m, class)).fold(0.0, f64::max);
                (most <= tol).then_some(most)
            };
            let found = match (share, part.axis) {
                (Some(_), None) => earlier.chunks(width).enumerate().find_map(|(c, class)| near(class).map(|e| (c, e))),
                _ => None,
            };
            if let Some((c, e)) = found {
                matrices.truncate(start);
                classes.class[p] = c;
                classes.slack[p] = 2.0 * e;
                classes.own[p] = false;
                runs[c + 1] += 1;
            } else {
                classes.class[p] = runs.len() - 1;
                runs.push(1);
            }
        }
        // the parts by class, so that every class is a run, and where each run starts and the end
        g.sort_by_key(|&p| classes.class[p]);
        for c in 1..runs.len() {
            runs[c] += runs[c - 1];
        }
        let mut places = Vec::with_capacity(3 * g.len());
        for d in 0..3 {
            places.extend(g.iter().map(|&i| moved.x[i][d] - moved.c[d]));
        }
        let stride = matrices.len();
        let mut products = vec![0.0; 9 * stride];
        for (cm, product) in matrices.iter().enumerate() {
            for (e, v) in product.iter().enumerate() {
                products[e * stride + cm] = *v;
            }
        }
        let mut fixed = 0.0;
        let mut floor = 0.0;
        if share.is_some() {
            let a = slots.distances(g);
            let b = moved.distances(g);
            floor = part.mass * a.iter().zip(&b).map(|(a, b)| (a - b) * (a - b)).sum::<f64>();
            fixed = g.iter().map(|&i| offsets(part, slots, moved, i) - classes.slack[i]).sum();
        }
        Group { parts: g.to_vec(), runs, products, places, fixed, floor }
    }
}

impl Singles {
    /// Add part i, of a key of its own.
    fn add(&mut self, part: &Part, i: usize, slots: &Side, moved: &Side) {
        self.fixed += offsets(part, slots, moved, i) + 2.0 * part.s.trace();
        self.linear += (slots.x[i] - slots.c) * (moved.x[i] - moved.c).transpose() * part.mass;
        let mats: Vec<M3> = part.mats.iter().map(|m| slots.r[i] * m.transpose() * moved.r[i].transpose()).collect();
        if part.axis.is_none() && mats.len() == 1 {
            self.linear += mats[0];
        } else {
            self.others.push((i, mats));
        }
    }
}

/// Moved parts: per part its position, per slot the rotation of the part it holds and that part; their rotation from
/// the input, and per group the assignment's prices.
#[derive(Clone, Default)]
struct State {
    x: Vec<V3>,
    q: Vec<Q>,
    rotation: Q,
    original: Vec<usize>,
    prices: Vec<Vec<f64>>,
}

impl State {
    /// Copy `from` into this state's buffers.
    fn copy(&mut self, from: &Self) {
        self.x.clone_from(&from.x);
        self.q.clone_from(&from.q);
        self.rotation = from.rotation;
        self.original.clone_from(&from.original);
        self.prices.clone_from(&from.prices);
    }
}

/// Parts at (x1, q1) to move onto slots at (x0, q0) by a permutation of interchangeable parts (equal keys), each
/// part's symmetries, and a rigid motion, at the cost sum m |x0 - x|^2 + tr((R - R0) S (R - R0)^T).
struct Problem<'a> {
    parts: &'a [&'a Part],
    keys: &'a [usize],
    slots: Side<'a>,
    moved: Side<'a>,
    order: Vec<usize>,  // the parts by key, each group's by class
    spread: Vec<M3>,    // per slot S R0^T of its part
    groups: Vec<Group>, // the keys of two or more parts
    plan: Vec<usize>,   // the groups cheapest first
    class: Vec<usize>,  // per part, its class in its group
    own: Vec<bool>,     // per part, whether its class's products are its own
    deficit: f64,       // the parts' slack: the least cost less `value` is at most this
    singles: Singles,
    work: RefCell<Work>,
}

impl<'a> Problem<'a> {
    /// Problem of `parts` with `keys`; with `share` (a tolerance relative to tr S) it can be evaluated at rotations
    /// (`value`), the parts of a group whose products are each within it of one of a class's sharing the class's
    /// traces.
    fn new(parts: &'a [&'a Part], keys: &'a [usize], slots: Side<'a>, moved: Side<'a>, share: Option<f64>) -> Self {
        let k = parts.len();
        let mut order: Vec<usize> = (0..k).collect();
        order.sort_by_key(|&i| (keys[i], i));
        let mut classes = Classes { class: vec![0; k], own: vec![true; k], slack: vec![0.0; k] };
        let mut groups = Vec::new();
        let mut singles = Singles::default();
        for g in order.chunk_by_mut(|&a, &b| keys[a] == keys[b]) {
            let part = parts[g[0]];
            if g.len() > 1 {
                groups.push(Group::new(part, g, &slots, &moved, share, &mut classes));
            } else if share.is_some() {
                singles.add(part, g[0], &slots, &moved);
            }
        }
        let mut plan: Vec<usize> = (0..groups.len()).collect();
        plan.sort_by_key(|&i| groups[i].parts.len() * groups[i].products.len());
        let spread = parts.iter().zip(&slots.r).map(|(p, r)| p.s * r.transpose()).collect();
        let Classes { class, own, slack } = classes;
        let deficit = slack.iter().sum();
        let work = RefCell::default();
        Problem { parts, keys, slots, moved, order, spread, groups, plan, class, own, deficit, singles, work }
    }

    /// Parts of each key.
    fn identical(&self) -> impl Iterator<Item = &[usize]> + '_ {
        self.order.chunk_by(|&a, &b| self.keys[a] == self.keys[b])
    }

    /// Rotation of the parts' weighted principal axes onto the slots' under the proper sign flip (of 4) least by the
    /// mass-weighted squared distance of each rotated part to the nearest slot of its key, the first of equal ones.
    fn flip(&self) -> Q {
        let (s0, s1) = (&self.slots, &self.moved);
        let (e0, e1) = (s0.axes(self.parts), s1.axes(self.parts));
        let score = |r: &M3| {
            let mut total = 0.0;
            for g in self.identical() {
                for &i in g {
                    let moved = r * (s1.x[i] - s1.c) + s0.c;
                    let nearest = g.iter().map(|&j| (moved - s0.x[j]).norm_squared()).fold(f64::INFINITY, f64::min);
                    total += self.parts[i].mass * nearest;
                }
            }
            total
        };
        let flips = [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]].map(|flip| {
            let mut r = e0 * M3::from_diagonal(&V3::from(flip)) * e1.transpose();
            if r.determinant() < 0.0 {
                r = e0 * M3::from_diagonal(&V3::new(flip[0], flip[1], -flip[2])) * e1.transpose();
            }
            (r, score(&r))
        });
        let (r, _) = flips.into_iter().min_by(|a, b| a.1.total_cmp(&b.1)).expect("four flips");
        quat(&r)
    }

    /// Start `state`: the parts in input order, rotated by r about their centroid onto the slots' centroid.
    fn start(&self, r: Q, state: &mut State) {
        state.x.clear();
        state.x.extend_from_slice(self.moved.x);
        state.q.clear();
        state.q.extend_from_slice(self.moved.q);
        state.rotation = r;
        state.original.clear();
        state.original.extend(0..self.parts.len());
        state.prices.resize(self.groups.len(), Vec::new());
        state.prices.iter_mut().for_each(Vec::clear);
        self.place(&r, &self.moved.c, state);
    }

    /// Rotate the parts by r about their centroid c onto the slots' centroid.
    fn place(&self, r: &Q, c: &V3, state: &mut State) {
        for (x, q) in state.x.iter_mut().zip(&mut state.q) {
            *x = r * (*x - c) + self.slots.c;
            *q = r * *q;
        }
    }

    /// Cost of matching a group's slots s to its parts p (n x n, row-major) less its `fixed` (the same least
    /// assignment, being constant per row and per column): -2 m (x0_s - c0) . R (x1_p - c1) + 2 (tr S - max_g
    /// tr(R0_s^T R R1_p g S)), R^T = `turned`. The traces as <R^T R0_s, R1_p M> over the group's products, all classes
    /// at once, per slot and class the best matrix into `best`; at warm prices (n of them), per row its cheapest column
    /// and least reduced cost into `least`.
    fn costs(&self, group: &Group, turned: &M3, work: &mut Work, price: &[f64]) {
        let n = group.parts.len();
        let part = self.parts[group.parts[0]];
        let width = part.mats.len();
        let trace = part.s.trace();
        let stride = group.products.len() / 9;
        let k = group.runs.len() - 1;
        let rows: [&[f64]; 9] = std::array::from_fn(|e| &group.products[e * stride..][..stride]);
        let (y0, y1, y2) = (&group.places[..n], &group.places[n..2 * n], &group.places[2 * n..]);
        // every entry is written
        work.lap.cost.resize(n * n, 0.0);
        work.best.resize(n * k, 0);
        work.traces.resize(stride, 0.0);
        work.rotations.resize(n, 0.0);
        work.lap.least.clear();
        for (s, &slot) in group.parts.iter().enumerate() {
            let w = turned * self.slots.r[slot];
            for (c, t) in work.traces.iter_mut().enumerate() {
                *t = (0..9).map(|e| w[e] * rows[e][c]).sum();
            }
            let best = &mut work.best[s * k..][..k];
            for ((b, t), run) in best.iter_mut().zip(work.traces.chunks(width)).zip(group.runs.windows(2)) {
                let (m, most) = part.pick(|m| t[m]);
                *b = m;
                work.rotations[run[0]..run[1]].fill(2.0 * (trace - most));
            }
            let a = turned * (self.slots.x[slot] - self.slots.c) * (-2.0 * part.mass);
            let row = &mut work.lap.cost[s * n..][..n];
            for ((((c, y0), y1), y2), r) in row.iter_mut().zip(y0).zip(y1).zip(y2).zip(&work.rotations) {
                *c = a[0] * y0 + a[1] * y1 + a[2] * y2 + r;
            }
            if price.len() == n {
                let (mut j, mut least) = (0, f64::INFINITY);
                for (p, (&c, &v)) in row.iter().zip(price).enumerate() {
                    if c - v < least {
                        (j, least) = (p, c - v);
                    }
                }
                work.lap.least.push((j, least));
            }
        }
    }

    /// One round: `matching`, then rotate the parts by the best global rotation, maximizing tr(R H), H = sum m a b^T +
    /// R S R0^T (a, b: centered positions).
    fn round(&self, state: &mut State) {
        self.matching(state);
        let ca = centroid(self.parts, state.original.iter().map(|&p| &state.x[p]));
        let mut h = M3::zeros();
        for (i, part) in self.parts.iter().enumerate() {
            let x = state.x[state.original[i]];
            let r = state.q[i].to_rotation_matrix().into_inner();
            h += (x - ca) * (self.slots.x[i] - self.slots.c).transpose() * part.mass + r * self.spread[i];
        }
        let r = quat(&fit(&h));
        self.place(&r, &ca, state);
        state.rotation = r * state.rotation;
    }

    /// Match each group's slots to its parts at the parts' rotation, and turn every part toward its slot's rotation by
    /// its best symmetry.
    fn matching(&self, state: &mut State) {
        let work = &mut *self.work.borrow_mut();
        let turned = state.rotation.to_rotation_matrix().into_inner().transpose();
        reset(&mut work.known, self.parts.len(), None);
        for (group, price) in self.groups.iter().zip(&mut state.prices) {
            let n = group.parts.len();
            self.costs(group, &turned, work, price);
            work.lap.assign(n, price, f64::INFINITY);
            let k = work.best.len() / n;
            for (s, &p) in work.lap.col.iter().enumerate() {
                let (slot, part) = (group.parts[s], group.parts[p]);
                state.original[slot] = part;
                work.known[slot] = self.own[part].then_some(work.best[s * k + self.class[part]]);
            }
        }
        for (i, part) in self.parts.iter().enumerate() {
            let q = state.rotation * self.moved.q[state.original[i]];
            state.q[i] = part.turn(q, &self.slots.r[i], &self.slots.q[i], work.known[i]);
        }
    }

    /// Least cost at rotation r over the matchings and symmetries: the parts of keys of their own, then per group in
    /// plan order its least assignment, with column prices from and into `prices` (empty: none). A group of warm
    /// prices counts its dual bound at them first and its assignment after all groups. Once the cost so far plus the
    /// bounds of the groups left reaches `cap`, that sum.
    fn value(&self, r: Q, prices: &mut [Vec<f64>], cap: f64) -> f64 {
        let work = &mut *self.work.borrow_mut();
        let rotation = r.to_rotation_matrix().into_inner();
        let turned = rotation.transpose();
        let singles = &self.singles;
        let mut total = singles.fixed - 2.0 * rotation.dot(&singles.linear);
        for (i, mats) in &singles.others {
            total -= 2.0 * self.parts[*i].pick(|m| rotation.dot(&mats[m])).1;
        }
        let mut rest: f64 = self.plan.iter().map(|&index| self.groups[index].floor).sum();
        work.deferred.clear();
        for &index in &self.plan {
            if total + rest >= cap {
                return total + rest;
            }
            let (group, price) = (&self.groups[index], &mut prices[index]);
            let n = group.parts.len();
            rest -= group.floor;
            self.costs(group, &turned, work, price);
            total += group.fixed;
            let c = &work.lap.cost;
            if n == 2 {
                total += (c[0] + c[3]).min(c[1] + c[2]);
            } else if price.len() == n {
                let dual = work.lap.least.iter().map(|l| l.1).sum::<f64>() + price.iter().sum::<f64>();
                total += dual;
                let j = work.deferred.len();
                if work.kept.len() == j {
                    work.kept.push(Lap::default());
                }
                std::mem::swap(&mut work.lap, &mut work.kept[j]);
                work.deferred.push((index, dual));
            } else {
                total += work.lap.assign(n, price, cap - total - rest);
            }
        }
        for (j, &(index, dual)) in work.deferred.iter().enumerate() {
            if total >= cap {
                break;
            }
            total += work.kept[j].assign(self.groups[index].parts.len(), &mut prices[index], f64::INFINITY) - dual;
        }
        total
    }

    /// Training coupling: two rounds from rotation r.
    fn couple(&self, r: Q) -> State {
        let mut state = State::default();
        self.start(r, &mut state);
        for _ in 0..2 {
            self.round(&mut state);
        }
        state
    }

    /// Per slot the squared displacements (translation, rotation) of a state, by explicit squares.
    fn split<'s>(&'s self, state: &'s State) -> impl Iterator<Item = [f64; 2]> + 's {
        let s0 = &self.slots;
        (0..self.parts.len()).map(|i| self.parts[i].cost(&s0.x[i], &s0.r[i], &state.x[state.original[i]], &state.q[i]))
    }

    /// Cost of a state.
    fn cost(&self, state: &State) -> f64 {
        self.split(state).map(|[t, r]| t + r).sum()
    }

    /// State refined in place, and its cost: rounds, each followed by turning every part toward its slot, until the
    /// cost rises (the state before kept, from the buffer `before`), falls by at most 1e-10 relative, or 1000 rounds.
    fn refine(&self, state: &mut State, before: &mut State) -> f64 {
        let mut cost = self.cost(state);
        for _ in 0..1000 {
            before.copy(state);
            self.round(state);
            for i in 0..self.parts.len() {
                state.q[i] = self.parts[i].turn(state.q[i], &self.slots.r[i], &self.slots.q[i], None);
            }
            let next = self.cost(state);
            if next > cost {
                std::mem::swap(state, before);
                return cost;
            }
            let converged = cost - next <= 1e-10 * cost.max(1.0);
            cost = next;
            if converged {
                break;
            }
        }
        cost
    }

    /// Constants of the bounds over rotations: the curvature K, the least of 2 c and 2 sqrt(lambda_max(I_A)
    /// (lambda_max(I_B) + eta)) (I_A and I_B the inertia tensors of slots and parts about their centroids, eta the
    /// parts' symmetry slack): from a local minimum the least cost rises by at most K (1 - cos angle); C, the sum over
    /// parts of m (|a|^2 + |b|^2) + 2 tr S; c, per key m times the sorted pairing of |a| and |b| plus n tr S (a, b:
    /// centered positions of slots and parts); and I_A and I_B + eta I.
    fn constants(&self) -> certify::Constants {
        let eta: f64 = self.parts.iter().map(|p| p.eta).sum();
        let slots = self.slots.inertia(self.parts);
        let parts = self.moved.inertia(self.parts) + M3::identity() * eta;
        let largest = |m: &M3| SymmetricEigen::new(*m).eigenvalues.max();
        let inertial = 2.0 * (largest(&slots) * largest(&parts)).sqrt();
        let mut total = 0.0;
        let mut pairing = 0.0;
        for g in self.identical() {
            let part = self.parts[g[0]];
            let (a, b) = (self.slots.distances(g), self.moved.distances(g));
            let n = g.len() as f64;
            total += part.mass * a.iter().chain(&b).map(|v| v * v).sum::<f64>() + 2.0 * n * part.s.trace();
            pairing += part.mass * a.iter().zip(&b).map(|(a, b)| a * b).sum::<f64>() + n * part.s.trace();
        }
        certify::Constants { curvature: inertial.min(2.0 * pairing), total, pairing, slots, parts }
    }
}

/// Mass movement: its cost, a lower bound on the least cost, whether the search certified it (cost - bound <= gap
/// max(cost, total weight)), per slot the squared displacements (translation, rotation) and the part moved onto it,
/// and the moved parts' rigid motion x -> R x + t.
pub struct Movement {
    pub cost: f64,
    pub bound: f64,
    pub certified: bool,
    pub trans2: Vec<f64>,
    pub rot2: Vec<f64>,
    pub order: Vec<usize>,
    pub rotation: Q,
    pub translation: V3,
}

/// Objective of the certified search: f by `value` of the first problem (traces shared within a tolerance), or of the
/// second (shared only when equal) once the first's deficit could reach a tenth of the gap (`gap` times the least cost
/// found, at least `floor`); each group's assignment warm-started from the last evaluation's prices; at a rotation
/// below the least cost found, the movement matched there and refined.
struct Evaluator<'p, 'a> {
    problems: [&'p Problem<'a>; 2],
    prices: [Vec<Vec<f64>>; 2],
    gap: f64,
    floor: f64,
    state: State,
    before: State,
    best: (f64, State),
}

impl certify::Objective for Evaluator<'_, '_> {
    fn value(&mut self, r: &Q, cap: f64) -> f64 {
        let k = usize::from(10.0 * self.problems[0].deficit > (self.gap * self.best.0).max(self.floor));
        let problem = self.problems[k];
        let value = problem.value(*r, &mut self.prices[k], cap);
        if value < self.best.0 {
            problem.start(*r, &mut self.state);
            problem.matching(&mut self.state);
            let cost = problem.refine(&mut self.state, &mut self.before);
            if cost < self.best.0 {
                self.best = (cost, self.state.clone());
            }
        }
        value
    }

    fn upper(&self) -> f64 {
        self.best.0
    }
}

/// Least mass movement of parts at (x1, q1) onto slots at (x0, q0), parts of equal keys interchangeable: the
/// branch and bound over rotations about the training coupling's rotation to a relative `gap` (at least `gap` times the
/// total weight), the least cost found matched again at its own rotation.
pub fn movement(parts: &[&Part], keys: &[usize], x0: &[V3], q0: &[Q], x1: &[V3], q1: &[Q], gap: f64) -> Movement {
    let (slots, moved) = (Side::new(parts, x0, q0), Side::new(parts, x1, q1));
    let problem = Problem::new(parts, keys, slots.clone(), moved.clone(), Some(0.0));
    let coupled = problem.couple(problem.flip());
    let incumbent = coupled.rotation;
    // traces shared within a thirtieth of the gap (at most 1e-5) of tr S
    let shared = Problem::new(parts, keys, slots, moved, Some((gap / 30.0).min(1e-5)));
    let floor = gap * parts.iter().map(|p| p.mass).sum::<f64>();
    let groups = problem.groups.len();
    let mut evaluator = Evaluator {
        problems: [&shared, &problem],
        prices: [vec![Vec::new(); groups], vec![Vec::new(); groups]],
        gap,
        floor,
        state: State::default(),
        before: State::default(),
        best: (problem.cost(&coupled), coupled),
    };
    let (bound, certified) = certify::search(&mut evaluator, &problem.constants(), gap, floor, &incumbent);
    let mut state = evaluator.best.1;
    problem.matching(&mut state);
    let cost = problem.cost(&state);
    let (trans2, rot2) = problem.split(&state).map(|[t, r]| (t, r)).unzip();
    let State { original: order, rotation, .. } = state;
    let translation = problem.slots.c - rotation * problem.moved.c;
    Movement { cost, bound, certified, trans2, rot2, order, rotation, translation }
}

/// Directory of trees and the coding of its atlas.
struct Trees {
    set: Records,
    coding: Coding,
}

impl Trees {
    /// Part ids and poses of the first k parts of assembly `index`.
    fn read(&self, index: usize, k: usize) -> Result<(Vec<usize>, Vec<M4>)> {
        let (ids, poses) = self.coding.poses(&self.set, index, k)?;
        Ok((ids.iter().map(|&id| id as usize).collect(), poses.iter().map(|p| M4::from_row_slice(p)).collect()))
    }
}

/// Training pairs: the trees, parts by part id, seed, coordinate scale, noise std per part count, and time.
pub struct Pairs {
    trees: Option<Trees>,
    parts: Vec<Option<Part>>,
    seed: u64,
    scale: f64,
    noise: Vec<f64>,
    time: (f64, f64),
}

impl Pairs {
    /// Pairs of the trees in a directory (root, assets, the part id of each class) if given, else of given parts only.
    pub fn new(
        trees: Option<(&Path, &Path, Vec<i32>)>,
        parts: Vec<Option<Part>>,
        seed: u64,
        scale: f64,
        noise: Vec<f64>,
        time: (f64, f64),
    ) -> Result<Self> {
        let open = |(root, assets, representatives)| -> Result<Trees> {
            Ok(Trees { set: Records::open(root)?, coding: Coding::open(assets, representatives)? })
        };
        Ok(Pairs { trees: trees.map(open).transpose()?, parts, seed, scale, noise, time })
    }

    /// Given parts (part ids `ty`) coupled to their noise at stream position g; an error for a part not in the table
    /// or without a noise std.
    pub fn sample(&self, g: u64, ty: &[usize], keys: &[usize], poses: &[M4], time: Option<f64>) -> Result<Sample> {
        let k = ty.len();
        let sigma = *self.noise.get(k).ok_or_else(|| bad(&format!("no noise std for {k} parts")))?;
        let part = |&t: &usize| self.parts.get(t).and_then(Option::as_ref).ok_or_else(|| bad(&format!("no part {t}")));
        let parts: Vec<&Part> = ty.iter().map(part).collect::<Result<_>>()?;
        let (x1, q1) = frames(&parts, poses, self.scale);
        let mut rng = rng(self.seed, g, 1);
        let mut normal = || -> f64 { rng.sample(StandardNormal) };
        let drawn = 1.0 / (1.0 + (-(self.time.0 + self.time.1 * normal())).exp()); // drawn even when given
        let t = time.unwrap_or(drawn);
        let x0: Vec<V3> = (0..k).map(|_| V3::from_fn(|_, _| normal() * sigma)).collect();
        let q0: Vec<Q> = (0..k).map(|_| haar(&mut normal)).collect();
        let c0 = centroid(&parts, &x0);
        let x0: Vec<V3> = x0.iter().map(|p| p - c0).collect();
        let problem = Problem::new(&parts, keys, Side::new(&parts, &x0, &q0), Side::new(&parts, &x1, &q1), None);
        let State { x, q, original: order, .. } = problem.couple(problem.flip());
        let token = |i: usize| {
            let rel = toward(q0[i].inverse() * q[i], &Q::identity());
            let quat = q0[i] * rel.powf(t);
            let pos = x0[i] * (1.0 - t) + x[order[i]] * t;
            let vel = x[order[i]] - x0[i];
            let tangent = rel.scaled_axis() * (1.0 - t);
            let values = [
                pos[0], pos[1], pos[2], quat.w, quat.i, quat.j, quat.k, vel[0], vel[1], vel[2], tangent[0], tangent[1],
                tangent[2],
            ];
            values.map(|v| v as f32)
        };
        Ok(Sample { t, out: (0..k).map(token).collect(), order })
    }

    /// Stream position g -> index in [0, span) by a random affine bijection per pass: (a (g mod span) + b) mod span
    /// with a coprime to span.
    pub fn shuffled(&self, g: u64, span: u64) -> u64 {
        let mut shuffle = rng(self.seed, g / span, 3);
        let gcd = |mut a: u64, mut b: u64| {
            while b != 0 {
                (a, b) = (b, a % b);
            }
            a
        };
        let a = loop {
            let a = shuffle.random_range(1..span.max(2));
            if gcd(a, span) == 1 {
                break a;
            }
        };
        let b = shuffle.random_range(0..span);
        ((u128::from(a) * u128::from(g % span) + u128::from(b)) % u128::from(span)) as u64
    }

    /// k-part prefix of the assembly at stream position g (shuffled per pass): its types and sample.
    fn prefix(&self, g: u64, k: usize, time: Option<f64>) -> Result<(Vec<usize>, Sample)> {
        let trees = self.trees.as_ref().ok_or_else(|| bad("pairs without trees"))?;
        let (ty, poses) = trees.read(self.shuffled(g, trees.set.count as u64) as usize, k)?;
        let s = self.sample(g, &ty, &ty, &poses, time)?;
        Ok((s.order.iter().map(|&i| ty[i]).collect(), s))
    }

    /// One row: k ~ U{2..100} parts at position ga, the complement at gb with ga's time (as float32).
    pub fn row(&self, ga: u64, gb: u64) -> Result<Row> {
        let k = rng(self.seed, ga, 2).random_range(2..=PARTS);
        let (ta, a) = self.prefix(ga, k, None)?;
        let (tb, b) = self.prefix(gb, ROW - k, Some(f64::from(a.t as f32)))?;
        let all: Vec<usize> = [ta, tb].concat();
        let types = std::array::from_fn(|i| all[i] as i64);
        Ok(Row { types, lengths: [k as i64, (ROW - k) as i64], t: a.t as f32, tokens: [a.out, b.out].concat() })
    }
}
