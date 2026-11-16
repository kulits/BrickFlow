//! Certified mass movement by rotation bounds.

// The least cost is f* = min over rotations Q of f(Q), f the least cost over matchings and symmetries at Q. At a
// global minimizer the active choice (matching, symmetries, twists) has a local minimum, from which its cost rises
// along each geodesic by 2 (1 - cos s) u^T N u, N = tr H - H for the cross-covariance H of its vector pairs (part
// centers and principal axes). The largest eigenvalue of N is at most K / 2 (the inertia bound), at most 2 tr H, and
// at most (tr H + c) / 2, where tr H = (C - f*) / 2; and per direction 4 e^T N e <= 4 sqrt(e^T I_A e e^T I_B e),
// e = sin(s / 2) u the vector part of the move's unit quaternion and I_A, I_B the inertia tensors of the slots and of
// the parts at the cell's center (plus their symmetry slack). So a cell of SO(3) holding a global minimizer has f* at
// least the elimination bound of f at its center: cells bounded above the least cost found (less the gap) hold none.
// Cells are boxes on the 4 gnomonic faces of the quaternion cube in the slots' principal frame about the training
// coupling's rotation, least bound first, cut in three across the axis of most growth (the middle box keeps its
// center's value); the least bound of the cells left is a lower bound on f*, and 0 is one. The search certifies the
// least cost found within the gap of f* unless it runs out of evaluations first.

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use nalgebra::{Quaternion, SymmetricEigen, Vector4};

use crate::geometry::{ordered, quat, M3, Q, V3};

/// Cost over rotations.
pub trait Objective {
    /// f at rotation q, or a lower bound on it of at least `cap` once f is known to reach `cap`.
    fn value(&mut self, q: &Q, cap: f64) -> f64;
    /// Least cost found.
    fn upper(&self) -> f64;
}

/// Constants of the growth bounds: the curvature K, the total C of the vector pairs' squared lengths, their sorted
/// pairing c (at least the nuclear norm of H), and the inertia tensors of the slots and of the parts (in their own
/// frame, plus the symmetry slack).
pub struct Constants {
    pub curvature: f64,
    pub total: f64,
    pub pairing: f64,
    pub slots: M3,
    pub parts: M3,
}

const ONE: i64 = (1 << 36) * 531_441; // gnomonic coordinates in units of 1 / (2^36 3^12)
const ROOT: i64 = 4; // the root boxes: 4^3 per face

const CORNERS: [[i64; 3]; 8] =
    [[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1], [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]];

/// Box on a gnomonic face: the face, center, and half sides in units of 1 / ONE.
type Region = (usize, [i64; 3], [i64; 3]);

/// Bounded box: its region, f at its center, its bound, and the stiffness of each axis at its center.
struct Cell {
    region: Region,
    value: f64,
    lower: f64,
    stiffness: [f64; 3],
}

/// Gnomonic charts of the 4 faces of the quaternion cube in a frame E about an origin q0: the point u of face k is
/// E p E^-1 q0, p the unit quaternion along 1 at coordinate k and u at the others (`raw`); its Gibbs vector in E's axes
/// has coordinate axes[k][i].0 equal to axes[k][i].1 u[i].
struct Chart {
    frame: Q,
    right: Q,
    axes: [[(usize, f64); 3]; 4],
}

impl Chart {
    fn new(frame: Q, origin: Q) -> Self {
        let basis = |i: usize| Q::new_unchecked(Quaternion::from(Vector4::ith(i, 1.0)));
        let axes = std::array::from_fn(|face| {
            let others: Vec<usize> = (0..4).filter(|&i| i != face).collect();
            std::array::from_fn(|i| {
                let w = (basis(others[i]) * basis(face).inverse()).imag();
                let j = w.iamax();
                (j, w[j].signum())
            })
        });
        Chart { frame, right: frame.inverse() * origin, axes }
    }

    /// Unit quaternion (w, x, y, z) along 1 at coordinate `face` and u at the others.
    fn raw(face: usize, u: [i64; 3]) -> Vector4<f64> {
        let at = |i: usize| match i.cmp(&face) {
            Ordering::Less => u[i],
            Ordering::Equal => ONE,
            Ordering::Greater => u[i - 1],
        };
        Vector4::from_fn(|i, _| at(i) as f64).normalize()
    }

    /// Gibbs vector (frame axes) of coordinates u of a face.
    fn gibbs(&self, face: usize, u: [i64; 3]) -> V3 {
        let mut w = V3::zeros();
        for (i, &(j, sign)) in self.axes[face].iter().enumerate() {
            w[j] = sign * u[i] as f64 / ONE as f64;
        }
        w
    }
}

/// Least f* for which f* rises to `value` within angle r (cos(r / 2) = `cosine`), by the K, 2 tr H, and
/// (tr H + c) / 2 bounds; and the least value whose bound reaches `limit`.
fn eliminate(value: f64, cosine: f64, constants: &Constants, limit: f64) -> (f64, f64) {
    let b = 2.0 * (1.0 - cosine * cosine); // 1 - cos r
    let Constants { curvature, total, pairing, .. } = *constants;
    // f <= f* + K b, f <= f* + b ((C - f*) / 2 + c), f <= f* + 2 b (C - f*)
    let mut lower = (value - curvature * b).max((value - b * (total / 2.0 + pairing)) / (1.0 - b / 2.0));
    let mut cap = (limit + curvature * b).min(limit * (1.0 - b / 2.0) + b * (total / 2.0 + pairing));
    if 2.0 * b < 1.0 {
        lower = lower.max((value - 2.0 * b * total) / (1.0 - 2.0 * b));
        cap = cap.min(limit * (1.0 - 2.0 * b) + 2.0 * b * total);
    }
    (lower, cap)
}

/// Constants and chart of a search, and the slots' inertia tensor in the chart's frame.
struct Search<'c> {
    constants: &'c Constants,
    chart: Chart,
    slots: M3,
}

impl Search<'_> {
    /// Bounded box: f at its center (`known`, else evaluated and stopped once bounded out) through `eliminate`, r its
    /// largest angle from the center (at a corner), and less the growth 4 sqrt(e^T I_A e e^T I_B e) over its moves to
    /// the center: from b (Gibbs vector) to the center a the move's quaternion is (1 + a . b, L d) over
    /// sqrt((1 + |a|^2) (1 + |b|^2)) (d = a - b), the quadratic forms L^T I L in d (chart frame, L = I + [a]x) largest
    /// at corners; each axis's stiffness is sqrt(I_A I_B) on it.
    fn bound(&self, objective: &mut impl Objective, limit: f64, region: Region, known: f64) -> Cell {
        let (face, center, half) = region;
        let c = Chart::raw(face, center);
        let corner = |k: &[i64; 3]| Chart::raw(face, [0, 1, 2].map(|i| center[i] + half[i] * k[i])).dot(&c);
        let cosine = CORNERS.iter().map(corner).fold(1.0, f64::min);
        let local = Q::new_unchecked(Quaternion::from(c)) * self.chart.right;
        let q = self.chart.frame * local;
        let (a, h) = (self.chart.gibbs(face, center), self.chart.gibbs(face, half).abs());
        let gibbs = 1.0 + (0..3).map(|j| (a[j] * (a[j] - h[j])).min(a[j] * (a[j] + h[j]))).sum::<f64>();
        let nearest = (0..3).map(|j| (a[j].abs() - h[j]).max(0.0).powi(2)).sum::<f64>();
        let r = local.to_rotation_matrix().into_inner();
        let l = M3::identity() + a.cross_matrix();
        let slots = l.transpose() * self.slots * l;
        let parts = l.transpose() * (r * self.constants.parts * r.transpose()) * l;
        let stiffness = self.chart.axes[face].map(|(j, _)| (slots[(j, j)] * parts[(j, j)]).sqrt());
        let largest = |x: M3| {
            CORNERS.iter().map(|k| V3::from_fn(|i, _| h[i] * k[i] as f64)).map(|d| d.dot(&(x * d))).fold(0.0, f64::max)
        };
        let growth = 4.0 * (largest(slots) * largest(parts)).sqrt()
            / (gibbs * gibbs).max((1.0 + a.norm_squared()) * (1.0 + nearest));
        let value = if known.is_nan() {
            let cap = eliminate(0.0, cosine, self.constants, limit).1.min(limit + growth);
            objective.value(&q, cap.max(objective.upper()))
        } else {
            known
        };
        let lower = eliminate(value, cosine, self.constants, limit).0.max(value - growth);
        Cell { region, value, lower, stiffness }
    }

    /// Children of a box: cut across the axis of most growth (largest h^2 sqrt(I_A I_B) on it, at the center),
    /// in three while its half side there divides by 3 (the middle child keeps the center and its value), else in two.
    fn split(cell: &Cell) -> impl Iterator<Item = (Region, f64)> + '_ {
        let (face, center, half) = cell.region;
        let axis = V3::from_fn(|i, _| (half[i] as f64).powi(2) * cell.stiffness[i]).imax();
        let pieces = if half[axis] % 3 == 0 { 3 } else { 2 };
        let step = half[axis] / pieces;
        (0..pieces).map(move |k| {
            let (mut c, mut h) = (center, half);
            c[axis] += step * (2 * k - (pieces - 1));
            h[axis] = step;
            let known = if 2 * k == pieces - 1 { cell.value } else { f64::NAN };
            ((face, c, h), known)
        })
    }
}

/// Evaluations after which the search stops, its bound then the least of the cells left.
const BUDGET: usize = 1_000_000;

/// Search to a gap between the least cost found and the least bound of a cell left of `gap` relative (at least
/// `floor`), or until BUDGET evaluations, the chart about rotation `incumbent`: the lower bound on f*, and whether the
/// gap was reached (false: the budget ran out first).
pub fn search(
    objective: &mut impl Objective,
    constants: &Constants,
    gap: f64,
    floor: f64,
    incumbent: &Q,
) -> (f64, bool) {
    let threshold = |upper: f64| if upper.is_finite() { upper - (gap * upper).max(floor) } else { f64::INFINITY };
    // the slots' principal frame (a rotation)
    let mut frame = SymmetricEigen::new(constants.slots).eigenvectors;
    if frame.determinant() < 0.0 {
        frame.column_mut(0).neg_mut();
    }
    let slots = frame.transpose() * constants.slots * frame;
    let search = Search { constants, chart: Chart::new(quat(&frame), *incumbent), slots };
    let half = ONE / ROOT;
    let mut bounded = Vec::new();
    let mut evaluations = 0;
    for face in 0..4 {
        for i in 0..ROOT * ROOT * ROOT {
            let limit = threshold(objective.upper());
            if limit <= 0.0 {
                break;
            }
            let at = |j: i64| -ONE + half * (2 * j + 1);
            let center = [at(i / (ROOT * ROOT)), at(i / ROOT % ROOT), at(i % ROOT)];
            bounded.push(search.bound(objective, limit, (face, center, [half; 3]), f64::NAN));
            evaluations += 1;
        }
    }
    // cells by index, the least bound first out of the max-heap
    let mut cells = Vec::new();
    let mut heap = BinaryHeap::new();
    let mut pruned = f64::INFINITY;
    let mut certified = true;
    loop {
        let limit = threshold(objective.upper());
        for cell in bounded.drain(..) {
            if cell.lower >= limit {
                pruned = pruned.min(cell.lower);
            } else {
                heap.push((Reverse(ordered(cell.lower)), cells.len()));
                cells.push(cell);
            }
        }
        // f >= 0 certifies a least cost found within the gap of 0
        if limit <= 0.0 {
            return (0.0, true);
        }
        let Some((_, at)) = heap.pop() else { break };
        let cell = &cells[at];
        if cell.lower >= limit || evaluations >= BUDGET {
            certified = cell.lower >= limit;
            pruned = pruned.min(cell.lower);
            break;
        }
        for (child, known) in Search::split(cell) {
            let limit = threshold(objective.upper());
            bounded.push(search.bound(objective, limit, child, known));
            evaluations += usize::from(known.is_nan());
        }
    }
    let open = heap.peek().map_or(f64::INFINITY, |&(_, at)| cells[at].lower);
    (open.min(pruned).min(objective.upper()).max(0.0), certified)
}
