//! Rigid poses, rotations, and joint coordinates (LDU, degrees).

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use nalgebra::{Matrix3, Matrix4, Quaternion, Rotation3, SymmetricEigen, Unit, UnitQuaternion, Vector3, Vector4};

use crate::atlas::{Cell, Region, BALL, FIXED};

pub type M3 = Matrix3<f64>;
pub type M4 = Matrix4<f64>;
pub type V3 = Vector3<f64>;
pub type Q = UnitQuaternion<f64>;

const LEAF: usize = 8; // cells per leaf of a ball cell tree

/// Total order of f64 (`total_cmp`) as an integer, for heap keys.
pub fn ordered(x: f64) -> i64 {
    let bits = x.to_bits() as i64;
    bits ^ (((bits >> 63) as u64) >> 1) as i64
}

/// Rotation by `deg` degrees about coordinate axis `axis`.
pub fn rot(axis: usize, deg: f64) -> M3 {
    Rotation3::from_axis_angle(&Unit::new_unchecked(V3::ith(axis, 1.0)), deg.to_radians()).into_inner()
}

/// Pose [R | t].
pub fn pose(r: &M3, t: &V3) -> M4 {
    r.to_homogeneous().append_translation(t)
}

/// Rotation R of a pose [R | t].
pub fn rotation(p: &M4) -> M3 {
    p.fixed_view::<3, 3>(0, 0).into()
}

/// Translation t of a pose [R | t].
pub fn origin(p: &M4) -> V3 {
    p.fixed_view::<3, 1>(0, 3).into()
}

/// Unit quaternion of a rotation matrix.
pub fn quat(r: &M3) -> Q {
    Q::from_rotation_matrix(&Rotation3::from_matrix_unchecked(*r))
}

/// [R^T | -R^T t] of a rigid pose [R | t].
pub fn inverse(p: &M4) -> M4 {
    let rt = rotation(p).transpose();
    pose(&rt, &-(rt * origin(p)))
}

/// Maximum of tr(R H) over rotations R for each of N matrices: the largest root of the characteristic polynomial of
/// Horn's N(H) (see `fit`), by Halley's method from an upper bound, the N roots found in step.
pub fn largest<const N: usize>(h: [M3; N]) -> [f64; N] {
    // per matrix: |H|^2, det H, det N(H), and the bound
    let c = h.map(|h| {
        let h2 = h.norm_squared();
        let adj = [h.row(1).cross(&h.row(2)), h.row(2).cross(&h.row(0)), h.row(0).cross(&h.row(1))];
        let det_h = h[(0, 0)] * adj[0][0] + h[(0, 1)] * adj[0][1] + h[(0, 2)] * adj[0][2];
        let adj2 = adj.iter().map(|r| r.norm_squared()).sum::<f64>();
        [h2, det_h, h2 * h2 - 4.0 * adj2, (h2 + 2.0 * 3f64.sqrt() * adj2.sqrt()).sqrt()]
    });
    let mut l = c.map(|c| c[3]);
    let mut done = [false; N];
    for _ in 0..64 {
        for (i, [h2, det_h, det_n, _]) in c.into_iter().enumerate() {
            if done[i] {
                continue;
            }
            let p = ((l[i] * l[i] - 2.0 * h2) * l[i] - 8.0 * det_h) * l[i] + det_n;
            let dp = (4.0 * l[i] * l[i] - 4.0 * h2) * l[i] - 8.0 * det_h;
            let ddp = 12.0 * l[i] * l[i] - 4.0 * h2;
            let step = 2.0 * p * dp / (2.0 * dp * dp - p * ddp);
            // 0 / 0 at the root itself (H = 0)
            if step.is_finite() {
                l[i] -= step;
            }
            done[i] = !step.is_finite() || step.abs() <= 1e-6 * l[i];
        }
        if done == [true; N] {
            break;
        }
    }
    l
}

/// (x_12 - x_21, x_20 - x_02, x_01 - x_10): twice the axial vector of the skew part of x, negated.
pub fn vee(x: &M3) -> V3 {
    V3::new(x[(1, 2)] - x[(2, 1)], x[(2, 0)] - x[(0, 2)], x[(0, 1)] - x[(1, 0)])
}

/// Rotation maximizing tr(R H): the top eigenvector q = (w, x, y, z) of Horn's N(H), q^T N q = tr(R(q) H), as a
/// column of adj(N - l I) at the largest eigenvalue l (`largest`), or by eigendecomposition if l is nearly repeated.
pub fn fit(h: &M3) -> M3 {
    let [a, b, c] = vee(h).into();
    let [xy, xz, yz] = [h[(0, 1)] + h[(1, 0)], h[(0, 2)] + h[(2, 0)], h[(1, 2)] + h[(2, 1)]];
    let [x, y, z] = [h[(0, 0)], h[(1, 1)], h[(2, 2)]];
    let n = M4::new(x + y + z, a, b, c, a, x - y - z, xy, xz, b, xy, y - x - z, yz, c, xz, yz, z - x - y);
    let [l] = largest([*h]);
    // adj(N - l I) = q q^T times the product of (l_k - l) over the other eigenvalues l_k: its longest column is the one
    // with the largest diagonal entry
    let shifted = n - M4::from_diagonal_element(l);
    let cofactor = |i: usize, j: usize| {
        let minor = M3::from_fn(|r, c| shifted[(r + usize::from(r >= i), c + usize::from(c >= j))]);
        minor.determinant() * if (i + j) % 2 == 0 { 1.0 } else { -1.0 }
    };
    let diagonal = Vector4::from_fn(|j, _| cofactor(j, j).abs());
    let j = diagonal.imax();
    let q = if diagonal[j] > 1e-3 * l.powi(3) {
        Vector4::from_fn(|i, _| cofactor(i, j))
    } else {
        let eig = SymmetricEigen::new(n);
        eig.eigenvectors.column(eig.eigenvalues.imax()).into_owned()
    };
    Q::from_quaternion(Quaternion::new(q[0], q[1], q[2], q[3])).to_rotation_matrix().into_inner()
}

/// [R | t] of the joint J(q) of a family (q = (flip, yaw, slide) for stud, hinge, and axle), or of J(q)^-1 if
/// reversed.
pub fn motion(family: u8, reversed: bool, q: &[f64; 3]) -> (M3, V3) {
    let (m, t) = match family {
        BALL => (rot(2, q[2]) * rot(1, q[1]) * rot(0, q[0]), V3::zeros()),
        FIXED => (M3::identity(), V3::zeros()),
        _ => {
            let yaw = rot(1, q[1]);
            let turned = if q[0] != 0.0 { yaw * M3::from_diagonal(&V3::new(1.0, -1.0, -1.0)) } else { yaw };
            // the slide runs along -y (0 for stud and hinge)
            (turned, V3::new(0.0, -q[2], 0.0))
        }
    };
    if reversed {
        (m.transpose(), -(m.transpose() * t))
    } else {
        (m, t)
    }
}

/// Pose T(q) = F_a J(q) F_b^-1 of b in a's frame.
pub fn placement(r: &Region, q: &[f64; 3]) -> M4 {
    let (m, t) = motion(r.family, r.reversed, q);
    r.fa * pose(&m, &t) * inverse(&r.fb)
}

/// Joint F_a^-1 rel F_b of a placement rel of b in a's frame (inverted for a reversed region).
pub fn joint(r: &Region, rel: &M4) -> M4 {
    let j = inverse(&r.fa) * rel * r.fb;
    if r.reversed {
        inverse(&j)
    } else {
        j
    }
}

/// Joint coordinates of a placement as two candidates: one per flip, or the `twins` of a ball joint.
pub fn actual(r: &Region, rel: &M4) -> [[f64; 3]; 2] {
    let j = joint(r, rel);
    if r.family == BALL {
        let x = j[(2, 1)].atan2(j[(2, 2)]).to_degrees();
        let y = (-j[(2, 0)]).clamp(-1.0, 1.0).asin().to_degrees();
        let z = j[(1, 0)].atan2(j[(0, 0)]).to_degrees();
        return twins([x, y, z]);
    }
    let yaw = |sign: f64| (sign * j[(0, 2)]).atan2(j[(0, 0)]).to_degrees();
    let slide = -j[(1, 3)];
    [[0.0, yaw(1.0), slide], [1.0, yaw(-1.0), slide]]
}

/// Two Euler forms (x, y, z) and (x + 180, 180 - y, z + 180) of a ball joint J = Rz(z) Ry(y) Rx(x), in [0, 360).
pub fn twins(x: [f64; 3]) -> [[f64; 3]; 2] {
    [x, [x[0] + 180.0, 180.0 - x[1], x[2] + 180.0]].map(|e| e.map(|v| v.rem_euclid(360.0)))
}

/// x on the circle of period 360 moved into [lo, hi]: (its distance, the nearest point).
fn wrap(x: f64, lo: f64, hi: f64) -> (f64, f64) {
    let y = x + 360.0 * ((lo - x) / 360.0).ceil();
    let above = y - hi;
    let below = lo - (y - 360.0);
    if y <= hi {
        (0.0, y)
    } else if above <= below {
        (above, hi)
    } else {
        (below, lo)
    }
}

/// Nearest of the candidates of `actual` moved into a cell (angles by whole turns, the slide clamped), with the
/// squared distance moved (degrees^2, LDU^2).
pub fn into_cell(family: u8, candidates: &[[f64; 3]; 2], cell: &Cell) -> (f64, [f64; 3]) {
    match family {
        FIXED => (0.0, [0.0; 3]),
        BALL => {
            let mut best = (f64::INFINITY, [0.0; 3]);
            for x in candidates {
                let moved: [(f64, f64); 3] = std::array::from_fn(|a| wrap(x[a], cell[0][a], cell[1][a]));
                let cost = moved.iter().map(|(distance, _)| distance * distance).sum::<f64>();
                if cost < best.0 {
                    best = (cost, moved.map(|(_, y)| y));
                }
            }
            best
        }
        _ => {
            let x = candidates[usize::from(cell[0][0] != 0.0)];
            let (dy, yaw) = wrap(x[1], cell[0][1], cell[1][1]);
            let slide = x[2].clamp(cell[0][2], cell[1][2]);
            (dy * dy + (slide - x[2]).powi(2), [x[0], yaw, slide])
        }
    }
}

/// Ball cells in a tree of bounding boxes: node k (root 1, children 2k and 2k + 1) bounds cells `order[lo..hi]`, a
/// range halved down to LEAF cells.
pub struct Tree {
    order: Vec<usize>,
    bounds: Vec<Cell>,
}

impl Tree {
    /// Tree of `cells`.
    pub fn new(cells: &[Cell]) -> Tree {
        let size = 2 * cells.len().div_ceil(LEAF).next_power_of_two();
        let mut tree = Tree { order: (0..cells.len()).collect(), bounds: vec![[[0.0; 3]; 2]; size] };
        tree.split(cells, 1, 0, cells.len());
        tree
    }

    /// Bound node `node` over cells `order[lo..hi]`, then halve them at the median center along the axis where the
    /// centers spread most.
    fn split(&mut self, cells: &[Cell], node: usize, lo: usize, hi: usize) {
        let center = |c: usize, a: usize| cells[c][0][a] + cells[c][1][a]; // doubled
        let mut bound = [[f64::INFINITY; 3], [f64::NEG_INFINITY; 3]];
        let mut centers = bound;
        for &c in &self.order[lo..hi] {
            for a in 0..3 {
                bound[0][a] = bound[0][a].min(cells[c][0][a]);
                bound[1][a] = bound[1][a].max(cells[c][1][a]);
                centers[0][a] = centers[0][a].min(center(c, a));
                centers[1][a] = centers[1][a].max(center(c, a));
            }
        }
        self.bounds[node] = bound;
        if hi - lo <= LEAF {
            return;
        }
        let axis = V3::from_fn(|a, _| centers[1][a] - centers[0][a]).imax();
        let mid = (lo + hi) / 2;
        self.order[lo..hi].select_nth_unstable_by(mid - lo, |&a, &b| center(a, axis).total_cmp(&center(b, axis)));
        self.split(cells, 2 * node, lo, mid);
        self.split(cells, 2 * node + 1, mid, hi);
    }

    /// k cells nearest the ball joint candidates x as `into_cell` (distance, point, index), least first by
    /// distance raised to `floor`, then index: best first over the nodes, whose bounds' distance is a lower bound.
    pub fn nearest(&self, cells: &[Cell], x: &[[f64; 3]; 2], k: usize, floor: f64) -> Vec<(f64, [f64; 3], usize)> {
        let low = |node: usize| Reverse(ordered(into_cell(BALL, x, &self.bounds[node]).0.max(floor)));
        let mut queue = BinaryHeap::from([(low(1), 1, 0, self.order.len())]);
        let mut found: Vec<(f64, [f64; 3], usize)> = Vec::with_capacity(k + 1);
        while let Some((Reverse(bound), node, lo, hi)) = queue.pop() {
            // past the k-th cell, with room for rounding (a bound can exceed the distances it bounds by an ulp)
            if found.len() == k && bound > ordered(found[k - 1].0.max(floor) + 1e-6) {
                break;
            }
            if hi - lo > LEAF {
                let mid = (lo + hi) / 2;
                queue.push((low(2 * node), 2 * node, lo, mid));
                queue.push((low(2 * node + 1), 2 * node + 1, mid, hi));
                continue;
            }
            for &c in &self.order[lo..hi] {
                let (distance, q) = into_cell(BALL, x, &cells[c]);
                let at = found.partition_point(|f| (f.0.max(floor), f.2) < (distance.max(floor), c));
                found.insert(at, (distance, q, c));
                found.truncate(k);
            }
        }
        found
    }
}
