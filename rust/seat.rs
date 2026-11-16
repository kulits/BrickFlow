//! Connection hypotheses and seat distances.

// A hypothesis is one branch (group of admitted cells) of one mating connector pair of two parts. Its seat distance is
// the least mass-weighted mean squared displacement of both parts' material, once the common rigid motion is removed,
// over the joint coordinates q in its cells:
//
// ```text
// D^2 = min over q in its cells, E in SO(3) of
//      [m_a tr((E - I) S_a (E - I)^T) + m_b tr((E R1 - R0) S_b (E R1 - R0)^T) + mu |E g - h|^2] / m
// ```
//
// E is Horn's closed form. A search starts from the actual joint coordinates moved into the branch's nearest cell.
// Ball branches are searched from the free minimum over all rotations and its nearest cells. The energy polishes stud,
// hinge, and axle branches by projected Newton in that cell; the snap searches all their cells by branch and bound,
// projected Newton polishing the best few points. The gradient is that of the cost at the optimal E (envelope
// theorem).

use std::rc::Rc;

use crate::atlas::{Atlas, Body, Cell, Moments, Region, Regions, AXLE, BALL, FIXED, HINGE, STUD};
use crate::geometry::{
    actual, fit, into_cell, inverse, largest, motion, origin, pose, rot, rotation, twins, vee, Tree, M3, M4, V3,
};

const STEP_ANGLE: f64 = 30.0; // stud, hinge, and axle: smallest boxes of the search, degrees
const STEP_SLIDE: f64 = 6.0; // and LDU
const STARTS: usize = 4; // and points polished
const CELLS: usize = 64; // ball: cells searched
pub const SEATED: f64 = 1e-8; // D^2 below which a hypothesis is seated (a start point the search keeps), LDU^2
const H: f64 = 1e-2; // central-difference step of projected Newton

/// Seat of a hypothesis: D^2 (LDU^2), joint coordinates q*, argmin cell, and the twist gradients of D^2.
#[derive(Clone, Copy)]
pub struct Seat {
    pub d2: f64,
    pub q: [f64; 3],
    pub cell: usize,
    pub grad: [[f64; 6]; 2],
}

/// Part pair with hypotheses: parts (a, b), the regions of its hypotheses, rel = P_a^-1 P_b, and moments.
pub struct Pair<'a> {
    pub a: usize,
    pub b: usize,
    pub regions: Vec<Rc<Region>>,
    pub rel: M4,
    pub ma: &'a Moments,
    pub mb: &'a Moments,
}

/// Branch `group` of region `region` of part pair `pair`; gap: connector-origin distance less reach.
pub struct Hyp {
    pub pair: usize,
    pub region: usize,
    pub group: usize,
    pub gap: f64,
}

/// Joint coordinates q in cell `cell` with their cost.
#[derive(Clone, Copy)]
struct Point {
    cost: f64,
    q: [f64; 3],
    cell: usize,
}

/// Part pairs with hypotheses, and every hypothesis with connector-origin gap below `radius`. A pair farther apart
/// than its connector pairs' extent is skipped whole; each type pair's connector pairs are listed once, and only for
/// a pair within the bound `Atlas::span` of their extent.
pub fn hypotheses<'a>(
    atlas: &'a Atlas,
    ids: &[u32],
    poses: &[M4],
    radius: f64,
    keep: impl Fn(usize, usize) -> bool,
) -> (Vec<Pair<'a>>, Vec<Hyp>) {
    let mut pairs = Vec::new();
    let mut hyps = Vec::new();
    let mut types = ids.to_vec();
    types.sort_unstable();
    types.dedup();
    let rank: Vec<usize> = ids.iter().map(|t| types.binary_search(t).expect("a listed type")).collect();
    let mut cached: Vec<Option<Rc<Regions>>> = vec![None; types.len() * types.len()];
    for i in 0..ids.len() {
        for j in (i + 1..ids.len()).filter(|&j| keep(i, j)) {
            let (a, b) = if ids[i] <= ids[j] { (i, j) } else { (j, i) };
            let distance = (origin(&poses[a]) - origin(&poses[b])).norm();
            if distance > atlas.span(ids[a], ids[b]) + radius {
                continue;
            }
            let candidates =
                cached[rank[a] * types.len() + rank[b]].get_or_insert_with(|| atlas.regions(ids[a], ids[b]));
            if distance > candidates.extent + radius {
                continue;
            }
            let rel = inverse(&poses[a]) * poses[b];
            let mut regions = Vec::new();
            atlas.near(candidates, &rel, radius, |region, distance| {
                let first = hyps.len();
                for (group, branch) in atlas.body(&region).groups.iter().enumerate() {
                    let gap = distance - branch.reach;
                    if gap < radius {
                        hyps.push(Hyp { pair: pairs.len(), region: regions.len(), group, gap });
                    }
                }
                if hyps.len() > first {
                    regions.push(region);
                }
            });
            if !regions.is_empty() {
                let moments = |p: usize| atlas.moments(ids[p]).expect("parts with regions have moments");
                pairs.push(Pair { a, b, regions, rel, ma: moments(a), mb: moments(b) });
            }
        }
    }
    (pairs, hyps)
}

/// D^2 over the joint coordinates of one branch (cells `index` of `cells`) at one actual pose.
struct Objective<'a> {
    region: &'a Region,
    cells: &'a [Cell],
    index: &'a [usize],
    tree: Option<&'a Tree>,
    pair: &'a Pair<'a>,
    r0: M3,
    h: V3,
    w: V3,
    cb: V3,
    k0: M3,
    k1: M3,
    m: f64,
    mu: f64,
    constant: f64,
    // stud, hinge, and axle: K and g are affine in (1, cos yaw, sin yaw, slide) at the branch's flip
    line: Option<([M3; 4], [V3; 4])>,
    stop: f64, // projected Newton's last step (degrees, LDU)
}

impl<'a> Objective<'a> {
    fn new(region: &'a Region, pair: &'a Pair<'a>, body: &'a Body, index: &'a [usize]) -> Self {
        let ma = pair.ma;
        let mb = pair.mb;
        let r0 = rotation(&pair.rel);
        let m = ma.mass + mb.mass;
        let mu = ma.mass * mb.mass / m;
        let h = ma.c - (r0 * mb.c + origin(&pair.rel));
        Objective {
            region,
            cells: &body.cells,
            index,
            tree: body.tree.as_ref(),
            pair,
            r0,
            h,
            w: ma.c - origin(&region.fa),
            cb: rotation(&region.fb).transpose() * (mb.c - origin(&region.fb)),
            k0: ma.s * ma.mass,
            k1: mb.s * r0.transpose() * mb.mass,
            m,
            mu,
            constant: 2.0 * (ma.mass * ma.s.trace() + mb.mass * mb.s.trace()) + mu * h.norm_squared(),
            line: None,
            stop: 1e-5,
        }
    }

    /// Copy to search to steps of `stop`; stud, hinge, and axle: K and g in their affine form along the branch's flip.
    fn searching(&self, stop: f64) -> Self {
        let mut obj = Objective { stop, ..*self };
        if matches!(self.region.family, STUD | HINGE | AXLE) {
            let flip = self.cells[self.index[0]][0][0];
            let at = [(0.0, 0.0), (180.0, 0.0), (90.0, 0.0), (0.0, 1.0)];
            let [(k_0, g_0), (k_180, g_180), (k_90, g_90), (k_s, g_s)] =
                at.map(|(yaw, slide)| self.kg(&[flip, yaw, slide]));
            let (k_mean, g_mean) = ((k_0 + k_180) * 0.5, (g_0 + g_180) * 0.5);
            obj.line = Some((
                [k_mean, (k_0 - k_180) * 0.5, k_90 - k_mean, k_s - k_0],
                [g_mean, (g_0 - g_180) * 0.5, g_90 - g_mean, g_s - g_0],
            ));
        }
        obj
    }

    /// R1 and g at q.
    fn target(&self, q: &[f64; 3]) -> (M3, V3) {
        let (m, t) = motion(self.region.family, self.region.reversed, q);
        let a = rotation(&self.region.fa);
        let am = a * m;
        (am * rotation(&self.region.fb).transpose(), self.w - (am * self.cb + a * t))
    }

    /// K and g at q.
    fn kg(&self, q: &[f64; 3]) -> (M3, V3) {
        if let Some((k, g)) = &self.line {
            let (s, c) = q[1].to_radians().sin_cos();
            return (k[0] + k[1] * c + k[2] * s + k[3] * q[2], g[0] + g[1] * c + g[2] * s + g[3] * q[2]);
        }
        let (r1, g) = self.target(q);
        (self.k(&r1, &g), g)
    }

    /// K = k0 + R1 k1 + mu g h^T.
    fn k(&self, r1: &M3, g: &V3) -> M3 {
        self.k0 + r1 * self.k1 + g * self.h.transpose() * self.mu
    }

    /// D^2 at points q from the largest eigenvalue, (C + mu |g|^2 - 2 lambda_max(K)) / m, the roots found in step.
    fn values<const N: usize>(&self, q: [[f64; 3]; N]) -> [f64; N] {
        let kg = q.map(|q| self.kg(&q));
        let l = largest(kg.map(|(k, _)| k));
        std::array::from_fn(|i| (self.constant + self.mu * kg[i].1.norm_squared() - 2.0 * l[i]) / self.m)
    }

    fn value(&self, q: &[f64; 3]) -> f64 {
        self.values([*q])[0]
    }

    /// D^2 on the central-difference stencil of x in a cell, in step: lanes i, 3 + i, and 6 + i at x + H e_i,
    /// x - H e_i, and x + H (1 - e_i); stud, hinge, and axle joints along yaw and slide only, in a cell of one slide
    /// along yaw only (the other lanes 0).
    fn stencil(&self, x: &[f64; 3], cell: &Cell) -> [f64; 9] {
        let at = |n: usize| {
            let d = match n / 3 {
                0 => V3::ith(n % 3, H),
                1 => V3::ith(n % 3, -H),
                _ => V3::repeat(H) - V3::ith(n % 3, H),
            };
            std::array::from_fn(|k| x[k] + d[k])
        };
        if self.line.is_none() {
            return self.values(std::array::from_fn(at));
        }
        if cell[0][2] == cell[1][2] {
            let f = self.values([at(1), at(4)]);
            return [0.0, f[0], 0.0, 0.0, f[1], 0.0, 0.0, 0.0, 0.0];
        }
        let f = self.values([1, 2, 4, 5, 6].map(at));
        [0.0, f[0], f[1], 0.0, f[2], f[3], f[4], 0.0, 0.0]
    }

    /// Seat at point p: D^2 in residual form and its gradient in the body twists [w, v] of a and b at the gauge E*
    /// (envelope).
    fn seat(&self, p: &Point) -> Seat {
        let Pair { ma, mb, .. } = self.pair;
        let (r1, g) = self.target(&p.q);
        let e = fit(&self.k(&r1, &g));
        let da = e - M3::identity();
        let db = e * r1 - self.r0;
        let r = e * g - self.h;
        let d2 =
            (ma.mass * (da * ma.s).dot(&da) + mb.mass * (db * mb.s).dot(&db) + self.mu * r.norm_squared()) / self.m;
        let dbs = db * mb.s;
        let rb = self.r0.transpose() * r;
        let centroid_b = self.r0 * mb.c + origin(&self.pair.rel);
        let s = 2.0 / self.m;
        let wa = (vee(&(self.r0 * dbs.transpose())) * mb.mass - centroid_b.cross(&r) * self.mu) * s;
        let wb = (mb.c.cross(&rb) * self.mu - vee(&(dbs.transpose() * self.r0)) * mb.mass) * s;
        let va = r * (-self.mu * s);
        let vb = rb * (self.mu * s);
        let grad = [[wa[0], wa[1], wa[2], va[0], va[1], va[2]], [wb[0], wb[1], wb[2], vb[0], vb[1], vb[2]]];
        Seat { d2, q: p.q, cell: p.cell, grad }
    }

    /// Actual joint coordinates moved into the first nearest cell of the branch (cost: the squared distance moved).
    fn nearest(&self) -> Point {
        let candidates = actual(self.region, &self.pair.rel);
        if let Some(tree) = self.tree {
            let (cost, q, cell) = tree.nearest(self.cells, &candidates, 1, 0.0)[0];
            return Point { cost, q, cell };
        }
        let mut best = Point { cost: f64::INFINITY, q: [0.0; 3], cell: 0 };
        for &c in self.index {
            let (cost, q) = into_cell(self.region.family, &candidates, &self.cells[c]);
            if cost < best.cost {
                best = Point { cost, q, cell: c };
            }
            if cost == 0.0 {
                break;
            }
        }
        best
    }

    /// Ball joints: the free minimum over all rotations (unbounded Newton from the best of the actual joint and the 24
    /// cube turns of a decoupled fit, each part's rotation fitted alone), the seat if it lies in a cell; else projected
    /// Newton from the best of `from` and its projections onto the CELLS nearest cells.
    fn ball(&self, from: Point) -> Point {
        let d = self.pair.mb.c - origin(&self.region.fb);
        let pa = self.k0 + self.w * self.h.transpose() * self.mu;
        let pb = self.k1 - d * self.h.transpose() * self.mu;
        let decoupled = fit(&pa).transpose() * fit(&pb);
        let faces = [rot(2, 0.0), rot(2, 90.0), rot(2, 180.0), rot(2, 270.0), rot(1, 90.0), rot(1, 270.0)];
        let cube =
            faces.map(|face| [0.0, 90.0, 180.0, 270.0].map(|turn| decoupled * (face * rot(0, turn)).map(f64::round)));
        let euler = |r1: &M3| actual(self.region, &pose(r1, &V3::zeros()))[0];
        let rotations = std::iter::once(self.r0).chain(cube.into_iter().flatten());
        let mut starts: Vec<Point> = rotations.map(|r1| self.point(euler(&r1), 0)).collect();
        starts.sort_by(|a, b| a.cost.total_cmp(&b.cost));
        let unbounded = [[f64::NEG_INFINITY; 3], [f64::INFINITY; 3]];
        let free = starts[..4].iter().map(|&s| self.descend(s, &unbounded));
        let free = free.min_by(|a, b| a.cost.total_cmp(&b.cost)).expect("starts");
        // the first cell within 1e-20 of the free minimum, else the CELLS nearest
        let near = self.tree.expect("ball bodies have trees").nearest(self.cells, &twins(free.q), CELLS, 1e-20);
        if near[0].0 <= 1e-20 {
            return self.point(near[0].1, near[0].2);
        }
        let mut scored: Vec<Point> = near.iter().map(|&(_, q, cell)| self.point(q, cell)).collect();
        scored.push(from);
        scored.sort_by(|a, b| a.cost.total_cmp(&b.cost));
        let polished = scored.iter().take(8).map(|&s| self.descend(s, &self.cells[s.cell]));
        polished.min_by(|a, b| a.cost.total_cmp(&b.cost)).expect("a start")
    }

    /// Stud, hinge, and axle joints: branch and bound over yaw x slide boxes, bounded below from D^2 at the corners and
    /// its curvature (along yaw 2 L^2 + 2 L sqrt(D^2), L the smaller of sqrt(m_x / m) times part x's rms distance from
    /// the axis; along the slide 2 mu / m), and cells within half a step from their center (sqrt(D^2) is L-Lipschitz).
    /// Projected Newton polishes the best point and up to STARTS distinct leaves.
    fn revolute(&self, from: Point) -> Point {
        let rates = self.rates();
        let mut best = from;
        let mut boxes = self.cover(rates, &mut best);
        boxes.retain(|b| b.low < limit(&best));
        let mut seeds = vec![best];
        while let Some(i) = (0..boxes.len()).min_by(|&a, &b| boxes[a].low.total_cmp(&boxes[b].low)) {
            let Patch { low, lo, hi, f, cell } = boxes.swap_remove(i);
            if low >= limit(&best) {
                break;
            }
            let h = (hi[1] - lo[1]) / STEP_ANGLE;
            let h_slide = (hi[2] - lo[2]) / STEP_SLIDE;
            if h <= 1.0 && h_slide <= 1.0 {
                let i = (0..4).min_by(|&a, &b| f[a].total_cmp(&f[b])).expect("corners");
                seeds.push(Point { cost: f[i], q: corner(&lo, &hi, i), cell });
                continue;
            }
            // halve the longer axis k (in steps): corners with bit k set lie on its hi side, the middle line is new
            let k = if h >= h_slide { 1 } else { 2 };
            let mut middle_lo = lo;
            let mut middle_hi = hi;
            middle_lo[k] = 0.5 * (lo[k] + hi[k]);
            middle_hi[k] = middle_lo[k];
            let middle = self.corners(&middle_lo, &hi, k, cell, &mut best);
            let below = std::array::from_fn(|i| if i & k == 0 { f[i] } else { middle[i - k] });
            let above = std::array::from_fn(|i| if i & k == 0 { middle[i] } else { f[i] });
            let halves = [self.patch(rates, lo, middle_hi, below, cell), self.patch(rates, middle_lo, hi, above, cell)];
            boxes.extend(halves.into_iter().filter(|half| half.low < limit(&best)));
        }
        seeds[0] = best;
        self.polish(best, seeds)
    }

    /// Rates L (per radian of yaw) and L_slide (per LDU of slide) at which sqrt(D^2) changes at most.
    fn rates(&self) -> (f64, f64) {
        let rms = |m: &Moments, f: &M4| {
            let offset = m.c - origin(f);
            let axis: V3 = f.fixed_view::<3, 1>(0, 1).into();
            let along = (axis.transpose() * m.s * axis)[0] + offset.dot(&axis).powi(2);
            (m.mass / self.m * (m.s.trace() + offset.norm_squared() - along)).sqrt()
        };
        let l = rms(self.pair.ma, &self.region.fa).min(rms(self.pair.mb, &self.region.fb));
        let l_slide = (self.pair.ma.mass.min(self.pair.mb.mass) / self.m).sqrt();
        (l, l_slide)
    }

    /// D^2 at the corners of box [lo, hi] with `bit` clear, the distinct ones in step (copied across a degenerate
    /// side), the least point found updated.
    fn corners(&self, lo: &[f64; 3], hi: &[f64; 3], bit: usize, cell: usize, best: &mut Point) -> [f64; 4] {
        let flat = usize::from(lo[1] == hi[1]) | (usize::from(lo[2] == hi[2]) << 1);
        let free = 3 & !(bit | flat);
        let mut f = [0.0; 4];
        match free {
            0 => f[0] = self.value(&corner(lo, hi, 0)),
            3 => f = self.values(std::array::from_fn(|i| corner(lo, hi, i))),
            _ => [f[0], f[free]] = self.values([corner(lo, hi, 0), corner(lo, hi, free)]),
        }
        for i in (0..4).filter(|i| i & bit == 0) {
            f[i] = f[i & !flat];
            if f[i] < best.cost {
                *best = Point { cost: f[i], q: corner(lo, hi, i), cell };
            }
        }
        f
    }

    /// Box [lo, hi] of cell `cell` with D^2 f at its corners and its lower bound, at `rates`.
    fn patch(&self, rates: (f64, f64), lo: [f64; 3], hi: [f64; 3], f: [f64; 4], cell: usize) -> Patch {
        let (l, l_slide) = rates;
        let h = (hi[1] - lo[1]).to_radians();
        let h_slide = hi[2] - lo[2];
        let least = f.iter().copied().fold(f64::INFINITY, f64::min);
        let a = l * (l + least.max(0.0).sqrt() + l * h + l_slide * h_slide) * h * h;
        let low = dip(dip(f[0], f[1], a), dip(f[2], f[3], a), self.mu / self.m * h_slide * h_slide);
        Patch { low, lo, hi, f, cell }
    }

    /// Boxes of the branch's cells, clipped to the slab of slides where D^2 can be stationary along the slide, the
    /// least point found updated. Cells within half a step are bounded from D^2 at their center (four in step), whose
    /// root falls by at most (L h + L_slide h_slide) / 2.
    fn cover(&self, rates: (f64, f64), best: &mut Point) -> Vec<Patch> {
        let (l, l_slide) = rates;
        // the slab: there g . g_s = h . E g_s, at most |h| |g_s|, and g . g_s is affine in the slide alone (the yaw
        // turns about the slide axis g_s)
        let g = &self.line.as_ref().expect("a revolute line").1;
        let along = g[3].norm_squared();
        let center = -g[0].dot(&g[3]) / along;
        let half = (self.h.norm() * along.sqrt() + g[1].dot(&g[3]).abs() + g[2].dot(&g[3]).abs()) / along;
        let mut boxes = Vec::new();
        let mut small = Vec::new();
        for &c in self.index {
            let [mut lo, mut hi] = self.cells[c];
            lo[2] = (center - half).clamp(self.cells[c][0][2], self.cells[c][1][2]);
            hi[2] = (center + half).clamp(self.cells[c][0][2], self.cells[c][1][2]);
            let h = hi[1] - lo[1];
            let h_slide = hi[2] - lo[2];
            if h > STEP_ANGLE / 2.0 || h_slide > STEP_SLIDE / 2.0 {
                let f = self.corners(&lo, &hi, 0, c, best);
                boxes.push(self.patch(rates, lo, hi, f, c));
            } else {
                let fall = (l * h.to_radians() + l_slide * h_slide) / 2.0;
                small.push(([lo[0], lo[1] + h / 2.0, lo[2] + h_slide / 2.0], fall, c));
            }
        }
        for four in small.chunks(4) {
            let cost = self.values::<4>(std::array::from_fn(|i| four[i.min(four.len() - 1)].0));
            for (&(q, fall, c), cost) in four.iter().zip(cost) {
                if cost < best.cost {
                    *best = Point { cost, q, cell: c };
                }
                let root = (cost.max(0.0).sqrt() - fall).max(0.0);
                boxes.push(Patch { low: root * root, lo: q, hi: q, f: [cost; 4], cell: c });
            }
        }
        boxes
    }

    /// Projected Newton from the best point and up to STARTS seeds apart from those taken, the least point found.
    fn polish(&self, best: Point, seeds: Vec<Point>) -> Point {
        let mut used: Vec<Point> = Vec::with_capacity(STARTS);
        for s in seeds {
            let near = |p: &Point| {
                p.cell == s.cell && (p.q[1] - s.q[1]).abs() <= STEP_ANGLE && (p.q[2] - s.q[2]).abs() <= STEP_SLIDE
            };
            if used.len() < STARTS && !used.iter().any(near) {
                used.push(s);
            }
        }
        let polished = used.into_iter().map(|s| self.descend(s, &self.cells[s.cell]));
        polished.fold(best, |a, b| if b.cost < a.cost { b } else { a })
    }

    fn point(&self, q: [f64; 3], cell: usize) -> Point {
        Point { cost: self.value(&q), q, cell }
    }

    /// Projected Newton for the minimum of D^2 over a cell from p, on the central differences of `Self::stencil`:
    /// coordinates at a bound with an outward slope are held, and a step halves until D^2 decreases. Where the Hessian
    /// is not positive definite it steps down the gradient, or (stud, hinge, and axle joints) each coordinate by its
    /// own curvature, to its bound where that is not positive.
    fn descend(&self, p: Point, cell: &Cell) -> Point {
        let [lo, hi] = cell;
        if lo == hi {
            return p;
        }
        let Point { q: mut x, cost: mut fx, .. } = p;
        for _ in 0..50 {
            let mut g = V3::zeros();
            let mut hessian = M3::identity();
            let f = self.stencil(&x, cell);
            for i in (0..3).filter(|&i| hi[i] > lo[i]) {
                g[i] = (f[i] - f[i + 3]) / (2.0 * H);
                hessian[(i, i)] = (f[i] - 2.0 * fx + f[i + 3]) / (H * H);
                if x[i] <= lo[i] && g[i] > 0.0 || x[i] >= hi[i] && g[i] < 0.0 {
                    g[i] = 0.0;
                    hessian[(i, i)] = 1.0;
                }
            }
            for i in 0..3 {
                for j in (0..i).filter(|&j| g[i] != 0.0 && g[j] != 0.0) {
                    hessian[(i, j)] = (f[9 - i - j] - f[i] - f[j] + fx) / (H * H); // lane 6 + the third axis
                    hessian[(j, i)] = hessian[(i, j)];
                }
            }
            if g.norm() == 0.0 {
                break;
            }
            let alone =
                |i, _| if hessian[(i, i)] > 0.0 { -g[i] / hessian[(i, i)] } else { -g[i].signum() * (hi[i] - lo[i]) };
            let fallback = if self.line.is_some() { V3::from_fn(alone) } else { -g / g.norm() };
            let d = hessian.cholesky().map(|c| -c.solve(&g)).filter(|d| d.dot(&g) < 0.0).unwrap_or(fallback);
            let mut t = 1.0;
            loop {
                let y: [f64; 3] = std::array::from_fn(|k| (x[k] + t * d[k]).clamp(lo[k], hi[k]));
                let fy = self.value(&y);
                if fy < fx {
                    x = y;
                    fx = fy;
                    break;
                }
                t *= 0.5;
                if t * d.norm() < 1e-4 * self.stop {
                    return Point { cost: fx, q: x, ..p };
                }
            }
            if t * d.amax() < self.stop {
                break;
            }
        }
        Point { cost: fx, q: x, ..p }
    }
}

/// Corner i of a box [lo, hi]: bit 0 yaw at hi, bit 1 slide at hi.
fn corner(lo: &[f64; 3], hi: &[f64; 3], i: usize) -> [f64; 3] {
    [lo[0], [lo[1], hi[1]][i & 1], [lo[2], hi[2]][i >> 1]]
}

/// Boxes of the revolute search below this are searched, the others dropped as made.
fn limit(best: &Point) -> f64 {
    best.cost * (1.0 - 1e-4) - 1e-12
}

/// Yaw x slide box [lo, hi] of cell `cell`: D^2 f at its corners and a lower bound `low` on it.
struct Patch {
    low: f64,
    lo: [f64; 3],
    hi: [f64; 3],
    f: [f64; 4],
    cell: usize,
}

/// Least of x0 + (x1 - x0) t - c t (1 - t) over t in [0, 1]: a lower bound along an edge of a function with second
/// derivative at most 2 c in t, from its ends.
fn dip(x0: f64, x1: f64, c: f64) -> f64 {
    let s = x1 - x0;
    if s.abs() < c {
        x0 - (c - s).powi(2) / (4.0 * c)
    } else {
        x0.min(x1)
    }
}

/// Hypothesis h's objective and its seat at its start point.
pub struct Start<'a> {
    obj: Objective<'a>,
    pub seat: Seat,
}

/// Start of hypothesis h: its objective and the seat at the nearest point of its branch.
pub fn start<'a>(atlas: &'a Atlas, pairs: &'a [Pair<'a>], h: &Hyp) -> Start<'a> {
    let pair = &pairs[h.pair];
    let region = &pair.regions[h.region];
    let body = atlas.body(region);
    let obj = Objective::new(region, pair, body, &body.groups[h.group].cells);
    let seat = obj.seat(&obj.nearest());
    Start { obj, seat }
}

impl Start<'_> {
    /// Least seat the search finds from the start point, which stands if it is seated. `local` (the energy's):
    /// projected Newton to steps of 1e-3 in the start point's cell for stud, hinge, and axle branches, and no search
    /// at all from a start point not below `ceiling`; else (the snap's) branch and bound over all cells.
    pub fn search(&self, local: bool, ceiling: f64) -> Seat {
        if self.obj.region.family == FIXED || self.seat.d2 <= SEATED || local && self.seat.d2 >= ceiling {
            return self.seat;
        }
        let obj = self.obj.searching(if local { 1e-3 } else { 1e-5 });
        let from = Point { cost: self.seat.d2, q: self.seat.q, cell: self.seat.cell };
        let found = match (obj.region.family, local) {
            (BALL, _) => obj.ball(from),
            (_, true) => obj.descend(from, &obj.cells[from.cell]),
            (_, false) => obj.revolute(from),
        };
        obj.seat(&found)
    }
}
