//! Connector regions of the connection atlas.

use std::cell::{OnceCell, RefCell};
use std::collections::HashMap;
use std::io;
use std::ops::Range;
use std::path::Path;
use std::rc::Rc;

use nalgebra::Vector4;
use procbrick::parts::{self, pairing, Kind, Pairing, PartConnector};
use procbrick::store::{bit, wrap, Domain, Row, Store};
pub use procbrick::store::{AXLE, BALL, FIXED, HINGE, STUD};
use procbrick::trees::matrix;

use crate::geometry::{inverse, origin, Tree, M3, M4, V3};

const BUDGET: usize = 1 << 27; // bytes of decoded atlas units cached

pub type Cell = [[f64; 3]; 2];

/// Mass, centroid, and covariance per unit mass of a part in its own frame (LDU).
pub struct Moments {
    pub mass: f64,
    pub c: V3,
    pub s: M3,
}

/// Branch of a region: its cells (indices into the region's cells) and its reach, the largest |slide| (LDU).
pub struct Group {
    pub cells: Vec<usize>,
    pub reach: f64,
}

/// Cells of a region, largest first, and its branches.
pub struct Body {
    pub cells: Vec<Cell>,
    pub groups: Vec<Group>,
    pub tree: Option<Tree>, // ball: the cells' tree
}

/// One admitted connector pair (ca of type a, cb of type b).
pub struct Region {
    pub ca: u32,
    pub cb: u32,
    pub family: u8,
    pub reversed: bool,
    pub fa: M4,
    pub fb: M4,
    domain: Option<Domain>,
    body: OnceCell<Body>,
}

/// Mating connector pairs of a type pair: per connector of ta that has some, a run of its mates among the connectors
/// of tb in order of their origin's x; `extent` is the largest |fa origin| + |fb origin| + reach. For culling: the
/// box of the mates' origins, the sphere (center, radius) of the runs' origins, and their largest reach.
pub struct Regions {
    ta: u32,
    tb: u32,
    runs: Vec<Run>,
    mates: Vec<Mate>,
    bounds: [V3; 2],
    sphere: (Vector4<f64>, f64),
    reach: f64,
    pub extent: f64,
}

/// Run of mates[range] of connector a of ta: its origin (homogeneous) and the largest reach of their regions.
struct Run {
    a: usize,
    range: Range<usize>,
    origin: Vector4<f64>,
    reach: f64,
}

/// Mate of a run's connector: connector b of tb, its origin, the reach of their region (bounding its branches'
/// reaches), and the region, made when it first comes within reach.
struct Mate {
    b: u32,
    at: V3,
    reach: f64,
    region: OnceCell<Rc<Region>>,
}

/// Connector of a part: its index, frame, kind (fixed or profiled), and length (LDU; 0 for fixed connectors).
#[derive(Clone)]
pub struct Connector {
    index: u32,
    pub frame: M4,
    kind: Kind,
    pub length: f64,
}

/// Atlas of an assets directory and the mass properties of its parts.
pub struct Atlas {
    store: RefCell<Store>,
    parts: Vec<Vec<Connector>>,
    spans: Vec<(f64, f64)>, // per part type: its connector origins' largest distance and largest reach (LDU)
    bounds: Vec<Option<[V3; 2]>>, // per part type: its collider box [lo, hi] in its frame (None: no collider)
    moments: Vec<Option<Moments>>,
    cached: RefCell<HashMap<(u32, u32), Rc<Regions>>>,
}

/// Root of i's component under the union-find parents `root`.
pub fn top(root: &[usize], mut i: usize) -> usize {
    while root[i] != i {
        i = root[i];
    }
    i
}

/// Join the components of a and b under the union-find parents `root` (the lower root stays).
pub fn join(root: &mut [usize], a: usize, b: usize) {
    let ra = top(root, a);
    let rb = top(root, b);
    root[ra.max(rb)] = ra.min(rb);
}

/// Components of n nodes joined by `edges`, each in increasing order, ordered by their lowest node.
pub fn components(n: usize, edges: impl IntoIterator<Item = (usize, usize)>) -> Vec<Vec<usize>> {
    let mut root: Vec<usize> = (0..n).collect();
    for (a, b) in edges {
        join(&mut root, a, b);
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| top(&root, i));
    order.chunk_by(|&a, &b| top(&root, a) == top(&root, b)).map(<[usize]>::to_vec).collect()
}

/// Row with samples `f(z)`, z < 360, wrapped.
fn row(f: impl Fn(usize) -> bool) -> Row {
    let mut r = [0; 6];
    for z in (0..360).filter(|&z| f(z)) {
        r[z / 64] |= 1 << (z % 64);
    }
    wrap(r)
}

/// Samples admitted in both rows.
fn and(a: Row, b: Row) -> Row {
    std::array::from_fn(|k| a[k] & b[k])
}

/// Maximal runs [lo, hi] of admitted samples, in order.
fn spans(r: &Row) -> impl Iterator<Item = (i32, i32)> + '_ {
    // the first sample from k on that is occupied (with `occupied` false: clear), 361 if none; a 64-bit word at a time
    let next = |mut k: usize, occupied: bool| loop {
        let word = if occupied { r[k / 64] } else { !r[k / 64] } >> (k % 64);
        if word != 0 || k >= 320 {
            return (k + word.trailing_zeros() as usize).min(361);
        }
        k = (k / 64 + 1) * 64;
    };
    let mut k = next(0, true);
    std::iter::from_fn(move || {
        if k >= 361 {
            return None;
        }
        let stop = next(k, false);
        let run = (k as i32, stop as i32 - 1);
        k = next(stop, true);
        Some(run)
    })
}

/// Maximal yaw windows [lo, hi] of admitted samples, a run through 360 continuing at 0 (ending past 360).
fn yaw_windows(r: &Row) -> Vec<(f64, f64)> {
    let mut out: Vec<(i32, i32)> = spans(r).collect();
    let n = out.len();
    if n > 1 && out[0].0 == 0 && out[n - 1].1 == 360 {
        let first = out.remove(0);
        out[n - 2].1 = 360 + first.1;
    }
    out.into_iter().map(|(lo, hi)| (f64::from(lo), f64::from(hi))).collect()
}

/// Closed intersection of two yaw windows on the circle (windows may run past 360).
fn overlap(a: (f64, f64), b: (f64, f64)) -> Vec<(f64, f64)> {
    if a.1 - a.0 >= 360.0 {
        return vec![b];
    }
    if b.1 - b.0 >= 360.0 {
        return vec![a];
    }
    let shifted = [-360.0, 0.0, 360.0].map(|k| (a.0.max(b.0 + k), a.1.min(b.1 + k)));
    shifted.into_iter().filter(|w| w.0 <= w.1).collect()
}

/// Ball box [[x, y, z], [x, y, z]] of grid samples, its corners at most 360.
type Box3 = [[i32; 3]; 2];

/// Merge boxes that agree on the two axes other than `axis` and touch along it.
fn unite(boxes: &mut Vec<Box3>, axis: usize) {
    // a box as the key of its fields b[0][o], b[1][o], b[0][p], b[1][p], b[0][axis], b[1][axis] (o, p the other axes),
    // 9 bits each, so that keys sort as the fields
    let fields =
        [(0, (axis + 1) % 3), (1, (axis + 1) % 3), (0, (axis + 2) % 3), (1, (axis + 2) % 3), (0, axis), (1, axis)];
    let mut keys: Vec<u64> =
        boxes.iter().map(|b| fields.iter().fold(0, |k, &(c, i)| k << 9 | b[c][i] as u64)).collect();
    keys.sort_unstable();
    // a key joins the box m before it if their other axes agree and it starts at most one past m's end
    let mut merged: Vec<u64> = Vec::with_capacity(keys.len());
    for k in keys {
        match merged.last_mut() {
            Some(m) if *m >> 18 == k >> 18 && (k >> 9 & 511) <= (*m & 511) + 1 => {
                *m = *m & !511 | (*m & 511).max(k & 511)
            }
            _ => merged.push(k),
        }
    }
    boxes.clear();
    for k in merged {
        let mut b = [[0; 3]; 2];
        for (n, &(c, i)) in fields.iter().enumerate() {
            b[c][i] = (k >> (9 * (5 - n)) & 511) as i32;
        }
        boxes.push(b);
    }
}

/// Merge ball boxes, `made` of them united along x as they were made, where they agree on two axes and touch along
/// the third (rounds over x, y, and z, the first x pass done, while the count falls, at most three); drop boxes
/// inside others.
fn merge(boxes: &mut Vec<Box3>, made: usize) {
    let mut before = made;
    for round in 0..3 {
        for axis in (0..3).filter(|&axis| round > 0 || axis > 0) {
            unite(boxes, axis);
        }
        if boxes.len() == before {
            break;
        }
        before = boxes.len();
    }
    // a box inside another has its lower corner in a grid cell that the other overlaps: cells of 16 samples, 23 per
    // axis to cover samples 0..=360
    const SIDE: i32 = 16;
    const CELLS: usize = 23;
    let cell = |p: [i32; 3]| p.iter().fold(0, |k, &v| k * CELLS + (v / SIDE) as usize);
    let mut grid = vec![Vec::new(); CELLS * CELLS * CELLS];
    for b in boxes.iter() {
        for x in b[0][0] / SIDE..=b[1][0] / SIDE {
            for y in b[0][1] / SIDE..=b[1][1] / SIDE {
                for z in b[0][2] / SIDE..=b[1][2] / SIDE {
                    grid[cell([SIDE * x, SIDE * y, SIDE * z])].push(*b);
                }
            }
        }
    }
    let inside = |b: &Box3, a: &Box3| a != b && (0..3).all(|k| a[0][k] <= b[0][k] && b[1][k] <= a[1][k]);
    let kept = boxes.iter().filter(|b| !grid[cell(b[0])].iter().any(|a| inside(b, a)));
    *boxes = kept.copied().collect();
}

/// Branches of a region's cells: for hinge and axle the components of cells of one flip (axle: touching slides and
/// overlapping yaws), else one.
fn branches(cells: &[Cell], family: u8) -> Vec<Group> {
    if !matches!(family, HINGE | AXLE) {
        return vec![Group { cells: (0..cells.len()).collect(), reach: 0.0 }];
    }
    let linked = |a: &Cell, b: &Cell| {
        let touching = a[0][2] <= b[1][2] && b[0][2] <= a[1][2];
        let yaws = || !overlap((a[0][1], a[1][1]), (b[0][1], b[1][1])).is_empty();
        a[0][0] == b[0][0] && (family == HINGE || (touching && yaws()))
    };
    let edges = (0..cells.len()).flat_map(|i| (0..i).map(move |j| (i, j)));
    let reach = |&i: &usize| cells[i][0][2].abs().max(cells[i][1][2].abs());
    let group = |g: Vec<usize>| Group { reach: g.iter().map(reach).fold(0.0, f64::max), cells: g };
    let linked_edges = edges.filter(|&(i, j)| linked(&cells[i], &cells[j]));
    components(cells.len(), linked_edges).into_iter().map(group).collect()
}

/// Domain (none: fixed) of connectors ca and cb and whether it is stored reversed, if they mate: their types mate
/// and, unless fixed, some sample of their domain is admitted.
fn mate(store: &mut Store, ca: &Connector, cb: &Connector) -> Option<(Option<Domain>, bool)> {
    match pairing(&store.chart, ca.kind, cb.kind)? {
        Pairing::Fixed(_) => Some((None, false)),
        Pairing::Profiled { domain, reversed, flag } => store.flag(flag).then_some((Some(domain), reversed)),
    }
}

impl Atlas {
    /// Atlas in the assets directory `root` (its connectors from ProcBrick's `pose_dictionary` and `parts.json`);
    /// `moments` by part id.
    pub fn open(root: &Path, moments: Vec<Option<Moments>>) -> io::Result<Self> {
        let store = Store::open(root, BUDGET)?;
        let assets = parts::open(root, &store.chart)?;
        let connector = |c: &PartConnector| Connector {
            index: u32::from(c.connector.id),
            frame: M4::from_row_slice(&matrix(c.pose)),
            kind: c.kind(),
            length: c.length,
        };
        let bounds = assets.parts.part_bounds()?.into_iter().map(|b| b.map(|b| b.map(V3::from))).collect();
        let parts: Vec<Vec<Connector>> =
            assets.connectors.iter().map(|cs| cs.iter().map(connector).collect()).collect();
        let reaches = store.chart.reaches();
        let reach = |c: &Connector| match c.kind {
            Kind::Profiled(p) => reaches[p],
            Kind::Fixed(_) => 0.0,
        };
        let span = |conns: &Vec<Connector>| {
            let radius = conns.iter().map(|c| origin(&c.frame).norm()).fold(0.0, f64::max);
            (radius, conns.iter().map(reach).fold(0.0, f64::max))
        };
        let spans = parts.iter().map(span).collect();
        Ok(Atlas { store: RefCell::new(store), parts, spans, bounds, moments, cached: Default::default() })
    }

    /// Collider box [lo, hi] of part type t in its frame (None: no collider).
    pub fn bounds(&self, t: u32) -> Option<[V3; 2]> {
        self.bounds.get(t as usize).copied().flatten()
    }

    /// Mass properties of part `part`, if known.
    pub fn moments(&self, part: u32) -> Option<&Moments> {
        self.moments.get(part as usize)?.as_ref()
    }

    /// Bound on the extent of the connector pairs of types ta, tb that needs no listing.
    pub fn span(&self, ta: u32, tb: u32) -> f64 {
        let [a, b] = [ta, tb].map(|t| self.spans.get(t as usize).copied().unwrap_or((0.0, 0.0)));
        a.0 + b.0 + a.1.min(b.1)
    }

    /// Connector pairs of the type pair ta <= tb (cached).
    pub fn regions(&self, ta: u32, tb: u32) -> Rc<Regions> {
        self.cached.borrow_mut().entry((ta, tb)).or_insert_with(|| Rc::new(self.collect_regions(ta, tb))).clone()
    }

    /// Connectors of type t.
    pub fn connectors(&self, t: u32) -> &[Connector] {
        self.parts.get(t as usize).map_or(&[], Vec::as_slice)
    }

    /// How connectors ca and cb pair, if their types mate.
    pub fn pairing(&self, ca: &Connector, cb: &Connector) -> Option<Pairing> {
        pairing(&self.store.borrow().chart, ca.kind, cb.kind)
    }

    /// Mating connector pairs of the type pair ta <= tb; none if a type has no moments.
    fn collect_regions(&self, ta: u32, tb: u32) -> Regions {
        let a = self.connectors(ta);
        let b = self.connectors(tb);
        let mut order: Vec<usize> = Vec::new();
        if self.moments(ta).is_some() && self.moments(tb).is_some() {
            order = (0..b.len()).collect();
            order.sort_by(|&p, &q| b[p].frame[(0, 3)].total_cmp(&b[q].frame[(0, 3)]));
        }
        let mut regions = Regions {
            ta,
            tb,
            runs: Vec::new(),
            mates: Vec::new(),
            bounds: [V3::repeat(f64::INFINITY), V3::repeat(f64::NEG_INFINITY)],
            sphere: (Vector4::w(), 0.0),
            reach: 0.0,
            extent: f64::MIN,
        };
        let mut store = self.store.borrow_mut();
        for (i, ca) in a.iter().enumerate() {
            let first = regions.mates.len();
            let mut largest: f64 = 0.0;
            for &k in &order {
                let cb = &b[k];
                let Some((domain, _)) = mate(&mut store, ca, cb) else {
                    continue;
                };
                let at = origin(&cb.frame);
                let reach = domain.map_or(0.0, |d| d.reach());
                regions.mates.push(Mate { b: k as u32, at, reach, region: OnceCell::new() });
                regions.bounds = [regions.bounds[0].inf(&at), regions.bounds[1].sup(&at)];
                largest = largest.max(reach);
                regions.extent = regions.extent.max(origin(&ca.frame).norm() + at.norm() + reach);
            }
            if regions.mates.len() > first {
                let range = first..regions.mates.len();
                regions.runs.push(Run { a: i, range, origin: ca.frame.column(3).into(), reach: largest });
                regions.reach = regions.reach.max(largest);
            }
        }
        let center = regions.runs.iter().map(|run| run.origin).sum::<Vector4<f64>>() / regions.runs.len().max(1) as f64;
        let distance = |run: &Run| (run.origin - center).xyz().norm();
        regions.sphere = (center, regions.runs.iter().map(distance).fold(0.0, f64::max));
        regions
    }

    /// Call `found` with regions whose connector origins come within `radius` plus the region's reach at
    /// the placement rel = P_a^-1 P_b, and that distance; by connector of ta, then by the x of tb's, sought within
    /// reach of the x of ta's origin in b's frame.
    pub fn near(&self, regions: &Regions, rel: &M4, radius: f64, mut found: impl FnMut(Rc<Region>, f64)) {
        let back = inverse(rel);
        // how far a point in b's frame lies outside the box of the mates' origins grown by `grow`
        let outside = |x: &V3, grow: f64| {
            (x - x.sup(&regions.bounds[0].add_scalar(-grow)).inf(&regions.bounds[1].add_scalar(grow))).norm()
        };
        if outside(&(back * regions.sphere.0).xyz(), radius + regions.reach) > regions.sphere.1 {
            return;
        }
        for run in &regions.runs {
            let xa = (back * run.origin).xyz();
            let reach = radius + run.reach;
            if outside(&xa, reach) > 0.0 {
                continue;
            }
            let slab = &regions.mates[run.range.clone()];
            let from = slab.partition_point(|m| m.at[0] < xa[0] - reach);
            for m in slab[from..].iter().take_while(|m| m.at[0] <= xa[0] + reach) {
                let distance = (xa - m.at).norm();
                if distance - m.reach < radius {
                    found(self.region(regions, run.a, m), distance);
                }
            }
        }
    }

    /// Region of connector i of ta and its mate m in `regions` (made once).
    fn region(&self, regions: &Regions, i: usize, m: &Mate) -> Rc<Region> {
        let made = m.region.get_or_init(|| {
            let (ca, cb) = (&self.connectors(regions.ta)[i], &self.connectors(regions.tb)[m.b as usize]);
            let (domain, reversed) = mate(&mut self.store.borrow_mut(), ca, cb).expect("a listed mate");
            Rc::new(Region {
                ca: ca.index,
                cb: cb.index,
                family: domain.map_or(FIXED, |d| d.family),
                reversed,
                fa: ca.frame,
                fb: cb.frame,
                domain,
                body: OnceCell::new(),
            })
        });
        made.clone()
    }

    /// Cells and groups of a region (decoded on first use).
    pub fn body<'a>(&self, region: &'a Region) -> &'a Body {
        region.body.get_or_init(|| {
            let mut cells = match region.domain {
                None => vec![[[0.0; 3]; 2]],
                Some(d) if d.family == BALL => self.ball_cells(&d),
                Some(d) => self.yaw_cells(&d),
            };
            let volume = |c: &Cell| (0..3).map(|k| c[1][k] - c[0][k] + 1.0).product::<f64>();
            cells.sort_by(|a, b| volume(b).total_cmp(&volume(a)));
            let groups = branches(&cells, region.family);
            let tree = (region.family == BALL).then(|| Tree::new(&cells));
            Body { cells, groups, tree }
        })
    }

    /// Stud, hinge, and axle cells [[flip, lo, slide], [flip, hi, slide]]: the cyclic yaw runs of the rows (flip,
    /// slide) (stud: flip 0 only), merged per flip into slide x yaw boxes where runs of consecutive slides overlap;
    /// runs inside a box are dropped.
    fn yaw_cells(&self, d: &Domain) -> Vec<Cell> {
        let flips = if d.family == STUD { 1 } else { 2 };
        let n = d.chunks / flips;
        let masks = self.store.borrow_mut().rows(d, 0..d.chunks);
        let slide = |s: usize| f64::from(s as i32 + d.slide);
        let mut out = Vec::new();
        for flip in 0..flips {
            // rows by line (flip, slide), else sample (flip, yaw, slide) at (360 flip + yaw) n + slide: alike if n = 1
            let pose = |s: usize, yaw: usize| (360 * flip + yaw) * n + s;
            let line = |s: usize| {
                if d.by_line || n == 1 {
                    masks[flip * n + s]
                } else {
                    row(|yaw| bit(&masks[pose(s, yaw) / 360], pose(s, yaw) % 360))
                }
            };
            let windows: Vec<Vec<(f64, f64)>> = (0..n).map(|s| yaw_windows(&line(s))).collect();
            let f = flip as f64;
            // `open` boxes end at the previous slide and extend while the next window overlap is the same
            let mut boxes: Vec<Cell> = Vec::new();
            let mut open: Vec<Cell> = Vec::new();
            for s in 1..n {
                let mut next = Vec::new();
                for &a in &windows[s - 1] {
                    for &b in &windows[s] {
                        for (lo, hi) in overlap(a, b) {
                            let first = match open.iter().position(|o| o[0][1] == lo && o[1][1] == hi) {
                                Some(k) => open.swap_remove(k)[0][2],
                                None => slide(s - 1),
                            };
                            next.push([[f, lo, first], [f, hi, slide(s)]]);
                        }
                    }
                }
                boxes.append(&mut open);
                open = next;
            }
            boxes.append(&mut open);
            // a window lies in a box when the box spans its slide and, at one of the turns -360, 0, 360 of the window
            // (windows may run past 360), holds the window, or when the box's window is a full turn
            let contains = |b: &Cell, y: f64, w: (f64, f64)| {
                let wide = b[1][1] - b[0][1] >= 360.0;
                let holds = || [-360.0, 0.0, 360.0].iter().any(|k| b[0][1] <= w.0 + k && w.1 + k <= b[1][1]);
                b[0][2] <= y && y <= b[1][2] && (wide || holds())
            };
            for (s, here) in windows.iter().enumerate() {
                for &w in here {
                    if !boxes.iter().any(|b| contains(b, slide(s), w)) {
                        out.push([[f, w.0, slide(s)], [f, w.1, slide(s)]]);
                    }
                }
            }
            out.append(&mut boxes);
        }
        out
    }

    /// Ball cells: boxes [x, x + dx] x [y, y + dy] x [lo, hi] of the runs around each grid point, merged.
    fn ball_cells(&self, d: &Domain) -> Vec<Cell> {
        let rows = self.store.borrow_mut().ball_rows(d, 0..360 * 360);
        let full = row(|_| true);
        if rows.iter().all(|r| *r == full) {
            return vec![[[0.0; 3], [360.0; 3]]];
        }
        let at = |x: usize, y: usize| rows[x % 360 * 360 + y % 360];
        // per line (y, dy) the samples s(x) of [x, x] (dy = 1: those of lines y and y + 1) and s(x) & s(x + 1) of
        // [x, x + 1]; a run's boxes are united along x as made, a box open while the next x can touch it
        let mut boxes = Vec::new();
        let mut made = 0;
        for y in 0..360 {
            for dy in 0..2 {
                let line = |x: usize| if dy == 0 { at(x, y) } else { and(at(x, y), at(x, y + 1)) };
                let mut open: Vec<Box3> = Vec::new();
                let mut here = line(0);
                for x in 0..360 {
                    let next = line(x + 1);
                    for (samples, dx) in [(here, 0), (and(here, next), 1)] {
                        for (lo, hi) in spans(&samples) {
                            made += 1;
                            let end = (x + dx) as i32;
                            match open.iter_mut().find(|b| b[0][2] == lo && b[1][2] == hi) {
                                Some(b) => b[1][0] = b[1][0].max(end),
                                None => open.push([[x as i32, y as i32, lo], [end, (y + dy) as i32, hi]]),
                            }
                        }
                    }
                    boxes.extend(open.extract_if(.., |b| b[1][0] < x as i32));
                    here = next;
                }
                boxes.append(&mut open);
            }
        }
        merge(&mut boxes, made);
        boxes.iter().map(|b| b.map(|c| c.map(f64::from))).collect()
    }
}
