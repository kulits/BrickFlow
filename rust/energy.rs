//! Connectivity energy and its gradient.

use nalgebra::DMatrix;

use crate::atlas::{join, top, Atlas};
use crate::geometry::{origin, rotation, M4};
use crate::seat::{self, Hyp, Pair, Seat, Start, SEATED};

const RHO: f64 = 1e-8; // resistance floor of a seated hypothesis, LDU^2
const CAP: f64 = 16.0; // the ground conducts 1 / (CAP n) between every two parts
const INNER: f64 = 12.0; // taper of the connector-origin gap, LDU: weight 1 up to INNER, 0 from RADIUS
const RADIUS: f64 = 16.0;
const DOMINATED: f64 = 0.2; // a hypothesis below this times its pair's best conductance at its start is not searched

/// Hypothesis in the network: link (parts a < b), conductance g = w / rq, taper w, dw/dgap, q'(c), and
/// rq = RHO + q(c).
struct Term {
    a: usize,
    b: usize,
    g: f64,
    w: f64,
    dw: f64,
    dq: f64,
    rq: f64,
}

impl Term {
    fn new(pair: &Pair, gap: f64, c: f64) -> Term {
        let (w, dw) = taper(gap);
        let (q, dq) = phi((c - SEATED) / SEATED, 1.0);
        let rq = RHO + SEATED * q;
        Term { a: pair.a.min(pair.b), b: pair.a.max(pair.b), g: w / rq, w, dw, dq, rq }
    }
}

/// Taper w of a connector-origin gap, and dw/dgap.
fn taper(gap: f64) -> (f64, f64) {
    let s = ((RADIUS - gap) / (RADIUS - INNER)).clamp(0.0, 1.0);
    (s * s * s * (s * (6.0 * s - 15.0) + 10.0), -30.0 * s * s * (1.0 - s) * (1.0 - s) / (RADIUS - INNER))
}

/// E and dE/d(body twist [w, v]) (n, 6) of parts `ids` at `poses` (LDU, at each part's LDraw origin).
pub fn energy(atlas: &Atlas, ids: &[u32], poses: &[M4]) -> (f64, Vec<[f64; 6]>) {
    let n = ids.len();
    let mut grad = vec![[0.0; 6]; n];
    let (pairs, hyps) = seat::hypotheses(atlas, ids, poses, RADIUS, |_, _| true);
    let mut starts: Vec<Option<Start>> = hyps.iter().map(|_| None).collect();
    if seated(atlas, &pairs, &hyps, &mut starts, n) {
        return (0.0, grad);
    }
    let started = hyps.iter().zip(starts).map(|(h, start)| start.unwrap_or_else(|| seat::start(atlas, &pairs, h)));
    let seats = search(&hyps, &started.collect::<Vec<Start>>());
    let terms: Vec<Term> = hyps.iter().zip(&seats).map(|(h, s)| Term::new(&pairs[h.pair], h.gap, s.d2)).collect();
    let (value, slopes) = network(n, &terms);
    for ((h, s), [dc, dgap]) in hyps.iter().zip(&seats).zip(slopes) {
        let p = &pairs[h.pair];
        for k in 0..6 {
            grad[p.a][k] += dc * s.grad[0][k];
            grad[p.b][k] += dc * s.grad[1][k];
        }
        if dgap != 0.0 {
            let r = &p.regions[h.region];
            let e = (origin(&(poses[p.a] * r.fa)) - origin(&(poses[p.b] * r.fb))).normalize();
            for (part, frame, sign) in [(p.a, &r.fa, dgap), (p.b, &r.fb, -dgap)] {
                let l = rotation(&poses[part]).transpose() * e;
                let w = origin(frame).cross(&l);
                for k in 0..3 {
                    grad[part][k] += sign * w[k];
                    grad[part][k + 3] += sign * l[k];
                }
            }
        }
    }
    (value, grad)
}

/// Seats of hypotheses from their starts, by the energy's local search. A pair's hypotheses go in order of their
/// conductance at the start; one whose start point conducts less than DOMINATED times the best of its pair so far is
/// not searched (its share of the pair's 4-norm is below about DOMINATED^4).
fn search(hyps: &[Hyp], starts: &[Start]) -> Vec<Seat> {
    let conductance = |h: usize, c: f64| taper(hyps[h].gap).0 / c.max(SEATED);
    let mut order: Vec<usize> = (0..hyps.len()).collect();
    let at_start = |h: &usize| conductance(*h, starts[*h].seat.d2);
    order.sort_by(|x, y| hyps[*x].pair.cmp(&hyps[*y].pair).then(at_start(y).total_cmp(&at_start(x))));
    let mut seats: Vec<Seat> = starts.iter().map(|s| s.seat).collect();
    let mut best = (usize::MAX, 0.0); // a pair and its best conductance so far
    for h in order {
        if hyps[h].pair != best.0 {
            best = (hyps[h].pair, 0.0);
        }
        // D^2 from which conductance stays below DOMINATED times the best (rq >= D^2 - SEATED / 2 there)
        let ceiling = if best.1 > 0.0 { taper(hyps[h].gap).0 / (DOMINATED * best.1) + SEATED } else { f64::INFINITY };
        seats[h] = starts[h].search(true, ceiling);
        best.1 = f64::max(best.1, conductance(h, seats[h].d2));
    }
    seats
}

/// Whether hypotheses seated at the actual joint coordinates connect all n parts (always for n < 2).
fn seated<'a>(atlas: &'a Atlas, pairs: &'a [Pair], hyps: &[Hyp], starts: &mut [Option<Start<'a>>], n: usize) -> bool {
    let mut order: Vec<usize> = (0..hyps.len()).filter(|&h| hyps[h].gap <= INNER).collect();
    order.sort_by(|&x, &y| hyps[x].gap.total_cmp(&hyps[y].gap));
    let mut root: Vec<usize> = (0..n).collect();
    let mut joined = 0;
    for h in order {
        let Pair { a, b, .. } = pairs[hyps[h].pair];
        if top(&root, a) == top(&root, b) {
            continue;
        }
        let start = starts[h].insert(seat::start(atlas, pairs, &hyps[h]));
        if start.seat.d2 <= SEATED {
            join(&mut root, a, b);
            joined += 1;
        }
    }
    joined + 1 >= n
}

/// phi_s(x) and phi_s'(x).
fn phi(x: f64, s: f64) -> (f64, f64) {
    let d = x * x + s * s;
    if x > 0.0 {
        (x * x * x / d, x * x * (x * x + 3.0 * s * s) / (d * d))
    } else {
        (0.0, 0.0)
    }
}

/// Resistance kernel K (n x n) of links `link` (upper triangle) and the ground: Omega_ab = K_aa + K_bb - 2 K_ab.
/// Star-mesh elimination of nodes 0..n-2 adds only positive conductances; with its pivots d and factor l (I - l unit
/// lower), K = Y^T D^-1 Y for Y = (I - l)^-1, node n - 1 the reference (its row and column 0).
fn kernel(link: &DMatrix<f64>) -> DMatrix<f64> {
    let n = link.nrows();
    let r = n - 1;
    let ground = 1.0 / (CAP * n as f64);
    // column k below the diagonal: node k's conductances to nodes k + 1..n, then l_ik = w_ik / d_k
    let mut w = DMatrix::from_fn(n, n, |i, j| if i > j { ground + link[(j, i)] } else { 0.0 });
    let mut d = vec![0.0; r];
    let s = w.as_mut_slice();
    for k in 0..r {
        let (done, rest) = s.split_at_mut((k + 1) * n);
        let star = &mut done[k * n..];
        d[k] = star[k + 1..].iter().sum();
        for l in &mut star[k + 1..] {
            *l /= d[k];
        }
        // the mesh of node k's star: w_ij += w_ik w_jk / d_k
        for j in k + 1..n {
            let w_jk = star[j] * d[k];
            let column = &mut rest[(j - k - 1) * n..(j - k) * n];
            for i in j + 1..n {
                column[i] += star[i] * w_jk;
            }
        }
    }
    // X = D^-1/2 Y, column a by forward substitution (zero above row a, and in the reference's column)
    let mut x = DMatrix::zeros(r, n);
    for a in 0..r {
        let y = &mut x.as_mut_slice()[a * r..(a + 1) * r];
        y[a] = 1.0;
        for k in a..r {
            let l = &s[k * n..k * n + r];
            for i in k + 1..r {
                y[i] += l[i] * y[k];
            }
            y[k] /= d[k].sqrt();
        }
    }
    x.transpose() * &x
}

/// E of n parts and (dE/dc, dE/dgap) of every hypothesis, through the link conductances G (upper triangle), each the
/// 4-norm of its pair's terms, and the Laplacian L_psi of the weights dE/dOmega_ab.
fn network(n: usize, terms: &[Term]) -> (f64, Vec<[f64; 2]>) {
    let mut link: DMatrix<f64> = DMatrix::zeros(n, n);
    for t in terms {
        link[(t.a, t.b)] += t.g.powi(4);
    }
    link.apply(|g| *g = g.sqrt().sqrt());
    let k = kernel(&link);
    let omega0 = (n - 1) as f64 * RHO;
    let scale = 1.0 / (n - 1) as f64;
    let mut value = 0.0;
    let mut psi = DMatrix::zeros(n, n);
    for c in 1..n {
        for a in 0..c {
            let (phi_value, dphi) = phi(k[(a, a)] + k[(c, c)] - 2.0 * k[(a, c)] - omega0, omega0);
            value += scale * phi_value;
            psi[(a, c)] = -scale * dphi;
            psi[(c, a)] = -scale * dphi;
        }
    }
    for a in 0..n {
        psi[(a, a)] = -psi.column(a).sum();
    }
    // dE/dG_ij = -v^T L_psi v for v = K (e_i - e_j): from columns i and j of K and of L_psi K
    let m = psi * &k;
    let slope = |t: &Term| {
        let g_link = link[(t.a, t.b)];
        let [ka, kb, ma, mb] = [(&k, t.a), (&k, t.b), (&m, t.a), (&m, t.b)].map(|(x, j)| x.column(j).data.into_slice());
        let v_psi_v: f64 = ka.iter().zip(kb).zip(ma.iter().zip(mb)).map(|((p, q), (s, u))| (p - q) * (s - u)).sum();
        let de_dg = if g_link == 0.0 { 0.0 } else { -v_psi_v * (t.g / g_link).powi(3) };
        [-de_dg * t.w * t.dq / (t.rq * t.rq), de_dg * t.dw / t.rq]
    };
    (value, terms.iter().map(slope).collect())
}
