//! BrickFlow Python bindings.

use std::collections::HashMap;
use std::io;
use std::path::PathBuf;

use nalgebra::Quaternion;
use numpy::ndarray::{Array2, Array3, Array4, ArrayD, IxDyn};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray3, PyReadonlyArrayDyn};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

mod atlas;
mod certify;
mod energy;
mod geometry;
mod pairs;
mod seat;
mod snap;

use geometry::{rotation, M3, M4, Q, V3};

/// Poses (n, 4, 4) from Python.
type Poses<'py> = PyReadonlyArray3<'py, f64>;

/// Part ids, table rows, or keys (n,) from Python.
type Ints<'py> = PyReadonlyArray1<'py, i64>;

/// Float64 array of any shape from Python.
type Floats<'py> = PyReadonlyArrayDyn<'py, f64>;

/// Part table from Python, by row: part id, mass, center of mass, second moment about it, symmetry orbit (w, x, y, z),
/// orbit size, and continuous symmetry axis, in LDU; shapes (n,), (n,), (n, 3), (n, 3, 3), (n, k, 4), (n,), (n, 3).
type Table<'py> = (Ints<'py>, Floats<'py>, Floats<'py>, Floats<'py>, Floats<'py>, Ints<'py>, Floats<'py>);

/// Core error as a Python one: invalid data (`bad`) as ValueError, file system errors as OSError.
fn io_err(error: io::Error) -> PyErr {
    match error.kind() {
        io::ErrorKind::Other => PyValueError::new_err(error.to_string()),
        _ => error.into(),
    }
}

/// ValueError for invalid input.
fn invalid(message: String) -> PyErr {
    PyValueError::new_err(message)
}

/// Row of the part table.
struct Row {
    id: usize,
    mass: f64,
    com: V3,
    second: M3,
    orbit: Vec<Q>,
    axis: V3,
}

/// Rows of a part table; every row has a part id, a finite mass > 0, and 1 to k symmetries.
fn table(table: &Table) -> PyResult<Vec<Row>> {
    let (ids, mass, com, second, orbit, size, axis) = table;
    let n = ids.as_array().len();
    let k = orbit.as_array().shape().get(1).copied().unwrap_or(0);
    let shapes = [
        (mass.as_array().shape().to_vec(), vec![n]),
        (com.as_array().shape().to_vec(), vec![n, 3]),
        (second.as_array().shape().to_vec(), vec![n, 3, 3]),
        (orbit.as_array().shape().to_vec(), vec![n, k, 4]),
        (size.as_array().shape().to_vec(), vec![n]),
        (axis.as_array().shape().to_vec(), vec![n, 3]),
    ];
    if shapes.iter().any(|(shape, want)| shape != want) {
        let need = "(n,), (n,), (n, 3), (n, 3, 3), (n, k, 4), (n,), (n, 3)";
        return Err(invalid(format!("part table: need columns of {n} rows, shaped {need}")));
    }
    let ids = ids.as_slice()?;
    let mass = mass.as_slice()?;
    let com = com.as_slice()?;
    let second = second.as_slice()?;
    let orbit = orbit.as_slice()?;
    let size = size.as_slice()?;
    let axis = axis.as_slice()?;
    let unit = |o: &[f64]| Q::from_quaternion(Quaternion::new(o[0], o[1], o[2], o[3]));
    let row = |p: usize| {
        let id = usize::try_from(ids[p]).ok();
        let count = usize::try_from(size[p]).ok().filter(|count| (1..=k).contains(count));
        let (Some(id), Some(count), true) = (id, count, mass[p] > 0.0 && mass[p].is_finite()) else {
            return Err(invalid(format!(
                "part table row {p}: need a part id >= 0, a mass > 0, and 1 to {k} symmetries"
            )));
        };
        Ok(Row {
            id,
            mass: mass[p],
            com: V3::from_row_slice(&com[3 * p..3 * p + 3]),
            second: M3::from_row_slice(&second[9 * p..9 * p + 9]),
            orbit: orbit[4 * k * p..4 * k * p + 4 * count].chunks(4).map(unit).collect(),
            axis: V3::from_row_slice(&axis[3 * p..3 * p + 3]),
        })
    };
    (0..n).map(row).collect()
}

/// Finite poses of an (n, 4, 4) array.
fn poses(p: &Poses, n: usize) -> PyResult<Vec<M4>> {
    let array = p.as_array();
    if array.dim() != (n, 4, 4) || array.iter().any(|v| !v.is_finite()) {
        return Err(invalid(format!("need finite poses ({n}, 4, 4)")));
    }
    Ok(array.outer_iter().map(|m| M4::from_fn(|i, j| m[[i, j]])).collect())
}

/// Whether a pose is rigid: an orthonormal rotation of determinant 1 and the last row (0, 0, 0, 1), within 1e-2.
fn rigid(p: &M4) -> bool {
    let r = rotation(p);
    let orthonormal = (r.transpose() * r - M3::identity()).amax() <= 1e-2;
    let last = (p.row(3) - M4::identity().row(3)).amax() <= 1e-2;
    orthonormal && last && r.determinant() > 0.0
}

/// Connection atlas of the physical-class representatives, cached as used.
#[pyclass(unsendable, module = "brickflow._core")]
struct Atlas {
    atlas: atlas::Atlas,
}

#[pymethods]
impl Atlas {
    /// Atlas of assets directory `root`, its parts' mass properties from the part table `table`.
    #[new]
    fn new(root: PathBuf, table: Table) -> PyResult<Self> {
        let mut moments: Vec<Option<atlas::Moments>> = Vec::new();
        for row in self::table(&table)? {
            if moments.len() <= row.id {
                moments.resize_with(row.id + 1, || None);
            }
            moments[row.id] = Some(atlas::Moments { mass: row.mass, c: row.com, s: row.second / row.mass });
        }
        let atlas = atlas::Atlas::open(&root, moments).map_err(io_err)?;
        Ok(Atlas { atlas })
    }

    /// E and dE/d(body twist [w, v]) (n, 6) of parts `ids` at `poses` (n, 4, 4).
    fn energy<'py>(&self, py: Python<'py>, ids: Vec<u32>, poses: Poses) -> PyResult<(f64, Bound<'py, PyArray2<f64>>)> {
        let poses = self::poses(&poses, ids.len())?;
        let (e, grad) = energy::energy(&self.atlas, &ids, &poses);
        Ok((e, Array2::from(grad).into_pyarray(py)))
    }

    /// Snapped poses (n, 4, 4) of parts `ids` at rigid `poses`.
    /// `depths(pairs (m, 2), poses (m, 2, 4, 4))` returns the pairs' penetration depths in LDU.
    fn snap<'py>(
        &self,
        py: Python<'py>,
        ids: Vec<u32>,
        poses: Poses,
        depths: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray3<f64>>> {
        let poses = self::poses(&poses, ids.len())?;
        if !poses.iter().all(rigid) {
            return Err(invalid("need rigid poses".to_string()));
        }
        // the first error of the callback is raised after the snap
        let mut error = None;
        let mut query = |q: &[(usize, usize, M4, M4)]| -> Vec<f64> {
            if error.is_none() {
                match measure(py, &depths, q) {
                    Ok(found) => return found,
                    Err(e) => error = Some(e),
                }
            }
            vec![0.0; q.len()]
        };
        let snapped = snap::snap(&self.atlas, &ids, &poses, &mut query);
        if let Some(e) = error {
            return Err(e);
        }
        let poses = snapped.map_err(io_err)?;
        Ok(Array3::from_shape_fn((ids.len(), 4, 4), |(k, i, j)| poses[k][(i, j)]).into_pyarray(py))
    }
}

/// Penetration depths of pairs (i, j) at poses (P_i, P_j) from the Python callback `depths`.
fn measure(py: Python, depths: &Bound<PyAny>, q: &[(usize, usize, M4, M4)]) -> PyResult<Vec<f64>> {
    let pairs = Array2::from_shape_fn((q.len(), 2), |(k, c)| [q[k].0, q[k].1][c] as i64);
    let at = Array4::from_shape_fn((q.len(), 2, 4, 4), |(k, side, i, j)| [&q[k].2, &q[k].3][side][(i, j)]);
    let found: Vec<f64> = depths.call1((pairs.into_pyarray(py), at.into_pyarray(py)))?.extract()?;
    if found.len() != q.len() {
        return Err(invalid(format!("depths: need {} depths, got {}", q.len(), found.len())));
    }
    Ok(found)
}

/// Parts of table rows `types`.
fn rows<'a>(parts: &'a [pairs::Part], types: &Ints) -> PyResult<Vec<&'a pairs::Part>> {
    let row = |&t: &i64| {
        let part = usize::try_from(t).ok().and_then(|t| parts.get(t));
        part.ok_or_else(|| invalid(format!("no part table row {t}")))
    };
    types.as_array().iter().map(row).collect()
}

/// Centers of mass and rotations of parts at finite poses (n, 4, 4) at their LDraw origins.
fn frames(parts: &[&pairs::Part], poses: &Poses) -> PyResult<(Vec<V3>, Vec<Q>)> {
    Ok(pairs::frames(parts, &self::poses(poses, parts.len())?, 1.0))
}

/// Part types for comparing assemblies, each of mass 1: the rows of a part table.
#[pyclass(frozen, module = "brickflow._core")]
struct Parts(Vec<pairs::Part>);

#[pymethods]
impl Parts {
    #[new]
    fn new(table: Table) -> PyResult<Self> {
        let part = |row: Row| pairs::Part::new(1.0, row.com, row.second * (1.0 / row.mass), &row.orbit, row.axis, 1.0);
        Ok(Parts(self::table(&table)?.into_iter().map(part).collect()))
    }

    /// Least mass movement of parts `types` (table rows) at `target` onto `poses` (n, 4, 4) at their LDraw origins,
    /// each part weighing 1; parts of equal `keys` (n,) are interchangeable. Dict of cost, bound (a lower bound on
    /// cost), certified, per slot trans2, rot2, and order (the part moved onto it), and rotation (w, x, y, z) and
    /// translation of the parts. With certified true, cost - bound <= certify max(cost, n): per part, MM^2 - bound^2
    /// <= certify max(MM^2, 1 LDU^2); false when the branch and bound over rotations ran out of evaluations first.
    fn movement<'py>(
        &self,
        py: Python<'py>,
        types: Ints,
        keys: Ints,
        poses: Poses,
        target: Poses,
        certify: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let parts = rows(&self.0, &types)?;
        let (x0, q0) = frames(&parts, &poses)?;
        let (x1, q1) = frames(&parts, &target)?;
        let need = "need parts, and keys (>= 0) that each hold one part type";
        if parts.is_empty() || keys.as_array().len() != parts.len() {
            return Err(invalid(need.to_string()));
        }
        let mut type_of = HashMap::new();
        for (&k, &t) in keys.as_array().iter().zip(types.as_array()) {
            let first = *type_of.entry(k).or_insert(t);
            if k < 0 || first != t {
                return Err(invalid(need.to_string()));
            }
        }
        if !(certify > 0.0 && certify.is_finite()) {
            return Err(invalid("certify: need a gap > 0".to_string()));
        }
        let keys: Vec<usize> = keys.as_array().iter().map(|&k| k as usize).collect();
        let m = py.detach(|| pairs::movement(&parts, &keys, &x0, &q0, &x1, &q1, certify));
        let d = PyDict::new(py);
        let r = m.rotation;
        let t = m.translation;
        d.set_item("cost", m.cost)?;
        d.set_item("bound", m.bound)?;
        d.set_item("certified", m.certified)?;
        d.set_item("trans2", m.trans2.into_pyarray(py))?;
        d.set_item("rot2", m.rot2.into_pyarray(py))?;
        d.set_item("order", m.order.iter().map(|&i| i as i64).collect::<Vec<i64>>().into_pyarray(py))?;
        d.set_item("rotation", vec![r.w, r.i, r.j, r.k].into_pyarray(py))?;
        d.set_item("translation", vec![t.x, t.y, t.z].into_pyarray(py))?;
        Ok(d)
    }

    /// Per part `types` (table rows) the squared distance d^2 per unit mass from `target` to `poses` (n, 4, 4) at their
    /// LDraw origins, over its symmetries.
    fn distance2<'py>(
        &self,
        py: Python<'py>,
        types: Ints,
        poses: Poses,
        target: Poses,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let parts = rows(&self.0, &types)?;
        let (x0, q0) = frames(&parts, &poses)?;
        let (x1, q1) = frames(&parts, &target)?;
        let d2: Vec<f64> = (0..parts.len()).map(|i| parts[i].distance2(&x0[i], &q0[i], &x1[i], &q1[i])).collect();
        Ok(d2.into_pyarray(py))
    }
}

/// Training pairs: `assemblies` (root, atlas assets, the part id of each physical class; None: given parts
/// only) coupled to noise; the parts by part id from the part table `table`.
#[pyclass(frozen, module = "brickflow._core")]
struct Pairs {
    pairs: pairs::Pairs,
    pool: rayon::ThreadPool,
}

/// Token outputs (trans_t, quat_t, trans_vel, rot_tangent) of samples as arrays of shape (lead..., width).
fn fields<'py>(py: Python<'py>, samples: &[&[[f32; 13]]], lead: &[usize]) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for (name, from, to) in [("trans_t", 0, 3), ("quat_t", 3, 7), ("trans_vel", 7, 10), ("rot_tangent", 10, 13)] {
        let mut data = Vec::with_capacity(lead.iter().product::<usize>() * (to - from));
        for token in samples.iter().copied().flatten() {
            data.extend_from_slice(&token[from..to]);
        }
        let shape = IxDyn(&[lead, &[to - from]].concat());
        let array = ArrayD::from_shape_vec(shape, data).map_err(|e| invalid(format!("{name}: {e}")))?;
        d.set_item(name, array.into_pyarray(py))?;
    }
    Ok(d)
}

#[pymethods]
impl Pairs {
    #[new]
    fn new(
        assemblies: Option<(PathBuf, PathBuf, Vec<i32>)>,
        table: Table,
        seed: u64,
        scale: f64,
        noise: Vec<f64>,
        time: (f64, f64),
        threads: usize,
    ) -> PyResult<Self> {
        let mut parts: Vec<Option<pairs::Part>> = Vec::new();
        for row in self::table(&table)? {
            if parts.len() <= row.id {
                parts.resize_with(row.id + 1, || None);
            }
            parts[row.id] = Some(pairs::Part::new(row.mass, row.com, row.second, &row.orbit, row.axis, scale));
        }
        let assemblies =
            assemblies.as_ref().map(|(root, assets, classes)| (root.as_path(), assets.as_path(), classes.clone()));
        let pairs = pairs::Pairs::new(assemblies, parts, seed, scale, noise, time).map_err(io_err)?;
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads.max(1)).build();
        let pool = pool.map_err(|e| PyRuntimeError::new_err(format!("cannot start {threads} threads: {e}")))?;
        Ok(Pairs { pairs, pool })
    }

    /// Stream position -> index in [0, span) by a random affine bijection per pass.
    fn shuffled(&self, position: u64, span: u64) -> PyResult<u64> {
        if span == 0 {
            return Err(invalid("need a span > 0".to_string()));
        }
        Ok(self.pairs.shuffled(position, span))
    }

    /// Rows of stream positions (2r, 2r + 1): a dict of types, lengths, t, and the per-token fields.
    fn batch<'py>(&self, py: Python<'py>, positions: Vec<u64>) -> PyResult<Bound<'py, PyDict>> {
        if positions.len() % 2 == 1 {
            return Err(invalid("positions come in pairs".to_string()));
        }
        let r = positions.len() / 2;
        let make = |i: usize| self.pairs.row(positions[2 * i], positions[2 * i + 1]);
        let prepare = || (0..r).into_par_iter().map(make).collect::<io::Result<Vec<pairs::Row>>>();
        let rows = py.detach(|| self.pool.install(prepare)).map_err(io_err)?;
        let samples: Vec<&[[f32; 13]]> = rows.iter().map(|row| row.tokens.as_slice()).collect();
        let d = fields(py, &samples, &[r, pairs::ROW])?;
        d.set_item("types", Array2::from_shape_fn((r, pairs::ROW), |(i, j)| rows[i].types[j]).into_pyarray(py))?;
        d.set_item("lengths", Array2::from_shape_fn((r, 2), |(i, j)| rows[i].lengths[j]).into_pyarray(py))?;
        d.set_item("t", rows.iter().map(|row| row.t).collect::<Vec<f32>>().into_pyarray(py))?;
        Ok(d)
    }

    /// (time, per-slot fields, part in each slot) of given parts `types` (part ids) with `keys` at `poses`, coupled as
    /// the sample at stream `position` at `time` (else drawn).
    #[pyo3(signature = (position, types, keys, poses, time=None))]
    fn couple<'py>(
        &self,
        py: Python<'py>,
        position: u64,
        types: Vec<usize>,
        keys: Vec<usize>,
        poses: Poses,
        time: Option<f64>,
    ) -> PyResult<(f64, Bound<'py, PyDict>, Vec<usize>)> {
        if keys.len() != types.len() {
            return Err(invalid(format!("need {} keys, got {}", types.len(), keys.len())));
        }
        let poses = self::poses(&poses, types.len())?;
        let s = self.pairs.sample(position, &types, &keys, &poses, time).map_err(io_err)?;
        Ok((s.t, fields(py, &[s.out.as_slice()], &[types.len()])?, s.order))
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Atlas>()?;
    m.add_class::<Pairs>()?;
    m.add_class::<Parts>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))
}
