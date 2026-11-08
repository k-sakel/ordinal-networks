//! Rust core of the `ordinal-networks` Python package.

pub mod counter;
pub mod encode;
pub mod perm;
pub mod systems;

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::PathBuf;

use counter::{count_block, SpillCounter};
use encode::{encode_windows_par, EncodeStats, Params};
use perm::{Ranking, FACT, MAX_M};

fn parse_ranking(s: &str) -> PyResult<Ranking> {
    Ranking::parse(s).ok_or_else(|| PyValueError::new_err(format!("unknown ranking '{s}' (use 'chronological' or 'amplitude')")))
}

fn split_ids(ids: &[u128]) -> (Vec<u64>, Vec<u64>) {
    (ids.iter().map(|v| (v >> 64) as u64).collect(), ids.iter().map(|v| *v as u64).collect())
}

fn join_ids(hi: &[u64], lo: &[u64]) -> PyResult<Vec<u128>> {
    if hi.len() != lo.len() {
        return Err(PyValueError::new_err("hi and lo must have the same length"));
    }
    Ok(hi.iter().zip(lo).map(|(h, l)| ((*h as u128) << 64) | *l as u128).collect())
}

fn pool(threads: usize) -> PyResult<rayon::ThreadPool> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Streaming counter of ordinal patterns and transitions.
///
/// Feed consecutive segments of a series with `feed(segment, n_windows)`;
/// window k of a segment starts at `segment[k*w]`. Transitions are counted
/// between consecutive windows, including across `feed` calls, unless
/// `break_chain()` is called in between.
#[pyclass(module = "ordinal_networks._core")]
struct Counter {
    params: Params,
    prev: u128,
    patterns: Option<SpillCounter<u128>>,
    transitions: Option<SpillCounter<(u128, u128)>>,
    stats: EncodeStats,
    n_windows: u64,
    pool: rayon::ThreadPool,
}

#[pymethods]
impl Counter {
    #[new]
    #[pyo3(signature = (m, tau=1, w=1, ranking="chronological", threads=0, memory_budget=4_294_967_296u64, spill_dir=None))]
    fn new(m: usize, tau: usize, w: usize, ranking: &str, threads: usize, memory_budget: u64, spill_dir: Option<PathBuf>) -> PyResult<Self> {
        let params = Params { m, tau, w, ranking: parse_ranking(ranking)? };
        params.validate().map_err(PyValueError::new_err)?;
        // hash-table allocation limits; spilling briefly needs ~2x this
        let pat_max = (memory_budget as f64 * 0.3 / 2.0) as usize;
        let tr_max = (memory_budget as f64 * 0.7 / 2.0) as usize;
        Ok(Counter {
            params,
            prev: 0,
            patterns: Some(SpillCounter::new(pat_max, spill_dir.clone(), "pat")),
            transitions: Some(SpillCounter::new(tr_max, spill_dir, "trn")),
            stats: EncodeStats::default(),
            n_windows: 0,
            pool: pool(threads)?,
        })
    }

    /// Encode and count `n_windows` windows of `segment`. If `return_ids`,
    /// returns the pattern IDs as (hi, lo) uint64 arrays (0 = skipped window).
    #[pyo3(signature = (segment, n_windows, return_ids=false))]
    fn feed<'py>(&mut self, py: Python<'py>, segment: PyReadonlyArray1<'py, f64>, n_windows: usize, return_ids: bool)
        -> PyResult<Option<(Bound<'py, PyArray1<u64>>, Bound<'py, PyArray1<u64>>)>> {
        let x = segment.as_slice()?;
        let p = self.params;
        if n_windows == 0 {
            return Ok(None);
        }
        let need = (n_windows - 1) * p.w + p.span();
        if x.len() < need {
            return Err(PyValueError::new_err(format!("segment too short: {} points for {} windows (need {})", x.len(), n_windows, need)));
        }
        let prev = self.prev;
        let pool = &self.pool;
        let (ids, st, pm, tm) = py.detach(|| {
            pool.install(|| {
                let mut ids = vec![0u128; n_windows];
                let st = encode_windows_par(x, 0, &mut ids, &p);
                let (pm, tm) = count_block(&ids, prev);
                (ids, st, pm, tm)
            })
        });
        let pc = self.patterns.as_mut().ok_or_else(|| PyValueError::new_err("counter already finalized"))?;
        let tc = self.transitions.as_mut().unwrap();
        let r: std::io::Result<()> = py.detach(|| pool.install(|| { pc.merge_map(pm)?; tc.merge_map(tm) }));
        r.map_err(|e| PyIOError::new_err(e.to_string()))?;
        self.stats.add(&st);
        self.n_windows += n_windows as u64;
        self.prev = *ids.last().unwrap();
        if return_ids {
            let (hi, lo) = split_ids(&ids);
            Ok(Some((hi.into_pyarray(py), lo.into_pyarray(py))))
        } else {
            Ok(None)
        }
    }

    /// Do not count a transition between the last fed window and the next one.
    fn break_chain(&mut self) {
        self.prev = 0;
    }

    /// Finish counting. Returns a dict with sorted pattern IDs (hi/lo), their
    /// counts, and the transition edge list in node indices (src, dst, count).
    fn finalize<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let pc = self.patterns.take().ok_or_else(|| PyValueError::new_err("counter already finalized"))?;
        let tc = self.transitions.take().unwrap();
        let pool = &self.pool;
        let res: std::io::Result<_> = py.detach(|| {
            pool.install(|| {
                use rayon::prelude::*;
                let pats = pc.finish()?;
                let keys: Vec<u128> = pats.iter().map(|(k, _)| *k).collect();
                let counts: Vec<u64> = pats.iter().map(|(_, c)| *c).collect();
                drop(pats);
                let trans = tc.finish()?;
                let n = trans.len();
                let mut src = vec![0u64; n];
                let mut dst = vec![0u64; n];
                let mut wgt = vec![0u64; n];
                src.par_iter_mut()
                    .zip(dst.par_iter_mut())
                    .zip(wgt.par_iter_mut())
                    .zip(trans.par_iter())
                    .for_each(|(((s, d), w), ((a, b), c))| {
                        *s = keys.binary_search(a).expect("transition source not in patterns") as u64;
                        *d = keys.binary_search(b).expect("transition target not in patterns") as u64;
                        *w = *c;
                    });
                drop(trans);
                Ok((keys, counts, src, dst, wgt))
            })
        });
        let (keys, counts, src, dst, wgt) = res.map_err(|e| PyIOError::new_err(e.to_string()))?;
        let (hi, lo) = split_ids(&keys);
        drop(keys);
        let d = PyDict::new(py);
        d.set_item("id_hi", hi.into_pyarray(py))?;
        d.set_item("id_lo", lo.into_pyarray(py))?;
        d.set_item("count", counts.into_pyarray(py))?;
        d.set_item("src", src.into_pyarray(py))?;
        d.set_item("dst", dst.into_pyarray(py))?;
        d.set_item("weight", wgt.into_pyarray(py))?;
        d.set_item("n_windows", self.n_windows)?;
        d.set_item("n_valid", self.stats.valid)?;
        d.set_item("n_nonfinite_windows", self.stats.nonfinite_windows)?;
        d.set_item("n_tied_windows", self.stats.tied_windows)?;
        d.set_item("n_tie_pairs", self.stats.tie_pairs)?;
        Ok(d)
    }

    /// Finish counting and stream the results to raw little-endian files in
    /// `dir` (constant memory): nodes_hi/nodes_lo/counts (u64, sorted by ID),
    /// edge_src (u64 node index), edge_dst_hi/edge_dst_lo (u64 pattern ID of
    /// the target, to be resolved to a node index), edge_weight (u64).
    fn finalize_to_disk<'py>(&mut self, py: Python<'py>, dir: PathBuf) -> PyResult<Bound<'py, PyDict>> {
        use std::io::{BufReader, BufWriter, Read, Write};
        let pc = self.patterns.take().ok_or_else(|| PyValueError::new_err("counter already finalized"))?;
        let tc = self.transitions.take().unwrap();
        let pool = &self.pool;
        let dir2 = dir.clone();
        let res: std::io::Result<(u64, u64)> = py.detach(|| {
            pool.install(|| {
                std::fs::create_dir_all(&dir2)?;
                let wr = |name: &str| -> std::io::Result<BufWriter<std::fs::File>> {
                    Ok(BufWriter::with_capacity(1 << 22, std::fs::File::create(dir2.join(name))?))
                };
                let (mut nh, mut nl, mut nc) = (wr("nodes_hi.bin")?, wr("nodes_lo.bin")?, wr("counts.bin")?);
                let mut v = 0u64;
                pc.finish_with(|k, c| {
                    nh.write_all(&((k >> 64) as u64).to_le_bytes())?;
                    nl.write_all(&(k as u64).to_le_bytes())?;
                    nc.write_all(&c.to_le_bytes())?;
                    v += 1;
                    Ok(())
                })?;
                nh.flush()?; nl.flush()?; nc.flush()?;
                drop((nh, nl, nc));
                // merge-join edges (sorted by source ID) with the sorted node list
                let rd = |name: &str| -> std::io::Result<BufReader<std::fs::File>> {
                    Ok(BufReader::with_capacity(1 << 20, std::fs::File::open(dir2.join(name))?))
                };
                let (mut rh, mut rl) = (rd("nodes_hi.bin")?, rd("nodes_lo.bin")?);
                let mut b8 = [0u8; 8];
                let mut next_node = |rh: &mut BufReader<std::fs::File>, rl: &mut BufReader<std::fs::File>| -> std::io::Result<u128> {
                    rh.read_exact(&mut b8)?;
                    let h = u64::from_le_bytes(b8);
                    rl.read_exact(&mut b8)?;
                    Ok(((h as u128) << 64) | u64::from_le_bytes(b8) as u128)
                };
                let (mut es, mut edh, mut edl, mut ew) = (wr("edge_src.bin")?, wr("edge_dst_hi.bin")?, wr("edge_dst_lo.bin")?, wr("edge_weight.bin")?);
                let mut idx: u64 = 0;
                let mut cur: u128 = if v > 0 { next_node(&mut rh, &mut rl)? } else { 0 };
                let mut e = 0u64;
                tc.finish_with(|(a, b), c| {
                    while cur < a {
                        cur = next_node(&mut rh, &mut rl)?;
                        idx += 1;
                    }
                    debug_assert!(cur == a);
                    es.write_all(&idx.to_le_bytes())?;
                    edh.write_all(&((b >> 64) as u64).to_le_bytes())?;
                    edl.write_all(&(b as u64).to_le_bytes())?;
                    ew.write_all(&c.to_le_bytes())?;
                    e += 1;
                    Ok(())
                })?;
                es.flush()?; edh.flush()?; edl.flush()?; ew.flush()?;
                Ok((v, e))
            })
        });
        let (v, e) = res.map_err(|e| PyIOError::new_err(e.to_string()))?;
        let d = PyDict::new(py);
        d.set_item("n_patterns", v)?;
        d.set_item("n_edges", e)?;
        d.set_item("n_windows", self.n_windows)?;
        d.set_item("n_valid", self.stats.valid)?;
        d.set_item("n_nonfinite_windows", self.stats.nonfinite_windows)?;
        d.set_item("n_tied_windows", self.stats.tied_windows)?;
        d.set_item("n_tie_pairs", self.stats.tie_pairs)?;
        Ok(d)
    }

    /// Number of windows of a series of length n.
    fn n_windows_for(&self, n: usize) -> usize {
        self.params.n_windows(n)
    }

    #[getter]
    fn span(&self) -> usize {
        self.params.span()
    }

    #[getter]
    fn n_spill_runs(&self) -> usize {
        self.patterns.as_ref().map(|p| p.n_runs()).unwrap_or(0) + self.transitions.as_ref().map(|p| p.n_runs()).unwrap_or(0)
    }
}

/// Pattern IDs of every window of `x` (0 = window with NaN/Inf).
/// Returns (hi, lo, stats_dict).
#[pyfunction]
#[pyo3(signature = (x, m, tau=1, w=1, ranking="chronological", threads=0))]
#[pyo3(name = "encode")]
fn encode_py<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, f64>, m: usize, tau: usize, w: usize, ranking: &str, threads: usize)
    -> PyResult<(Bound<'py, PyArray1<u64>>, Bound<'py, PyArray1<u64>>, Bound<'py, PyDict>)> {
    let p = Params { m, tau, w, ranking: parse_ranking(ranking)? };
    p.validate().map_err(PyValueError::new_err)?;
    let xs = x.as_slice()?;
    let nw = p.n_windows(xs.len());
    let tp = pool(threads)?;
    let (ids, st) = py.detach(|| {
        tp.install(|| {
            let mut ids = vec![0u128; nw];
            let st = encode_windows_par(xs, 0, &mut ids, &p);
            (ids, st)
        })
    });
    let (hi, lo) = split_ids(&ids);
    let d = PyDict::new(py);
    d.set_item("n_valid", st.valid)?;
    d.set_item("n_nonfinite_windows", st.nonfinite_windows)?;
    d.set_item("n_tied_windows", st.tied_windows)?;
    d.set_item("n_tie_pairs", st.tie_pairs)?;
    Ok((hi.into_pyarray(py), lo.into_pyarray(py), d))
}

/// Decode pattern IDs into permutations (n x m, uint8, 1-based) in the convention of `ranking`.
#[pyfunction]
#[pyo3(signature = (hi, lo, m, ranking="chronological"))]
fn decode<'py>(py: Python<'py>, hi: PyReadonlyArray1<'py, u64>, lo: PyReadonlyArray1<'py, u64>, m: usize, ranking: &str)
    -> PyResult<Bound<'py, PyArray2<u8>>> {
    let _ = parse_ranking(ranking)?;
    if m < 1 || m > MAX_M {
        return Err(PyValueError::new_err(format!("m must be in 1..={MAX_M}")));
    }
    let ids = join_ids(hi.as_slice()?, lo.as_slice()?)?;
    let mut out = Array2::<u8>::zeros((ids.len(), m));
    for (i, id) in ids.iter().enumerate() {
        if *id == 0 || *id > FACT[m] {
            return Err(PyValueError::new_err(format!("invalid pattern id {id} for m={m}")));
        }
        let mut buf = [0u8; MAX_M];
        perm::lehmer_unrank(id - 1, m, &mut buf);
        for k in 0..m {
            out[[i, k]] = buf[k];
        }
    }
    Ok(out.into_pyarray(py))
}

/// Pattern IDs of permutations (n x m, 1-based) given in their own convention.
#[pyfunction]
fn encode_permutations<'py>(py: Python<'py>, perms: PyReadonlyArray2<'py, i64>)
    -> PyResult<(Bound<'py, PyArray1<u64>>, Bound<'py, PyArray1<u64>>)> {
    let a = perms.as_array();
    let (n, m) = (a.nrows(), a.ncols());
    let mut ids = Vec::with_capacity(n);
    let mut buf = vec![0u8; m];
    for i in 0..n {
        for k in 0..m {
            let v = a[[i, k]];
            if v < 1 || v as usize > m {
                return Err(PyValueError::new_err(format!("row {i} is not a permutation of 1..{m}")));
            }
            buf[k] = v as u8;
        }
        ids.push(perm::id_from_perm(&buf).ok_or_else(|| PyValueError::new_err(format!("row {i} is not a permutation of 1..{m}")))?);
    }
    let (hi, lo) = split_ids(&ids);
    Ok((hi.into_pyarray(py), lo.into_pyarray(py)))
}

/// Pattern IDs (length m-1) of the first m-1 points of windows with pattern IDs (length m).
#[pyfunction]
#[pyo3(signature = (hi, lo, m, ranking="chronological"))]
fn prefix_ids<'py>(py: Python<'py>, hi: PyReadonlyArray1<'py, u64>, lo: PyReadonlyArray1<'py, u64>, m: usize, ranking: &str)
    -> PyResult<(Bound<'py, PyArray1<u64>>, Bound<'py, PyArray1<u64>>)> {
    let r = parse_ranking(ranking)?;
    if m < 2 || m > MAX_M {
        return Err(PyValueError::new_err(format!("m must be in 2..={MAX_M}")));
    }
    let ids = join_ids(hi.as_slice()?, lo.as_slice()?)?;
    for id in &ids {
        if *id == 0 || *id > FACT[m] {
            return Err(PyValueError::new_err(format!("invalid pattern id {id} for m={m}")));
        }
    }
    let out: Vec<u128> = ids.iter().map(|id| perm::prefix_id(*id, m, r)).collect();
    let (h, l) = split_ids(&out);
    Ok((h.into_pyarray(py), l.into_pyarray(py)))
}

/// m! as a Python int.
#[pyfunction]
fn factorial(m: usize) -> PyResult<u128> {
    if m > MAX_M {
        return Err(PyValueError::new_err(format!("m must be <= {MAX_M}")));
    }
    Ok(FACT[m])
}

// ---------------------------------------------------------------- systems --

#[pyfunction]
#[pyo3(signature = (name, params, x0, n, transient=0))]
fn iterate_map<'py>(py: Python<'py>, name: &str, params: Vec<f64>, x0: Vec<f64>, n: usize, transient: usize)
    -> PyResult<(Bound<'py, PyArray2<f64>>, Vec<f64>)> {
    let (d, np) = systems::map_spec(name).ok_or_else(|| PyValueError::new_err(format!("unknown map '{name}'")))?;
    if x0.len() != d || params.len() != np {
        return Err(PyValueError::new_err(format!("map '{name}' needs x0 of length {d} and {np} parameters")));
    }
    let name = name.to_string();
    let (v, last) = py.detach(|| systems::iterate_map(&name, &params, &x0, n, transient));
    let arr = Array2::from_shape_vec((n, d), v).unwrap();
    Ok((arr.into_pyarray(py), last))
}

#[pyfunction]
#[pyo3(signature = (name, params, x0, dt, n, every=1, transient=0))]
fn integrate_flow<'py>(py: Python<'py>, name: &str, params: Vec<f64>, x0: Vec<f64>, dt: f64, n: usize, every: usize, transient: usize)
    -> PyResult<(Bound<'py, PyArray2<f64>>, Vec<f64>)> {
    let (d, np) = systems::flow_spec(name).ok_or_else(|| PyValueError::new_err(format!("unknown flow '{name}'")))?;
    if x0.len() != d || params.len() != np {
        return Err(PyValueError::new_err(format!("flow '{name}' needs x0 of length {d} and {np} parameters")));
    }
    let name = name.to_string();
    let (v, last) = py.detach(|| systems::integrate(&name, &params, &x0, dt, n, every, transient));
    let arr = Array2::from_shape_vec((n, d), v).unwrap();
    Ok((arr.into_pyarray(py), last))
}

#[pyfunction]
#[pyo3(signature = (name, params, x0, dt, n, normal, offset, direction=1.0, transient=0, max_steps=u64::MAX))]
fn poincare_section<'py>(py: Python<'py>, name: &str, params: Vec<f64>, x0: Vec<f64>, dt: f64, n: usize, normal: Vec<f64>, offset: f64,
    direction: f64, transient: usize, max_steps: u64) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>, Vec<f64>)> {
    let (d, np) = systems::flow_spec(name).ok_or_else(|| PyValueError::new_err(format!("unknown flow '{name}'")))?;
    if x0.len() != d || params.len() != np || normal.len() != d {
        return Err(PyValueError::new_err(format!("flow '{name}' needs x0 and normal of length {d} and {np} parameters")));
    }
    let name = name.to_string();
    let (v, t, last) = py
        .detach(|| systems::poincare(&name, &params, &x0, dt, n, &normal, offset, direction, transient, max_steps))
        .map_err(PyValueError::new_err)?;
    let arr = Array2::from_shape_vec((n, d), v).unwrap();
    Ok((arr.into_pyarray(py), t.into_pyarray(py), last))
}

#[pyfunction]
#[pyo3(signature = (params, x0, dt, n, every=1, transient=0))]
fn mackey_glass<'py>(py: Python<'py>, params: Vec<f64>, x0: f64, dt: f64, n: usize, every: usize, transient: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if params.len() != 5 {
        return Err(PyValueError::new_err("mackey_glass needs 5 parameters (beta, gamma, n, tau, alpha)"));
    }
    let v = py.detach(|| systems::mackey_glass(&params, x0, dt, n, every, transient));
    Ok(v.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (indptr, indices, weights, start, steps, seed=0))]
fn random_walk<'py>(py: Python<'py>, indptr: PyReadonlyArray1<'py, i64>, indices: PyReadonlyArray1<'py, i64>, weights: PyReadonlyArray1<'py, f64>,
    start: usize, steps: usize, seed: u64) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let (ip, ix, wt) = (indptr.as_slice()?, indices.as_slice()?, weights.as_slice()?);
    if start + 1 >= ip.len() {
        return Err(PyValueError::new_err("start node out of range"));
    }
    let v = py.detach(|| systems::random_walk(ip, ix, wt, start, steps, seed));
    Ok(v.into_pyarray(py))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Counter>()?;
    m.add_function(wrap_pyfunction!(encode_py, m)?)?;
    m.add_function(wrap_pyfunction!(decode, m)?)?;
    m.add_function(wrap_pyfunction!(encode_permutations, m)?)?;
    m.add_function(wrap_pyfunction!(prefix_ids, m)?)?;
    m.add_function(wrap_pyfunction!(factorial, m)?)?;
    m.add_function(wrap_pyfunction!(iterate_map, m)?)?;
    m.add_function(wrap_pyfunction!(integrate_flow, m)?)?;
    m.add_function(wrap_pyfunction!(poincare_section, m)?)?;
    m.add_function(wrap_pyfunction!(mackey_glass, m)?)?;
    m.add_function(wrap_pyfunction!(random_walk, m)?)?;
    m.add("MAX_M", MAX_M)?;
    Ok(())
}
