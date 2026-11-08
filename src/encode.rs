//! Window segmentation and ordinal encoding.
//!
//! Window k (0-based) of a series starts at index `k*w` and contains the
//! points `x[k*w + j*tau]`, j = 0..m-1. When consecutive windows share points
//! (w a multiple of tau and w/tau < m) the ascending rank vector is updated
//! incrementally in O(m) per shifted point; otherwise it is recomputed.

use crate::perm::{id_from_ranks, Ranking, MAX_M};

#[derive(Clone, Copy, Debug)]
pub struct Params {
    pub m: usize,
    pub tau: usize,
    pub w: usize,
    pub ranking: Ranking,
}

impl Params {
    pub fn validate(&self) -> Result<(), String> {
        if self.m < 1 || self.m > MAX_M {
            return Err(format!("m must be in 1..={MAX_M} (got {})", self.m));
        }
        if self.tau < 1 {
            return Err("tau must be >= 1".into());
        }
        if self.w < 1 {
            return Err("w must be >= 1".into());
        }
        Ok(())
    }
    /// Number of series points spanned by one window.
    pub fn span(&self) -> usize {
        (self.m - 1) * self.tau + 1
    }
    /// Number of windows in a series of length n.
    pub fn n_windows(&self, n: usize) -> usize {
        let s = self.span();
        if n < s {
            0
        } else {
            (n - s) / self.w + 1
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct EncodeStats {
    /// windows mapped to a pattern
    pub valid: u64,
    /// windows skipped because they contain NaN/Inf
    pub nonfinite_windows: u64,
    /// valid windows containing at least one tie
    pub tied_windows: u64,
    /// total number of tied pairs over valid windows
    pub tie_pairs: u64,
}

impl EncodeStats {
    pub fn add(&mut self, o: &EncodeStats) {
        self.valid += o.valid;
        self.nonfinite_windows += o.nonfinite_windows;
        self.tied_windows += o.tied_windows;
        self.tie_pairs += o.tie_pairs;
    }
}

/// Full O(m^2) ranking of a window. Returns (n_nonfinite, tie_pairs).
#[inline]
fn full_rank(v: &[f64], r: &mut [u8]) -> (usize, u32) {
    let m = v.len();
    let mut nf = 0usize;
    for &a in v {
        if !a.is_finite() {
            nf += 1;
        }
    }
    if nf > 0 {
        return (nf, 0);
    }
    let mut ties = 0u32;
    for k in 0..m {
        let vk = v[k];
        let mut c = 1u8;
        for j in 0..k {
            // earlier point: smaller if <= (tie -> earlier is smaller)
            c += (v[j] <= vk) as u8;
            ties += (v[j] == vk) as u32;
        }
        for j in k + 1..m {
            c += (v[j] < vk) as u8;
        }
        r[k] = c;
    }
    (0, ties)
}

/// Encode `n` consecutive windows, the first one starting at `x[first]`.
/// `out[k]` receives the pattern ID of window k (0 if it contains non-finite
/// values). The slice `x` must contain every point the windows touch.
pub fn encode_windows(x: &[f64], first: usize, out: &mut [u128], p: &Params) -> EncodeStats {
    let m = p.m;
    let tau = p.tau;
    let w = p.w;
    let mut st = EncodeStats::default();
    let n = out.len();
    if n == 0 {
        return st;
    }
    let incremental = w % tau == 0 && w / tau < m;
    let s = if incremental { w / tau } else { 0 };

    let mut vals = [0f64; MAX_M];
    let mut ranks = [0u8; MAX_M];
    let mut nf: usize = 0; // non-finite points in current window
    let mut ties: u32 = 0;
    let mut ranks_ok = false;

    for k in 0..n {
        let start = first + k * w;
        if !incremental || k == 0 {
            for j in 0..m {
                vals[j] = x[start + j * tau];
            }
            let (f, t) = full_rank(&vals[..m], &mut ranks[..m]);
            nf = f;
            ties = t;
            ranks_ok = nf == 0;
        } else {
            for t in 0..s {
                // drop oldest point
                let u = vals[0];
                if !u.is_finite() {
                    nf -= 1;
                }
                if ranks_ok {
                    let ru = ranks[0];
                    for j in 1..m {
                        ranks[j] -= (ranks[j] > ru) as u8;
                        ties -= (vals[j] == u) as u32;
                    }
                }
                vals.copy_within(1..m, 0);
                ranks.copy_within(1..m, 0);
                // append newest point (latest in time: ties -> it is larger)
                let v = x[start + (m - s + t) * tau];
                vals[m - 1] = v;
                if !v.is_finite() {
                    nf += 1;
                    ranks_ok = false;
                } else if ranks_ok {
                    let mut rv = 1u8;
                    for j in 0..m - 1 {
                        let le = vals[j] <= v;
                        rv += le as u8;
                        ranks[j] += (!le) as u8;
                        ties += (vals[j] == v) as u32;
                    }
                    ranks[m - 1] = rv;
                }
            }
            if nf == 0 && !ranks_ok {
                let (_, t2) = full_rank(&vals[..m], &mut ranks[..m]);
                ties = t2;
                ranks_ok = true;
            }
        }
        if nf == 0 {
            out[k] = id_from_ranks(&ranks[..m], p.ranking);
            st.valid += 1;
            if ties > 0 {
                st.tied_windows += 1;
                st.tie_pairs += ties as u64;
            }
        } else {
            out[k] = 0;
            st.nonfinite_windows += 1;
        }
    }
    st
}

/// Parallel version: splits the windows into contiguous blocks.
pub fn encode_windows_par(x: &[f64], first: usize, out: &mut [u128], p: &Params) -> EncodeStats {
    use rayon::prelude::*;
    let n = out.len();
    let nt = rayon::current_num_threads().max(1);
    let block = ((n + nt * 4 - 1) / (nt * 4)).max(1 << 14);
    out.par_chunks_mut(block)
        .enumerate()
        .map(|(bi, o)| encode_windows(x, first + bi * block * p.w, o, p))
        .reduce(EncodeStats::default, |mut a, b| {
            a.add(&b);
            a
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perm::lehmer_rank;

    /// Reference: MATLAB-style stable argsort per window.
    fn reference(x: &[f64], p: &Params) -> Vec<u128> {
        let nw = p.n_windows(x.len());
        (0..nw)
            .map(|k| {
                let v: Vec<f64> = (0..p.m).map(|j| x[k * p.w + j * p.tau]).collect();
                if v.iter().any(|a| !a.is_finite()) {
                    return 0;
                }
                let mut idx: Vec<usize> = (0..p.m).collect();
                idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap()); // stable
                let bp: Vec<u8> = idx.iter().map(|&i| (i + 1) as u8).collect();
                match p.ranking {
                    Ranking::Chronological => lehmer_rank(&bp) + 1,
                    Ranking::Amplitude => {
                        let mut r = vec![0u8; p.m];
                        for (pos, &i) in idx.iter().enumerate() {
                            r[i] = (p.m - pos) as u8;
                        }
                        lehmer_rank(&r) + 1
                    }
                }
            })
            .collect()
    }

    fn lcg(n: usize, seed: u64, levels: u64) -> Vec<f64> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 33) % levels) as f64
            })
            .collect()
    }

    #[test]
    fn matches_reference() {
        for &levels in &[3u64, 1000, 1 << 30] {
            let mut x = lcg(3000, levels, levels);
            x[100] = f64::NAN;
            x[101] = f64::INFINITY;
            x[2000] = f64::NAN;
            for m in [1usize, 2, 3, 5, 8, 21, 25] {
                for tau in [1usize, 2, 3] {
                    for w in [1usize, 2, 3, 4, 7, m * tau, m * tau + 1] {
                        for ranking in [Ranking::Chronological, Ranking::Amplitude] {
                            let p = Params { m, tau, w, ranking };
                            let nw = p.n_windows(x.len());
                            let mut out = vec![0u128; nw];
                            encode_windows(&x, 0, &mut out, &p);
                            assert_eq!(out, reference(&x, &p), "m={m} tau={tau} w={w} {ranking:?} lv={levels}");
                        }
                    }
                }
            }
        }
    }
}
