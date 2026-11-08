//! Permutation <-> pattern-ID arithmetic.
//!
//! A pattern of length `m` is identified by its 1-based lexicographic rank in
//! the symmetric group S_m (so IDs run from 1 to m!, and 0 is reserved for
//! "no pattern", e.g. a window containing non-finite values). IDs are carried
//! as `u128`, which is exact for m <= 34 (34! < 2^128).

pub const MAX_M: usize = 34;

/// n! for n = 0..=34, exact in u128.
pub const FACT: [u128; MAX_M + 1] = {
    let mut f = [1u128; MAX_M + 1];
    let mut i = 1;
    while i <= MAX_M {
        f[i] = f[i - 1] * (i as u128);
        i += 1;
    }
    f
};

/// How a window is turned into a permutation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ranking {
    /// Bandt & Pompe chronological index ranking: pi = (p1..pm) such that
    /// x_{p1} < x_{p2} < ... < x_{pm} (ties: earlier point is smaller).
    /// This is MATLAB's `[~,s] = sort(window,'ascend')`.
    Chronological,
    /// Small's amplitude ranking: the j-th point of the window gets its rank in
    /// *descending* order of amplitude (1 = largest). Ties: earlier is smaller.
    Amplitude,
}

impl Ranking {
    pub fn parse(s: &str) -> Option<Ranking> {
        match s.to_ascii_lowercase().as_str() {
            "chronological" | "bp" | "bandt-pompe" | "index" => Some(Ranking::Chronological),
            "amplitude" | "small" | "rank" => Some(Ranking::Amplitude),
            _ => None,
        }
    }
}

/// 0-based lexicographic rank of a permutation of 1..=m (Horner form of the
/// Lehmer code). Uses a bitmask so it is O(m).
#[inline]
pub fn lehmer_rank(p: &[u8]) -> u128 {
    let m = p.len();
    let mut used: u64 = 0;
    if m <= 20 {
        let mut acc: u64 = 0;
        for k in 0..m {
            let v = (p[k] - 1) as u32;
            let c = v - (used & ((1u64 << v) - 1)).count_ones();
            acc = acc * ((m - k) as u64) + c as u64;
            used |= 1u64 << v;
        }
        acc as u128
    } else {
        let mut acc: u128 = 0;
        for k in 0..m {
            let v = (p[k] - 1) as u32;
            let c = v - (used & ((1u64 << v) - 1)).count_ones();
            acc = acc * ((m - k) as u128) + c as u128;
            used |= 1u64 << v;
        }
        acc
    }
}

/// Inverse of `lehmer_rank`: 0-based rank -> permutation of 1..=m.
pub fn lehmer_unrank(mut r: u128, m: usize, out: &mut [u8]) {
    let mut used: u64 = 0;
    for k in 0..m {
        let f = FACT[m - 1 - k];
        let c = (r / f) as u32;
        r %= f;
        // select the c-th (0-based) unused value
        let mut cnt = 0u32;
        let mut v = 0u32;
        loop {
            if used & (1u64 << v) == 0 {
                if cnt == c {
                    break;
                }
                cnt += 1;
            }
            v += 1;
        }
        used |= 1u64 << v;
        out[k] = (v + 1) as u8;
    }
}

/// Pattern ID (1-based) from the *ascending rank vector* of a window:
/// `r[k]` = 1 + number of window points that are "smaller" than point k
/// (ties: earlier point is smaller).
#[inline]
pub fn id_from_ranks(r: &[u8], ranking: Ranking) -> u128 {
    let m = r.len();
    let mut p = [0u8; MAX_M];
    match ranking {
        Ranking::Chronological => {
            for k in 0..m {
                p[(r[k] - 1) as usize] = (k + 1) as u8;
            }
        }
        Ranking::Amplitude => {
            for k in 0..m {
                p[k] = (m as u8 + 1) - r[k];
            }
        }
    }
    lehmer_rank(&p[..m]) + 1
}

/// Permutation (in the convention of `ranking`) from a 1-based pattern ID.
pub fn perm_from_id(id: u128, m: usize, ranking: Ranking, out: &mut [u8]) {
    let _ = ranking; // the ID is the rank of the permutation in its own convention
    lehmer_unrank(id - 1, m, out);
}

/// Ascending rank vector of the window (time order) from a pattern ID.
pub fn ranks_from_id(id: u128, m: usize, ranking: Ranking, out: &mut [u8]) {
    let mut p = [0u8; MAX_M];
    lehmer_unrank(id - 1, m, &mut p);
    match ranking {
        Ranking::Chronological => {
            for j in 0..m {
                out[(p[j] - 1) as usize] = (j + 1) as u8;
            }
        }
        Ranking::Amplitude => {
            for k in 0..m {
                out[k] = (m as u8 + 1) - p[k];
            }
        }
    }
}

/// ID of a permutation given in the convention of `ranking` (validated).
pub fn id_from_perm(p: &[u8]) -> Option<u128> {
    let m = p.len();
    if m == 0 || m > MAX_M {
        return None;
    }
    let mut seen: u64 = 0;
    for &v in p {
        if v == 0 || v as usize > m || seen & (1u64 << (v - 1)) != 0 {
            return None;
        }
        seen |= 1u64 << (v - 1);
    }
    Some(lehmer_rank(p) + 1)
}

/// ID of the pattern formed by the first `m-1` points of the window whose
/// pattern (length m) is `id` -- the "prefix" used for the outgrowth ratio.
pub fn prefix_id(id: u128, m: usize, ranking: Ranking) -> u128 {
    let mut r = [0u8; MAX_M];
    ranks_from_id(id, m, ranking, &mut r);
    let last = r[m - 1];
    let mut q = [0u8; MAX_M];
    for k in 0..m - 1 {
        q[k] = if r[k] > last { r[k] - 1 } else { r[k] };
    }
    id_from_ranks(&q[..m - 1], ranking)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_perms(m: usize) -> Vec<Vec<u8>> {
        // lexicographic order
        let mut out = vec![];
        let mut p: Vec<u8> = (1..=m as u8).collect();
        loop {
            out.push(p.clone());
            // next permutation
            let mut i = m as isize - 2;
            while i >= 0 && p[i as usize] >= p[i as usize + 1] {
                i -= 1;
            }
            if i < 0 {
                break;
            }
            let mut j = m - 1;
            while p[j] <= p[i as usize] {
                j -= 1;
            }
            p.swap(i as usize, j);
            p[i as usize + 1..].reverse();
        }
        out
    }

    #[test]
    fn rank_unrank_roundtrip() {
        for m in 1..=6 {
            for (k, p) in all_perms(m).iter().enumerate() {
                assert_eq!(lehmer_rank(p), k as u128);
                let mut q = vec![0u8; m];
                lehmer_unrank(k as u128, m, &mut q);
                assert_eq!(&q, p);
            }
        }
    }

    #[test]
    fn big_m() {
        let m = 34;
        let p: Vec<u8> = (1..=m as u8).rev().collect();
        assert_eq!(lehmer_rank(&p), FACT[m] - 1);
        let mut q = vec![0u8; m];
        lehmer_unrank(FACT[m] - 1, m, &mut q);
        assert_eq!(p, q);
    }
}
