//! Frequency counting of patterns and transitions, with optional spilling of
//! sorted runs to disk when the in-memory hash tables exceed a budget.

use rustc_hash::FxHashMap;
use std::collections::BinaryHeap;
use std::cmp::Reverse;
use std::fs::File;
use std::hash::Hash;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::PathBuf;

pub trait Key: Copy + Eq + Hash + Ord + Send + Sync {
    const SIZE: usize;
    fn put(&self, buf: &mut Vec<u8>);
    fn get(buf: &[u8]) -> Self;
}

impl Key for u128 {
    const SIZE: usize = 16;
    fn put(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.to_le_bytes());
    }
    fn get(b: &[u8]) -> Self {
        u128::from_le_bytes(b[..16].try_into().unwrap())
    }
}

impl Key for (u128, u128) {
    const SIZE: usize = 32;
    fn put(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.0.to_le_bytes());
        buf.extend_from_slice(&self.1.to_le_bytes());
    }
    fn get(b: &[u8]) -> Self {
        (
            u128::from_le_bytes(b[..16].try_into().unwrap()),
            u128::from_le_bytes(b[16..32].try_into().unwrap()),
        )
    }
}

pub struct SpillCounter<K: Key> {
    pub map: FxHashMap<K, u64>,
    max_bytes: usize,
    dir: Option<PathBuf>,
    tag: String,
    runs: Vec<PathBuf>,
}

impl<K: Key> SpillCounter<K> {
    /// Bytes per hash-table bucket (entry + control byte).
    pub const BUCKET: usize = std::mem::size_of::<(K, u64)>() + 1;

    /// `max_bytes`: allowed size of the hash table allocation. Spilling sorts
    /// the table into a vector of about the same size, so the peak is ~2x.
    pub fn new(max_bytes: usize, dir: Option<PathBuf>, tag: &str) -> Self {
        SpillCounter {
            map: FxHashMap::default(),
            max_bytes: max_bytes.max(1 << 20),
            dir,
            tag: tag.to_string(),
            runs: vec![],
        }
    }

    pub fn n_runs(&self) -> usize {
        self.runs.len()
    }

    fn table_bytes(cap: usize) -> usize {
        cap * Self::BUCKET
    }

    pub fn merge_map(&mut self, other: FxHashMap<K, u64>) -> std::io::Result<()> {
        let can_spill = self.dir.is_some();
        // if absorbing `other` could force the table past the budget, spill first
        if can_spill && !self.map.is_empty() {
            let need = (self.map.len() + other.len()) * 8 / 7 + 1;
            if Self::table_bytes(need.next_power_of_two()) > self.max_bytes {
                self.spill()?;
            }
        }
        if self.map.is_empty() {
            self.map = other;
        } else {
            for (k, c) in other {
                *self.map.entry(k).or_insert(0) += c;
            }
        }
        if can_spill && Self::table_bytes(self.map.capacity()) > self.max_bytes {
            self.spill()?;
        }
        Ok(())
    }

    fn sorted_take(&mut self) -> Vec<(K, u64)> {
        let map = std::mem::take(&mut self.map);
        let mut v: Vec<(K, u64)> = map.into_iter().collect();
        use rayon::prelude::*;
        v.par_sort_unstable_by(|a, b| a.0.cmp(&b.0));
        v
    }

    fn spill(&mut self) -> std::io::Result<()> {
        if self.map.is_empty() {
            return Ok(());
        }
        let v = self.sorted_take();
        let dir = self.dir.as_ref().unwrap();
        std::fs::create_dir_all(dir)?;
        let path = dir.join(format!(
            "ordnet_{}_{}_{:p}_{}.run",
            self.tag,
            std::process::id(),
            self as *const _,
            self.runs.len()
        ));
        let mut wtr = BufWriter::with_capacity(1 << 22, File::create(&path)?);
        let mut buf = Vec::with_capacity(K::SIZE + 8);
        for (k, c) in v {
            buf.clear();
            k.put(&mut buf);
            buf.extend_from_slice(&c.to_le_bytes());
            wtr.write_all(&buf)?;
        }
        wtr.flush()?;
        self.runs.push(path);
        Ok(())
    }

    /// Stream every (key, total count) in ascending key order into `f`.
    pub fn finish_with<F: FnMut(K, u64) -> std::io::Result<()>>(mut self, mut f: F) -> std::io::Result<()> {
        if self.runs.is_empty() {
            for (k, c) in self.sorted_take() {
                f(k, c)?;
            }
            return Ok(());
        }
        self.spill()?;
        let rec = K::SIZE + 8;
        let mut readers: Vec<BufReader<File>> = self
            .runs
            .iter()
            .map(|p| File::open(p).map(|f| BufReader::with_capacity(1 << 20, f)))
            .collect::<Result<_, _>>()?;
        let mut buf = vec![0u8; rec];
        let next = |r: &mut BufReader<File>, buf: &mut Vec<u8>| -> std::io::Result<Option<(K, u64)>> {
            match r.read_exact(buf) {
                Ok(()) => Ok(Some((K::get(&buf[..K::SIZE]), u64::from_le_bytes(buf[K::SIZE..].try_into().unwrap())))),
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Ok(None),
                Err(e) => Err(e),
            }
        };
        let mut heap = BinaryHeap::new();
        for (i, r) in readers.iter_mut().enumerate() {
            if let Some((k, c)) = next(r, &mut buf)? {
                heap.push(Reverse((k, i, c)));
            }
        }
        let mut cur: Option<(K, u64)> = None;
        while let Some(Reverse((k, i, c))) = heap.pop() {
            match cur.as_mut() {
                Some(last) if last.0 == k => last.1 += c,
                _ => {
                    if let Some((pk, pc)) = cur.take() {
                        f(pk, pc)?;
                    }
                    cur = Some((k, c));
                }
            }
            if let Some((k2, c2)) = next(&mut readers[i], &mut buf)? {
                heap.push(Reverse((k2, i, c2)));
            }
        }
        if let Some((pk, pc)) = cur {
            f(pk, pc)?;
        }
        drop(readers);
        for p in self.runs.drain(..) {
            let _ = std::fs::remove_file(p);
        }
        Ok(())
    }

    /// Sorted (key, count) list of everything counted.
    pub fn finish(self) -> std::io::Result<Vec<(K, u64)>> {
        let mut out = Vec::new();
        self.finish_with(|k, c| {
            out.push((k, c));
            Ok(())
        })?;
        Ok(out)
    }
}

impl<K: Key> Drop for SpillCounter<K> {
    fn drop(&mut self) {
        for p in &self.runs {
            let _ = std::fs::remove_file(p);
        }
    }
}

/// Count patterns and transitions of one block of IDs in parallel.
/// `prev` is the ID preceding `ids[0]` (0 = none).
pub fn count_block(
    ids: &[u128],
    prev: u128,
) -> (FxHashMap<u128, u64>, FxHashMap<(u128, u128), u64>) {
    use rayon::prelude::*;
    let n = ids.len();
    let nt = rayon::current_num_threads().max(1);
    let block = ((n + nt - 1) / nt).max(1 << 16);
    let nblocks = (n + block - 1) / block;
    let parts: Vec<(FxHashMap<u128, u64>, FxHashMap<(u128, u128), u64>)> = (0..nblocks)
        .into_par_iter()
        .map(|b| {
            let lo = b * block;
            let hi = (lo + block).min(n);
            let mut pm: FxHashMap<u128, u64> = FxHashMap::default();
            let mut tm: FxHashMap<(u128, u128), u64> = FxHashMap::default();
            let mut p = if lo == 0 { prev } else { ids[lo - 1] };
            for &id in &ids[lo..hi] {
                if id != 0 {
                    *pm.entry(id).or_insert(0) += 1;
                    if p != 0 {
                        *tm.entry((p, id)).or_insert(0) += 1;
                    }
                }
                p = id;
            }
            (pm, tm)
        })
        .collect();
    let mut it = parts.into_iter();
    let (mut pm, mut tm) = it.next().unwrap_or_default();
    for (p2, t2) in it {
        let (big, small) = if p2.len() > pm.len() { (p2, std::mem::take(&mut pm)) } else { (std::mem::take(&mut pm), p2) };
        pm = big;
        for (k, c) in small {
            *pm.entry(k).or_insert(0) += c;
        }
        let (big, small) = if t2.len() > tm.len() { (t2, std::mem::take(&mut tm)) } else { (std::mem::take(&mut tm), t2) };
        tm = big;
        for (k, c) in small {
            *tm.entry(k).or_insert(0) += c;
        }
    }
    (pm, tm)
}
