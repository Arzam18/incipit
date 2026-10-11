// Transposition table: 16-byte entries in 4-entry buckets (one cache line),
// full key verification, depth- and age-aware replacement.
use crate::position::Move;

pub const BOUND_NONE: u8 = 0;
pub const BOUND_UPPER: u8 = 1;
pub const BOUND_LOWER: u8 = 2;
pub const BOUND_EXACT: u8 = 3;

#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct Entry {
    pub key: u64,
    pub mv: Move,
    pub score: i16,
    pub eval: i16,
    pub depth: u8,
    pub bound: u8,
}

/// Entries per bucket: one 64-byte cache line.
const WAYS: usize = 4;

#[derive(Clone, Copy, Default)]
#[repr(C, align(64))]
struct Bucket([Entry; WAYS]);

/// Buckets of 4 entries. An entry's `bound` byte also holds the generation
/// (search number) it was written in: bits 0-1 the bound, bits 2-7 the
/// generation. Replacement prefers, in a bucket: the same key; else an empty
/// entry; else the entry with the lowest depth - 8 * age (old and shallow
/// first), so quiescence stores no longer evict deep or PV entries.
pub struct TT {
    table: Table,
    gen: u8,
}

/// The buckets: zeroed memory straight from the allocator, 64-byte aligned
/// by hand. A zeroed request at the type's own alignment (64) is served
/// by clearing the memory, which touches every page; one at a plain
/// alignment is a calloc, whose large allocations come from the OS as
/// untouched zero pages, so a big table costs only the memory the search
/// goes on to use. An all-zero bucket is four empty entries.
struct Table {
    base: *mut u8,
    layout: std::alloc::Layout,
    buckets: *mut Bucket,
    len: usize,
}

// The allocation is owned exclusively, like a Box<[Bucket]>.
unsafe impl Send for Table {}
unsafe impl Sync for Table {}

impl Table {
    fn new(len: usize) -> Option<Table> {
        let align = std::mem::align_of::<Bucket>();
        let layout =
            std::alloc::Layout::from_size_align(len.checked_mul(std::mem::size_of::<Bucket>())? + align, 16).ok()?;
        // SAFETY: the layout has a non-zero size.
        let base = unsafe { std::alloc::alloc_zeroed(layout) };
        if base.is_null() {
            return None;
        }
        // SAFETY: the extra `align` bytes leave room for the offset.
        let buckets = unsafe { base.add(base.align_offset(align)) } as *mut Bucket;
        Some(Table { base, layout, buckets, len })
    }
}

impl Drop for Table {
    fn drop(&mut self) {
        // SAFETY: allocated in `new` with this layout.
        unsafe { std::alloc::dealloc(self.base, self.layout) }
    }
}

impl std::ops::Deref for Table {
    type Target = [Bucket];
    fn deref(&self) -> &[Bucket] {
        // SAFETY: `buckets` is aligned, zero-initialised (valid) and `len` long.
        unsafe { std::slice::from_raw_parts(self.buckets, self.len) }
    }
}

impl std::ops::DerefMut for Table {
    fn deref_mut(&mut self) -> &mut [Bucket] {
        unsafe { std::slice::from_raw_parts_mut(self.buckets, self.len) }
    }
}

impl TT {
    pub fn new(mb: usize) -> TT {
        Self::try_new(mb).unwrap_or_else(|| {
            eprintln!("error: cannot allocate a {} MB hash table", mb);
            std::process::exit(1)
        })
    }
    /// As `new`, but None instead of aborting when the memory isn't there
    /// (see `Table` for why a big table is cheap until used).
    pub fn try_new(mb: usize) -> Option<TT> {
        let n = ((mb.max(1) * 1024 * 1024) / std::mem::size_of::<Bucket>()).max(1);
        Some(TT { table: Table::new(n)?, gen: 0 })
    }
    pub fn clear(&mut self) {
        // SAFETY: all-zero buckets are valid (and empty).
        unsafe { std::ptr::write_bytes(self.table.as_mut_ptr(), 0, self.table.len()) };
        self.gen = 0;
    }
    /// Starts a new search: entries written before it age by one.
    pub fn new_search(&mut self) {
        self.gen = (self.gen + 1) & 63;
    }
    #[inline(always)]
    fn idx(&self, key: u64) -> usize {
        ((key as u128 * self.table.len() as u128) >> 64) as usize
    }
    #[inline(always)]
    pub fn prefetch(&self, key: u64) {
        let p = unsafe { self.table.as_ptr().add(self.idx(key)) } as *const i8;
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_prefetch(p, std::arch::x86_64::_MM_HINT_T0)
        };
        #[cfg(target_arch = "aarch64")]
        unsafe {
            std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) p, options(nostack, preserves_flags, readonly))
        };
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let _ = p;
    }
    #[inline(always)]
    pub fn probe(&self, key: u64) -> Option<Entry> {
        let b = unsafe { self.table.get_unchecked(self.idx(key)) };
        for e in b.0.iter() {
            if e.key == key && e.bound & 3 != BOUND_NONE {
                return Some(Entry { bound: e.bound & 3, ..*e });
            }
        }
        None
    }
    #[inline(always)]
    pub fn store(&mut self, key: u64, mv: Move, score: i32, eval: i32, depth: i32, bound: u8) {
        let gen = self.gen;
        let i = self.idx(key);
        let b = unsafe { self.table.get_unchecked_mut(i) };
        let age = |e: &Entry| (gen.wrapping_sub(e.bound >> 2) & 63) as i32;
        let mut slot = 0;
        let mut found = false;
        let mut worst = i32::MAX;
        for (k, e) in b.0.iter().enumerate() {
            if e.key == key && e.bound & 3 != BOUND_NONE {
                slot = k;
                found = true;
                break;
            }
            let q = if e.bound & 3 == BOUND_NONE { i32::MIN } else { e.depth as i32 - 8 * age(e) };
            if q < worst {
                worst = q;
                slot = k;
            }
        }
        let e = &mut b.0[slot];
        // Same position: keep the deeper result unless this one is exact,
        // nearly as deep, or the old one is from an earlier search.
        if found && !(bound == BOUND_EXACT || depth + 3 >= e.depth as i32 || age(e) != 0) {
            return;
        }
        let mv = if mv == 0 && found { e.mv } else { mv };
        *e = Entry {
            key,
            mv,
            score: score as i16,
            eval: eval as i16,
            depth: depth.clamp(0, 255) as u8,
            bound: bound | (gen << 2),
        };
    }
    /// Permille of entries in use from the current search (first 1000).
    pub fn hashfull(&self) -> usize {
        self.table
            .iter()
            .take(1000 / WAYS)
            .flat_map(|b| b.0.iter())
            .filter(|e| e.bound & 3 != BOUND_NONE && e.bound >> 2 == self.gen)
            .count()
    }
}
