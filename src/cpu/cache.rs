//! VR4300 I-cache (16 KiB, 2-way) and D-cache (8 KiB, 2-way) models.
//!
//! ## Architecture
//!
//! - **I-cache**: 16 KiB, 2-way set-associative, 32-byte lines → 256 sets × 2 ways
//! - **D-cache**: 8 KiB, 2-way set-associative, 16-byte lines → 256 sets × 2 ways
//!
//! Both caches use pseudo-LRU (single bit per set) for replacement. The D-cache is write-back
//! with dirty tracking. Cache coherency with DMA is managed via explicit CACHE instructions.
//!
//! ## References
//!
//! - [n64brew: VR4300 Cache](https://n64brew.dev/wiki/VR4300#Cache)
//! - VR4300 User's Manual, Chapter 11 (Cache)

/// I-cache line size in bytes (32 bytes = 8 words).
pub const ICACHE_LINE_SIZE: usize = 32;
/// I-cache total size (16 KiB).
pub const ICACHE_SIZE: usize = 16 * 1024;
/// I-cache associativity.
pub const ICACHE_WAYS: usize = 2;
/// I-cache sets (16KB / 32B / 2-way = 256).
pub const ICACHE_SETS: usize = ICACHE_SIZE / ICACHE_LINE_SIZE / ICACHE_WAYS;

/// D-cache line size in bytes (16 bytes = 4 words).
pub const DCACHE_LINE_SIZE: usize = 16;
/// D-cache total size (8 KiB).
pub const DCACHE_SIZE: usize = 8 * 1024;
/// D-cache associativity.
pub const DCACHE_WAYS: usize = 2;
/// D-cache sets (8KB / 16B / 2-way = 256).
pub const DCACHE_SETS: usize = DCACHE_SIZE / DCACHE_LINE_SIZE / DCACHE_WAYS;

/// I-cache line index bits (log2(32) = 5).
const ICACHE_LINE_BITS: u32 = 5;
/// I-cache set index bits (log2(256) = 8).
const ICACHE_SET_BITS: u32 = 8;

/// D-cache line index bits (log2(16) = 4).
const DCACHE_LINE_BITS: u32 = 4;
/// D-cache set index bits (log2(256) = 8).
const DCACHE_SET_BITS: u32 = 8;

/// CACHE instruction operation codes (bits 20:16 of the instruction).
/// Format: `CACHE op, offset(base)` where op selects cache and operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum CacheOp {
    /// Index Invalidate (I-cache): Invalidate line at index.
    IndexInvalidateI = 0x00,
    /// Index Load Tag (I-cache): Load tag into TagLo/TagHi.
    IndexLoadTagI = 0x04,
    /// Index Store Tag (I-cache): Store TagLo/TagHi into cache.
    IndexStoreTagI = 0x08,
    /// Hit Invalidate (I-cache): Invalidate if tag matches.
    HitInvalidateI = 0x10,
    /// Fill (I-cache): Fill line from memory.
    FillI = 0x14,
    /// Hit Writeback Invalidate (I-cache): Same as HitInvalidate for I-cache.
    HitWritebackInvalidateI = 0x18,

    /// Index Writeback Invalidate (D-cache): Writeback and invalidate at index.
    IndexWritebackInvalidateD = 0x01,
    /// Index Load Tag (D-cache): Load tag into TagLo/TagHi.
    IndexLoadTagD = 0x05,
    /// Index Store Tag (D-cache): Store TagLo/TagHi into cache.
    IndexStoreTagD = 0x09,
    /// Create Dirty Exclusive (D-cache): Allocate line without fill, mark dirty.
    CreateDirtyExclusiveD = 0x0D,
    /// Hit Invalidate (D-cache): Invalidate if tag matches (no writeback).
    HitInvalidateD = 0x11,
    /// Hit Writeback Invalidate (D-cache): Writeback dirty and invalidate if hit.
    HitWritebackInvalidateD = 0x15,
    /// Hit Writeback (D-cache): Writeback dirty if hit, don't invalidate.
    HitWritebackD = 0x19,
}

impl CacheOp {
    /// Decode the 5-bit cache operation field.
    pub fn from_u8(op: u8) -> Option<Self> {
        match op {
            0x00 => Some(Self::IndexInvalidateI),
            0x04 => Some(Self::IndexLoadTagI),
            0x08 => Some(Self::IndexStoreTagI),
            0x10 => Some(Self::HitInvalidateI),
            0x14 => Some(Self::FillI),
            0x18 => Some(Self::HitWritebackInvalidateI),
            0x01 => Some(Self::IndexWritebackInvalidateD),
            0x05 => Some(Self::IndexLoadTagD),
            0x09 => Some(Self::IndexStoreTagD),
            0x0D => Some(Self::CreateDirtyExclusiveD),
            0x11 => Some(Self::HitInvalidateD),
            0x15 => Some(Self::HitWritebackInvalidateD),
            0x19 => Some(Self::HitWritebackD),
            _ => None,
        }
    }

    /// True if this operation targets the I-cache.
    #[inline]
    pub fn is_icache(self) -> bool {
        (self as u8) & 1 == 0
    }
}

/// Single cache line state.
#[derive(Clone, Copy, Debug, Default)]
pub struct CacheLine {
    /// Physical tag (upper bits after set index).
    pub tag: u32,
    /// Line is valid.
    pub valid: bool,
    /// Line is dirty (D-cache only, needs writeback).
    pub dirty: bool,
}

/// Single cache set (2-way associative).
#[derive(Clone, Copy, Debug)]
pub struct CacheSet<const WAYS: usize> {
    pub lines: [CacheLine; WAYS],
    /// Pseudo-LRU bit: 0 = way 0 is LRU, 1 = way 1 is LRU.
    pub lru: u8,
}

impl Default for CacheSet<2> {
    fn default() -> Self {
        Self {
            lines: [CacheLine::default(); 2],
            lru: 0,
        }
    }
}

impl<const WAYS: usize> CacheSet<WAYS> {
    /// Find which way (if any) contains the given tag.
    #[inline]
    pub fn find_way(&self, tag: u32) -> Option<usize> {
        for (i, line) in self.lines.iter().enumerate() {
            if line.valid && line.tag == tag {
                return Some(i);
            }
        }
        None
    }

    /// Update LRU after accessing the given way.
    #[inline]
    pub fn touch(&mut self, way: usize) {
        // For 2-way, LRU bit points to the *other* way.
        self.lru = if way == 0 { 1 } else { 0 };
    }

    /// Get the way to replace (the LRU one).
    #[inline]
    pub fn victim_way(&self) -> usize {
        self.lru as usize
    }
}

/// Instruction cache (16 KiB, 2-way, 32-byte lines).
#[derive(Clone, Debug)]
pub struct ICache {
    pub sets: Box<[CacheSet<ICACHE_WAYS>; ICACHE_SETS]>,
    /// Cache hit count (statistics).
    pub hits: u64,
    /// Cache miss count (statistics).
    pub misses: u64,
}

impl Default for ICache {
    fn default() -> Self {
        Self::new()
    }
}

impl ICache {
    pub fn new() -> Self {
        Self {
            sets: Box::new([CacheSet::default(); ICACHE_SETS]),
            hits: 0,
            misses: 0,
        }
    }

    /// Reset all cache lines to invalid.
    pub fn invalidate_all(&mut self) {
        for set in self.sets.iter_mut() {
            for line in set.lines.iter_mut() {
                line.valid = false;
            }
        }
    }

    /// Extract set index from physical address.
    #[inline]
    fn set_index(paddr: u32) -> usize {
        ((paddr >> ICACHE_LINE_BITS) & ((1 << ICACHE_SET_BITS) - 1)) as usize
    }

    /// Extract tag from physical address.
    #[inline]
    fn tag(paddr: u32) -> u32 {
        paddr >> (ICACHE_LINE_BITS + ICACHE_SET_BITS)
    }

    /// Check if address hits in cache. Returns (hit, set_index, way).
    pub fn probe(&mut self, paddr: u32) -> (bool, usize, Option<usize>) {
        let set_idx = Self::set_index(paddr);
        let tag = Self::tag(paddr);
        let set = &mut self.sets[set_idx];
        if let Some(way) = set.find_way(tag) {
            self.hits += 1;
            set.touch(way);
            (true, set_idx, Some(way))
        } else {
            self.misses += 1;
            (false, set_idx, None)
        }
    }

    /// Fill a cache line (on miss). Returns the way used.
    pub fn fill(&mut self, paddr: u32) -> usize {
        let set_idx = Self::set_index(paddr);
        let tag = Self::tag(paddr);
        let set = &mut self.sets[set_idx];
        let way = set.victim_way();
        set.lines[way] = CacheLine {
            tag,
            valid: true,
            dirty: false,
        };
        set.touch(way);
        way
    }

    /// Invalidate a specific line by index (for CACHE Index Invalidate).
    pub fn invalidate_index(&mut self, set_idx: usize, way: usize) {
        if set_idx < ICACHE_SETS && way < ICACHE_WAYS {
            self.sets[set_idx].lines[way].valid = false;
        }
    }

    /// Invalidate by address if it hits (for CACHE Hit Invalidate).
    pub fn invalidate_hit(&mut self, paddr: u32) -> bool {
        let set_idx = Self::set_index(paddr);
        let tag = Self::tag(paddr);
        let set = &mut self.sets[set_idx];
        if let Some(way) = set.find_way(tag) {
            set.lines[way].valid = false;
            true
        } else {
            false
        }
    }
}

/// Data cache (8 KiB, 2-way, 16-byte lines).
#[derive(Clone, Debug)]
pub struct DCache {
    pub sets: Box<[CacheSet<DCACHE_WAYS>; DCACHE_SETS]>,
    /// Cache hit count (statistics).
    pub hits: u64,
    /// Cache miss count (statistics).
    pub misses: u64,
    /// Writebacks performed.
    pub writebacks: u64,
}

impl Default for DCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a D-cache probe or fill operation.
#[derive(Clone, Copy, Debug)]
pub struct DCacheResult {
    /// True if hit.
    pub hit: bool,
    /// Set index.
    pub set_idx: usize,
    /// Way if hit, or victim way if miss.
    pub way: usize,
    /// True if victim needs writeback (dirty eviction).
    pub needs_writeback: bool,
    /// Physical address of line to writeback (if needs_writeback).
    pub writeback_addr: u32,
}

impl DCache {
    pub fn new() -> Self {
        Self {
            sets: Box::new([CacheSet::default(); DCACHE_SETS]),
            hits: 0,
            misses: 0,
            writebacks: 0,
        }
    }

    /// Reset all cache lines to invalid.
    pub fn invalidate_all(&mut self) {
        for set in self.sets.iter_mut() {
            for line in set.lines.iter_mut() {
                line.valid = false;
                line.dirty = false;
            }
        }
    }

    /// Extract set index from physical address.
    #[inline]
    fn set_index(paddr: u32) -> usize {
        ((paddr >> DCACHE_LINE_BITS) & ((1 << DCACHE_SET_BITS) - 1)) as usize
    }

    /// Extract tag from physical address.
    #[inline]
    fn tag(paddr: u32) -> u32 {
        paddr >> (DCACHE_LINE_BITS + DCACHE_SET_BITS)
    }

    /// Reconstruct physical address from tag and set index (line-aligned).
    #[inline]
    fn addr_from_tag(tag: u32, set_idx: usize) -> u32 {
        (tag << (DCACHE_LINE_BITS + DCACHE_SET_BITS))
            | ((set_idx as u32) << DCACHE_LINE_BITS)
    }

    /// Probe cache for read. Updates LRU on hit.
    pub fn probe_read(&mut self, paddr: u32) -> DCacheResult {
        let set_idx = Self::set_index(paddr);
        let tag = Self::tag(paddr);
        let set = &mut self.sets[set_idx];

        if let Some(way) = set.find_way(tag) {
            self.hits += 1;
            set.touch(way);
            DCacheResult {
                hit: true,
                set_idx,
                way,
                needs_writeback: false,
                writeback_addr: 0,
            }
        } else {
            self.misses += 1;
            let victim = set.victim_way();
            let victim_line = &set.lines[victim];
            let needs_wb = victim_line.valid && victim_line.dirty;
            let wb_addr = if needs_wb {
                Self::addr_from_tag(victim_line.tag, set_idx)
            } else {
                0
            };
            DCacheResult {
                hit: false,
                set_idx,
                way: victim,
                needs_writeback: needs_wb,
                writeback_addr: wb_addr,
            }
        }
    }

    /// Probe cache for write. Updates LRU on hit, marks dirty.
    pub fn probe_write(&mut self, paddr: u32) -> DCacheResult {
        let set_idx = Self::set_index(paddr);
        let tag = Self::tag(paddr);
        let set = &mut self.sets[set_idx];

        if let Some(way) = set.find_way(tag) {
            self.hits += 1;
            set.lines[way].dirty = true;
            set.touch(way);
            DCacheResult {
                hit: true,
                set_idx,
                way,
                needs_writeback: false,
                writeback_addr: 0,
            }
        } else {
            self.misses += 1;
            let victim = set.victim_way();
            let victim_line = &set.lines[victim];
            let needs_wb = victim_line.valid && victim_line.dirty;
            let wb_addr = if needs_wb {
                Self::addr_from_tag(victim_line.tag, set_idx)
            } else {
                0
            };
            DCacheResult {
                hit: false,
                set_idx,
                way: victim,
                needs_writeback: needs_wb,
                writeback_addr: wb_addr,
            }
        }
    }

    /// Fill a cache line after a miss. Call after handling any writeback.
    pub fn fill(&mut self, paddr: u32, dirty: bool) -> usize {
        let set_idx = Self::set_index(paddr);
        let tag = Self::tag(paddr);
        let set = &mut self.sets[set_idx];
        let way = set.victim_way();
        set.lines[way] = CacheLine {
            tag,
            valid: true,
            dirty,
        };
        set.touch(way);
        way
    }

    /// Create dirty exclusive (allocate without fill, mark dirty).
    pub fn create_dirty_exclusive(&mut self, paddr: u32) -> DCacheResult {
        let set_idx = Self::set_index(paddr);
        let tag = Self::tag(paddr);
        let set = &mut self.sets[set_idx];

        // Check if already present
        if let Some(way) = set.find_way(tag) {
            set.lines[way].dirty = true;
            set.touch(way);
            return DCacheResult {
                hit: true,
                set_idx,
                way,
                needs_writeback: false,
                writeback_addr: 0,
            };
        }

        let victim = set.victim_way();
        let victim_line = &set.lines[victim];
        let needs_wb = victim_line.valid && victim_line.dirty;
        let wb_addr = if needs_wb {
            Self::addr_from_tag(victim_line.tag, set_idx)
        } else {
            0
        };

        // Allocate the line
        set.lines[victim] = CacheLine {
            tag,
            valid: true,
            dirty: true,
        };
        set.touch(victim);

        if needs_wb {
            self.writebacks += 1;
        }

        DCacheResult {
            hit: false,
            set_idx,
            way: victim,
            needs_writeback: needs_wb,
            writeback_addr: wb_addr,
        }
    }

    /// Invalidate by index (no writeback check).
    pub fn invalidate_index(&mut self, set_idx: usize, way: usize) {
        if set_idx < DCACHE_SETS && way < DCACHE_WAYS {
            self.sets[set_idx].lines[way].valid = false;
            self.sets[set_idx].lines[way].dirty = false;
        }
    }

    /// Writeback and invalidate by index.
    pub fn writeback_invalidate_index(&mut self, set_idx: usize, way: usize) -> Option<u32> {
        if set_idx >= DCACHE_SETS || way >= DCACHE_WAYS {
            return None;
        }
        let line = &self.sets[set_idx].lines[way];
        let wb_addr = if line.valid && line.dirty {
            self.writebacks += 1;
            Some(Self::addr_from_tag(line.tag, set_idx))
        } else {
            None
        };
        self.sets[set_idx].lines[way].valid = false;
        self.sets[set_idx].lines[way].dirty = false;
        wb_addr
    }

    /// Hit invalidate (invalidate if address hits, no writeback).
    pub fn hit_invalidate(&mut self, paddr: u32) -> bool {
        let set_idx = Self::set_index(paddr);
        let tag = Self::tag(paddr);
        let set = &mut self.sets[set_idx];
        if let Some(way) = set.find_way(tag) {
            set.lines[way].valid = false;
            set.lines[way].dirty = false;
            true
        } else {
            false
        }
    }

    /// Hit writeback invalidate (writeback dirty and invalidate if hit).
    pub fn hit_writeback_invalidate(&mut self, paddr: u32) -> Option<u32> {
        let set_idx = Self::set_index(paddr);
        let tag = Self::tag(paddr);
        let set = &mut self.sets[set_idx];
        if let Some(way) = set.find_way(tag) {
            let line = &set.lines[way];
            let wb_addr = if line.dirty {
                self.writebacks += 1;
                Some(Self::addr_from_tag(line.tag, set_idx))
            } else {
                None
            };
            set.lines[way].valid = false;
            set.lines[way].dirty = false;
            wb_addr
        } else {
            None
        }
    }

    /// Hit writeback (writeback dirty if hit, don't invalidate).
    pub fn hit_writeback(&mut self, paddr: u32) -> Option<u32> {
        let set_idx = Self::set_index(paddr);
        let tag = Self::tag(paddr);
        let set = &mut self.sets[set_idx];
        if let Some(way) = set.find_way(tag) {
            let line = &mut set.lines[way];
            if line.dirty {
                self.writebacks += 1;
                line.dirty = false;
                Some(Self::addr_from_tag(line.tag, set_idx))
            } else {
                None
            }
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn icache_set_index_and_tag() {
        // Address 0x8000_1000: physical 0x0000_1000
        // Line bits = 5, set bits = 8, so set = (0x1000 >> 5) & 0xFF = 0x80
        // Tag = 0x1000 >> 13 = 0
        let paddr = 0x0000_1000u32;
        assert_eq!(ICache::set_index(paddr), 0x80);
        assert_eq!(ICache::tag(paddr), 0);

        // Address 0x0002_0000: set = (0x20000 >> 5) & 0xFF = 0, tag = 0x20000 >> 13 = 0x10
        let paddr2 = 0x0002_0000u32;
        assert_eq!(ICache::set_index(paddr2), 0);
        assert_eq!(ICache::tag(paddr2), 0x10);
    }

    #[test]
    fn dcache_set_index_and_tag() {
        // D-cache: line bits = 4, set bits = 8
        let paddr = 0x0000_1000u32;
        // set = (0x1000 >> 4) & 0xFF = 0
        // tag = 0x1000 >> 12 = 1
        assert_eq!(DCache::set_index(paddr), 0);
        assert_eq!(DCache::tag(paddr), 1);
    }

    #[test]
    fn icache_miss_then_hit() {
        let mut cache = ICache::new();
        let paddr = 0x0000_1000u32;

        // First access: miss
        let (hit, _, _) = cache.probe(paddr);
        assert!(!hit);
        assert_eq!(cache.misses, 1);

        // Fill the line
        cache.fill(paddr);

        // Second access: hit
        let (hit, _, way) = cache.probe(paddr);
        assert!(hit);
        assert!(way.is_some());
        assert_eq!(cache.hits, 1);
    }

    #[test]
    fn dcache_write_marks_dirty() {
        let mut cache = DCache::new();
        let paddr = 0x0000_1000u32;
        let step = (DCACHE_SETS as u32) * (DCACHE_LINE_SIZE as u32);
        let paddr_b = paddr.wrapping_add(step);
        let paddr_c = paddr_b.wrapping_add(step);

        // Write miss
        let result = cache.probe_write(paddr);
        assert!(!result.hit);
        assert!(!result.needs_writeback);

        // Fill as dirty
        cache.fill(paddr, true);

        // Read hit
        let result = cache.probe_read(paddr);
        assert!(result.hit);

        // Fill the other way in the same set (2-way); both lines valid+dirty.
        let result = cache.probe_write(paddr_b);
        assert!(!result.hit);
        cache.fill(paddr_b, true);

        // Third distinct tag in the same set must evict a dirty victim.
        let result = cache.probe_read(paddr_c);
        assert!(!result.hit);
        assert!(result.needs_writeback);
    }

    #[test]
    fn cache_lru_replacement() {
        let mut cache = ICache::new();
        let set_idx = 0x10;
        let base = (set_idx as u32) << ICACHE_LINE_BITS;

        // Fill way 0
        let way0 = cache.fill(base);
        assert_eq!(way0, 0); // First fill goes to LRU way

        // Fill way 1 (different tag, same set)
        let addr1 = base + ((ICACHE_SETS as u32) << ICACHE_LINE_BITS);
        let way1 = cache.fill(addr1);
        assert_eq!(way1, 1); // LRU was way 1 after touching way 0

        // Access way 0 again
        let (hit, _, _) = cache.probe(base);
        assert!(hit);

        // Now way 1 is LRU, fill a third address should evict way 1
        let addr2 = base + 2 * ((ICACHE_SETS as u32) << ICACHE_LINE_BITS);
        let way2 = cache.fill(addr2);
        assert_eq!(way2, 1);

        // Original way 0 should still hit
        let (hit, _, _) = cache.probe(base);
        assert!(hit);

        // Way 1 (addr1) should miss now
        let (hit, _, _) = cache.probe(addr1);
        assert!(!hit);
    }
}
