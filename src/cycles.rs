//! Cycle costs for retired VR4300 operations.
//!
//! Values follow a simplified single-issue model: most integer ops cost one
//! cycle; multiply/divide use MDU latencies; loads/stores model I-cache and
//! D-cache hit/miss latencies.
//!
//! On retail N64 the VR4300 core runs at **93.75 MHz**, the same as the RCP
//! master clock. [`crate::Machine::step`] therefore adds each instruction’s
//! retired cost plus deferred VI/RDP debt into **one** RCP timeline — see
//! [`crate::timing`] — without an extra CPU↔RCP frequency scaler.
//!
//! **MDU interlock:** [`crate::cpu::R4300i`] keeps `mdu_issue_remain` so `MFHI`/`MFLO` can add
//! stall cycles if HI/LO are read before the MDU would finish.

/// One-cycle integer / logical operations (ideal pipeline).
pub const ALU: u64 = 1;

/// Branch / jump dispatch (delay slot is a separate retire).
pub const BRANCH: u64 = 1;

/// Extra cycles when a **conditional** branch is taken (fetch bubble stand-in).
pub const BRANCH_TAKEN_EXTRA: u64 = 1;

/// `MULT` / `MULTU` — result available to `MFHI`/`MFLO` after MDU latency.
pub const MULT_LATENCY: u64 = 5;

/// `DIV` / `DIVU` — worst-case latency bound used until operand-specific
/// timing is implemented.
pub const DIV_LATENCY: u64 = 36;

/// Base load/store latency before RI RDRAM tuning ([`crate::bus::Bus::rdram_access_extra_cycles`]).
pub const MEM_ACCESS_BASE: u64 = 1;

/// Default load/store latency when caches are not modeled (same as [`MEM_ACCESS_BASE`] for flat RAM tests).
pub const MEM_ACCESS: u64 = MEM_ACCESS_BASE;

/// Coprocessor register move (MFC/MTC).
pub const COP_MOVE: u64 = 1;

/// External interrupt exception taken (RCP / MI line).
pub const INTERRUPT: u64 = 2;

/// Synchronous general exception (`SYSCALL`, `BREAK`, …) cycle cost.
pub const EXCEPTION: u64 = INTERRUPT;

// --- Cache timing -----------------------------------------------------------

/// I-cache hit latency (single cycle fetch from cache).
pub const ICACHE_HIT: u64 = 1;

/// D-cache hit latency (single cycle load/store to cache).
pub const DCACHE_HIT: u64 = 1;

/// I-cache line fill overhead: Rambus packet setup + 32-byte line.
/// Approx 40 cycles setup + 32 bytes × 2 cycles/byte = 104 cycles.
pub const ICACHE_MISS_FILL: u64 = 104;

/// D-cache line fill overhead: Rambus packet setup + 16-byte line.
/// Approx 40 cycles setup + 16 bytes × 2 cycles/byte = 72 cycles.
pub const DCACHE_MISS_FILL: u64 = 72;

/// D-cache writeback cost (dirty eviction): 16 bytes to RDRAM.
/// Approx 24 cycles setup + 16 bytes × 2 cycles/byte = 56 cycles.
pub const DCACHE_WRITEBACK: u64 = 56;

/// CACHE instruction base cost (tag manipulation).
pub const CACHE_OP: u64 = 2;
