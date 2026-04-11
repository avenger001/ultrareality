//! Cycle costs for retired R4300i operations.
//!
//! Values follow the simplified “best case” single-issue model from VR4300
//! documentation: most integer ops retire in one cycle when the pipeline is
//! full; multiply/divide occupy the MDU for several cycles; memory ops pay
//! whatever the bus model charges (here, a flat latency until the cache/bus
//! layer is modeled).

/// One-cycle integer / logical operations (ideal pipeline).
pub const ALU: u64 = 1;

/// Branch resolved in the branch slot (no taken-penalty modeled yet).
pub const BRANCH: u64 = 1;

/// `MULT` / `MULTU` — result available to `MFHI`/`MFLO` after MDU latency.
pub const MULT_LATENCY: u64 = 5;

/// `DIV` / `DIVU` — worst-case latency bound used until operand-specific
/// timing is implemented.
pub const DIV_LATENCY: u64 = 36;

/// Default load/store latency when caches are not modeled (L1 hit stand-in).
pub const MEM_ACCESS: u64 = 1;

/// Coprocessor register move (MFC/MTC).
pub const COP_MOVE: u64 = 1;
