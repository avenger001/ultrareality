//! Cycle costs for retired VR4300 operations.
//!
//! Values follow a simplified single-issue model: most integer ops cost one
//! cycle; multiply/divide use MDU latencies; loads/stores use a flat L1-hit
//! stand-in until caches are modeled.
//!
//! On retail N64 the VR4300 core runs at **93.75 MHz**, the same as the RCP
//! master clock. [`crate::Machine::step`] therefore adds each instruction’s
//! retired cost plus deferred VI/RDP debt into **one** RCP timeline — see
//! [`crate::timing`] — without an extra CPU↔RCP frequency scaler.

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

/// External interrupt exception taken (RCP / MI line).
pub const INTERRUPT: u64 = 2;

/// Synchronous general exception (`SYSCALL`, `BREAK`, …) until pipeline timing is modeled.
pub const EXCEPTION: u64 = INTERRUPT;
