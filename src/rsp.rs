//! **Phase 2:** Reality Signal Processor — vector core not implemented yet.
//!
//! The bus already exposes DMEM/IMEM, DMA, `SP_STATUS` halt, and `SP_PC`. This module is the
//! future home for instruction fetch/decode and VU execution so graphics microcode can feed
//! [`crate::rdp::Rdp`] display lists.

/// Placeholder RSP execution state (extend with V0–V31, accumulators, etc.).
#[derive(Clone, Debug)]
pub struct RspState {
    /// Mirrors `SP_STATUS` bit 0 — RSP stopped until cleared by software.
    pub halted: bool,
    /// Word-aligned PC into IMEM (`& 0xFFC`).
    pub pc: u32,
}

impl Default for RspState {
    fn default() -> Self {
        Self { halted: true, pc: 0 }
    }
}

/// Owning wrapper for future stepping API (`step` / `run_quantum`).
#[derive(Debug)]
pub struct Rsp {
    pub state: RspState,
}

impl Default for Rsp {
    fn default() -> Self {
        Self::new()
    }
}

impl Rsp {
    pub fn new() -> Self {
        Self {
            state: RspState::default(),
        }
    }

    /// Single-instruction step — **not implemented** (returns `false` until Phase 2 lands).
    pub fn step(&mut self) -> bool {
        let _ = &mut self.state;
        false
    }
}
