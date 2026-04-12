//! Audio Interface (AI): RDRAM → DAC FIFO.
//!
//! A **non-zero** write to `AI_LEN` ([`AI_REG_LEN`]) raises [`crate::mi::MI_INTR_AI`] and records
//! **RCP cycle debt** for buffer playback ([`crate::timing::ai_pcm_buffer_cycles`]) so [`crate::Machine`]
//! advances [`crate::cpu::cop0::Cop0::count`] and the master timeline with PI/SI/VI.

use crate::mi::{Mi, MI_INTR_AI};
use crate::timing::ai_pcm_buffer_cycles;

pub const AI_REGS_BASE: u32 = 0x0450_0000;
pub const AI_REGS_LEN: usize = 0x20;

/// Byte offset of `AI_DRAM_ADDR` (PCM buffer in RDRAM).
pub const AI_REG_DRAM_ADDR: u32 = 0x00;
/// Byte offset of `AI_LEN` — non-zero write completes the audio DMA stub and raises `MI_INTR_AI`.
pub const AI_REG_LEN: u32 = 0x04;

/// Word index of `AI_LEN` in [`Ai::regs`] (same as `AI_REG_LEN / 4`).
const AI_WORD_INDEX_LEN: usize = 1;

#[derive(Debug)]
pub struct Ai {
    pub regs: [u32; AI_REGS_LEN / 4],
    pending_cycles: u64,
}

impl Ai {
    pub fn new() -> Self {
        Self {
            regs: [0u32; AI_REGS_LEN / 4],
            pending_cycles: 0,
        }
    }

    pub fn drain_cycles(&mut self) -> u64 {
        let c = self.pending_cycles;
        self.pending_cycles = 0;
        c
    }

    pub fn offset(paddr: u32) -> Option<usize> {
        if (AI_REGS_BASE..AI_REGS_BASE + AI_REGS_LEN as u32).contains(&paddr) {
            Some((paddr - AI_REGS_BASE) as usize)
        } else {
            None
        }
    }

    pub fn read(&self, paddr: u32) -> u32 {
        let Some(byte_off) = Self::offset(paddr) else {
            return 0;
        };
        if byte_off & 3 != 0 {
            return 0;
        }
        self.regs.get(byte_off / 4).copied().unwrap_or(0)
    }

    pub fn write(&mut self, paddr: u32, value: u32, mi: &mut Mi) {
        let Some(byte_off) = Self::offset(paddr) else {
            return;
        };
        if byte_off & 3 != 0 {
            return;
        }
        let wi = byte_off / 4;
        if let Some(r) = self.regs.get_mut(wi) {
            *r = value;
        }
        if wi == AI_WORD_INDEX_LEN && value != 0 {
            self.pending_cycles = self
                .pending_cycles
                .saturating_add(ai_pcm_buffer_cycles(value));
            mi.raise(MI_INTR_AI);
        }
    }
}

impl Default for Ai {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mi::Mi;

    #[test]
    fn len_write_nonzero_raises_ai_interrupt() {
        let mut ai = Ai::new();
        let mut mi = Mi::new();
        mi.mask = MI_INTR_AI;
        ai.write(AI_REGS_BASE + AI_REG_LEN, 0x800, &mut mi);
        assert!(mi.cpu_irq_pending());
        assert_eq!(ai.drain_cycles(), crate::timing::ai_pcm_buffer_cycles(0x800));
    }

    #[test]
    fn len_write_zero_does_not_raise() {
        let mut ai = Ai::new();
        let mut mi = Mi::new();
        mi.mask = MI_INTR_AI;
        ai.write(AI_REGS_BASE + AI_REG_LEN, 0, &mut mi);
        assert!(!mi.cpu_irq_pending());
    }
}
