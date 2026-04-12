//! Audio Interface (AI): RDRAM → DAC FIFO.
//!
//! A non-zero [`AI_REG_LEN`] write schedules **playback time** on the RCP clock; [`crate::mi::MI_INTR_AI`]
//! is raised when that duration elapses ([`Ai::advance_time`]), not on the register write.

use crate::mi::{Mi, MI_INTR_AI};
use crate::timing::ai_pcm_buffer_cycles;

pub const AI_REGS_BASE: u32 = 0x0450_0000;
pub const AI_REGS_LEN: usize = 0x20;

/// Byte offset of `AI_DRAM_ADDR` (PCM buffer in RDRAM).
pub const AI_REG_DRAM_ADDR: u32 = 0x00;
/// Byte offset of `AI_LEN` — non-zero write schedules buffer playback and (later) `MI_INTR_AI`.
pub const AI_REG_LEN: u32 = 0x04;

/// Word index of `AI_LEN` in [`Ai::regs`] (same as `AI_REG_LEN / 4`).
const AI_WORD_INDEX_LEN: usize = 1;

#[derive(Debug, Clone)]
struct AiPlayback {
    remaining_rcp_cycles: u64,
}

#[derive(Debug)]
pub struct Ai {
    pub regs: [u32; AI_REGS_LEN / 4],
    active: Option<AiPlayback>,
}

impl Ai {
    pub fn new() -> Self {
        Self {
            regs: [0u32; AI_REGS_LEN / 4],
            active: None,
        }
    }

    #[inline]
    pub fn playback_pending(&self) -> bool {
        self.active.is_some()
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
            mi.clear(MI_INTR_AI);
            let c = ai_pcm_buffer_cycles(value);
            self.active = Some(AiPlayback {
                remaining_rcp_cycles: c.max(1),
            });
        }
    }

    /// Apply `delta` RCP cycles; raises `MI_INTR_AI` when the current buffer’s playback interval ends.
    pub fn advance_time(&mut self, delta: u64, mi: &mut Mi) {
        let mut d = delta;
        while d > 0 {
            let Some(active) = self.active.as_mut() else {
                return;
            };
            let use_cycles = active.remaining_rcp_cycles.min(d);
            active.remaining_rcp_cycles -= use_cycles;
            d -= use_cycles;
            if active.remaining_rcp_cycles > 0 {
                return;
            }
            self.active = None;
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
    fn len_write_schedules_interrupt_after_cycles() {
        let mut ai = Ai::new();
        let mut mi = Mi::new();
        mi.mask = MI_INTR_AI;
        ai.write(AI_REGS_BASE + AI_REG_LEN, 0x800, &mut mi);
        assert!(!mi.cpu_irq_pending());
        let need = crate::timing::ai_pcm_buffer_cycles(0x800);
        ai.advance_time(need - 1, &mut mi);
        assert!(!mi.cpu_irq_pending());
        ai.advance_time(1, &mut mi);
        assert!(mi.cpu_irq_pending());
    }

    #[test]
    fn len_write_zero_does_not_raise() {
        let mut ai = Ai::new();
        let mut mi = Mi::new();
        mi.mask = MI_INTR_AI;
        ai.write(AI_REGS_BASE + AI_REG_LEN, 0, &mut mi);
        assert!(!mi.cpu_irq_pending());
        ai.advance_time(10_000, &mut mi);
        assert!(!mi.cpu_irq_pending());
    }
}
