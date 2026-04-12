//! Audio Interface (AI): RDRAM → DAC FIFO (no real sample timing yet).
//!
//! A **non-zero** write to `AI_LEN` (`0x0450_0004`) is treated as an audio DMA completing immediately
//! and raises [`crate::mi::MI_INTR_AI`], matching the usual “one IRQ per buffer” expectation.

use crate::mi::{Mi, MI_INTR_AI};

pub const AI_REGS_BASE: u32 = 0x0450_0000;
pub const AI_REGS_LEN: usize = 0x20;
/// Word index of `AI_LEN` (`0x04` from [`AI_REGS_BASE`]).
pub const AI_REG_LEN: usize = 1;

#[derive(Debug)]
pub struct Ai {
    pub regs: [u32; AI_REGS_LEN / 4],
}

impl Ai {
    pub fn new() -> Self {
        Self {
            regs: [0u32; AI_REGS_LEN / 4],
        }
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
        if wi == AI_REG_LEN && value != 0 {
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
        ai.write(AI_REGS_BASE + 0x04, 0x800, &mut mi);
        assert!(mi.cpu_irq_pending());
    }

    #[test]
    fn len_write_zero_does_not_raise() {
        let mut ai = Ai::new();
        let mut mi = Mi::new();
        mi.mask = MI_INTR_AI;
        ai.write(AI_REGS_BASE + 0x04, 0, &mut mi);
        assert!(!mi.cpu_irq_pending());
    }
}
