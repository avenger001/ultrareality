//! Video Interface (VI): framebuffer pointer, timing, and NTSC frame tick (stub).

use crate::mi::{Mi, MI_INTR_VI};

pub const VI_REGS_BASE: u32 = 0x0440_0000;
/// First 14 × 32-bit registers cover common VI programming.
pub const VI_REGS_LEN: usize = 0x38;

/// NTSC ~59.94 Hz vertical interrupt at 93.75 MHz CPU clock: `93_750_000 / 59.94`.
pub const VI_NTSC_CYCLES_PER_FRAME: u64 = 1_564_062;

#[derive(Debug)]
pub struct Vi {
    pub regs: [u32; 14],
    /// Cycles since last VI interrupt (for `advance`).
    pub cycle_accum: u64,
}

impl Vi {
    pub fn new() -> Self {
        Self {
            regs: [0u32; 14],
            cycle_accum: 0,
        }
    }

    /// Add retired cycles; when a frame elapses, raise `MI_INTR_VI` (stub timing).
    pub fn advance(&mut self, cycles: u64, mi: &mut Mi) {
        self.cycle_accum = self.cycle_accum.saturating_add(cycles);
        while self.cycle_accum >= VI_NTSC_CYCLES_PER_FRAME {
            self.cycle_accum -= VI_NTSC_CYCLES_PER_FRAME;
            mi.raise(MI_INTR_VI);
        }
    }

    pub fn offset(paddr: u32) -> Option<usize> {
        if (VI_REGS_BASE..VI_REGS_BASE + VI_REGS_LEN as u32).contains(&paddr) {
            Some((paddr - VI_REGS_BASE) as usize)
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
        let i = byte_off / 4;
        self.regs.get(i).copied().unwrap_or(0)
    }

    pub fn write(&mut self, paddr: u32, value: u32) {
        let Some(byte_off) = Self::offset(paddr) else {
            return;
        };
        if byte_off & 3 != 0 {
            return;
        }
        let i = byte_off / 4;
        if let Some(r) = self.regs.get_mut(i) {
            *r = value;
        }
    }
}

impl Default for Vi {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn advance_raises_vi_line_in_mi() {
        let mut vi = Vi::new();
        let mut mi = Mi::new();
        mi.mask = MI_INTR_VI;
        vi.advance(VI_NTSC_CYCLES_PER_FRAME, &mut mi);
        assert!(mi.cpu_irq_pending());
    }
}
