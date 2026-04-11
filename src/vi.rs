//! Video Interface (VI): framebuffer pointer, timing, and NTSC frame tick (stub).

use crate::mi::{Mi, MI_INTR_VI};

pub const VI_REGS_BASE: u32 = 0x0440_0000;
/// First 14 × 32-bit registers cover common VI programming.
pub const VI_REGS_LEN: usize = 0x38;

/// Word index into [`Vi::regs`] / n64brew [`Video_Interface`](https://n64brew.dev/wiki/Video_Interface).
pub const VI_REG_CONTROL: usize = 0;
pub const VI_REG_ORIGIN: usize = 1;
pub const VI_REG_WIDTH: usize = 2;
pub const VI_REG_V_INTR: usize = 3;
pub const VI_REG_V_CURRENT: usize = 4;
pub const VI_REG_BURST: usize = 5;
pub const VI_REG_V_SYNC: usize = 6;
pub const VI_REG_H_SYNC: usize = 7;
pub const VI_REG_LEAP: usize = 8;
pub const VI_REG_H_VIDEO: usize = 9;
pub const VI_REG_V_VIDEO: usize = 10;
pub const VI_REG_V_BURST: usize = 11;
pub const VI_REG_X_SCALE: usize = 12;
pub const VI_REG_Y_SCALE: usize = 13;

/// NTSC ~59.94 Hz vertical interrupt at 93.75 MHz CPU clock: `93_750_000 / 59.94`.
pub const VI_NTSC_CYCLES_PER_FRAME: u64 = 1_564_062;

#[derive(Debug)]
pub struct Vi {
    pub regs: [u32; 14],
    /// Cycles since last VI interrupt (for `advance`).
    pub cycle_accum: u64,
    /// Increments each time a full NTSC frame elapses (stub timing).
    pub frame_counter: u64,
}

impl Vi {
    pub fn new() -> Self {
        Self {
            regs: [0u32; 14],
            cycle_accum: 0,
            frame_counter: 0,
        }
    }

    /// Add retired cycles; when a frame elapses, raise `MI_INTR_VI` (stub timing).
    pub fn advance(&mut self, cycles: u64, mi: &mut Mi) {
        self.cycle_accum = self.cycle_accum.saturating_add(cycles);
        while self.cycle_accum >= VI_NTSC_CYCLES_PER_FRAME {
            self.cycle_accum -= VI_NTSC_CYCLES_PER_FRAME;
            self.frame_counter = self.frame_counter.wrapping_add(1);
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

    /// RDRAM byte offset for the framebuffer (`VI_ORIGIN & 0x00FF_FFFF`).
    #[inline]
    pub fn framebuffer_rdram_offset(&self) -> usize {
        (self.regs[VI_REG_ORIGIN] & 0x00FF_FFFF) as usize
    }

    /// Horizontal resolution in pixels (`VI_WIDTH`); defaults if unset.
    pub fn display_width(&self) -> u32 {
        let w = self.regs[VI_REG_WIDTH] & 0xFFF;
        if w == 0 {
            320
        } else {
            w.min(720)
        }
    }

    /// Stub vertical resolution (full VI timing not modeled yet).
    pub fn display_height(&self) -> u32 {
        240
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
