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

/// Byte offset of the word-sized VI register at `word_index` (`0` … `13`).
#[inline]
pub const fn vi_reg_byte_off(word_index: usize) -> u32 {
    (word_index as u32) * 4
}

/// Byte offset from [`VI_REGS_BASE`] for each VI word (same as [`vi_reg_byte_off`] of the matching [`VI_REG_*`] index).
pub const VI_OFF_CONTROL: u32 = vi_reg_byte_off(VI_REG_CONTROL);
pub const VI_OFF_ORIGIN: u32 = vi_reg_byte_off(VI_REG_ORIGIN);
pub const VI_OFF_WIDTH: u32 = vi_reg_byte_off(VI_REG_WIDTH);
pub const VI_OFF_V_INTR: u32 = vi_reg_byte_off(VI_REG_V_INTR);
pub const VI_OFF_V_CURRENT: u32 = vi_reg_byte_off(VI_REG_V_CURRENT);
pub const VI_OFF_BURST: u32 = vi_reg_byte_off(VI_REG_BURST);
pub const VI_OFF_V_SYNC: u32 = vi_reg_byte_off(VI_REG_V_SYNC);
pub const VI_OFF_H_SYNC: u32 = vi_reg_byte_off(VI_REG_H_SYNC);
pub const VI_OFF_LEAP: u32 = vi_reg_byte_off(VI_REG_LEAP);
pub const VI_OFF_H_VIDEO: u32 = vi_reg_byte_off(VI_REG_H_VIDEO);
pub const VI_OFF_V_VIDEO: u32 = vi_reg_byte_off(VI_REG_V_VIDEO);
pub const VI_OFF_V_BURST: u32 = vi_reg_byte_off(VI_REG_V_BURST);
pub const VI_OFF_X_SCALE: u32 = vi_reg_byte_off(VI_REG_X_SCALE);
pub const VI_OFF_Y_SCALE: u32 = vi_reg_byte_off(VI_REG_Y_SCALE);

/// NTSC ~59.94 Hz vertical interrupt; RCP cycles per field ([`crate::timing::RCP_MASTER_HZ_NTSC`]).
pub const VI_NTSC_CYCLES_PER_FRAME: u64 = crate::timing::VI_NTSC_CYCLES_PER_FRAME;
/// Stub: treat a frame as [`crate::timing::VI_NTSC_ACTIVE_SCANLINES`] lines for `VI_V_CURRENT` scaling.
pub const VI_NTSC_SCANLINES: u64 = crate::timing::VI_NTSC_ACTIVE_SCANLINES;

/// Default RDRAM cycles per byte for VI fetch ([`crate::timing::RDRAM_BUS_CYCLES_PER_BYTE`]).
pub const VI_RDRAM_CYCLES_PER_BYTE: u64 = crate::timing::RDRAM_BUS_CYCLES_PER_BYTE;

#[derive(Debug)]
pub struct Vi {
    pub regs: [u32; 14],
    /// Cycles elapsed within the current frame `[0, VI_NTSC_CYCLES_PER_FRAME)`.
    pub cycle_in_frame: u64,
    /// Increments each time a full NTSC frame elapses (stub timing).
    pub frame_counter: u64,
    /// Master-cycle debt for reading the framebuffer from RDRAM (billed via [`crate::bus::SystemBus::drain_deferred_cycles`]).
    pub fetch_debt: u64,
    /// Avoid re-raising `MI_INTR_VI` for the same `VI_V_INTR` line crossing within one field.
    v_line_intr_latched: bool,
}

/// Diagnostic counter: total times MI_INTR_VI was raised by vi.advance.
pub static VI_INT_RAISE_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Diagnostic counter: times VI_V_CURRENT was written (VI interrupt ack).
pub static VI_INT_ACK_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

impl Vi {
    pub fn new() -> Self {
        Self {
            regs: [0u32; 14],
            cycle_in_frame: 0,
            frame_counter: 0,
            fetch_debt: 0,
            v_line_intr_latched: false,
        }
    }

    /// RCP cycle offset within the field for `VI_V_INTR` (half-line / line interrupt). `None` if disabled.
    #[inline]
    pub fn v_intr_cycle_offset(&self) -> Option<u64> {
        let v = self.regs[VI_REG_V_INTR] & 0x3FF;
        if v == 0 {
            return None;
        }
        let line = (v as u64).min(VI_NTSC_SCANLINES - 1);
        Some(line.saturating_mul(VI_NTSC_CYCLES_PER_FRAME) / VI_NTSC_SCANLINES)
    }

    /// Schedule RDRAM bandwidth cost for reading `pixels` RGBA5551 texels (2 bytes each) for display.
    pub fn charge_framebuffer_fetch_rgba16_pixels(&mut self, pixels: u64, cycles_per_byte: u64) {
        let bytes = pixels.saturating_mul(2);
        self.fetch_debt = self
            .fetch_debt
            .saturating_add(bytes.saturating_mul(cycles_per_byte));
    }

    pub fn drain_fetch_debt(&mut self) -> u64 {
        std::mem::take(&mut self.fetch_debt)
    }

    /// Add retired cycles; VI interrupt fires when `VI_V_CURRENT` crosses `VI_V_INTR`.
    /// Frame counter increments at end of field (no separate interrupt at frame wrap).
    pub fn advance(&mut self, cycles: u64, mi: &mut Mi) {
        let start = self.cycle_in_frame;
        let end = start.saturating_add(cycles);
        if let Some(t) = self.v_intr_cycle_offset() {
            if !self.v_line_intr_latched && start < t && end >= t {
                self.v_line_intr_latched = true;
                mi.raise(MI_INTR_VI);
                VI_INT_RAISE_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
        self.cycle_in_frame = end;
        while self.cycle_in_frame >= VI_NTSC_CYCLES_PER_FRAME {
            self.cycle_in_frame -= VI_NTSC_CYCLES_PER_FRAME;
            self.frame_counter = self.frame_counter.wrapping_add(1);
            self.v_line_intr_latched = false;
            // Note: VI interrupt only fires at VI_V_INTR line, not at frame wrap
        }
    }

    /// Approximate current vertical line (for `VI_V_CURRENT` reads), 0–261.
    #[inline]
    pub fn v_current_line(&self) -> u32 {
        let line =
            (self.cycle_in_frame.saturating_mul(VI_NTSC_SCANLINES)) / VI_NTSC_CYCLES_PER_FRAME;
        (line as u32).min((VI_NTSC_SCANLINES - 1) as u32)
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
        if i == VI_REG_V_CURRENT {
            return self.v_current_line();
        }
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
        if i == VI_REG_V_CURRENT {
            return;
        }
        // Log first few VI_ORIGIN writes
        if i == VI_REG_ORIGIN {
            static VI_ORIGIN_LOG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            let n = VI_ORIGIN_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let old_val = self.regs.get(i).copied().unwrap_or(0);
            if n < 20 || (value != old_val && n < 100) {
                eprintln!("[VI_ORIGIN #{} write] frame={} old=0x{:08X} new=0x{:08X}",
                    n, self.frame_counter, old_val, value);
            }
        }
        if let Some(r) = self.regs.get_mut(i) {
            *r = value;
        }
        if i == VI_REG_V_INTR {
            self.v_line_intr_latched = false;
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
        // V_INTR must be non-zero for interrupt to fire (0 disables it)
        vi.regs[VI_REG_V_INTR] = 128; // Set interrupt at scanline 128
        vi.advance(VI_NTSC_CYCLES_PER_FRAME, &mut mi);
        assert!(mi.cpu_irq_pending());
    }

    #[test]
    fn v_current_tracks_cycle_in_frame() {
        let mut vi = Vi::new();
        let mut mi = Mi::new();
        assert_eq!(vi.read(VI_REGS_BASE + VI_OFF_V_CURRENT), 0);
        vi.advance(VI_NTSC_CYCLES_PER_FRAME / 2, &mut mi);
        let mid = vi.read(VI_REGS_BASE + VI_OFF_V_CURRENT);
        assert!(
            mid >= 120 && mid <= 140,
            "mid-frame V_CURRENT should be ~half of 262 lines, got {mid}"
        );
        vi.advance(VI_NTSC_CYCLES_PER_FRAME / 2, &mut mi);
        assert_eq!(vi.cycle_in_frame, 0);
        assert_eq!(vi.read(VI_REGS_BASE + VI_OFF_V_CURRENT), 0);
    }

    #[test]
    fn v_current_register_is_read_only() {
        let mut vi = Vi::new();
        vi.write(VI_REGS_BASE + VI_OFF_V_CURRENT, 0xFFFF_FFFF);
        assert_ne!(vi.read(VI_REGS_BASE + VI_OFF_V_CURRENT), 0xFFFF_FFFF);
    }

    #[test]
    fn vi_off_aliases_match_vi_reg_byte_off() {
        assert_eq!(VI_OFF_ORIGIN, vi_reg_byte_off(VI_REG_ORIGIN));
        assert_eq!(VI_OFF_V_CURRENT, 0x10);
    }

    #[test]
    fn v_intr_triggers_mid_field_mi() {
        let mut vi = Vi::new();
        let mut mi = Mi::new();
        mi.mask = MI_INTR_VI;
        vi.regs[VI_REG_V_INTR] = 128;
        let t = vi.v_intr_cycle_offset().expect("line intr");
        vi.advance(t.saturating_sub(1), &mut mi);
        assert!(!mi.cpu_irq_pending());
        vi.advance(1, &mut mi);
        assert!(mi.cpu_irq_pending());
    }
}
