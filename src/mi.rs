//! MIPS Interface (MI): RCP interrupt routing and mode.
//!
//! **Acknowledge:** a write to `MI_INTR` (`0x0430_0008`) clears bits that are set in the value
//! (write-1-to-clear / W1C), matching common N64 behavior.

pub const MI_REGS_BASE: u32 = 0x0430_0000;
pub const MI_REGS_LEN: usize = 0x10;

/// Hardware version / revision (read-only), typical retail value.
pub const MI_VERSION_DEFAULT: u32 = 0x0202_0102;

/// `MI_INTR` / `MI_MASK` bits (RCP sources → CPU via MIPS Interface).
pub const MI_INTR_SP: u32 = 1 << 0;
pub const MI_INTR_SI: u32 = 1 << 1;
pub const MI_INTR_AI: u32 = 1 << 2;
pub const MI_INTR_VI: u32 = 1 << 3;
pub const MI_INTR_PI: u32 = 1 << 4;
pub const MI_INTR_DP: u32 = 1 << 5;

#[derive(Debug)]
pub struct Mi {
    pub mode: u32,
    pub intr: u32,
    pub mask: u32,
}

impl Mi {
    pub fn new() -> Self {
        Self {
            mode: 0,
            intr: 0,
            mask: 0,
        }
    }

    /// OR interrupt bits into `MI_INTR` (devices call when an event is pending).
    #[inline]
    pub fn raise(&mut self, bits: u32) {
        self.intr |= bits;
    }

    /// Clear interrupt bits (used by acknowledge paths).
    #[inline]
    pub fn clear(&mut self, bits: u32) {
        self.intr &= !bits;
    }

    /// True if any raised line is unmasked (CPU may take external interrupt).
    #[inline]
    pub fn cpu_irq_pending(&self) -> bool {
        (self.intr & self.mask) != 0
    }

    pub fn offset(paddr: u32) -> Option<u32> {
        if (MI_REGS_BASE..MI_REGS_BASE + MI_REGS_LEN as u32).contains(&paddr) {
            Some(paddr - MI_REGS_BASE)
        } else {
            None
        }
    }

    pub fn read(&self, paddr: u32) -> u32 {
        let Some(off) = Self::offset(paddr) else {
            return 0;
        };
        match off {
            0x00 => self.mode,
            0x04 => MI_VERSION_DEFAULT,
            0x08 => self.intr,
            0x0C => self.mask,
            _ => 0,
        }
    }

    pub fn write(&mut self, paddr: u32, value: u32) {
        let Some(off) = Self::offset(paddr) else {
            return;
        };
        match off {
            0x00 => self.mode = value,
            0x08 => self.intr &= !value,
            0x0C => self.mask = value,
            _ => {}
        }
    }
}

impl Default for Mi {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_intr_ack_clears_selected_bits() {
        let mut mi = Mi::new();
        mi.raise(MI_INTR_PI | MI_INTR_SI | MI_INTR_VI);
        mi.write(MI_REGS_BASE + 0x08, MI_INTR_PI);
        assert_eq!(mi.intr & MI_INTR_PI, 0);
        assert_ne!(mi.intr & MI_INTR_SI, 0);
        assert_ne!(mi.intr & MI_INTR_VI, 0);
        mi.write(MI_REGS_BASE + 0x08, MI_INTR_SI | MI_INTR_VI);
        assert_eq!(mi.intr, 0);
    }

    #[test]
    fn cpu_irq_pending_respects_mask() {
        let mut mi = Mi::new();
        mi.raise(MI_INTR_DP);
        mi.mask = MI_INTR_VI;
        assert!(!mi.cpu_irq_pending());
        mi.mask |= MI_INTR_DP;
        assert!(mi.cpu_irq_pending());
    }
}
