//! MIPS Interface (MI): RCP interrupt routing and mode.
//!
//! **Acknowledge:** a write to [`MI_REG_INTR`] clears bits that are set in the value
//! (write-1-to-clear / W1C), matching common N64 behavior.

pub const MI_REGS_BASE: u32 = 0x0430_0000;
pub const MI_REGS_LEN: usize = 0x10;

/// `MI_MODE` — RCP mode ([n64brew: MI](https://n64brew.dev/wiki/MIPS_Interface)).
pub const MI_REG_MODE: u32 = 0x00;
/// `MI_VERSION` — read-only; returns [`MI_VERSION_DEFAULT`].
pub const MI_REG_VERSION: u32 = 0x04;
/// `MI_INTR` — pending RCP interrupt bits; **write** with bits set clears them (W1C).
pub const MI_REG_INTR: u32 = 0x08;
/// `MI_INTR_MASK` — which [`MI_INTR_*`] lines can assert the CPU interrupt.
pub const MI_REG_INTR_MASK: u32 = 0x0C;

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
            MI_REG_MODE => self.mode,
            MI_REG_VERSION => MI_VERSION_DEFAULT,
            MI_REG_INTR => self.intr,
            MI_REG_INTR_MASK => self.mask,
            _ => 0,
        }
    }

    pub fn write(&mut self, paddr: u32, value: u32) {
        let Some(off) = Self::offset(paddr) else {
            return;
        };
        match off {
            MI_REG_MODE => {
                // MI_MODE write decode (n64brew MIPS Interface):
                //   bits 0–6  init mode length (stored)
                //   bit  7    clear init mode
                //   bit  8    set   init mode
                //   bit  9    clear EBUS test mode
                //   bit  10   set   EBUS test mode
                //   bit  11   clear DP interrupt   ← libultra's __osDpSetNextBuffer/handler ack path
                //   bit  12   clear RDRAM register mode
                //   bit  13   set   RDRAM register mode
                // Mode register state below tracks bits 7–13 separately from the
                // raw store; the init-length field stays in the low 7 bits.
                self.mode = (self.mode & !0x7F) | (value & 0x7F);
                if value & (1 << 7)  != 0 { self.mode &= !(1 << 7); }
                if value & (1 << 8)  != 0 { self.mode |=  1 << 7;  }
                if value & (1 << 9)  != 0 { self.mode &= !(1 << 9); }
                if value & (1 << 10) != 0 { self.mode |=  1 << 9;  }
                if value & (1 << 11) != 0 { self.intr &= !MI_INTR_DP; }
                if value & (1 << 12) != 0 { self.mode &= !(1 << 13); }
                if value & (1 << 13) != 0 { self.mode |=  1 << 13;  }
            }
            MI_REG_INTR => self.intr &= !value,
            MI_REG_INTR_MASK => {
                // Set/clear bit pairs: even bits clear, odd bits set
                // Bit 0/1 = SP, 2/3 = SI, 4/5 = AI, 6/7 = VI, 8/9 = PI, 10/11 = DP
                for i in 0..6 {
                    let clear_bit = 1 << (i * 2);     // even: clear
                    let set_bit = 1 << (i * 2 + 1);   // odd: set
                    let mask_bit = 1 << i;            // actual mask bit
                    if value & clear_bit != 0 {
                        self.mask &= !mask_bit;
                    }
                    if value & set_bit != 0 {
                        self.mask |= mask_bit;
                    }
                }
            }
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
        mi.write(MI_REGS_BASE + MI_REG_INTR, MI_INTR_PI);
        assert_eq!(mi.intr & MI_INTR_PI, 0);
        assert_ne!(mi.intr & MI_INTR_SI, 0);
        assert_ne!(mi.intr & MI_INTR_VI, 0);
        mi.write(MI_REGS_BASE + MI_REG_INTR, MI_INTR_SI | MI_INTR_VI);
        assert_eq!(mi.intr, 0);
    }

    #[test]
    fn mode_init_length_writes_through_low_bits() {
        let mut mi = Mi::new();
        // Low 7 bits = init mode length
        mi.write(MI_REGS_BASE + MI_REG_MODE, 0x3F);
        assert_eq!(mi.read(MI_REGS_BASE + MI_REG_MODE) & 0x7F, 0x3F);
    }

    #[test]
    fn mode_write_bit11_clears_dp_interrupt() {
        let mut mi = Mi::new();
        mi.raise(MI_INTR_DP | MI_INTR_VI);
        // bit 11 of MI_MODE write acks DP IRQ on real hardware
        mi.write(MI_REGS_BASE + MI_REG_MODE, 1 << 11);
        assert_eq!(mi.intr & MI_INTR_DP, 0);
        assert_ne!(mi.intr & MI_INTR_VI, 0, "VI must not be touched");
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

    #[test]
    fn intr_mask_set_clear_pairs() {
        let mut mi = Mi::new();
        // Set VI mask (bit 7)
        mi.write(MI_REGS_BASE + MI_REG_INTR_MASK, 0x80);
        assert_eq!(mi.mask, MI_INTR_VI);
        // Set SP mask (bit 1)
        mi.write(MI_REGS_BASE + MI_REG_INTR_MASK, 0x02);
        assert_eq!(mi.mask, MI_INTR_VI | MI_INTR_SP);
        // Clear VI mask (bit 6)
        mi.write(MI_REGS_BASE + MI_REG_INTR_MASK, 0x40);
        assert_eq!(mi.mask, MI_INTR_SP);
        // Set and clear in same write: set wins (odd bit processed after even)
        mi.write(MI_REGS_BASE + MI_REG_INTR_MASK, 0x03); // clear SP (0x01), set SP (0x02)
        assert_eq!(mi.mask, MI_INTR_SP); // set wins
    }
}
