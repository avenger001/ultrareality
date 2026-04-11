//! COP0 subset: Status, Cause, EPC — exceptions and interrupt scaffolding.

/// Status: interrupt enable (IE).
pub const STATUS_IE: u32 = 1 << 0;
/// Status: exception level (EXL).
pub const STATUS_EXL: u32 = 1 << 1;
/// Status: error level (ERL).
pub const STATUS_ERL: u32 = 1 << 2;
/// Status: bootstrap exception vectors (BEV).
pub const STATUS_BEV: u32 = 1 << 22;

/// Cause: exception code field (bits 2–6).
pub const CAUSE_EXCCODE_SHIFT: u32 = 2;
pub const CAUSE_EXCCODE_MASK: u32 = 0x1F;

/// Exception code: interrupt.
pub const EXCCODE_INT: u32 = 0;

#[derive(Clone, Debug)]
pub struct Cop0 {
    pub status: u32,
    pub cause: u32,
    pub epc: u64,
    /// COP0 r30 — error return (ERET when `Status.ERL` is set).
    pub error_epc: u64,
    pub badvaddr: u64,
    pub compare: u32,
    pub count: u32,
}

impl Cop0 {
    pub fn new() -> Self {
        Self {
            // Kernel mode, interrupts off — typical cold reset approximation.
            status: 0x7040_0004,
            cause: 0,
            epc: 0,
            error_epc: 0,
            badvaddr: 0,
            compare: 0,
            count: 0,
        }
    }

    /// True if interrupts are globally enabled and not blocked by EXL/ERL.
    #[inline]
    pub fn interrupts_enabled(&self) -> bool {
        (self.status & STATUS_IE) != 0
            && (self.status & STATUS_EXL) == 0
            && (self.status & STATUS_ERL) == 0
    }

    /// Vector for external interrupt: `0x80000180` (cached) or `0xBFC00380` (BEV).
    #[inline]
    pub fn interrupt_vector(&self) -> u64 {
        if (self.status & STATUS_BEV) != 0 {
            0xFFFF_FFFF_BFC0_0380u64
        } else {
            0xFFFF_FFFF_8000_0180u64
        }
    }

    /// Record an interrupt exception before redirecting `PC` (caller sets `pc`).
    pub fn enter_interrupt_exception(&mut self, epc: u64) {
        self.epc = epc;
        self.cause = (self.cause & !(CAUSE_EXCCODE_MASK << CAUSE_EXCCODE_SHIFT))
            | (EXCCODE_INT << CAUSE_EXCCODE_SHIFT);
        self.status |= STATUS_EXL;
    }

    /// `ERET` return path (caller assigns `self.pc`).
    #[inline]
    pub fn apply_eret(&mut self) -> u64 {
        if (self.status & STATUS_ERL) != 0 {
            self.status &= !STATUS_ERL;
            self.error_epc
        } else {
            self.status &= !STATUS_EXL;
            self.epc
        }
    }

    pub fn read_32(&self, reg: u32) -> u32 {
        match reg {
            8 => self.badvaddr as u32,
            9 => (self.badvaddr >> 32) as u32,
            11 => self.compare,
            12 => self.status,
            13 => self.cause,
            14 => self.epc as u32,
            15 => {
                // PRId — VR4300 (games read COP0 r15; not EPC high).
                0x0B00_0002
            }
            30 => self.error_epc as u32,
            31 => (self.error_epc >> 32) as u32,
            _ => 0,
        }
    }

    pub fn write_32(&mut self, reg: u32, value: u32) {
        match reg {
            11 => self.compare = value,
            12 => self.status = value,
            13 => self.cause = value,
            14 => self.epc = (self.epc & !0xFFFF_FFFF) | u64::from(value),
            15 => self.epc = (self.epc & 0xFFFF_FFFF) | (u64::from(value) << 32),
            30 => self.error_epc = (self.error_epc & !0xFFFF_FFFF) | u64::from(value),
            31 => self.error_epc = (self.error_epc & 0xFFFF_FFFF) | (u64::from(value) << 32),
            _ => {}
        }
    }
}

impl Default for Cop0 {
    fn default() -> Self {
        Self::new()
    }
}
