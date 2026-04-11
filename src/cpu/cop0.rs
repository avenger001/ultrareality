//! COP0 subset: Status, Cause, EPC — enough for exception entry/return scaffolding.

#[derive(Clone, Debug)]
pub struct Cop0 {
    pub status: u32,
    pub cause: u32,
    pub epc: u64,
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
            badvaddr: 0,
            compare: 0,
            count: 0,
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
            15 => (self.epc >> 32) as u32,
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
            _ => {}
        }
    }
}

impl Default for Cop0 {
    fn default() -> Self {
        Self::new()
    }
}
