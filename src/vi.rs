//! Video Interface (VI): framebuffer pointer and timing (stub registers).

pub const VI_REGS_BASE: u32 = 0x0440_0000;
/// First 14 × 32-bit registers cover common VI programming.
pub const VI_REGS_LEN: usize = 0x38;

#[derive(Debug)]
pub struct Vi {
    pub regs: [u32; 14],
}

impl Vi {
    pub fn new() -> Self {
        Self { regs: [0u32; 14] }
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
