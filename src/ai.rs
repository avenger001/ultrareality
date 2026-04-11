//! Audio Interface (AI): DMA to DAC (register stub).

pub const AI_REGS_BASE: u32 = 0x0450_0000;
pub const AI_REGS_LEN: usize = 0x20;

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

    pub fn write(&mut self, paddr: u32, value: u32) {
        let Some(byte_off) = Self::offset(paddr) else {
            return;
        };
        if byte_off & 3 != 0 {
            return;
        }
        if let Some(r) = self.regs.get_mut(byte_off / 4) {
            *r = value;
        }
    }
}

impl Default for Ai {
    fn default() -> Self {
        Self::new()
    }
}
