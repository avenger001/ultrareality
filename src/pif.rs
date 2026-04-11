//! PIF (Peripheral Interface): boot ROM and 64-byte “scratch” RAM.
//!
//! Physical layout: ROM `0x1FC00000`–`0x1FC007BF`, RAM `0x1FC007C0`–`0x1FC007FF`.

pub const PIF_ROM_START: u32 = 0x1FC0_0000;
pub const PIF_ROM_LEN: usize = 0x7C0;
pub const PIF_RAM_START: u32 = 0x1FC0_07C0;
pub const PIF_RAM_LEN: usize = 64;
pub const PIF_WINDOW_END: u32 = 0x1FC0_0800;

#[derive(Debug)]
pub struct Pif {
    pub rom: Box<[u8]>,
    pub ram: [u8; PIF_RAM_LEN],
}

impl Pif {
    pub fn new() -> Self {
        Self {
            rom: vec![0u8; PIF_ROM_LEN].into_boxed_slice(),
            ram: [0u8; PIF_RAM_LEN],
        }
    }

    #[inline]
    pub fn paddr_index(paddr: u32) -> Option<usize> {
        match paddr {
            PIF_ROM_START..0x1FC0_07C0 => Some((paddr - PIF_ROM_START) as usize),
            PIF_RAM_START..PIF_WINDOW_END => {
                Some(PIF_ROM_LEN + (paddr - PIF_RAM_START) as usize)
            }
            _ => None,
        }
    }

    pub fn read_u8(&self, paddr: u32) -> Option<u8> {
        let i = Self::paddr_index(paddr)?;
        if i < PIF_ROM_LEN {
            return Some(self.rom[i]);
        }
        let ri = i - PIF_ROM_LEN;
        Some(self.ram[ri])
    }

    pub fn write_u8(&mut self, paddr: u32, v: u8) {
        let Some(i) = Self::paddr_index(paddr) else {
            return;
        };
        if i < PIF_ROM_LEN {
            return;
        }
        let ri = i - PIF_ROM_LEN;
        self.ram[ri] = v;
    }

    pub fn read_u32(&self, paddr: u32) -> Option<u32> {
        if paddr & 3 != 0 {
            return None;
        }
        let b0 = self.read_u8(paddr)?;
        let b1 = self.read_u8(paddr + 1)?;
        let b2 = self.read_u8(paddr + 2)?;
        let b3 = self.read_u8(paddr + 3)?;
        Some(u32::from_be_bytes([b0, b1, b2, b3]))
    }

    pub fn write_u32(&mut self, paddr: u32, v: u32) {
        if paddr & 3 != 0 {
            return;
        }
        for (k, byte) in v.to_be_bytes().iter().enumerate() {
            self.write_u8(paddr + k as u32, *byte);
        }
    }
}

impl Default for Pif {
    fn default() -> Self {
        Self::new()
    }
}
