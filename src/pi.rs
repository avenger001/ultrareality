//! Parallel Interface (PI): cartridge ROM access and PI DMA into RDRAM.

use crate::bus::{Bus, PhysicalMemory};
use crate::mi::{Mi, MI_INTR_PI};

/// PI register block base (physical).
pub const PI_REGS_BASE: u32 = 0x0460_0000;

/// Cartridge ROM mapping base (physical).
pub const CART_DOM1_ADDR2_BASE: u32 = 0x1000_0000;

/// Rough cycle cost for PI DMA (bytes × factor); refined when the RCP bus is modeled.
pub const PI_CYCLES_PER_DMA_BYTE: u64 = 1;

#[derive(Debug)]
pub struct Pi {
    pub rom: Vec<u8>,
    pub dram_addr: u32,
    pub cart_addr: u32,
    /// Minimal status: bit0 = DMA busy (stub), bit1 = interrupt (stub).
    pub status: u32,
    pending_cycles: u64,
}

impl Pi {
    pub fn new() -> Self {
        Self {
            rom: Vec::new(),
            dram_addr: 0,
            cart_addr: 0,
            status: 0,
            pending_cycles: 0,
        }
    }

    pub fn with_rom(rom: Vec<u8>) -> Self {
        Self {
            rom,
            dram_addr: 0,
            cart_addr: 0,
            status: 0,
            pending_cycles: 0,
        }
    }

    pub fn drain_cycles(&mut self) -> u64 {
        let c = self.pending_cycles;
        self.pending_cycles = 0;
        c
    }

    fn reg_offset(paddr: u32) -> Option<u32> {
        if (PI_REGS_BASE..PI_REGS_BASE + 0x20).contains(&paddr) {
            Some(paddr - PI_REGS_BASE)
        } else {
            None
        }
    }

    pub fn read_reg(&self, paddr: u32) -> u32 {
        let Some(off) = Self::reg_offset(paddr) else {
            return 0;
        };
        match off {
            0x00 => self.dram_addr,
            0x04 => self.cart_addr,
            0x0C => 0,
            0x10 => self.status,
            _ => 0,
        }
    }

    pub fn write_reg(&mut self, paddr: u32, value: u32, rdram: &mut PhysicalMemory, mi: &mut Mi) {
        let Some(off) = Self::reg_offset(paddr) else {
            return;
        };
        match off {
            0x00 => self.dram_addr = value,
            0x04 => self.cart_addr = value,
            0x0C => self.dma_read(rdram, value, mi),
            0x10 => {
                self.dma_write(rdram, value, mi);
            }
            _ => {}
        }
    }

    /// Cartridge ROM → RDRAM via the same DMA engine as MMIO (`PI_DRAM_ADDR`, `PI_CART_ADDR`, `PI_RD_LEN`).
    pub fn dma_cart_segment_to_rdram(
        &mut self,
        rdram: &mut PhysicalMemory,
        cart_bus_addr: u32,
        rdram_phys: u32,
        len: u32,
        mi: &mut Mi,
    ) {
        if len == 0 {
            return;
        }
        self.cart_addr = cart_bus_addr;
        self.dram_addr = rdram_phys;
        self.dma_read(rdram, len - 1, mi);
    }

    /// PI “read” DMA: cartridge ROM → RDRAM. `len_minus_one` is the value written to PI_RD_LEN.
    fn dma_read(&mut self, rdram: &mut PhysicalMemory, len_minus_one: u32, mi: &mut Mi) {
        let len = (len_minus_one as u64).saturating_add(1);
        if len == 0 {
            return;
        }
        // bit0 busy, bit1 interrupt (set at completion); clear old interrupt when starting.
        self.status = (self.status & !2) | 1;

        let rdram_mask = (rdram.data.len() as u64).saturating_sub(1);
        let mut dram = (self.dram_addr as u64) & rdram_mask;

        for i in 0..len {
            let cart_p = self.cart_addr.wrapping_add(i as u32);
            let b = self.read_cart_u8(cart_p);
            let p = (dram as u32) & (rdram_mask as u32);
            rdram.write_u8(p, b);
            dram = dram.wrapping_add(1) & rdram_mask;
        }

        self.pending_cycles = self.pending_cycles.saturating_add(len.saturating_mul(PI_CYCLES_PER_DMA_BYTE));
        self.status = (self.status & !1) | 2;
        mi.raise(MI_INTR_PI);
    }

    /// PI “write” DMA: RDRAM → cart (save chips). No writable retail ROM; consume cycles only.
    fn dma_write(&mut self, rdram: &mut PhysicalMemory, len_minus_one: u32, mi: &mut Mi) {
        let len = (len_minus_one as u64).saturating_add(1);
        self.pending_cycles = self.pending_cycles.saturating_add(len.saturating_mul(PI_CYCLES_PER_DMA_BYTE));
        let _ = rdram;
        self.status |= 2;
        mi.raise(MI_INTR_PI);
    }

    fn read_cart_u8(&self, cart_addr: u32) -> u8 {
        let off = cart_rom_offset(cart_addr) as usize;
        self.rom.get(off).copied().unwrap_or(0)
    }

    pub fn read_cart_u32(&self, cart_addr: u32) -> u32 {
        let off = cart_rom_offset(cart_addr) as usize;
        if off + 4 > self.rom.len() {
            return 0;
        }
        let s = &self.rom[off..off + 4];
        u32::from_be_bytes([s[0], s[1], s[2], s[3]])
    }
}

impl Default for Pi {
    fn default() -> Self {
        Self::new()
    }
}

#[inline]
pub fn cart_rom_offset(cart_addr: u32) -> u32 {
    cart_addr.wrapping_sub(CART_DOM1_ADDR2_BASE) & 0x0FFF_FFFF
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bus::Bus;
    use crate::mi::Mi;

    #[test]
    fn pi_dma_copy_header_to_rdram() {
        let mut rom = vec![0u8; 0x1000];
        rom[0x40..0x44].copy_from_slice(&0x8037_0012u32.to_be_bytes());

        let mut pi = Pi::with_rom(rom);
        let mut rdram = PhysicalMemory::new(4 * 1024 * 1024);
        let mut mi = Mi::new();
        pi.cart_addr = CART_DOM1_ADDR2_BASE + 0x40;
        pi.dram_addr = 0x0000_0000;

        pi.write_reg(PI_REGS_BASE + 0x0C, 3, &mut rdram, &mut mi);

        assert_eq!(rdram.read_u32(0).unwrap(), 0x8037_0012);
        assert_ne!(mi.intr & MI_INTR_PI, 0);
    }
}
