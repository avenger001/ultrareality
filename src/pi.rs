//! Parallel Interface (PI): cartridge ROM access and PI DMA into RDRAM.
//!
//! Cart → RDRAM DMA is **deferred**: the transfer runs to completion after enough RCP cycles
//! elapse on the master timeline ([`Pi::advance_time`]), keeping `PI_STATUS` bit0 busy until then.

use crate::bus::{Bus, PhysicalMemory};
use crate::mi::{Mi, MI_INTR_PI};
use crate::timing::{pi_cart_dma_total_cycles, PI_ROM_DMA_CYCLES_PER_BYTE};

/// PI register block base (physical).
pub const PI_REGS_BASE: u32 = 0x0460_0000;
pub const PI_REGS_LEN: usize = 0x20;

/// Cartridge ROM mapping base (physical).
pub const CART_DOM1_ADDR2_BASE: u32 = 0x1000_0000;

/// ROM byte offset used in PI DMA tests as a source dword (not a hardware constant; arbitrary pattern location).
pub const CART_ROM_TEST_DWORD_OFF: usize = 0x40;

/// `PI_DRAM_ADDR` / `PI_CART_ADDR` / length registers ([n64brew: PI](https://n64brew.dev/wiki/Peripheral_Interface)).
pub const PI_REG_DRAM_ADDR: u32 = 0x00;
pub const PI_REG_CART_ADDR: u32 = 0x04;
/// Cart → RDRAM DMA “read”; write starts transfer; read returns last written length − 1 (low 24 bits).
pub const PI_REG_RD_LEN: u32 = 0x0C;
/// RDRAM → cart DMA “write”; write starts transfer. **Read** at `0x10` is [`Pi::status`] (`PI_STATUS`).
pub const PI_REG_WR_LEN: u32 = 0x10;

/// RCP cycles per byte for PI cart ROM DMA ([`PI_ROM_DMA_CYCLES_PER_BYTE`](crate::timing::PI_ROM_DMA_CYCLES_PER_BYTE)).
pub const PI_CYCLES_PER_DMA_BYTE: u64 = PI_ROM_DMA_CYCLES_PER_BYTE;

#[derive(Debug, Clone)]
struct PiDmaActive {
    remaining_rcp_cycles: u64,
    cart_addr: u32,
    dram_addr: u32,
    len: u64,
    cart_to_rdram: bool,
}

#[derive(Debug)]
pub struct Pi {
    pub rom: Vec<u8>,
    pub dram_addr: u32,
    pub cart_addr: u32,
    /// Last value written to `PI_RD_LEN` (length − 1 field); returned on read.
    pub rd_len: u32,
    /// Last value written to `PI_WR_LEN` (length − 1); for debugging / extension.
    pub wr_len: u32,
    /// Minimal status: bit0 = DMA busy, bit1 = interrupt (completion).
    pub status: u32,
    active: Option<PiDmaActive>,
}

impl Pi {
    pub fn new() -> Self {
        Self {
            rom: Vec::new(),
            dram_addr: 0,
            cart_addr: 0,
            rd_len: 0,
            wr_len: 0,
            status: 0,
            active: None,
        }
    }

    pub fn with_rom(rom: Vec<u8>) -> Self {
        Self {
            rom,
            dram_addr: 0,
            cart_addr: 0,
            rd_len: 0,
            wr_len: 0,
            status: 0,
            active: None,
        }
    }

    /// True while a PI DMA is in flight (matches `PI_STATUS` bit0).
    #[inline]
    pub fn dma_busy(&self) -> bool {
        self.active.is_some()
    }

    fn reg_offset(paddr: u32) -> Option<u32> {
        if (PI_REGS_BASE..PI_REGS_BASE + PI_REGS_LEN as u32).contains(&paddr) {
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
            PI_REG_DRAM_ADDR => self.dram_addr,
            PI_REG_CART_ADDR => self.cart_addr,
            PI_REG_RD_LEN => self.rd_len & 0x00FF_FFFF,
            PI_REG_WR_LEN => self.status,
            _ => 0,
        }
    }

    pub fn write_reg(&mut self, paddr: u32, value: u32, _rdram: &mut PhysicalMemory, mi: &mut Mi) {
        let Some(off) = Self::reg_offset(paddr) else {
            return;
        };
        match off {
            PI_REG_DRAM_ADDR => self.dram_addr = value,
            PI_REG_CART_ADDR => self.cart_addr = value,
            PI_REG_RD_LEN => {
                self.rd_len = value & 0x00FF_FFFF;
                self.dma_read_kick(self.rd_len, mi);
            }
            PI_REG_WR_LEN => {
                self.wr_len = value & 0x00FF_FFFF;
                self.dma_write_kick(self.wr_len, mi);
            }
            _ => {}
        }
    }

    /// Cartridge ROM → RDRAM via the same DMA engine as MMIO (`PI_DRAM_ADDR`, `PI_CART_ADDR`, `PI_RD_LEN`).
    /// Fast-forwards the in-flight timer so IPL-style loaders see RDRAM filled without stepping the CPU.
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
        self.dma_read_kick(len - 1, mi);
        self.advance_time(u64::MAX, rdram, mi);
    }

    fn dma_read_kick(&mut self, len_minus_one: u32, mi: &mut Mi) {
        let len = (len_minus_one as u64).saturating_add(1);
        if len == 0 {
            return;
        }
        mi.clear(MI_INTR_PI);
        self.status = (self.status & !2) | 1;
        let cost = pi_cart_dma_total_cycles(len);
        self.active = Some(PiDmaActive {
            remaining_rcp_cycles: cost,
            cart_addr: self.cart_addr,
            dram_addr: self.dram_addr,
            len,
            cart_to_rdram: true,
        });
    }

    fn dma_write_kick(&mut self, len_minus_one: u32, mi: &mut Mi) {
        let len = (len_minus_one as u64).saturating_add(1);
        if len == 0 {
            return;
        }
        mi.clear(MI_INTR_PI);
        self.status = (self.status & !2) | 1;
        let cost = pi_cart_dma_total_cycles(len);
        self.active = Some(PiDmaActive {
            remaining_rcp_cycles: cost,
            cart_addr: self.cart_addr,
            dram_addr: self.dram_addr,
            len,
            cart_to_rdram: false,
        });
    }

    /// Apply `delta` RCP cycles to any in-flight PI DMA; completes copies and raises `MI_INTR_PI` when due.
    pub fn advance_time(&mut self, delta: u64, rdram: &mut PhysicalMemory, mi: &mut Mi) {
        let mut d = delta;
        while d > 0 {
            let Some(active) = self.active.as_mut() else {
                return;
            };
            let use_cycles = active.remaining_rcp_cycles.min(d);
            active.remaining_rcp_cycles -= use_cycles;
            d -= use_cycles;
            if active.remaining_rcp_cycles > 0 {
                return;
            }
            let op = self.active.take().unwrap();
            self.finish_dma(op, rdram, mi);
        }
    }

    fn finish_dma(&mut self, op: PiDmaActive, rdram: &mut PhysicalMemory, mi: &mut Mi) {
        if op.cart_to_rdram {
            let rdram_mask = (rdram.data.len() as u64).saturating_sub(1);
            let mut dram = (op.dram_addr as u64) & rdram_mask;
            for i in 0..op.len {
                let cart_p = op.cart_addr.wrapping_add(i as u32);
                let b = self.read_cart_u8(cart_p);
                let p = (dram as u32) & (rdram_mask as u32);
                rdram.write_u8(p, b);
                dram = dram.wrapping_add(1) & rdram_mask;
            }
        }
        self.status = (self.status & !1) | 2;
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
        rom[CART_ROM_TEST_DWORD_OFF..CART_ROM_TEST_DWORD_OFF + 4]
            .copy_from_slice(&0x8037_0012u32.to_be_bytes());

        let mut pi = Pi::with_rom(rom);
        let mut rdram = PhysicalMemory::new(4 * 1024 * 1024);
        let mut mi = Mi::new();
        pi.cart_addr = CART_DOM1_ADDR2_BASE + CART_ROM_TEST_DWORD_OFF as u32;
        pi.dram_addr = 0x0000_0000;

        pi.write_reg(PI_REGS_BASE + PI_REG_RD_LEN, 3, &mut rdram, &mut mi);
        assert!(pi.dma_busy());
        assert_ne!(rdram.read_u32(0).unwrap(), 0x8037_0012);

        pi.advance_time(1000, &mut rdram, &mut mi);
        assert!(!pi.dma_busy());
        assert_eq!(rdram.read_u32(0).unwrap(), 0x8037_0012);
        assert_ne!(mi.intr & MI_INTR_PI, 0);
        assert_eq!(pi.read_reg(PI_REGS_BASE + PI_REG_RD_LEN), 3);
    }

    #[test]
    fn pi_rd_len_readback_after_dma() {
        let rom = vec![0xAAu8; 0x10];
        let mut pi = Pi::with_rom(rom);
        let mut rdram = PhysicalMemory::new(4 * 1024 * 1024);
        let mut mi = Mi::new();
        pi.cart_addr = CART_DOM1_ADDR2_BASE;
        pi.dram_addr = 0;
        pi.write_reg(PI_REGS_BASE + PI_REG_RD_LEN, 0, &mut rdram, &mut mi);
        assert_eq!(pi.read_reg(PI_REGS_BASE + PI_REG_RD_LEN), 0);
    }
}
