//! Serial Interface (SI): PIF communication and 64-byte DMA to/from PIF RAM.

use crate::bus::{Bus, PhysicalMemory};
use crate::mi::{Mi, MI_INTR_SI};
use crate::pif::{Pif, PIF_RAM_LEN, PIF_RAM_START};

pub const SI_REGS_BASE: u32 = 0x0480_0000;
pub const SI_REGS_LEN: usize = 0x20;

/// Cycles charged per 64-byte SI DMA (stub until bus timing is modeled).
pub const SI_DMA_CYCLES: u64 = 640;

#[derive(Debug)]
pub struct Si {
    pub dram_addr: u32,
    /// Full PIF-side address used for DMA (often `0x1FC007C0` for RAM).
    pub pif_addr: u32,
    /// Bits 0–1: busy during DMA; bit 2: completion (until next DMA starts).
    pub status: u32,
    pending_cycles: u64,
}

impl Si {
    pub fn new() -> Self {
        Self {
            dram_addr: 0,
            pif_addr: 0,
            status: 0,
            pending_cycles: 0,
        }
    }

    pub fn drain_cycles(&mut self) -> u64 {
        let c = self.pending_cycles;
        self.pending_cycles = 0;
        c
    }

    pub fn offset(paddr: u32) -> Option<u32> {
        if (SI_REGS_BASE..SI_REGS_BASE + SI_REGS_LEN as u32).contains(&paddr) {
            Some(paddr - SI_REGS_BASE)
        } else {
            None
        }
    }

    pub fn read(&self, paddr: u32) -> u32 {
        let Some(off) = Self::offset(paddr) else {
            return 0;
        };
        match off {
            0x00 => self.dram_addr,
            0x04 => self.pif_addr,
            0x10 => self.pif_addr,
            0x18 => self.status,
            _ => 0,
        }
    }

    /// Writes to `SI_PIF_ADDR_RD64B` / `SI_PIF_ADDR_WR64B` load `pif_addr` and start DMA.
    pub fn write(
        &mut self,
        paddr: u32,
        value: u32,
        rdram: &mut PhysicalMemory,
        pif: &mut Pif,
        mi: &mut Mi,
    ) {
        let Some(off) = Self::offset(paddr) else {
            return;
        };
        match off {
            0x00 => self.dram_addr = value,
            0x04 => {
                self.pif_addr = value;
                self.dma_rd64(rdram, pif, mi);
            }
            0x10 => {
                self.pif_addr = value;
                self.dma_wr64(rdram, pif, mi);
            }
            _ => {}
        }
    }

    /// DMA: PIF RAM → RDRAM (64 bytes).
    fn dma_rd64(&mut self, rdram: &mut PhysicalMemory, pif: &Pif, mi: &mut Mi) {
        self.status = (self.status & !4) | 3;
        let base = pif_ram_byte_index(self.pif_addr);
        let dram = (self.dram_addr & 0x00FF_FFFF) as usize;
        let rdram_len = rdram.data.len();
        for i in 0..PIF_RAM_LEN {
            let b = pif.ram[base.wrapping_add(i) & (PIF_RAM_LEN - 1)];
            let p = dram.wrapping_add(i);
            if p < rdram_len {
                rdram.write_u8(p as u32, b);
            }
        }
        self.pending_cycles = self.pending_cycles.saturating_add(SI_DMA_CYCLES);
        self.status = (self.status & !3) | 4;
        mi.raise(MI_INTR_SI);
    }

    /// DMA: RDRAM → PIF RAM (64 bytes).
    fn dma_wr64(&mut self, rdram: &mut PhysicalMemory, pif: &mut Pif, mi: &mut Mi) {
        self.status = (self.status & !4) | 3;
        let base = pif_ram_byte_index(self.pif_addr);
        let dram = (self.dram_addr & 0x00FF_FFFF) as usize;
        let rdram_len = rdram.data.len();
        for i in 0..PIF_RAM_LEN {
            let p = dram.wrapping_add(i);
            let b = if p < rdram_len {
                rdram.read_u8(p as u32).unwrap_or(0)
            } else {
                0
            };
            pif.ram[base.wrapping_add(i) & (PIF_RAM_LEN - 1)] = b;
        }
        self.pending_cycles = self.pending_cycles.saturating_add(SI_DMA_CYCLES);
        self.status = (self.status & !3) | 4;
        mi.raise(MI_INTR_SI);
    }
}

#[inline]
fn pif_ram_byte_index(pif_addr: u32) -> usize {
    (pif_addr.wrapping_sub(PIF_RAM_START) as usize) & (PIF_RAM_LEN - 1)
}

impl Default for Si {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bus::PhysicalMemory;
    use crate::mi::Mi;

    #[test]
    fn si_dma_round_trip_pif_ram() {
        let mut si = Si::new();
        let mut rdram = PhysicalMemory::new(4 * 1024 * 1024);
        let mut pif = Pif::new();
        let mut mi = Mi::new();
        pif.ram[0..4].copy_from_slice(&0xDEAD_BEEFu32.to_be_bytes());

        // PIF RAM → RDRAM (SI read DMA)
        si.write(SI_REGS_BASE, 0x0010_0000, &mut rdram, &mut pif, &mut mi);
        si.write(SI_REGS_BASE + 0x04, PIF_RAM_START, &mut rdram, &mut pif, &mut mi);
        assert_ne!(mi.intr & MI_INTR_SI, 0);
        mi.clear(MI_INTR_SI);

        pif.ram[0..4].fill(0);
        // RDRAM → PIF RAM (SI write DMA)
        si.write(SI_REGS_BASE + 0x10, PIF_RAM_START, &mut rdram, &mut pif, &mut mi);

        assert_eq!(&pif.ram[0..4], &0xDEAD_BEEFu32.to_be_bytes());
        assert_ne!(mi.intr & MI_INTR_SI, 0);
    }
}
