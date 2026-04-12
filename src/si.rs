//! Serial Interface (SI): PIF communication and 64-byte DMA to/from PIF RAM.
//!
//! `RD64B` / `WR64B` DMA completes after [`SI_DMA_64_BLOCK_CYCLES`] on the RCP timeline; data moves and
//! `MI_INTR_SI` are deferred until then ([`Si::advance_time`]).

use crate::bus::{Bus, PhysicalMemory};
use crate::mi::{Mi, MI_INTR_SI};
use crate::pif::{Pif, PIF_RAM_LEN, PIF_RAM_START};
use crate::timing::SI_DMA_64_BLOCK_CYCLES;

pub const SI_REGS_BASE: u32 = 0x0480_0000;
pub const SI_REGS_LEN: usize = 0x20;

/// `SI_DRAM_ADDR` — RDRAM physical for the 64-byte SI DMA ([n64brew: SI](https://n64brew.dev/wiki/Serial_Interface)).
pub const SI_REG_DRAM_ADDR: u32 = 0x00;
/// PIF address for **PIF RAM → RDRAM** DMA (`RD64B`).
pub const SI_REG_PIF_ADDR_RD64B: u32 = 0x04;
/// PIF address for **RDRAM → PIF RAM** DMA (`WR64B`).
pub const SI_REG_PIF_ADDR_WR64B: u32 = 0x10;
/// `SI_STATUS` — busy / complete bits.
pub const SI_REG_STATUS: u32 = 0x18;

/// RCP cycles charged per 64-byte SI DMA ([`SI_DMA_64_BLOCK_CYCLES`](crate::timing::SI_DMA_64_BLOCK_CYCLES)).
pub const SI_DMA_CYCLES: u64 = SI_DMA_64_BLOCK_CYCLES;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SiDmaKind {
    Rd64,
    Wr64,
}

#[derive(Debug, Clone)]
struct SiDmaActive {
    remaining_rcp_cycles: u64,
    kind: SiDmaKind,
    dram_addr: u32,
    pif_base: usize,
}

#[derive(Debug)]
pub struct Si {
    pub dram_addr: u32,
    /// Last PIF address written for a **read** DMA (`SI_REG_PIF_ADDR_RD64B`).
    pub pif_addr_rd: u32,
    /// Last PIF address written for a **write** DMA (`SI_REG_PIF_ADDR_WR64B`).
    pub pif_addr_wr: u32,
    /// Bits 0–1: busy during DMA; bit 2: completion (until next DMA starts).
    pub status: u32,
    active: Option<SiDmaActive>,
}

impl Si {
    pub fn new() -> Self {
        Self {
            dram_addr: 0,
            pif_addr_rd: 0,
            pif_addr_wr: 0,
            status: 0,
            active: None,
        }
    }

    #[inline]
    pub fn dma_busy(&self) -> bool {
        self.active.is_some()
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
            SI_REG_DRAM_ADDR => self.dram_addr,
            SI_REG_PIF_ADDR_RD64B => self.pif_addr_rd,
            SI_REG_PIF_ADDR_WR64B => self.pif_addr_wr,
            SI_REG_STATUS => self.status,
            _ => 0,
        }
    }

    /// Writes to [`SI_REG_PIF_ADDR_RD64B`] / [`SI_REG_PIF_ADDR_WR64B`] start DMA and update the matching latch.
    pub fn write(
        &mut self,
        paddr: u32,
        value: u32,
        _rdram: &mut PhysicalMemory,
        _pif: &mut Pif,
        mi: &mut Mi,
    ) {
        let Some(off) = Self::offset(paddr) else {
            return;
        };
        match off {
            SI_REG_DRAM_ADDR => self.dram_addr = value,
            SI_REG_PIF_ADDR_RD64B => {
                self.pif_addr_rd = value;
                self.dma_rd64_kick(value, mi);
            }
            SI_REG_PIF_ADDR_WR64B => {
                self.pif_addr_wr = value;
                self.dma_wr64_kick(value, mi);
            }
            _ => {}
        }
    }

    fn dma_rd64_kick(&mut self, pif_addr: u32, mi: &mut Mi) {
        mi.clear(MI_INTR_SI);
        self.status = (self.status & !4) | 3;
        self.active = Some(SiDmaActive {
            remaining_rcp_cycles: SI_DMA_64_BLOCK_CYCLES,
            kind: SiDmaKind::Rd64,
            dram_addr: self.dram_addr,
            pif_base: pif_ram_byte_index(pif_addr),
        });
    }

    fn dma_wr64_kick(&mut self, pif_addr: u32, mi: &mut Mi) {
        mi.clear(MI_INTR_SI);
        self.status = (self.status & !4) | 3;
        self.active = Some(SiDmaActive {
            remaining_rcp_cycles: SI_DMA_64_BLOCK_CYCLES,
            kind: SiDmaKind::Wr64,
            dram_addr: self.dram_addr,
            pif_base: pif_ram_byte_index(pif_addr),
        });
    }

    /// Apply `delta` RCP cycles to in-flight SI DMA; performs the 64-byte transfer and raises `MI_INTR_SI` when due.
    pub fn advance_time(
        &mut self,
        delta: u64,
        rdram: &mut PhysicalMemory,
        pif: &mut Pif,
        mi: &mut Mi,
    ) {
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
            self.finish_dma(op, rdram, pif, mi);
        }
    }

    fn finish_dma(&mut self, op: SiDmaActive, rdram: &mut PhysicalMemory, pif: &mut Pif, mi: &mut Mi) {
        match op.kind {
            SiDmaKind::Rd64 => {
                let dram = (op.dram_addr & 0x00FF_FFFF) as usize;
                let rdram_len = rdram.data.len();
                for i in 0..PIF_RAM_LEN {
                    let b = pif.ram[op.pif_base.wrapping_add(i) & (PIF_RAM_LEN - 1)];
                    let p = dram.wrapping_add(i);
                    if p < rdram_len {
                        rdram.write_u8(p as u32, b);
                    }
                }
            }
            SiDmaKind::Wr64 => {
                let dram = (op.dram_addr & 0x00FF_FFFF) as usize;
                let rdram_len = rdram.data.len();
                for i in 0..PIF_RAM_LEN {
                    let p = dram.wrapping_add(i);
                    let b = if p < rdram_len {
                        rdram.read_u8(p as u32).unwrap_or(0)
                    } else {
                        0
                    };
                    pif.ram[op.pif_base.wrapping_add(i) & (PIF_RAM_LEN - 1)] = b;
                }
            }
        }
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

        si.write(SI_REGS_BASE + SI_REG_DRAM_ADDR, 0x0010_0000, &mut rdram, &mut pif, &mut mi);
        si.write(
            SI_REGS_BASE + SI_REG_PIF_ADDR_RD64B,
            PIF_RAM_START,
            &mut rdram,
            &mut pif,
            &mut mi,
        );
        assert_eq!(si.read(SI_REGS_BASE + SI_REG_PIF_ADDR_RD64B), PIF_RAM_START);
        assert!(si.dma_busy());
        si.advance_time(SI_DMA_64_BLOCK_CYCLES, &mut rdram, &mut pif, &mut mi);
        assert_ne!(mi.intr & MI_INTR_SI, 0);
        mi.clear(MI_INTR_SI);

        pif.ram[0..4].fill(0);
        si.write(
            SI_REGS_BASE + SI_REG_PIF_ADDR_WR64B,
            PIF_RAM_START,
            &mut rdram,
            &mut pif,
            &mut mi,
        );
        assert_eq!(si.read(SI_REGS_BASE + SI_REG_PIF_ADDR_WR64B), PIF_RAM_START);
        si.advance_time(SI_DMA_64_BLOCK_CYCLES, &mut rdram, &mut pif, &mut mi);

        assert_eq!(&pif.ram[0..4], &0xDEAD_BEEFu32.to_be_bytes());
        assert_ne!(mi.intr & MI_INTR_SI, 0);
    }
}
