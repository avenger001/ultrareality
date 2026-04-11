//! Physical memory, PI/cartridge, RSP DMEM/IMEM, and RCP MMIO (MI/VI/AI/SI/PIF).

use crate::ai::{Ai, AI_REGS_BASE, AI_REGS_LEN};
use crate::mi::{Mi, MI_REGS_BASE, MI_REGS_LEN};
use crate::pif::{Pif, PIF_ROM_START, PIF_WINDOW_END};
use crate::pi::{Pi, CART_DOM1_ADDR2_BASE, PI_REGS_BASE};
use crate::si::{Si, SI_REGS_BASE, SI_REGS_LEN};
use crate::vi::{Vi, VI_REGS_BASE, VI_REGS_LEN};

/// Default retail RDRAM size (4 MiB). Expansion Pak (8 MiB) can be enabled later.
pub const DEFAULT_RDRAM_SIZE: usize = 4 * 1024 * 1024;

pub const RSP_DMEM_START: u32 = 0x0400_0000;
pub const RSP_DMEM_END: u32 = 0x0400_1000;
pub const RSP_IMEM_START: u32 = 0x0400_1000;
pub const RSP_IMEM_END: u32 = 0x0400_2000;

pub trait Bus {
    fn read_u32(&mut self, paddr: u32) -> Option<u32>;
    fn write_u32(&mut self, paddr: u32, value: u32);
    fn read_u8(&mut self, paddr: u32) -> Option<u8>;
    fn write_u8(&mut self, paddr: u32, value: u8);
}

/// Contiguous big-endian RAM backing store (RDRAM).
pub struct PhysicalMemory {
    pub data: Box<[u8]>,
}

impl PhysicalMemory {
    pub fn new(size: usize) -> Self {
        Self {
            data: vec![0u8; size].into_boxed_slice(),
        }
    }

    #[inline]
    fn in_bounds(&self, paddr: u32, len: u32) -> bool {
        (paddr as u64) + (len as u64) <= self.data.len() as u64
    }
}

impl Bus for PhysicalMemory {
    fn read_u32(&mut self, paddr: u32) -> Option<u32> {
        if !self.in_bounds(paddr, 4) {
            return None;
        }
        let i = paddr as usize;
        Some(u32::from_be_bytes(self.data[i..i + 4].try_into().unwrap()))
    }

    fn write_u32(&mut self, paddr: u32, value: u32) {
        if !self.in_bounds(paddr, 4) {
            return;
        }
        let i = paddr as usize;
        self.data[i..i + 4].copy_from_slice(&value.to_be_bytes());
    }

    fn read_u8(&mut self, paddr: u32) -> Option<u8> {
        if !self.in_bounds(paddr, 1) {
            return None;
        }
        Some(self.data[paddr as usize])
    }

    fn write_u8(&mut self, paddr: u32, value: u8) {
        if !self.in_bounds(paddr, 1) {
            return;
        }
        self.data[paddr as usize] = value;
    }
}

/// Map CPU virtual address to physical (direct segments; **TLB not implemented**).
///
/// `kseg0` / `kseg1` (`0x80000000`–`0xBFFFFFFF`): physical = `vaddr & 0x1FFF_FFFF`.
/// `kuseg` low (`0x00000000`–`0x7FFFFFFF`): identity-mapped physical for bring-up.
#[inline]
pub fn virt_to_phys(vaddr: u64) -> Option<u32> {
    let v = vaddr as u32;
    if (0x8000_0000..=0xBFFF_FFFF).contains(&v) {
        return Some(v & 0x1FFF_FFFF);
    }
    if v < 0x8000_0000 {
        return Some(v);
    }
    None
}

/// RDRAM-only helper used by unit tests that only install instructions in low physical RAM.
#[inline]
pub fn virt_to_phys_rdram(vaddr: u64, rdram_size: usize) -> Option<u32> {
    let p = virt_to_phys(vaddr)?;
    if (p as usize) < rdram_size {
        Some(p)
    } else {
        None
    }
}

/// Full physical map: RDRAM, RSP, RCP MMIO, PI, cartridge ROM, PIF.
pub struct SystemBus {
    pub rdram: PhysicalMemory,
    pub rsp_dmem: Box<[u8; RSP_DMEM_END as usize - RSP_DMEM_START as usize]>,
    pub rsp_imem: Box<[u8; RSP_IMEM_END as usize - RSP_IMEM_START as usize]>,
    pub mi: Mi,
    pub vi: Vi,
    pub ai: Ai,
    pub si: Si,
    pub pif: Pif,
    pub pi: Pi,
}

impl SystemBus {
    pub fn new() -> Self {
        Self {
            rdram: PhysicalMemory::new(DEFAULT_RDRAM_SIZE),
            rsp_dmem: Box::new([0u8; (RSP_DMEM_END - RSP_DMEM_START) as usize]),
            rsp_imem: Box::new([0u8; (RSP_IMEM_END - RSP_IMEM_START) as usize]),
            mi: Mi::new(),
            vi: Vi::new(),
            ai: Ai::new(),
            si: Si::new(),
            pif: Pif::new(),
            pi: Pi::new(),
        }
    }

    pub fn with_rdram_size(rdram_size: usize) -> Self {
        Self {
            rdram: PhysicalMemory::new(rdram_size),
            rsp_dmem: Box::new([0u8; (RSP_DMEM_END - RSP_DMEM_START) as usize]),
            rsp_imem: Box::new([0u8; (RSP_IMEM_END - RSP_IMEM_START) as usize]),
            mi: Mi::new(),
            vi: Vi::new(),
            ai: Ai::new(),
            si: Si::new(),
            pif: Pif::new(),
            pi: Pi::new(),
        }
    }

    pub fn drain_pi_cycles(&mut self) -> u64 {
        self.pi.drain_cycles()
    }

    /// PI DMA + SI DMA cycle debt (until RCP runs on the master timeline).
    pub fn drain_deferred_cycles(&mut self) -> u64 {
        self.drain_pi_cycles().saturating_add(self.si.drain_cycles())
    }

    #[inline]
    fn rdram_len_u32(&self) -> u32 {
        self.rdram.data.len() as u32
    }

    fn rsp_read_u8(&self, paddr: u32) -> Option<u8> {
        if (RSP_DMEM_START..RSP_DMEM_END).contains(&paddr) {
            return Some(self.rsp_dmem[(paddr - RSP_DMEM_START) as usize]);
        }
        if (RSP_IMEM_START..RSP_IMEM_END).contains(&paddr) {
            return Some(self.rsp_imem[(paddr - RSP_IMEM_START) as usize]);
        }
        None
    }

    fn rsp_write_u8(&mut self, paddr: u32, v: u8) {
        if (RSP_DMEM_START..RSP_DMEM_END).contains(&paddr) {
            self.rsp_dmem[(paddr - RSP_DMEM_START) as usize] = v;
        } else if (RSP_IMEM_START..RSP_IMEM_END).contains(&paddr) {
            self.rsp_imem[(paddr - RSP_IMEM_START) as usize] = v;
        }
    }
}

impl Bus for SystemBus {
    fn read_u32(&mut self, paddr: u32) -> Option<u32> {
        if paddr & 3 != 0 {
            return None;
        }
        if paddr.saturating_add(4) <= self.rdram_len_u32() {
            return self.rdram.read_u32(paddr);
        }
        if (RSP_DMEM_START..RSP_DMEM_END).contains(&paddr) {
            let i = (paddr - RSP_DMEM_START) as usize;
            let b = [
                self.rsp_dmem[i],
                self.rsp_dmem[i + 1],
                self.rsp_dmem[i + 2],
                self.rsp_dmem[i + 3],
            ];
            return Some(u32::from_be_bytes(b));
        }
        if (RSP_IMEM_START..RSP_IMEM_END).contains(&paddr) {
            let i = (paddr - RSP_IMEM_START) as usize;
            let b = [
                self.rsp_imem[i],
                self.rsp_imem[i + 1],
                self.rsp_imem[i + 2],
                self.rsp_imem[i + 3],
            ];
            return Some(u32::from_be_bytes(b));
        }
        if (MI_REGS_BASE..MI_REGS_BASE + MI_REGS_LEN as u32).contains(&paddr) {
            return Some(self.mi.read(paddr));
        }
        if (VI_REGS_BASE..VI_REGS_BASE + VI_REGS_LEN as u32).contains(&paddr) {
            return Some(self.vi.read(paddr));
        }
        if (AI_REGS_BASE..AI_REGS_BASE + AI_REGS_LEN as u32).contains(&paddr) {
            return Some(self.ai.read(paddr));
        }
        if (PI_REGS_BASE..PI_REGS_BASE + 0x20).contains(&paddr) {
            return Some(self.pi.read_reg(paddr));
        }
        if (SI_REGS_BASE..SI_REGS_BASE + SI_REGS_LEN as u32).contains(&paddr) {
            return Some(self.si.read(paddr));
        }
        if (PIF_ROM_START..PIF_WINDOW_END).contains(&paddr) {
            return self.pif.read_u32(paddr);
        }
        if (CART_DOM1_ADDR2_BASE..=0x1FBF_FFFF).contains(&paddr) {
            return Some(self.pi.read_cart_u32(paddr));
        }
        None
    }

    fn write_u32(&mut self, paddr: u32, value: u32) {
        if paddr & 3 != 0 {
            return;
        }
        if paddr.saturating_add(4) <= self.rdram_len_u32() {
            self.rdram.write_u32(paddr, value);
            return;
        }
        if (MI_REGS_BASE..MI_REGS_BASE + MI_REGS_LEN as u32).contains(&paddr) {
            self.mi.write(paddr, value);
            return;
        }
        if (VI_REGS_BASE..VI_REGS_BASE + VI_REGS_LEN as u32).contains(&paddr) {
            self.vi.write(paddr, value);
            return;
        }
        if (AI_REGS_BASE..AI_REGS_BASE + AI_REGS_LEN as u32).contains(&paddr) {
            self.ai.write(paddr, value);
            return;
        }
        if (PI_REGS_BASE..PI_REGS_BASE + 0x20).contains(&paddr) {
            self.pi.write_reg(paddr, value, &mut self.rdram);
            return;
        }
        if (SI_REGS_BASE..SI_REGS_BASE + SI_REGS_LEN as u32).contains(&paddr) {
            self.si.write(paddr, value, &mut self.rdram, &mut self.pif);
            return;
        }
        if (PIF_ROM_START..PIF_WINDOW_END).contains(&paddr) {
            self.pif.write_u32(paddr, value);
            return;
        }
        if (RSP_DMEM_START..RSP_DMEM_END).contains(&paddr) {
            let i = (paddr - RSP_DMEM_START) as usize;
            self.rsp_dmem[i..i + 4].copy_from_slice(&value.to_be_bytes());
            return;
        }
        if (RSP_IMEM_START..RSP_IMEM_END).contains(&paddr) {
            let i = (paddr - RSP_IMEM_START) as usize;
            self.rsp_imem[i..i + 4].copy_from_slice(&value.to_be_bytes());
        }
    }

    fn read_u8(&mut self, paddr: u32) -> Option<u8> {
        if (paddr as usize) < self.rdram.data.len() {
            return self.rdram.read_u8(paddr);
        }
        if let Some(b) = self.rsp_read_u8(paddr) {
            return Some(b);
        }
        if (PIF_ROM_START..PIF_WINDOW_END).contains(&paddr) {
            return self.pif.read_u8(paddr);
        }
        let wa = paddr & !3;
        if let Some(w) = self.read_u32(wa) {
            let shift = 8 * (3 - (paddr & 3));
            return Some(((w >> shift) & 0xFF) as u8);
        }
        None
    }

    fn write_u8(&mut self, paddr: u32, value: u8) {
        if (paddr as usize) < self.rdram.data.len() {
            self.rdram.write_u8(paddr, value);
            return;
        }
        if (RSP_DMEM_START..RSP_DMEM_END).contains(&paddr)
            || (RSP_IMEM_START..RSP_IMEM_END).contains(&paddr)
        {
            self.rsp_write_u8(paddr, value);
            return;
        }
        if (PIF_ROM_START..PIF_WINDOW_END).contains(&paddr) {
            self.pif.write_u8(paddr, value);
            return;
        }
        let wa = paddr & !3;
        if let Some(cur) = self.read_u32(wa) {
            let shift = 8 * (3 - (paddr & 3));
            let new = (cur & !(0xFFu32 << shift)) | ((value as u32) << shift);
            self.write_u32(wa, new);
        }
    }
}

impl Default for SystemBus {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mi::MI_VERSION_DEFAULT;

    #[test]
    fn mi_version_register() {
        let mut bus = SystemBus::with_rdram_size(1024 * 1024);
        assert_eq!(
            bus.read_u32(0x0430_0004),
            Some(MI_VERSION_DEFAULT)
        );
    }
}
