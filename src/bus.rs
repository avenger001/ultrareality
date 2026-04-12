//! Physical memory, PI/cartridge, RSP DMEM/IMEM, and RCP MMIO (SP/DPC, MI/VI/AI/SI/PIF).

use crate::ai::{Ai, AI_REGS_BASE, AI_REGS_LEN};
use crate::mi::{Mi, MI_INTR_DP, MI_INTR_SP, MI_REGS_BASE, MI_REGS_LEN};
use crate::pif::{Pif, PIF_ROM_START, PIF_WINDOW_END};
use crate::pi::{Pi, CART_DOM1_ADDR2_BASE, PI_REGS_BASE, PI_REGS_LEN};
use crate::ri::{Ri, RI_REGS_BASE, RI_REGS_LEN};
use crate::rcp::{
    sp_dma_decode, sp_dma_end_addresses, DpcEndKick, DpcRegs, SpRegs, DPC_REGS_BASE,
    DPC_REGS_LEN, SP_PC_REG_IBIST, SP_PC_REG_PC, SP_PC_REGS_BASE, SP_PC_REGS_LEN, SP_REG_DMA_BUSY,
    SP_REG_DMA_FULL, SP_REG_RD_LEN, SP_REG_SEMAPHORE,
    SP_REG_STATUS, SP_REG_WR_LEN, SP_REGS_BASE, SP_REGS_LEN, SP_WORD_DRAM_ADDR, SP_WORD_MEM_ADDR,
};
use crate::rdp::Rdp;
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

/// Map CPU virtual address to physical **without** the TLB (bootstrap / PI helpers only).
///
/// `kseg0` / `kseg1` (`0x80000000`–`0xBFFFFFFF`): physical = `vaddr & 0x1FFF_FFFF`.
/// `kuseg` (`0x00000000`–`0x7FFFFFFF`): identity-mapped for DMA / loaders — **CPU** accesses use
/// [`crate::cpu::cop0::Cop0::translate_virt`].
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

#[derive(Debug)]
struct SpDmaPending {
    remaining_rcp_cycles: u64,
    len_reg: u32,
    mem: u32,
    dram: u32,
    to_rsp: bool,
}

#[derive(Debug)]
struct DpcRdpPending {
    remaining_rcp_cycles: u64,
    kick: DpcEndKick,
}

/// Full physical map: RDRAM, RSP, RCP MMIO, RI, PI, cartridge ROM, PIF.
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
    pub ri: Ri,
    pub sp_regs: SpRegs,
    pub dpc_regs: DpcRegs,
    pub rdp: Rdp,
    /// RDP list processing cycles billed to the master clock via [`SystemBus::drain_deferred_cycles`].
    pub rdp_deferred_cycles: u64,
    /// RSP halted (bit 0 of `SP_STATUS`); cold reset keeps the RSP stopped until software clears halt.
    pub sp_halted: bool,
    pub sp_broke: bool,
    /// `SP_SEMAPHORE` data bit; a read latches 1 until the register is written.
    pub sp_semaphore: u32,
    /// `SP_PC` (`0x0408_0000`): RSP scalar PC into IMEM (word-aligned, [`SP_PC_REG_PC`]).
    pub rsp_pc: u32,
    /// `SP_IBIST` — stub storage for bring-up.
    pub rsp_ibist: u32,
    /// RSP scalar GPR file (mirrors MIPS `r0`–`r31` for bring-up).
    pub rsp_scalar_regs: [u32; 32],
    sp_dma_pending: Option<SpDmaPending>,
    dpc_rdp_pending: Option<DpcRdpPending>,
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
            ri: Ri::new(),
            sp_regs: SpRegs::new(),
            dpc_regs: DpcRegs::new(),
            rdp: Rdp::new(),
            rdp_deferred_cycles: 0,
            sp_halted: true,
            sp_broke: false,
            sp_semaphore: 0,
            rsp_pc: 0,
            rsp_ibist: 0,
            rsp_scalar_regs: [0u32; 32],
            sp_dma_pending: None,
            dpc_rdp_pending: None,
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
            ri: Ri::new(),
            sp_regs: SpRegs::new(),
            dpc_regs: DpcRegs::new(),
            rdp: Rdp::new(),
            rdp_deferred_cycles: 0,
            sp_halted: true,
            sp_broke: false,
            sp_semaphore: 0,
            rsp_pc: 0,
            rsp_ibist: 0,
            rsp_scalar_regs: [0u32; 32],
            sp_dma_pending: None,
            dpc_rdp_pending: None,
        }
    }

    /// Charge VI readout of the current `VI_ORIGIN` / `VI_WIDTH` framebuffer (RGBA16 halfwords → RDRAM cycles).
    pub fn schedule_vi_frame_fetch(&mut self) {
        let px = self.vi.display_width() as u64 * self.vi.display_height() as u64;
        self.vi.charge_framebuffer_fetch_rgba16_pixels(px);
    }

    /// RDP list **processing** cycle debt (applied when a deferred `DPC_END` completes) and VI framebuffer
    /// read debt. SP DMA completion is timed via [`Self::rcp_advance_dma_in_flight`]; PI/SI/AI likewise.
    pub fn drain_deferred_cycles(&mut self) -> u64 {
        self.vi
            .drain_fetch_debt()
            .saturating_add(std::mem::take(&mut self.rdp_deferred_cycles))
    }

    /// Advance in-flight PI / SI / AI / SP / DPC work by `delta` RCP cycles (same quantum as the current CPU step).
    pub fn rcp_advance_dma_in_flight(&mut self, delta: u64) {
        self.pi.advance_time(delta, &mut self.rdram, &mut self.mi);
        self.si
            .advance_time(delta, &mut self.rdram, &mut self.pif, &mut self.mi);
        self.ai.advance_time(delta, &mut self.mi);
        self.advance_sp_dma(delta);
        self.advance_dpc_rdp(delta);
        crate::rsp::run_for_rcp_quantum(self, delta);
    }

    fn advance_sp_dma(&mut self, mut delta: u64) {
        while delta > 0 {
            let Some(p) = self.sp_dma_pending.as_mut() else {
                return;
            };
            let u = p.remaining_rcp_cycles.min(delta);
            p.remaining_rcp_cycles -= u;
            delta -= u;
            if p.remaining_rcp_cycles > 0 {
                return;
            }
            let job = self.sp_dma_pending.take().unwrap();
            if job.to_rsp {
                self.sp_dma_rdram_to_rsp_inner(job.len_reg, job.mem, job.dram);
            } else {
                self.sp_dma_rsp_to_rdram_inner(job.len_reg, job.mem, job.dram);
            }
            self.mi.raise(MI_INTR_SP);
        }
    }

    fn advance_dpc_rdp(&mut self, mut delta: u64) {
        while delta > 0 {
            let Some(p) = self.dpc_rdp_pending.as_mut() else {
                return;
            };
            let u = p.remaining_rcp_cycles.min(delta);
            p.remaining_rcp_cycles -= u;
            delta -= u;
            if p.remaining_rcp_cycles > 0 {
                return;
            }
            let job = self.dpc_rdp_pending.take().unwrap();
            let k = job.kick;
            let c = self.rdp.process_display_list(
                &mut self.rdram.data,
                &self.rsp_dmem[..],
                &self.rsp_imem[..],
                k.start,
                k.end,
                self.dpc_regs.status,
            );
            self.rdp_deferred_cycles = self.rdp_deferred_cycles.saturating_add(c);
            self.dpc_regs.mark_display_list_complete(k.end);
            self.mi.raise(MI_INTR_DP);
        }
    }

    /// NTSC VI line: accumulate cycles and raise `MI_INTR_VI` each frame (stub).
    pub fn advance_vi_frame_timing(&mut self, cycles: u64) {
        self.vi.advance(cycles, &mut self.mi);
    }

    #[inline]
    fn rsp_write_flat(&mut self, flat: usize, b: u8) {
        let i = flat & 0x1FFF;
        if i < 0x1000 {
            self.rsp_dmem[i] = b;
        } else {
            self.rsp_imem[i - 0x1000] = b;
        }
    }

    #[inline]
    fn rsp_read_flat(&self, flat: usize) -> u8 {
        let i = flat & 0x1FFF;
        if i < 0x1000 {
            self.rsp_dmem[i]
        } else {
            self.rsp_imem[i - 0x1000]
        }
    }

    fn sp_dma_rdram_to_rsp_inner(&mut self, len_reg: u32, mem: u32, dram: u32) {
        let (line_bytes, line_count, dram_skip) = sp_dma_decode(len_reg);
        let mut dram_p = (dram & 0x00FF_FFFF) as usize;
        let mut sp_flat =
            (((mem & 0x1000) != 0) as usize).saturating_mul(0x1000) + (mem & 0xFFF) as usize;
        let rd_len = self.rdram.data.len();

        for _line in 0..line_count {
            for _ in 0..line_bytes {
                let b = if dram_p < rd_len {
                    self.rdram.data[dram_p]
                } else {
                    0
                };
                self.rsp_write_flat(sp_flat, b);
                dram_p = dram_p.saturating_add(1);
                sp_flat = (sp_flat + 1) & 0x1FFF;
            }
            dram_p = dram_p.saturating_add(dram_skip);
        }

        let (new_mem, new_dram) = sp_dma_end_addresses(mem, dram, line_bytes, line_count, dram_skip);
        self.sp_regs.words[SP_WORD_MEM_ADDR] = new_mem;
        self.sp_regs.words[SP_WORD_DRAM_ADDR] = new_dram;
    }

    fn sp_dma_rsp_to_rdram_inner(&mut self, len_reg: u32, mem: u32, dram: u32) {
        let (line_bytes, line_count, dram_skip) = sp_dma_decode(len_reg);
        let mut dram_p = (dram & 0x00FF_FFFF) as usize;
        let mut sp_flat =
            (((mem & 0x1000) != 0) as usize).saturating_mul(0x1000) + (mem & 0xFFF) as usize;
        let rd_len = self.rdram.data.len();

        for _line in 0..line_count {
            for _ in 0..line_bytes {
                let b = self.rsp_read_flat(sp_flat);
                if dram_p < rd_len {
                    self.rdram.data[dram_p] = b;
                }
                dram_p = dram_p.saturating_add(1);
                sp_flat = (sp_flat + 1) & 0x1FFF;
            }
            dram_p = dram_p.saturating_add(dram_skip);
        }

        let (new_mem, new_dram) = sp_dma_end_addresses(mem, dram, line_bytes, line_count, dram_skip);
        self.sp_regs.words[SP_WORD_MEM_ADDR] = new_mem;
        self.sp_regs.words[SP_WORD_DRAM_ADDR] = new_dram;
    }

    fn sp_status_read(&self) -> u32 {
        let mut v = 0u32;
        if self.sp_halted {
            v |= 1;
        }
        if self.sp_broke {
            v |= 2;
        }
        v
    }

    /// `SP_STATUS` write: W1S/W1C pairs for halt and MI `SP` interrupt ([ares RSP](https://github.com/ares-emulator/ares)).
    fn sp_status_write(&mut self, value: u32) {
        if (value & 1) != 0 && (value & 2) == 0 {
            self.sp_halted = false;
        }
        if (value & 2) != 0 && (value & 1) == 0 {
            self.sp_halted = true;
        }
        if (value & 4) != 0 {
            self.sp_broke = false;
        }
        if (value & 8) != 0 && (value & 0x10) == 0 {
            self.mi.clear(MI_INTR_SP);
        }
        if (value & 0x10) != 0 && (value & 8) == 0 {
            self.mi.raise(MI_INTR_SP);
        }
    }

    fn sp_read_mmio(&mut self, paddr: u32) -> u32 {
        let off = paddr.wrapping_sub(SP_REGS_BASE);
        match off {
            SP_REG_STATUS => self.sp_status_read(),
            SP_REG_DMA_FULL => 0,
            SP_REG_DMA_BUSY => u32::from(self.sp_dma_pending.is_some()),
            SP_REG_SEMAPHORE => {
                let r = self.sp_semaphore & 1;
                self.sp_semaphore = 1;
                r
            }
            _ => self.sp_regs.read(paddr),
        }
    }

    fn sp_write_mmio(&mut self, paddr: u32, value: u32) {
        let off = paddr.wrapping_sub(SP_REGS_BASE);
        match off {
            SP_REG_STATUS => self.sp_status_write(value),
            SP_REG_DMA_FULL | SP_REG_DMA_BUSY => {}
            SP_REG_SEMAPHORE => {
                self.sp_semaphore = 0;
            }
            SP_REG_RD_LEN => {
                self.sp_regs.store_u32(paddr, value);
                if value != 0 {
                    self.mi.clear(MI_INTR_SP);
                    let mem = self.sp_regs.words[SP_WORD_MEM_ADDR];
                    let dram = self.sp_regs.words[SP_WORD_DRAM_ADDR];
                    self.sp_dma_pending = Some(SpDmaPending {
                        remaining_rcp_cycles: crate::timing::sp_rsp_dma_total_cycles(value),
                        len_reg: value,
                        mem,
                        dram,
                        to_rsp: true,
                    });
                }
            }
            SP_REG_WR_LEN => {
                self.sp_regs.store_u32(paddr, value);
                if value != 0 {
                    self.mi.clear(MI_INTR_SP);
                    let mem = self.sp_regs.words[SP_WORD_MEM_ADDR];
                    let dram = self.sp_regs.words[SP_WORD_DRAM_ADDR];
                    self.sp_dma_pending = Some(SpDmaPending {
                        remaining_rcp_cycles: crate::timing::sp_rsp_dma_total_cycles(value),
                        len_reg: value,
                        mem,
                        dram,
                        to_rsp: false,
                    });
                }
            }
            _ => {
                self.sp_regs.store_u32(paddr, value);
            }
        }
    }

    fn sp_pc_read(&self, paddr: u32) -> u32 {
        let o = paddr.wrapping_sub(SP_PC_REGS_BASE) & 0xF;
        match o {
            SP_PC_REG_PC => self.rsp_pc,
            SP_PC_REG_IBIST => self.rsp_ibist,
            _ => 0,
        }
    }

    fn sp_pc_write(&mut self, paddr: u32, value: u32) {
        let o = paddr.wrapping_sub(SP_PC_REGS_BASE) & 0xF;
        match o {
            SP_PC_REG_PC => self.rsp_pc = value & 0xFFC,
            SP_PC_REG_IBIST => self.rsp_ibist = value,
            _ => {}
        }
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
        if (SP_REGS_BASE..SP_REGS_BASE + SP_REGS_LEN as u32).contains(&paddr) {
            return Some(self.sp_read_mmio(paddr));
        }
        if (SP_PC_REGS_BASE..SP_PC_REGS_BASE + SP_PC_REGS_LEN as u32).contains(&paddr) {
            return Some(self.sp_pc_read(paddr));
        }
        if (DPC_REGS_BASE..DPC_REGS_BASE + DPC_REGS_LEN as u32).contains(&paddr) {
            return Some(self.dpc_regs.read(paddr));
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
        if (PI_REGS_BASE..PI_REGS_BASE + PI_REGS_LEN as u32).contains(&paddr) {
            return Some(self.pi.read_reg(paddr));
        }
        if (RI_REGS_BASE..RI_REGS_BASE + RI_REGS_LEN as u32).contains(&paddr) {
            return Some(self.ri.read(paddr));
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
            self.ai.write(paddr, value, &mut self.mi);
            return;
        }
        if (SP_REGS_BASE..SP_REGS_BASE + SP_REGS_LEN as u32).contains(&paddr) {
            self.sp_write_mmio(paddr, value);
            return;
        }
        if (SP_PC_REGS_BASE..SP_PC_REGS_BASE + SP_PC_REGS_LEN as u32).contains(&paddr) {
            self.sp_pc_write(paddr, value);
            return;
        }
        if (DPC_REGS_BASE..DPC_REGS_BASE + DPC_REGS_LEN as u32).contains(&paddr) {
            if let Some(k) = self.dpc_regs.write(paddr, value) {
                self.mi.clear(MI_INTR_DP);
                let est = Rdp::estimate_display_list_cycles(k.start, k.end);
                self.dpc_rdp_pending = Some(DpcRdpPending {
                    remaining_rcp_cycles: est,
                    kick: k,
                });
            }
            return;
        }
        if (PI_REGS_BASE..PI_REGS_BASE + PI_REGS_LEN as u32).contains(&paddr) {
            self.pi.write_reg(paddr, value, &mut self.rdram, &mut self.mi);
            return;
        }
        if (RI_REGS_BASE..RI_REGS_BASE + RI_REGS_LEN as u32).contains(&paddr) {
            self.ri.write(paddr, value);
            return;
        }
        if (SI_REGS_BASE..SI_REGS_BASE + SI_REGS_LEN as u32).contains(&paddr) {
            self.si.write(paddr, value, &mut self.rdram, &mut self.pif, &mut self.mi);
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
    use crate::ai::{AI_REG_LEN, AI_REGS_BASE};
    use crate::mi::{MI_INTR_AI, MI_INTR_SP, MI_REG_VERSION, MI_VERSION_DEFAULT};
    use crate::rcp::{
        SP_PC_REG_IBIST, SP_PC_REGS_BASE, SP_REG_DRAM_ADDR, SP_REG_MEM_ADDR, SP_REG_RD_LEN,
        SP_REG_SEMAPHORE, SP_REG_STATUS, SP_REGS_BASE,
    };
    use crate::rdp::Rdp;
    use crate::ri::{RI_MODE_DEFAULT, RI_REG_MODE, RI_REGS_BASE};
    use crate::vi::{VI_OFF_ORIGIN, VI_REG_ORIGIN, VI_REGS_BASE};

    #[test]
    fn mi_version_register() {
        let mut bus = SystemBus::with_rdram_size(1024 * 1024);
        assert_eq!(
            bus.read_u32(MI_REGS_BASE + MI_REG_VERSION),
            Some(MI_VERSION_DEFAULT)
        );
    }

    #[test]
    fn ri_mode_reset_visible_on_bus() {
        let mut bus = SystemBus::with_rdram_size(1024 * 1024);
        assert_eq!(bus.read_u32(RI_REGS_BASE + RI_REG_MODE), Some(RI_MODE_DEFAULT));
    }

    #[test]
    fn ai_len_write_sets_mi_ai() {
        let mut bus = SystemBus::with_rdram_size(1024 * 1024);
        bus.mi.mask = MI_INTR_AI;
        bus.write_u32(AI_REGS_BASE + AI_REG_LEN, 0x1000);
        assert!(!bus.mi.cpu_irq_pending());
        let need = crate::timing::ai_pcm_buffer_cycles(0x1000);
        bus.rcp_advance_dma_in_flight(need);
        assert!(bus.mi.cpu_irq_pending());
    }

    #[test]
    fn vi_origin_round_trip_on_bus() {
        let mut bus = SystemBus::with_rdram_size(1024 * 1024);
        bus.write_u32(VI_REGS_BASE + VI_OFF_ORIGIN, 0x0012_3456);
        assert_eq!(bus.read_u32(VI_REGS_BASE + VI_OFF_ORIGIN), Some(0x0012_3456));
        assert_eq!(bus.vi.regs[VI_REG_ORIGIN], 0x0012_3456);
    }

    #[test]
    fn dpc_end_runs_rdp_fill_rect_list() {
        use crate::rcp::{DPC_REG_END, DPC_REG_START, DPC_REGS_BASE};

        let mut bus = SystemBus::with_rdram_size(1024 * 1024);
        let origin = 0x0008_0000u32;
        bus.write_u32(VI_REGS_BASE + VI_OFF_ORIGIN, origin);

        let base = 0x1000usize;
        let w0_c = 0xFF080140u32;
        let w1_c = origin;
        let cmd1 = ((w0_c as u64) << 32) | (w1_c as u64);
        let w0_f = 0xF700_FFFFu32;
        let cmd2 = (w0_f as u64) << 32;
        let w0_r = (0xF6u32 << 24) | (8 << 12) | 8;
        let w1_r = (4 << 12) | 4;
        let cmd3 = ((w0_r as u64) << 32) | (w1_r as u64);
        for (i, cmd) in [cmd1, cmd2, cmd3].iter().enumerate() {
            bus.rdram.data[base + i * 8..base + i * 8 + 8].copy_from_slice(&cmd.to_be_bytes());
        }

        bus.mi.intr = 0;
        bus.write_u32(DPC_REGS_BASE + DPC_REG_START, base as u32);
        bus.write_u32(DPC_REGS_BASE + DPC_REG_END, (base + 24) as u32);
        assert_eq!(bus.mi.intr & crate::mi::MI_INTR_DP, 0);
        let est = Rdp::estimate_display_list_cycles(base as u32, (base + 24) as u32);
        bus.rcp_advance_dma_in_flight(est);
        assert_ne!(bus.mi.intr & crate::mi::MI_INTR_DP, 0);

        let off = origin as usize + (1usize * 320 + 1) * 2;
        assert_eq!(
            u16::from_be_bytes([bus.rdram.data[off], bus.rdram.data[off + 1]]),
            0xFFFF
        );
    }

    #[test]
    fn sp_rd_dma_copies_rdram_to_dmem() {
        let mut bus = SystemBus::with_rdram_size(1024 * 1024);
        bus.mi.intr = 0;
        bus.rdram.data[0x100..0x104].copy_from_slice(&[0xAA, 0xBB, 0xCC, 0xDD]);
        bus.write_u32(SP_REGS_BASE + SP_REG_MEM_ADDR, 0x0400_0000);
        bus.write_u32(SP_REGS_BASE + SP_REG_DRAM_ADDR, 0x0000_0100);
        bus.write_u32(SP_REGS_BASE + SP_REG_RD_LEN, 3);
        assert_eq!(&bus.rsp_dmem[0..4], &[0, 0, 0, 0]);
        bus.rcp_advance_dma_in_flight(crate::timing::sp_rsp_dma_total_cycles(3));
        assert_eq!(&bus.rsp_dmem[0..4], &[0xAA, 0xBB, 0xCC, 0xDD]);
        assert_ne!(bus.mi.intr & MI_INTR_SP, 0);
    }

    #[test]
    fn sp_rd_dma_updates_mem_and_dram_addresses() {
        let mut bus = SystemBus::with_rdram_size(1024 * 1024);
        bus.mi.intr = 0;
        bus.rdram.data[0x100..0x104].fill(0x5A);
        bus.write_u32(SP_REGS_BASE + SP_REG_MEM_ADDR, 0x0400_0000);
        bus.write_u32(SP_REGS_BASE + SP_REG_DRAM_ADDR, 0x0000_0100);
        bus.write_u32(SP_REGS_BASE + SP_REG_RD_LEN, 3);
        bus.rcp_advance_dma_in_flight(crate::timing::sp_rsp_dma_total_cycles(3));
        assert_eq!(bus.sp_regs.words[0], 0x0400_0004);
        assert_eq!(bus.sp_regs.words[1], 0x0000_0104);
    }

    #[test]
    fn sp_status_halt_and_clear_mi_via_status() {
        let mut bus = SystemBus::with_rdram_size(1024 * 1024);
        assert_eq!(bus.read_u32(SP_REGS_BASE + SP_REG_STATUS), Some(1));
        bus.write_u32(SP_REGS_BASE + SP_REG_STATUS, 1);
        assert_eq!(bus.read_u32(SP_REGS_BASE + SP_REG_STATUS), Some(0));
        bus.mi.raise(MI_INTR_SP);
        bus.write_u32(SP_REGS_BASE + SP_REG_STATUS, 8);
        assert_eq!(bus.mi.intr & MI_INTR_SP, 0);
    }

    #[test]
    fn sp_semaphore_acquire_and_release() {
        let mut bus = SystemBus::with_rdram_size(1024 * 1024);
        assert_eq!(bus.read_u32(SP_REGS_BASE + SP_REG_SEMAPHORE), Some(0));
        assert_eq!(bus.read_u32(SP_REGS_BASE + SP_REG_SEMAPHORE), Some(1));
        bus.write_u32(SP_REGS_BASE + SP_REG_SEMAPHORE, 0);
        assert_eq!(bus.read_u32(SP_REGS_BASE + SP_REG_SEMAPHORE), Some(0));
    }

    #[test]
    fn sp_pc_word_aligned_mask() {
        let mut bus = SystemBus::with_rdram_size(1024 * 1024);
        bus.write_u32(SP_PC_REGS_BASE, 0xFFFF_FF01);
        assert_eq!(bus.read_u32(SP_PC_REGS_BASE), Some(0xF00));
        bus.write_u32(SP_PC_REGS_BASE, 0xFFFF_FFFF);
        assert_eq!(bus.read_u32(SP_PC_REGS_BASE), Some(0xFFC));
        bus.write_u32(SP_PC_REGS_BASE, 0x0000_0200);
        assert_eq!(bus.read_u32(SP_PC_REGS_BASE), Some(0x200));
        assert_eq!(bus.read_u32(SP_PC_REGS_BASE + SP_PC_REG_IBIST), Some(0));
        bus.write_u32(SP_PC_REGS_BASE + SP_PC_REG_IBIST, 0x1234_5678);
        assert_eq!(bus.read_u32(SP_PC_REGS_BASE + SP_PC_REG_IBIST), Some(0x1234_5678));
    }

    #[test]
    fn sp_wr_dma_copies_dmem_to_rdram() {
        let mut bus = SystemBus::with_rdram_size(1024 * 1024);
        bus.mi.intr = 0;
        bus.rsp_dmem[0..4].copy_from_slice(&[0x11, 0x22, 0x33, 0x44]);
        bus.write_u32(SP_REGS_BASE + SP_REG_MEM_ADDR, 0x0400_0000);
        bus.write_u32(SP_REGS_BASE + SP_REG_DRAM_ADDR, 0x0000_0200);
        bus.write_u32(SP_REGS_BASE + SP_REG_WR_LEN, 3);
        bus.rcp_advance_dma_in_flight(crate::timing::sp_rsp_dma_total_cycles(3));
        assert_eq!(&bus.rdram.data[0x200..0x204], &[0x11, 0x22, 0x33, 0x44]);
        assert_ne!(bus.mi.intr & MI_INTR_SP, 0);
    }

    /// Two 2-byte lines from RDRAM with 8-byte inter-line skip; DMEM receives four contiguous bytes.
    #[test]
    fn sp_rd_dma_two_lines_with_rdram_skip() {
        let mut bus = SystemBus::with_rdram_size(1024 * 1024);
        bus.mi.intr = 0;
        bus.rdram.data[0x300] = 0x10;
        bus.rdram.data[0x301] = 0x11;
        bus.rdram.data[0x30A] = 0xaa;
        bus.rdram.data[0x30B] = 0xbb;
        bus.write_u32(SP_REGS_BASE + SP_REG_MEM_ADDR, 0x0400_0000);
        bus.write_u32(SP_REGS_BASE + SP_REG_DRAM_ADDR, 0x0000_0300);
        bus.write_u32(SP_REGS_BASE + SP_REG_RD_LEN, 0x0080_1001);
        bus.rcp_advance_dma_in_flight(crate::timing::sp_rsp_dma_total_cycles(0x0080_1001));
        assert_eq!(&bus.rsp_dmem[0..4], &[0x10, 0x11, 0xaa, 0xbb]);
        assert_ne!(bus.mi.intr & MI_INTR_SP, 0);
    }
}
