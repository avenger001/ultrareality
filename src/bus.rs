//! Physical memory, PI/cartridge, RSP DMEM/IMEM, and RCP MMIO (SP/DPC, MI/VI/AI/SI/PIF).

use crate::ai::{Ai, AI_REGS_BASE, AI_REGS_LEN};
use crate::mi::{Mi, MI_INTR_DP, MI_INTR_SP, MI_INTR_VI, MI_REGS_BASE, MI_REGS_LEN};
use crate::pif::{Pif, PIF_ROM_START, PIF_WINDOW_END};
use crate::pi::{Pi, CART_DOM1_ADDR2_BASE, PI_REGS_BASE, PI_REGS_LEN};
use crate::ri::{Ri, RI_REG_LATENCY, RI_REGS_BASE, RI_REGS_LEN};
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

/// RDRAM hardware config registers base (`0x03F0_0000`); per-chip stride `0x0008_0000`.
/// Games (via libultra `osInitialize`) probe these to detect memory size and configure refresh.
pub const RDRAM_REGS_BASE: u32 = 0x03F0_0000;
pub const RDRAM_REGS_END: u32 = 0x0400_0000;

pub const RSP_DMEM_START: u32 = 0x0400_0000;
pub const RSP_DMEM_END: u32 = 0x0400_1000;
pub const RSP_IMEM_START: u32 = 0x0400_1000;
pub const RSP_IMEM_END: u32 = 0x0400_2000;

pub trait Bus {
    fn read_u32(&mut self, paddr: u32) -> Option<u32>;
    fn write_u32(&mut self, paddr: u32, value: u32);
    fn read_u8(&mut self, paddr: u32) -> Option<u8>;
    fn write_u8(&mut self, paddr: u32, value: u8);

    /// Rambus RDRAM access cycles for physical `paddr` with `access_bytes`.
    ///
    /// This uses the Rambus packet model with row buffer tracking:
    /// - **Row hit**: Only column access cost (~2 cycles per 8 bytes)
    /// - **Row miss**: Precharge + row open + column (~17+ cycles)
    ///
    /// The RI_LATENCY register nibble adds controller timing overhead.
    fn rdram_access_cycles(&mut self, _paddr: u32, _access_bytes: u32) -> u64 {
        0
    }
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
    /// RSP Vector Unit (COP2) state.
    pub rsp_vu: crate::rsp_vu::VectorUnit,
    /// RSP delay slot: latched branch target, applied after the next instruction executes.
    pub rsp_delay_slot_target: Option<u32>,
    /// SP_STATUS signal flags (bits 0–7 correspond to sig0–sig7).
    pub sp_signal: u8,
    sp_dma_pending: Option<SpDmaPending>,
    dpc_rdp_pending: Option<DpcRdpPending>,
    /// Stub RDRAM hardware config registers (`0x03F0_0000` – `0x03FF_FFFF`).
    /// Two chips × 10 words each. Reads return configured defaults; writes are absorbed.
    rdram_config: [u32; 20],
}

/// Default RDRAM hardware config register values (two chips × 10 words).
///
/// Per-chip register layout (stride 0x0008_0000):
///   0x00 RDRAM_CONFIG, 0x04 RDRAM_DEVICE_ID, 0x08 RDRAM_DELAY,
///   0x0C RDRAM_MODE, 0x10 RDRAM_REF_INTERVAL, 0x14 RDRAM_REF_ROW,
///   0x18 RDRAM_RAS_INTERVAL, 0x1C RDRAM_MIN_INTERVAL, 0x20 RDRAM_ADDR_SELECT,
///   0x24 RDRAM_DEVICE_MANUF
///
/// Values from retail hardware observation / other emulators.
fn rdram_config_defaults() -> [u32; 20] {
    let chip = |dev_id: u32| -> [u32; 10] {
        [
            0xB419_0010, // CONFIG: row=9 col=9 bank=1 (4MB chip), valid bit
            dev_id,      // DEVICE_ID
            0x2B3B_1A0F, // DELAY
            0xC0C0_C0C0, // MODE (default active)
            0x0000_0400, // REF_INTERVAL
            0x0000_0000, // REF_ROW
            0x0000_0000, // RAS_INTERVAL
            0x0000_0000, // MIN_INTERVAL
            0x0000_0000, // ADDR_SELECT
            0x0000_0500, // DEVICE_MANUF (Rambus)
        ]
    };
    let c0 = chip(0x0000_0000);
    let c1 = chip(0x0000_0001);
    let mut out = [0u32; 20];
    out[..10].copy_from_slice(&c0);
    out[10..].copy_from_slice(&c1);
    out
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
            rsp_vu: crate::rsp_vu::VectorUnit::new(),
            rsp_delay_slot_target: None,
            sp_signal: 0,
            sp_dma_pending: None,
            dpc_rdp_pending: None,
            rdram_config: rdram_config_defaults(),
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
            rsp_vu: crate::rsp_vu::VectorUnit::new(),
            rsp_delay_slot_target: None,
            sp_signal: 0,
            sp_dma_pending: None,
            dpc_rdp_pending: None,
            rdram_config: rdram_config_defaults(),
        }
    }

    /// RCP cycles per RDRAM byte for VI/RDP traffic ([`crate::timing::rdram_byte_cost_from_ri_latency`]).
    #[inline]
    pub fn rdram_byte_cycle_cost(&self) -> u64 {
        crate::timing::rdram_byte_cost_from_ri_latency(self.ri.regs[(RI_REG_LATENCY / 4) as usize])
    }

    /// Charge VI readout of the current `VI_ORIGIN` / `VI_WIDTH` framebuffer (RGBA16 halfwords → RDRAM cycles).
    /// Only charges cycles if VI is actually enabled (VI_CTRL format bits nonzero).
    pub fn schedule_vi_frame_fetch(&mut self) {
        // VI_CTRL bits 0-1 select format: 0=blank, 2=RGBA5551, 3=RGBA8888
        let vi_ctrl = self.vi.regs[0];
        if vi_ctrl & 0b11 == 0 {
            return; // VI disabled or blank, no fetch needed
        }
        let px = self.vi.display_width() as u64 * self.vi.display_height() as u64;
        let cppb = self.rdram_byte_cycle_cost();
        self.vi
            .charge_framebuffer_fetch_rgba16_pixels(px, cppb);
    }

    /// RDP list **processing** cycle debt (applied when a deferred `DPC_END` completes) and VI framebuffer
    /// read debt. SP DMA completion is timed via [`Self::rcp_advance_dma_in_flight`]; PI/SI/AI likewise.
    pub fn drain_deferred_cycles(&mut self) -> u64 {
        self.vi
            .drain_fetch_debt()
            .saturating_add(std::mem::take(&mut self.rdp_deferred_cycles))
    }

    /// Advance in-flight PI / SI / AI / SP / DPC work by `delta` **RCP master cycles**.
    /// Each subsystem’s timers see the **same** `delta` in parallel (real hardware overlap), not sequential draining.
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
            // NOTE: SP DMA completion does NOT raise MI_INTR_SP on real
            // hardware — only BREAK (with intbreak enabled) or explicit
            // SET_INTR writes do. The DMA-busy bit clears via
            // `sp_dma_pending = None` above, which is enough.
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
            // Real hardware only raises MI_INTR_DP when the command stream
            // executes OP_SYNC_FULL. A gfx task typically issues multiple
            // DPC kicks (e.g. SetOtherModes, LoadBlock, triangles, SyncFull) —
            // if we fire DP IRQ on every kick, libultra gets duplicate
            // OS_EVENT_DP dispatches which corrupt scheduler state (observed
            // in SM64 as duplicate msg=0x65 at frame 342, leading to deadlock
            // after gfx task #326 completes).
            if self.rdp.last_list_had_sync_full {
                self.mi.raise(MI_INTR_DP);
            }
        }
    }

    /// VI field timer: same `delta` as [`Self::rcp_advance_dma_in_flight`] for lockstep with DMA/audio.
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
            v |= 1 << 1;
        }
        if self.sp_dma_pending.is_some() {
            v |= 1 << 2; // DMA busy
        }
        // bit 3 = DMA full (always 0 for now)
        // bit 4 = IO full (always 0)
        // bit 5 = single-step, bit 6 = intbreak (not modelled yet)
        // bits 7–14 = signal flags 0–7 (per n64brew SP_STATUS layout).
        // libultra's interrupt handler at __osHandleInterrupt ANDs the value
        // with 0x300 (bits 8+9 = SIG1+SIG2 = YIELDED/TASKDONE) to decide
        // whether an SP IRQ corresponds to a real task completion vs. a CPU
        // break event — placing the signals at the wrong bit positions makes
        // it route the audio task-done IRQ to OS_EVENT_CPU_BREAK and drop
        // the OS_EVENT_SP dispatch entirely.
        v |= (self.sp_signal as u32) << 7;
        v
    }

    /// `SP_STATUS` write: W1S/W1C pairs for halt, broke, interrupt, and signals 0–7.
    ///
    /// Write bits ([n64brew](https://n64brew.dev/wiki/Reality_Signal_Processor/Interface#0x0404_0010_-_SP_STATUS)):
    ///   0: clear halt    1: set halt
    ///   2: clear broke
    ///   3: clear intr    4: set intr
    ///   5: clear sstep   6: set sstep  (not modeled)
    ///   7: clear intr on break   8: set intr on break (not modeled)
    ///   9–22: clear/set signal 0–6 (paired)
    ///   Actually: bit  9 = clear sig0, bit 10 = set sig0
    ///             bit 11 = clear sig1, bit 12 = set sig1
    ///             ...
    ///             bit 23 = clear sig7, bit 24 = set sig7
    fn sp_status_write(&mut self, value: u32) {
        // Trace every SP_STATUS write so we can see if the CPU ever tries to
        // start a new RSP task after the first BREAK. Read __osRunningThread
        // (pointer at RDRAM 0x3359B0) so we know which OS thread made the write.
        {
            static SPSW_LOG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            let n = SPSW_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            // Only log writes that actually start the RSP (clr_halt+clr_broke = 0x125)
            // or set/clear interrupt bits — these are the load-bearing transitions.
            // Skip the high-volume sigs writes that flood the log.
            // Only log the actual "start RSP" pattern (clr_halt+clr_broke = bits 0+2 = 0x5)
            // — skip the noisy signal-set writes that happen before every start.
            let is_start = (value & 0x5) == 0x5;
            if is_start {
                let running = if 0x3359B0 + 4 <= self.rdram.data.len() {
                    u32::from_be_bytes([
                        self.rdram.data[0x3359B0],
                        self.rdram.data[0x3359B1],
                        self.rdram.data[0x3359B2],
                        self.rdram.data[0x3359B3],
                    ])
                } else { 0 };
                // Read the running thread's id field (offset 0x14).
                let tid = if running >= 0x80000000 {
                    let pa = (running & 0x1FFFFFFF) as usize + 0x14;
                    if pa + 4 <= self.rdram.data.len() {
                        u32::from_be_bytes([
                            self.rdram.data[pa], self.rdram.data[pa+1],
                            self.rdram.data[pa+2], self.rdram.data[pa+3],
                        ])
                    } else { 0 }
                } else { 0 };
                eprintln!(
                    "[SP_STATUS_W #{} f={}] value=0x{:08X} (clr_halt={} clr_broke={} clr_intr={} set_intr={} sigs=0x{:04X}) thread=0x{:08X} id={}",
                    n, self.vi.frame_counter, value,
                    (value & 1) != 0, (value & 4) != 0,
                    (value & 8) != 0, (value & 0x10) != 0,
                    (value >> 9) & 0xFFFF,
                    running, tid,
                );
            }
        }
        // Halt: clear (bit 0) / set (bit 1)
        if (value & 1) != 0 && (value & 2) == 0 {
            static RSP_START_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            let n = RSP_START_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            // Periodic frame-tagged trace so we can see if RSP starts continue
            // after the gameThread stalls. Without this it's impossible to tell
            // whether the scheduler is still trying to start tasks late in the run.
            let osk_type = u32::from_be_bytes([
                self.rsp_dmem[0xFC0], self.rsp_dmem[0xFC1],
                self.rsp_dmem[0xFC2], self.rsp_dmem[0xFC3],
            ]);
            if n < 8 || n % 100 == 0 {
                eprintln!(
                    "[RSP] START_TICK #{} f={} type={} PC={:03X}",
                    n, self.vi.frame_counter, osk_type, self.rsp_pc,
                );
            }
            // Arm the gfx PC trace on every gfx start so the dump on BREAK shows
            // the path actually taken (working tasks dump short traces; the
            // stuck task will show a clear loop pattern).
            if osk_type == 1 {
                if let Ok(mut t) = crate::rsp::GFX_TRACE.try_lock() {
                    t.arm(n);
                }
                // Dump IMEM around the polling-loop area at the moment of the
                // gfx task start so we can decode the trace against the *correct*
                // IMEM (rspboot may DMA different content in over time).
                static GFX_START_DUMPS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                let d = GFX_START_DUMPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if d < 6 {
                    let mut chunks = Vec::new();
                    for off in (0x000..0x100).step_by(4) {
                        let c = &self.rsp_imem[off..off+4];
                        chunks.push(format!("{:03X}:{:02X}{:02X}{:02X}{:02X}", off, c[0], c[1], c[2], c[3]));
                    }
                    eprintln!("[GFX_START dump#{} f={} task#{}] IMEM[000..100]: {}",
                        d, self.vi.frame_counter, n, chunks.join(" "));
                    eprintln!("[GFX_START dump#{} sp_dma_pending={} sp_halted={}",
                        d, self.sp_dma_pending.is_some(), self.sp_halted);
                }
            }
            if n < 8 {
                eprintln!("[RSP] START #{}: PC={:03X}", n, self.rsp_pc);
                // Dump OSTask currently in DMEM so we know what task is being run.
                let osk: Vec<String> = self.rsp_dmem[0xFC0..0x1000]
                    .chunks(4)
                    .enumerate()
                    .map(|(i, c)| {
                        if i == 0 {
                            format!("{:02X}{:02X}{:02X}{:02X}", c[0], c[1], c[2], c[3])
                        } else {
                            format!("{:02X}{:02X}{:02X}{:02X}", c[0], c[1], c[2], c[3])
                        }
                    })
                    .collect();
                eprintln!("[RSP] OSTask at START #{}: {}", n, osk.join(" "));
            }
            self.sp_halted = false;
        }
        if (value & 2) != 0 && (value & 1) == 0 {
            self.sp_halted = true;
        }
        // Broke: clear (bit 2)
        if (value & 4) != 0 {
            self.sp_broke = false;
        }
        // MI SP interrupt: clear (bit 3) / set (bit 4)
        if (value & 8) != 0 && (value & 0x10) == 0 {
            self.mi.clear(MI_INTR_SP);
        }
        if (value & 0x10) != 0 && (value & 8) == 0 {
            self.mi.raise(MI_INTR_SP);
        }
        // Signal flags 0–7: paired clear/set at bits 9–24
        for i in 0u32..8 {
            let clr_bit = 9 + i * 2;
            let set_bit = 10 + i * 2;
            if (value & (1 << clr_bit)) != 0 && (value & (1 << set_bit)) == 0 {
                self.sp_signal &= !(1 << i);
            }
            if (value & (1 << set_bit)) != 0 && (value & (1 << clr_bit)) == 0 {
                self.sp_signal |= 1 << i;
            }
        }
    }

    /// Public SP_STATUS read (used by RSP COP0 MFC0).
    pub fn sp_status_read_internal(&self) -> u32 {
        self.sp_status_read()
    }

    /// Public SP_STATUS write (used by RSP COP0 MTC0).
    pub fn sp_status_write_internal(&mut self, value: u32) {
        self.sp_status_write(value);
    }

    /// Public DMA full query.
    pub fn sp_dma_full(&self) -> bool {
        false // single-buffered for now
    }

    /// Public DMA busy query.
    pub fn sp_dma_busy(&self) -> bool {
        self.sp_dma_pending.is_some()
    }

    /// Initiate SP RD_LEN DMA from RSP COP0 MTC0.
    pub fn sp_write_rd_len(&mut self, value: u32) {
        self.sp_regs.store_u32(SP_REGS_BASE + SP_REG_RD_LEN, value);
        if value != 0 {
            self.mi.clear(MI_INTR_SP);
            let mem = self.sp_regs.words[SP_WORD_MEM_ADDR];
            let dram = self.sp_regs.words[SP_WORD_DRAM_ADDR];
            // DL ring buffer reload tracking: any DMA targeting DMEM[0x6A0]
            // (Fast3D DL ring buffer), regardless of who initiated it. Dumps
            // the RDRAM source so we can see whether the source contains zeros
            // (game-side bug or wrong segment) vs. our DMA returning garbage.
            if (mem & 0xFFF) == 0x6A0 {
                static DLLOAD_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                let n = DLLOAD_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if n < 50 {
                    let src = (dram & 0x00FF_FFF8) as usize;
                    let len = (value as usize & 0x0FFF) + 1;
                    let mut s = String::new();
                    let take = len.min(0x80);
                    for i in (0..take).step_by(4) {
                        if i > 0 && i % 16 == 0 { s.push('\n'); s.push_str("        "); }
                        else if i > 0 { s.push(' '); }
                        if src + i + 4 <= self.rdram.data.len() {
                            s.push_str(&format!("{:08X}",
                                u32::from_be_bytes([self.rdram.data[src+i], self.rdram.data[src+i+1],
                                                     self.rdram.data[src+i+2], self.rdram.data[src+i+3]])));
                        }
                    }
                    eprintln!("[DLLOAD #{}] DMEM<-0x{:08X} len=0x{:X}\n        {}",
                        n, dram, len, s);
                    // Also dump DMEM[0xF20..0xFC0] which spans the typical F3D
                    // segment table location. We don't know exactly where the
                    // ucode keeps it; the dump lets us see segment values.
                    let mut seg = String::new();
                    for off in (0xF00..0x1000).step_by(0x10) {
                        seg.push_str(&format!("\n        {:03X}: ", off));
                        for i in (0..0x10).step_by(4) {
                            if i > 0 { seg.push(' '); }
                            seg.push_str(&format!("{:08X}",
                                u32::from_be_bytes([self.rsp_dmem[off+i], self.rsp_dmem[off+i+1],
                                                     self.rsp_dmem[off+i+2], self.rsp_dmem[off+i+3]])));
                        }
                    }
                    eprintln!("[DLLOAD #{}] DMEM tail:{}", n, seg);
                    // Scan DMEM[0x000..0xFC0] for the known seg values
                    // 0x00214550 (seg 1) and 0x00084620 (seg 7 in task#326)
                    // to locate where the segment table actually lives.
                    let mut found = String::new();
                    for off in (0x000..0xFC0).step_by(4) {
                        let w = u32::from_be_bytes([self.rsp_dmem[off], self.rsp_dmem[off+1],
                                                     self.rsp_dmem[off+2], self.rsp_dmem[off+3]]);
                        if w == 0x00214550 || w == 0x00084620 || w == 0x00064F80 || w == 0x00207D00
                           || w == 0x00220C88 || w == 0x002146F8 || w == 0x00220B68 {
                            found.push_str(&format!(" [{:03X}]=0x{:08X}", off, w));
                        }
                    }
                    if !found.is_empty() {
                        eprintln!("[DLLOAD #{}] seg-scan:{}", n, found);
                    }
                    // For DLLOAD #10 (the one just before the bad segment-7
                    // G_DL fires), dump the ENTIRE first 0x700 bytes of DMEM
                    // (below the ring buffer). This lets us find the segment
                    // table no matter where the ucode put it.
                    if n == 10 {
                        let mut full = String::new();
                        for off in (0x000..0x6A0).step_by(0x10) {
                            full.push_str(&format!("\n        {:03X}: ", off));
                            for i in (0..0x10).step_by(4) {
                                if i > 0 { full.push(' '); }
                                full.push_str(&format!("{:08X}",
                                    u32::from_be_bytes([self.rsp_dmem[off+i], self.rsp_dmem[off+i+1],
                                                         self.rsp_dmem[off+i+2], self.rsp_dmem[off+i+3]])));
                            }
                        }
                        eprintln!("[DLLOAD #{}] DMEM[000..6A0]:{}", n, full);
                    }
                }
            }
            self.sp_dma_pending = Some(SpDmaPending {
                remaining_rcp_cycles: crate::timing::sp_rsp_dma_total_cycles(value),
                len_reg: value,
                mem,
                dram,
                to_rsp: true,
            });
        }
    }

    /// Initiate SP WR_LEN DMA from RSP COP0 MTC0.
    pub fn sp_write_wr_len(&mut self, value: u32) {
        self.sp_regs.store_u32(SP_REGS_BASE + SP_REG_WR_LEN, value);
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

    /// Kick the RDP from RSP COP0 DPC_END write.
    pub fn kick_rdp(&mut self, k: crate::rcp::DpcEndKick) {
        static KICK_LOG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = KICK_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if n < 30 {
            // Source label: distinguish CPU MMIO kicks from RSP MTC0 kicks
            // by inspecting whether the SP is currently halted (CPU MMIO) or
            // running (RSP MTC0). When called from `sp_write_mmio`, the SP
            // may be running OR halted; inspecting the kick's start/end vs
            // dpc_status bits isn't decisive. Best we can do here: print SP
            // state at kick time as a hint.
            let sp_state = if self.sp_halted { "HALT" } else { "RUN " };
            eprintln!("[RDP kick #{}] sp={} start=0x{:08X} end=0x{:08X}", n, sp_state, k.start, k.end);
        }
        self.mi.clear(MI_INTR_DP);
        let est = crate::rdp::Rdp::estimate_display_list_cycles(k.start, k.end);
        self.dpc_rdp_pending = Some(DpcRdpPending {
            remaining_rcp_cycles: est,
            kick: k,
        });
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
            SP_REG_RD_LEN => self.sp_write_rd_len(value),
            SP_REG_WR_LEN => self.sp_write_wr_len(value),
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
    fn rdram_access_cycles(&mut self, paddr: u32, access_bytes: u32) -> u64 {
        let len = self.rdram_len_u32();
        if paddr < len {
            self.ri.cpu_rdram_access_cycles(paddr, access_bytes)
        } else {
            0
        }
    }

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
            let v = self.mi.read(paddr);
            // Track when MI_INTR is read with VI bit set
            if paddr == crate::mi::MI_REGS_BASE + crate::mi::MI_REG_INTR && (v & crate::mi::MI_INTR_VI) != 0 {
                static MI_INTR_VI_READ_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                MI_INTR_VI_READ_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            return Some(v);
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
        // RDRAM hardware config registers (0x03F0_0000 – 0x03FF_FFFF)
        if (RDRAM_REGS_BASE..RDRAM_REGS_END).contains(&paddr) {
            let off = paddr - RDRAM_REGS_BASE;
            // Chip 0: 0x03F0_0000, Chip 1: 0x03F8_0000; 10 words each
            let chip = (off >> 19) as usize; // 0x80000 stride
            let word = ((off & 0x3F) >> 2) as usize;
            if chip < 2 && word < 10 {
                return Some(self.rdram_config[chip * 10 + word]);
            }
            return Some(0);
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
            let off = paddr - VI_REGS_BASE;
            // Writing to VI_V_CURRENT (offset 0x10) clears the VI interrupt
            if off == 0x10 {
                self.mi.clear(MI_INTR_VI);
                crate::vi::VI_INT_ACK_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
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
                self.kick_rdp(k);
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
            return;
        }
        // RDRAM hardware config registers — absorb writes
        if (RDRAM_REGS_BASE..RDRAM_REGS_END).contains(&paddr) {
            let off = paddr - RDRAM_REGS_BASE;
            let chip = (off >> 19) as usize;
            let word = ((off & 0x3F) >> 2) as usize;
            if chip < 2 && word < 10 {
                self.rdram_config[chip * 10 + word] = value;
            }
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
        // Real hardware only raises MI_INTR_DP on OP_SYNC_FULL (0xE9).
        let cmd4 = 0xE900_0000_0000_0000u64;
        for (i, cmd) in [cmd1, cmd2, cmd3, cmd4].iter().enumerate() {
            bus.rdram.data[base + i * 8..base + i * 8 + 8].copy_from_slice(&cmd.to_be_bytes());
        }

        bus.mi.intr = 0;
        bus.write_u32(DPC_REGS_BASE + DPC_REG_START, base as u32);
        bus.write_u32(DPC_REGS_BASE + DPC_REG_END, (base + 32) as u32);
        assert_eq!(bus.mi.intr & crate::mi::MI_INTR_DP, 0);
        let est = Rdp::estimate_display_list_cycles(base as u32, (base + 32) as u32);
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
        // SP DMA completion does not raise MI_INTR_SP on real hardware.
        assert_eq!(bus.mi.intr & MI_INTR_SP, 0);
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
        assert_eq!(bus.mi.intr & MI_INTR_SP, 0);
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
        assert_eq!(bus.mi.intr & MI_INTR_SP, 0);
    }
}
