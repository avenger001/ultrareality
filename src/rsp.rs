//! Reality Signal Processor — scalar ISA + COP0 + COP2 stub + delay slots.
//!
//! Execution state lives on [`crate::bus::SystemBus`]: `rsp_pc`, `rsp_scalar_regs`, `sp_halted`.
//! Vector (COP2) ops advance PC but do not emulate the VU yet.

use crate::bus::SystemBus;
use crate::mi::MI_INTR_SP;
use crate::rsp_vu;

/// RCP cycles charged per scalar instruction (stub until cycle-accurate RSP exists).
pub const RSP_CYCLES_PER_INSTR: u64 = 2;

#[inline]
fn gpr_load(regs: &[u32; 32], r: usize) -> u32 {
    if r == 0 {
        0
    } else {
        regs[r]
    }
}

#[inline]
fn gpr_store(regs: &mut [u32; 32], r: usize, v: u32) {
    if r != 0 {
        regs[r] = v;
    }
}

/// Placeholder for a future instance-based API (state is on [`SystemBus`]).
#[derive(Clone, Debug, Default)]
pub struct RspState {
    pub halted: bool,
    pub pc: u32,
}

#[derive(Debug, Default)]
pub struct Rsp;

impl Rsp {
    pub fn new() -> Self {
        Self
    }

    /// Deprecated: use [`step_instruction`]. Kept for callers that still use the old API.
    pub fn step(&mut self) -> bool {
        let _ = self;
        false
    }
}

// ---------------------------------------------------------------------------
// RSP memory helpers (DMEM/IMEM flat address space 0x000..0x1FFF)
// ---------------------------------------------------------------------------

/// Fetch one big-endian word from RSP IMEM (`pc` is byte offset `0..0xFFF`).
#[inline]
fn imem_load_word(imem: &[u8; 4096], pc: usize) -> u32 {
    let i = pc & 0xFFC;
    u32::from_be_bytes(imem[i..i + 4].try_into().unwrap())
}

// NOTE: RSP has NO alignment checks for scalar loads/stores. LW/LH/LHU/SW/SH
// all operate on consecutive bytes starting at the given address, wrapping
// within the 4KB DMEM on overflow. F3D ucode RELIES on unaligned LH/LHU
// (e.g. `LH $a1, 0x30E($at=6)` reads DMEM[0x314], and `LHU $v0, -7($k1)`
// reads an odd-aligned halfword from the DL ring buffer).
#[inline]
fn dmem_load_word(dmem: &[u8; 4096], addr: usize) -> u32 {
    let a0 = addr & 0xFFF;
    let a1 = (addr + 1) & 0xFFF;
    let a2 = (addr + 2) & 0xFFF;
    let a3 = (addr + 3) & 0xFFF;
    u32::from_be_bytes([dmem[a0], dmem[a1], dmem[a2], dmem[a3]])
}

#[inline]
fn dmem_store_word(dmem: &mut [u8; 4096], addr: usize, v: u32) {
    let b = v.to_be_bytes();
    dmem[addr & 0xFFF] = b[0];
    dmem[(addr + 1) & 0xFFF] = b[1];
    dmem[(addr + 2) & 0xFFF] = b[2];
    dmem[(addr + 3) & 0xFFF] = b[3];
}

#[inline]
fn dmem_load_u8(dmem: &[u8; 4096], addr: usize) -> u8 {
    dmem[addr & 0xFFF]
}

#[inline]
fn dmem_store_u8(dmem: &mut [u8; 4096], addr: usize, v: u8) {
    dmem[addr & 0xFFF] = v;
}

#[inline]
fn dmem_load_u16(dmem: &[u8; 4096], addr: usize) -> u16 {
    let a0 = addr & 0xFFF;
    let a1 = (addr + 1) & 0xFFF;
    u16::from_be_bytes([dmem[a0], dmem[a1]])
}

#[inline]
fn dmem_store_u16(dmem: &mut [u8; 4096], addr: usize, v: u16) {
    let b = v.to_be_bytes();
    dmem[addr & 0xFFF] = b[0];
    dmem[(addr + 1) & 0xFFF] = b[1];
}

// ---------------------------------------------------------------------------
// RSP flat memory load/store (DMEM 0x000–0xFFF, IMEM 0x1000–0x1FFF)
// ---------------------------------------------------------------------------

fn rsp_load_flat(bus: &SystemBus, addr: u32) -> u32 {
    let a = (addr & 0x1FFF) as usize;
    if a < 0x1000 {
        dmem_load_word(&bus.rsp_dmem, a)
    } else {
        imem_load_word(&bus.rsp_imem, a & 0xFFC)
    }
}

fn rsp_store_flat(bus: &mut SystemBus, addr: u32, v: u32) {
    let a = (addr & 0x1FFF) as usize;
    if a < 0x1000 {
        dmem_store_word(&mut bus.rsp_dmem, a, v);
    } else if a + 4 <= 0x2000 {
        let i = a - 0x1000;
        bus.rsp_imem[i..i + 4].copy_from_slice(&v.to_be_bytes());
    }
}

fn rsp_load_flat_u8(bus: &SystemBus, addr: u32) -> u8 {
    let a = (addr & 0x1FFF) as usize;
    if a < 0x1000 {
        dmem_load_u8(&bus.rsp_dmem, a)
    } else {
        bus.rsp_imem[(a - 0x1000) & 0xFFF]
    }
}

fn rsp_store_flat_u8(bus: &mut SystemBus, addr: u32, v: u8) {
    let a = (addr & 0x1FFF) as usize;
    if a < 0x1000 {
        dmem_store_u8(&mut bus.rsp_dmem, a, v);
    } else {
        bus.rsp_imem[(a - 0x1000) & 0xFFF] = v;
    }
}

fn rsp_load_flat_u16(bus: &SystemBus, addr: u32) -> u16 {
    let a = (addr & 0x1FFF) as usize;
    if a < 0x1000 {
        dmem_load_u16(&bus.rsp_dmem, a)
    } else {
        let i = (a - 0x1000) & 0xFFE;
        u16::from_be_bytes(bus.rsp_imem[i..i + 2].try_into().unwrap())
    }
}

fn rsp_store_flat_u16(bus: &mut SystemBus, addr: u32, v: u16) {
    let a = (addr & 0x1FFF) as usize;
    if a < 0x1000 {
        dmem_store_u16(&mut bus.rsp_dmem, a, v);
    } else {
        let i = (a - 0x1000) & 0xFFE;
        bus.rsp_imem[i..i + 2].copy_from_slice(&v.to_be_bytes());
    }
}

// ---------------------------------------------------------------------------
// RSP COP0: MFC0/MTC0 — read/write SP and DPC registers from microcode
// ---------------------------------------------------------------------------

/// COP0 register mapping for RSP (rd index 0–15):
///   0: SP_MEM_ADDR      4: SP_STATUS     8: DPC_START     12: DPC_STATUS
///   1: SP_DRAM_ADDR     5: SP_DMA_FULL   9: DPC_END       13: DPC_CLOCK
///   2: SP_RD_LEN        6: SP_DMA_BUSY  10: DPC_CURRENT   14: DPC_BUF_BUSY
///   3: SP_WR_LEN        7: SP_SEMAPHORE 11: (reserved)    15: DPC_TMEM_BUSY
fn rsp_cop0_read(bus: &mut SystemBus, rd: usize) -> u32 {
    use crate::rcp::*;
    match rd {
        0 => bus.sp_regs.read(SP_REGS_BASE + SP_REG_MEM_ADDR),
        1 => bus.sp_regs.read(SP_REGS_BASE + SP_REG_DRAM_ADDR),
        2 => bus.sp_regs.read(SP_REGS_BASE + SP_REG_RD_LEN),
        3 => bus.sp_regs.read(SP_REGS_BASE + SP_REG_WR_LEN),
        4 => bus.sp_status_read_internal(),
        5 => u32::from(bus.sp_dma_full()),
        6 => u32::from(bus.sp_dma_busy()),
        7 => {
            let r = bus.sp_semaphore & 1;
            bus.sp_semaphore = 1;
            r
        }
        8 => bus.dpc_regs.read(DPC_REGS_BASE + DPC_REG_START),
        9 => bus.dpc_regs.read(DPC_REGS_BASE + DPC_REG_END),
        10 => bus.dpc_regs.read(DPC_REGS_BASE + DPC_REG_CURRENT),
        11 => bus.dpc_regs.read(DPC_REGS_BASE + DPC_REG_STATUS),
        12 => 0, // DPC_CLOCK
        13 => 0, // DPC_BUF_BUSY
        14 => 0, // DPC_PIPE_BUSY
        15 => 0, // DPC_TMEM_BUSY
        _ => 0,
    }
}

fn rsp_cop0_write(bus: &mut SystemBus, rd: usize, value: u32) {
    use crate::rcp::*;
    match rd {
        0 => bus.sp_regs.store_u32(SP_REGS_BASE + SP_REG_MEM_ADDR, value),
        1 => bus.sp_regs.store_u32(SP_REGS_BASE + SP_REG_DRAM_ADDR, value),
        2 => bus.sp_write_rd_len(value),
        3 => bus.sp_write_wr_len(value),
        4 => bus.sp_status_write_internal(value),
        5 => {} // SP_DMA_FULL — read-only
        6 => {} // SP_DMA_BUSY — read-only
        7 => bus.sp_semaphore = 0,
        8 => { bus.dpc_regs.write(DPC_REGS_BASE + DPC_REG_START, value); }
        9 => {
            if let Some(k) = bus.dpc_regs.write(DPC_REGS_BASE + DPC_REG_END, value) {
                bus.kick_rdp(k);
            }
        }
        10 => {} // DPC_CURRENT — read-only
        11 => { bus.dpc_regs.write(DPC_REGS_BASE + DPC_REG_STATUS, value); }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// RSP interpreter core
// ---------------------------------------------------------------------------

/// Execute one RSP scalar instruction, handling delay slots.
/// Returns **0** if the RSP is halted.
/// PC histogram bucket (one slot per IMEM word) for diagnosing stuck microcode.
pub static RSP_PC_HIST: [std::sync::atomic::AtomicU32; 1024] = {
    const Z: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    [Z; 1024]
};
/// Count of unimplemented opcodes seen by the RSP interpreter.
pub static RSP_UNIMPL_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
/// Last unimplemented instruction word (op<<26 | funct or whatever the decode hits).
pub static RSP_LAST_UNIMPL: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Per-gfx-task PC trace ring (captures last N PCs of a gfx task; circular).
/// Used to inspect what the stuck task is doing at quit time.
pub const GFX_TRACE_LEN: usize = 65536;
pub static GFX_TRACE: std::sync::Mutex<GfxTrace> = std::sync::Mutex::new(GfxTrace::new());

pub struct GfxTrace {
    pub pcs: [u16; GFX_TRACE_LEN],
    pub past_rspboot: bool,
    /// Total number of records seen since arm (may exceed GFX_TRACE_LEN).
    pub total: usize,
    /// Index where the next record will be written (circular).
    pub head: usize,
    pub task_id: u32,
    pub armed: bool,
}

impl GfxTrace {
    pub const fn new() -> Self {
        Self { pcs: [0; GFX_TRACE_LEN], past_rspboot: false, total: 0, head: 0, task_id: 0, armed: false }
    }
    pub fn arm(&mut self, task_id: u32) {
        self.task_id = task_id;
        self.total = 0;
        self.head = 0;
        self.armed = true;
        self.past_rspboot = false;
    }
    pub fn record(&mut self, pc: u16) {
        if self.armed {
            self.total += 1;
            // Skip rspboot (0x000-0x0FF) until the task has left it at least once.
            if !self.past_rspboot {
                if pc < 0x100 { return; }
                self.past_rspboot = true;
            }
            // Dedupe short loops: if this pc appeared in the last 4 recorded
            // slots, skip it. That collapses 3-PC loops like 164/168/16C
            // into a single entry without losing forward progress.
            if self.head >= 1 {
                let start = self.head.saturating_sub(4);
                if self.pcs[start..self.head].contains(&pc) {
                    return;
                }
            }
            if self.head < GFX_TRACE_LEN {
                self.pcs[self.head] = pc;
                self.head += 1;
            }
        }
    }
    /// Dump last N entries (chronological order).
    /// Non-wrapping: records live in pcs[..head], in order. Dump the last n of them.
    pub fn dump_tail(&self, n: usize, label: &str) {
        let n = n.min(self.head);
        let start = self.head - n;
        let s: Vec<String> = self.pcs[start..self.head]
            .iter()
            .map(|p| format!("{:03X}", p))
            .collect();
        eprintln!("[GFX_TRACE {} task#{} total={} recorded={} tail({})]: {}",
            label, self.task_id, self.total, self.head, n, s.join(" "));
    }
    /// Dump the FIRST N recorded entries (chronological).
    pub fn dump_head(&self, n: usize, label: &str) {
        let n = n.min(self.head);
        let s: Vec<String> = self.pcs[..n]
            .iter()
            .map(|p| format!("{:03X}", p))
            .collect();
        eprintln!("[GFX_TRACE {} task#{} total={} recorded={} head({})]: {}",
            label, self.task_id, self.total, self.head, n, s.join(" "));
    }
    pub fn dump(&self, label: &str) {
        self.dump_tail(GFX_TRACE_LEN, label);
    }
}

static HANDLER_TRACE_ARM: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
static HANDLER_TRACE_REMAINING: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
static HANDLER_TRACE_TAG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

pub fn step_instruction(bus: &mut SystemBus) -> u64 {
    if bus.sp_halted {
        return 0;
    }

    let pc = (bus.rsp_pc as usize) & 0xFFC;
    if pc + 4 > bus.rsp_imem.len() {
        bus.sp_broke = true;
        bus.sp_halted = true;
        bus.mi.raise(MI_INTR_SP);
        return RSP_CYCLES_PER_INSTR;
    }

    // PC histogram (one slot per word, 1024 entries cover the full IMEM).
    RSP_PC_HIST[(pc >> 2) & 0x3FF].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    // Per-task PC trace (cheap when not armed)
    if let Ok(mut t) = GFX_TRACE.try_lock() {
        t.record(pc as u16);
    }
    // Handler-PC trace: when armed via HANDLER_TRACE_ARM, record every
    // executed instruction (PC + raw opcode) for a fixed budget.
    {
        let a = HANDLER_TRACE_ARM.swap(0, std::sync::atomic::Ordering::Relaxed);
        if a > 0 {
            HANDLER_TRACE_REMAINING.store(a, std::sync::atomic::Ordering::Relaxed);
        }
        let r = HANDLER_TRACE_REMAINING.load(std::sync::atomic::Ordering::Relaxed);
        if r > 0 {
            HANDLER_TRACE_REMAINING.store(r - 1, std::sync::atomic::Ordering::Relaxed);
            let iw_trace = imem_load_word(&bus.rsp_imem, pc);
            let tag = HANDLER_TRACE_TAG.load(std::sync::atomic::Ordering::Relaxed);
            eprintln!("[HTRACE t={} rem={}] pc=0x{:03X} iw=0x{:08X} op=0x{:02X} t9=0x{:08X} t8=0x{:08X} at=0x{:08X} v0=0x{:08X} v1=0x{:08X} t3=0x{:08X} t4=0x{:08X} t0=0x{:08X} t1=0x{:08X} k1=0x{:04X}",
                tag, r, pc, iw_trace, iw_trace >> 26,
                bus.rsp_scalar_regs[25] as u32, bus.rsp_scalar_regs[24] as u32,
                bus.rsp_scalar_regs[1] as u32, bus.rsp_scalar_regs[2] as u32,
                bus.rsp_scalar_regs[3] as u32,
                bus.rsp_scalar_regs[11] as u32, bus.rsp_scalar_regs[12] as u32,
                bus.rsp_scalar_regs[8] as u32, bus.rsp_scalar_regs[9] as u32,
                bus.rsp_scalar_regs[27] as u32 & 0xFFF);
            // At the G_TRI1 cull-test point (pc=0x9B0), snapshot the three
            // vertex outcode bytes + first 8 bytes of each vertex record.
            if pc == 0x9B0 {
                let at = bus.rsp_scalar_regs[1] as u32 as usize & 0xFFF;
                let v0 = bus.rsp_scalar_regs[2] as u32 as usize & 0xFFF;
                let v1 = bus.rsp_scalar_regs[3] as u32 as usize & 0xFFF;
                let read_u16 = |off: usize| -> u16 {
                    if off + 2 <= bus.rsp_dmem.len() {
                        u16::from_be_bytes([bus.rsp_dmem[off], bus.rsp_dmem[off+1]])
                    } else { 0 }
                };
                let read_bytes = |off: usize, n: usize| -> String {
                    (0..n).map(|i| {
                        if off + i < bus.rsp_dmem.len() {
                            format!("{:02X}", bus.rsp_dmem[off + i])
                        } else { "??".into() }
                    }).collect::<Vec<_>>().join("")
                };
                eprintln!("[CULL v0@{:03X} flag24={:04X} bytes={}] [v1@{:03X} flag24={:04X} bytes={}] [v2@{:03X} flag24={:04X} bytes={}]",
                    at, read_u16(at + 0x24), read_bytes(at, 40),
                    v0, read_u16(v0 + 0x24), read_bytes(v0, 40),
                    v1, read_u16(v1 + 0x24), read_bytes(v1, 40));
            }
        }
    }
    // DL command log: every time the ucode reads a DL command at 0x060
    // (LW $t9, 0($k1)), snapshot $k1, $gp, and the command words. This is
    // how we see whether the DL terminates or loops.
    if pc == 0x060 {
        static DL_CMD_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        static FIRST_NOOP_DUMPED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        let n = DL_CMD_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let k1 = bus.rsp_scalar_regs[27] as u32;
        let gp = bus.rsp_scalar_regs[28] as u32;
        let k1d = (k1 as usize) & 0xFFF;
        let w0 = if k1d + 8 <= bus.rsp_dmem.len() {
            u32::from_be_bytes([bus.rsp_dmem[k1d], bus.rsp_dmem[k1d+1],
                                 bus.rsp_dmem[k1d+2], bus.rsp_dmem[k1d+3]])
        } else { 0 };
        let w1 = if k1d + 8 <= bus.rsp_dmem.len() {
            u32::from_be_bytes([bus.rsp_dmem[k1d+4], bus.rsp_dmem[k1d+5],
                                 bus.rsp_dmem[k1d+6], bus.rsp_dmem[k1d+7]])
        } else { 0 };
        if n < 200 || n % 10000 == 0 {
            eprintln!("[DL #{}] k1=0x{:04X} gp=0x{:08X} cmd=0x{:08X} 0x{:08X} (op=0x{:02X})",
                n, k1, gp, w0, w1, (w0 >> 24) & 0xFF);
        }
        // Arm handler trace for the segment-7 G_MOVEWORD and the
        // following G_DL that would dereference segment 7. Also trace
        // G_ENDDL so we can see how termination is supposed to work.
        // Only arm once per cmd so we see clean handler paths.
        static HT_ARMED_SEG7: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        static HT_ARMED_G_DL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        static HT_ARMED_ENDDL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if w0 == 0xBC001C06 && w1 != 0
           && !HT_ARMED_SEG7.swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            eprintln!("[HT ARM seg7 nz] DL#{} cmd={:08X} val={:08X}", n, w0, w1);
            HANDLER_TRACE_TAG.store(0xBC001C06, std::sync::atomic::Ordering::Relaxed);
            HANDLER_TRACE_ARM.store(80, std::sync::atomic::Ordering::Relaxed);
        }
        // Also capture the very first G_MOVEWORD segment 0 setup (usually the
        // segment-table init at task start) so we can compare with seg-7.
        static HT_ARMED_SEG0: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if w0 == 0xBC000406 && w1 != 0
           && !HT_ARMED_SEG0.swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            eprintln!("[HT ARM seg1 nz] DL#{} cmd={:08X} val={:08X}", n, w0, w1);
            HANDLER_TRACE_TAG.store(0xBC000406, std::sync::atomic::Ordering::Relaxed);
            HANDLER_TRACE_ARM.store(80, std::sync::atomic::Ordering::Relaxed);
        }
        // G_DL with segment 7 (opcode 0x06, upper byte of w1 = 0x07)
        if (w0 >> 24) == 0x06 && (w1 >> 24) == 0x07
           && !HT_ARMED_G_DL.swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            eprintln!("[HT ARM G_DL seg7] DL#{} cmd={:08X} addr={:08X}", n, w0, w1);
            HANDLER_TRACE_TAG.store(0x06000007, std::sync::atomic::Ordering::Relaxed);
            HANDLER_TRACE_ARM.store(120, std::sync::atomic::Ordering::Relaxed);
        }
        // G_ENDDL (op 0xB8)
        if (w0 >> 24) == 0xB8
           && !HT_ARMED_ENDDL.swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            eprintln!("[HT ARM G_ENDDL] DL#{} cmd={:08X}", n, w0);
            HANDLER_TRACE_TAG.store(0xB8000000, std::sync::atomic::Ordering::Relaxed);
            HANDLER_TRACE_ARM.store(80, std::sync::atomic::Ordering::Relaxed);
        }
        // G_VTX (op 0x04) — trace the vertex load handler. This is the
        // key path we need to verify: does it segment-resolve, kick an
        // SP DMA, and land the vertices in the vertex cache?
        static HT_ARMED_GVTX: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if (w0 >> 24) == 0x04
           && !HT_ARMED_GVTX.swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            eprintln!("[HT ARM G_VTX] DL#{} cmd={:08X} addr={:08X}", n, w0, w1);
            HANDLER_TRACE_TAG.store(0x04000000, std::sync::atomic::Ordering::Relaxed);
            HANDLER_TRACE_ARM.store(300, std::sync::atomic::Ordering::Relaxed);
        }
        // G_TRI1 (op 0xBF) — trace the triangle setup/emission handler.
        // This is where the "no triangles reach RDP" bug lives. We want
        // to see: transform, clip/cull test, cross-product setup,
        // edge coefficient computation, and SB/SH writes to the RDP
        // output buffer in DMEM.
        static HT_ARMED_GTRI1: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if (w0 >> 24) == 0xBF
           && !HT_ARMED_GTRI1.swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            eprintln!("[HT ARM G_TRI1] DL#{} cmd={:08X} indices={:08X}", n, w0, w1);
            HANDLER_TRACE_TAG.store(0xBF000000, std::sync::atomic::Ordering::Relaxed);
            HANDLER_TRACE_ARM.store(800, std::sync::atomic::Ordering::Relaxed);
        }
        // First time we see a NOOP in a stuck task (after task #310's break),
        // dump the entire ring buffer + segment table area so we can see what's there.
        if w0 == 0 && w1 == 0 && n > 200
           && !FIRST_NOOP_DUMPED.swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            eprintln!("[DL FIRST_NOOP] n={} k1=0x{:04X} gp=0x{:08X}", n, k1, gp);
            for base in (0x6A0..0x800).step_by(0x10) {
                let mut s = String::new();
                for i in (0..0x10).step_by(4) {
                    if i > 0 { s.push(' '); }
                    s.push_str(&format!("{:08X}",
                        u32::from_be_bytes([bus.rsp_dmem[base+i], bus.rsp_dmem[base+i+1],
                                             bus.rsp_dmem[base+i+2], bus.rsp_dmem[base+i+3]])));
                }
                eprintln!("  DMEM[{:03X}]: {}", base, s);
            }
        }
    }

    let iw = imem_load_word(&bus.rsp_imem, pc);
    let op = iw >> 26;
    let rs = ((iw >> 21) & 31) as usize;
    let rt = ((iw >> 16) & 31) as usize;
    let rd = ((iw >> 11) & 31) as usize;
    let sa = (iw >> 6) & 31;
    let funct = iw & 0x3F;
    let simm = (iw & 0xFFFF) as i16 as i32 as u32;
    let immu = iw & 0xFFFF;

    // Default: advance to next sequential instruction
    let mut branch_target: Option<u32> = None;

    match op {
        // --- SPECIAL (op 0) --------------------------------------------------
        0 => match funct {
            0x00 => {
                // SLL (NOP when iw == 0)
                if iw != 0 {
                    let v = gpr_load(&bus.rsp_scalar_regs, rt).wrapping_shl(sa);
                    gpr_store(&mut bus.rsp_scalar_regs, rd, v);
                }
            }
            0x02 => {
                // SRL
                let v = gpr_load(&bus.rsp_scalar_regs, rt).wrapping_shr(sa);
                gpr_store(&mut bus.rsp_scalar_regs, rd, v);
            }
            0x03 => {
                // SRA
                let v = (gpr_load(&bus.rsp_scalar_regs, rt) as i32).wrapping_shr(sa) as u32;
                gpr_store(&mut bus.rsp_scalar_regs, rd, v);
            }
            0x04 => {
                // SLLV
                let v = gpr_load(&bus.rsp_scalar_regs, rt)
                    .wrapping_shl(gpr_load(&bus.rsp_scalar_regs, rs) & 31);
                gpr_store(&mut bus.rsp_scalar_regs, rd, v);
            }
            0x06 => {
                // SRLV
                let v = gpr_load(&bus.rsp_scalar_regs, rt)
                    .wrapping_shr(gpr_load(&bus.rsp_scalar_regs, rs) & 31);
                gpr_store(&mut bus.rsp_scalar_regs, rd, v);
            }
            0x07 => {
                // SRAV
                let v = (gpr_load(&bus.rsp_scalar_regs, rt) as i32)
                    .wrapping_shr(gpr_load(&bus.rsp_scalar_regs, rs) & 31)
                    as u32;
                gpr_store(&mut bus.rsp_scalar_regs, rd, v);
            }
            0x08 => {
                // JR
                branch_target = Some(gpr_load(&bus.rsp_scalar_regs, rs) & 0xFFC);
            }
            0x09 => {
                // JALR
                branch_target = Some(gpr_load(&bus.rsp_scalar_regs, rs) & 0xFFC);
                gpr_store(&mut bus.rsp_scalar_regs, rd, (pc as u32 + 8) & 0xFFC);
            }
            0x0D => {
                // BREAK
                static RSP_BREAK_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                let n = RSP_BREAK_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                // Always log BREAK with OSTask type (audio=2, gfx=1) so we can
                // distinguish audio task completions from gfx task completions
                // throughout the entire run, not just the first 10.
                let osk_type = u32::from_be_bytes([
                    bus.rsp_dmem[0xFC0], bus.rsp_dmem[0xFC1],
                    bus.rsp_dmem[0xFC2], bus.rsp_dmem[0xFC3],
                ]);
                static GFX_BREAK_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                static AUD_BREAK_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                if osk_type == 1 {
                    let g = GFX_BREAK_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let dpc_end_now = bus.dpc_regs.read(crate::rcp::DPC_REGS_BASE + crate::rcp::DPC_REG_END);
                    let dpc_start_now = bus.dpc_regs.read(crate::rcp::DPC_REGS_BASE + crate::rcp::DPC_REG_START);
                    let frame = bus.vi.frame_counter;
                    if g < 5 || g >= 5 && g % 10 == 0 {
                        let sz = dpc_end_now.wrapping_sub(dpc_start_now);
                        eprintln!("[RSP] GFX-BREAK #{} f={} PC={:03X} ra={:03X} DPC_START=0x{:08X} DPC_END=0x{:08X} sz=0x{:X}",
                            g, frame, pc, bus.rsp_scalar_regs[31] & 0xFFF, dpc_start_now, dpc_end_now, sz);
                        // Dump the RDP command stream bytes in RDRAM for
                        // this task so we can see whether triangle opcodes
                        // (0x08-0x0F) actually appear or whether it's only
                        // framebuffer-setup bytes. Limit to 512 bytes.
                        let s = dpc_start_now as usize;
                        let e = (dpc_end_now as usize).min(s + 512);
                        if e > s && e <= bus.rdram.data.len() {
                            let bytes: Vec<String> = bus.rdram.data[s..e]
                                .chunks(8)
                                .enumerate()
                                .map(|(i, c)| {
                                    let op = c[0];
                                    format!("{:04X}:{:02X}{:02X}{:02X}{:02X} {:02X}{:02X}{:02X}{:02X}(op{:02X}){}",
                                        s + i*8, c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], op,
                                        if i % 4 == 3 { "\n    " } else { " " })
                                })
                                .collect();
                            eprintln!("[RSP] GFX-BREAK #{} RDRAM[{:X}..{:X}]:\n    {}", g, s, e, bytes.join(""));
                        }
                    }
                    // Dump the captured trace from this gfx task (whichever was armed),
                    // then disarm so subsequent audio tasks don't pollute the trace
                    // / MTC0 logging.
                    if let Ok(mut t) = GFX_TRACE.try_lock() {
                        if t.total > 0 && g < 5 {
                            t.dump_tail(400, &format!("BREAK f={}", frame));
                        }
                        t.armed = false;
                    }
                    // Snapshot IMEM[0x000..0x100] at gfx-break time so we can
                    // diff a working task's handler region against the stuck
                    // task's pure-vector handler (project_sm64_gfx_ucode_loop).
                    if g < 5 {
                        let dump: Vec<String> = bus.rsp_imem[0x000..0x100]
                            .chunks(4)
                            .enumerate()
                            .map(|(i, c)| {
                                if i % 8 == 0 {
                                    format!("\n  {:03X}: {:02X}{:02X}{:02X}{:02X}", i * 4, c[0], c[1], c[2], c[3])
                                } else {
                                    format!("{:02X}{:02X}{:02X}{:02X}", c[0], c[1], c[2], c[3])
                                }
                            })
                            .collect();
                        eprintln!("[RSP] GFX-BREAK #{} IMEM[000..100]:{}", g, dump.join(" "));
                        // Also dump the dispatch table area DMEM[0x0B0..0x180]
                        // — Fast3D reads LH $v0, 0xBC($at) from here.
                        let dmem: Vec<String> = bus.rsp_dmem[0x0B0..0x180]
                            .chunks(4)
                            .enumerate()
                            .map(|(i, c)| {
                                if i % 8 == 0 {
                                    format!("\n  {:03X}: {:02X}{:02X}{:02X}{:02X}", 0x0B0 + i * 4, c[0], c[1], c[2], c[3])
                                } else {
                                    format!("{:02X}{:02X}{:02X}{:02X}", c[0], c[1], c[2], c[3])
                                }
                            })
                            .collect();
                        eprintln!("[RSP] GFX-BREAK #{} DMEM[0B0..180] (dispatch tbl):{}", g, dmem.join(" "));
                    }
                    // Vertex-cache snapshot: Fast3D stores the transformed
                    // vertex buffer around DMEM[0x400..0x580]. Dump it at
                    // break time so we can see whether screen x/y/z are
                    // sane or garbage. 16 verts x ~24 bytes each. Also
                    // dump the RDP output buffer area DMEM[0xFB0..0xFC0]
                    // which holds the current DP tail pointer.
                    if g < 3 || (g >= 5 && g % 50 == 0) {
                        let vtx: Vec<String> = bus.rsp_dmem[0x400..0x580]
                            .chunks(4)
                            .enumerate()
                            .map(|(i, c)| {
                                if i % 8 == 0 {
                                    format!("\n  {:03X}: {:02X}{:02X}{:02X}{:02X}", 0x400 + i * 4, c[0], c[1], c[2], c[3])
                                } else {
                                    format!("{:02X}{:02X}{:02X}{:02X}", c[0], c[1], c[2], c[3])
                                }
                            })
                            .collect();
                        eprintln!("[RSP] GFX-BREAK #{} DMEM[400..580] (vtx cache?):{}", g, vtx.join(" "));
                        // Also try 0x800..0xC00 — some F3D variants place
                        // the vertex cache here (between RDP output FIFO
                        // and scratch area).
                        let vtx2: Vec<String> = bus.rsp_dmem[0x800..0xC00]
                            .chunks(4)
                            .enumerate()
                            .map(|(i, c)| {
                                if i % 8 == 0 {
                                    format!("\n  {:03X}: {:02X}{:02X}{:02X}{:02X}", 0x800 + i * 4, c[0], c[1], c[2], c[3])
                                } else {
                                    format!("{:02X}{:02X}{:02X}{:02X}", c[0], c[1], c[2], c[3])
                                }
                            })
                            .collect();
                        eprintln!("[RSP] GFX-BREAK #{} DMEM[800..C00]:{}", g, vtx2.join(" "));
                    }
                } else {
                    let a = AUD_BREAK_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let frame = bus.vi.frame_counter;
                    if a < 3 || a % 100 == 0 {
                        eprintln!("[RSP] AUD-BREAK #{} f={} PC={:03X}", a, frame, pc);
                    }
                }
                if n < 10 {
                    eprintln!("[RSP] BREAK #{} type={} at PC={:03X} ra={:03X}", n, osk_type, pc, bus.rsp_scalar_regs[31] & 0xFFF);
                    // Dump a wide window of IMEM to follow the path from rspboot's
                    // entry (around 0x080) to the break point.
                    let base = 0x080usize.min(pc & !0x3);
                    let end = ((pc + 0x20) & !0x3).max(base + 0x10).min(bus.rsp_imem.len());
                    let dump: Vec<String> = bus.rsp_imem[base..end]
                        .chunks(4)
                        .enumerate()
                        .map(|(i, c)| {
                            if i % 8 == 0 {
                                format!("\n  {:03X}: {:02X}{:02X}{:02X}{:02X}", base + i * 4, c[0], c[1], c[2], c[3])
                            } else {
                                format!("{:02X}{:02X}{:02X}{:02X}", c[0], c[1], c[2], c[3])
                            }
                        })
                        .collect();
                    eprintln!("[RSP] IMEM[{:03X}..{:03X}] at BREAK:{}", base, end, dump.join(" "));
                    // Dump OSTask (DMEM[0xFC0..0x1000]) and DL buffer area
                    // (DMEM[0x380..0x400]) to see what data the ucode received.
                    let osk: Vec<String> = bus.rsp_dmem[0xFC0..0x1000]
                        .chunks(4)
                        .enumerate()
                        .map(|(i, c)| {
                            if i % 8 == 0 {
                                format!("\n  {:03X}: {:02X}{:02X}{:02X}{:02X}", 0xFC0 + i * 4, c[0], c[1], c[2], c[3])
                            } else {
                                format!("{:02X}{:02X}{:02X}{:02X}", c[0], c[1], c[2], c[3])
                            }
                        })
                        .collect();
                    eprintln!("[RSP] DMEM[FC0..1000] (OSTask) at BREAK:{}", osk.join(" "));
                    let dlb: Vec<String> = bus.rsp_dmem[0x380..0x400]
                        .chunks(4)
                        .enumerate()
                        .map(|(i, c)| {
                            if i % 8 == 0 {
                                format!("\n  {:03X}: {:02X}{:02X}{:02X}{:02X}", 0x380 + i * 4, c[0], c[1], c[2], c[3])
                            } else {
                                format!("{:02X}{:02X}{:02X}{:02X}", c[0], c[1], c[2], c[3])
                            }
                        })
                        .collect();
                    eprintln!("[RSP] DMEM[380..400] (DL buf) at BREAK:{}", dlb.join(" "));
                    // Also check DPC state at break time — was a display list kicked?
                    let dpc_start = bus.dpc_regs.read(crate::rcp::DPC_REGS_BASE + crate::rcp::DPC_REG_START);
                    let dpc_end = bus.dpc_regs.read(crate::rcp::DPC_REGS_BASE + crate::rcp::DPC_REG_END);
                    let dpc_curr = bus.dpc_regs.read(crate::rcp::DPC_REGS_BASE + crate::rcp::DPC_REG_CURRENT);
                    let dpc_status = bus.dpc_regs.read(crate::rcp::DPC_REGS_BASE + crate::rcp::DPC_REG_STATUS);
                    eprintln!("[RSP] DPC at BREAK: start=0x{:08X} end=0x{:08X} curr=0x{:08X} status=0x{:08X}",
                        dpc_start, dpc_end, dpc_curr, dpc_status);
                }
                bus.sp_broke = true;
                bus.sp_halted = true;
                // Mirror Mupen64Plus `rsp_interrupt_event`: on RSP task
                // completion via BREAK, hardware auto-sets TASKDONE along with
                // BROKE|HALT. TASKDONE = SP_STATUS bit 9 = SIG2 = bit 2 of
                // `sp_signal`. libultra's SP interrupt handler ANDs SP_STATUS
                // with 0x300 (SIG1|SIG2 = YIELDED|TASKDONE) to route the IRQ
                // to OS_EVENT_SP; without SIG2 set, a gfx BREAK gets routed to
                // OS_EVENT_CPU_BREAK and the scheduler never acks the task,
                // deadlocking after gfx task #326 in SM64.
                bus.sp_signal |= 1 << 2;
                bus.mi.raise(MI_INTR_SP);
                return RSP_CYCLES_PER_INSTR;
            }
            0x20 | 0x21 => {
                // ADD / ADDU
                let v = gpr_load(&bus.rsp_scalar_regs, rs)
                    .wrapping_add(gpr_load(&bus.rsp_scalar_regs, rt));
                gpr_store(&mut bus.rsp_scalar_regs, rd, v);
            }
            0x22 | 0x23 => {
                // SUB / SUBU
                let v = gpr_load(&bus.rsp_scalar_regs, rs)
                    .wrapping_sub(gpr_load(&bus.rsp_scalar_regs, rt));
                gpr_store(&mut bus.rsp_scalar_regs, rd, v);
            }
            0x24 => {
                // AND
                let v = gpr_load(&bus.rsp_scalar_regs, rs)
                    & gpr_load(&bus.rsp_scalar_regs, rt);
                gpr_store(&mut bus.rsp_scalar_regs, rd, v);
            }
            0x25 => {
                // OR
                let v = gpr_load(&bus.rsp_scalar_regs, rs)
                    | gpr_load(&bus.rsp_scalar_regs, rt);
                gpr_store(&mut bus.rsp_scalar_regs, rd, v);
            }
            0x26 => {
                // XOR
                let v = gpr_load(&bus.rsp_scalar_regs, rs)
                    ^ gpr_load(&bus.rsp_scalar_regs, rt);
                gpr_store(&mut bus.rsp_scalar_regs, rd, v);
            }
            0x27 => {
                // NOR
                let v = !(gpr_load(&bus.rsp_scalar_regs, rs)
                    | gpr_load(&bus.rsp_scalar_regs, rt));
                gpr_store(&mut bus.rsp_scalar_regs, rd, v);
            }
            0x2A => {
                // SLT
                let v = u32::from(
                    (gpr_load(&bus.rsp_scalar_regs, rs) as i32)
                        < (gpr_load(&bus.rsp_scalar_regs, rt) as i32),
                );
                gpr_store(&mut bus.rsp_scalar_regs, rd, v);
            }
            0x2B => {
                // SLTU
                let v = u32::from(
                    gpr_load(&bus.rsp_scalar_regs, rs)
                        < gpr_load(&bus.rsp_scalar_regs, rt),
                );
                gpr_store(&mut bus.rsp_scalar_regs, rd, v);
            }
            _ => {} // Unknown SPECIAL funct — NOP
        },

        // --- REGIMM (op 1) ---------------------------------------------------
        1 => {
            let off = ((iw & 0xFFFF) as i16 as i32) << 2;
            let target = ((pc as i32 + 4 + off) as u32) & 0xFFC;
            match rt {
                0x00 => {
                    // BLTZ
                    if (gpr_load(&bus.rsp_scalar_regs, rs) as i32) < 0 {
                        branch_target = Some(target);
                    }
                }
                0x01 => {
                    // BGEZ
                    if (gpr_load(&bus.rsp_scalar_regs, rs) as i32) >= 0 {
                        branch_target = Some(target);
                    }
                }
                0x10 => {
                    // BLTZAL
                    gpr_store(&mut bus.rsp_scalar_regs, 31, (pc as u32 + 8) & 0xFFC);
                    if (gpr_load(&bus.rsp_scalar_regs, rs) as i32) < 0 {
                        branch_target = Some(target);
                    }
                }
                0x11 => {
                    // BGEZAL
                    gpr_store(&mut bus.rsp_scalar_regs, 31, (pc as u32 + 8) & 0xFFC);
                    if (gpr_load(&bus.rsp_scalar_regs, rs) as i32) >= 0 {
                        branch_target = Some(target);
                    }
                }
                _ => {}
            }
        }

        // --- J (op 2) --------------------------------------------------------
        2 => {
            let t = iw & 0x03FF_FFFF;
            branch_target = Some((t << 2) & 0xFFC);
        }

        // --- JAL (op 3) ------------------------------------------------------
        3 => {
            let t = iw & 0x03FF_FFFF;
            branch_target = Some((t << 2) & 0xFFC);
            gpr_store(&mut bus.rsp_scalar_regs, 31, (pc as u32 + 8) & 0xFFC);
        }

        // --- BEQ (op 4) ------------------------------------------------------
        4 => {
            let off = ((iw & 0xFFFF) as i16 as i32) << 2;
            if gpr_load(&bus.rsp_scalar_regs, rs) == gpr_load(&bus.rsp_scalar_regs, rt) {
                branch_target = Some(((pc as i32 + 4 + off) as u32) & 0xFFC);
            }
        }

        // --- BNE (op 5) ------------------------------------------------------
        5 => {
            let off = ((iw & 0xFFFF) as i16 as i32) << 2;
            if gpr_load(&bus.rsp_scalar_regs, rs) != gpr_load(&bus.rsp_scalar_regs, rt) {
                branch_target = Some(((pc as i32 + 4 + off) as u32) & 0xFFC);
            }
        }

        // --- BLEZ (op 6) -----------------------------------------------------
        6 => {
            let off = ((iw & 0xFFFF) as i16 as i32) << 2;
            if (gpr_load(&bus.rsp_scalar_regs, rs) as i32) <= 0 {
                branch_target = Some(((pc as i32 + 4 + off) as u32) & 0xFFC);
            }
        }

        // --- BGTZ (op 7) -----------------------------------------------------
        7 => {
            let off = ((iw & 0xFFFF) as i16 as i32) << 2;
            if (gpr_load(&bus.rsp_scalar_regs, rs) as i32) > 0 {
                branch_target = Some(((pc as i32 + 4 + off) as u32) & 0xFFC);
            }
        }

        // --- ADDI / ADDIU (op 8/9) -------------------------------------------
        8 | 9 => {
            let v = gpr_load(&bus.rsp_scalar_regs, rs).wrapping_add(simm);
            gpr_store(&mut bus.rsp_scalar_regs, rt, v);
        }

        // --- SLTI (op 10) ----------------------------------------------------
        10 => {
            let v = u32::from((gpr_load(&bus.rsp_scalar_regs, rs) as i32) < (simm as i32));
            gpr_store(&mut bus.rsp_scalar_regs, rt, v);
        }

        // --- SLTIU (op 11) ---------------------------------------------------
        11 => {
            let v = u32::from(gpr_load(&bus.rsp_scalar_regs, rs) < simm);
            gpr_store(&mut bus.rsp_scalar_regs, rt, v);
        }

        // --- ANDI (op 12) ----------------------------------------------------
        12 => {
            let v = gpr_load(&bus.rsp_scalar_regs, rs) & immu;
            gpr_store(&mut bus.rsp_scalar_regs, rt, v);
        }

        // --- ORI (op 13) -----------------------------------------------------
        13 => {
            let v = gpr_load(&bus.rsp_scalar_regs, rs) | immu;
            gpr_store(&mut bus.rsp_scalar_regs, rt, v);
        }

        // --- XORI (op 14) ----------------------------------------------------
        14 => {
            let v = gpr_load(&bus.rsp_scalar_regs, rs) ^ immu;
            gpr_store(&mut bus.rsp_scalar_regs, rt, v);
        }

        // --- LUI (op 15) -----------------------------------------------------
        15 => {
            gpr_store(&mut bus.rsp_scalar_regs, rt, immu << 16);
        }

        // --- COP0 (op 16) — MFC0 / MTC0 for SP/DPC registers ----------------
        16 => {
            match rs {
                0x00 => {
                    // MFC0 rt, rd — move from COP0 register
                    let v = rsp_cop0_read(bus, rd);
                    gpr_store(&mut bus.rsp_scalar_regs, rt, v);
                }
                0x04 => {
                    // MTC0 rt, rd — move to COP0 register
                    let v = gpr_load(&bus.rsp_scalar_regs, rt);
                    // SP DMA / DPC kick logging — only when a gfx task is armed.
                    if rd <= 3 || rd == 8 || rd == 9 || rd == 11 {
                        let gfx_armed = GFX_TRACE.try_lock().map(|t| t.armed).unwrap_or(false);
                        if gfx_armed {
                            static MTC0_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                            let n = MTC0_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if n < 800 || n % 50000 == 0 {
                                let regname = match rd {
                                    0 => "SP_MEM_ADDR",
                                    1 => "SP_DRAM_ADDR",
                                    2 => "SP_RD_LEN",
                                    3 => "SP_WR_LEN",
                                    8 => "DPC_START",
                                    9 => "DPC_END",
                                    11 => "DPC_STATUS",
                                    _ => "?",
                                };
                                let k1 = bus.rsp_scalar_regs[27] as u32;
                                let gp = bus.rsp_scalar_regs[28] as u32;
                                eprintln!("[GFX MTC0 #{}] pc=0x{:03X} {} <- 0x{:08X}  k1=0x{:04X} gp=0x{:08X}",
                                    n, pc, regname, v, k1, gp);
                            }
                        }
                    }
                    rsp_cop0_write(bus, rd, v);
                }
                _ => {} // Other COP0 sub-ops ignored
            }
        }

        // --- COP2 (op 18) — vector unit -----------------------------------------
        18 => {
            rsp_vu::execute_cop2(
                &mut bus.rsp_vu,
                &mut bus.rsp_scalar_regs,
                &mut bus.rsp_dmem,
                iw,
            );
        }

        // --- LB (op 32) — sign-extended byte load ----------------------------
        32 => {
            let addr = gpr_load(&bus.rsp_scalar_regs, rs).wrapping_add(simm);
            let v = rsp_load_flat_u8(bus, addr) as i8 as i32 as u32;
            gpr_store(&mut bus.rsp_scalar_regs, rt, v);
        }

        // --- LH (op 33) — sign-extended halfword load ------------------------
        33 => {
            let addr = gpr_load(&bus.rsp_scalar_regs, rs).wrapping_add(simm);
            let v = rsp_load_flat_u16(bus, addr) as i16 as i32 as u32;
            gpr_store(&mut bus.rsp_scalar_regs, rt, v);
        }

        // --- LW (op 35) ------------------------------------------------------
        35 => {
            let base = gpr_load(&bus.rsp_scalar_regs, rs).wrapping_add(simm);
            let v = rsp_load_flat(bus, base);
            gpr_store(&mut bus.rsp_scalar_regs, rt, v);
        }

        // --- LBU (op 36) — zero-extended byte load ---------------------------
        36 => {
            let addr = gpr_load(&bus.rsp_scalar_regs, rs).wrapping_add(simm);
            let v = rsp_load_flat_u8(bus, addr) as u32;
            gpr_store(&mut bus.rsp_scalar_regs, rt, v);
        }

        // --- LHU (op 37) — zero-extended halfword load -----------------------
        37 => {
            let addr = gpr_load(&bus.rsp_scalar_regs, rs).wrapping_add(simm);
            let v = rsp_load_flat_u16(bus, addr) as u32;
            gpr_store(&mut bus.rsp_scalar_regs, rt, v);
        }

        // --- SB (op 40) — byte store -----------------------------------------
        40 => {
            let addr = gpr_load(&bus.rsp_scalar_regs, rs).wrapping_add(simm);
            rsp_store_flat_u8(bus, addr, gpr_load(&bus.rsp_scalar_regs, rt) as u8);
        }

        // --- SH (op 41) — halfword store -------------------------------------
        41 => {
            let addr = gpr_load(&bus.rsp_scalar_regs, rs).wrapping_add(simm);
            let val = gpr_load(&bus.rsp_scalar_regs, rt) as u16;
            // GFX store-trace: log any SH to DMEM[0x000..0x6A0] (segment table
            // / scratch area, below the DL ring buffer) during a gfx task.
            let daddr = (addr & 0xFFF) as usize;
            if daddr < 0x6A0 {
                let gfx_armed = GFX_TRACE.try_lock().map(|t| t.armed).unwrap_or(false);
                if gfx_armed {
                    static SH_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                    let n = SH_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if n < 600 {
                        eprintln!("[GFX SH #{}] pc=0x{:03X} dmem[0x{:03X}]<-0x{:04X} (base r{}=0x{:04X}+off)",
                            n, pc, daddr, val, rs, gpr_load(&bus.rsp_scalar_regs, rs) & 0xFFF);
                    }
                }
            }
            rsp_store_flat_u16(bus, addr, val);
        }

        // --- SW (op 43) ------------------------------------------------------
        43 => {
            let base = gpr_load(&bus.rsp_scalar_regs, rs).wrapping_add(simm);
            let val = gpr_load(&bus.rsp_scalar_regs, rt);
            // GFX store-trace: log any SW to DMEM[0x000..0x6A0] during gfx task.
            let daddr = (base & 0xFFF) as usize;
            if daddr < 0x6A0 {
                let gfx_armed = GFX_TRACE.try_lock().map(|t| t.armed).unwrap_or(false);
                if gfx_armed {
                    static SW_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                    let n = SW_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if n < 600 {
                        eprintln!("[GFX SW #{}] pc=0x{:03X} dmem[0x{:03X}]<-0x{:08X} (base r{}=0x{:04X}+off)",
                            n, pc, daddr, val, rs, gpr_load(&bus.rsp_scalar_regs, rs) & 0xFFF);
                    }
                }
            }
            rsp_store_flat(bus, base, val);
        }

        // --- LWC2 (op 50) — vector load from DMEM ----------------------------
        50 => {
            let vt = rt;
            let element = (rd >> 0) & 0xF; // element field from bits 10:7 is actually (iw >> 7) & 0xF
            let el = ((iw >> 7) & 0xF) as usize;
            let load_op = (rd >> 0) & 0x1F; // rd field = load sub-opcode
            let offset7 = (iw & 0x7F) as i8 as i32; // 7-bit signed offset
            // Scale depends on load type
            let scale = match load_op as usize {
                0 => 1, // LBV
                1 => 2, // LSV
                2 => 4, // LLV
                3 => 8, // LDV
                4 => 16, // LQV
                5 => 16, // LRV
                6 => 8, // LPV
                7 => 8, // LUV
                11 => 16, // LTV
                _ => 1,
            };
            let _ = (vt, element);
            let base = gpr_load(&bus.rsp_scalar_regs, rs);
            let addr = base.wrapping_add((offset7 * scale) as u32) as usize;
            rsp_vu::vector_load(
                &mut bus.rsp_vu,
                &bus.rsp_dmem,
                vt,
                el,
                addr,
                load_op as usize,
            );
        }

        // --- SWC2 (op 58) — vector store to DMEM ----------------------------
        58 => {
            let vt = rt;
            let el = ((iw >> 7) & 0xF) as usize;
            let store_op = (rd >> 0) & 0x1F;
            let offset7 = (iw & 0x7F) as i8 as i32;
            let scale = match store_op as usize {
                0 => 1,
                1 => 2,
                2 => 4,
                3 => 8,
                4 => 16,
                5 => 16,
                6 => 8,
                7 => 8,
                11 => 16,
                _ => 1,
            };
            let base = gpr_load(&bus.rsp_scalar_regs, rs);
            let addr = base.wrapping_add((offset7 * scale) as u32) as usize;
            rsp_vu::vector_store(
                &bus.rsp_vu,
                &mut bus.rsp_dmem,
                vt,
                el,
                addr,
                store_op as usize,
            );
        }

        _ => {
            RSP_UNIMPL_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            RSP_LAST_UNIMPL.store(iw, std::sync::atomic::Ordering::Relaxed);
        }
    }

    // --- Delay slot handling -------------------------------------------------
    // If we already have a pending delay slot, the current instruction was the
    // delay slot instruction. Apply the previously-latched branch target.
    if let Some(target) = bus.rsp_delay_slot_target.take() {
        bus.rsp_pc = target & 0xFFC;
    } else if let Some(target) = branch_target {
        // Branch was just decided — latch target, execute next instruction
        // (the delay slot) at pc+4 first.
        bus.rsp_delay_slot_target = Some(target);
        bus.rsp_pc = (pc as u32 + 4) & 0xFFC;
    } else {
        bus.rsp_pc = (pc as u32 + 4) & 0xFFC;
    }

    RSP_CYCLES_PER_INSTR
}

/// Run RSP work for the same RCP quantum as CPU/DMA (coarse).
pub fn run_for_rcp_quantum(bus: &mut SystemBus, rcp_cycles: u64) {
    if bus.sp_halted || rcp_cycles == 0 {
        return;
    }
    let mut budget = rcp_cycles.saturating_mul(8).min(65_536);
    while budget >= RSP_CYCLES_PER_INSTR {
        let c = step_instruction(bus);
        if c == 0 {
            break;
        }
        budget -= c;
        if bus.sp_halted {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bus::SystemBus;

    fn write_imem(bus: &mut SystemBus, off: usize, w: u32) {
        bus.rsp_imem[off..off + 4].copy_from_slice(&w.to_be_bytes());
    }

    fn run_n(bus: &mut SystemBus, n: usize) {
        for _ in 0..n {
            step_instruction(bus);
        }
    }

    fn fresh_bus() -> SystemBus {
        let mut bus = SystemBus::with_rdram_size(1024 * 1024);
        bus.sp_halted = false;
        bus.rsp_pc = 0;
        bus
    }

    #[test]
    fn nop_then_break_halts() {
        let mut bus = fresh_bus();
        write_imem(&mut bus, 0, 0x0000_0000); // NOP
        write_imem(&mut bus, 4, 0x0000_000D); // BREAK
        assert_eq!(step_instruction(&mut bus), RSP_CYCLES_PER_INSTR);
        assert_eq!(bus.rsp_pc, 4);
        assert!(!bus.sp_halted);
        assert_eq!(step_instruction(&mut bus), RSP_CYCLES_PER_INSTR);
        assert!(bus.sp_halted);
        assert!(bus.sp_broke);
    }

    #[test]
    fn addiu_lw_sw_round_trip() {
        let mut bus = fresh_bus();
        // ADDIU r1, r0, 0x40
        write_imem(&mut bus, 0, 0x2401_0040);
        // SW r1, 0(r0)
        write_imem(&mut bus, 4, 0xAC01_0000);
        // LW r2, 0(r0)
        write_imem(&mut bus, 8, 0x8C02_0000);
        run_n(&mut bus, 3);
        assert_eq!(bus.rsp_scalar_regs[2], 0x40);
    }

    #[test]
    fn addiu_to_r0_discards_result() {
        let mut bus = fresh_bus();
        // ADDIU r0, r0, 0x123
        write_imem(&mut bus, 0, 0x2400_0123);
        step_instruction(&mut bus);
        assert_eq!(bus.rsp_scalar_regs[0], 0);
    }

    // --- New scalar instruction tests ----------------------------------------

    #[test]
    fn and_or_xor_nor() {
        let mut bus = fresh_bus();
        bus.rsp_scalar_regs[1] = 0xFF00_FF00;
        bus.rsp_scalar_regs[2] = 0x0F0F_0F0F;
        // AND r3, r1, r2  => SPECIAL rd=3, rs=1, rt=2, funct=0x24
        write_imem(&mut bus, 0, 0x0022_1824); // AND
        // OR  r4, r1, r2  => funct=0x25
        write_imem(&mut bus, 4, 0x0022_2025); // OR
        // XOR r5, r1, r2  => funct=0x26
        write_imem(&mut bus, 8, 0x0022_2826); // XOR
        // NOR r6, r1, r2  => funct=0x27
        write_imem(&mut bus, 12, 0x0022_3027); // NOR
        run_n(&mut bus, 4);
        assert_eq!(bus.rsp_scalar_regs[3], 0x0F00_0F00);
        assert_eq!(bus.rsp_scalar_regs[4], 0xFF0F_FF0F);
        assert_eq!(bus.rsp_scalar_regs[5], 0xF00F_F00F);
        assert_eq!(bus.rsp_scalar_regs[6], !(0xFF0F_FF0F));
    }

    #[test]
    fn slt_sltu() {
        let mut bus = fresh_bus();
        bus.rsp_scalar_regs[1] = 0xFFFF_FFFF; // -1 signed, large unsigned
        bus.rsp_scalar_regs[2] = 1;
        // SLT r3, r1, r2 => funct=0x2A: -1 < 1 => 1
        write_imem(&mut bus, 0, 0x0022_182A);
        // SLTU r4, r1, r2 => funct=0x2B: 0xFFFFFFFF < 1 => 0
        write_imem(&mut bus, 4, 0x0022_202B);
        run_n(&mut bus, 2);
        assert_eq!(bus.rsp_scalar_regs[3], 1);
        assert_eq!(bus.rsp_scalar_regs[4], 0);
    }

    #[test]
    fn srl_sra_sllv_srlv_srav() {
        let mut bus = fresh_bus();
        bus.rsp_scalar_regs[1] = 0x8000_0000;
        bus.rsp_scalar_regs[2] = 4; // shift amount in register
        // SRL r3, r1, 4 => funct=0x02, sa=4
        write_imem(&mut bus, 0, 0x0001_1902); // SRL rd=3, rt=1, sa=4
        // SRA r4, r1, 4 => funct=0x03, sa=4
        write_imem(&mut bus, 4, 0x0001_2103); // SRA rd=4, rt=1, sa=4
        // SLLV r5, r1, r2 => funct=0x04: r1 << r2
        write_imem(&mut bus, 8, 0x0041_2804); // SLLV rd=5, rt=1, rs=2
        // SRLV r6, r1, r2 => funct=0x06: r1 >> r2 (logical)
        write_imem(&mut bus, 12, 0x0041_3006); // SRLV rd=6, rt=1, rs=2
        // SRAV r7, r1, r2 => funct=0x07: r1 >> r2 (arithmetic)
        write_imem(&mut bus, 16, 0x0041_3807); // SRAV rd=7, rt=1, rs=2
        run_n(&mut bus, 5);
        assert_eq!(bus.rsp_scalar_regs[3], 0x0800_0000);
        assert_eq!(bus.rsp_scalar_regs[4], 0xF800_0000u32);
        assert_eq!(bus.rsp_scalar_regs[5], 0x0000_0000); // 0x80000000 << 4 = 0
        assert_eq!(bus.rsp_scalar_regs[6], 0x0800_0000);
        assert_eq!(bus.rsp_scalar_regs[7], 0xF800_0000u32);
    }

    #[test]
    fn andi_xori_slti_sltiu() {
        let mut bus = fresh_bus();
        bus.rsp_scalar_regs[1] = 0xFF00_FF00;
        // ANDI r2, r1, 0x00FF
        write_imem(&mut bus, 0, 0x3022_00FF);
        // XORI r3, r1, 0x00FF
        write_imem(&mut bus, 4, 0x3823_00FF);
        run_n(&mut bus, 2);
        assert_eq!(bus.rsp_scalar_regs[2], 0x0000_0000);
        assert_eq!(bus.rsp_scalar_regs[3], 0xFF00_FFFF);

        // SLTI: -1 < 0 => true
        bus.rsp_scalar_regs[4] = 0xFFFF_FFFF;
        // SLTI r5, r4, 0 (op=10, rs=4, rt=5, imm=0)
        write_imem(&mut bus, 8, 0x2885_0000);
        step_instruction(&mut bus);
        assert_eq!(bus.rsp_scalar_regs[5], 1);

        // SLTIU: 0xFFFFFFFF < 0 (sign-extended to 0xFFFFFFFF) => false (equal)
        // SLTIU r6, r4, 0 (op=11, rs=4, rt=6, imm=0)
        write_imem(&mut bus, 12, 0x2C86_0000);
        step_instruction(&mut bus);
        assert_eq!(bus.rsp_scalar_regs[6], 0);
    }

    #[test]
    fn lb_lbu_lh_lhu_sb_sh() {
        let mut bus = fresh_bus();
        // Store 0xDEAD at DMEM[0] via direct write
        bus.rsp_dmem[0] = 0xDE;
        bus.rsp_dmem[1] = 0xAD;
        bus.rsp_dmem[2] = 0xBE;
        bus.rsp_dmem[3] = 0xEF;

        // LBU r1, 0(r0) => op=36: zero-extended 0xDE = 0x000000DE
        write_imem(&mut bus, 0, 0x9001_0000);
        // LB r2, 0(r0) => op=32: sign-extended 0xDE = 0xFFFFFFDE
        write_imem(&mut bus, 4, 0x8002_0000);
        // LHU r3, 0(r0) => op=37: zero-extended 0xDEAD
        write_imem(&mut bus, 8, 0x9403_0000);
        // LH r4, 0(r0) => op=33: sign-extended 0xDEAD = 0xFFFFDEAD
        write_imem(&mut bus, 12, 0x8404_0000);
        run_n(&mut bus, 4);
        assert_eq!(bus.rsp_scalar_regs[1], 0x0000_00DE);
        assert_eq!(bus.rsp_scalar_regs[2], 0xFFFF_FFDE);
        assert_eq!(bus.rsp_scalar_regs[3], 0x0000_DEAD);
        assert_eq!(bus.rsp_scalar_regs[4], 0xFFFF_DEAD);

        // SB r1 (0xDE), 8(r0) => op=40
        write_imem(&mut bus, 16, 0xA001_0008);
        // SH r3 (0xDEAD), 10(r0) => op=41: store halfword at DMEM[10]
        write_imem(&mut bus, 20, 0xA403_000A);
        run_n(&mut bus, 2);
        assert_eq!(bus.rsp_dmem[8], 0xDE);
        assert_eq!(bus.rsp_dmem[10], 0xDE);
        assert_eq!(bus.rsp_dmem[11], 0xAD);
    }

    #[test]
    fn branch_delay_slot_j() {
        let mut bus = fresh_bus();
        // 0x000: J 0x010 (target word addr = 4 => byte addr 0x010)
        //        J target = (iw & 0x03FFFFFF) << 2 = 4 << 2 = 0x10
        write_imem(&mut bus, 0x000, 0x0800_0004); // J target_word=4
        // 0x004: ADDIU r1, r0, 42 — delay slot, should execute
        write_imem(&mut bus, 0x004, 0x2401_002A);
        // 0x008: ADDIU r2, r0, 99 — should NOT execute
        write_imem(&mut bus, 0x008, 0x2402_0063);
        // 0x010: ADDIU r3, r0, 7 — jump target
        write_imem(&mut bus, 0x010, 0x2403_0007);

        run_n(&mut bus, 3);
        assert_eq!(bus.rsp_scalar_regs[1], 42, "delay slot should execute");
        assert_eq!(bus.rsp_scalar_regs[2], 0, "instruction after delay slot should be skipped");
        assert_eq!(bus.rsp_scalar_regs[3], 7, "jump target should execute");
        assert_eq!(bus.rsp_pc, 0x014);
    }

    #[test]
    fn branch_delay_slot_bne_taken() {
        let mut bus = fresh_bus();
        bus.rsp_scalar_regs[1] = 1;
        // 0x000: BNE r1, r0, +2 (offset +2 instructions = +8 bytes from pc+4)
        //        BNE: op=5, rs=1, rt=0, offset=2
        write_imem(&mut bus, 0x000, 0x1420_0002);
        // 0x004: ADDIU r2, r0, 55 — delay slot, should execute
        write_imem(&mut bus, 0x004, 0x2402_0037);
        // 0x008: ADDIU r3, r0, 88 — should NOT execute (branched past)
        write_imem(&mut bus, 0x008, 0x2403_0058);
        // 0x00C: ADDIU r4, r0, 77 — branch target (pc+4 + 8 = 0x00C)
        write_imem(&mut bus, 0x00C, 0x2404_004D);

        run_n(&mut bus, 3);
        assert_eq!(bus.rsp_scalar_regs[2], 55, "delay slot executes");
        assert_eq!(bus.rsp_scalar_regs[3], 0, "skipped instruction");
        assert_eq!(bus.rsp_scalar_regs[4], 77, "branch target executes");
    }

    #[test]
    fn branch_delay_slot_bne_not_taken() {
        let mut bus = fresh_bus();
        // r1 = 0, r0 = 0 => BNE not taken
        // 0x000: BNE r1, r0, +2
        write_imem(&mut bus, 0x000, 0x1420_0002);
        // 0x004: ADDIU r2, r0, 55 — delay slot still executes (not-taken branch)
        write_imem(&mut bus, 0x004, 0x2402_0037);
        // 0x008: ADDIU r3, r0, 88 — falls through, should execute
        write_imem(&mut bus, 0x008, 0x2403_0058);

        run_n(&mut bus, 3);
        assert_eq!(bus.rsp_scalar_regs[2], 55, "delay slot executes even on not-taken");
        assert_eq!(bus.rsp_scalar_regs[3], 88, "fall-through executes");
    }

    #[test]
    fn jal_stores_return_address() {
        let mut bus = fresh_bus();
        // 0x000: JAL target_word=8 => byte addr 0x020
        write_imem(&mut bus, 0x000, 0x0C00_0008);
        // 0x004: NOP (delay slot)
        write_imem(&mut bus, 0x004, 0x0000_0000);
        // 0x020: BREAK
        write_imem(&mut bus, 0x020, 0x0000_000D);

        run_n(&mut bus, 2); // JAL + delay slot
        // RA should be pc+8 = 0x008
        assert_eq!(bus.rsp_scalar_regs[31], 0x008);
        assert_eq!(bus.rsp_pc, 0x020);
    }

    #[test]
    fn beq_blez_bgtz() {
        // BEQ r1, r2, +1 => target = pc+4 + 4 = 0x008 (delay slot at 0x004)
        let mut bus = fresh_bus();
        bus.rsp_scalar_regs[1] = 0;
        bus.rsp_scalar_regs[2] = 0;
        write_imem(&mut bus, 0x000, 0x1022_0001); // BEQ r1, r2, +1
        write_imem(&mut bus, 0x004, 0x0000_0000); // delay slot NOP
        write_imem(&mut bus, 0x008, 0x0000_000D); // BREAK at target
        run_n(&mut bus, 2); // branch + delay slot
        assert_eq!(bus.rsp_pc, 0x008);

        // BLEZ with r1 = 0 => taken
        let mut bus2 = fresh_bus();
        bus2.rsp_scalar_regs[1] = 0;
        write_imem(&mut bus2, 0x000, 0x1820_0001); // BLEZ r1, +1
        write_imem(&mut bus2, 0x004, 0x0000_0000);
        run_n(&mut bus2, 2);
        assert_eq!(bus2.rsp_pc, 0x008);

        // BGTZ with r1 = 5 => taken
        let mut bus3 = fresh_bus();
        bus3.rsp_scalar_regs[1] = 5;
        write_imem(&mut bus3, 0x000, 0x1C20_0001); // BGTZ r1, +1
        write_imem(&mut bus3, 0x004, 0x0000_0000);
        run_n(&mut bus3, 2);
        assert_eq!(bus3.rsp_pc, 0x008);
    }

    #[test]
    fn regimm_bltz_bgez() {
        // BLTZ: taken when rs < 0; target = pc+4 + 4 = 0x008
        let mut bus = fresh_bus();
        bus.rsp_scalar_regs[1] = 0xFFFF_FFFF; // -1
        write_imem(&mut bus, 0x000, 0x0420_0001); // BLTZ r1, +1
        write_imem(&mut bus, 0x004, 0x0000_0000); // delay slot
        run_n(&mut bus, 2);
        assert_eq!(bus.rsp_pc, 0x008);

        // BGEZ: taken when rs >= 0
        let mut bus2 = fresh_bus();
        bus2.rsp_scalar_regs[1] = 0;
        write_imem(&mut bus2, 0x000, 0x0421_0001); // BGEZ r1, +1
        write_imem(&mut bus2, 0x004, 0x0000_0000);
        run_n(&mut bus2, 2);
        assert_eq!(bus2.rsp_pc, 0x008);
    }

    #[test]
    fn cop0_mfc0_reads_sp_status() {
        let mut bus = fresh_bus();
        bus.sp_halted = false;
        bus.sp_broke = false;
        // MFC0 r1, $4 (SP_STATUS) => op=16, rs=0, rt=1, rd=4
        write_imem(&mut bus, 0, 0x4001_2000);
        step_instruction(&mut bus);
        // SP_STATUS should have halt=0, broke=0 => 0
        assert_eq!(bus.rsp_scalar_regs[1], 0);
    }

    #[test]
    fn cop0_mtc0_writes_sp_mem_addr() {
        let mut bus = fresh_bus();
        bus.rsp_scalar_regs[1] = 0x0400_0100;
        // MTC0 r1, $0 (SP_MEM_ADDR) => op=16, rs=4, rt=1, rd=0
        write_imem(&mut bus, 0, 0x4081_0000);
        step_instruction(&mut bus);
        assert_eq!(
            bus.sp_regs.read(crate::rcp::SP_REGS_BASE + crate::rcp::SP_REG_MEM_ADDR),
            0x0400_0100
        );
    }

    #[test]
    fn sp_signal_flags_read_write() {
        // Per n64brew SP_STATUS: read-side signal flags SIG0..SIG7 occupy bits
        // 7..14 (NOT 5..12). Write-side pairs are clr_sigN @ (9 + 2*N) and
        // set_sigN @ (10 + 2*N). libultra masks the read with 0x300 (bits 8+9
        // = SIG1+SIG2 = YIELDED+TASKDONE) to detect task completion, so the
        // bit positions are load-bearing — getting them wrong routes audio
        // task done IRQs to OS_EVENT_CPU_BREAK.
        let mut bus = fresh_bus();
        // Set signal 0: write bit 10 (set_sig0). Read-side bit is 7.
        bus.sp_status_write_internal(1 << 10);
        let status = bus.sp_status_read_internal();
        assert_ne!(status & (1 << 7), 0, "signal 0 should be set (bit 7 of read)");

        // Clear signal 0: write bit 9 (clr_sig0)
        bus.sp_status_write_internal(1 << 9);
        let status = bus.sp_status_read_internal();
        assert_eq!(status & (1 << 7), 0, "signal 0 should be cleared");

        // Set signal 2 (TASKDONE): write bit 14 (set_sig2). Read-side bit is 9.
        bus.sp_status_write_internal(1 << 14);
        let status = bus.sp_status_read_internal();
        assert_ne!(status & (1 << 9), 0, "signal 2 should be set (bit 9 of read)");
        assert_ne!(status & 0x300, 0, "libultra mask 0x300 must see SIG2");
    }
}
