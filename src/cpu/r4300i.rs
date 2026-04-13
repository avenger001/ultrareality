use super::cop0::{
    Cop0, EXCCODE_ADEL, EXCCODE_ADES, EXCCODE_BP, EXCCODE_SYSCALL,
};

/// Global gate for matrix-watch logging (enabled from `main.rs` frame loop
/// when approaching the SM64 Goddard panic frame).
pub static MATRIX_WATCH_ARMED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
use super::tlb::MapFault;
use super::cop1::{
    cond_f32, cond_f64, f32_to_i32_ceil, f32_to_i32_floor, f32_to_i32_rm, f32_to_i32_trunc,
    f32_to_i64_rm, f64_to_i32_ceil, f64_to_i32_floor, f64_to_i32_rm, f64_to_i32_trunc,
    f64_to_i64_rm, fcsr_rm, Cop1,
};
use super::cache::{CacheOp, DCache, ICache};
use super::scoreboard::{fp_latency, Scoreboard, LOAD_USE_STALL};
use crate::bus::Bus;
use crate::cycles;

/// Map `Result<T, ()>` from address-translation helpers into an early return.
/// The exception has already been delivered (PC → vector, flag set), so we
/// just return the exception pipeline-flush cost.
macro_rules! try_mem {
    ($e:expr) => {
        match $e {
            Ok(v) => v,
            Err(()) => return Ok(cycles::EXCEPTION),
        }
    };
}

#[derive(Clone, Debug)]
pub enum CpuHalt {
    UnimplementedOpcode { pc: u64, word: u32 },
}

/// MIPS III R4300i — interpreter with per-instruction cycle accounting.
///
/// Branch and jump instructions execute the architectural delay slot in the
/// same `step` call (two retired instructions worth of work when a branch is
/// taken), and cycle counts include both the branch/jump and the delay slot.
///
/// **References:** N64 CPU / VR4300 overview — [en64: N64 CPU](https://en64.shoutwiki.com/wiki/N64_CPU);
/// opcode tables / decoding — [en64: Opcodes](https://en64.shoutwiki.com/wiki/Opcodes) (see also repo `docs/n64-opcodes-reference.md`);
/// architecture and chip notes — [n64brew: VR4300](https://n64brew.dev/wiki/VR4300).
pub struct R4300i {
    pub regs: [u64; 32],
    pub hi: u64,
    pub lo: u64,
    pub pc: u64,
    pub cop0: Cop0,
    pub cop1: Cop1,
    pub ll_bit: bool,
    pub ll_addr: u64,
    /// COP2 (RSP) GPR file stub — real CP2 is the RSP scalar/vector unit; enough for `LWC2`/`SWC2` bring-up.
    pub cop2: [u64; 32],
    /// When executing a branch/jump **delay slot**, PC of the branch/jump (for `EPC` + `Cause.BD`).
    delay_slot_branch_pc: Option<u64>,
    /// MDU issue–use cycles remaining before `MFHI`/`MFLO` may read a fresh `MULT`/`DIV` result.
    mdu_issue_remain: u64,
    /// Instruction cache (16 KiB, 2-way, 32-byte lines).
    pub icache: ICache,
    /// Data cache (8 KiB, 2-way, 16-byte lines).
    pub dcache: DCache,
    /// Accumulated I-cache miss penalty for current instruction.
    icache_stall: u64,
    /// Accumulated D-cache miss/writeback penalty for current instruction.
    dcache_stall: u64,
    /// Pipeline scoreboard for register hazard detection.
    pub scoreboard: Scoreboard,
    /// Set by exception delivery; checked by callers instead of comparing
    /// a cycle-count sentinel (avoids collisions with real instruction costs).
    exception_taken: bool,
}

impl R4300i {
    pub fn new() -> Self {
        Self {
            regs: [0u64; 32],
            hi: 0,
            lo: 0,
            // Typical IPL entry in kseg1 bootstrap (PI ROM copies to RDRAM first).
            pc: 0xA400_0040,
            cop0: Cop0::new(),
            cop1: Cop1::new(),
            ll_bit: false,
            ll_addr: 0,
            cop2: [0u64; 32],
            delay_slot_branch_pc: None,
            mdu_issue_remain: 0,
            icache: ICache::new(),
            dcache: DCache::new(),
            icache_stall: 0,
            dcache_stall: 0,
            scoreboard: Scoreboard::new(),
            exception_taken: false,
        }
    }

    pub fn reset(&mut self, entry_pc: u64) {
        self.regs = [0u64; 32];
        self.hi = 0;
        self.lo = 0;
        self.pc = entry_pc;
        self.cop0 = Cop0::new();
        self.cop1.reset();
        self.ll_bit = false;
        self.ll_addr = 0;
        self.cop2 = [0u64; 32];
        self.delay_slot_branch_pc = None;
        self.mdu_issue_remain = 0;
        self.icache.invalidate_all();
        self.dcache.invalidate_all();
        self.icache_stall = 0;
        self.dcache_stall = 0;
        self.scoreboard.reset();
        self.exception_taken = false;
    }

    /// Check if a virtual address is in a cacheable region.
    /// kseg0 (0x8000_0000–0x9FFF_FFFF) is cached.
    /// kseg1 (0xA000_0000–0xBFFF_FFFF) is uncached.
    /// kuseg and kseg2/3 depend on TLB (TODO: check C bits).
    #[inline]
    fn is_cacheable(vaddr: u64) -> bool {
        let v = vaddr as u32;
        // kseg0: cached
        (0x8000_0000..=0x9FFF_FFFF).contains(&v)
    }

    /// Compute memory access cycles with Rambus packet model timing.
    #[inline]
    fn mem_access_cycles(bus: &mut impl Bus, paddr: u32, access_bytes: u32) -> u64 {
        cycles::MEM_ACCESS_BASE.saturating_add(bus.rdram_access_cycles(paddr, access_bytes))
    }

    /// Compute cycles for a memory access at virtual address.
    #[inline]
    fn cycles_for_mem_vaddr(
        &self,
        bus: &mut impl Bus,
        vaddr: u64,
        access_bytes: u32,
        write: bool,
    ) -> Result<u64, ()> {
        let paddr = match self.cop0.translate_virt(vaddr, write) {
            Ok(p) => p,
            Err(_) => return Err(()),
        };
        Ok(Self::mem_access_cycles(bus, paddr, access_bytes))
    }

    #[inline]
    fn retire_mdu_cycles(&mut self, retired: u64) {
        self.mdu_issue_remain = self.mdu_issue_remain.saturating_sub(retired);
    }

    #[inline]
    fn deliver_general_exception(&mut self, current_pc: u64, exccode: u32) -> u64 {
        // Must get vector BEFORE enter_general_exception sets EXL, for correct TLB refill handling
        let v = self.cop0.exception_vector(exccode);
        let (epc, bd) = match self.delay_slot_branch_pc.take() {
            Some(branch_pc) => (branch_pc, true),
            None => (current_pc, false),
        };
        // DIAG: count general exceptions by exccode
        crate::cpu::cop0::GEN_EXC_COUNT[(exccode & 0x1F) as usize].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.cop0.enter_general_exception(epc, exccode, bd);
        self.exception_taken = true;
        v
    }

    /// Run `exec_non_branch` as a branch/jump delay slot; clears `delay_slot_branch_pc` on bus errors.
    #[inline]
    fn exec_non_branch_delay_slot(
        &mut self,
        branch_pc: u64,
        delay_pc: u64,
        word: u32,
        bus: &mut impl Bus,
    ) -> Result<u64, CpuHalt> {
        self.delay_slot_branch_pc = Some(branch_pc);
        let r = self.exec_non_branch(delay_pc, word, bus);
        if r.is_err() {
            self.delay_slot_branch_pc = None;
        }
        r
    }

    #[inline]
    fn gpr(&self, i: usize) -> u64 {
        if i == 0 {
            0
        } else {
            self.regs[i]
        }
    }

    #[inline]
    fn set_gpr(&mut self, i: usize, v: u64) {
        if i != 0 {
            self.regs[i] = v;
        }
    }

    #[inline]
    fn deliver_map_fault(
        &mut self,
        inst_pc: u64,
        vaddr: u64,
        fault: MapFault,
        instr_fetch: bool,
    ) {
        self.cop0.set_tlb_fault_regs(vaddr);
        let exc = Cop0::exccode_for_map_fault(fault, instr_fetch);
        self.pc = self.deliver_general_exception(inst_pc, exc);
    }

    #[inline]
    fn deliver_bus_fault(
        &mut self,
        inst_pc: u64,
        vaddr: u64,
        store: bool,
        instr_fetch: bool,
    ) {
        self.cop0.set_tlb_fault_regs(vaddr);
        let exc = Cop0::exccode_for_map_fault(MapFault::BusError { store }, instr_fetch);
        self.pc = self.deliver_general_exception(inst_pc, exc);
    }

    fn fetch32(
        &mut self,
        bus: &mut impl Bus,
        inst_pc: u64,
        vaddr: u64,
    ) -> Result<u32, ()> {
        if vaddr & 3 != 0 {
            self.cop0.badvaddr = vaddr;
            self.pc = self.deliver_general_exception(inst_pc, EXCCODE_ADEL);
            return Err(());
        }
        let paddr = match self.cop0.translate_virt(vaddr, false) {
            Ok(p) => p,
            Err(f) => {
                self.deliver_map_fault(inst_pc, vaddr, f, true);
                return Err(());
            }
        };

        // I-cache lookup for cacheable regions
        if Self::is_cacheable(vaddr) {
            let (hit, _, _) = self.icache.probe(paddr);
            if hit {
                // I-cache hit: no extra stall (pipeline hides 1-cycle access)
            } else {
                // I-cache miss: fill line and charge miss penalty
                self.icache.fill(paddr);
                self.icache_stall = self
                    .icache_stall
                    .saturating_add(cycles::ICACHE_MISS_FILL);
            }
        }

        match bus.read_u32(paddr) {
            Some(w) => Ok(w),
            None => {
                self.deliver_bus_fault(inst_pc, vaddr, false, true);
                Err(())
            }
        }
    }

    fn load32(&mut self, bus: &mut impl Bus, inst_pc: u64, vaddr: u64) -> Result<u32, ()> {
        if vaddr & 3 != 0 {
            self.cop0.badvaddr = vaddr;
            self.pc = self.deliver_general_exception(inst_pc, EXCCODE_ADEL);
            return Err(());
        }
        let paddr = match self.cop0.translate_virt(vaddr, false) {
            Ok(p) => p,
            Err(f) => {
                self.deliver_map_fault(inst_pc, vaddr, f, false);
                return Err(());
            }
        };

        // D-cache lookup for cacheable regions
        if Self::is_cacheable(vaddr) {
            let result = self.dcache.probe_read(paddr);
            if result.hit {
                // D-cache hit: 1 cycle (already accounted in base MEM_ACCESS)
            } else {
                // D-cache miss: handle writeback if needed, then fill
                if result.needs_writeback {
                    self.dcache_stall = self
                        .dcache_stall
                        .saturating_add(cycles::DCACHE_WRITEBACK);
                }
                self.dcache.fill(paddr, false);
                self.dcache_stall = self
                    .dcache_stall
                    .saturating_add(cycles::DCACHE_MISS_FILL);
            }
        }

        match bus.read_u32(paddr) {
            Some(w) => Ok(w),
            None => {
                self.deliver_bus_fault(inst_pc, vaddr, false, false);
                Err(())
            }
        }
    }

    fn store32(
        &mut self,
        bus: &mut impl Bus,
        inst_pc: u64,
        vaddr: u64,
        value: u32,
    ) -> Result<(), ()> {
        if vaddr & 3 != 0 {
            self.cop0.badvaddr = vaddr;
            self.pc = self.deliver_general_exception(inst_pc, EXCCODE_ADES);
            return Err(());
        }
        let paddr = match self.cop0.translate_virt(vaddr, true) {
            Ok(p) => p,
            Err(f) => {
                self.deliver_map_fault(inst_pc, vaddr, f, false);
                return Err(());
            }
        };

        // D-cache lookup for cacheable regions (write-back policy)
        if Self::is_cacheable(vaddr) {
            let result = self.dcache.probe_write(paddr);
            if result.hit {
                // D-cache hit: mark dirty (already done in probe_write)
            } else {
                // D-cache miss: handle writeback if needed, then allocate
                if result.needs_writeback {
                    self.dcache_stall = self
                        .dcache_stall
                        .saturating_add(cycles::DCACHE_WRITEBACK);
                }
                // Write-allocate: fill the line, then mark dirty
                self.dcache.fill(paddr, true);
                self.dcache_stall = self
                    .dcache_stall
                    .saturating_add(cycles::DCACHE_MISS_FILL);
            }
        }

        bus.write_u32(paddr, value);
        // Matrix write-watch for SM64 Goddard debug. Watches:
        //   dst   0x000B77F8..0x000B7838 (GdObj local transform)
        //   srcA  0x000B7878..0x000B78B8 (earlier-frame src)
        //   srcB  0x00206B00..0x00206B40 (panic-frame src)
        if MATRIX_WATCH_ARMED.load(std::sync::atomic::Ordering::Relaxed) {
            let in_dst = paddr >= 0x000B_77F8 && paddr < 0x000B_7838;
            let in_srca = paddr >= 0x000B_7878 && paddr < 0x000B_78B8;
            let in_srcb = paddr >= 0x0020_6B78 && paddr < 0x0020_6BB8;
            // Log any store into the watched ranges, regardless of PC.
            let pc32 = inst_pc as u32;
            if in_dst || in_srca || in_srcb {
                let label = if in_dst { "DST" } else if in_srca { "SRA" } else { "SRB" };
                eprintln!(
                    "[MW32_{}] PC=0x{:08X} paddr=0x{:08X} val=0x{:08X} ({:+e})",
                    label,
                    pc32,
                    paddr,
                    value,
                    f32::from_bits(value),
                );
            }
        }
        Ok(())
    }

    fn load64(&mut self, bus: &mut impl Bus, inst_pc: u64, vaddr: u64) -> Result<u64, ()> {
        if vaddr & 7 != 0 {
            self.cop0.badvaddr = vaddr;
            self.pc = self.deliver_general_exception(inst_pc, EXCCODE_ADEL);
            return Err(());
        }
        let paddr = match self.cop0.translate_virt(vaddr, false) {
            Ok(p) => p,
            Err(f) => {
                self.deliver_map_fault(inst_pc, vaddr, f, false);
                return Err(());
            }
        };

        // D-cache lookup for cacheable regions
        if Self::is_cacheable(vaddr) {
            let result = self.dcache.probe_read(paddr);
            if !result.hit {
                if result.needs_writeback {
                    self.dcache_stall = self
                        .dcache_stall
                        .saturating_add(cycles::DCACHE_WRITEBACK);
                }
                self.dcache.fill(paddr, false);
                self.dcache_stall = self
                    .dcache_stall
                    .saturating_add(cycles::DCACHE_MISS_FILL);
            }
        }

        let hi = match bus.read_u32(paddr) {
            Some(w) => w,
            None => {
                self.deliver_bus_fault(inst_pc, vaddr, false, false);
                return Err(());
            }
        };
        let lo = match bus.read_u32(paddr.wrapping_add(4)) {
            Some(w) => w,
            None => {
                self.deliver_bus_fault(inst_pc, vaddr.wrapping_add(4), false, false);
                return Err(());
            }
        };
        Ok((u64::from(hi) << 32) | u64::from(lo))
    }

    fn store64(&mut self, bus: &mut impl Bus, inst_pc: u64, vaddr: u64, value: u64) -> Result<(), ()> {
        if vaddr & 7 != 0 {
            self.cop0.badvaddr = vaddr;
            self.pc = self.deliver_general_exception(inst_pc, EXCCODE_ADES);
            return Err(());
        }
        let paddr = match self.cop0.translate_virt(vaddr, true) {
            Ok(p) => p,
            Err(f) => {
                self.deliver_map_fault(inst_pc, vaddr, f, false);
                return Err(());
            }
        };

        // D-cache lookup for cacheable regions
        if Self::is_cacheable(vaddr) {
            let result = self.dcache.probe_write(paddr);
            if !result.hit {
                if result.needs_writeback {
                    self.dcache_stall = self
                        .dcache_stall
                        .saturating_add(cycles::DCACHE_WRITEBACK);
                }
                self.dcache.fill(paddr, true);
                self.dcache_stall = self
                    .dcache_stall
                    .saturating_add(cycles::DCACHE_MISS_FILL);
            }
        }

        bus.write_u32(paddr, (value >> 32) as u32);
        bus.write_u32(paddr.wrapping_add(4), value as u32);
        if MATRIX_WATCH_ARMED.load(std::sync::atomic::Ordering::Relaxed) {
            let in_dst = paddr >= 0x000B_77F8 && paddr < 0x000B_7838;
            let in_srca = paddr >= 0x000B_7878 && paddr < 0x000B_78B8;
            let in_srcb = paddr >= 0x0020_6B78 && paddr < 0x0020_6BB8;
            if in_dst || in_srca || in_srcb {
                let label = if in_dst { "DST" } else if in_srca { "SRA" } else { "SRB" };
                let hi = (value >> 32) as u32;
                let lo = value as u32;
                eprintln!(
                    "[MW64_{}] PC=0x{:08X} paddr=0x{:08X} hi=0x{:08X} lo=0x{:08X} (f32 hi={:+e} lo={:+e})",
                    label,
                    inst_pc as u32,
                    paddr,
                    hi,
                    lo,
                    f32::from_bits(hi),
                    f32::from_bits(lo),
                );
            }
        }
        Ok(())
    }

    /// `LWL` — merge loaded high bytes into `rt` (32-bit result sign-extended to GPR).
    fn merge_lwl(&self, cur_rt: u64, mem_word: u32, eff: u64) -> u32 {
        let sh = ((eff & 3) * 8) as u32;
        let cur = cur_rt as u32;
        (mem_word << sh) | (cur & ((1u32 << sh).wrapping_sub(1)))
    }

    /// `LWR` — merge loaded low bytes into `rt`.
    fn merge_lwr(&self, cur_rt: u64, mem_word: u32, eff: u64) -> u32 {
        let o = (eff & 3) as u32;
        let cur = cur_rt as u32;
        match o {
            0 => (cur & 0xFFFF_FF00u32) | (mem_word >> 24),
            1 => (cur & 0xFFFF_0000u32) | (mem_word >> 16),
            2 => (cur & 0xFF00_0000u32) | (mem_word >> 8),
            3 => mem_word,
            _ => unreachable!(),
        }
    }

    /// Store word left: high-order bytes of rt from eff through end of aligned word.
    fn store_swl(&mut self, bus: &mut impl Bus, inst_pc: u64, eff: u64, val: u64) -> Result<(), ()> {
        let b = (val as u32).to_be_bytes();
        let al = eff & !3;
        let o = (eff & 3) as usize;
        for i in o..4 {
            let va = al.wrapping_add(i as u64);
            let paddr = match self.cop0.translate_virt(va, true) {
                Ok(p) => p,
                Err(f) => {
                    self.deliver_map_fault(inst_pc, va, f, false);
                    return Err(());
                }
            };
            bus.write_u8(paddr, b[i]);
        }
        Ok(())
    }

    /// Store word right: low-order bytes of rt from aligned address through eff.
    fn store_swr(&mut self, bus: &mut impl Bus, inst_pc: u64, eff: u64, val: u64) -> Result<(), ()> {
        let al = eff & !3;
        let o = (eff & 3) as usize;
        if o == 0 {
            return self.store32(bus, inst_pc, eff, val as u32);
        }
        let b = (val as u32).to_be_bytes();
        for j in 0..=o {
            let va = al.wrapping_add(j as u64);
            let paddr = match self.cop0.translate_virt(va, true) {
                Ok(p) => p,
                Err(f) => {
                    self.deliver_map_fault(inst_pc, va, f, false);
                    return Err(());
                }
            };
            bus.write_u8(paddr, b[3 - o + j]);
        }
        Ok(())
    }

    fn load16_signed(&mut self, bus: &mut impl Bus, inst_pc: u64, vaddr: u64) -> Result<u32, ()> {
        if vaddr & 1 != 0 {
            self.cop0.badvaddr = vaddr;
            self.pc = self.deliver_general_exception(inst_pc, EXCCODE_ADEL);
            return Err(());
        }
        let paddr = match self.cop0.translate_virt(vaddr, false) {
            Ok(p) => p,
            Err(f) => {
                self.deliver_map_fault(inst_pc, vaddr, f, false);
                return Err(());
            }
        };
        let hi = match bus.read_u8(paddr) {
            Some(b) => b,
            None => {
                self.deliver_bus_fault(inst_pc, vaddr, false, false);
                return Err(());
            }
        };
        let lo = match bus.read_u8(paddr.wrapping_add(1)) {
            Some(b) => b,
            None => {
                self.deliver_bus_fault(inst_pc, vaddr.wrapping_add(1), false, false);
                return Err(());
            }
        };
        let h = u16::from_be_bytes([hi, lo]);
        Ok(i32::from(h as i16) as u32)
    }

    fn load16_unsigned(&mut self, bus: &mut impl Bus, inst_pc: u64, vaddr: u64) -> Result<u32, ()> {
        if vaddr & 1 != 0 {
            self.cop0.badvaddr = vaddr;
            self.pc = self.deliver_general_exception(inst_pc, EXCCODE_ADEL);
            return Err(());
        }
        let paddr = match self.cop0.translate_virt(vaddr, false) {
            Ok(p) => p,
            Err(f) => {
                self.deliver_map_fault(inst_pc, vaddr, f, false);
                return Err(());
            }
        };
        let hi = match bus.read_u8(paddr) {
            Some(b) => b,
            None => {
                self.deliver_bus_fault(inst_pc, vaddr, false, false);
                return Err(());
            }
        };
        let lo = match bus.read_u8(paddr.wrapping_add(1)) {
            Some(b) => b,
            None => {
                self.deliver_bus_fault(inst_pc, vaddr.wrapping_add(1), false, false);
                return Err(());
            }
        };
        Ok(u32::from(u16::from_be_bytes([hi, lo])))
    }

    fn load8_signed(&mut self, bus: &mut impl Bus, inst_pc: u64, vaddr: u64) -> Result<u32, ()> {
        let paddr = match self.cop0.translate_virt(vaddr, false) {
            Ok(p) => p,
            Err(f) => {
                self.deliver_map_fault(inst_pc, vaddr, f, false);
                return Err(());
            }
        };
        let b = match bus.read_u8(paddr) {
            Some(b) => b,
            None => {
                self.deliver_bus_fault(inst_pc, vaddr, false, false);
                return Err(());
            }
        };
        Ok(i32::from(b as i8) as u32)
    }

    fn load8_unsigned(&mut self, bus: &mut impl Bus, inst_pc: u64, vaddr: u64) -> Result<u32, ()> {
        let paddr = match self.cop0.translate_virt(vaddr, false) {
            Ok(p) => p,
            Err(f) => {
                self.deliver_map_fault(inst_pc, vaddr, f, false);
                return Err(());
            }
        };
        let b = match bus.read_u8(paddr) {
            Some(b) => b,
            None => {
                self.deliver_bus_fault(inst_pc, vaddr, false, false);
                return Err(());
            }
        };
        Ok(u32::from(b))
    }

    fn store16(&mut self, bus: &mut impl Bus, inst_pc: u64, vaddr: u64, value: u32) -> Result<(), ()> {
        if vaddr & 1 != 0 {
            self.cop0.badvaddr = vaddr;
            self.pc = self.deliver_general_exception(inst_pc, EXCCODE_ADES);
            return Err(());
        }
        let paddr = match self.cop0.translate_virt(vaddr, true) {
            Ok(p) => p,
            Err(f) => {
                self.deliver_map_fault(inst_pc, vaddr, f, false);
                return Err(());
            }
        };
        let half = (value & 0xFFFF) as u16;
        let [a, b] = half.to_be_bytes();
        bus.write_u8(paddr, a);
        bus.write_u8(paddr.wrapping_add(1), b);
        Ok(())
    }

    fn store8(&mut self, bus: &mut impl Bus, inst_pc: u64, vaddr: u64, value: u32) -> Result<(), ()> {
        let paddr = match self.cop0.translate_virt(vaddr, true) {
            Ok(p) => p,
            Err(f) => {
                self.deliver_map_fault(inst_pc, vaddr, f, false);
                return Err(());
            }
        };
        bus.write_u8(paddr, (value & 0xFF) as u8);
        Ok(())
    }

    /// Execute one **architectural** instruction (including its delay slot for
    /// branches/jumps). Returns CPU cycles consumed for this retirement.
    ///
    /// `rcp_interrupt`: MI-driven external interrupt line (pending and mask). If
    /// [`crate::cpu::cop0::Cop0::interrupts_enabled`], takes exception before fetch.
    ///
    /// **Compare / timer:** if `Cause.IP7` is set (Count hit `Compare`) and `Status.IM7` is set,
    /// delivers the same interrupt vector as RCP (ExcCode 0); `Cause.IP7` is cleared on take.
    pub fn step(&mut self, bus: &mut impl Bus, rcp_interrupt: bool) -> Result<u64, CpuHalt> {
        // Reset per-instruction stall accumulators
        self.icache_stall = 0;
        self.dcache_stall = 0;
        self.exception_taken = false;

        let r = self.step_inner(bus, rcp_interrupt)?;
        self.retire_mdu_cycles(r);

        // Include cache miss penalties in total cycle count
        let total = r
            .saturating_add(self.icache_stall)
            .saturating_add(self.dcache_stall);

        // Advance scoreboard by cycles consumed
        self.scoreboard.advance(total);

        Ok(total)
    }

    fn step_inner(&mut self, bus: &mut impl Bus, rcp_interrupt: bool) -> Result<u64, CpuHalt> {
        // Update Cause.IP2 to reflect external RCP interrupt line state
        self.cop0.set_external_interrupt_pending(rcp_interrupt);

        // Check if any interrupt should be taken (IE=1, EXL=0, ERL=0, and IP&IM != 0)
        if self.cop0.interrupts_enabled() && self.cop0.any_interrupt_pending_masked() {
            let epc = self.pc;
            let v = self.cop0.interrupt_vector();

            // DIAGNOSTIC: count how many times CPU enters the interrupt vector
            #[cfg(feature = "boot_diag")]
            {
                static INT_TAKEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                let n = INT_TAKEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if n < 30 {
                    eprintln!("[INT] #{} taken at PC=0x{:08X}, EPC=0x{:08X}, vector=0x{:08X}, cause=0x{:08X}",
                        n, self.pc as u32, epc as u32, v as u32, self.cop0.cause);
                }
            }
            crate::cpu::cop0::INT_TAKEN_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            // Per-bit IP histogram (each bit set during this interrupt entry)
            {
                let ip = (self.cop0.cause >> 8) & 0xFF;
                for b in 0..8u32 {
                    if (ip >> b) & 1 != 0 {
                        crate::cpu::cop0::INT_TAKEN_IP_HISTOGRAM[b as usize]
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                }
            }

            // Note: Cause.IP7 is level-sensitive on VR4300 — it is held high by the
            // Count==Compare comparator and only cleared when software writes Compare.
            // Do NOT clear it on interrupt entry, otherwise the kernel ISR reads
            // Cause.IP7=0 and never dispatches the timer handler.
            self.cop0.enter_interrupt_exception(epc);
            self.pc = v;

            return Ok(cycles::INTERRUPT);
        }


        self.cop0.advance_random();

        let pc = self.pc;


        let word = try_mem!(self.fetch32(bus, pc, pc));
        let op = word >> 26;

        match op {
            0 => self.exec_special(pc, word, bus),
            1 => self.exec_regimm(pc, word, bus),
            2 | 3 => self.exec_j_type(pc, word, bus),
            _ => self.exec_common_i_type(pc, word, bus, op),
        }
    }

    fn exec_j_type(
        &mut self,
        pc: u64,
        word: u32,
        bus: &mut impl Bus,
    ) -> Result<u64, CpuHalt> {
        let op = word >> 26;
        let target = (pc & 0xFFFF_FFFF_F000_0000) | u64::from(word & 0x03FF_FFFF) << 2;
        let mut cycles = cycles::BRANCH;

        if op == 3 {
            // JAL
            self.set_gpr(31, pc.wrapping_add(8));
        }

        let delay_pc = pc.wrapping_add(4);
        self.delay_slot_branch_pc = Some(pc);
        let delay_word = try_mem!(self.fetch32(bus, pc, delay_pc));

        let d = self.exec_non_branch_delay_slot(pc, delay_pc, delay_word, bus)?;

        cycles += d;
        if self.exception_taken {
            return Ok(cycles);
        }
        self.delay_slot_branch_pc = None;
        self.pc = target;
        Ok(cycles)
    }

    fn exec_regimm(
        &mut self,
        pc: u64,
        word: u32,
        bus: &mut impl Bus,
    ) -> Result<u64, CpuHalt> {
        let rt_field = ((word >> 16) & 0x1F) as usize;
        let rs = ((word >> 21) & 0x1F) as usize;
        let imm = (word & 0xFFFF) as i16 as i64;

        // Stall on rs for condition check (not for BC0F/BC0T which don't use rs)
        let stall = match rt_field {
            0x08 | 0x09 => 0,
            _ => self.scoreboard.gpr_stall(rs),
        };

        let cond = match rt_field {
            0x00 | 0x02 | 0x10 | 0x12 => (self.gpr(rs) as i64) < 0,
            0x01 | 0x03 | 0x11 | 0x13 => (self.gpr(rs) as i64) >= 0,
            // `BC0F` / `BC0T` — branch on COP0 condition; no COP0 cond model yet (never take).
            0x08 | 0x09 => false,
            _ => return Err(CpuHalt::UnimplementedOpcode { pc, word }),
        };

        let likely = matches!(rt_field, 0x02 | 0x03 | 0x12 | 0x13);
        let link = matches!(rt_field, 0x10 | 0x11 | 0x12 | 0x13);

        if link {
            self.set_gpr(31, pc.wrapping_add(8));
        }

        if likely && !cond {
            self.pc = pc.wrapping_add(8);
            return Ok(stall.saturating_add(cycles::BRANCH));
        }

        let delay_pc = pc.wrapping_add(4);
        self.delay_slot_branch_pc = Some(pc);
        let delay_word = try_mem!(self.fetch32(bus, pc, delay_pc));
        let mut cycles = stall.saturating_add(cycles::BRANCH);
        let d = self.exec_non_branch_delay_slot(pc, delay_pc, delay_word, bus)?;
        cycles += d;
        if self.exception_taken {
            return Ok(cycles);
        }
        self.delay_slot_branch_pc = None;

        self.pc = if cond {
            pc.wrapping_add((imm << 2) as u64).wrapping_add(4)
        } else {
            pc.wrapping_add(8)
        };
        if cond {
            cycles += cycles::BRANCH_TAKEN_EXTRA;
        }
        Ok(cycles)
    }

    fn exec_special(
        &mut self,
        pc: u64,
        word: u32,
        bus: &mut impl Bus,
    ) -> Result<u64, CpuHalt> {
        let funct = word & 0x3F;
        let _sa = ((word >> 6) & 0x1F) as u32;
        let rd = ((word >> 11) & 0x1F) as usize;
        let _rt = ((word >> 16) & 0x1F) as usize;
        let rs = ((word >> 21) & 0x1F) as usize;

        match funct {
            0x08 => {
                // JR: stall on target register rs
                let stall = self.scoreboard.gpr_stall(rs);
                let target = self.gpr(rs);

                let mut cycles = stall.saturating_add(cycles::BRANCH);
                let delay_pc = pc.wrapping_add(4);
                self.delay_slot_branch_pc = Some(pc);
                let delay_word = try_mem!(self.fetch32(bus, pc, delay_pc));
                let d = self.exec_non_branch_delay_slot(pc, delay_pc, delay_word, bus)?;
                cycles += d;
                if self.exception_taken {
                    return Ok(cycles);
                }
                self.delay_slot_branch_pc = None;
                self.pc = target;
                Ok(cycles)
            }
            0x09 => {
                // JALR: stall on target register rs
                let stall = self.scoreboard.gpr_stall(rs);
                let target = self.gpr(rs);
                self.set_gpr(rd, pc.wrapping_add(8));
                let mut cycles = stall.saturating_add(cycles::BRANCH);
                let delay_pc = pc.wrapping_add(4);
                self.delay_slot_branch_pc = Some(pc);
                let delay_word = try_mem!(self.fetch32(bus, pc, delay_pc));
                let d = self.exec_non_branch_delay_slot(pc, delay_pc, delay_word, bus)?;
                cycles += d;
                if self.exception_taken {
                    return Ok(cycles);
                }
                self.delay_slot_branch_pc = None;
                self.pc = target;
                Ok(cycles)
            }
            0x0C => {
                self.pc = self.deliver_general_exception(pc, EXCCODE_SYSCALL);
                Ok(cycles::EXCEPTION)
            }
            0x0D => {
                self.pc = self.deliver_general_exception(pc, EXCCODE_BP);
                Ok(cycles::EXCEPTION)
            }
            _ => {
                let c = self.exec_non_branch(pc, word, bus)?;
                if !self.exception_taken && !Self::is_eret(word) {
                    self.pc = pc.wrapping_add(4);
                }
                Ok(c)
            }
        }
    }

    fn exec_common_i_type(
        &mut self,
        pc: u64,
        word: u32,
        bus: &mut impl Bus,
        op: u32,
    ) -> Result<u64, CpuHalt> {
        match op {
            0x04 | 0x05 | 0x06 | 0x07 => self.exec_branch(pc, word, bus, op),
            0x14 | 0x15 | 0x16 | 0x17 => self.exec_branch_likely(pc, word, bus, op),
            _ => {
                // COP1 BC1F/BC1T/BC1FL/BC1TL (op=0x11, fmt/rs=0x08) are branches:
                // exec_bc1 sets self.pc itself, so we must not overwrite it here.
                let is_cop1_bc = op == 0x11 && ((word >> 21) & 0x1F) == 0x08;
                let c = self.exec_non_branch(pc, word, bus)?;
                if !self.exception_taken && !Self::is_eret(word) && !is_cop1_bc {
                    self.pc = pc.wrapping_add(4);
                }
                Ok(c)
            }
        }
    }

    /// Branch likely (`BEQL`…`BGTZL`): if not taken, delay slot is **not** executed.
    fn exec_branch_likely(
        &mut self,
        pc: u64,
        word: u32,
        bus: &mut impl Bus,
        op: u32,
    ) -> Result<u64, CpuHalt> {
        let rs = ((word >> 21) & 0x1F) as usize;
        let rt = ((word >> 16) & 0x1F) as usize;
        let imm = ((word & 0xFFFF) as i16 as i64) << 2;

        // Stall on source registers used for condition
        let stall = match op {
            0x14 | 0x15 => self.scoreboard.gpr_stall_2(rs, rt), // BEQL, BNEL
            0x16 | 0x17 => self.scoreboard.gpr_stall(rs),       // BLEZL, BGTZL
            _ => 0,
        };

        let take = match op {
            0x14 => self.gpr(rs) == self.gpr(rt),
            0x15 => self.gpr(rs) != self.gpr(rt),
            0x16 => (self.gpr(rs) as i64) <= 0,
            0x17 => (self.gpr(rs) as i64) > 0,
            _ => unreachable!(),
        };

        if !take {
            self.pc = pc.wrapping_add(8);
            return Ok(stall.saturating_add(cycles::BRANCH));
        }

        let delay_pc = pc.wrapping_add(4);
        self.delay_slot_branch_pc = Some(pc);
        let delay_word = try_mem!(self.fetch32(bus, pc, delay_pc));
        let mut cycles = stall.saturating_add(cycles::BRANCH);
        let d = self.exec_non_branch_delay_slot(pc, delay_pc, delay_word, bus)?;
        cycles += d;
        if self.exception_taken {
            return Ok(cycles);
        }
        self.delay_slot_branch_pc = None;
        self.pc = pc.wrapping_add(imm as u64).wrapping_add(4);
        cycles += cycles::BRANCH_TAKEN_EXTRA;
        Ok(cycles)
    }

    #[inline]
    fn is_eret(word: u32) -> bool {
        word >> 26 == 0x10 && (word >> 21) & 0x1F == 0x10 && word & 0x3F == 0x18
    }

    fn exec_branch(
        &mut self,
        pc: u64,
        word: u32,
        bus: &mut impl Bus,
        op: u32,
    ) -> Result<u64, CpuHalt> {
        let rs = ((word >> 21) & 0x1F) as usize;
        let rt = ((word >> 16) & 0x1F) as usize;
        let imm = ((word & 0xFFFF) as i16 as i64) << 2;

        // Stall on source registers used for condition
        let stall = match op {
            0x04 | 0x05 => self.scoreboard.gpr_stall_2(rs, rt), // BEQ, BNE
            0x06 | 0x07 => self.scoreboard.gpr_stall(rs),       // BLEZ, BGTZ
            _ => 0,
        };
        let mut cycles = stall.saturating_add(cycles::BRANCH);

        let take = match op {
            0x04 => self.gpr(rs) == self.gpr(rt),
            0x05 => self.gpr(rs) != self.gpr(rt),
            0x06 => (self.gpr(rs) as i64) <= 0,
            0x07 => (self.gpr(rs) as i64) > 0,
            _ => unreachable!(),
        };

        let delay_pc = pc.wrapping_add(4);
        self.delay_slot_branch_pc = Some(pc);
        let delay_word = try_mem!(self.fetch32(bus, pc, delay_pc));
        let d = self.exec_non_branch_delay_slot(pc, delay_pc, delay_word, bus)?;
        cycles += d;
        if self.exception_taken {
            return Ok(cycles);
        }
        self.delay_slot_branch_pc = None;

        self.pc = if take {
            pc.wrapping_add(imm as u64).wrapping_add(4)
        } else {
            pc.wrapping_add(8)
        };
        if take {
            cycles += cycles::BRANCH_TAKEN_EXTRA;
        }
        Ok(cycles)
    }

    fn exec_non_branch(
        &mut self,
        pc: u64,
        word: u32,
        bus: &mut impl Bus,
    ) -> Result<u64, CpuHalt> {
        let op = word >> 26;
        let rs = ((word >> 21) & 0x1F) as usize;
        let rt = ((word >> 16) & 0x1F) as usize;
        let rd = ((word >> 11) & 0x1F) as usize;
        let sa = ((word >> 6) & 0x1F) as u32;
        let funct = word & 0x3F;
        let imm_u = word & 0xFFFF;
        let imm_s = (imm_u as i16) as i64;

        match op {
            0 => {
                match funct {
                    0x00 => {
                        // SLL: 32-bit shift left, result sign-extended to 64 bits.
                        let stall = self.scoreboard.gpr_stall(rt);
                        let v = (self.gpr(rt) as u32) << sa;
                        self.set_gpr(rd, v as i32 as i64 as u64);
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x02 => {
                        // SRL: 32-bit logical shift right, result sign-extended to 64 bits.
                        let stall = self.scoreboard.gpr_stall(rt);
                        let v = (self.gpr(rt) as u32) >> sa;
                        self.set_gpr(rd, v as i32 as i64 as u64);
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x03 => {
                        // SRA: 32-bit arithmetic shift right, result sign-extended to 64 bits.
                        let stall = self.scoreboard.gpr_stall(rt);
                        let v = (self.gpr(rt) as i32) >> sa;
                        self.set_gpr(rd, v as i64 as u64);
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x04 => {
                        // SLLV: 32-bit shift left variable, result sign-extended to 64 bits.
                        let stall = self.scoreboard.gpr_stall_2(rs, rt);
                        let s = self.gpr(rs) & 0x1F;
                        let v = (self.gpr(rt) as u32) << s;
                        self.set_gpr(rd, v as i32 as i64 as u64);
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x06 => {
                        // SRLV: 32-bit logical shift right variable, result sign-extended to 64 bits.
                        let stall = self.scoreboard.gpr_stall_2(rs, rt);
                        let s = self.gpr(rs) & 0x1F;
                        let v = (self.gpr(rt) as u32) >> s;
                        self.set_gpr(rd, v as i32 as i64 as u64);
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x07 => {
                        // SRAV: 32-bit arithmetic shift right variable, result sign-extended to 64 bits.
                        let stall = self.scoreboard.gpr_stall_2(rs, rt);
                        let s = self.gpr(rs) & 0x1F;
                        let v = (self.gpr(rt) as i32) >> s;
                        self.set_gpr(rd, v as i64 as u64);
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x01 => {
                        // MOVF/MOVT: conditional move based on FPU condition code.
                        // tf = bit 16: 0 = MOVF (move if cc false), 1 = MOVT (move if cc true)
                        // cc = bits 20:18: condition code number (0-7)
                        let stall = self.scoreboard.gpr_stall(rs);
                        let tf = (word >> 16) & 1;
                        let cc = (word >> 18) & 7;
                        let cond = (self.cop1.fcsr >> (23 + cc)) & 1;
                        let do_move = (tf == 1 && cond != 0) || (tf == 0 && cond == 0);
                        if do_move {
                            self.set_gpr(rd, self.gpr(rs));
                        }
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x21 => {
                        // ADDU: 32-bit add, result sign-extended to 64 bits.
                        let stall = self.scoreboard.gpr_stall_2(rs, rt);
                        let v = (self.gpr(rs) as u32).wrapping_add(self.gpr(rt) as u32);
                        self.set_gpr(rd, v as i32 as i64 as u64);
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x23 => {
                        // SUBU: 32-bit subtract, result sign-extended to 64 bits.
                        let stall = self.scoreboard.gpr_stall_2(rs, rt);
                        let v = (self.gpr(rs) as u32).wrapping_sub(self.gpr(rt) as u32);
                        self.set_gpr(rd, v as i32 as i64 as u64);
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x24 => {
                        // AND: stall on rs and rt
                        let stall = self.scoreboard.gpr_stall_2(rs, rt);
                        self.set_gpr(rd, self.gpr(rs) & self.gpr(rt));
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x25 => {
                        // OR: stall on rs and rt
                        let stall = self.scoreboard.gpr_stall_2(rs, rt);
                        self.set_gpr(rd, self.gpr(rs) | self.gpr(rt));
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x26 => {
                        // XOR: stall on rs and rt
                        let stall = self.scoreboard.gpr_stall_2(rs, rt);
                        self.set_gpr(rd, self.gpr(rs) ^ self.gpr(rt));
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x27 => {
                        // NOR: stall on rs and rt
                        let stall = self.scoreboard.gpr_stall_2(rs, rt);
                        self.set_gpr(rd, !(self.gpr(rs) | self.gpr(rt)));
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x2A => {
                        // SLT: stall on rs and rt
                        let stall = self.scoreboard.gpr_stall_2(rs, rt);
                        let v = (self.gpr(rs) as i64) < (self.gpr(rt) as i64);
                        self.set_gpr(rd, if v { 1 } else { 0 });
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x2B => {
                        // SLTU: stall on rs and rt
                        let stall = self.scoreboard.gpr_stall_2(rs, rt);
                        let v = self.gpr(rs) < self.gpr(rt);
                        self.set_gpr(rd, if v { 1 } else { 0 });
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    // ADD / SUB (overflow traps not modeled — 32-bit, sign-extended like ADDU/SUBU).
                    0x20 => {
                        // ADD: 32-bit add, result sign-extended to 64 bits.
                        let stall = self.scoreboard.gpr_stall_2(rs, rt);
                        let v = (self.gpr(rs) as u32).wrapping_add(self.gpr(rt) as u32);
                        self.set_gpr(rd, v as i32 as i64 as u64);
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x22 => {
                        // SUB: 32-bit subtract, result sign-extended to 64 bits.
                        let stall = self.scoreboard.gpr_stall_2(rs, rt);
                        let v = (self.gpr(rs) as u32).wrapping_sub(self.gpr(rt) as u32);
                        self.set_gpr(rd, v as i32 as i64 as u64);
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x18 => {
                        // MULT: stall on rs and rt
                        let stall = self.scoreboard.gpr_stall_2(rs, rt);
                        let a = self.gpr(rs) as u32 as i32;
                        let b = self.gpr(rt) as u32 as i32;
                        let p = (a as i64).wrapping_mul(b as i64);
                        self.lo = ((p as u32) as i32 as i64) as u64;
                        self.hi = (((p >> 32) as u32) as i32 as i64) as u64;
                        self.mdu_issue_remain =
                            self.mdu_issue_remain.saturating_add(cycles::MULT_LATENCY);
                        Ok(stall.saturating_add(cycles::MULT_LATENCY))
                    }
                    0x19 => {
                        // MULTU: stall on rs and rt. Results sign-extended to 64 bits.
                        let stall = self.scoreboard.gpr_stall_2(rs, rt);
                        let a = self.gpr(rs) as u32 as u64;
                        let b = self.gpr(rt) as u32 as u64;
                        let p = a.wrapping_mul(b);
                        self.lo = (p as u32) as i32 as i64 as u64;
                        self.hi = ((p >> 32) as u32) as i32 as i64 as u64;
                        self.mdu_issue_remain =
                            self.mdu_issue_remain.saturating_add(cycles::MULT_LATENCY);
                        Ok(stall.saturating_add(cycles::MULT_LATENCY))
                    }
                    0x1A => {
                        // DIV: stall on rs and rt
                        let stall = self.scoreboard.gpr_stall_2(rs, rt);
                        let s = self.gpr(rs) as u32 as i32;
                        let t = self.gpr(rt) as u32 as i32;
                        if t == 0 {
                            self.lo = 0;
                            self.hi = 0;
                        } else {
                            let q = s / t;
                            let r = s % t;
                            self.lo = (q as i64) as u64;
                            self.hi = (r as i64) as u64;
                        }
                        self.mdu_issue_remain =
                            self.mdu_issue_remain.saturating_add(cycles::DIV_LATENCY);
                        Ok(stall.saturating_add(cycles::DIV_LATENCY))
                    }
                    0x1B => {
                        // DIVU: stall on rs and rt. Results sign-extended to 64 bits.
                        let stall = self.scoreboard.gpr_stall_2(rs, rt);
                        let s = self.gpr(rs) as u32;
                        let t = self.gpr(rt) as u32;
                        if t == 0 {
                            self.lo = 0;
                            self.hi = 0;
                        } else {
                            self.lo = (s / t) as i32 as i64 as u64;
                            self.hi = (s % t) as i32 as i64 as u64;
                        }
                        self.mdu_issue_remain =
                            self.mdu_issue_remain.saturating_add(cycles::DIV_LATENCY);
                        Ok(stall.saturating_add(cycles::DIV_LATENCY))
                    }
                    0x10 => {
                        // MFHI: MDU interlock handles stall
                        let stall = self.mdu_issue_remain;
                        self.mdu_issue_remain = 0;
                        self.set_gpr(rd, self.hi);
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x11 => {
                        // MTHI: stall on rs
                        let stall = self.scoreboard.gpr_stall(rs);
                        self.hi = self.gpr(rs);
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x12 => {
                        // MFLO: MDU interlock handles stall
                        let stall = self.mdu_issue_remain;
                        self.mdu_issue_remain = 0;
                        self.set_gpr(rd, self.lo);
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x13 => {
                        // MTLO: stall on rs
                        let stall = self.scoreboard.gpr_stall(rs);
                        self.lo = self.gpr(rs);
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x0F => {
                        // SYNC — memory barrier; no bus effect in this model.
                        Ok(cycles::ALU)
                    }
                    0x0C => {
                        self.pc = self.deliver_general_exception(pc, EXCCODE_SYSCALL);
                        Ok(cycles::EXCEPTION)
                    }
                    0x0D => {
                        self.pc = self.deliver_general_exception(pc, EXCCODE_BP);
                        Ok(cycles::EXCEPTION)
                    }
                    0x14 => {
                        // DSLLV: stall on rs and rt
                        let stall = self.scoreboard.gpr_stall_2(rs, rt);
                        let s = self.gpr(rs) & 0x3F;
                        self.set_gpr(rd, self.gpr(rt) << s);
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x16 => {
                        // DSRLV: stall on rs and rt
                        let stall = self.scoreboard.gpr_stall_2(rs, rt);
                        let s = self.gpr(rs) & 0x3F;
                        self.set_gpr(rd, self.gpr(rt) >> s);
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x17 => {
                        // DSRAV: stall on rs and rt
                        let stall = self.scoreboard.gpr_stall_2(rs, rt);
                        let s = self.gpr(rs) & 0x3F;
                        self.set_gpr(rd, ((self.gpr(rt) as i64) >> s) as u64);
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x1C => {
                        // DMULT: stall on rs and rt
                        let stall = self.scoreboard.gpr_stall_2(rs, rt);
                        let a = self.gpr(rs) as i128;
                        let b = self.gpr(rt) as i128;
                        let p = a.wrapping_mul(b);
                        self.lo = p as u64;
                        self.hi = (p >> 64) as u64;
                        self.mdu_issue_remain =
                            self.mdu_issue_remain.saturating_add(cycles::MULT_LATENCY);
                        Ok(stall.saturating_add(cycles::MULT_LATENCY))
                    }
                    0x1D => {
                        // DMULTU: stall on rs and rt
                        let stall = self.scoreboard.gpr_stall_2(rs, rt);
                        let a = self.gpr(rs) as u128;
                        let b = self.gpr(rt) as u128;
                        let p = a.wrapping_mul(b);
                        self.lo = p as u64;
                        self.hi = (p >> 64) as u64;
                        self.mdu_issue_remain =
                            self.mdu_issue_remain.saturating_add(cycles::MULT_LATENCY);
                        Ok(stall.saturating_add(cycles::MULT_LATENCY))
                    }
                    0x1E => {
                        // DDIV: stall on rs and rt
                        let stall = self.scoreboard.gpr_stall_2(rs, rt);
                        let s = self.gpr(rs) as i64;
                        let t = self.gpr(rt) as i64;
                        if t == 0 {
                            self.lo = 0;
                            self.hi = 0;
                        } else {
                            self.lo = (s / t) as u64;
                            self.hi = (s % t) as u64;
                        }
                        self.mdu_issue_remain =
                            self.mdu_issue_remain.saturating_add(cycles::DIV_LATENCY);
                        Ok(stall.saturating_add(cycles::DIV_LATENCY))
                    }
                    0x1F => {
                        // DDIVU: stall on rs and rt
                        let stall = self.scoreboard.gpr_stall_2(rs, rt);
                        let s = self.gpr(rs);
                        let t = self.gpr(rt);
                        if t == 0 {
                            self.lo = 0;
                            self.hi = 0;
                        } else {
                            self.lo = s / t;
                            self.hi = s % t;
                        }
                        self.mdu_issue_remain =
                            self.mdu_issue_remain.saturating_add(cycles::DIV_LATENCY);
                        Ok(stall.saturating_add(cycles::DIV_LATENCY))
                    }
                    0x2D => {
                        // DADDU: stall on rs and rt
                        let stall = self.scoreboard.gpr_stall_2(rs, rt);
                        self.set_gpr(rd, self.gpr(rs).wrapping_add(self.gpr(rt)));
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x2F => {
                        // DSUBU: stall on rs and rt
                        let stall = self.scoreboard.gpr_stall_2(rs, rt);
                        self.set_gpr(rd, self.gpr(rs).wrapping_sub(self.gpr(rt)));
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    // DADD / DSUB (overflow traps not modeled — same as DADDU/DSUBU).
                    0x2C => {
                        // DADD: stall on rs and rt
                        let stall = self.scoreboard.gpr_stall_2(rs, rt);
                        self.set_gpr(rd, self.gpr(rs).wrapping_add(self.gpr(rt)));
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x2E => {
                        // DSUB: stall on rs and rt
                        let stall = self.scoreboard.gpr_stall_2(rs, rt);
                        self.set_gpr(rd, self.gpr(rs).wrapping_sub(self.gpr(rt)));
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x38 => {
                        // DSLL: stall on rt
                        let stall = self.scoreboard.gpr_stall(rt);
                        self.set_gpr(rd, self.gpr(rt) << sa);
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x3A => {
                        // DSRL: stall on rt
                        let stall = self.scoreboard.gpr_stall(rt);
                        self.set_gpr(rd, self.gpr(rt) >> sa);
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x3B => {
                        // DSRA: stall on rt
                        let stall = self.scoreboard.gpr_stall(rt);
                        self.set_gpr(rd, ((self.gpr(rt) as i64) >> sa) as u64);
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x3C => {
                        // DSLL32: stall on rt
                        let stall = self.scoreboard.gpr_stall(rt);
                        self.set_gpr(rd, self.gpr(rt) << (sa + 32));
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x3E => {
                        // DSRL32: stall on rt
                        let stall = self.scoreboard.gpr_stall(rt);
                        self.set_gpr(rd, self.gpr(rt) >> (sa + 32));
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x3F => {
                        // DSRA32: stall on rt
                        let stall = self.scoreboard.gpr_stall(rt);
                        self.set_gpr(rd, ((self.gpr(rt) as i64) >> (sa + 32)) as u64);
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    // `TGE`…`TNE` — traps not modeled (no `SignalException`); retire as ALU.
                    0x30 | 0x31 | 0x32 | 0x33 | 0x34 | 0x35 => Ok(cycles::ALU),
                    _ => Err(CpuHalt::UnimplementedOpcode { pc, word }),
                }
            }
            0x08 | 0x09 => {
                // ADDI/ADDIU: 32-bit add, result sign-extended to 64 bits.
                let stall = self.scoreboard.gpr_stall(rs);
                let v = (self.gpr(rs) as u32).wrapping_add(imm_s as u32);
                self.set_gpr(rt, v as i32 as i64 as u64);
                Ok(stall.saturating_add(cycles::ALU))
            }
            0x18 | 0x19 => {
                // DADDI/DADDIU: 64-bit add.
                let stall = self.scoreboard.gpr_stall(rs);
                let v = self.gpr(rs).wrapping_add(imm_s as u64);
                self.set_gpr(rt, v);
                Ok(stall.saturating_add(cycles::ALU))
            }
            0x0C => {
                // ANDI: stall on rs
                let stall = self.scoreboard.gpr_stall(rs);
                self.set_gpr(rt, self.gpr(rs) & u64::from(imm_u));
                Ok(stall.saturating_add(cycles::ALU))
            }
            0x0D => {
                // ORI: stall on rs
                let stall = self.scoreboard.gpr_stall(rs);
                self.set_gpr(rt, self.gpr(rs) | u64::from(imm_u));
                Ok(stall.saturating_add(cycles::ALU))
            }
            0x0E => {
                // XORI: stall on rs
                let stall = self.scoreboard.gpr_stall(rs);
                self.set_gpr(rt, self.gpr(rs) ^ u64::from(imm_u));
                Ok(stall.saturating_add(cycles::ALU))
            }
            0x0A => {
                // SLTI: stall on rs
                let stall = self.scoreboard.gpr_stall(rs);
                let v = (self.gpr(rs) as i64) < imm_s;
                self.set_gpr(rt, if v { 1 } else { 0 });
                Ok(stall.saturating_add(cycles::ALU))
            }
            0x0B => {
                // SLTIU: unsigned compare of GPR[rs] vs sign-extended immediate.
                let stall = self.scoreboard.gpr_stall(rs);
                let v = self.gpr(rs) < (imm_s as u64);
                self.set_gpr(rt, if v { 1 } else { 0 });
                Ok(stall.saturating_add(cycles::ALU))
            }
            0x0F => {
                // LUI: place imm in upper 16 bits of 32-bit word, sign-extend to 64 bits.
                let v = (imm_u as u32) << 16;
                self.set_gpr(rt, v as i32 as i64 as u64);
                Ok(cycles::ALU)
            }
            0x33 => Ok(cycles::ALU),
            0x23 => {
                // LW: check stall on base register, set load-use latency on destination
                let stall = self.scoreboard.gpr_stall(rs);
                let addr = self.gpr(rs).wrapping_add(imm_s as u64);
                let v = try_mem!(self.load32(bus, pc, addr));
                self.set_gpr(rt, i64::from(v as i32) as u64);
                self.scoreboard.set_gpr_latency(rt, LOAD_USE_STALL);
                Ok(stall.saturating_add(try_mem!(self.cycles_for_mem_vaddr(bus, addr, 4, false))))
            }
            0x22 => {
                // LWL: stall on base and partial destination (it merges)
                let stall = self.scoreboard.gpr_stall_2(rs, rt);
                let eff = self.gpr(rs).wrapping_add(imm_s as u64);
                let al = eff & !3;
                let mem_word = try_mem!(self.load32(bus, pc, al));
                let merged = self.merge_lwl(self.gpr(rt), mem_word, eff);
                self.set_gpr(rt, i64::from(merged as i32) as u64);
                self.scoreboard.set_gpr_latency(rt, LOAD_USE_STALL);
                Ok(stall.saturating_add(try_mem!(self.cycles_for_mem_vaddr(bus, eff, 4, false))))
            }
            0x26 => {
                // LWR: stall on base and partial destination (it merges)
                let stall = self.scoreboard.gpr_stall_2(rs, rt);
                let eff = self.gpr(rs).wrapping_add(imm_s as u64);
                let al = eff & !3;
                let mem_word = try_mem!(self.load32(bus, pc, al));
                let merged = self.merge_lwr(self.gpr(rt), mem_word, eff);
                self.set_gpr(rt, i64::from(merged as i32) as u64);
                self.scoreboard.set_gpr_latency(rt, LOAD_USE_STALL);
                Ok(stall.saturating_add(try_mem!(self.cycles_for_mem_vaddr(bus, eff, 4, false))))
            }
            0x27 => {
                // LWU: load word unsigned with scoreboard
                let stall = self.scoreboard.gpr_stall(rs);
                let addr = self.gpr(rs).wrapping_add(imm_s as u64);
                let v = try_mem!(self.load32(bus, pc, addr));
                self.set_gpr(rt, u64::from(v));
                self.scoreboard.set_gpr_latency(rt, LOAD_USE_STALL);
                Ok(stall.saturating_add(try_mem!(self.cycles_for_mem_vaddr(bus, addr, 4, false))))
            }
            0x24 => {
                // LBU
                let stall = self.scoreboard.gpr_stall(rs);
                let addr = self.gpr(rs).wrapping_add(imm_s as u64);
                let v = try_mem!(self.load8_unsigned(bus, pc, addr));
                self.set_gpr(rt, u64::from(v));
                self.scoreboard.set_gpr_latency(rt, LOAD_USE_STALL);
                Ok(stall.saturating_add(try_mem!(self.cycles_for_mem_vaddr(bus, addr, 1, false))))
            }
            0x20 => {
                // LB
                let stall = self.scoreboard.gpr_stall(rs);
                let addr = self.gpr(rs).wrapping_add(imm_s as u64);
                let v = try_mem!(self.load8_signed(bus, pc, addr));
                self.set_gpr(rt, i64::from(v as i32) as u64);
                self.scoreboard.set_gpr_latency(rt, LOAD_USE_STALL);
                Ok(stall.saturating_add(try_mem!(self.cycles_for_mem_vaddr(bus, addr, 1, false))))
            }
            0x25 => {
                // LHU
                let stall = self.scoreboard.gpr_stall(rs);
                let addr = self.gpr(rs).wrapping_add(imm_s as u64);
                let v = try_mem!(self.load16_unsigned(bus, pc, addr));
                self.set_gpr(rt, u64::from(v));
                self.scoreboard.set_gpr_latency(rt, LOAD_USE_STALL);
                Ok(stall.saturating_add(try_mem!(self.cycles_for_mem_vaddr(bus, addr, 2, false))))
            }
            0x21 => {
                // LH
                let stall = self.scoreboard.gpr_stall(rs);
                let addr = self.gpr(rs).wrapping_add(imm_s as u64);
                let v = try_mem!(self.load16_signed(bus, pc, addr));
                self.set_gpr(rt, i64::from(v as i32) as u64);
                self.scoreboard.set_gpr_latency(rt, LOAD_USE_STALL);
                Ok(stall.saturating_add(try_mem!(self.cycles_for_mem_vaddr(bus, addr, 2, false))))
            }
            0x2B => {
                // SW: check stalls on both base and value registers
                let stall = self.scoreboard.gpr_stall_2(rs, rt);
                let addr = self.gpr(rs).wrapping_add(imm_s as u64);
                try_mem!(self.store32(bus, pc, addr, self.gpr(rt) as u32));
                Ok(stall.saturating_add(try_mem!(self.cycles_for_mem_vaddr(bus, addr, 4, true))))
            }
            0x28 => {
                // SB
                let stall = self.scoreboard.gpr_stall_2(rs, rt);
                let addr = self.gpr(rs).wrapping_add(imm_s as u64);
                try_mem!(self.store8(bus, pc, addr, self.gpr(rt) as u32));
                Ok(stall.saturating_add(try_mem!(self.cycles_for_mem_vaddr(bus, addr, 1, true))))
            }
            0x29 => {
                // SH
                let stall = self.scoreboard.gpr_stall_2(rs, rt);
                let addr = self.gpr(rs).wrapping_add(imm_s as u64);
                try_mem!(self.store16(bus, pc, addr, self.gpr(rt) as u32));
                Ok(stall.saturating_add(try_mem!(self.cycles_for_mem_vaddr(bus, addr, 2, true))))
            }
            0x2A => {
                // SWL
                let stall = self.scoreboard.gpr_stall_2(rs, rt);
                let eff = self.gpr(rs).wrapping_add(imm_s as u64);
                try_mem!(self.store_swl(bus, pc, eff, self.gpr(rt)));
                Ok(stall.saturating_add(try_mem!(self.cycles_for_mem_vaddr(bus, eff, 4, true))))
            }
            0x2E => {
                // SWR
                let stall = self.scoreboard.gpr_stall_2(rs, rt);
                let eff = self.gpr(rs).wrapping_add(imm_s as u64);
                try_mem!(self.store_swr(bus, pc, eff, self.gpr(rt)));
                Ok(stall.saturating_add(try_mem!(self.cycles_for_mem_vaddr(bus, eff, 4, true))))
            }
            0x30 => {
                // LL (load linked)
                let stall = self.scoreboard.gpr_stall(rs);
                let addr = self.gpr(rs).wrapping_add(imm_s as u64);
                if addr & 3 != 0 {
                    self.cop0.badvaddr = addr;
                    self.pc = self.deliver_general_exception(pc, EXCCODE_ADEL);
                    return Ok(cycles::EXCEPTION);
                }
                let v = try_mem!(self.load32(bus, pc, addr));
                self.ll_bit = true;
                self.ll_addr = addr;
                self.set_gpr(rt, i64::from(v as i32) as u64);
                self.scoreboard.set_gpr_latency(rt, LOAD_USE_STALL);
                Ok(stall.saturating_add(try_mem!(self.cycles_for_mem_vaddr(bus, addr, 4, false))))
            }
            0x38 => {
                // SC: stall on base and value registers
                let stall = self.scoreboard.gpr_stall_2(rs, rt);
                let addr = self.gpr(rs).wrapping_add(imm_s as u64);
                if addr & 3 != 0 {
                    self.cop0.badvaddr = addr;
                    self.pc = self.deliver_general_exception(pc, EXCCODE_ADEL);
                    return Ok(cycles::EXCEPTION);
                }
                let ok = self.ll_bit && self.ll_addr == addr;
                let val = self.gpr(rt) as u32;
                if ok {
                    try_mem!(self.store32(bus, pc, addr, val));
                }
                self.ll_bit = false;
                self.set_gpr(rt, if ok { 1 } else { 0 });
                Ok(stall.saturating_add(try_mem!(self.cycles_for_mem_vaddr(bus, addr, 4, true))))
            }
            0x34 => {
                // LLD: load linked double with scoreboard
                let stall = self.scoreboard.gpr_stall(rs);
                let addr = self.gpr(rs).wrapping_add(imm_s as u64);
                if addr & 7 != 0 {
                    self.cop0.badvaddr = addr;
                    self.pc = self.deliver_general_exception(pc, EXCCODE_ADEL);
                    return Ok(cycles::EXCEPTION);
                }
                let v = try_mem!(self.load64(bus, pc, addr));
                self.ll_bit = true;
                self.ll_addr = addr;
                self.set_gpr(rt, v);
                self.scoreboard.set_gpr_latency(rt, LOAD_USE_STALL);
                Ok(stall.saturating_add(try_mem!(self.cycles_for_mem_vaddr(bus, addr, 8, false))))
            }
            0x3C => {
                // SCD: store conditional double with scoreboard
                let stall = self.scoreboard.gpr_stall_2(rs, rt);
                let addr = self.gpr(rs).wrapping_add(imm_s as u64);
                if addr & 7 != 0 {
                    self.cop0.badvaddr = addr;
                    self.pc = self.deliver_general_exception(pc, EXCCODE_ADEL);
                    return Ok(cycles::EXCEPTION);
                }
                let ok = self.ll_bit && self.ll_addr == addr;
                let val = self.gpr(rt);
                if ok {
                    try_mem!(self.store64(bus, pc, addr, val));
                }
                self.ll_bit = false;
                self.set_gpr(rt, if ok { 1 } else { 0 });
                Ok(stall.saturating_add(try_mem!(self.cycles_for_mem_vaddr(bus, addr, 8, true))))
            }
            // CACHE — I-type; rt field is cache operation code.
            0x2F => {
                // CACHE: stall on base GPR
                let stall = self.scoreboard.gpr_stall(rs);
                let addr = self.gpr(rs).wrapping_add(imm_s as u64);
                let paddr = match self.cop0.translate_virt(addr, false) {
                    Ok(p) => p,
                    Err(_) => {
                        // Cache ops on unmapped addresses are often NOPs
                        return Ok(stall.saturating_add(cycles::CACHE_OP));
                    }
                };
                self.exec_cache_op(rt as u8, paddr);
                Ok(stall.saturating_add(cycles::CACHE_OP))
            }
            // MIPS III / 64-bit loads & stores (GPR)
            0x37 => {
                // LD: load double with scoreboard
                let stall = self.scoreboard.gpr_stall(rs);
                let addr = self.gpr(rs).wrapping_add(imm_s as u64);
                let v = try_mem!(self.load64(bus, pc, addr));
                self.set_gpr(rt, v);
                self.scoreboard.set_gpr_latency(rt, LOAD_USE_STALL);
                Ok(stall.saturating_add(try_mem!(self.cycles_for_mem_vaddr(bus, addr, 8, false))))
            }
            0x3F => {
                // SD: store double with scoreboard
                let stall = self.scoreboard.gpr_stall_2(rs, rt);
                let addr = self.gpr(rs).wrapping_add(imm_s as u64);
                try_mem!(self.store64(bus, pc, addr, self.gpr(rt)));
                Ok(stall.saturating_add(try_mem!(self.cycles_for_mem_vaddr(bus, addr, 8, true))))
            }
            // CP1 load / store (I-type: `rt` index is `ft`)
            0x31 => {
                // LWC1: stall on base GPR, set FPR load-use latency
                let stall = self.scoreboard.gpr_stall(rs);
                let addr = self.gpr(rs).wrapping_add(imm_s as u64);
                let v = try_mem!(self.load32(bus, pc, addr));
                self.cop1.set_fpr_u32(rt, v);
                self.scoreboard.set_fpr_latency(rt, LOAD_USE_STALL);
                Ok(stall.saturating_add(try_mem!(self.cycles_for_mem_vaddr(bus, addr, 4, false))))
            }
            0x39 => {
                // SWC1: stall on base GPR and source FPR
                let gpr_stall = self.scoreboard.gpr_stall(rs);
                let fpr_stall = self.scoreboard.fpr_stall(rt);
                let stall = gpr_stall.max(fpr_stall);
                let addr = self.gpr(rs).wrapping_add(imm_s as u64);
                try_mem!(self.store32(bus, pc, addr, self.cop1.fpr_u32(rt)));
                Ok(stall.saturating_add(try_mem!(self.cycles_for_mem_vaddr(bus, addr, 4, true))))
            }
            0x35 => {
                // LDC1: stall on base GPR, set FPR load-use latency
                let stall = self.scoreboard.gpr_stall(rs);
                let addr = self.gpr(rs).wrapping_add(imm_s as u64);
                let v = try_mem!(self.load64(bus, pc, addr));
                let fr = (self.cop0.status >> 26) & 1 != 0;
                self.cop1.set_fpr_u64(rt, v, fr);
                self.scoreboard.set_fpr_latency(rt, LOAD_USE_STALL);
                Ok(stall.saturating_add(try_mem!(self.cycles_for_mem_vaddr(bus, addr, 8, false))))
            }
            0x3D => {
                // SDC1: stall on base GPR and source FPR
                let gpr_stall = self.scoreboard.gpr_stall(rs);
                let fpr_stall = self.scoreboard.fpr_stall(rt);
                let stall = gpr_stall.max(fpr_stall);
                let addr = self.gpr(rs).wrapping_add(imm_s as u64);
                let fr = (self.cop0.status >> 26) & 1 != 0;
                try_mem!(self.store64(bus, pc, addr, self.cop1.fpr_u64(rt, fr)));
                Ok(stall.saturating_add(try_mem!(self.cycles_for_mem_vaddr(bus, addr, 8, true))))
            }
            // COP2 load / store — RSP registers stubbed as `cop2[]` until the RSP core exists.
            0x32 => {
                // LWC2: stall on base GPR
                let stall = self.scoreboard.gpr_stall(rs);
                let addr = self.gpr(rs).wrapping_add(imm_s as u64);
                let v = try_mem!(self.load32(bus, pc, addr));
                self.cop2[rt] = i64::from(v as i32) as u64;
                Ok(stall.saturating_add(try_mem!(self.cycles_for_mem_vaddr(bus, addr, 4, false))))
            }
            // `LDC2` shares primary `0x34` with `LLD` on MIPS III — only `LLD` is decoded here.
            0x3A => {
                // SWC2: stall on base GPR (COP2 regs have no scoreboard tracking yet)
                let stall = self.scoreboard.gpr_stall(rs);
                let addr = self.gpr(rs).wrapping_add(imm_s as u64);
                try_mem!(self.store32(bus, pc, addr, self.cop2[rt] as u32));
                Ok(stall.saturating_add(try_mem!(self.cycles_for_mem_vaddr(bus, addr, 4, true))))
            }
            0x3E => {
                // SDC2: stall on base GPR
                let stall = self.scoreboard.gpr_stall(rs);
                let addr = self.gpr(rs).wrapping_add(imm_s as u64);
                try_mem!(self.store64(bus, pc, addr, self.cop2[rt]));
                Ok(stall.saturating_add(try_mem!(self.cycles_for_mem_vaddr(bus, addr, 8, true))))
            }
            0x10 => {
                let sub = (word >> 21) & 0x1F;
                let funct = word & 0x3F;
                // `CO` class: TLB ops + `ERET`.
                if sub == 0x10 {
                    match funct {
                        0x01 => {
                            self.cop0.tlb_read();
                            return Ok(cycles::ALU);
                        }
                        0x02 => {
                            self.cop0.tlb_write_indexed();
                            return Ok(cycles::ALU);
                        }
                        0x06 => {
                            self.cop0.tlb_write_random();
                            return Ok(cycles::ALU);
                        }
                        0x08 => {
                            self.cop0.tlb_probe();
                            return Ok(cycles::ALU);
                        }
                        0x05 | 0x09 => {
                            return Ok(cycles::ALU);
                        }
                        0x18 => {
                            // ERET
                            self.pc = self.cop0.apply_eret();
                            self.ll_bit = false;
                            return Ok(cycles::ALU);
                        }
                        // `WAIT`, `EHB`, and other COP0 `CO` ops — no side effects modeled.
                        _ => return Ok(cycles::ALU),
                    }
                }
                let rd_cop = ((word >> 11) & 0x1F) as u32;
                match sub {
                    0x00 => {
                        let v = self.cop0.read_32(rd_cop);
                        self.set_gpr(rt, i64::from(v as i32) as u64);
                        Ok(cycles::COP_MOVE)
                    }
                    0x01 => {
                        // DMFC0
                        let v = self.cop0.read_xpr64(rd_cop);
                        self.set_gpr(rt, v);
                        Ok(cycles::COP_MOVE)
                    }
                    0x02 => {
                        // CFC0 — control path; mirror `MFC0` for bring-up.
                        let v = self.cop0.read_32(rd_cop);
                        self.set_gpr(rt, i64::from(v as i32) as u64);
                        Ok(cycles::COP_MOVE)
                    }
                    0x04 => {
                        let v = self.gpr(rt) as u32;
                        if rd_cop == 11 {
                            static MTC0_CMP_LOG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                            let n = MTC0_CMP_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if n < 10 {
                                let ra = self.gpr(31) as u32;
                                let current_count = self.cop0.read_32(9); // Count register
                                eprintln!("[MTC0_COMPARE #{}] PC=0x{:08X} RA=0x{:08X} value=0x{:08X} (Count=0x{:08X}, delta={})",
                                    n, pc as u32, ra, v, current_count,
                                    if v >= current_count { v.wrapping_sub(current_count) } else { 0 });
                            }
                        }
                        self.cop0.write_32(rd_cop, v);
                        Ok(cycles::COP_MOVE)
                    }
                    0x05 => {
                        // DMTC0
                        self.cop0.write_xpr64(rd_cop, self.gpr(rt));
                        Ok(cycles::COP_MOVE)
                    }
                    0x06 => {
                        // CTC0 — mirror `MTC0` for bring-up.
                        let v = self.gpr(rt) as u32;
                        self.cop0.write_32(rd_cop, v);
                        Ok(cycles::COP_MOVE)
                    }
                    _ => Err(CpuHalt::UnimplementedOpcode { pc, word }),
                }
            }
            0x11 => self.exec_cop1(pc, word, bus),
            // COP2 — RSP vector unit; no ucode execution yet (bring-up NOP).
            0x12 => Ok(cycles::ALU),
            // COP1X — fused FP ops (`MADD.S`, …); not modeled.
            0x13 => Ok(cycles::ALU),
            _ => Err(CpuHalt::UnimplementedOpcode { pc, word }),
        }
    }

    /// COP1: moves (`MFC1`, `CFC1`, …), `BC1F`/`BC1T`, and FP compute when bit 25 = 1 (common `.S`/`.D`).
    fn exec_cop1(
        &mut self,
        pc: u64,
        word: u32,
        bus: &mut impl Bus,
    ) -> Result<u64, CpuHalt> {
        let fmt = (word >> 21) & 0x1F;
        let rt = ((word >> 16) & 0x1F) as usize;
        let fs = ((word >> 11) & 0x1F) as usize;
        let _fd = ((word >> 6) & 0x1F) as usize;

        if fmt == 0x08 {
            return self.exec_bc1(pc, word, bus);
        }

        if (word >> 25) & 1 != 0 {
            return self.exec_cop1_fp(pc, word);
        }

        match fmt {
            0x00 => {
                // MFC1: stall on FPR fs, set GPR latency. In FR=0, MFC1 to an odd
                // index reads the high half of the pair; our FPR array stores each
                // 32-bit slot independently so `fpr_u32` returns the right bits.
                let stall = self.scoreboard.fpr_stall(fs);
                let v = self.cop1.fpr_u32(fs);
                self.set_gpr(rt, i64::from(v as i32) as u64);
                self.scoreboard.set_gpr_latency(rt, LOAD_USE_STALL);
                Ok(stall.saturating_add(cycles::COP_MOVE))
            }
            0x01 => {
                // DMFC1: stall on FPR fs, set GPR latency
                let stall = self.scoreboard.fpr_stall(fs);
                let fr = (self.cop0.status >> 26) & 1 != 0;
                self.set_gpr(rt, self.cop1.fpr_u64(fs, fr));
                self.scoreboard.set_gpr_latency(rt, LOAD_USE_STALL);
                Ok(stall.saturating_add(cycles::COP_MOVE))
            }
            0x04 => {
                // MTC1: stall on GPR rt, set FPR latency.
                // In FR=0 mode MTC1 to an odd index writes the high 32 bits of the
                // pair, which lands on the odd slot of our FPR array — either way
                // `set_fpr_u32` writes the addressed 32-bit slot.
                let stall = self.scoreboard.gpr_stall(rt);
                self.cop1.set_fpr_u32(fs, self.gpr(rt) as u32);
                self.scoreboard.set_fpr_latency(fs, LOAD_USE_STALL);
                Ok(stall.saturating_add(cycles::COP_MOVE))
            }
            0x05 => {
                // DMTC1: stall on GPR rt, set FPR latency
                let stall = self.scoreboard.gpr_stall(rt);
                let fr = (self.cop0.status >> 26) & 1 != 0;
                self.cop1.set_fpr_u64(fs, self.gpr(rt), fr);
                self.scoreboard.set_fpr_latency(fs, LOAD_USE_STALL);
                Ok(stall.saturating_add(cycles::COP_MOVE))
            }
            0x02 => {
                // CFC1: FCSR read, set GPR latency
                let v = self.cop1.read_fcr(fs);
                self.set_gpr(rt, i64::from(v as i32) as u64);
                self.scoreboard.set_gpr_latency(rt, LOAD_USE_STALL);
                Ok(cycles::COP_MOVE)
            }
            0x06 => {
                // CTC1: stall on GPR rt
                let stall = self.scoreboard.gpr_stall(rt);
                self.cop1.write_fcr(fs, self.gpr(rt) as u32);
                Ok(stall.saturating_add(cycles::COP_MOVE))
            }
            _ => Ok(cycles::ALU),
        }
    }

    /// COP1 floating-point compute (`CO` / bit 25 = 1). Covers common `.S` / `.D` ops and compares.
    fn exec_cop1_fp(&mut self, _pc: u64, word: u32) -> Result<u64, CpuHalt> {
        let fmt5 = (word >> 21) & 0x1F;
        let ft = ((word >> 16) & 0x1F) as usize;
        let fs = ((word >> 11) & 0x1F) as usize;
        let fd = ((word >> 6) & 0x1F) as usize;
        let funct = word & 0x3F;
        // Status.FR (bit 26) selects the FPR file layout: FR=1 = 32 × 64-bit regs,
        // FR=0 = 16 doubles accessed as even/odd pairs. libultra/SM64 runs FR=0.
        let fr = (self.cop0.status >> 26) & 1 != 0;

        // MIPS III: `fmt` in bits 24–21 with `CO`=1 → 0x10=.S, 0x11=.D, 0x14=.W, 0x15=.L
        match fmt5 {
            0x10 => {
                // .S (single-precision)
                match funct {
                    0x00 => {
                        // ADD.S: stall on fs/ft, latency on fd
                        let stall = self.scoreboard.fpr_stall_2(fs, ft);
                        let r = self.cop1.fpr_f32(fs) + self.cop1.fpr_f32(ft);
                        self.cop1.set_fpr_f32(fd, r);
                        self.scoreboard.set_fpr_latency(fd, fp_latency::ADD_MUL_S);
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x01 => {
                        // SUB.S
                        let stall = self.scoreboard.fpr_stall_2(fs, ft);
                        let r = self.cop1.fpr_f32(fs) - self.cop1.fpr_f32(ft);
                        self.cop1.set_fpr_f32(fd, r);
                        self.scoreboard.set_fpr_latency(fd, fp_latency::ADD_MUL_S);
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x02 => {
                        // MUL.S
                        let stall = self.scoreboard.fpr_stall_2(fs, ft);
                        let r = self.cop1.fpr_f32(fs) * self.cop1.fpr_f32(ft);
                        self.cop1.set_fpr_f32(fd, r);
                        self.scoreboard.set_fpr_latency(fd, fp_latency::ADD_MUL_S);
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x03 => {
                        // DIV.S
                        let stall = self.scoreboard.fpr_stall_2(fs, ft);
                        let a = self.cop1.fpr_f32(fs);
                        let b = self.cop1.fpr_f32(ft);
                        let r = if b == 0.0 { 0.0 } else { a / b };
                        self.cop1.set_fpr_f32(fd, r);
                        self.scoreboard.set_fpr_latency(fd, fp_latency::DIV_S);
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x04 => {
                        // SQRT.S
                        let stall = self.scoreboard.fpr_stall(fs);
                        let r = self.cop1.fpr_f32(fs).sqrt();
                        self.cop1.set_fpr_f32(fd, r);
                        self.scoreboard.set_fpr_latency(fd, fp_latency::SQRT_S);
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x05 => {
                        // ABS.S
                        let stall = self.scoreboard.fpr_stall(fs);
                        let r = self.cop1.fpr_f32(fs).abs();
                        self.cop1.set_fpr_f32(fd, r);
                        self.scoreboard.set_fpr_latency(fd, fp_latency::CVT);
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x06 => {
                        // MOV.S
                        let stall = self.scoreboard.fpr_stall(fs);
                        let v = self.cop1.fpr_f32(fs);
                        self.cop1.set_fpr_f32(fd, v);
                        self.scoreboard.set_fpr_latency(fd, fp_latency::CVT);
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x07 => {
                        // NEG.S
                        let stall = self.scoreboard.fpr_stall(fs);
                        let r = -self.cop1.fpr_f32(fs);
                        self.cop1.set_fpr_f32(fd, r);
                        self.scoreboard.set_fpr_latency(fd, fp_latency::CVT);
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x0C => {
                        // ROUND.W.S — to word using FCSR RM
                        let stall = self.scoreboard.fpr_stall(fs);
                        let rm = fcsr_rm(self.cop1.fcsr);
                        let i = f32_to_i32_rm(self.cop1.fpr_f32(fs), rm);
                        self.cop1.set_fpr_u32(fd, i as u32);
                        self.scoreboard.set_fpr_latency(fd, fp_latency::CVT);
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x0D => {
                        // TRUNC.W.S — toward zero
                        let stall = self.scoreboard.fpr_stall(fs);
                        let i = f32_to_i32_trunc(self.cop1.fpr_f32(fs));
                        self.cop1.set_fpr_u32(fd, i as u32);
                        self.scoreboard.set_fpr_latency(fd, fp_latency::CVT);
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x0E => {
                        // CEIL.W.S
                        let stall = self.scoreboard.fpr_stall(fs);
                        let i = f32_to_i32_ceil(self.cop1.fpr_f32(fs));
                        self.cop1.set_fpr_u32(fd, i as u32);
                        self.scoreboard.set_fpr_latency(fd, fp_latency::CVT);
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x0F => {
                        // FLOOR.W.S
                        let stall = self.scoreboard.fpr_stall(fs);
                        let i = f32_to_i32_floor(self.cop1.fpr_f32(fs));
                        self.cop1.set_fpr_u32(fd, i as u32);
                        self.scoreboard.set_fpr_latency(fd, fp_latency::CVT);
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x21 => {
                        // CVT.D.S — single in `fs` → double in `fd`
                        let stall = self.scoreboard.fpr_stall(fs);
                        let r = f64::from(self.cop1.fpr_f32(fs));
                        self.cop1.set_fpr_f64(fd, r, fr);
                        self.scoreboard.set_fpr_latency(fd, fp_latency::CVT);
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x24 => {
                        // CVT.W.S — float → signed 32-bit word in FPR (uses FCSR RM)
                        let stall = self.scoreboard.fpr_stall(fs);
                        let rm = fcsr_rm(self.cop1.fcsr);
                        let i = f32_to_i32_rm(self.cop1.fpr_f32(fs), rm);
                        self.cop1.set_fpr_u32(fd, i as u32);
                        self.scoreboard.set_fpr_latency(fd, fp_latency::CVT);
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x25 => {
                        // CVT.L.S — single → signed 64-bit integer in FPR (uses FCSR RM)
                        let stall = self.scoreboard.fpr_stall(fs);
                        let rm = fcsr_rm(self.cop1.fcsr);
                        let i = f32_to_i64_rm(self.cop1.fpr_f32(fs), rm);
                        self.cop1.set_fpr_u64(fd, i as u64, fr);
                        self.scoreboard.set_fpr_latency(fd, fp_latency::CVT);
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    f @ 0x30..=0x3F => {
                        // C.cond.S — compare, sets FCSR CC
                        let stall = self.scoreboard.fpr_stall_2(fs, ft);
                        self.cop1.set_cc0(cond_f32(
                            self.cop1.fpr_f32(fs),
                            self.cop1.fpr_f32(ft),
                            f,
                        ));
                        self.scoreboard.set_fcsr_cc_latency(fp_latency::CMP);
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    _ => Ok(cycles::ALU),
                }
            }
            0x11 => {
                // .D (double-precision)
                match funct {
                    0x00 => {
                        // ADD.D
                        let stall = self.scoreboard.fpr_stall_2(fs, ft);
                        let r = self.cop1.fpr_f64(fs, fr) + self.cop1.fpr_f64(ft, fr);
                        self.cop1.set_fpr_f64(fd, r, fr);
                        self.scoreboard.set_fpr_latency(fd, fp_latency::ADD_MUL_D);
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x01 => {
                        // SUB.D
                        let stall = self.scoreboard.fpr_stall_2(fs, ft);
                        let r = self.cop1.fpr_f64(fs, fr) - self.cop1.fpr_f64(ft, fr);
                        self.cop1.set_fpr_f64(fd, r, fr);
                        self.scoreboard.set_fpr_latency(fd, fp_latency::ADD_MUL_D);
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x02 => {
                        // MUL.D
                        let stall = self.scoreboard.fpr_stall_2(fs, ft);
                        let r = self.cop1.fpr_f64(fs, fr) * self.cop1.fpr_f64(ft, fr);
                        self.cop1.set_fpr_f64(fd, r, fr);
                        self.scoreboard.set_fpr_latency(fd, fp_latency::ADD_MUL_D);
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x03 => {
                        // DIV.D
                        let stall = self.scoreboard.fpr_stall_2(fs, ft);
                        let a = self.cop1.fpr_f64(fs, fr);
                        let b = self.cop1.fpr_f64(ft, fr);
                        let r = if b == 0.0 { 0.0 } else { a / b };
                        self.cop1.set_fpr_f64(fd, r, fr);
                        self.scoreboard.set_fpr_latency(fd, fp_latency::DIV_D);
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x04 => {
                        // SQRT.D
                        let stall = self.scoreboard.fpr_stall(fs);
                        let r = self.cop1.fpr_f64(fs, fr).sqrt();
                        self.cop1.set_fpr_f64(fd, r, fr);
                        self.scoreboard.set_fpr_latency(fd, fp_latency::SQRT_D);
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x05 => {
                        // ABS.D
                        let stall = self.scoreboard.fpr_stall(fs);
                        let r = self.cop1.fpr_f64(fs, fr).abs();
                        self.cop1.set_fpr_f64(fd, r, fr);
                        self.scoreboard.set_fpr_latency(fd, fp_latency::CVT);
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x06 => {
                        // MOV.D
                        let stall = self.scoreboard.fpr_stall(fs);
                        let v = self.cop1.fpr_f64(fs, fr);
                        self.cop1.set_fpr_f64(fd, v, fr);
                        self.scoreboard.set_fpr_latency(fd, fp_latency::CVT);
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x07 => {
                        // NEG.D
                        let stall = self.scoreboard.fpr_stall(fs);
                        let r = -self.cop1.fpr_f64(fs, fr);
                        self.cop1.set_fpr_f64(fd, r, fr);
                        self.scoreboard.set_fpr_latency(fd, fp_latency::CVT);
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x0C => {
                        // ROUND.W.D
                        let stall = self.scoreboard.fpr_stall(fs);
                        let rm = fcsr_rm(self.cop1.fcsr);
                        let i = f64_to_i32_rm(self.cop1.fpr_f64(fs, fr), rm);
                        self.cop1.set_fpr_u32(fd, i as u32);
                        self.scoreboard.set_fpr_latency(fd, fp_latency::CVT);
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x0D => {
                        // TRUNC.W.D
                        let stall = self.scoreboard.fpr_stall(fs);
                        let i = f64_to_i32_trunc(self.cop1.fpr_f64(fs, fr));
                        self.cop1.set_fpr_u32(fd, i as u32);
                        self.scoreboard.set_fpr_latency(fd, fp_latency::CVT);
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x0E => {
                        // CEIL.W.D
                        let stall = self.scoreboard.fpr_stall(fs);
                        let i = f64_to_i32_ceil(self.cop1.fpr_f64(fs, fr));
                        self.cop1.set_fpr_u32(fd, i as u32);
                        self.scoreboard.set_fpr_latency(fd, fp_latency::CVT);
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x0F => {
                        // FLOOR.W.D
                        let stall = self.scoreboard.fpr_stall(fs);
                        let i = f64_to_i32_floor(self.cop1.fpr_f64(fs, fr));
                        self.cop1.set_fpr_u32(fd, i as u32);
                        self.scoreboard.set_fpr_latency(fd, fp_latency::CVT);
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x20 => {
                        // CVT.S.D
                        let stall = self.scoreboard.fpr_stall(fs);
                        let r = self.cop1.fpr_f64(fs, fr) as f32;
                        self.cop1.set_fpr_f32(fd, r);
                        self.scoreboard.set_fpr_latency(fd, fp_latency::CVT);
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x24 => {
                        // CVT.W.D
                        let stall = self.scoreboard.fpr_stall(fs);
                        let rm = fcsr_rm(self.cop1.fcsr);
                        let w = f64_to_i32_rm(self.cop1.fpr_f64(fs, fr), rm);
                        self.cop1.set_fpr_u32(fd, w as u32);
                        self.scoreboard.set_fpr_latency(fd, fp_latency::CVT);
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x25 => {
                        // CVT.L.D
                        let stall = self.scoreboard.fpr_stall(fs);
                        let rm = fcsr_rm(self.cop1.fcsr);
                        let i = f64_to_i64_rm(self.cop1.fpr_f64(fs, fr), rm);
                        self.cop1.set_fpr_u64(fd, i as u64, fr);
                        self.scoreboard.set_fpr_latency(fd, fp_latency::CVT);
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    f @ 0x30..=0x3F => {
                        // C.cond.D — compare, sets FCSR CC
                        let stall = self.scoreboard.fpr_stall_2(fs, ft);
                        self.cop1.set_cc0(cond_f64(
                            self.cop1.fpr_f64(fs, fr),
                            self.cop1.fpr_f64(ft, fr),
                            f,
                        ));
                        self.scoreboard.set_fcsr_cc_latency(fp_latency::CMP);
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    _ => Ok(cycles::ALU),
                }
            }
            0x14 => {
                // .W (fixed-point / integer in FPR)
                match funct {
                    0x20 => {
                        // CVT.S.W — signed word in `fs` → single in `fd`
                        let stall = self.scoreboard.fpr_stall(fs);
                        let i = self.cop1.fpr_u32(fs) as i32;
                        self.cop1.set_fpr_f32(fd, i as f32);
                        self.scoreboard.set_fpr_latency(fd, fp_latency::CVT);
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x21 => {
                        // CVT.D.W
                        let stall = self.scoreboard.fpr_stall(fs);
                        let i = self.cop1.fpr_u32(fs) as i32;
                        self.cop1.set_fpr_f64(fd, i as f64, fr);
                        self.scoreboard.set_fpr_latency(fd, fp_latency::CVT);
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x25 => {
                        // CVT.L.W — 32-bit fixed word in FPR → signed 64-bit in FPR
                        let stall = self.scoreboard.fpr_stall(fs);
                        let i = self.cop1.fpr_u32(fs) as i32 as i64;
                        self.cop1.set_fpr_u64(fd, i as u64, fr);
                        self.scoreboard.set_fpr_latency(fd, fp_latency::CVT);
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    _ => Ok(cycles::ALU),
                }
            }
            0x15 => {
                // `.L` — 64-bit integer in FPR (MIPS fmt L)
                match funct {
                    0x20 => {
                        // CVT.S.L
                        let stall = self.scoreboard.fpr_stall(fs);
                        let v = self.cop1.fpr_u64(fs, fr) as i64;
                        self.cop1.set_fpr_f32(fd, v as f32);
                        self.scoreboard.set_fpr_latency(fd, fp_latency::CVT);
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    0x21 => {
                        // CVT.D.L
                        let stall = self.scoreboard.fpr_stall(fs);
                        let v = self.cop1.fpr_u64(fs, fr) as i64;
                        self.cop1.set_fpr_f64(fd, v as f64, fr);
                        self.scoreboard.set_fpr_latency(fd, fp_latency::CVT);
                        Ok(stall.saturating_add(cycles::ALU))
                    }
                    _ => Ok(cycles::ALU),
                }
            }
            _ => Ok(cycles::ALU),
        }
    }

    /// Execute a CACHE instruction operation.
    fn exec_cache_op(&mut self, op: u8, paddr: u32) {
        use super::cache::{DCACHE_SETS, ICACHE_SETS};

        // Extract set index and way from address for index operations
        // I-cache: line_bits=5, set_bits=8
        let icache_set = ((paddr >> 5) & 0xFF) as usize;
        let icache_way = ((paddr >> 13) & 1) as usize;
        // D-cache: line_bits=4, set_bits=8
        let dcache_set = ((paddr >> 4) & 0xFF) as usize;
        let dcache_way = ((paddr >> 12) & 1) as usize;

        match CacheOp::from_u8(op) {
            Some(CacheOp::IndexInvalidateI) => {
                self.icache.invalidate_index(icache_set, icache_way);
            }
            Some(CacheOp::IndexLoadTagI) => {
                // Load tag into COP0 TagLo/TagHi - simplified: just read the tag
                if icache_set < ICACHE_SETS && icache_way < 2 {
                    let line = &self.icache.sets[icache_set].lines[icache_way];
                    self.cop0.taglo = if line.valid {
                        (line.tag << 8) | 0x80 // PState valid bit
                    } else {
                        0
                    };
                }
            }
            Some(CacheOp::IndexStoreTagI) => {
                // Store TagLo into I-cache tag
                if icache_set < ICACHE_SETS && icache_way < 2 {
                    let taglo = self.cop0.taglo;
                    let line = &mut self.icache.sets[icache_set].lines[icache_way];
                    line.tag = (taglo >> 8) & 0x00FF_FFFF;
                    line.valid = (taglo & 0x80) != 0;
                }
            }
            Some(CacheOp::HitInvalidateI) => {
                self.icache.invalidate_hit(paddr);
            }
            Some(CacheOp::FillI) | Some(CacheOp::HitWritebackInvalidateI) => {
                // Fill I-cache line from memory (or hit invalidate for I$)
                self.icache.invalidate_hit(paddr);
                self.icache.fill(paddr);
            }
            Some(CacheOp::IndexWritebackInvalidateD) => {
                self.dcache.writeback_invalidate_index(dcache_set, dcache_way);
            }
            Some(CacheOp::IndexLoadTagD) => {
                if dcache_set < DCACHE_SETS && dcache_way < 2 {
                    let line = &self.dcache.sets[dcache_set].lines[dcache_way];
                    self.cop0.taglo = if line.valid {
                        let mut v = (line.tag << 8) | 0x80;
                        if line.dirty {
                            v |= 0x40; // Dirty bit
                        }
                        v
                    } else {
                        0
                    };
                }
            }
            Some(CacheOp::IndexStoreTagD) => {
                if dcache_set < DCACHE_SETS && dcache_way < 2 {
                    let taglo = self.cop0.taglo;
                    let line = &mut self.dcache.sets[dcache_set].lines[dcache_way];
                    line.tag = (taglo >> 8) & 0x00FF_FFFF;
                    line.valid = (taglo & 0x80) != 0;
                    line.dirty = (taglo & 0x40) != 0;
                }
            }
            Some(CacheOp::CreateDirtyExclusiveD) => {
                self.dcache.create_dirty_exclusive(paddr);
            }
            Some(CacheOp::HitInvalidateD) => {
                self.dcache.hit_invalidate(paddr);
            }
            Some(CacheOp::HitWritebackInvalidateD) => {
                self.dcache.hit_writeback_invalidate(paddr);
            }
            Some(CacheOp::HitWritebackD) => {
                self.dcache.hit_writeback(paddr);
            }
            None => {
                // Unknown cache op - NOP
            }
        }
    }

    /// `BC1F` / `BC1T`: branch on floating-point condition (uses FCSR CC bits).
    fn exec_bc1(
        &mut self,
        pc: u64,
        word: u32,
        bus: &mut impl Bus,
    ) -> Result<u64, CpuHalt> {
        // BC1F/BC1T: stall on FCSR condition code
        let stall = self.scoreboard.fcsr_cc_stall();
        let imm = (((word & 0xFFFF) as i16) as i64) << 2;
        let tf = (word >> 16) & 1 != 0;
        let nd = (word >> 17) & 1 != 0;
        let cc = (word >> 18) & 0x7;
        let cond = (self.cop1.fcsr >> (23 + cc)) & 1;
        let take = if tf { cond != 0 } else { cond == 0 };

        if nd && !take {
            self.pc = pc.wrapping_add(8);
            return Ok(stall.saturating_add(cycles::BRANCH));
        }

        let delay_pc = pc.wrapping_add(4);
        self.delay_slot_branch_pc = Some(pc);
        let delay_word = try_mem!(self.fetch32(bus, pc, delay_pc));
        let mut cycles = cycles::BRANCH;
        let d = self.exec_non_branch_delay_slot(pc, delay_pc, delay_word, bus)?;
        cycles += d;
        if self.exception_taken {
            return Ok(stall.saturating_add(cycles));
        }
        self.delay_slot_branch_pc = None;

        self.pc = if take {
            pc.wrapping_add(imm as u64).wrapping_add(4)
        } else {
            pc.wrapping_add(8)
        };
        if take {
            cycles += cycles::BRANCH_TAKEN_EXTRA;
        }
        Ok(stall.saturating_add(cycles))
    }
}

impl Default for R4300i {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bus::PhysicalMemory;

    fn write_be32(mem: &mut PhysicalMemory, paddr: u32, v: u32) {
        mem.write_u32(paddr, v);
    }

    #[test]
    fn addiu_and_lui() {
        let mut cpu = R4300i::new();
        cpu.reset(0xA000_0000);
        let mut mem = PhysicalMemory::new(1024 * 1024);
        // lui t0, 0x1234
        write_be32(&mut mem, 0, 0x3C081234);
        // addiu t1, t0, 0x5678  (encoding: rs=t0=8, rt=t1=9)
        write_be32(&mut mem, 4, 0x2509_5678);

        assert_eq!(cpu.step(&mut mem, false).unwrap(), cycles::ALU);
        assert_eq!(cpu.pc, 0xA000_0004);
        assert_eq!(cpu.regs[8], 0x1234_0000);

        assert_eq!(cpu.step(&mut mem, false).unwrap(), cycles::ALU);
        assert_eq!(cpu.pc, 0xA000_0008);
        assert_eq!(cpu.regs[9], 0x1234_5678);
    }

    #[test]
    fn branch_delay_slot_skipped_when_not_taken() {
        let mut cpu = R4300i::new();
        let mut mem = PhysicalMemory::new(1024 * 1024);
        // beq r0,r0,+2  -> skip 8 bytes if taken; we use r0!=r0 so not taken
        // actually beq r0,r0 is always taken. Use bne r0, r0
        // 0x80000000: bne $0,$0,+2  (not taken)  opcode 0x05
        write_be32(&mut mem, 0, 0x1400_0002);
        // delay: addiu t0, t0, 1
        write_be32(&mut mem, 4, 0x2508_0001);
        // next
        write_be32(&mut mem, 8, 0x2508_0002);

        cpu.pc = 0xA000_0000;
        cpu.regs[8] = 0;
        let c = cpu.step(&mut mem, false).unwrap();
        assert!(c > cycles::ALU);
        assert_eq!(cpu.regs[8], 1, "delay slot must run");
        assert_eq!(cpu.pc, 0xA000_0008);
    }

    #[test]
    fn jump_and_link_sets_ra() {
        let mut cpu = R4300i::new();
        let mut mem = PhysicalMemory::new(1024 * 1024);
        // jal 0xA0000020  (word index: target byte 0xA0000020)
        // Encoding: upper 4 bits from PC | imm26<<2
        write_be32(&mut mem, 0, 0x0C00_0008);
        // delay: nop
        write_be32(&mut mem, 4, 0x0000_0000);
        cpu.pc = 0xA000_0000;
        cpu.step(&mut mem, false).unwrap();
        assert_eq!(cpu.pc, 0xA000_0020);
        assert_eq!(cpu.regs[31], 0xA000_0008);
    }

    #[test]
    fn eret_restores_epc_and_clears_exl() {
        use crate::cpu::cop0::{CAUSE_BD, STATUS_ERL, STATUS_EXL};

        let mut cpu = R4300i::new();
        cpu.reset(0xFFFF_FFFF_A000_0180);
        cpu.cop0.epc = 0xFFFF_FFFF_A000_4000;
        cpu.cop0.cause |= CAUSE_BD;
        // Avoid default `ERL` so `ERET` uses `EPC` (not `ErrorEPC`).
        cpu.cop0.status = (cpu.cop0.status & !STATUS_ERL) | STATUS_EXL;

        let mut mem = PhysicalMemory::new(1024 * 1024);
        write_be32(&mut mem, 0x180, 0x4200_0018);

        assert_eq!(cpu.step(&mut mem, false).unwrap(), crate::cycles::ALU);
        assert_eq!(cpu.pc, 0xFFFF_FFFF_A000_4000);
        assert!((cpu.cop0.status & STATUS_EXL) == 0);
        assert!((cpu.cop0.cause & CAUSE_BD) == 0);
    }

    #[test]
    fn syscall_and_break_general_exception() {
        use crate::cpu::cop0::{
            CAUSE_EXCCODE_MASK, CAUSE_EXCCODE_SHIFT, EXCCODE_BP, EXCCODE_SYSCALL, STATUS_BEV,
            STATUS_EXL,
        };

        let mut cpu = R4300i::new();
        cpu.reset(0xA000_0000);
        cpu.cop0.status &= !STATUS_BEV;
        let mut mem = PhysicalMemory::new(1024 * 1024);
        // SPECIAL + SYSCALL (funct 0x0C)
        write_be32(&mut mem, 0, 0x0000_000C);
        assert_eq!(cpu.step(&mut mem, false).unwrap(), cycles::EXCEPTION);
        assert_eq!(cpu.pc, 0xFFFF_FFFF_8000_0180);
        assert_eq!(cpu.cop0.epc, 0xA000_0000);
        assert!((cpu.cop0.status & STATUS_EXL) != 0);
        let exc = (cpu.cop0.cause >> CAUSE_EXCCODE_SHIFT) & CAUSE_EXCCODE_MASK;
        assert_eq!(exc, EXCCODE_SYSCALL);

        // Handler at vector: nop, then re-run at next PC with BREAK
        // Use kseg1 to avoid I-cache effects in test
        write_be32(&mut mem, 0x180, 0x0000_0000);
        write_be32(&mut mem, 0x184, 0x0000_000D);
        cpu.cop0.status &= !STATUS_EXL;
        cpu.pc = 0xFFFF_FFFF_A000_0180;
        assert_eq!(cpu.step(&mut mem, false).unwrap(), cycles::ALU);
        assert_eq!(cpu.pc, 0xFFFF_FFFF_A000_0184);
        assert_eq!(cpu.step(&mut mem, false).unwrap(), cycles::EXCEPTION);
        assert_eq!(cpu.cop0.epc, 0xFFFF_FFFF_A000_0184);
        let exc2 = (cpu.cop0.cause >> CAUSE_EXCCODE_SHIFT) & CAUSE_EXCCODE_MASK;
        assert_eq!(exc2, EXCCODE_BP);
    }

    #[test]
    fn kuseg_tlb_miss_exception() {
        use crate::cpu::cop0::{
            CAUSE_EXCCODE_MASK, CAUSE_EXCCODE_SHIFT, EXCCODE_TLBL, STATUS_BEV,
            KSEG0_TLB_REFILL_VECTOR_PC,
        };

        let mut cpu = R4300i::new();
        cpu.reset(0xA000_0000);
        cpu.cop0.status &= !STATUS_BEV;
        let mut mem = PhysicalMemory::new(1024 * 1024);
        // `LW $1, 0x100($0)` — kuseg `0x100`, no TLB entry.
        write_be32(&mut mem, 0, 0x8C01_0100);
        assert_eq!(cpu.step(&mut mem, false).unwrap(), cycles::EXCEPTION);
        // TLB refill with EXL=0 goes to offset 0x000, not 0x180
        assert_eq!(cpu.pc, KSEG0_TLB_REFILL_VECTOR_PC);
        assert_eq!(cpu.cop0.badvaddr, 0x100);
        let exc = (cpu.cop0.cause >> CAUSE_EXCCODE_SHIFT) & CAUSE_EXCCODE_MASK;
        assert_eq!(exc, EXCCODE_TLBL);
    }

    #[test]
    fn kuseg_load_maps_via_tlb() {
        use crate::cpu::tlb::TlbEntry;

        let mut cpu = R4300i::new();
        cpu.reset(0xA000_0000);
        cpu.cop0.tlb[0] = TlbEntry {
            page_mask: 0,
            hi: 0,
            lo0: 0x7,
            lo1: 0,
        };
        let mut mem = PhysicalMemory::new(1024 * 1024);
        mem.write_u32(0x100, 0xCAFE_BABE);
        write_be32(&mut mem, 0, 0x8C01_0100);
        assert_eq!(cpu.step(&mut mem, false).unwrap(), cycles::MEM_ACCESS);
        assert_eq!(cpu.regs[1], i64::from(0xCAFE_BABEu32 as i32) as u64);
    }

    #[test]
    fn kuseg_load_maps_via_tlb_16k_pagemask() {
        use crate::cpu::tlb::TlbEntry;

        let mut cpu = R4300i::new();
        cpu.reset(0xA000_0000);
        cpu.cop0.tlb[0] = TlbEntry {
            page_mask: 0x6000,
            hi: 0,
            lo0: 0x7,
            lo1: 0,
        };
        let mut mem = PhysicalMemory::new(1024 * 1024);
        mem.write_u32(0x2000, 0x1122_3344);
        write_be32(&mut mem, 0, 0x8C01_2000);
        assert_eq!(cpu.step(&mut mem, false).unwrap(), cycles::MEM_ACCESS);
        assert_eq!(cpu.regs[1], i64::from(0x1122_3344u32 as i32) as u64);
    }

    #[test]
    fn ksseg_load_maps_via_tlb() {
        use crate::cpu::tlb::TlbEntry;

        let mut cpu = R4300i::new();
        cpu.reset(0xA000_0000);
        cpu.cop0.tlb[0] = TlbEntry {
            page_mask: 0,
            hi: 0xC000_0000,
            lo0: 0x7,
            lo1: 0,
        };
        let mut mem = PhysicalMemory::new(1024 * 1024);
        mem.write_u32(0x100, 0x55AA_66BB);
        // `LW $1, 0x100($1)` — rs=1 (0x8C21), not `0x8C01` (rs=$0).
        write_be32(&mut mem, 0, 0x8C21_0100);
        cpu.regs[1] = 0xC000_0000;
        assert_eq!(cpu.step(&mut mem, false).unwrap(), cycles::MEM_ACCESS);
        assert_eq!(cpu.regs[1], i64::from(0x55AA_66BBu32 as i32) as u64);
    }

    #[test]
    fn cfc0_ctc0_compare_mirror_mtc0() {
        let mut cpu = R4300i::new();
        cpu.reset(0xA000_0000);
        let mut mem = PhysicalMemory::new(1024 * 1024);
        // `MTC0 $8, $Compare` — sub 0x04, rt=8, rd=11
        write_be32(&mut mem, 0, 0x4088_5800);
        // `CFC0 $9, $Compare` — sub 0x02, rt=9, rd=11
        write_be32(&mut mem, 4, 0x4049_5800);
        cpu.regs[8] = 0xABCD_1234;
        assert_eq!(cpu.step(&mut mem, false).unwrap(), cycles::COP_MOVE);
        assert_eq!(cpu.step(&mut mem, false).unwrap(), cycles::COP_MOVE);
        assert_eq!(cpu.cop0.compare, 0xABCD_1234);
        // `CFC0` mirrors `MFC0`: 32-bit value is sign-extended into the GPR.
        assert_eq!(cpu.regs[9] as i64, i64::from(0xABCD_1234u32 as i32));

        // `CTC0 $8, $Compare` — sub 0x06
        write_be32(&mut mem, 8, 0x40C8_5800);
        cpu.regs[8] = 0x1000_0000;
        assert_eq!(cpu.step(&mut mem, false).unwrap(), cycles::COP_MOVE);
        assert_eq!(cpu.cop0.compare, 0x1000_0000);
    }

    #[test]
    fn lwc2_swc2_cop2_stub_round_trip() {
        let mut cpu = R4300i::new();
        cpu.reset(0xA000_0000);
        let mut mem = PhysicalMemory::new(1024 * 1024);
        mem.write_u32(0x100, 0xDEAD_BEEF);
        // Use kseg1 base to avoid D-cache effects in test
        cpu.regs[1] = 0xA000_0000;
        // `LWC2 $5,0x100($1)` — opcode 0x32
        write_be32(&mut mem, 0, 0xC825_0100);
        assert_eq!(cpu.step(&mut mem, false).unwrap(), cycles::MEM_ACCESS);
        assert_eq!(cpu.cop2[5], 0xFFFF_FFFF_DEAD_BEEF);

        // `SWC2 $5,0x104($1)` — opcode 0x3A
        write_be32(&mut mem, 4, 0xE825_0104);
        assert_eq!(cpu.step(&mut mem, false).unwrap(), cycles::MEM_ACCESS);
        assert_eq!(mem.read_u32(0x104).unwrap(), 0xDEAD_BEEF);
    }

    #[test]
    fn syscall_in_branch_delay_slot_sets_bd_and_branch_epc() {
        use crate::cpu::cop0::{CAUSE_BD, STATUS_BEV};

        let mut cpu = R4300i::new();
        cpu.reset(0xA000_0000);
        cpu.cop0.status &= !STATUS_BEV;
        let mut mem = PhysicalMemory::new(1024 * 1024);
        // `BEQ $0,$0,+2` — taken; delay slot at +4 is `SYSCALL`.
        write_be32(&mut mem, 0, 0x1000_0002);
        write_be32(&mut mem, 4, 0x0000_000C);

        let c = cpu.step(&mut mem, false).unwrap();
        assert_eq!(c, cycles::BRANCH + cycles::EXCEPTION);
        assert_eq!(cpu.pc, 0xFFFF_FFFF_8000_0180);
        assert_eq!(cpu.cop0.epc, 0xA000_0000);
        assert!((cpu.cop0.cause & CAUSE_BD) != 0);
    }

    #[test]
    fn timer_compare_delivers_interrupt() {
        use crate::cpu::cop0::{CAUSE_IP7, STATUS_BEV, STATUS_IE, STATUS_IM7};

        let mut cpu = R4300i::new();
        cpu.reset(0xA000_0000);
        cpu.cop0.status = (STATUS_IE | STATUS_IM7) & !STATUS_BEV;
        cpu.cop0.compare = 3;
        cpu.cop0.advance_count_wrapped(2);
        assert_eq!(cpu.cop0.count, 2);
        cpu.cop0.advance_count_wrapped(1);
        assert_eq!(cpu.cop0.count, 3);
        assert!((cpu.cop0.cause & CAUSE_IP7) != 0);

        let mut mem = PhysicalMemory::new(1024 * 1024);
        write_be32(&mut mem, 0, 0x0000_0000);
        assert_eq!(cpu.step(&mut mem, false).unwrap(), cycles::INTERRUPT);
        assert_eq!(cpu.pc, 0xFFFF_FFFF_8000_0180);
        // IP7 is level-sensitive on VR4300: it stays asserted until software writes
        // Compare. The kernel ISR relies on this to decode the interrupt source.
        assert!((cpu.cop0.cause & CAUSE_IP7) != 0);
        // Writing Compare clears IP7.
        cpu.cop0.write_32(11, 10);
        assert!((cpu.cop0.cause & CAUSE_IP7) == 0);
    }
}
