use super::cop0::Cop0;
use super::cop1::{
    cond_f32, cond_f64, f32_to_i32_ceil, f32_to_i32_floor, f32_to_i32_rm, f32_to_i32_trunc,
    f32_to_i64_rm, f64_to_i32_ceil, f64_to_i32_floor, f64_to_i32_rm, f64_to_i32_trunc,
    f64_to_i64_rm, fcsr_rm, Cop1,
};
use crate::bus::{virt_to_phys, Bus};
use crate::cycles;

#[derive(Clone, Debug)]
pub enum CpuHalt {
    UnmappedAddress { vaddr: u64 },
    UnalignedFetch { vaddr: u64 },
    UnalignedAccess { vaddr: u64, width: u8 },
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

    fn fetch32(&self, bus: &mut impl Bus, vaddr: u64) -> Result<u32, CpuHalt> {
        if vaddr & 3 != 0 {
            return Err(CpuHalt::UnalignedFetch { vaddr });
        }
        let p = virt_to_phys(vaddr).ok_or(CpuHalt::UnmappedAddress { vaddr })?;
        bus.read_u32(p).ok_or(CpuHalt::UnmappedAddress { vaddr })
    }

    fn load32(&mut self, bus: &mut impl Bus, vaddr: u64) -> Result<u32, CpuHalt> {
        if vaddr & 3 != 0 {
            return Err(CpuHalt::UnalignedAccess { vaddr, width: 4 });
        }
        let p = virt_to_phys(vaddr).ok_or(CpuHalt::UnmappedAddress { vaddr })?;
        bus.read_u32(p).ok_or(CpuHalt::UnmappedAddress { vaddr })
    }

    fn store32(&mut self, bus: &mut impl Bus, vaddr: u64, value: u32) -> Result<(), CpuHalt> {
        if vaddr & 3 != 0 {
            return Err(CpuHalt::UnalignedAccess { vaddr, width: 4 });
        }
        let p = virt_to_phys(vaddr).ok_or(CpuHalt::UnmappedAddress { vaddr })?;
        bus.write_u32(p, value);
        Ok(())
    }

    fn load64(&mut self, bus: &mut impl Bus, vaddr: u64) -> Result<u64, CpuHalt> {
        if vaddr & 7 != 0 {
            return Err(CpuHalt::UnalignedAccess { vaddr, width: 8 });
        }
        let p = virt_to_phys(vaddr).ok_or(CpuHalt::UnmappedAddress { vaddr })?;
        let hi = bus.read_u32(p).ok_or(CpuHalt::UnmappedAddress { vaddr })?;
        let lo = bus
            .read_u32(p.wrapping_add(4))
            .ok_or(CpuHalt::UnmappedAddress { vaddr })?;
        Ok((u64::from(hi) << 32) | u64::from(lo))
    }

    fn store64(&mut self, bus: &mut impl Bus, vaddr: u64, value: u64) -> Result<(), CpuHalt> {
        if vaddr & 7 != 0 {
            return Err(CpuHalt::UnalignedAccess { vaddr, width: 8 });
        }
        let p = virt_to_phys(vaddr).ok_or(CpuHalt::UnmappedAddress { vaddr })?;
        bus.write_u32(p, (value >> 32) as u32);
        bus.write_u32(p.wrapping_add(4), value as u32);
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
    fn store_swl(&mut self, bus: &mut impl Bus, eff: u64, val: u64) -> Result<(), CpuHalt> {
        let b = (val as u32).to_be_bytes();
        let al = eff & !3;
        let o = (eff & 3) as usize;
        for i in o..4 {
            let pa = virt_to_phys(al.wrapping_add(i as u64)).ok_or(CpuHalt::UnmappedAddress { vaddr: al.wrapping_add(i as u64) })?;
            bus.write_u8(pa, b[i]);
        }
        Ok(())
    }

    /// Store word right: low-order bytes of rt from aligned address through eff.
    fn store_swr(&mut self, bus: &mut impl Bus, eff: u64, val: u64) -> Result<(), CpuHalt> {
        let al = eff & !3;
        let o = (eff & 3) as usize;
        if o == 0 {
            return self.store32(bus, eff, val as u32);
        }
        let b = (val as u32).to_be_bytes();
        for j in 0..=o {
            let pa = virt_to_phys(al.wrapping_add(j as u64)).ok_or(CpuHalt::UnmappedAddress { vaddr: al.wrapping_add(j as u64) })?;
            bus.write_u8(pa, b[3 - o + j]);
        }
        Ok(())
    }

    fn load16_signed(&mut self, bus: &mut impl Bus, vaddr: u64) -> Result<u32, CpuHalt> {
        if vaddr & 1 != 0 {
            return Err(CpuHalt::UnalignedAccess { vaddr, width: 2 });
        }
        let p = virt_to_phys(vaddr).ok_or(CpuHalt::UnmappedAddress { vaddr })?;
        let hi = bus.read_u8(p).ok_or(CpuHalt::UnmappedAddress { vaddr })?;
        let lo = bus.read_u8(p.wrapping_add(1)).ok_or(CpuHalt::UnmappedAddress { vaddr })?;
        let h = u16::from_be_bytes([hi, lo]);
        Ok(i32::from(h as i16) as u32)
    }

    fn load16_unsigned(&mut self, bus: &mut impl Bus, vaddr: u64) -> Result<u32, CpuHalt> {
        if vaddr & 1 != 0 {
            return Err(CpuHalt::UnalignedAccess { vaddr, width: 2 });
        }
        let p = virt_to_phys(vaddr).ok_or(CpuHalt::UnmappedAddress { vaddr })?;
        let hi = bus.read_u8(p).ok_or(CpuHalt::UnmappedAddress { vaddr })?;
        let lo = bus.read_u8(p.wrapping_add(1)).ok_or(CpuHalt::UnmappedAddress { vaddr })?;
        Ok(u32::from(u16::from_be_bytes([hi, lo])))
    }

    fn load8_signed(&mut self, bus: &mut impl Bus, vaddr: u64) -> Result<u32, CpuHalt> {
        let p = virt_to_phys(vaddr).ok_or(CpuHalt::UnmappedAddress { vaddr })?;
        let b = bus.read_u8(p).ok_or(CpuHalt::UnmappedAddress { vaddr })?;
        Ok(i32::from(b as i8) as u32)
    }

    fn load8_unsigned(&mut self, bus: &mut impl Bus, vaddr: u64) -> Result<u32, CpuHalt> {
        let p = virt_to_phys(vaddr).ok_or(CpuHalt::UnmappedAddress { vaddr })?;
        let b = bus.read_u8(p).ok_or(CpuHalt::UnmappedAddress { vaddr })?;
        Ok(u32::from(b))
    }

    fn store16(&mut self, bus: &mut impl Bus, vaddr: u64, value: u32) -> Result<(), CpuHalt> {
        if vaddr & 1 != 0 {
            return Err(CpuHalt::UnalignedAccess { vaddr, width: 2 });
        }
        let p = virt_to_phys(vaddr).ok_or(CpuHalt::UnmappedAddress { vaddr })?;
        let half = (value & 0xFFFF) as u16;
        let [a, b] = half.to_be_bytes();
        bus.write_u8(p, a);
        bus.write_u8(p.wrapping_add(1), b);
        Ok(())
    }

    fn store8(&mut self, bus: &mut impl Bus, vaddr: u64, value: u32) -> Result<(), CpuHalt> {
        let p = virt_to_phys(vaddr).ok_or(CpuHalt::UnmappedAddress { vaddr })?;
        bus.write_u8(p, (value & 0xFF) as u8);
        Ok(())
    }

    /// Execute one **architectural** instruction (including its delay slot for
    /// branches/jumps). Returns CPU cycles consumed for this retirement.
    ///
    /// `rcp_interrupt`: MI-driven external interrupt line (pending and mask). If
    /// [`crate::cpu::cop0::Cop0::interrupts_enabled`], takes exception before fetch.
    pub fn step(&mut self, bus: &mut impl Bus, rcp_interrupt: bool) -> Result<u64, CpuHalt> {
        if rcp_interrupt && self.cop0.interrupts_enabled() {
            let epc = self.pc;
            let v = self.cop0.interrupt_vector();
            self.cop0.enter_interrupt_exception(epc);
            self.pc = v;
            return Ok(cycles::INTERRUPT);
        }

        let pc = self.pc;
        let word = self.fetch32(bus, pc)?;
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
        let target = (pc & 0xF000_0000) | u64::from(word & 0x03FF_FFFF) << 2;
        let mut cycles = cycles::BRANCH;

        if op == 3 {
            // JAL
            self.set_gpr(31, pc.wrapping_add(8));
        }

        let delay_pc = pc.wrapping_add(4);
        let delay_word = self.fetch32(bus, delay_pc)?;
        cycles += self.exec_non_branch(delay_pc, delay_word, bus)?;
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

        let cond = match rt_field {
            0x00 | 0x02 | 0x10 | 0x12 => (self.gpr(rs) as i64) < 0,
            0x01 | 0x03 | 0x11 | 0x13 => (self.gpr(rs) as i64) >= 0,
            _ => return Err(CpuHalt::UnimplementedOpcode { pc, word }),
        };

        let likely = matches!(rt_field, 0x02 | 0x03 | 0x12 | 0x13);
        let link = matches!(rt_field, 0x10 | 0x11 | 0x12 | 0x13);

        if link {
            self.set_gpr(31, pc.wrapping_add(8));
        }

        if likely && !cond {
            self.pc = pc.wrapping_add(8);
            return Ok(cycles::BRANCH);
        }

        let delay_pc = pc.wrapping_add(4);
        let delay_word = self.fetch32(bus, delay_pc)?;
        let mut cycles = cycles::BRANCH;
        cycles += self.exec_non_branch(delay_pc, delay_word, bus)?;

        self.pc = if cond {
            pc.wrapping_add((imm << 2) as u64).wrapping_add(4)
        } else {
            pc.wrapping_add(8)
        };
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
                // JR
                let target = self.gpr(rs);
                let mut cycles = cycles::BRANCH;
                let delay_pc = pc.wrapping_add(4);
                let delay_word = self.fetch32(bus, delay_pc)?;
                cycles += self.exec_non_branch(delay_pc, delay_word, bus)?;
                self.pc = target;
                Ok(cycles)
            }
            0x09 => {
                // JALR
                let target = self.gpr(rs);
                self.set_gpr(rd, pc.wrapping_add(8));
                let mut cycles = cycles::BRANCH;
                let delay_pc = pc.wrapping_add(4);
                let delay_word = self.fetch32(bus, delay_pc)?;
                cycles += self.exec_non_branch(delay_pc, delay_word, bus)?;
                self.pc = target;
                Ok(cycles)
            }
            _ => {
                let c = self.exec_non_branch(pc, word, bus)?;
                if !Self::is_eret(word) {
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
                let c = self.exec_non_branch(pc, word, bus)?;
                if !Self::is_eret(word) {
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

        let take = match op {
            0x14 => self.gpr(rs) == self.gpr(rt),
            0x15 => self.gpr(rs) != self.gpr(rt),
            0x16 => (self.gpr(rs) as i64) <= 0,
            0x17 => (self.gpr(rs) as i64) > 0,
            _ => unreachable!(),
        };

        if !take {
            self.pc = pc.wrapping_add(8);
            return Ok(cycles::BRANCH);
        }

        let delay_pc = pc.wrapping_add(4);
        let delay_word = self.fetch32(bus, delay_pc)?;
        let mut cycles = cycles::BRANCH;
        cycles += self.exec_non_branch(delay_pc, delay_word, bus)?;
        self.pc = pc.wrapping_add(imm as u64).wrapping_add(4);
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
        let mut cycles = cycles::BRANCH;

        let take = match op {
            0x04 => self.gpr(rs) == self.gpr(rt),
            0x05 => self.gpr(rs) != self.gpr(rt),
            0x06 => (self.gpr(rs) as i64) <= 0,
            0x07 => (self.gpr(rs) as i64) > 0,
            _ => unreachable!(),
        };

        let delay_pc = pc.wrapping_add(4);
        let delay_word = self.fetch32(bus, delay_pc)?;
        cycles += self.exec_non_branch(delay_pc, delay_word, bus)?;

        self.pc = if take {
            pc.wrapping_add(imm as u64).wrapping_add(4)
        } else {
            pc.wrapping_add(8)
        };
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
                        // SLL
                        self.set_gpr(rd, self.gpr(rt) << sa);
                        Ok(cycles::ALU)
                    }
                    0x02 => {
                        self.set_gpr(rd, self.gpr(rt) >> sa);
                        Ok(cycles::ALU)
                    }
                    0x03 => {
                        self.set_gpr(rd, ((self.gpr(rt) as i64) >> sa) as u64);
                        Ok(cycles::ALU)
                    }
                    0x04 => {
                        let s = self.gpr(rs) & 0x1F;
                        self.set_gpr(rd, self.gpr(rt) << s);
                        Ok(cycles::ALU)
                    }
                    0x06 => {
                        let s = self.gpr(rs) & 0x1F;
                        self.set_gpr(rd, self.gpr(rt) >> s);
                        Ok(cycles::ALU)
                    }
                    0x07 => {
                        let s = self.gpr(rs) & 0x1F;
                        self.set_gpr(rd, ((self.gpr(rt) as i64) >> s) as u64);
                        Ok(cycles::ALU)
                    }
                    0x21 => {
                        self.set_gpr(rd, self.gpr(rs).wrapping_add(self.gpr(rt)));
                        Ok(cycles::ALU)
                    }
                    0x23 => {
                        self.set_gpr(rd, self.gpr(rs).wrapping_sub(self.gpr(rt)));
                        Ok(cycles::ALU)
                    }
                    0x24 => {
                        self.set_gpr(rd, self.gpr(rs) & self.gpr(rt));
                        Ok(cycles::ALU)
                    }
                    0x25 => {
                        self.set_gpr(rd, self.gpr(rs) | self.gpr(rt));
                        Ok(cycles::ALU)
                    }
                    0x26 => {
                        self.set_gpr(rd, self.gpr(rs) ^ self.gpr(rt));
                        Ok(cycles::ALU)
                    }
                    0x27 => {
                        self.set_gpr(rd, !(self.gpr(rs) | self.gpr(rt)));
                        Ok(cycles::ALU)
                    }
                    0x2A => {
                        let v = (self.gpr(rs) as i64) < (self.gpr(rt) as i64);
                        self.set_gpr(rd, if v { 1 } else { 0 });
                        Ok(cycles::ALU)
                    }
                    0x2B => {
                        let v = self.gpr(rs) < self.gpr(rt);
                        self.set_gpr(rd, if v { 1 } else { 0 });
                        Ok(cycles::ALU)
                    }
                    0x18 => {
                        let a = self.gpr(rs) as u32 as i32;
                        let b = self.gpr(rt) as u32 as i32;
                        let p = (a as i64).wrapping_mul(b as i64);
                        self.lo = ((p as u32) as i32 as i64) as u64;
                        self.hi = (((p >> 32) as u32) as i32 as i64) as u64;
                        Ok(cycles::MULT_LATENCY)
                    }
                    0x19 => {
                        let a = self.gpr(rs) as u32 as u64;
                        let b = self.gpr(rt) as u32 as u64;
                        let p = a.wrapping_mul(b);
                        self.lo = u64::from(p as u32);
                        self.hi = u64::from((p >> 32) as u32);
                        Ok(cycles::MULT_LATENCY)
                    }
                    0x1A => {
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
                        Ok(cycles::DIV_LATENCY)
                    }
                    0x1B => {
                        let s = self.gpr(rs) as u32;
                        let t = self.gpr(rt) as u32;
                        if t == 0 {
                            self.lo = 0;
                            self.hi = 0;
                        } else {
                            self.lo = u64::from(s / t);
                            self.hi = u64::from(s % t);
                        }
                        Ok(cycles::DIV_LATENCY)
                    }
                    0x10 => {
                        self.set_gpr(rd, self.hi);
                        Ok(cycles::ALU)
                    }
                    0x11 => {
                        self.hi = self.gpr(rs);
                        Ok(cycles::ALU)
                    }
                    0x12 => {
                        self.set_gpr(rd, self.lo);
                        Ok(cycles::ALU)
                    }
                    0x13 => {
                        self.lo = self.gpr(rs);
                        Ok(cycles::ALU)
                    }
                    0x0F => {
                        // SYNC — memory barrier; no bus effect in this model.
                        Ok(cycles::ALU)
                    }
                    0x0C => {
                        // SYSCALL — no exception delivery model yet; no-op for bring-up.
                        Ok(cycles::ALU)
                    }
                    0x0D => {
                        // BREAK — no debugger; no-op for bring-up.
                        Ok(cycles::ALU)
                    }
                    0x14 => {
                        // DSLLV
                        let s = self.gpr(rs) & 0x3F;
                        self.set_gpr(rd, self.gpr(rt) << s);
                        Ok(cycles::ALU)
                    }
                    0x16 => {
                        // DSRLV
                        let s = self.gpr(rs) & 0x3F;
                        self.set_gpr(rd, self.gpr(rt) >> s);
                        Ok(cycles::ALU)
                    }
                    0x17 => {
                        // DSRAV
                        let s = self.gpr(rs) & 0x3F;
                        self.set_gpr(rd, ((self.gpr(rt) as i64) >> s) as u64);
                        Ok(cycles::ALU)
                    }
                    0x1C => {
                        let a = self.gpr(rs) as i128;
                        let b = self.gpr(rt) as i128;
                        let p = a.wrapping_mul(b);
                        self.lo = p as u64;
                        self.hi = (p >> 64) as u64;
                        Ok(cycles::MULT_LATENCY)
                    }
                    0x1D => {
                        let a = self.gpr(rs) as u128;
                        let b = self.gpr(rt) as u128;
                        let p = a.wrapping_mul(b);
                        self.lo = p as u64;
                        self.hi = (p >> 64) as u64;
                        Ok(cycles::MULT_LATENCY)
                    }
                    0x1E => {
                        let s = self.gpr(rs) as i64;
                        let t = self.gpr(rt) as i64;
                        if t == 0 {
                            self.lo = 0;
                            self.hi = 0;
                        } else {
                            self.lo = (s / t) as u64;
                            self.hi = (s % t) as u64;
                        }
                        Ok(cycles::DIV_LATENCY)
                    }
                    0x1F => {
                        let s = self.gpr(rs);
                        let t = self.gpr(rt);
                        if t == 0 {
                            self.lo = 0;
                            self.hi = 0;
                        } else {
                            self.lo = s / t;
                            self.hi = s % t;
                        }
                        Ok(cycles::DIV_LATENCY)
                    }
                    0x2D => {
                        self.set_gpr(rd, self.gpr(rs).wrapping_add(self.gpr(rt)));
                        Ok(cycles::ALU)
                    }
                    0x2F => {
                        self.set_gpr(rd, self.gpr(rs).wrapping_sub(self.gpr(rt)));
                        Ok(cycles::ALU)
                    }
                    0x38 => {
                        self.set_gpr(rd, self.gpr(rt) << sa);
                        Ok(cycles::ALU)
                    }
                    0x3A => {
                        self.set_gpr(rd, self.gpr(rt) >> sa);
                        Ok(cycles::ALU)
                    }
                    0x3B => {
                        self.set_gpr(rd, ((self.gpr(rt) as i64) >> sa) as u64);
                        Ok(cycles::ALU)
                    }
                    0x3C => {
                        self.set_gpr(rd, self.gpr(rt) << (sa + 32));
                        Ok(cycles::ALU)
                    }
                    0x3E => {
                        self.set_gpr(rd, self.gpr(rt) >> (sa + 32));
                        Ok(cycles::ALU)
                    }
                    0x3F => {
                        self.set_gpr(rd, ((self.gpr(rt) as i64) >> (sa + 32)) as u64);
                        Ok(cycles::ALU)
                    }
                    _ => Err(CpuHalt::UnimplementedOpcode { pc, word }),
                }
            }
            0x08 | 0x09 | 0x18 | 0x19 => {
                let v = self.gpr(rs).wrapping_add(imm_s as u64);
                self.set_gpr(rt, v);
                Ok(cycles::ALU)
            }
            0x0C => {
                self.set_gpr(rt, self.gpr(rs) & u64::from(imm_u));
                Ok(cycles::ALU)
            }
            0x0D => {
                self.set_gpr(rt, self.gpr(rs) | u64::from(imm_u));
                Ok(cycles::ALU)
            }
            0x0E => {
                self.set_gpr(rt, self.gpr(rs) ^ u64::from(imm_u));
                Ok(cycles::ALU)
            }
            0x0A => {
                let v = (self.gpr(rs) as i64) < imm_s;
                self.set_gpr(rt, if v { 1 } else { 0 });
                Ok(cycles::ALU)
            }
            0x0B => {
                let v = self.gpr(rs) < (imm_u as u64);
                self.set_gpr(rt, if v { 1 } else { 0 });
                Ok(cycles::ALU)
            }
            0x0F => {
                self.set_gpr(rt, u64::from(imm_u) << 16);
                Ok(cycles::ALU)
            }
            0x33 => Ok(cycles::ALU),
            0x23 => {
                let addr = self.gpr(rs).wrapping_add(imm_s as u64);
                let v = self.load32(bus, addr)?;
                self.set_gpr(rt, i64::from(v as i32) as u64);
                Ok(cycles::MEM_ACCESS)
            }
            0x22 => {
                let eff = self.gpr(rs).wrapping_add(imm_s as u64);
                let al = eff & !3;
                let mem_word = self.load32(bus, al)?;
                let merged = self.merge_lwl(self.gpr(rt), mem_word, eff);
                self.set_gpr(rt, i64::from(merged as i32) as u64);
                Ok(cycles::MEM_ACCESS)
            }
            0x26 => {
                let eff = self.gpr(rs).wrapping_add(imm_s as u64);
                let al = eff & !3;
                let mem_word = self.load32(bus, al)?;
                let merged = self.merge_lwr(self.gpr(rt), mem_word, eff);
                self.set_gpr(rt, i64::from(merged as i32) as u64);
                Ok(cycles::MEM_ACCESS)
            }
            0x27 => {
                let addr = self.gpr(rs).wrapping_add(imm_s as u64);
                let v = self.load32(bus, addr)?;
                self.set_gpr(rt, u64::from(v));
                Ok(cycles::MEM_ACCESS)
            }
            0x24 => {
                let addr = self.gpr(rs).wrapping_add(imm_s as u64);
                let v = self.load8_unsigned(bus, addr)?;
                self.set_gpr(rt, u64::from(v));
                Ok(cycles::MEM_ACCESS)
            }
            0x20 => {
                let addr = self.gpr(rs).wrapping_add(imm_s as u64);
                let v = self.load8_signed(bus, addr)?;
                self.set_gpr(rt, i64::from(v as i32) as u64);
                Ok(cycles::MEM_ACCESS)
            }
            0x25 => {
                let addr = self.gpr(rs).wrapping_add(imm_s as u64);
                let v = self.load16_unsigned(bus, addr)?;
                self.set_gpr(rt, u64::from(v));
                Ok(cycles::MEM_ACCESS)
            }
            0x21 => {
                let addr = self.gpr(rs).wrapping_add(imm_s as u64);
                let v = self.load16_signed(bus, addr)?;
                self.set_gpr(rt, i64::from(v as i32) as u64);
                Ok(cycles::MEM_ACCESS)
            }
            0x2B => {
                let addr = self.gpr(rs).wrapping_add(imm_s as u64);
                self.store32(bus, addr, self.gpr(rt) as u32)?;
                Ok(cycles::MEM_ACCESS)
            }
            0x28 => {
                let addr = self.gpr(rs).wrapping_add(imm_s as u64);
                self.store8(bus, addr, self.gpr(rt) as u32)?;
                Ok(cycles::MEM_ACCESS)
            }
            0x29 => {
                let addr = self.gpr(rs).wrapping_add(imm_s as u64);
                self.store16(bus, addr, self.gpr(rt) as u32)?;
                Ok(cycles::MEM_ACCESS)
            }
            0x2A => {
                let eff = self.gpr(rs).wrapping_add(imm_s as u64);
                self.store_swl(bus, eff, self.gpr(rt))?;
                Ok(cycles::MEM_ACCESS)
            }
            0x2E => {
                let eff = self.gpr(rs).wrapping_add(imm_s as u64);
                self.store_swr(bus, eff, self.gpr(rt))?;
                Ok(cycles::MEM_ACCESS)
            }
            0x30 => {
                let addr = self.gpr(rs).wrapping_add(imm_s as u64);
                if addr & 3 != 0 {
                    return Err(CpuHalt::UnalignedAccess { vaddr: addr, width: 4 });
                }
                let v = self.load32(bus, addr)?;
                self.ll_bit = true;
                self.ll_addr = addr;
                self.set_gpr(rt, i64::from(v as i32) as u64);
                Ok(cycles::MEM_ACCESS)
            }
            0x38 => {
                let addr = self.gpr(rs).wrapping_add(imm_s as u64);
                if addr & 3 != 0 {
                    return Err(CpuHalt::UnalignedAccess { vaddr: addr, width: 4 });
                }
                let ok = self.ll_bit && self.ll_addr == addr;
                let val = self.gpr(rt) as u32;
                if ok {
                    self.store32(bus, addr, val)?;
                }
                self.ll_bit = false;
                self.set_gpr(rt, if ok { 1 } else { 0 });
                Ok(cycles::MEM_ACCESS)
            }
            0x34 => {
                let addr = self.gpr(rs).wrapping_add(imm_s as u64);
                if addr & 7 != 0 {
                    return Err(CpuHalt::UnalignedAccess { vaddr: addr, width: 8 });
                }
                let v = self.load64(bus, addr)?;
                self.ll_bit = true;
                self.ll_addr = addr;
                self.set_gpr(rt, v);
                Ok(cycles::MEM_ACCESS)
            }
            0x3C => {
                let addr = self.gpr(rs).wrapping_add(imm_s as u64);
                if addr & 7 != 0 {
                    return Err(CpuHalt::UnalignedAccess { vaddr: addr, width: 8 });
                }
                let ok = self.ll_bit && self.ll_addr == addr;
                let val = self.gpr(rt);
                if ok {
                    self.store64(bus, addr, val)?;
                }
                self.ll_bit = false;
                self.set_gpr(rt, if ok { 1 } else { 0 });
                Ok(cycles::MEM_ACCESS)
            }
            // CACHE — I-type; no L1/I$/D$ model yet (commercial ROMs use this constantly).
            0x2F => Ok(cycles::ALU),
            // MIPS III / 64-bit loads & stores (GPR)
            0x37 => {
                let addr = self.gpr(rs).wrapping_add(imm_s as u64);
                let v = self.load64(bus, addr)?;
                self.set_gpr(rt, v);
                Ok(cycles::MEM_ACCESS)
            }
            0x3F => {
                let addr = self.gpr(rs).wrapping_add(imm_s as u64);
                self.store64(bus, addr, self.gpr(rt))?;
                Ok(cycles::MEM_ACCESS)
            }
            // CP1 load / store (I-type: `rt` index is `ft`)
            0x31 => {
                let addr = self.gpr(rs).wrapping_add(imm_s as u64);
                let v = self.load32(bus, addr)?;
                self.cop1.set_fpr_u32(rt, v);
                Ok(cycles::MEM_ACCESS)
            }
            0x39 => {
                let addr = self.gpr(rs).wrapping_add(imm_s as u64);
                self.store32(bus, addr, self.cop1.fpr_u32(rt))?;
                Ok(cycles::MEM_ACCESS)
            }
            0x35 => {
                let addr = self.gpr(rs).wrapping_add(imm_s as u64);
                let v = self.load64(bus, addr)?;
                self.cop1.fpr[rt] = v;
                Ok(cycles::MEM_ACCESS)
            }
            0x3D => {
                let addr = self.gpr(rs).wrapping_add(imm_s as u64);
                self.store64(bus, addr, self.cop1.fpr[rt])?;
                Ok(cycles::MEM_ACCESS)
            }
            0x10 => {
                let sub = (word >> 21) & 0x1F;
                let funct = word & 0x3F;
                // `CO` class: TLB ops + `ERET` (no full TLB yet).
                if sub == 0x10 {
                    match funct {
                        0x01 | 0x02 | 0x05 | 0x06 | 0x08 => {
                            // TLBR, TLBWI, ?, TLBWR, TLBP
                            return Ok(cycles::ALU);
                        }
                        0x18 => {
                            self.pc = self.cop0.apply_eret();
                            self.ll_bit = false;
                            return Ok(cycles::ALU);
                        }
                        _ => return Err(CpuHalt::UnimplementedOpcode { pc, word }),
                    }
                }
                let rd_cop = ((word >> 11) & 0x1F) as u32;
                match sub {
                    0x00 => {
                        let v = self.cop0.read_32(rd_cop);
                        self.set_gpr(rt, i64::from(v as i32) as u64);
                        Ok(cycles::COP_MOVE)
                    }
                    0x04 => {
                        let v = self.gpr(rt) as u32;
                        self.cop0.write_32(rd_cop, v);
                        Ok(cycles::COP_MOVE)
                    }
                    _ => Err(CpuHalt::UnimplementedOpcode { pc, word }),
                }
            }
            0x11 => self.exec_cop1(pc, word, bus),
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
                // MFC1
                let v = self.cop1.fpr[fs] as u32;
                self.set_gpr(rt, i64::from(v as i32) as u64);
                Ok(cycles::COP_MOVE)
            }
            0x01 => {
                // DMFC1
                self.set_gpr(rt, self.cop1.fpr[fs]);
                Ok(cycles::COP_MOVE)
            }
            0x04 => {
                // MTC1
                self.cop1.fpr[fs] = u64::from(self.gpr(rt) as u32);
                Ok(cycles::COP_MOVE)
            }
            0x05 => {
                // DMTC1
                self.cop1.fpr[fs] = self.gpr(rt);
                Ok(cycles::COP_MOVE)
            }
            0x02 => {
                // CFC1
                let v = self.cop1.read_fcr(fs);
                self.set_gpr(rt, i64::from(v as i32) as u64);
                Ok(cycles::COP_MOVE)
            }
            0x06 => {
                // CTC1
                self.cop1.write_fcr(fs, self.gpr(rt) as u32);
                Ok(cycles::COP_MOVE)
            }
            _ => Err(CpuHalt::UnimplementedOpcode { pc, word }),
        }
    }

    /// COP1 floating-point compute (`CO` / bit 25 = 1). Covers common `.S` / `.D` ops and compares.
    fn exec_cop1_fp(&mut self, pc: u64, word: u32) -> Result<u64, CpuHalt> {
        let fmt5 = (word >> 21) & 0x1F;
        let ft = ((word >> 16) & 0x1F) as usize;
        let fs = ((word >> 11) & 0x1F) as usize;
        let fd = ((word >> 6) & 0x1F) as usize;
        let funct = word & 0x3F;

        // MIPS III: `fmt` in bits 24–21 with `CO`=1 → 0x10=.S, 0x11=.D, 0x14=.W, 0x15=.L
        match fmt5 {
            0x10 => {
                // .S
                match funct {
                    0x00 => {
                        let r = self.cop1.fpr_f32(fs) + self.cop1.fpr_f32(ft);
                        self.cop1.set_fpr_f32(fd, r);
                        Ok(cycles::COP_MOVE)
                    }
                    0x01 => {
                        let r = self.cop1.fpr_f32(fs) - self.cop1.fpr_f32(ft);
                        self.cop1.set_fpr_f32(fd, r);
                        Ok(cycles::COP_MOVE)
                    }
                    0x02 => {
                        let r = self.cop1.fpr_f32(fs) * self.cop1.fpr_f32(ft);
                        self.cop1.set_fpr_f32(fd, r);
                        Ok(cycles::COP_MOVE)
                    }
                    0x03 => {
                        let a = self.cop1.fpr_f32(fs);
                        let b = self.cop1.fpr_f32(ft);
                        let r = if b == 0.0 { 0.0 } else { a / b };
                        self.cop1.set_fpr_f32(fd, r);
                        Ok(cycles::COP_MOVE)
                    }
                    0x04 => {
                        let r = self.cop1.fpr_f32(fs).sqrt();
                        self.cop1.set_fpr_f32(fd, r);
                        Ok(cycles::COP_MOVE)
                    }
                    0x05 => {
                        let r = self.cop1.fpr_f32(fs).abs();
                        self.cop1.set_fpr_f32(fd, r);
                        Ok(cycles::COP_MOVE)
                    }
                    0x06 => {
                        let v = self.cop1.fpr_f32(fs);
                        self.cop1.set_fpr_f32(fd, v);
                        Ok(cycles::COP_MOVE)
                    }
                    0x07 => {
                        let r = -self.cop1.fpr_f32(fs);
                        self.cop1.set_fpr_f32(fd, r);
                        Ok(cycles::COP_MOVE)
                    }
                    0x0C => {
                        // ROUND.W.S — to word using FCSR RM
                        let rm = fcsr_rm(self.cop1.fcsr);
                        let i = f32_to_i32_rm(self.cop1.fpr_f32(fs), rm);
                        self.cop1.set_fpr_u32(fd, i as u32);
                        Ok(cycles::COP_MOVE)
                    }
                    0x0D => {
                        // TRUNC.W.S — toward zero
                        let i = f32_to_i32_trunc(self.cop1.fpr_f32(fs));
                        self.cop1.set_fpr_u32(fd, i as u32);
                        Ok(cycles::COP_MOVE)
                    }
                    0x0E => {
                        // CEIL.W.S
                        let i = f32_to_i32_ceil(self.cop1.fpr_f32(fs));
                        self.cop1.set_fpr_u32(fd, i as u32);
                        Ok(cycles::COP_MOVE)
                    }
                    0x0F => {
                        // FLOOR.W.S
                        let i = f32_to_i32_floor(self.cop1.fpr_f32(fs));
                        self.cop1.set_fpr_u32(fd, i as u32);
                        Ok(cycles::COP_MOVE)
                    }
                    0x21 => {
                        // CVT.D.S — single in `fs` → double in `fd`
                        let r = f64::from(self.cop1.fpr_f32(fs));
                        self.cop1.set_fpr_f64(fd, r);
                        Ok(cycles::COP_MOVE)
                    }
                    0x24 => {
                        // CVT.W.S — float → signed 32-bit word in FPR (uses FCSR RM)
                        let rm = fcsr_rm(self.cop1.fcsr);
                        let i = f32_to_i32_rm(self.cop1.fpr_f32(fs), rm);
                        self.cop1.set_fpr_u32(fd, i as u32);
                        Ok(cycles::COP_MOVE)
                    }
                    0x25 => {
                        // CVT.L.S — single → signed 64-bit integer in FPR (uses FCSR RM)
                        let rm = fcsr_rm(self.cop1.fcsr);
                        let i = f32_to_i64_rm(self.cop1.fpr_f32(fs), rm);
                        self.cop1.fpr[fd] = i as u64;
                        Ok(cycles::COP_MOVE)
                    }
                    f @ 0x30..=0x3F => {
                        self.cop1.set_cc0(cond_f32(
                            self.cop1.fpr_f32(fs),
                            self.cop1.fpr_f32(ft),
                            f,
                        ));
                        Ok(cycles::ALU)
                    }
                    _ => Err(CpuHalt::UnimplementedOpcode { pc, word }),
                }
            }
            0x11 => {
                // .D
                match funct {
                    0x00 => {
                        let r = self.cop1.fpr_f64(fs) + self.cop1.fpr_f64(ft);
                        self.cop1.set_fpr_f64(fd, r);
                        Ok(cycles::COP_MOVE)
                    }
                    0x01 => {
                        let r = self.cop1.fpr_f64(fs) - self.cop1.fpr_f64(ft);
                        self.cop1.set_fpr_f64(fd, r);
                        Ok(cycles::COP_MOVE)
                    }
                    0x02 => {
                        let r = self.cop1.fpr_f64(fs) * self.cop1.fpr_f64(ft);
                        self.cop1.set_fpr_f64(fd, r);
                        Ok(cycles::COP_MOVE)
                    }
                    0x03 => {
                        let a = self.cop1.fpr_f64(fs);
                        let b = self.cop1.fpr_f64(ft);
                        let r = if b == 0.0 { 0.0 } else { a / b };
                        self.cop1.set_fpr_f64(fd, r);
                        Ok(cycles::COP_MOVE)
                    }
                    0x04 => {
                        let r = self.cop1.fpr_f64(fs).sqrt();
                        self.cop1.set_fpr_f64(fd, r);
                        Ok(cycles::COP_MOVE)
                    }
                    0x05 => {
                        let r = self.cop1.fpr_f64(fs).abs();
                        self.cop1.set_fpr_f64(fd, r);
                        Ok(cycles::COP_MOVE)
                    }
                    0x06 => {
                        let v = self.cop1.fpr_f64(fs);
                        self.cop1.set_fpr_f64(fd, v);
                        Ok(cycles::COP_MOVE)
                    }
                    0x07 => {
                        let r = -self.cop1.fpr_f64(fs);
                        self.cop1.set_fpr_f64(fd, r);
                        Ok(cycles::COP_MOVE)
                    }
                    0x0C => {
                        let rm = fcsr_rm(self.cop1.fcsr);
                        let i = f64_to_i32_rm(self.cop1.fpr_f64(fs), rm);
                        self.cop1.set_fpr_u32(fd, i as u32);
                        Ok(cycles::COP_MOVE)
                    }
                    0x0D => {
                        let i = f64_to_i32_trunc(self.cop1.fpr_f64(fs));
                        self.cop1.set_fpr_u32(fd, i as u32);
                        Ok(cycles::COP_MOVE)
                    }
                    0x0E => {
                        let i = f64_to_i32_ceil(self.cop1.fpr_f64(fs));
                        self.cop1.set_fpr_u32(fd, i as u32);
                        Ok(cycles::COP_MOVE)
                    }
                    0x0F => {
                        let i = f64_to_i32_floor(self.cop1.fpr_f64(fs));
                        self.cop1.set_fpr_u32(fd, i as u32);
                        Ok(cycles::COP_MOVE)
                    }
                    0x20 => {
                        // CVT.S.D
                        let r = self.cop1.fpr_f64(fs) as f32;
                        self.cop1.set_fpr_f32(fd, r);
                        Ok(cycles::COP_MOVE)
                    }
                    0x24 => {
                        let rm = fcsr_rm(self.cop1.fcsr);
                        let w = f64_to_i32_rm(self.cop1.fpr_f64(fs), rm);
                        self.cop1.set_fpr_u32(fd, w as u32);
                        Ok(cycles::COP_MOVE)
                    }
                    0x25 => {
                        let rm = fcsr_rm(self.cop1.fcsr);
                        let i = f64_to_i64_rm(self.cop1.fpr_f64(fs), rm);
                        self.cop1.fpr[fd] = i as u64;
                        Ok(cycles::COP_MOVE)
                    }
                    f @ 0x30..=0x3F => {
                        self.cop1.set_cc0(cond_f64(
                            self.cop1.fpr_f64(fs),
                            self.cop1.fpr_f64(ft),
                            f,
                        ));
                        Ok(cycles::ALU)
                    }
                    _ => Err(CpuHalt::UnimplementedOpcode { pc, word }),
                }
            }
            0x14 => {
                // .W (fixed-point / integer in FPR)
                match funct {
                    0x20 => {
                        // CVT.S.W — signed word in `fs` → single in `fd`
                        let i = self.cop1.fpr_u32(fs) as i32;
                        self.cop1.set_fpr_f32(fd, i as f32);
                        Ok(cycles::COP_MOVE)
                    }
                    0x21 => {
                        // CVT.D.W
                        let i = self.cop1.fpr_u32(fs) as i32;
                        self.cop1.set_fpr_f64(fd, i as f64);
                        Ok(cycles::COP_MOVE)
                    }
                    0x25 => {
                        // CVT.L.W — 32-bit fixed word in FPR → signed 64-bit in FPR
                        let i = self.cop1.fpr_u32(fs) as i32 as i64;
                        self.cop1.fpr[fd] = i as u64;
                        Ok(cycles::COP_MOVE)
                    }
                    _ => Err(CpuHalt::UnimplementedOpcode { pc, word }),
                }
            }
            0x15 => {
                // `.L` — 64-bit integer in FPR (MIPS fmt L)
                match funct {
                    0x20 => {
                        // CVT.S.L
                        let v = self.cop1.fpr[fs] as i64;
                        self.cop1.set_fpr_f32(fd, v as f32);
                        Ok(cycles::COP_MOVE)
                    }
                    0x21 => {
                        // CVT.D.L
                        let v = self.cop1.fpr[fs] as i64;
                        self.cop1.set_fpr_f64(fd, v as f64);
                        Ok(cycles::COP_MOVE)
                    }
                    _ => Err(CpuHalt::UnimplementedOpcode { pc, word }),
                }
            }
            _ => Err(CpuHalt::UnimplementedOpcode { pc, word }),
        }
    }

    /// `BC1F` / `BC1T`: branch on floating-point condition (uses FCSR CC bits).
    fn exec_bc1(
        &mut self,
        pc: u64,
        word: u32,
        bus: &mut impl Bus,
    ) -> Result<u64, CpuHalt> {
        let imm = (((word & 0xFFFF) as i16) as i64) << 2;
        let tf = (word >> 16) & 1 != 0;
        let nd = (word >> 17) & 1 != 0;
        let cc = (word >> 18) & 0x7;
        let cond = (self.cop1.fcsr >> (23 + cc)) & 1;
        let take = if tf { cond != 0 } else { cond == 0 };

        if nd && !take {
            self.pc = pc.wrapping_add(8);
            return Ok(cycles::BRANCH);
        }

        let delay_pc = pc.wrapping_add(4);
        let delay_word = self.fetch32(bus, delay_pc)?;
        let mut cycles = cycles::BRANCH;
        cycles += self.exec_non_branch(delay_pc, delay_word, bus)?;

        self.pc = if take {
            pc.wrapping_add(imm as u64).wrapping_add(4)
        } else {
            pc.wrapping_add(8)
        };
        Ok(cycles)
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
        cpu.reset(0x8000_0000);
        let mut mem = PhysicalMemory::new(1024 * 1024);
        // lui t0, 0x1234
        write_be32(&mut mem, 0, 0x3C081234);
        // addiu t1, t0, 0x5678  (encoding: rs=t0=8, rt=t1=9)
        write_be32(&mut mem, 4, 0x2509_5678);

        assert_eq!(cpu.step(&mut mem, false).unwrap(), cycles::ALU);
        assert_eq!(cpu.pc, 0x8000_0004);
        assert_eq!(cpu.regs[8], 0x1234_0000);

        assert_eq!(cpu.step(&mut mem, false).unwrap(), cycles::ALU);
        assert_eq!(cpu.pc, 0x8000_0008);
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

        cpu.pc = 0x8000_0000;
        cpu.regs[8] = 0;
        let c = cpu.step(&mut mem, false).unwrap();
        assert!(c > cycles::ALU);
        assert_eq!(cpu.regs[8], 1, "delay slot must run");
        assert_eq!(cpu.pc, 0x8000_0008);
    }

    #[test]
    fn jump_and_link_sets_ra() {
        let mut cpu = R4300i::new();
        let mut mem = PhysicalMemory::new(1024 * 1024);
        // j 0x80000020  (word index: target byte 0x80000020)
        // Encoding: upper 4 bits from PC | imm26<<2
        write_be32(&mut mem, 0, 0x0C00_0008);
        // delay: nop
        write_be32(&mut mem, 4, 0x0000_0000);
        cpu.pc = 0x8000_0000;
        cpu.step(&mut mem, false).unwrap();
        assert_eq!(cpu.pc, 0x8000_0020);
        assert_eq!(cpu.regs[31], 0x8000_0008);
    }

    #[test]
    fn eret_restores_epc_and_clears_exl() {
        use crate::cpu::cop0::{STATUS_ERL, STATUS_EXL};

        let mut cpu = R4300i::new();
        cpu.reset(0xFFFF_FFFF_8000_0180);
        cpu.cop0.epc = 0xFFFF_FFFF_8000_4000;
        // Avoid default `ERL` so `ERET` uses `EPC` (not `ErrorEPC`).
        cpu.cop0.status = (cpu.cop0.status & !STATUS_ERL) | STATUS_EXL;

        let mut mem = PhysicalMemory::new(1024 * 1024);
        write_be32(&mut mem, 0x180, 0x4200_0018);

        assert_eq!(cpu.step(&mut mem, false).unwrap(), crate::cycles::ALU);
        assert_eq!(cpu.pc, 0xFFFF_FFFF_8000_4000);
        assert!((cpu.cop0.status & STATUS_EXL) == 0);
    }

    #[test]
    fn syscall_and_break_special_no_halt() {
        let mut cpu = R4300i::new();
        cpu.reset(0x8000_0000);
        let mut mem = PhysicalMemory::new(1024 * 1024);
        // SPECIAL + SYSCALL (funct 0x0C)
        write_be32(&mut mem, 0, 0x0000_000C);
        assert_eq!(cpu.step(&mut mem, false).unwrap(), cycles::ALU);
        assert_eq!(cpu.pc, 0x8000_0004);
        // SPECIAL + BREAK (funct 0x0D)
        write_be32(&mut mem, 4, 0x0000_000D);
        assert_eq!(cpu.step(&mut mem, false).unwrap(), cycles::ALU);
        assert_eq!(cpu.pc, 0x8000_0008);
    }
}
