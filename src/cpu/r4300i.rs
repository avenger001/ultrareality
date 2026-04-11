use super::cop0::Cop0;
use crate::bus::{virt_to_phys_rdram, Bus};
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
pub struct R4300i {
    pub regs: [u64; 32],
    pub hi: u64,
    pub lo: u64,
    pub pc: u64,
    pub cop0: Cop0,
    pub ll_bit: bool,
    pub ll_addr: u32,
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

    fn fetch32(&self, bus: &mut impl Bus, rdram_size: usize, vaddr: u64) -> Result<u32, CpuHalt> {
        if vaddr & 3 != 0 {
            return Err(CpuHalt::UnalignedFetch { vaddr });
        }
        let p = virt_to_phys_rdram(vaddr, rdram_size).ok_or(CpuHalt::UnmappedAddress { vaddr })?;
        bus.read_u32(p).ok_or(CpuHalt::UnmappedAddress { vaddr })
    }

    fn load32(&mut self, bus: &mut impl Bus, rdram_size: usize, vaddr: u64) -> Result<u32, CpuHalt> {
        if vaddr & 3 != 0 {
            return Err(CpuHalt::UnalignedAccess { vaddr, width: 4 });
        }
        let p = virt_to_phys_rdram(vaddr, rdram_size).ok_or(CpuHalt::UnmappedAddress { vaddr })?;
        bus.read_u32(p).ok_or(CpuHalt::UnmappedAddress { vaddr })
    }

    fn store32(&mut self, bus: &mut impl Bus, rdram_size: usize, vaddr: u64, value: u32) -> Result<(), CpuHalt> {
        if vaddr & 3 != 0 {
            return Err(CpuHalt::UnalignedAccess { vaddr, width: 4 });
        }
        let p = virt_to_phys_rdram(vaddr, rdram_size).ok_or(CpuHalt::UnmappedAddress { vaddr })?;
        bus.write_u32(p, value);
        Ok(())
    }

    fn load16_signed(&mut self, bus: &mut impl Bus, rdram_size: usize, vaddr: u64) -> Result<u32, CpuHalt> {
        if vaddr & 1 != 0 {
            return Err(CpuHalt::UnalignedAccess { vaddr, width: 2 });
        }
        let p = virt_to_phys_rdram(vaddr, rdram_size).ok_or(CpuHalt::UnmappedAddress { vaddr })?;
        let hi = bus.read_u8(p).ok_or(CpuHalt::UnmappedAddress { vaddr })?;
        let lo = bus.read_u8(p.wrapping_add(1)).ok_or(CpuHalt::UnmappedAddress { vaddr })?;
        let h = u16::from_be_bytes([hi, lo]);
        Ok(i32::from(h as i16) as u32)
    }

    fn load16_unsigned(&mut self, bus: &mut impl Bus, rdram_size: usize, vaddr: u64) -> Result<u32, CpuHalt> {
        if vaddr & 1 != 0 {
            return Err(CpuHalt::UnalignedAccess { vaddr, width: 2 });
        }
        let p = virt_to_phys_rdram(vaddr, rdram_size).ok_or(CpuHalt::UnmappedAddress { vaddr })?;
        let hi = bus.read_u8(p).ok_or(CpuHalt::UnmappedAddress { vaddr })?;
        let lo = bus.read_u8(p.wrapping_add(1)).ok_or(CpuHalt::UnmappedAddress { vaddr })?;
        Ok(u32::from(u16::from_be_bytes([hi, lo])))
    }

    fn load8_signed(&mut self, bus: &mut impl Bus, rdram_size: usize, vaddr: u64) -> Result<u32, CpuHalt> {
        let p = virt_to_phys_rdram(vaddr, rdram_size).ok_or(CpuHalt::UnmappedAddress { vaddr })?;
        let b = bus.read_u8(p).ok_or(CpuHalt::UnmappedAddress { vaddr })?;
        Ok(i32::from(b as i8) as u32)
    }

    fn load8_unsigned(&mut self, bus: &mut impl Bus, rdram_size: usize, vaddr: u64) -> Result<u32, CpuHalt> {
        let p = virt_to_phys_rdram(vaddr, rdram_size).ok_or(CpuHalt::UnmappedAddress { vaddr })?;
        let b = bus.read_u8(p).ok_or(CpuHalt::UnmappedAddress { vaddr })?;
        Ok(u32::from(b))
    }

    fn store16(&mut self, bus: &mut impl Bus, rdram_size: usize, vaddr: u64, value: u32) -> Result<(), CpuHalt> {
        if vaddr & 1 != 0 {
            return Err(CpuHalt::UnalignedAccess { vaddr, width: 2 });
        }
        let p = virt_to_phys_rdram(vaddr, rdram_size).ok_or(CpuHalt::UnmappedAddress { vaddr })?;
        let half = (value & 0xFFFF) as u16;
        let [a, b] = half.to_be_bytes();
        bus.write_u8(p, a);
        bus.write_u8(p.wrapping_add(1), b);
        Ok(())
    }

    fn store8(&mut self, bus: &mut impl Bus, rdram_size: usize, vaddr: u64, value: u32) -> Result<(), CpuHalt> {
        let p = virt_to_phys_rdram(vaddr, rdram_size).ok_or(CpuHalt::UnmappedAddress { vaddr })?;
        bus.write_u8(p, (value & 0xFF) as u8);
        Ok(())
    }

    /// Execute one **architectural** instruction (including its delay slot for
    /// branches/jumps). Returns CPU cycles consumed for this retirement.
    pub fn step(&mut self, bus: &mut impl Bus, rdram_size: usize) -> Result<u64, CpuHalt> {
        let pc = self.pc;
        let word = self.fetch32(bus, rdram_size, pc)?;
        let op = word >> 26;

        match op {
            0 => self.exec_special(pc, word, bus, rdram_size),
            1 => self.exec_regimm(pc, word, bus, rdram_size),
            2 | 3 => self.exec_j_type(pc, word, bus, rdram_size),
            _ => self.exec_common_i_type(pc, word, bus, rdram_size, op),
        }
    }

    fn exec_j_type(
        &mut self,
        pc: u64,
        word: u32,
        bus: &mut impl Bus,
        rdram_size: usize,
    ) -> Result<u64, CpuHalt> {
        let op = word >> 26;
        let target = (pc & 0xF000_0000) | u64::from(word & 0x03FF_FFFF) << 2;
        let mut cycles = cycles::BRANCH;

        if op == 3 {
            // JAL
            self.set_gpr(31, pc.wrapping_add(8));
        }

        let delay_pc = pc.wrapping_add(4);
        let delay_word = self.fetch32(bus, rdram_size, delay_pc)?;
        cycles += self.exec_non_branch(delay_pc, delay_word, bus, rdram_size)?;
        self.pc = target;
        Ok(cycles)
    }

    fn exec_regimm(
        &mut self,
        pc: u64,
        word: u32,
        bus: &mut impl Bus,
        rdram_size: usize,
    ) -> Result<u64, CpuHalt> {
        let rt = ((word >> 16) & 0x1F) as usize;
        let rs = ((word >> 21) & 0x1F) as usize;
        let imm = (word & 0xFFFF) as i16 as i64;
        let mut cycles = cycles::BRANCH;

        let take = match rt {
            0x00 => (self.gpr(rs) as i64) < 0,
            0x01 => (self.gpr(rs) as i64) >= 0,
            0x10 => self.gpr(rs) == 0,
            0x11 => self.gpr(rs) != 0,
            _ => {
                return Err(CpuHalt::UnimplementedOpcode { pc, word });
            }
        };

        let delay_pc = pc.wrapping_add(4);
        let delay_word = self.fetch32(bus, rdram_size, delay_pc)?;
        cycles += self.exec_non_branch(delay_pc, delay_word, bus, rdram_size)?;

        self.pc = if take {
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
        rdram_size: usize,
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
                let delay_word = self.fetch32(bus, rdram_size, delay_pc)?;
                cycles += self.exec_non_branch(delay_pc, delay_word, bus, rdram_size)?;
                self.pc = target;
                Ok(cycles)
            }
            0x09 => {
                // JALR
                let target = self.gpr(rs);
                self.set_gpr(rd, pc.wrapping_add(8));
                let mut cycles = cycles::BRANCH;
                let delay_pc = pc.wrapping_add(4);
                let delay_word = self.fetch32(bus, rdram_size, delay_pc)?;
                cycles += self.exec_non_branch(delay_pc, delay_word, bus, rdram_size)?;
                self.pc = target;
                Ok(cycles)
            }
            _ => {
                let c = self.exec_non_branch(pc, word, bus, rdram_size)?;
                self.pc = pc.wrapping_add(4);
                Ok(c)
            }
        }
    }

    fn exec_common_i_type(
        &mut self,
        pc: u64,
        word: u32,
        bus: &mut impl Bus,
        rdram_size: usize,
        op: u32,
    ) -> Result<u64, CpuHalt> {
        match op {
            0x04 | 0x05 | 0x06 | 0x07 => self.exec_branch(pc, word, bus, rdram_size, op),
            _ => {
                let c = self.exec_non_branch(pc, word, bus, rdram_size)?;
                self.pc = pc.wrapping_add(4);
                Ok(c)
            }
        }
    }

    fn exec_branch(
        &mut self,
        pc: u64,
        word: u32,
        bus: &mut impl Bus,
        rdram_size: usize,
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
        let delay_word = self.fetch32(bus, rdram_size, delay_pc)?;
        cycles += self.exec_non_branch(delay_pc, delay_word, bus, rdram_size)?;

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
        rdram_size: usize,
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
                    _ => Err(CpuHalt::UnimplementedOpcode { pc, word }),
                }
            }
            0x08 | 0x09 => {
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
            0x23 => {
                let addr = self.gpr(rs).wrapping_add(imm_s as u64);
                let v = self.load32(bus, rdram_size, addr)?;
                self.set_gpr(rt, i64::from(v as i32) as u64);
                Ok(cycles::MEM_ACCESS)
            }
            0x24 => {
                let addr = self.gpr(rs).wrapping_add(imm_s as u64);
                let v = self.load8_unsigned(bus, rdram_size, addr)?;
                self.set_gpr(rt, u64::from(v));
                Ok(cycles::MEM_ACCESS)
            }
            0x20 => {
                let addr = self.gpr(rs).wrapping_add(imm_s as u64);
                let v = self.load8_signed(bus, rdram_size, addr)?;
                self.set_gpr(rt, i64::from(v as i32) as u64);
                Ok(cycles::MEM_ACCESS)
            }
            0x25 => {
                let addr = self.gpr(rs).wrapping_add(imm_s as u64);
                let v = self.load16_unsigned(bus, rdram_size, addr)?;
                self.set_gpr(rt, u64::from(v));
                Ok(cycles::MEM_ACCESS)
            }
            0x21 => {
                let addr = self.gpr(rs).wrapping_add(imm_s as u64);
                let v = self.load16_signed(bus, rdram_size, addr)?;
                self.set_gpr(rt, i64::from(v as i32) as u64);
                Ok(cycles::MEM_ACCESS)
            }
            0x2B => {
                let addr = self.gpr(rs).wrapping_add(imm_s as u64);
                self.store32(bus, rdram_size, addr, self.gpr(rt) as u32)?;
                Ok(cycles::MEM_ACCESS)
            }
            0x28 => {
                let addr = self.gpr(rs).wrapping_add(imm_s as u64);
                self.store8(bus, rdram_size, addr, self.gpr(rt) as u32)?;
                Ok(cycles::MEM_ACCESS)
            }
            0x29 => {
                let addr = self.gpr(rs).wrapping_add(imm_s as u64);
                self.store16(bus, rdram_size, addr, self.gpr(rt) as u32)?;
                Ok(cycles::MEM_ACCESS)
            }
            0x10 => {
                let sub = (word >> 21) & 0x1F;
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
            _ => Err(CpuHalt::UnimplementedOpcode { pc, word }),
        }
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
        let sz = mem.data.len();
        // lui t0, 0x1234
        write_be32(&mut mem, 0, 0x3C081234);
        // addiu t1, t0, 0x5678  (encoding: rs=t0=8, rt=t1=9)
        write_be32(&mut mem, 4, 0x2509_5678);

        assert_eq!(cpu.step(&mut mem, sz).unwrap(), cycles::ALU);
        assert_eq!(cpu.pc, 0x8000_0004);
        assert_eq!(cpu.regs[8], 0x1234_0000);

        assert_eq!(cpu.step(&mut mem, sz).unwrap(), cycles::ALU);
        assert_eq!(cpu.pc, 0x8000_0008);
        assert_eq!(cpu.regs[9], 0x1234_5678);
    }

    #[test]
    fn branch_delay_slot_skipped_when_not_taken() {
        let mut cpu = R4300i::new();
        let mut mem = PhysicalMemory::new(1024 * 1024);
        let sz = mem.data.len();
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
        let c = cpu.step(&mut mem, sz).unwrap();
        assert!(c > cycles::ALU);
        assert_eq!(cpu.regs[8], 1, "delay slot must run");
        assert_eq!(cpu.pc, 0x8000_0008);
    }

    #[test]
    fn jump_and_link_sets_ra() {
        let mut cpu = R4300i::new();
        let mut mem = PhysicalMemory::new(1024 * 1024);
        let sz = mem.data.len();
        // j 0x80000020  (word index: target byte 0x80000020)
        // Encoding: upper 4 bits from PC | imm26<<2
        write_be32(&mut mem, 0, 0x0C00_0008);
        // delay: nop
        write_be32(&mut mem, 4, 0x0000_0000);
        cpu.pc = 0x8000_0000;
        cpu.step(&mut mem, sz).unwrap();
        assert_eq!(cpu.pc, 0x8000_0020);
        assert_eq!(cpu.regs[31], 0x8000_0008);
    }
}
