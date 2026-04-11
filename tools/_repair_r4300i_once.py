"""One-shot repair: strip junk between merge_lwr and step(), insert clean helpers + opcode patches."""
from pathlib import Path

p = Path(r"F:\Projects\UltraReality\src\cpu\r4300i.rs")
lines = p.read_text(encoding="utf-8").splitlines(keepends=True)

def head_through_merge_lwr(src: list[str]) -> list[str]:
    """First lines through the closing `}` of `fn merge_lwr` (do not cut inside `match`)."""
    start = next(
        i
        for i, l in enumerate(src)
        if "fn merge_lwr" in l and "cur_rt" in l
    )
    depth = 0
    for i in range(start, min(start + 40, len(src))):
        depth += src[i].count("{") - src[i].count("}")
        if i > start and depth == 0:
            return src[: i + 1]
    raise SystemExit("merge_lwr closing brace not found")


head_lines = head_through_merge_lwr(lines)
head = "".join(head_lines)
tail_idx = next(i for i, l in enumerate(lines) if l.startswith("    /// Execute one **architectural**"))
tail = "".join(lines[tail_idx:])
mid = r"""    /// Store word left: high-order bytes of rt from eff through end of aligned word.
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

"""
text = head + mid + tail
if "use super::cop1::Cop1;" in text and "cond_f32" not in text.split("\n")[1]:
    text = text.replace("use super::cop1::Cop1;", "use super::cop1::{cond_f32, cond_f64, Cop1};", 1)
text = text.replace("    pub ll_addr: u32,", "    pub ll_addr: u64,")

# Opcode patches (same as prior session)
old = """            0x23 => {
                let addr = self.gpr(rs).wrapping_add(imm_s as u64);
                let v = self.load32(bus, addr)?;
                self.set_gpr(rt, i64::from(v as i32) as u64);
                Ok(cycles::MEM_ACCESS)
            }
            0x24 => {"""
new = """            0x23 => {
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
            0x24 => {"""
if old not in text:
    raise SystemExit("LW block not found")
text = text.replace(old, new, 1)

old2 = """            0x29 => {
                let addr = self.gpr(rs).wrapping_add(imm_s as u64);
                self.store16(bus, addr, self.gpr(rt) as u32)?;
                Ok(cycles::MEM_ACCESS)
            }
            // CACHE"""
new2 = """            0x29 => {
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
            // CACHE"""
if old2 not in text:
    raise SystemExit("store half / CACHE not found")
text = text.replace(old2, new2, 1)

old3 = """            // CACHE — I-type; no L1/I$/D$ model yet (commercial ROMs use this constantly).
            0x2F => Ok(cycles::ALU),"""
if old3 not in text:
    old3 = old3.replace("—", "\u2014")
new3 = """            0x30 => {
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
            0x2F => Ok(cycles::ALU),"""
if old3 not in text:
    raise SystemExit("CACHE comment block not found")
text = text.replace(old3, new3, 1)

old4 = """            0x08 | 0x09 => {
                let v = self.gpr(rs).wrapping_add(imm_s as u64);
                self.set_gpr(rt, v);
                Ok(cycles::ALU)
            }"""
new4 = """            0x08 | 0x09 | 0x18 | 0x19 => {
                let v = self.gpr(rs).wrapping_add(imm_s as u64);
                self.set_gpr(rt, v);
                Ok(cycles::ALU)
            }"""
if old4 not in text:
    raise SystemExit("ADDI block not found")
text = text.replace(old4, new4, 1)

old5 = """            0x0F => {
                self.set_gpr(rt, u64::from(imm_u) << 16);
                Ok(cycles::ALU)
            }
            0x23 => {"""
new5 = """            0x0F => {
                self.set_gpr(rt, u64::from(imm_u) << 16);
                Ok(cycles::ALU)
            }
            0x33 => Ok(cycles::ALU),
            0x23 => {"""
if old5 not in text:
    raise SystemExit("LUI block not found")
text = text.replace(old5, new5, 1)

p.write_text(text, encoding="utf-8")
print("repaired", len(text.splitlines()), "lines")
