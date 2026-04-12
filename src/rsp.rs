//! **Phase 2:** Reality Signal Processor — scalar subset + COP2 stub.
//!
//! Execution state lives on [`crate::bus::SystemBus`]: `rsp_pc`, `rsp_scalar_regs`, `sp_halted`.
//! Vector (COP2) ops advance PC but do not emulate the VU yet.

use crate::bus::SystemBus;
use crate::mi::MI_INTR_SP;

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

/// Fetch one big-endian word from RSP IMEM (`pc` is word index offset `0..0x1000`).
#[inline]
fn imem_load_word(imem: &[u8; 4096], pc: usize) -> u32 {
    let i = pc & 0xFFC;
    u32::from_be_bytes(imem[i..i + 4].try_into().unwrap())
}

#[inline]
fn dmem_load_word(dmem: &[u8; 4096], addr: usize) -> u32 {
    let a = addr & 0xFFF;
    u32::from_be_bytes(dmem[a..a + 4].try_into().unwrap())
}

#[inline]
fn dmem_store_word(dmem: &mut [u8; 4096], addr: usize, v: u32) {
    let a = addr & 0xFFF;
    dmem[a..a + 4].copy_from_slice(&v.to_be_bytes());
}

/// Execute one RSP scalar instruction. Returns **0** if the RSP is halted.
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

    let iw = imem_load_word(&bus.rsp_imem, pc);
    let op = iw >> 26;
    let rs = ((iw >> 21) & 31) as usize;
    let rt = ((iw >> 16) & 31) as usize;
    let rd = ((iw >> 11) & 31) as usize;
    let sa = (iw >> 6) & 31;
    let funct = iw & 0x3F;
    let simm = (iw & 0xFFFF) as i16 as i32 as u32;
    let immu = iw & 0xFFFF;

    let mut next = pc as u32 + 4;

    match op {
        0 => match funct {
            0x00 => {
                // SLL — if not all-zero, shift; `0x00000000` is NOP.
                if iw != 0 {
                    let v = gpr_load(&bus.rsp_scalar_regs, rt).wrapping_shl(sa);
                    gpr_store(&mut bus.rsp_scalar_regs, rd, v);
                }
            }
            0x08 => {
                // JR
                next = gpr_load(&bus.rsp_scalar_regs, rs) & 0xFFC;
            }
            0x0D => {
                // BREAK
                bus.sp_broke = true;
                bus.sp_halted = true;
                bus.mi.raise(MI_INTR_SP);
            }
            _ => {}
        },
        2 => {
            // J — target 26 bits, word address in IMEM window
            let t = iw & 0x03FF_FFFF;
            next = ((t << 2) & 0xFFC) as u32;
        }
        5 => {
            // BNE — branch if rs != rt
            let off = ((iw & 0xFFFF) as i16 as i32 as i32) << 2;
            if gpr_load(&bus.rsp_scalar_regs, rs) != gpr_load(&bus.rsp_scalar_regs, rt) {
                next = (pc as i32 + 4 + off) as u32;
                next &= 0xFFC;
            }
        }
        9 => {
            // ADDIU
            let v = gpr_load(&bus.rsp_scalar_regs, rs).wrapping_add(simm);
            gpr_store(&mut bus.rsp_scalar_regs, rt, v);
        }
        13 => {
            // ORI
            let v = gpr_load(&bus.rsp_scalar_regs, rs) | immu;
            gpr_store(&mut bus.rsp_scalar_regs, rt, v);
        }
        15 => {
            // LUI
            gpr_store(&mut bus.rsp_scalar_regs, rt, immu << 16);
        }
        35 => {
            // LW
            let base = gpr_load(&bus.rsp_scalar_regs, rs).wrapping_add(simm);
            let v = rsp_load_flat(bus, base);
            gpr_store(&mut bus.rsp_scalar_regs, rt, v);
        }
        43 => {
            // SW
            let base = gpr_load(&bus.rsp_scalar_regs, rs).wrapping_add(simm);
            rsp_store_flat(bus, base, gpr_load(&bus.rsp_scalar_regs, rt));
        }
        18 => {
            // COP2 — vector unit; not emulated: consume cycles and advance PC.
        }
        _ => {}
    }

    bus.rsp_pc = next & 0xFFC;
    RSP_CYCLES_PER_INSTR
}

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

/// Run RSP work for the same RCP quantum as CPU/DMA (coarse).
pub fn run_for_rcp_quantum(bus: &mut SystemBus, rcp_cycles: u64) {
    if bus.sp_halted || rcp_cycles == 0 {
        return;
    }
    // Budget scales with master quantum; cap avoids starving the CPU test harness.
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

    #[test]
    fn nop_then_break_halts() {
        let mut bus = SystemBus::with_rdram_size(1024 * 1024);
        bus.sp_halted = false;
        bus.rsp_pc = 0;
        write_imem(&mut bus, 0, 0x0000_0000);
        write_imem(&mut bus, 4, 0x0000_000D);
        assert_eq!(step_instruction(&mut bus), RSP_CYCLES_PER_INSTR);
        assert_eq!(bus.rsp_pc, 4);
        assert!(!bus.sp_halted);
        assert_eq!(step_instruction(&mut bus), RSP_CYCLES_PER_INSTR);
        assert!(bus.sp_halted);
        assert!(bus.sp_broke);
    }

    #[test]
    fn addiu_lw_sw_round_trip() {
        let mut bus = SystemBus::with_rdram_size(1024 * 1024);
        bus.sp_halted = false;
        bus.rsp_pc = 0;
        // ADDIU r1, r0, 0x40  → r1 = 0x40
        write_imem(&mut bus, 0, 0x2401_0040);
        // SW r1, 0(r0) — store 0x40 at DMEM 0
        write_imem(&mut bus, 4, 0xAC01_0000);
        // LW r2, 0(r0)
        write_imem(&mut bus, 8, 0x8C02_0000);
        step_instruction(&mut bus);
        step_instruction(&mut bus);
        step_instruction(&mut bus);
        assert_eq!(bus.rsp_scalar_regs[2], 0x40);
    }

    #[test]
    fn addiu_to_r0_discards_result() {
        let mut bus = SystemBus::with_rdram_size(1024 * 1024);
        bus.sp_halted = false;
        bus.rsp_pc = 0;
        // ADDIU r0, r0, 0x123
        write_imem(&mut bus, 0, 0x2400_0123);
        step_instruction(&mut bus);
        assert_eq!(bus.rsp_scalar_regs[0], 0);
    }
}
