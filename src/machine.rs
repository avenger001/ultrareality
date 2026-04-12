//! Top-level machine: master cycle counter and devices that will eventually
//! run on the same 93.75 MHz (NTSC) timeline as the RCP.

use crate::boot::{ipl3_load_via_pi_dma, DEFAULT_GAME_SP};
use crate::bus::SystemBus;
use crate::cpu::{CpuHalt, R4300i};
use crate::pif::{PifRomLoadError, PIF_KSEG1_RESET_PC};

pub struct Machine {
    pub cpu: R4300i,
    pub bus: SystemBus,
    pub master_cycles: u64,
    /// Retired cycles from the previous `cpu.step` (incl. bus debt), fed into `Count` before the next instruction.
    last_retired_cycles: u64,
}

impl Machine {
    pub fn new() -> Self {
        Self {
            cpu: R4300i::new(),
            bus: SystemBus::new(),
            master_cycles: 0,
            last_retired_cycles: 0,
        }
    }

    /// Replace cartridge ROM (e.g. after reading a `.z64` / `.n64` file).
    pub fn set_cartridge_rom(&mut self, rom: Vec<u8>) {
        self.bus.pi = crate::pi::Pi::with_rom(rom);
    }

    /// Load PIF boot ROM bytes (e.g. from a dump). Must be at least [`crate::pif::PIF_ROM_LEN`].
    pub fn set_pif_rom(&mut self, data: &[u8]) -> Result<(), PifRomLoadError> {
        self.bus.pif.replace_rom(data)
    }

    /// Cold reset with PC at the PIF ROM entry in kseg1 ([`PIF_KSEG1_RESET_PC`]).
    /// Does not preload the cart into RDRAM; real firmware is expected to drive PI DMA and transfer control.
    pub fn bootstrap_from_pif_reset(&mut self) {
        self.cpu.reset(PIF_KSEG1_RESET_PC);
    }

    /// Reset CPU and load the IPL3 ROM region into RDRAM via **PI DMA** (see [`crate::boot::ipl3_load_via_pi_dma`]).
    /// Entry PC still comes from the cart header until we boot from the PIF reset vector with a loaded PIF ROM.
    pub fn bootstrap_cart_from_rom(&mut self) {
        let Some(pc) =
            ipl3_load_via_pi_dma(&mut self.bus.pi, &mut self.bus.rdram, &mut self.bus.mi) else {
            return;
        };
        self.cpu.reset(pc);
        self.cpu.regs[29] = DEFAULT_GAME_SP;
    }

    pub fn step(&mut self) -> Result<(), CpuHalt> {
        self.cpu
            .cop0
            .advance_count_wrapped(self.last_retired_cycles);
        self.last_retired_cycles = 0;

        let irq = self.bus.mi.cpu_irq_pending();
        let c = self.cpu.step(&mut self.bus, irq)?;
        let def = self.bus.drain_deferred_cycles();
        let delta = c.saturating_add(def);
        self.master_cycles = self.master_cycles.wrapping_add(delta);
        self.last_retired_cycles = delta;
        self.bus.advance_vi_frame_timing(delta);
        Ok(())
    }

    /// Run up to `max_steps` CPU steps (each may include a branch delay slot).
    pub fn run(&mut self, max_steps: u64) -> Result<(), CpuHalt> {
        for _ in 0..max_steps {
            self.cpu
                .cop0
                .advance_count_wrapped(self.last_retired_cycles);
            self.last_retired_cycles = 0;

            let irq = self.bus.mi.cpu_irq_pending();
            let c = self.cpu.step(&mut self.bus, irq)?;
            let def = self.bus.drain_deferred_cycles();
            let delta = c.saturating_add(def);
            self.master_cycles = self.master_cycles.wrapping_add(delta);
            self.last_retired_cycles = delta;
            self.bus.advance_vi_frame_timing(delta);
        }
        Ok(())
    }
}

impl Default for Machine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bus::Bus;
    use crate::cpu::cop0::{EXCCODE_INT, STATUS_EXL, STATUS_IE};
    use crate::ai::AI_REGS_BASE;
    use crate::mi::{
        MI_INTR_AI, MI_INTR_DP, MI_INTR_PI, MI_INTR_SI, MI_INTR_SP, MI_INTR_VI,
    };
    use crate::rcp::{DPC_REG_END, DPC_REGS_BASE, SP_REG_RD_LEN, SP_REGS_BASE};

    #[test]
    fn mi_interrupt_delivers_to_handler_vector() {
        use crate::cpu::cop0::CAUSE_BD;

        let mut m = Machine::new();
        // Handler at 0x80000180 → physical 0x180: `break` (syscall) as placeholder
        m.bus.rdram.write_u32(0x180, 0x0000_000D);

        m.cpu.reset(0x8000_2000);
        // Fresh `reset` COP0 still has ERL/BEV bits that block external interrupts.
        m.cpu.cop0.status = STATUS_IE;
        m.cpu.cop0.cause |= CAUSE_BD;

        m.bus.mi.mask = 0xFF;
        m.bus.mi.raise(MI_INTR_VI);

        m.step().unwrap();

        assert_eq!(m.cpu.pc, 0xFFFF_FFFF_8000_0180u64);
        assert_eq!(m.cpu.cop0.epc, 0x8000_2000);
        assert!((m.cpu.cop0.status & STATUS_EXL) != 0);
        assert!((m.cpu.cop0.cause & CAUSE_BD) == 0);
    }

    #[test]
    fn bootstrap_from_pif_reset_sets_entry_pc() {
        use crate::pif::PIF_ROM_LEN;

        let mut m = Machine::new();
        m.set_pif_rom(&vec![0x5Au8; PIF_ROM_LEN]).unwrap();
        m.bootstrap_from_pif_reset();
        assert_eq!(m.cpu.pc, PIF_KSEG1_RESET_PC);
    }

    #[test]
    fn pi_dma_sets_mi_and_interrupt_delivers() {
        use crate::cpu::cop0::CAUSE_EXCCODE_MASK;
        use crate::cpu::cop0::CAUSE_EXCCODE_SHIFT;
        use crate::pi::{CART_DOM1_ADDR2_BASE, PI_REGS_BASE};

        let mut m = Machine::new();
        let mut rom = vec![0u8; 0x200];
        rom[0x40..0x44].copy_from_slice(&0x1122_3344u32.to_be_bytes());
        m.set_cartridge_rom(rom);

        m.bus.rdram.write_u32(0x180, 0x0000_000D);
        m.cpu.reset(0x8000_4000);
        m.cpu.cop0.status = STATUS_IE;
        m.bus.mi.mask = MI_INTR_PI;
        m.bus.mi.intr = 0;

        m.bus.write_u32(PI_REGS_BASE + 0x00, 0x100);
        m.bus.write_u32(PI_REGS_BASE + 0x04, CART_DOM1_ADDR2_BASE + 0x40);
        m.bus.write_u32(PI_REGS_BASE + 0x0C, 3);

        assert_ne!(m.bus.mi.intr & MI_INTR_PI, 0);
        assert_eq!(m.bus.rdram.read_u32(0x100).unwrap(), 0x1122_3344);

        m.step().unwrap();
        assert_eq!(m.cpu.pc, 0xFFFF_FFFF_8000_0180u64);
        assert_eq!(m.cpu.cop0.epc, 0x8000_4000);
        let exc = (m.cpu.cop0.cause >> CAUSE_EXCCODE_SHIFT) & CAUSE_EXCCODE_MASK;
        assert_eq!(exc, EXCCODE_INT);
    }

    #[test]
    fn si_dma_sets_mi_and_interrupt_delivers() {
        use crate::cpu::cop0::CAUSE_EXCCODE_MASK;
        use crate::cpu::cop0::CAUSE_EXCCODE_SHIFT;
        use crate::pif::{PIF_RAM_START, PIF_ROM_LEN};
        use crate::si::SI_REGS_BASE;

        let mut m = Machine::new();
        m.set_pif_rom(&vec![0x3Cu8; PIF_ROM_LEN]).unwrap();
        m.bus.pif.ram[0..4].copy_from_slice(&0x99AA_BBCCu32.to_be_bytes());

        m.bus.rdram.write_u32(0x180, 0x0000_000D);
        m.cpu.reset(0x8000_5000);
        m.cpu.cop0.status = STATUS_IE;
        m.bus.mi.mask = MI_INTR_SI;
        m.bus.mi.intr = 0;

        m.bus.write_u32(SI_REGS_BASE, 0x200);
        m.bus.write_u32(SI_REGS_BASE + 0x04, PIF_RAM_START);

        assert_ne!(m.bus.mi.intr & MI_INTR_SI, 0);
        assert_eq!(m.bus.rdram.read_u32(0x200).unwrap(), 0x99AA_BBCC);

        m.step().unwrap();
        assert_eq!(m.cpu.pc, 0xFFFF_FFFF_8000_0180u64);
        assert_eq!(m.cpu.cop0.epc, 0x8000_5000);
        let exc = (m.cpu.cop0.cause >> CAUSE_EXCCODE_SHIFT) & CAUSE_EXCCODE_MASK;
        assert_eq!(exc, EXCCODE_INT);
    }

    #[test]
    fn ai_len_write_sets_mi_and_interrupt_delivers() {
        use crate::cpu::cop0::CAUSE_EXCCODE_MASK;
        use crate::cpu::cop0::CAUSE_EXCCODE_SHIFT;

        let mut m = Machine::new();
        m.bus.rdram.write_u32(0x180, 0x0000_000D);
        m.cpu.reset(0x8000_6000);
        m.cpu.cop0.status = STATUS_IE;
        m.bus.mi.mask = MI_INTR_AI;
        m.bus.mi.intr = 0;

        m.bus.write_u32(AI_REGS_BASE + 0x04, 0x400);

        assert_ne!(m.bus.mi.intr & MI_INTR_AI, 0);
        m.step().unwrap();
        assert_eq!(m.cpu.pc, 0xFFFF_FFFF_8000_0180u64);
        assert_eq!(m.cpu.cop0.epc, 0x8000_6000);
        let exc = (m.cpu.cop0.cause >> CAUSE_EXCCODE_SHIFT) & CAUSE_EXCCODE_MASK;
        assert_eq!(exc, EXCCODE_INT);
    }

    #[test]
    fn sp_rd_len_sets_mi_and_interrupt_delivers() {
        use crate::cpu::cop0::CAUSE_EXCCODE_MASK;
        use crate::cpu::cop0::CAUSE_EXCCODE_SHIFT;

        let mut m = Machine::new();
        m.bus.rdram.write_u32(0x180, 0x0000_000D);
        m.cpu.reset(0x8000_7000);
        m.cpu.cop0.status = STATUS_IE;
        m.bus.mi.mask = MI_INTR_SP;
        m.bus.mi.intr = 0;

        m.bus.write_u32(SP_REGS_BASE + SP_REG_RD_LEN, 0x200);

        assert_ne!(m.bus.mi.intr & MI_INTR_SP, 0);
        m.step().unwrap();
        assert_eq!(m.cpu.pc, 0xFFFF_FFFF_8000_0180u64);
        assert_eq!(m.cpu.cop0.epc, 0x8000_7000);
        let exc = (m.cpu.cop0.cause >> CAUSE_EXCCODE_SHIFT) & CAUSE_EXCCODE_MASK;
        assert_eq!(exc, EXCCODE_INT);
    }

    #[test]
    fn dpc_end_sets_mi_and_interrupt_delivers() {
        use crate::cpu::cop0::CAUSE_EXCCODE_MASK;
        use crate::cpu::cop0::CAUSE_EXCCODE_SHIFT;

        let mut m = Machine::new();
        m.bus.rdram.write_u32(0x180, 0x0000_000D);
        m.cpu.reset(0x8000_8000);
        m.cpu.cop0.status = STATUS_IE;
        m.bus.mi.mask = MI_INTR_DP;
        m.bus.mi.intr = 0;

        m.bus.write_u32(DPC_REGS_BASE + DPC_REG_END, 0x0010_0000);

        assert_ne!(m.bus.mi.intr & MI_INTR_DP, 0);
        m.step().unwrap();
        assert_eq!(m.cpu.pc, 0xFFFF_FFFF_8000_0180u64);
        assert_eq!(m.cpu.cop0.epc, 0x8000_8000);
        let exc = (m.cpu.cop0.cause >> CAUSE_EXCCODE_SHIFT) & CAUSE_EXCCODE_MASK;
        assert_eq!(exc, EXCCODE_INT);
    }
}
