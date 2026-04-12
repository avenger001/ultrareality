//! Top-level machine: [`crate::timing::RCP_MASTER_HZ_NTSC`] master cycle counter.
//! Each step adds CPU cycles plus VI/RDP deferred debt ([`crate::bus::SystemBus::drain_deferred_cycles`]),
//! then advances in-flight PI/SI/AI DMA on the same quantum ([`crate::bus::SystemBus::rcp_advance_dma_in_flight`])
//! so `Count`, `master_cycles`, and VI frame timing stay aligned.

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
        self.bus.rcp_advance_dma_in_flight(delta);
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
            self.bus.rcp_advance_dma_in_flight(delta);
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
    use crate::cpu::cop0::{
        CAUSE_BD, CAUSE_EXCCODE_MASK, CAUSE_EXCCODE_SHIFT, EXCCODE_INT, GENERAL_EXCEPTION_OFFSET,
        KSEG0_INTERRUPT_VECTOR_PC, MIPS_OPCODE_BREAK, STATUS_EXL, STATUS_IE,
    };
    use crate::ai::{AI_REG_LEN, AI_REGS_BASE};
    use crate::timing::{ai_pcm_buffer_cycles, sp_rsp_dma_total_cycles};
    use crate::si::SI_DMA_CYCLES;

    /// Test RDRAM destinations for PI/SI DMA integration tests (physical).
    const RDRAM_TEST_PI_DST: u32 = 0x100;
    const RDRAM_TEST_SI_DST: u32 = 0x200;
    use crate::mi::{
        MI_INTR_AI, MI_INTR_DP, MI_INTR_PI, MI_INTR_SI, MI_INTR_SP, MI_INTR_VI,
    };
    use crate::rcp::{DPC_REG_END, DPC_REGS_BASE, SP_REG_RD_LEN, SP_REGS_BASE};
    use crate::rdp::Rdp;

    #[test]
    fn mi_interrupt_delivers_to_handler_vector() {
        let mut m = Machine::new();
        // Placeholder handler at KSEG0 general vector (physical GENERAL_EXCEPTION_OFFSET): `break`.
        m.bus
            .rdram
            .write_u32(GENERAL_EXCEPTION_OFFSET, MIPS_OPCODE_BREAK);

        m.cpu.reset(0x8000_2000);
        // Fresh `reset` COP0 still has ERL/BEV bits that block external interrupts.
        m.cpu.cop0.status = STATUS_IE;
        m.cpu.cop0.cause |= CAUSE_BD;

        m.bus.mi.mask = 0xFF;
        m.bus.mi.raise(MI_INTR_VI);

        m.step().unwrap();

        assert_eq!(m.cpu.pc, KSEG0_INTERRUPT_VECTOR_PC);
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
        use crate::pi::{
            CART_DOM1_ADDR2_BASE, CART_ROM_TEST_DWORD_OFF, PI_REG_CART_ADDR, PI_REG_DRAM_ADDR,
            PI_REG_RD_LEN, PI_REGS_BASE,
        };

        let mut m = Machine::new();
        let mut rom = vec![0u8; 0x200];
        rom[CART_ROM_TEST_DWORD_OFF..CART_ROM_TEST_DWORD_OFF + 4]
            .copy_from_slice(&0x1122_3344u32.to_be_bytes());
        m.set_cartridge_rom(rom);

        m.bus
            .rdram
            .write_u32(GENERAL_EXCEPTION_OFFSET, MIPS_OPCODE_BREAK);
        m.cpu.reset(0x8000_4000);
        m.cpu.cop0.status = STATUS_IE;
        m.bus.mi.mask = MI_INTR_PI;
        m.bus.mi.intr = 0;

        m.bus.write_u32(PI_REGS_BASE + PI_REG_DRAM_ADDR, RDRAM_TEST_PI_DST);
        m.bus.write_u32(
            PI_REGS_BASE + PI_REG_CART_ADDR,
            CART_DOM1_ADDR2_BASE + CART_ROM_TEST_DWORD_OFF as u32,
        );
        m.bus.write_u32(PI_REGS_BASE + PI_REG_RD_LEN, 3);

        assert_eq!(m.bus.mi.intr & MI_INTR_PI, 0);
        let pi_cost = m.bus.pi.cart_dma_cost_cycles(4);
        m.bus.rcp_advance_dma_in_flight(pi_cost);
        assert_ne!(m.bus.mi.intr & MI_INTR_PI, 0);
        assert_eq!(
            m.bus.rdram.read_u32(RDRAM_TEST_PI_DST).unwrap(),
            0x1122_3344
        );

        m.step().unwrap();
        assert_eq!(m.cpu.pc, KSEG0_INTERRUPT_VECTOR_PC);
        assert_eq!(m.cpu.cop0.epc, 0x8000_4000);
        let exc = (m.cpu.cop0.cause >> CAUSE_EXCCODE_SHIFT) & CAUSE_EXCCODE_MASK;
        assert_eq!(exc, EXCCODE_INT);
    }

    #[test]
    fn si_dma_sets_mi_and_interrupt_delivers() {
        use crate::pif::{PIF_RAM_START, PIF_ROM_LEN};
        use crate::si::{SI_REG_DRAM_ADDR, SI_REG_PIF_ADDR_RD64B, SI_REGS_BASE};

        let mut m = Machine::new();
        m.set_pif_rom(&vec![0x3Cu8; PIF_ROM_LEN]).unwrap();
        m.bus.pif.ram[0..4].copy_from_slice(&0x99AA_BBCCu32.to_be_bytes());

        m.bus
            .rdram
            .write_u32(GENERAL_EXCEPTION_OFFSET, MIPS_OPCODE_BREAK);
        m.cpu.reset(0x8000_5000);
        m.cpu.cop0.status = STATUS_IE;
        m.bus.mi.mask = MI_INTR_SI;
        m.bus.mi.intr = 0;

        m.bus.write_u32(SI_REGS_BASE + SI_REG_DRAM_ADDR, RDRAM_TEST_SI_DST);
        m.bus.write_u32(SI_REGS_BASE + SI_REG_PIF_ADDR_RD64B, PIF_RAM_START);

        assert_eq!(m.bus.mi.intr & MI_INTR_SI, 0);
        m.bus.rcp_advance_dma_in_flight(SI_DMA_CYCLES);
        assert_ne!(m.bus.mi.intr & MI_INTR_SI, 0);
        assert_eq!(
            m.bus.rdram.read_u32(RDRAM_TEST_SI_DST).unwrap(),
            0x99AA_BBCC
        );

        m.step().unwrap();
        assert_eq!(m.cpu.pc, KSEG0_INTERRUPT_VECTOR_PC);
        assert_eq!(m.cpu.cop0.epc, 0x8000_5000);
        let exc = (m.cpu.cop0.cause >> CAUSE_EXCCODE_SHIFT) & CAUSE_EXCCODE_MASK;
        assert_eq!(exc, EXCCODE_INT);
    }

    #[test]
    fn ai_len_write_sets_mi_and_interrupt_delivers() {
        let mut m = Machine::new();
        m.bus
            .rdram
            .write_u32(GENERAL_EXCEPTION_OFFSET, MIPS_OPCODE_BREAK);
        m.cpu.reset(0x8000_6000);
        m.cpu.cop0.status = STATUS_IE;
        m.bus.mi.mask = MI_INTR_AI;
        m.bus.mi.intr = 0;

        m.bus.write_u32(AI_REGS_BASE + AI_REG_LEN, 0x400);

        assert_eq!(m.bus.mi.intr & MI_INTR_AI, 0);
        m.bus
            .rcp_advance_dma_in_flight(ai_pcm_buffer_cycles(0x400));
        assert_ne!(m.bus.mi.intr & MI_INTR_AI, 0);
        m.step().unwrap();
        assert_eq!(m.cpu.pc, KSEG0_INTERRUPT_VECTOR_PC);
        assert_eq!(m.cpu.cop0.epc, 0x8000_6000);
        let exc = (m.cpu.cop0.cause >> CAUSE_EXCCODE_SHIFT) & CAUSE_EXCCODE_MASK;
        assert_eq!(exc, EXCCODE_INT);
    }

    #[test]
    fn sp_rd_len_sets_mi_and_interrupt_delivers() {
        let mut m = Machine::new();
        m.bus
            .rdram
            .write_u32(GENERAL_EXCEPTION_OFFSET, MIPS_OPCODE_BREAK);
        m.cpu.reset(0x8000_7000);
        m.cpu.cop0.status = STATUS_IE;
        m.bus.mi.mask = MI_INTR_SP;
        m.bus.mi.intr = 0;

        m.bus.write_u32(SP_REGS_BASE + SP_REG_RD_LEN, 0x200);

        assert_eq!(m.bus.mi.intr & MI_INTR_SP, 0);
        m.bus
            .rcp_advance_dma_in_flight(sp_rsp_dma_total_cycles(0x200));
        assert_ne!(m.bus.mi.intr & MI_INTR_SP, 0);
        m.step().unwrap();
        assert_eq!(m.cpu.pc, KSEG0_INTERRUPT_VECTOR_PC);
        assert_eq!(m.cpu.cop0.epc, 0x8000_7000);
        let exc = (m.cpu.cop0.cause >> CAUSE_EXCCODE_SHIFT) & CAUSE_EXCCODE_MASK;
        assert_eq!(exc, EXCCODE_INT);
    }

    #[test]
    fn dpc_end_sets_mi_and_interrupt_delivers() {
        let mut m = Machine::new();
        m.bus
            .rdram
            .write_u32(GENERAL_EXCEPTION_OFFSET, MIPS_OPCODE_BREAK);
        m.cpu.reset(0x8000_8000);
        m.cpu.cop0.status = STATUS_IE;
        m.bus.mi.mask = MI_INTR_DP;
        m.bus.mi.intr = 0;

        m.bus.write_u32(DPC_REGS_BASE + DPC_REG_END, 0x0010_0000);

        assert_eq!(m.bus.mi.intr & MI_INTR_DP, 0);
        let est = Rdp::estimate_display_list_cycles(0, 0x0010_0000);
        m.bus.rcp_advance_dma_in_flight(est);
        assert_ne!(m.bus.mi.intr & MI_INTR_DP, 0);
        m.step().unwrap();
        assert_eq!(m.cpu.pc, KSEG0_INTERRUPT_VECTOR_PC);
        assert_eq!(m.cpu.cop0.epc, 0x8000_8000);
        let exc = (m.cpu.cop0.cause >> CAUSE_EXCCODE_SHIFT) & CAUSE_EXCCODE_MASK;
        assert_eq!(exc, EXCCODE_INT);
    }
}
