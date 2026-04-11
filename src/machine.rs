//! Top-level machine: master cycle counter and devices that will eventually
//! run on the same 93.75 MHz (NTSC) timeline as the RCP.

use crate::boot::{hle_ipl3_load_rom, DEFAULT_GAME_SP};
use crate::bus::SystemBus;
use crate::cpu::{CpuHalt, R4300i};

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

    /// HLE bootstrap: IPL3-style ROM→RDRAM copy, then PC from header boot address (`0x08`).
    /// Does **not** run PIF/CIC (no CIC 6103/6106 PC adjust); use for bring-up until PIF is emulated.
    pub fn bootstrap_hle_cart_entry(&mut self) {
        let Some(pc) = hle_ipl3_load_rom(&self.bus.pi.rom, &mut self.bus.rdram) else {
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
    use crate::cpu::cop0::{STATUS_EXL, STATUS_IE};

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
        m.bus.mi.raise(crate::mi::MI_INTR_VI);

        m.step().unwrap();

        assert_eq!(m.cpu.pc, 0xFFFF_FFFF_8000_0180u64);
        assert_eq!(m.cpu.cop0.epc, 0x8000_2000);
        assert!((m.cpu.cop0.status & STATUS_EXL) != 0);
        assert!((m.cpu.cop0.cause & CAUSE_BD) == 0);
    }
}
