//! Top-level machine: master cycle counter and devices that will eventually
//! run on the same 93.75 MHz (NTSC) timeline as the RCP.

use crate::bus::{PhysicalMemory, DEFAULT_RDRAM_SIZE};
use crate::cpu::{CpuHalt, R4300i};

pub struct Machine {
    pub cpu: R4300i,
    pub mem: PhysicalMemory,
    pub master_cycles: u64,
}

impl Machine {
    pub fn new() -> Self {
        Self {
            cpu: R4300i::new(),
            mem: PhysicalMemory::new(DEFAULT_RDRAM_SIZE),
            master_cycles: 0,
        }
    }

    pub fn step(&mut self) -> Result<(), CpuHalt> {
        let rdram_size = self.mem.data.len();
        let c = self.cpu.step(&mut self.mem, rdram_size)?;
        self.master_cycles = self.master_cycles.wrapping_add(c);
        Ok(())
    }

    /// Run up to `max_steps` CPU steps (each may include a branch delay slot).
    pub fn run(&mut self, max_steps: u64) -> Result<(), CpuHalt> {
        let rdram_size = self.mem.data.len();
        for _ in 0..max_steps {
            let c = self.cpu.step(&mut self.mem, rdram_size)?;
            self.master_cycles = self.master_cycles.wrapping_add(c);
        }
        Ok(())
    }
}

impl Default for Machine {
    fn default() -> Self {
        Self::new()
    }
}
