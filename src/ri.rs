//! RDRAM Interface (RI): timing and mode registers at `0x0470_0000` ([n64brew: RI](https://n64brew.dev/wiki/RDRAM_Interface)).
//!
//! No serial Rambus protocol is simulated; values are **retail-style defaults** sufficient for
//! libultra-style init and polling.

pub const RI_REGS_BASE: u32 = 0x0470_0000;
pub const RI_REGS_LEN: usize = 0x20;

/// Byte offsets within the RI block ([n64brew: RI](https://n64brew.dev/wiki/RDRAM_Interface)).
pub const RI_REG_MODE: u32 = 0x00;
pub const RI_REG_CONFIG: u32 = 0x04;
pub const RI_REG_CURRENT_LOAD: u32 = 0x08;
pub const RI_REG_SELECT: u32 = 0x0C;
pub const RI_REG_REFRESH: u32 = 0x10;
pub const RI_REG_LATENCY: u32 = 0x14;

/// Typical `RI_MODE` after reset (retail observation / emulators).
pub const RI_MODE_DEFAULT: u32 = 0x0000_000E;
/// Typical `RI_CONFIG`.
pub const RI_CONFIG_DEFAULT: u32 = 0x0000_0040;
/// Typical `RI_SELECT`.
pub const RI_SELECT_DEFAULT: u32 = 0x0000_0014;
/// Typical `RI_REFRESH` / refresh interval field.
pub const RI_REFRESH_DEFAULT: u32 = 0x0006_3634;
/// `RI_LATENCY` default overlap field (low nibble often `0xF`).
pub const RI_LATENCY_DEFAULT: u32 = 0x0000_000F;

#[derive(Debug)]
pub struct Ri {
    /// Eight word registers (`0x00` … `0x1C`).
    pub regs: [u32; RI_REGS_LEN / 4],
}

impl Ri {
    pub fn new() -> Self {
        let mut s = Self {
            regs: [0u32; RI_REGS_LEN / 4],
        };
        s.reset();
        s
    }

    /// Power-on values aligned with common HLE defaults ([Project64-style cold init](https://github.com/project64/project64)).
    pub fn reset(&mut self) {
        self.regs.fill(0);
        self.regs[(RI_REG_MODE / 4) as usize] = RI_MODE_DEFAULT;
        self.regs[(RI_REG_CONFIG / 4) as usize] = RI_CONFIG_DEFAULT;
        self.regs[(RI_REG_SELECT / 4) as usize] = RI_SELECT_DEFAULT;
        self.regs[(RI_REG_REFRESH / 4) as usize] = RI_REFRESH_DEFAULT;
        self.regs[(RI_REG_LATENCY / 4) as usize] = RI_LATENCY_DEFAULT;
    }

    fn offset(paddr: u32) -> Option<usize> {
        if (RI_REGS_BASE..RI_REGS_BASE + RI_REGS_LEN as u32).contains(&paddr) && paddr & 3 == 0 {
            Some((paddr - RI_REGS_BASE) as usize / 4)
        } else {
            None
        }
    }

    pub fn read(&self, paddr: u32) -> u32 {
        let Some(i) = Self::offset(paddr) else {
            return 0;
        };
        self.regs.get(i).copied().unwrap_or(0)
    }

    pub fn write(&mut self, paddr: u32, value: u32) {
        let Some(i) = Self::offset(paddr) else {
            return;
        };
        if let Some(r) = self.regs.get_mut(i) {
            *r = value;
        }
    }
}

impl Default for Ri {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ri_mode_reset_matches_retail_stub() {
        let ri = Ri::new();
        assert_eq!(ri.read(RI_REGS_BASE + RI_REG_MODE), RI_MODE_DEFAULT);
        assert_eq!(ri.read(RI_REGS_BASE + RI_REG_CONFIG), RI_CONFIG_DEFAULT);
        assert_eq!(ri.read(RI_REGS_BASE + RI_REG_SELECT), RI_SELECT_DEFAULT);
    }

    #[test]
    fn ri_write_round_trip() {
        let mut ri = Ri::new();
        ri.write(RI_REGS_BASE + RI_REG_MODE, 0xAA);
        assert_eq!(ri.read(RI_REGS_BASE + RI_REG_MODE), 0xAA);
    }
}
