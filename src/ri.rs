//! RDRAM Interface (RI): timing and mode registers at `0x0470_0000` ([n64brew: RI](https://n64brew.dev/wiki/RDRAM_Interface)).
//!
//! ## Rambus Packet Model
//!
//! The N64 uses Rambus RDRAM with a packetized interface. Each access involves:
//! - **ROW (Row Open)**: Activates a row in a bank (~10 cycles)
//! - **COL (Column Access)**: Reads/writes from open row (~2 cycles per 8 bytes)
//! - **PRE (Precharge)**: Closes a row before opening another in same bank (~5 cycles)
//!
//! The retail N64 has 2 RDRAM chips (4MB each, 8MB total; Expansion Pak adds 2 more for 8MB total
//! capacity). Each chip has 8 banks with 2KB row buffers. Addresses are interleaved between chips
//! for bandwidth, so consecutive cache lines often hit different banks.
//!
//! This model tracks the last open row per bank to differentiate:
//! - **Row hit**: Same row as last access → COL only
//! - **Row miss**: Different row in same bank → PRE + ROW + COL

pub const RI_REGS_BASE: u32 = 0x0470_0000;
pub const RI_REGS_LEN: usize = 0x20;

// --- Rambus RDRAM Timing Constants -------------------------------------------

/// Number of RDRAM banks per chip (standard Rambus RDRAM).
const RDRAM_BANKS_PER_CHIP: usize = 8;

/// Number of RDRAM chips in standard N64 (4MB each = 8MB total).
/// Expansion Pak doubles this, but we model a fixed array.
const RDRAM_CHIP_COUNT: usize = 2;

/// Total banks tracked (2 chips × 8 banks).
const RDRAM_TOTAL_BANKS: usize = RDRAM_CHIP_COUNT * RDRAM_BANKS_PER_CHIP;

/// Bits to shift address to get row number (2KB row = log2(2048) = 11).
const RDRAM_ROW_SHIFT: u32 = 11; // log2(2048)

/// Cycles for ROW (row activation / open).
const RAMBUS_ROW_CYCLES: u64 = 10;

/// Cycles for COL (column access, per 8-byte chunk on 2-byte bus).
const RAMBUS_COL_CYCLES: u64 = 2;

/// Cycles for PRE (precharge before row change in same bank).
const RAMBUS_PRE_CYCLES: u64 = 5;

/// No open row sentinel value.
const RAMBUS_ROW_CLOSED: u32 = u32::MAX;

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

/// Rambus bank state: tracks open row per bank for row hit/miss timing.
#[derive(Debug, Clone)]
struct RambusBankState {
    /// Currently open row number per bank (RAMBUS_ROW_CLOSED if none).
    open_row: [u32; RDRAM_TOTAL_BANKS],
}

impl RambusBankState {
    fn new() -> Self {
        Self {
            open_row: [RAMBUS_ROW_CLOSED; RDRAM_TOTAL_BANKS],
        }
    }

    fn reset(&mut self) {
        self.open_row.fill(RAMBUS_ROW_CLOSED);
    }

    /// Extract bank index from physical address.
    /// Interleaved: bits 11-13 select bank within chip, bits 22+ select chip.
    #[inline]
    fn bank_index(paddr: u32) -> usize {
        // Bank bits from within row: bits 11-13 (3 bits = 8 banks)
        let bank_in_chip = ((paddr >> RDRAM_ROW_SHIFT) & 0x7) as usize;
        // Chip select: bit 22 for 4MB chips (simpler model)
        let chip = ((paddr >> 22) & 0x1) as usize;
        chip * RDRAM_BANKS_PER_CHIP + bank_in_chip
    }

    /// Extract row number from physical address within a bank.
    #[inline]
    fn row_number(paddr: u32) -> u32 {
        // Row = upper bits after bank selection (simplified: bits 14+)
        paddr >> 14
    }

    /// Calculate access cycles and update bank state.
    /// Returns cycles for this access based on row hit/miss.
    fn access(&mut self, paddr: u32, access_bytes: u32) -> u64 {
        let bank = Self::bank_index(paddr);
        let row = Self::row_number(paddr);
        let current_open = self.open_row[bank];

        // Column cycles: ceil(access_bytes / 8) * RAMBUS_COL_CYCLES
        let col_cycles = ((access_bytes as u64 + 7) / 8) * RAMBUS_COL_CYCLES;

        if current_open == row {
            // Row hit: only column access cost
            col_cycles
        } else {
            // Row miss: precharge (if row was open) + row open + column access
            self.open_row[bank] = row;
            let pre = if current_open != RAMBUS_ROW_CLOSED {
                RAMBUS_PRE_CYCLES
            } else {
                0
            };
            pre + RAMBUS_ROW_CYCLES + col_cycles
        }
    }
}

#[derive(Debug)]
pub struct Ri {
    /// Eight word registers (`0x00` … `0x1C`).
    pub regs: [u32; RI_REGS_LEN / 4],
    /// Rambus bank state for row buffer tracking.
    rambus: RambusBankState,
}

impl Ri {
    pub fn new() -> Self {
        let mut s = Self {
            regs: [0u32; RI_REGS_LEN / 4],
            rambus: RambusBankState::new(),
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
        self.rambus.reset();
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

    /// Extra RCP cycles added to uncached RDRAM loads/stores on the CPU beyond [`crate::cycles::MEM_ACCESS_BASE`].
    /// Uses the `RI_LATENCY` low nibble (overlap / timing field) as a coarse stall — not a serial Rambus model.
    ///
    /// Deprecated: prefer [`cpu_rdram_access_cycles`] for Rambus packet model timing.
    #[inline]
    pub fn cpu_rdram_extra_cycles(&self) -> u64 {
        let lat = self.regs[(RI_REG_LATENCY / 4) as usize];
        (((lat & 0xF) as u64) * 1) / 4
    }

    /// Rambus packet model: compute access cycles based on row hit/miss state.
    ///
    /// This updates the internal row buffer state and returns cycles for:
    /// - **Row hit**: Only column access cost (fastest)
    /// - **Row miss**: Precharge + row open + column access (slowest)
    ///
    /// The RI_LATENCY nibble adds additional tuning on top of the base model.
    #[inline]
    pub fn cpu_rdram_access_cycles(&mut self, paddr: u32, access_bytes: u32) -> u64 {
        // Base Rambus packet timing from row buffer model
        let rambus_cycles = self.rambus.access(paddr, access_bytes);

        // Add RI_LATENCY nibble contribution (controller/signal timing)
        let lat = self.regs[(RI_REG_LATENCY / 4) as usize];
        let lat_extra = ((lat & 0xF) as u64) / 4;

        rambus_cycles.saturating_add(lat_extra)
    }

    /// Query whether an access would be a row hit without modifying state.
    #[inline]
    pub fn would_be_row_hit(&self, paddr: u32) -> bool {
        let bank = RambusBankState::bank_index(paddr);
        let row = RambusBankState::row_number(paddr);
        self.rambus.open_row[bank] == row
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

    #[test]
    fn cpu_rdram_extra_matches_latency_nibble() {
        let ri = Ri::new();
        assert_eq!(ri.cpu_rdram_extra_cycles(), 3);
        let mut ri2 = Ri::new();
        ri2.write(RI_REGS_BASE + RI_REG_LATENCY, 0);
        assert_eq!(ri2.cpu_rdram_extra_cycles(), 0);
    }

    #[test]
    fn rambus_row_hit_cheaper_than_miss() {
        let mut ri = Ri::new();
        ri.write(RI_REGS_BASE + RI_REG_LATENCY, 0); // Remove RI_LATENCY overhead for clarity

        // First access to address 0: row miss (no precharge needed, first open)
        // ROW (10) + COL (2 for 4 bytes) = 12 cycles
        let first = ri.cpu_rdram_access_cycles(0x0000_0000, 4);
        assert_eq!(first, RAMBUS_ROW_CYCLES + RAMBUS_COL_CYCLES);

        // Second access to same row (within 2KB): row hit
        // COL only = 2 cycles
        let second = ri.cpu_rdram_access_cycles(0x0000_0004, 4);
        assert_eq!(second, RAMBUS_COL_CYCLES);

        // Third access to different row in same bank: row miss with precharge
        // PRE (5) + ROW (10) + COL (2) = 17 cycles
        let third = ri.cpu_rdram_access_cycles(0x0000_4000, 4);
        assert_eq!(third, RAMBUS_PRE_CYCLES + RAMBUS_ROW_CYCLES + RAMBUS_COL_CYCLES);
    }

    #[test]
    fn rambus_different_banks_no_precharge() {
        let mut ri = Ri::new();
        ri.write(RI_REGS_BASE + RI_REG_LATENCY, 0);

        // Access bank 0, row 0
        let _ = ri.cpu_rdram_access_cycles(0x0000_0000, 4);

        // Access bank 1 (different bank bits 11-13) - no precharge needed
        // Bank 1 = bits 11-13 = 0x800
        let bank1 = ri.cpu_rdram_access_cycles(0x0000_0800, 4);
        assert_eq!(bank1, RAMBUS_ROW_CYCLES + RAMBUS_COL_CYCLES);
    }

    #[test]
    fn rambus_col_cycles_scale_with_size() {
        let mut ri = Ri::new();
        ri.write(RI_REGS_BASE + RI_REG_LATENCY, 0);

        // 4 bytes = 1 chunk = 2 cycles
        let c4 = ri.cpu_rdram_access_cycles(0x0000_0000, 4);
        assert_eq!(c4, RAMBUS_ROW_CYCLES + RAMBUS_COL_CYCLES);

        // Reset row state (also resets RI_LATENCY, so clear it again)
        ri.reset();
        ri.write(RI_REGS_BASE + RI_REG_LATENCY, 0);

        // 16 bytes = 2 chunks = 4 cycles
        let c16 = ri.cpu_rdram_access_cycles(0x0000_0000, 16);
        assert_eq!(c16, RAMBUS_ROW_CYCLES + 2 * RAMBUS_COL_CYCLES);
    }
}
