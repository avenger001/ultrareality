//! RCP: RSP **SP** MMIO (`0x0404_0000`) and RDP **DPC** (`0x0410_0000`).
//!
//! **SP**: `SP_MEM_ADDR` / `SP_DRAM_ADDR` must be programmed before a write to `SP_RD_LEN` /
//! `SP_WR_LEN`; those writes run **immediate** RDRAM↔RSP memory DMA (see [`crate::bus::SystemBus`]),
//! then update both address registers to the end of the transfer. `SP_STATUS` / `SP_SEMAPHORE`
//! are handled in [`crate::bus::SystemBus`] (halt, `MI_INTR_SP` clear/set, lock). The RSP PC lives
//! at [`SP_PC_REGS_BASE`] (not `SP_REGS_BASE`).
//!
//! **DPC**: command DMA addresses and [`crate::mi::MI_INTR_DP`] on **non-zero** `DPC_END` (no RDP
//! command processing yet — completion is **instant** and `DPC_CURRENT` reflects the end address).

use crate::mi::{Mi, MI_INTR_DP};

/// RSP SP register block ([n64brew: RSP](https://n64brew.dev/wiki/RSP)).
pub const SP_REGS_BASE: u32 = 0x0404_0000;
pub const SP_REGS_LEN: usize = 0x20;

/// `SP_RD_LEN` — DMA RDRAM → DMEM.
pub const SP_REG_RD_LEN: u32 = 0x08;
/// `SP_WR_LEN` — DMA DMEM → RDRAM.
pub const SP_REG_WR_LEN: u32 = 0x0C;
/// `SP_STATUS` — halt / broke / interrupt control (W1S/W1C pairs on write).
pub const SP_REG_STATUS: u32 = 0x10;
/// `SP_DMA_FULL` — DMA queue full (read-only in practice).
pub const SP_REG_DMA_FULL: u32 = 0x14;
/// `SP_DMA_BUSY` — DMA engine busy (read-only in practice).
pub const SP_REG_DMA_BUSY: u32 = 0x18;
/// `SP_SEMAPHORE` — CPU/RSP lock (read has side effects).
pub const SP_REG_SEMAPHORE: u32 = 0x1C;

/// Word offset of `SP_MEM_ADDR` / `SP_DRAM_ADDR` / length registers.
pub const SP_WORD_MEM_ADDR: usize = 0;
pub const SP_WORD_DRAM_ADDR: usize = 1;

/// Decoded `SP_RD_LEN` / `SP_WR_LEN` fields ([n64maps RSP](https://infrid.com/rcp64/docfiles/n64maps.txt)):
/// bits 11:0 → line length − 1, bits 19:12 → line count − 1, bits 31:20 → RDRAM skip between lines
/// (masked to 8-byte steps like common HLE cores).
#[inline]
pub fn sp_dma_decode(len_reg: u32) -> (usize, usize, usize) {
    let line_bytes = ((len_reg & 0xFFF) as usize).saturating_add(1);
    let line_count = (((len_reg >> 12) & 0xFF) as usize).saturating_add(1);
    let dram_skip = ((len_reg >> 20) & 0xFF8) as usize;
    (line_bytes, line_count, dram_skip)
}

/// RSP DMEM/IMEM linear index after DMA (13-bit) and updated `SP_MEM_ADDR` / `SP_DRAM_ADDR` values.
pub fn sp_dma_end_addresses(
    mem: u32,
    dram: u32,
    line_bytes: usize,
    line_count: usize,
    dram_skip: usize,
) -> (u32, u32) {
    let start_flat = (((mem & 0x1000) != 0) as usize).saturating_mul(0x1000) + (mem & 0xFFF) as usize;
    let end_flat = (start_flat + line_count.saturating_mul(line_bytes)) & 0x1FFF;
    let imem = (end_flat & 0x1000) != 0;
    let off = (end_flat & 0xFFF) as u32;
    let new_mem = (mem & 0xFFFF_E000) | (u32::from(imem) << 12) | off;

    let dram_lo = (dram & 0x00FF_FFFF) as usize;
    let delta = line_count
        .saturating_mul(line_bytes)
        .saturating_add(line_count.saturating_mul(dram_skip));
    let new_lo = dram_lo.saturating_add(delta) & 0x00FF_FFFF;
    let new_dram = (dram & !0x00FF_FFFF) | (new_lo as u32);

    (new_mem, new_dram)
}

#[derive(Debug)]
pub struct SpRegs {
    pub words: [u32; SP_REGS_LEN / 4],
}

impl SpRegs {
    pub fn new() -> Self {
        Self {
            words: [0u32; SP_REGS_LEN / 4],
        }
    }

    fn word_index(paddr: u32) -> Option<usize> {
        if (SP_REGS_BASE..SP_REGS_BASE + SP_REGS_LEN as u32).contains(&paddr) {
            let o = (paddr - SP_REGS_BASE) as usize;
            if o & 3 != 0 {
                return None;
            }
            Some(o / 4)
        } else {
            None
        }
    }

    pub fn read(&self, paddr: u32) -> u32 {
        let Some(i) = Self::word_index(paddr) else {
            return 0;
        };
        self.words.get(i).copied().unwrap_or(0)
    }

    pub fn store_u32(&mut self, paddr: u32, value: u32) {
        let Some(i) = Self::word_index(paddr) else {
            return;
        };
        if let Some(w) = self.words.get_mut(i) {
            *w = value;
        }
    }
}

impl Default for SpRegs {
    fn default() -> Self {
        Self::new()
    }
}

/// RSP scalar PC / built-in self-test ([patater: SP_PC](https://patater.com/gbaguy/day8n64.htm)).
/// Separate from [`SP_REGS_BASE`] (`0x0404_0000`); physical `0x0408_0000` / `0xA408_0000` (kseg1).
pub const SP_PC_REGS_BASE: u32 = 0x0408_0000;
pub const SP_PC_REGS_LEN: usize = 0x10;

/// `SP_PC` — RSP instruction address (IMEM); writes mask to word-aligned (`& 0xFFC`).
pub const SP_PC_REG_PC: u32 = 0x00;
/// `SP_IBIST` — memory BIST (stub until RSP execution exists).
pub const SP_PC_REG_IBIST: u32 = 0x04;

/// Display Processor / RDP command interface ([n64brew: RDP](https://n64brew.dev/wiki/Reality_Display_Processor)).
pub const DPC_REGS_BASE: u32 = 0x0410_0000;
pub const DPC_REGS_LEN: usize = 0x20;

/// RDRAM/DMEM command range start (8-byte aligned address bits).
pub const DPC_REG_START: u32 = 0x00;
/// Exclusive end of RDP command DMA (`DPC_END` write kicks processing + `MI_INTR_DP` when non-zero).
pub const DPC_REG_END: u32 = 0x04;
/// DMA position (after a completed `DPC_END`, equals masked end in this stub).
pub const DPC_REG_CURRENT: u32 = 0x08;
/// `DPC_STATUS` — busy / pending flags ([n64brew DPC_STATUS](https://n64brew.dev/wiki/Reality_Display_Processor/Interface)).
pub const DPC_REG_STATUS: u32 = 0x0C;
pub const DPC_REG_CLOCK: u32 = 0x10;
pub const DPC_REG_BUF_BUSY: u32 = 0x14;
pub const DPC_REG_PIPE_BUSY: u32 = 0x18;
pub const DPC_REG_TMEM_BUSY: u32 = 0x1C;

/// Status bits (low 11 bits; read side). See wiki — we only model a subset for bring-up.
pub const DPC_STATUS_XBUS_DMEM: u32 = 1 << 0;
pub const DPC_STATUS_TMEM_BUSY: u32 = 1 << 4;
pub const DPC_STATUS_PIPE_BUSY: u32 = 1 << 5;
pub const DPC_STATUS_CMD_BUSY: u32 = 1 << 6;
pub const DPC_STATUS_CBUF_READY: u32 = 1 << 7;
pub const DPC_STATUS_DMA_BUSY: u32 = 1 << 8;
pub const DPC_STATUS_END_PENDING: u32 = 1 << 9;
pub const DPC_STATUS_START_PENDING: u32 = 1 << 10;

#[inline]
fn dpc_addr_mask(v: u32) -> u32 {
    v & 0x00FF_FFF8
}

#[derive(Debug)]
pub struct DpcRegs {
    pub start: u32,
    pub end: u32,
    pub current: u32,
    /// Raw `DPC_STATUS` read value; no GPU — stays idle after each instant `DPC_END` completion.
    pub status: u32,
}

impl DpcRegs {
    pub fn new() -> Self {
        Self {
            start: 0,
            end: 0,
            current: 0,
            status: 0,
        }
    }

    fn reg_offset(paddr: u32) -> Option<u32> {
        if (DPC_REGS_BASE..DPC_REGS_BASE + DPC_REGS_LEN as u32).contains(&paddr) && paddr & 3 == 0 {
            Some((paddr.wrapping_sub(DPC_REGS_BASE)) & 0x1F)
        } else {
            None
        }
    }

    pub fn read(&self, paddr: u32) -> u32 {
        let Some(off) = Self::reg_offset(paddr) else {
            return 0;
        };
        match off {
            DPC_REG_START => dpc_addr_mask(self.start),
            DPC_REG_END => dpc_addr_mask(self.end),
            DPC_REG_CURRENT => dpc_addr_mask(self.current),
            DPC_REG_STATUS => self.status,
            DPC_REG_CLOCK | DPC_REG_BUF_BUSY | DPC_REG_PIPE_BUSY | DPC_REG_TMEM_BUSY => 0,
            _ => 0,
        }
    }

    pub fn write(&mut self, paddr: u32, value: u32, mi: &mut Mi) {
        let Some(off) = Self::reg_offset(paddr) else {
            return;
        };
        match off {
            DPC_REG_START => {
                self.start = dpc_addr_mask(value);
                self.status |= DPC_STATUS_START_PENDING;
            }
            DPC_REG_END => {
                self.end = dpc_addr_mask(value);
                if value != 0 {
                    self.current = self.end;
                    self.status &= !(DPC_STATUS_DMA_BUSY
                        | DPC_STATUS_END_PENDING
                        | DPC_STATUS_START_PENDING
                        | DPC_STATUS_CMD_BUSY
                        | DPC_STATUS_PIPE_BUSY
                        | DPC_STATUS_TMEM_BUSY);
                    mi.raise(MI_INTR_DP);
                }
            }
            DPC_REG_CURRENT => {}
            DPC_REG_STATUS => {
                self.status_write(value);
            }
            _ => {}
        }
    }

    /// `DPC_STATUS` writes: paired set/clear for DMEM vs RDRAM command source ([n64brew DPC_STATUS](https://n64brew.dev/wiki/Reality_Display_Processor/Interface)).
    fn status_write(&mut self, v: u32) {
        if (v & 1) != 0 && (v & 2) == 0 {
            self.status &= !DPC_STATUS_XBUS_DMEM;
        }
        if (v & 2) != 0 && (v & 1) == 0 {
            self.status |= DPC_STATUS_XBUS_DMEM;
        }
    }
}

impl Default for DpcRegs {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mi::Mi;

    #[test]
    fn sp_dma_decode_single_line_matches_len_minus_one() {
        assert_eq!(sp_dma_decode(3), (4, 1, 0));
        assert_eq!(sp_dma_decode(0), (1, 1, 0));
        assert_eq!(sp_dma_decode(0x200), (0x201, 1, 0));
    }

    #[test]
    fn sp_dma_decode_multi_line_and_skip() {
        assert_eq!(sp_dma_decode(0x0000_1003), (4, 2, 0));
        // Upper bits set skip; `>> 20` then `& 0xFF8` yields 16 for `0x0100_0000`.
        assert_eq!(sp_dma_decode(0x0100_0000), (1, 1, 16));
    }

    #[test]
    fn sp_dma_end_addresses_single_transfer() {
        let (m, d) = sp_dma_end_addresses(0x0400_0000, 0x0000_0100, 4, 1, 0);
        assert_eq!(m, 0x0400_0004);
        assert_eq!(d, 0x0000_0104);
    }

    #[test]
    fn dpc_end_raises_dp() {
        let mut dpc = DpcRegs::new();
        let mut mi = Mi::new();
        mi.mask = MI_INTR_DP;
        dpc.write(DPC_REGS_BASE + DPC_REG_END, 0x80123456, &mut mi);
        assert!(mi.cpu_irq_pending());
    }

    #[test]
    fn dpc_end_sets_current_instant_complete() {
        let mut dpc = DpcRegs::new();
        let mut mi = Mi::new();
        dpc.write(DPC_REGS_BASE + DPC_REG_START, 0x8012_3000, &mut mi);
        dpc.write(DPC_REGS_BASE + DPC_REG_END, 0x8012_3458, &mut mi);
        assert_eq!(dpc.read(DPC_REGS_BASE + DPC_REG_CURRENT), 0x8012_3458 & 0x00FF_FFF8);
        assert_eq!(
            dpc.read(DPC_REGS_BASE + DPC_REG_STATUS) & DPC_STATUS_DMA_BUSY,
            0
        );
    }

    #[test]
    fn dpc_status_xbus_dma_toggle() {
        let mut dpc = DpcRegs::new();
        let mut mi = Mi::new();
        dpc.write(DPC_REGS_BASE + DPC_REG_STATUS, 2, &mut mi);
        assert_ne!(dpc.read(DPC_REGS_BASE + DPC_REG_STATUS) & DPC_STATUS_XBUS_DMEM, 0);
        dpc.write(DPC_REGS_BASE + DPC_REG_STATUS, 1, &mut mi);
        assert_eq!(dpc.read(DPC_REGS_BASE + DPC_REG_STATUS) & DPC_STATUS_XBUS_DMEM, 0);
    }
}
