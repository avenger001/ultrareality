//! RCP master-clock timing (NTSC).
//!
//! ## Lockstep with [`crate::Machine`]
//!
//! Retail VR4300 and RCP I/O run from the same **93.75 MHz** crystal. The CPU interpreter retires
//! abstract “pipeline cycles” ([`crate::cycles`]); each [`crate::Machine::step`] adds those retired
//! cycles plus any deferred VI/RDP debt ([`crate::bus::SystemBus::drain_deferred_cycles`]), then
//! applies that **single** `delta` to every in-flight peripheral in parallel:
//! [`SystemBus::rcp_advance_dma_in_flight`](crate::bus::SystemBus::rcp_advance_dma_in_flight) and
//! [`SystemBus::advance_vi_frame_timing`](crate::bus::SystemBus::advance_vi_frame_timing). So
//! `master_cycles`, `Count`, PI/SI/AI/SP/DPC timers, and VI field time all advance on one RCP
//! timeline — **not** instant completion on register writes.
//!
//! Constants below are **RCP master cycles** (denominator [`RCP_MASTER_HZ_NTSC`]) unless noted.
//! PI cart throughput follows published ~5 MiB/s averages; SI PIF DMA uses the ~3.3 MiB/s figure from
//! [n64brew](https://n64brew.dev/wiki/Serial_Interface); AI PCM duration matches 44.1 kHz stereo
//! 16‑bit consumption. SP display-list RDRAM traffic uses [`RDRAM_BUS_CYCLES_PER_BYTE`].

use crate::rcp::sp_dma_decode;

/// NTSC CPU / RCP I/O master frequency (Hz).
pub const RCP_MASTER_HZ_NTSC: u64 = 93_750_000;

/// RCP master-cycle count (93.75 MHz NTSC): same unit as [`crate::Machine::master_cycles`].
pub type MasterCycles = u64;

// --- Video Interface (VI) ----------------------------------------------------

/// Vertical interrupt period for ~59.94 Hz NTSC: `RCP_MASTER_HZ_NTSC * 1001 / 60000` (MPEG NTSC rate).
pub const VI_NTSC_CYCLES_PER_FRAME: u64 = RCP_MASTER_HZ_NTSC * 1001 / 60_000;

/// Active scanlines used to spread [`VI_V_INTR`](crate::vi::VI_REG_V_INTR) within a field (uniform stub).
pub const VI_NTSC_ACTIVE_SCANLINES: u64 = 262;

// --- Parallel Interface (PI), cartridge ROM DMA ------------------------------

/// Typical average Game Pak read throughput (~5 MiB/s per Nintendo documentation).
pub const PI_CART_ROM_BYTES_PER_SEC: u64 = 5 * 1024 * 1024;

/// RCP cycles charged per byte for PI cart → RDRAM DMA (ceiling of `RCP_HZ / throughput`).
pub const PI_ROM_DMA_CYCLES_PER_BYTE: u64 = (RCP_MASTER_HZ_NTSC + PI_CART_ROM_BYTES_PER_SEC - 1)
    / PI_CART_ROM_BYTES_PER_SEC;

#[inline]
pub fn pi_cart_dma_total_cycles(byte_len: u64) -> u64 {
    byte_len.saturating_mul(PI_ROM_DMA_CYCLES_PER_BYTE)
}

// --- RDRAM (bus occupancy for RDP list DMA / VI fetch; no Rambus serial model) ---

/// RCP cycles billed per RDRAM byte for coarse bandwidth (VI blit, RDP list read/write stub).
pub const RDRAM_BUS_CYCLES_PER_BYTE: u64 = 2;

// --- Serial Interface (SI), 64-byte PIF block -------------------------------

/// Documented effective PIF DMA throughput (~3.3 MiB/s, [n64brew: Serial Interface](https://n64brew.dev/wiki/Serial_Interface)).
pub const SI_PIF_DMA_BYTES_PER_SEC: u64 = 3_460_300;

/// RCP cycles per byte: ceil(`RCP_MASTER_HZ_NTSC` / [`SI_PIF_DMA_BYTES_PER_SEC`]).
pub const SI_DMA_CYCLES_PER_BYTE: u64 =
    (RCP_MASTER_HZ_NTSC + SI_PIF_DMA_BYTES_PER_SEC - 1) / SI_PIF_DMA_BYTES_PER_SEC;

/// Fixed cost before the 64-byte payload (address / control phase; first-order).
pub const SI_DMA_64_BLOCK_OVERHEAD_CYCLES: u64 = 48;

/// Total RCP cycles for one 64-byte SI DMA (`RD64B` / `WR64B`).
pub const SI_DMA_64_BLOCK_CYCLES: u64 =
    SI_DMA_64_BLOCK_OVERHEAD_CYCLES.saturating_add(64 * SI_DMA_CYCLES_PER_BYTE);

// --- Audio Interface (AI) ---------------------------------------------------

/// Assumed PCM consumption: 44.1 kHz, stereo, 16-bit (typical libultra audio).
pub const AI_PCM_BYTES_PER_SEC: u64 = 44_100 * 2 * 2;

/// RCP cycles corresponding to playing back `byte_len` bytes at [`AI_PCM_BYTES_PER_SEC`].
#[inline]
pub fn ai_pcm_buffer_cycles(byte_len: u32) -> u64 {
    let len = byte_len as u64;
    if len == 0 {
        return 0;
    }
    len.saturating_mul(RCP_MASTER_HZ_NTSC) / AI_PCM_BYTES_PER_SEC
}

// --- RSP SP DMA (RDRAM ↔ DMEM/IMEM) -----------------------------------------

/// Fixed cost before byte pipeline (rough; see hcs64 RSP DMA measurements).
pub const SP_RSP_DMA_OVERHEAD_CYCLES: u64 = 40;
/// RCP cycles per byte transferred (simplified linear model).
pub const SP_RSP_CYCLES_PER_BYTE: u64 = 1;

#[inline]
pub fn sp_rsp_dma_total_cycles(len_reg: u32) -> u64 {
    let (line_bytes, line_count, _) = sp_dma_decode(len_reg);
    let bytes = (line_bytes as u64).saturating_mul(line_count as u64);
    SP_RSP_DMA_OVERHEAD_CYCLES.saturating_add(bytes.saturating_mul(SP_RSP_CYCLES_PER_BYTE))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pi_cycles_per_byte_matches_throughput() {
        assert_eq!(PI_ROM_DMA_CYCLES_PER_BYTE, 18);
        assert_eq!(pi_cart_dma_total_cycles(1024), 1024 * 18);
    }

    #[test]
    fn si_dma_derived_from_pif_throughput() {
        assert_eq!(SI_DMA_CYCLES_PER_BYTE, 28);
        assert!(SI_DMA_64_BLOCK_CYCLES > 64 * SI_DMA_CYCLES_PER_BYTE);
    }

    #[test]
    fn ai_buffer_cycles_order_of_one_frame_44k_stereo() {
        let one_sec = ai_pcm_buffer_cycles(AI_PCM_BYTES_PER_SEC as u32);
        assert_eq!(one_sec, RCP_MASTER_HZ_NTSC);
    }

    #[test]
    fn sp_dma_four_bytes_includes_overhead() {
        assert!(sp_rsp_dma_total_cycles(3) >= 40);
    }
}
