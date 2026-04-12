//! RCP master-clock timing (NTSC).
//!
//! The VR4300 and RCP I/O blocks share a **93.75 MHz** system clock on NTSC units. [`Machine`](crate::Machine)
//! advances [`SystemBus::rcp_advance_dma_in_flight`](crate::bus::SystemBus::rcp_advance_dma_in_flight),
//! [`SystemBus::drain_deferred_cycles`](crate::bus::SystemBus::drain_deferred_cycles), and
//! [`Vi::advance`](crate::vi::Vi::advance) in **RCP cycles** so the VI frame counter, PI/SI/AI DMA,
//! and other deferred work share one timeline.
//!
//! Values here are **first-order** models: PI cart throughput follows published ~5 MiB/s averages;
//! SI and AI use conservative stand-ins until measured against hardware or test ROMs (e.g.
//! [n64_pi_dma_test](https://github.com/rasky/n64_pi_dma_test)).

use crate::rcp::sp_dma_decode;

/// NTSC CPU / RCP I/O master frequency (Hz).
pub const RCP_MASTER_HZ_NTSC: u64 = 93_750_000;

// --- Video Interface (VI) ----------------------------------------------------

/// Vertical interrupt period for ~59.94 Hz NTSC: matches `93_750_000 / 59.940059…` within integer rounding.
pub const VI_NTSC_CYCLES_PER_FRAME: u64 = 1_564_062;

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

// --- Serial Interface (SI), 64-byte PIF block -------------------------------

/// RCP cycles per byte for SI PIF DMA (stub: order-of-magnitude; PIF is slower than PI cart).
pub const SI_DMA_CYCLES_PER_BYTE: u64 = 10;

/// Total RCP cycles for one 64-byte SI DMA (`RD64B` / `WR64B`).
pub const SI_DMA_64_BLOCK_CYCLES: u64 = 64 * SI_DMA_CYCLES_PER_BYTE;

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
    fn ai_buffer_cycles_order_of_one_frame_44k_stereo() {
        let one_sec = ai_pcm_buffer_cycles(AI_PCM_BYTES_PER_SEC as u32);
        assert_eq!(one_sec, RCP_MASTER_HZ_NTSC);
    }

    #[test]
    fn sp_dma_four_bytes_includes_overhead() {
        assert!(sp_rsp_dma_total_cycles(3) >= 40);
    }
}
