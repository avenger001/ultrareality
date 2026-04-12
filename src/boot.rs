//! ROM header parsing and IPL3 bootstrap using **PI DMA** (same mechanism as hardware after PIF hands off).
//!
//! For cold boot from real PIF firmware, use [`crate::Machine::bootstrap_from_pif_reset`] after loading a
//! PIF dump. The cart fast path ([`ipl3_load_via_pi_dma`]) takes the entry PC from the ROM header ([`ROM_OFF_BOOT_ADDRESS`])
//! and copies the IPL3-sized region via [`crate::pi::Pi`] DMA, raising [`crate::mi::MI_INTR_PI`] on
//! completion like hardware.

use crate::bus::{virt_to_phys, PhysicalMemory};
use crate::mi::Mi;
use crate::pi::{Pi, CART_DOM1_ADDR2_BASE};

/// Sign-extend a 32-bit address into the upper canonical MIPS III range.
#[inline]
pub fn sign_extend_word32(addr: u32) -> u64 {
    i64::from(addr as i32) as u64
}

/// Read big-endian `u32` at byte offset in ROM.
#[inline]
pub fn rom_u32_be(rom: &[u8], offset: usize) -> Option<u32> {
    if offset + 4 > rom.len() {
        return None;
    }
    Some(u32::from_be_bytes(rom[offset..offset + 4].try_into().ok()?))
}

/// Typical game entry after IPL; used when the header word is zero.
pub const DEFAULT_GAME_ENTRY_PC: u64 = 0xFFFF_FFFF_8000_0400;

/// Byte offset of the **Boot Address** word in a big-endian ROM image ([n64brew ROM_Header](https://n64brew.dev/wiki/ROM_Header)).
pub const ROM_OFF_BOOT_ADDRESS: usize = 0x08;

/// ROM offset where IPL3 starts copying (after the 4 KiB header + IPL3 slot).
pub const IPL3_ROM_DMA_START: usize = 0x1000;
/// IPL3 copies **1 MiB** from [`IPL3_ROM_DMA_START`] into RDRAM at the boot address.
pub const IPL3_COPY_LEN: usize = 0x0010_0000;

/// Boot PC from ROM header word at [`ROM_OFF_BOOT_ADDRESS`] (Boot Address), per [n64brew ROM_Header](https://n64brew.dev/wiki/ROM_Header).
/// Word at **0x0C** is libultra version metadata, not the entry PC.
pub fn cart_boot_pc(rom: &[u8]) -> Option<u64> {
    let w = rom_u32_be(rom, ROM_OFF_BOOT_ADDRESS)?;
    if w == 0 {
        return Some(DEFAULT_GAME_ENTRY_PC);
    }
    Some(sign_extend_word32(w))
}

/// IPL3-style load: first **1 MiB** of game ROM from cart offset `0x1000` into RDRAM at the physical
/// address of the boot PC from the header (`0x08`), using PI “read” DMA (not a host `memcpy`).
/// The transfer is timed like hardware; [`Pi::dma_cart_segment_to_rdram`] fast-forwards the RCP clock
/// so RDRAM is filled before the CPU runs.
pub fn ipl3_load_via_pi_dma(pi: &mut Pi, rdram: &mut PhysicalMemory, mi: &mut Mi) -> Option<u64> {
    let pc = cart_boot_pc(&pi.rom)?;
    let dst_phys = virt_to_phys(pc)? as usize;

    let src_remain = pi.rom.len().saturating_sub(IPL3_ROM_DMA_START);
    if src_remain == 0 {
        return Some(pc);
    }
    let n = src_remain.min(IPL3_COPY_LEN);
    let dst_avail = rdram.data.len().saturating_sub(dst_phys);
    let copy_n = n.min(dst_avail);
    if copy_n == 0 {
        return Some(pc);
    }
    let cart_addr = CART_DOM1_ADDR2_BASE + IPL3_ROM_DMA_START as u32;
    pi.dma_cart_segment_to_rdram(rdram, cart_addr, dst_phys as u32, copy_n as u32, mi);
    Some(pc)
}

/// Initial stack pointer many games expect after bootstrapping.
pub const DEFAULT_GAME_SP: u64 = 0xFFFF_FFFF_801F_FFF0;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cart_boot_reads_header_word() {
        let mut rom = vec![0u8; 0x1000];
        rom[ROM_OFF_BOOT_ADDRESS..ROM_OFF_BOOT_ADDRESS + 4]
            .copy_from_slice(&0x8000_1234u32.to_be_bytes());
        assert_eq!(cart_boot_pc(&rom), Some(sign_extend_word32(0x8000_1234)));
    }

    #[test]
    fn ipl3_pi_dma_copies_after_header() {
        use crate::mi::MI_INTR_PI;
        use crate::pi::Pi;

        let mut rom = vec![0u8; 0x2000];
        rom[ROM_OFF_BOOT_ADDRESS..ROM_OFF_BOOT_ADDRESS + 4]
            .copy_from_slice(&0x8000_0400u32.to_be_bytes());
        rom[0x1000..0x1004].copy_from_slice(&0xDEAD_BEEFu32.to_be_bytes());

        let mut rdram = PhysicalMemory::new(4 * 1024 * 1024);
        let mut pi = Pi::with_rom(rom);
        let mut mi = crate::mi::Mi::new();
        let pc = ipl3_load_via_pi_dma(&mut pi, &mut rdram, &mut mi).unwrap();
        assert_eq!(pc, sign_extend_word32(0x8000_0400));
        assert_eq!(
            u32::from_be_bytes(rdram.data[0x400..0x404].try_into().unwrap()),
            0xDEAD_BEEF
        );
        assert_ne!(mi.intr & MI_INTR_PI, 0);
    }

    #[test]
    fn cart_boot_zero_falls_back() {
        let rom = vec![0u8; 0x1000];
        assert_eq!(cart_boot_pc(&rom), Some(DEFAULT_GAME_ENTRY_PC));
    }
}
