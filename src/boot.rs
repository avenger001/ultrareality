//! HLE bootstrap helpers and ROM header parsing (no real PIF firmware).

use crate::bus::{virt_to_phys, PhysicalMemory};

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

/// ROM offset where IPL3 starts copying (after the 4 KiB header + IPL3 slot).
pub const IPL3_ROM_DMA_START: usize = 0x1000;
/// IPL3 copies **1 MiB** from [`IPL3_ROM_DMA_START`] into RDRAM at the boot address.
pub const IPL3_COPY_LEN: usize = 0x0010_0000;

/// Boot PC from ROM header word at **0x08** (Boot Address), per [n64brew ROM_Header](https://n64brew.dev/wiki/ROM_Header).
/// Word at **0x0C** is libultra version metadata, not the entry PC.
pub fn cart_boot_pc(rom: &[u8]) -> Option<u64> {
    let w = rom_u32_be(rom, 0x08)?;
    if w == 0 {
        return Some(DEFAULT_GAME_ENTRY_PC);
    }
    Some(sign_extend_word32(w))
}

/// HLE stand-in for IPL3: copy the first **1 MiB** of game code from ROM `0x1000` into RDRAM at
/// the physical address of the boot PC (same work as hardware after PIF). Returns the CPU entry PC.
pub fn hle_ipl3_load_rom(rom: &[u8], rdram: &mut PhysicalMemory) -> Option<u64> {
    let pc = cart_boot_pc(rom)?;
    let dst_phys = virt_to_phys(pc)? as usize;

    let src_remain = rom.len().saturating_sub(IPL3_ROM_DMA_START);
    if src_remain == 0 {
        return Some(pc);
    }
    let n = src_remain.min(IPL3_COPY_LEN);
    let dst_avail = rdram.data.len().saturating_sub(dst_phys);
    let copy_n = n.min(dst_avail);
    if copy_n == 0 {
        return Some(pc);
    }
    rdram.data[dst_phys..dst_phys + copy_n]
        .copy_from_slice(&rom[IPL3_ROM_DMA_START..IPL3_ROM_DMA_START + copy_n]);
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
        rom[0x08..0x0C].copy_from_slice(&0x8000_1234u32.to_be_bytes());
        assert_eq!(cart_boot_pc(&rom), Some(sign_extend_word32(0x8000_1234)));
    }

    #[test]
    fn hle_ipl3_copies_after_header() {
        let mut rom = vec![0u8; 0x2000];
        rom[0x08..0x0C].copy_from_slice(&0x8000_0400u32.to_be_bytes());
        rom[0x1000..0x1004].copy_from_slice(&0xDEAD_BEEFu32.to_be_bytes());

        let mut rdram = PhysicalMemory::new(4 * 1024 * 1024);
        let pc = hle_ipl3_load_rom(&rom, &mut rdram).unwrap();
        assert_eq!(pc, sign_extend_word32(0x8000_0400));
        assert_eq!(
            u32::from_be_bytes(rdram.data[0x400..0x404].try_into().unwrap()),
            0xDEAD_BEEF
        );
    }

    #[test]
    fn cart_boot_zero_falls_back() {
        let rom = vec![0u8; 0x1000];
        assert_eq!(cart_boot_pc(&rom), Some(DEFAULT_GAME_ENTRY_PC));
    }
}
