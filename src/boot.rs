//! HLE bootstrap helpers and ROM header parsing (no real PIF firmware).

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

/// Boot PC from ROM header word at **0x0C** (“Boot address” / PC), per [n64brew ROM header](https://n64brew.dev/wiki/ROM_Header).
pub fn cart_boot_pc(rom: &[u8]) -> Option<u64> {
    let w = rom_u32_be(rom, 0x0C)?;
    if w == 0 {
        return Some(DEFAULT_GAME_ENTRY_PC);
    }
    Some(sign_extend_word32(w))
}

/// Initial stack pointer many games expect after bootstrapping.
pub const DEFAULT_GAME_SP: u64 = 0xFFFF_FFFF_801F_FFF0;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cart_boot_reads_header_word() {
        let mut rom = vec![0u8; 0x1000];
        rom[0x0C..0x10].copy_from_slice(&0x8000_1234u32.to_be_bytes());
        assert_eq!(cart_boot_pc(&rom), Some(sign_extend_word32(0x8000_1234)));
    }

    #[test]
    fn cart_boot_zero_falls_back() {
        let rom = vec![0u8; 0x1000];
        assert_eq!(cart_boot_pc(&rom), Some(DEFAULT_GAME_ENTRY_PC));
    }
}
