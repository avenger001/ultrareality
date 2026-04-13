//! PIF (Peripheral Interface): boot ROM and 64-byte “scratch” RAM.
//!
//! Physical layout: ROM `0x1FC00000`–`0x1FC007BF`, RAM `0x1FC007C0`–`0x1FC007FF`.

pub const PIF_ROM_START: u32 = 0x1FC0_0000;
pub const PIF_ROM_LEN: usize = 0x7C0;
/// VR4300 reset fetch in **kseg1** (uncached PIF ROM), canonical 64-bit PC.
pub const PIF_KSEG1_RESET_PC: u64 = 0xFFFF_FFFF_BFC0_0000;
pub const PIF_RAM_START: u32 = 0x1FC0_07C0;
pub const PIF_RAM_LEN: usize = 64;
pub const PIF_WINDOW_END: u32 = 0x1FC0_0800;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PifRomLoadError {
    /// Fewer than [`PIF_ROM_LEN`] bytes (hardware PIF is fixed size).
    TooShort { got: usize, need: usize },
}

/// Controller port state for Joybus responses.
#[derive(Debug, Clone)]
pub struct ControllerState {
    /// Whether a controller is connected to this port.
    pub present: bool,
    /// Button/stick state: bits 31-16 = buttons, bits 15-8 = X axis, bits 7-0 = Y axis.
    pub buttons: u32,
}

impl Default for ControllerState {
    fn default() -> Self {
        Self {
            present: false,
            buttons: 0,
        }
    }
}

/// EEPROM size: 4 kbit = 512 bytes (SM64 uses this type).
pub const EEPROM_SIZE: usize = 512;

#[derive(Debug)]
pub struct Pif {
    pub rom: Box<[u8]>,
    pub ram: [u8; PIF_RAM_LEN],
    /// Controller state for ports 0–3.
    pub controllers: [ControllerState; 4],
    /// EEPROM storage (4 kbit = 512 bytes).
    pub eeprom: [u8; EEPROM_SIZE],
    /// Whether EEPROM is present (detected via osEepromProbe).
    pub eeprom_present: bool,
}

impl Pif {
    pub fn new() -> Self {
        let mut controllers: [ControllerState; 4] = Default::default();
        // Port 0 has a controller by default (SM64 requires at least one)
        controllers[0].present = true;
        Self {
            rom: vec![0u8; PIF_ROM_LEN].into_boxed_slice(),
            ram: [0u8; PIF_RAM_LEN],
            controllers,
            eeprom: [0u8; EEPROM_SIZE],
            eeprom_present: true, // SM64 expects EEPROM for save data
        }
    }

    /// Replace boot ROM contents. Longer dumps (e.g. 2048-byte files) use the first [`PIF_ROM_LEN`] bytes.
    pub fn replace_rom(&mut self, data: &[u8]) -> Result<(), PifRomLoadError> {
        if data.len() < PIF_ROM_LEN {
            return Err(PifRomLoadError::TooShort {
                got: data.len(),
                need: PIF_ROM_LEN,
            });
        }
        self.rom.copy_from_slice(&data[..PIF_ROM_LEN]);
        Ok(())
    }

    #[inline]
    pub fn paddr_index(paddr: u32) -> Option<usize> {
        match paddr {
            PIF_ROM_START..0x1FC0_07C0 => Some((paddr - PIF_ROM_START) as usize),
            PIF_RAM_START..PIF_WINDOW_END => {
                Some(PIF_ROM_LEN + (paddr - PIF_RAM_START) as usize)
            }
            _ => None,
        }
    }

    pub fn read_u8(&self, paddr: u32) -> Option<u8> {
        let i = Self::paddr_index(paddr)?;
        if i < PIF_ROM_LEN {
            return Some(self.rom[i]);
        }
        let ri = i - PIF_ROM_LEN;
        Some(self.ram[ri])
    }

    pub fn write_u8(&mut self, paddr: u32, v: u8) {
        let Some(i) = Self::paddr_index(paddr) else {
            return;
        };
        if i < PIF_ROM_LEN {
            return;
        }
        let ri = i - PIF_ROM_LEN;
        self.ram[ri] = v;
    }

    pub fn read_u32(&self, paddr: u32) -> Option<u32> {
        if paddr & 3 != 0 {
            return None;
        }
        let b0 = self.read_u8(paddr)?;
        let b1 = self.read_u8(paddr + 1)?;
        let b2 = self.read_u8(paddr + 2)?;
        let b3 = self.read_u8(paddr + 3)?;
        Some(u32::from_be_bytes([b0, b1, b2, b3]))
    }

    pub fn write_u32(&mut self, paddr: u32, v: u32) {
        if paddr & 3 != 0 {
            return;
        }
        for (k, byte) in v.to_be_bytes().iter().enumerate() {
            self.write_u8(paddr + k as u32, *byte);
        }
    }

    /// Process Joybus commands in PIF RAM after an SI WR64B DMA completes.
    ///
    /// PIF RAM layout is a sequence of command blocks:
    /// - Byte 0: TX byte count (data bytes to send to device)
    /// - Byte 1: RX byte count (expected response bytes)
    /// - Bytes 2..2+TX: TX data (command bytes from CPU)
    /// - Bytes 2+TX..2+TX+RX: RX data (response, filled by PIF)
    ///
    /// Special values in the TX-count slot:
    /// - `0xFE`: end of command stream.
    /// - `0xFD`: channel reset/skip — advance channel, consume 1 byte.
    /// - `0xFF`: alignment padding — consume 1 byte, do NOT advance channel.
    /// - `0x00`: NOP for this channel — advance channel by 1, consume 1 byte.
    ///
    /// libultra's `__osEepStatus` inserts a `0xFF` pad before the EEPROM command block
    /// (after the 4 controller skip bytes) to word-align the transmit data. Without
    /// handling this pad, the parser misreads `0xFF` as `tx_count = 63` and aborts,
    /// leaving the EEPROM status RX bytes unwritten — libultra then sees type != 0x8000
    /// and bails without releasing `__osSiAccessQueue`, hanging the game thread.
    pub fn process_commands(&mut self) {
        let mut channel = 0usize; // controller port index
        let mut i = 0usize;

        while i < PIF_RAM_LEN - 1 {
            let tx = self.ram[i];

            // End marker
            if tx == 0xFE {
                break;
            }
            // Skip marker (advance channel but no command)
            if tx == 0xFD {
                channel += 1;
                i += 1;
                continue;
            }
            // Alignment padding — consume one byte but do not advance channel.
            if tx == 0xFF {
                i += 1;
                continue;
            }
            // Zero TX = NOP for this channel, advance channel (do NOT treat as end).
            // libultra emits 4 consecutive 0x00 bytes to skip controller ports 0-3 before
            // the EEPROM command on channel 4; misreading this as end-of-stream means
            // channel 4 never gets processed and the EEPROM RX bytes stay as 0xFF, which
            // hangs __osEepStatus.
            let rx = self.ram.get(i + 1).copied().unwrap_or(0);
            if tx == 0 {
                channel += 1;
                i += 1;
                continue;
            }

            let tx_count = (tx & 0x3F) as usize;
            let rx_count = (rx & 0x3F) as usize;

            // Bounds check
            let cmd_start = i + 2;
            let rx_start = cmd_start + tx_count;
            let block_end = rx_start + rx_count;
            if block_end > PIF_RAM_LEN {
                break;
            }

            // Read the command byte
            let cmd = if tx_count > 0 { self.ram[cmd_start] } else { 0xFF };

            if channel < 4 {
                self.handle_controller_command(channel, cmd, rx_count, rx_start, i + 1);
            } else if channel == 4 {
                // Channel 4: EEPROM
                self.handle_eeprom_command(cmd, tx_count, rx_count, cmd_start, rx_start, i + 1);
            }
            // Channel 5+ ignored

            channel += 1;
            i = block_end;
        }

        // Set the PIF status byte (byte 63) to signal completion
        // Bit 7 of byte 63: clear the "execute commands" flag
        self.ram[PIF_RAM_LEN - 1] = 0;
    }

    fn handle_controller_command(&mut self, port: usize, cmd: u8, rx_count: usize, rx_start: usize, rx_count_byte_idx: usize) {
        if !self.controllers[port].present {
            // No controller: set error flag (bit 7 of RX count byte)
            self.ram[rx_count_byte_idx] |= 0x80;
            return;
        }

        match cmd {
            0x00 | 0xFF => {
                // Status / Reset: respond with controller type
                // [0x05, 0x00] = standard controller, [0x01] = pak status (no pak)
                if rx_count >= 3 {
                    self.ram[rx_start] = 0x05;     // type high: standard controller
                    self.ram[rx_start + 1] = 0x00; // type low
                    self.ram[rx_start + 2] = 0x01; // status: no pak inserted
                }
            }
            0x01 => {
                // Read buttons / stick
                if rx_count >= 4 {
                    let state = self.controllers[port].buttons;
                    self.ram[rx_start] = (state >> 24) as u8;     // buttons high
                    self.ram[rx_start + 1] = (state >> 16) as u8; // buttons low
                    self.ram[rx_start + 2] = (state >> 8) as u8;  // X axis
                    self.ram[rx_start + 3] = state as u8;          // Y axis
                }
            }
            0x02 => {
                // Read controller pak (stub: return zeros)
                for j in 0..rx_count {
                    self.ram[rx_start + j] = 0;
                }
            }
            0x03 => {
                // Write controller pak (stub: ignore)
            }
            _ => {
                // Unknown command
            }
        }
    }

    fn handle_eeprom_command(&mut self, cmd: u8, tx_count: usize, rx_count: usize, cmd_start: usize, rx_start: usize, rx_count_byte_idx: usize) {
        if !self.eeprom_present {
            // No EEPROM: set error flag
            self.ram[rx_count_byte_idx] |= 0x80;
            return;
        }

        match cmd {
            0x00 => {
                // EEPROM status/probe: respond with device type ID.
                // libultra reads this as a big-endian halfword and compares against
                // `CONT_EEPROM = 0x8000`, so the first byte must be 0x80 (4 kbit) and
                // the second byte 0x00. For 16 kbit EEPROM it would be 0x00, 0xC0.
                if rx_count >= 3 {
                    self.ram[rx_start] = 0x80;     // ID byte 0 = 4 kbit EEPROM
                    self.ram[rx_start + 1] = 0x00; // ID byte 1
                    self.ram[rx_start + 2] = 0x00; // status: ready, not busy
                }
            }
            0x04 => {
                // Read EEPROM: address in TX byte 1, read 8 bytes
                let addr = if tx_count >= 2 {
                    (self.ram[cmd_start + 1] as usize) * 8
                } else {
                    0
                };
                // Read 8 bytes from EEPROM
                for j in 0..rx_count.min(8) {
                    let eeprom_addr = (addr + j) % EEPROM_SIZE;
                    self.ram[rx_start + j] = self.eeprom[eeprom_addr];
                }
            }
            0x05 => {
                // Write EEPROM: address in TX byte 1, write 8 bytes from TX bytes 2-9
                let addr = if tx_count >= 2 {
                    (self.ram[cmd_start + 1] as usize) * 8
                } else {
                    0
                };
                // Write up to 8 bytes to EEPROM
                for j in 0..(tx_count.saturating_sub(2)).min(8) {
                    let eeprom_addr = (addr + j) % EEPROM_SIZE;
                    self.eeprom[eeprom_addr] = self.ram[cmd_start + 2 + j];
                }
                // Return a status byte (0 = success)
                if rx_count > 0 {
                    self.ram[rx_start] = 0x00;
                }
            }
            _ => {
                // Unknown EEPROM command
            }
        }
    }
}

impl Default for Pif {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replace_rom_accepts_exact_len() {
        let mut p = Pif::new();
        let mut blob = vec![0x5Au8; PIF_ROM_LEN];
        blob[0] = 0x12;
        p.replace_rom(&blob).unwrap();
        assert_eq!(p.rom[0], 0x12);
        assert_eq!(p.rom[PIF_ROM_LEN - 1], 0x5A);
    }

    #[test]
    fn replace_rom_truncates_long_dump() {
        let mut p = Pif::new();
        let mut blob = vec![0u8; PIF_ROM_LEN + 64];
        blob[PIF_ROM_LEN] = 0xEE;
        p.replace_rom(&blob).unwrap();
        assert_eq!(p.rom[PIF_ROM_LEN - 1], 0);
    }

    #[test]
    fn replace_rom_rejects_short() {
        let mut p = Pif::new();
        let r = p.replace_rom(&[1, 2, 3]);
        assert_eq!(
            r,
            Err(PifRomLoadError::TooShort {
                got: 3,
                need: PIF_ROM_LEN
            })
        );
    }

    #[test]
    fn joybus_status_query() {
        let mut pif = Pif::new();
        // Controller 0 is present by default
        // Build a status command in PIF RAM:
        // TX=1 (send 1 byte: cmd 0x00), RX=3 (expect 3 response bytes)
        pif.ram[0] = 1; // TX count
        pif.ram[1] = 3; // RX count
        pif.ram[2] = 0x00; // command: status
        pif.ram[3] = 0; // RX byte 0 (to be filled)
        pif.ram[4] = 0; // RX byte 1
        pif.ram[5] = 0; // RX byte 2
        pif.ram[6] = 0xFE; // end marker
        pif.process_commands();
        assert_eq!(pif.ram[3], 0x05, "type high = standard controller");
        assert_eq!(pif.ram[4], 0x00, "type low");
        assert_eq!(pif.ram[5], 0x01, "status: no pak");
    }

    #[test]
    fn joybus_poll_returns_button_state() {
        let mut pif = Pif::new();
        pif.controllers[0].buttons = 0x8000_4020; // A button + some stick
        pif.ram[0] = 1;    // TX=1
        pif.ram[1] = 4;    // RX=4
        pif.ram[2] = 0x01; // command: poll
        pif.ram[7] = 0xFE;
        pif.process_commands();
        assert_eq!(pif.ram[3], 0x80); // buttons high
        assert_eq!(pif.ram[4], 0x00); // buttons low
        assert_eq!(pif.ram[5], 0x40); // X axis
        assert_eq!(pif.ram[6], 0x20); // Y axis
    }

    #[test]
    fn joybus_no_controller_sets_error() {
        let mut pif = Pif::new();
        pif.controllers[0].present = false;
        pif.ram[0] = 1;
        pif.ram[1] = 3;
        pif.ram[2] = 0x00;
        pif.ram[6] = 0xFE;
        pif.process_commands();
        // RX count byte should have error bit set
        assert_ne!(pif.ram[1] & 0x80, 0, "error flag should be set for absent controller");
    }

    #[test]
    fn joybus_multiple_channels() {
        let mut pif = Pif::new();
        pif.controllers[0].present = true;
        pif.controllers[1].present = true;
        // Channel 0: status
        pif.ram[0] = 1;    // TX
        pif.ram[1] = 3;    // RX
        pif.ram[2] = 0x00; // status cmd
        // Response at [3..6]
        // Channel 1: status
        pif.ram[6] = 1;
        pif.ram[7] = 3;
        pif.ram[8] = 0x00;
        // Response at [9..12]
        pif.ram[12] = 0xFE;
        pif.process_commands();
        assert_eq!(pif.ram[3], 0x05, "ch0 type");
        assert_eq!(pif.ram[9], 0x05, "ch1 type");
    }
}
