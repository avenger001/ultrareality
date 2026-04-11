//! VR4300-style TLB: 32 dual-page entries; `PageMask` selects half-page size (Linux `PM_*` layout).

/// One TLB slot: shared `EntryHi` (VPN2 + ASID) and two `EntryLo` halves.
#[derive(Clone, Copy, Debug, Default)]
pub struct TlbEntry {
    pub page_mask: u32,
    pub hi: u32,
    pub lo0: u32,
    pub lo1: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MapFault {
    /// No matching TLB entry, or matched half `V = 0`.
    TlbMiss { store: bool },
    /// Store to a valid page with `D = 0`.
    TlbMod,
    /// Unmapped segment, unaligned access, or other address error.
    AddressError { store: bool },
    /// Physical address had no device / bus timeout (delivered as `AdEL`/`AdES` for now).
    BusError { store: bool },
}

/// Linux `PM_*` values (`arch/mips/include/asm/mipsregs.h`): mask with `0x01FFE000`.
/// Returns byte size of **one TLB half** (one `EntryLo`).
#[inline]
pub fn pagemask_half_size(pm: u32) -> u64 {
    match pm & 0x01ffe000 {
        0x0000_0000 => 4096,
        0x0000_2000 => 8192,
        0x0000_6000 => 16 * 1024,
        0x0000_e000 => 32 * 1024,
        0x0001_e000 => 64 * 1024,
        0x0003_e000 => 128 * 1024,
        0x0007_e000 => 256 * 1024,
        0x000f_e000 => 512 * 1024,
        0x001f_e000 => 1024 * 1024,
        0x003f_e000 => 2 * 1024 * 1024,
        0x007f_e000 => 4 * 1024 * 1024,
        0x00ff_e000 => 8 * 1024 * 1024,
        0x01ff_e000 => 16 * 1024 * 1024,
        0x03ff_e000 => 32 * 1024 * 1024,
        0x07ff_e000 => 64 * 1024 * 1024,
        0x1fff_e000 => 256 * 1024 * 1024,
        0x7fff_e000 => 1024 * 1024 * 1024,
        _ => 4096,
    }
}

#[inline]
fn pair_size(pm: u32) -> u64 {
    pagemask_half_size(pm).saturating_mul(2)
}

#[inline]
fn entry_lo_pfn(lo: u32) -> u32 {
    (lo >> 6) & 0xFFFF_F
}

#[inline]
fn entry_lo_v(lo: u32) -> bool {
    (lo & 2) != 0
}

#[inline]
fn entry_lo_d(lo: u32) -> bool {
    (lo & 4) != 0
}

#[inline]
fn entry_lo_g(lo: u32) -> bool {
    (lo & 1) != 0
}

impl TlbEntry {
    /// `current_asid` comes from CP0 `EntryHi[7:0]` (executing context).
    pub fn matches(&self, vaddr: u32, current_asid: u8) -> bool {
        let ps = pair_size(self.page_mask);
        let ps_u = u32::try_from(ps).unwrap_or(0x2000);
        if ps == 0 {
            return false;
        }
        let align_mask = !(ps_u.wrapping_sub(1));
        if (vaddr & align_mask) != (self.hi & align_mask) {
            return false;
        }
        let asid = (self.hi & 0xFF) as u8;
        let lo0 = self.lo0;
        let lo1 = self.lo1;
        entry_lo_g(lo0) || entry_lo_g(lo1) || asid == current_asid
    }

    pub fn translate(
        &self,
        vaddr: u32,
        current_asid: u8,
        store: bool,
    ) -> Result<u32, MapFault> {
        let half = u32::try_from(pagemask_half_size(self.page_mask)).unwrap_or(0x1000);

        if !self.matches(vaddr, current_asid) {
            return Err(MapFault::TlbMiss { store });
        }

        let odd = (vaddr & half) != 0;
        let lo = if odd { self.lo1 } else { self.lo0 };
        if !entry_lo_v(lo) {
            return Err(MapFault::TlbMiss { store });
        }
        if store && !entry_lo_d(lo) {
            return Err(MapFault::TlbMod);
        }
        let pfn = entry_lo_pfn(lo);
        let off = vaddr & (half.wrapping_sub(1));
        Ok((pfn << 12) | off)
    }
}

/// Probe: return index `0..32` if any entry matches `EntryHi` VPN2 + ASID rules.
pub fn probe_index(entries: &[TlbEntry; 32], probe_hi: u32) -> Option<usize> {
    let key_asid = (probe_hi & 0xFF) as u8;
    for (i, e) in entries.iter().enumerate() {
        let ps = pair_size(e.page_mask);
        let ps_u = u32::try_from(ps).unwrap_or(0x2000);
        if ps == 0 {
            continue;
        }
        let align_mask = !(ps_u.wrapping_sub(1));
        if (probe_hi & align_mask) != (e.hi & align_mask) {
            continue;
        }
        let asid = (e.hi & 0xFF) as u8;
        if entry_lo_g(e.lo0) || entry_lo_g(e.lo1) || asid == key_asid {
            return Some(i);
        }
    }
    None
}
