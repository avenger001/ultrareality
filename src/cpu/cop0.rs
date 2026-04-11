//! COP0 subset: Status, Cause, EPC — exceptions, **Compare / Count** timer (`Cause.IP7` / `Status.IM7`),
//! interrupt scaffolding, and a **32-entry TLB** (`kuseg` only; `kseg0`/`kseg1` stay direct-mapped).

use super::tlb::{MapFault, TlbEntry};

/// Status: interrupt enable (IE).
pub const STATUS_IE: u32 = 1 << 0;
/// Status: exception level (EXL).
pub const STATUS_EXL: u32 = 1 << 1;
/// Status: error level (ERL).
pub const STATUS_ERL: u32 = 1 << 2;
/// Status: bootstrap exception vectors (BEV).
pub const STATUS_BEV: u32 = 1 << 22;
/// Status: interrupt mask 7 — enables **Compare** / timer (`Cause.IP7`).
pub const STATUS_IM7: u32 = 1 << 15;

/// Cause: exception code field (bits 2–6).
pub const CAUSE_EXCCODE_SHIFT: u32 = 2;
pub const CAUSE_EXCCODE_MASK: u32 = 0x1F;

/// Exception code: interrupt.
pub const EXCCODE_INT: u32 = 0;
/// Exception code: system call (`SYSCALL`).
pub const EXCCODE_SYSCALL: u32 = 8;
/// Exception code: breakpoint (`BREAK`).
pub const EXCCODE_BP: u32 = 9;
/// TLB modification (store to clean page).
pub const EXCCODE_MOD: u32 = 1;
/// TLB miss on load or instruction fetch.
pub const EXCCODE_TLBL: u32 = 2;
/// TLB miss on store.
pub const EXCCODE_TLBS: u32 = 3;
/// Address error on load or instruction fetch.
pub const EXCCODE_ADEL: u32 = 4;
/// Address error on store.
pub const EXCCODE_ADES: u32 = 5;

/// Cause: interrupt pending 7 — timer (`Count` == `Compare`).
pub const CAUSE_IP7: u32 = 1 << 15;
/// Cause: branch delay — last exception was from a delay slot (`EPC` points at branch).
pub const CAUSE_BD: u32 = 1 << 31;

#[derive(Clone, Debug)]
pub struct Cop0 {
    pub status: u32,
    pub cause: u32,
    pub epc: u64,
    /// COP0 r30 — error return (ERET when `Status.ERL` is set).
    pub error_epc: u64,
    pub badvaddr: u64,
    pub compare: u32,
    pub count: u32,
    /// CP0 r0 — TLB index (`TLBWI` / `TLBR`); bit 31 set if `TLBP` missed.
    pub index: u32,
    /// CP0 r1 — lower bound for `TLBWR` random replacement.
    pub wired: u32,
    /// CP0 r2 / r3 — `EntryLo0` / `EntryLo1` staging for TLB ops.
    pub entry_lo0: u64,
    pub entry_lo1: u64,
    /// CP0 r5 — `PageMask` (Linux `PM_*` layout; drives TLB half-page size).
    pub page_mask: u32,
    /// CP0 r10 — `EntryHi` (VPN2, ASID).
    pub entry_hi: u64,
    /// Software TLB: 32 dual pages.
    pub tlb: [TlbEntry; 32],
    /// Instruction count for `Random` / `TLBWR` (not cycle-accurate vs. real Random decrement).
    random_tick: u64,
}

impl Cop0 {
    pub fn new() -> Self {
        Self {
            // Kernel mode, interrupts off — typical cold reset approximation.
            status: 0x7040_0004,
            cause: 0,
            epc: 0,
            error_epc: 0,
            badvaddr: 0,
            compare: 0,
            count: 0,
            index: 0,
            wired: 0,
            entry_lo0: 0,
            entry_lo1: 0,
            page_mask: 0,
            entry_hi: 0,
            tlb: [TlbEntry::default(); 32],
            random_tick: 0,
        }
    }

    #[inline]
    fn maps_via_tlb(v: u32) -> bool {
        v < 0x8000_0000 || (0xC000_0000..=0xDFFF_FFFF).contains(&v)
    }

    /// **`kseg0`/`kseg1`** are direct-mapped; **`kuseg`** and **`ksseg`** use the TLB.
    pub fn translate_virt(&self, vaddr: u64, store: bool) -> Result<u32, MapFault> {
        let v = vaddr as u32;
        if (0x8000_0000..=0xBFFF_FFFF).contains(&v) {
            return Ok(v & 0x1FFF_FFFF);
        }
        if Self::maps_via_tlb(v) {
            let asid = (self.entry_hi & 0xFF) as u8;
            for e in &self.tlb {
                match e.translate(v, asid, store) {
                    Ok(p) => return Ok(p),
                    Err(MapFault::TlbMod) => return Err(MapFault::TlbMod),
                    Err(MapFault::TlbMiss { .. }) => continue,
                    Err(MapFault::AddressError { .. }) | Err(MapFault::BusError { .. }) => unreachable!(),
                }
            }
            return Err(MapFault::TlbMiss { store });
        }
        Err(MapFault::AddressError { store })
    }

    #[inline]
    pub fn exccode_for_map_fault(fault: MapFault, instr_fetch: bool) -> u32 {
        match fault {
            MapFault::TlbMiss { store } => {
                if store {
                    EXCCODE_TLBS
                } else {
                    EXCCODE_TLBL
                }
            }
            MapFault::TlbMod => EXCCODE_MOD,
            MapFault::AddressError { store } => {
                if instr_fetch || !store {
                    EXCCODE_ADEL
                } else {
                    EXCCODE_ADES
                }
            }
            MapFault::BusError { store } => {
                if instr_fetch || !store {
                    EXCCODE_ADEL
                } else {
                    EXCCODE_ADES
                }
            }
        }
    }

    #[inline]
    pub fn random_index(&self) -> u32 {
        let w = self.wired.min(31);
        let span = 32u32.saturating_sub(w);
        if span == 0 {
            return 31;
        }
        w + ((self.random_tick % u64::from(span)) as u32)
    }

    #[inline]
    pub fn advance_random(&mut self) {
        self.random_tick = self.random_tick.wrapping_add(1);
    }

    /// Prepare CP0 regs after a TLB-related address fault (before `EPC` is written by the caller).
    pub fn set_tlb_fault_regs(&mut self, vaddr: u64) {
        self.badvaddr = vaddr;
        let vpn2 = (vaddr >> 13) << 13;
        self.entry_hi = vpn2 | (self.entry_hi & 0xFF);
    }

    pub fn tlb_read(&mut self) {
        let i = (self.index & 0x1F) as usize;
        let e = self.tlb[i];
        self.page_mask = e.page_mask;
        self.entry_hi = u64::from(e.hi);
        self.entry_lo0 = u64::from(e.lo0);
        self.entry_lo1 = u64::from(e.lo1);
    }

    pub fn tlb_write_indexed(&mut self) {
        let i = (self.index & 0x1F) as usize;
        self.tlb[i] = TlbEntry {
            page_mask: self.page_mask,
            hi: self.entry_hi as u32,
            lo0: self.entry_lo0 as u32,
            lo1: self.entry_lo1 as u32,
        };
    }

    pub fn tlb_write_random(&mut self) {
        let idx = self.random_index();
        self.index = (self.index & !0x1F) | (idx & 0x1F);
        self.tlb_write_indexed();
    }

    /// `TLBP`: set `Index` to matching entry or `1 << 31` on miss.
    pub fn tlb_probe(&mut self) {
        let eh = self.entry_hi as u32;
        if let Some(i) = super::tlb::probe_index(&self.tlb, eh) {
            self.index = (self.index & 0x8000_0000) | (i as u32 & 0x1F);
        } else {
            self.index = 0x8000_0000;
        }
    }

    /// True if interrupts are globally enabled and not blocked by EXL/ERL.
    #[inline]
    pub fn interrupts_enabled(&self) -> bool {
        (self.status & STATUS_IE) != 0
            && (self.status & STATUS_EXL) == 0
            && (self.status & STATUS_ERL) == 0
    }

    /// Vector for external interrupt: `0x80000180` (cached) or `0xBFC00380` (BEV).
    #[inline]
    pub fn interrupt_vector(&self) -> u64 {
        if (self.status & STATUS_BEV) != 0 {
            0xFFFF_FFFF_BFC0_0380u64
        } else {
            0xFFFF_FFFF_8000_0180u64
        }
    }

    /// General exception vector (TLB refill / mod / `SYSCALL` / `BREAK` / …): same base as interrupts.
    #[inline]
    pub fn general_exception_vector(&self) -> u64 {
        self.interrupt_vector()
    }

    /// Record an interrupt exception before redirecting `PC` (caller sets `pc`).
    pub fn enter_interrupt_exception(&mut self, epc: u64) {
        self.epc = epc;
        self.cause = (self.cause
            & !(CAUSE_EXCCODE_MASK << CAUSE_EXCCODE_SHIFT)
            & !CAUSE_BD)
            | (EXCCODE_INT << CAUSE_EXCCODE_SHIFT);
        self.status |= STATUS_EXL;
    }

    /// Synchronous exception (`SYSCALL`, `BREAK`, trap, …): sets `EPC`, `Cause.ExcCode`, `Status.EXL`.
    ///
    /// `branch_delay`: if true, sets `Cause.BD` (`EPC` is the branch/jump PC, not the delay slot).
    pub fn enter_general_exception(&mut self, epc: u64, exccode: u32, branch_delay: bool) {
        self.epc = epc;
        self.cause = (self.cause & !(CAUSE_EXCCODE_MASK << CAUSE_EXCCODE_SHIFT))
            | ((exccode & CAUSE_EXCCODE_MASK) << CAUSE_EXCCODE_SHIFT);
        if branch_delay {
            self.cause |= CAUSE_BD;
        } else {
            self.cause &= !CAUSE_BD;
        }
        self.status |= STATUS_EXL;
    }

    /// Timer IRQ pending in `Cause` and allowed by `Status.IM7` (does not check `IE` / `EXL` / `ERL`).
    #[inline]
    pub fn timer_interrupt_pending_masked(&self) -> bool {
        (self.cause & CAUSE_IP7) != 0 && (self.status & STATUS_IM7) != 0
    }

    /// Clear timer pending after delivering an interrupt exception (hardware clears edge).
    #[inline]
    pub fn clear_timer_interrupt_pending(&mut self) {
        self.cause &= !CAUSE_IP7;
    }

    /// Advance `Count` (CP0 r9); if it becomes `Compare` during this interval, set `Cause.IP7`.
    pub fn advance_count_wrapped(&mut self, delta: u64) {
        if delta == 0 {
            return;
        }
        let cmp = self.compare;
        let old = self.count;
        if compare_matched_in_u32_interval(old, delta, cmp) {
            self.cause |= CAUSE_IP7;
        }
        self.count = (old as u128 + delta as u128) as u32;
    }

    /// `ERET` return path (caller assigns `self.pc`).
    #[inline]
    pub fn apply_eret(&mut self) -> u64 {
        self.cause &= !CAUSE_BD;
        if (self.status & STATUS_ERL) != 0 {
            self.status &= !STATUS_ERL;
            self.error_epc
        } else {
            self.status &= !STATUS_EXL;
            self.epc
        }
    }

    pub fn read_32(&self, reg: u32) -> u32 {
        match reg {
            0 => self.index,
            1 => self.random_index() & 0x3F,
            2 => self.entry_lo0 as u32,
            3 => self.entry_lo1 as u32,
            5 => self.page_mask,
            6 => self.wired,
            8 => self.badvaddr as u32,
            // CP0 r9 — Count (not high half of BadVAddr; use `DMFC0` / 64-bit paths for full BadVAddr).
            9 => self.count,
            10 => self.entry_hi as u32,
            11 => self.compare,
            12 => self.status,
            13 => self.cause,
            14 => self.epc as u32,
            15 => {
                // PRId — VR4300 (games read COP0 r15; not EPC high).
                0x0B00_0002
            }
            30 => self.error_epc as u32,
            31 => (self.error_epc >> 32) as u32,
            _ => 0,
        }
    }

    pub fn write_32(&mut self, reg: u32, value: u32) {
        match reg {
            0 => self.index = value & 0x8000_001F,
            2 => self.entry_lo0 = u64::from(value),
            3 => self.entry_lo1 = u64::from(value),
            5 => self.page_mask = value,
            6 => self.wired = value.min(31),
            9 => self.count = value,
            10 => self.entry_hi = (self.entry_hi & !0xFFFF_FFFF) | u64::from(value),
            // Writing `Compare` clears the timer interrupt (VR4300).
            11 => {
                self.compare = value;
                self.cause &= !CAUSE_IP7;
            }
            12 => self.status = value,
            13 => self.cause = value,
            14 => self.epc = (self.epc & !0xFFFF_FFFF) | u64::from(value),
            15 => self.epc = (self.epc & 0xFFFF_FFFF) | (u64::from(value) << 32),
            30 => self.error_epc = (self.error_epc & !0xFFFF_FFFF) | u64::from(value),
            31 => self.error_epc = (self.error_epc & 0xFFFF_FFFF) | (u64::from(value) << 32),
            _ => {}
        }
    }

    /// `DMFC0` — full 64-bit read (MIPS III).
    pub fn read_xpr64(&self, reg: u32) -> u64 {
        match reg {
            0 => u64::from(self.index),
            1 => u64::from(self.read_32(1)),
            2 => self.entry_lo0,
            3 => self.entry_lo1,
            5 => u64::from(self.page_mask),
            6 => u64::from(self.wired),
            8 => self.badvaddr,
            9 => u64::from(self.count),
            10 => self.entry_hi,
            11 => u64::from(self.compare),
            12 => u64::from(self.status),
            13 => u64::from(self.cause),
            14 => self.epc,
            15 => u64::from(self.read_32(15)),
            30 => self.error_epc,
            31 => (self.error_epc >> 32) as u64,
            _ => u64::from(self.read_32(reg)),
        }
    }

    /// `DMTC0` — full 64-bit write (MIPS III).
    pub fn write_xpr64(&mut self, reg: u32, v: u64) {
        match reg {
            0 => self.index = (v as u32) & 0x8000_001F,
            2 => self.entry_lo0 = v,
            3 => self.entry_lo1 = v,
            5 => self.page_mask = v as u32,
            6 => self.wired = (v as u32).min(31),
            8 => self.badvaddr = v,
            9 => self.count = v as u32,
            10 => self.entry_hi = v,
            11 => {
                self.compare = v as u32;
                self.cause &= !CAUSE_IP7;
            }
            12 => self.status = v as u32,
            13 => self.cause = v as u32,
            14 => self.epc = v,
            15 => {}
            30 => self.error_epc = v,
            31 => {
                self.error_epc = (self.error_epc & 0xFFFF_FFFF) | (v << 32);
            }
            _ => self.write_32(reg, v as u32),
        }
    }
}

impl Default for Cop0 {
    fn default() -> Self {
        Self::new()
    }
}

/// True iff `compare` is visited when stepping from `old` by `delta` increments of +1 (mod 2³²).
fn compare_matched_in_u32_interval(old: u32, delta: u64, cmp: u32) -> bool {
    if delta == 0 {
        return false;
    }
    if delta >= (1u64 << 32) {
        return true;
    }
    let k = min_steps_forward_to_value(old, cmp);
    k <= delta
}

/// Smallest `k >= 1` such that `old.wrapping_add(k) == cmp`, as a `u64` distance.
fn min_steps_forward_to_value(old: u32, cmp: u32) -> u64 {
    if old == cmp {
        1u64 << 32
    } else {
        u64::from(cmp.wrapping_sub(old))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compare_interval_no_wrap() {
        assert!(!compare_matched_in_u32_interval(5, 3, 4));
        assert!(compare_matched_in_u32_interval(5, 3, 6));
        assert!(compare_matched_in_u32_interval(5, 3, 7));
        assert!(compare_matched_in_u32_interval(5, 3, 8));
        assert!(!compare_matched_in_u32_interval(5, 3, 9));
    }

    #[test]
    fn compare_interval_wrap() {
        assert!(compare_matched_in_u32_interval(0xFFFF_FFFE, 5, 0));
        assert!(compare_matched_in_u32_interval(0xFFFF_FFFE, 5, 1));
    }

    #[test]
    fn compare_full_cycle_distance() {
        assert!(!compare_matched_in_u32_interval(5, 10, 5));
        assert!(compare_matched_in_u32_interval(5, 1u64 << 32, 5));
    }

    #[test]
    fn dmtc0_dmfc0_epc_round_trip() {
        let mut c = Cop0::new();
        c.write_xpr64(14, 0xFFFF_FFFF_8000_4000);
        assert_eq!(c.read_xpr64(14), 0xFFFF_FFFF_8000_4000);
    }
}
