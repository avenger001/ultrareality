//! Top-level machine: one [`crate::timing::RCP_MASTER_HZ_NTSC`] timeline for CPU retirements and RCP I/O.
//! Each step adds interpreter cycles plus VI/RDP deferred debt ([`crate::bus::SystemBus::drain_deferred_cycles`]),
//! then applies that `delta` to PI/SI/AI/SP/DPC in parallel and to the VI field counter — see [`crate::timing`] module docs.

use crate::boot::{ipl3_load_via_pi_dma, DEFAULT_GAME_SP};
use crate::bus::{Bus, SystemBus};
use crate::cpu::{CpuHalt, R4300i};
use crate::pif::{PifRomLoadError, PIF_KSEG1_RESET_PC};
use crate::timing::MasterCycles;

pub struct Machine {
    pub cpu: R4300i,
    pub bus: SystemBus,
    /// Monotonic RCP master-cycle sum ([`crate::timing::RCP_MASTER_HZ_NTSC`]).
    pub master_cycles: MasterCycles,
    /// Retired cycles from the previous `cpu.step` (incl. bus debt), fed into `Count` before the next instruction.
    last_retired_cycles: MasterCycles,
    /// Accumulated master cycles for CP0 Count (which advances at half the CPU pipeline rate).
    /// Count increments by 1 for every 2 master cycles.
    count_cycle_accum: u64,
}

impl Machine {
    pub fn new() -> Self {
        Self {
            cpu: R4300i::new(),
            bus: SystemBus::new(),
            master_cycles: 0,
            last_retired_cycles: 0,
            count_cycle_accum: 0,
        }
    }

    /// Replace cartridge ROM (e.g. after reading a `.z64` / `.n64` file).
    pub fn set_cartridge_rom(&mut self, rom: Vec<u8>) {
        self.bus.pi = crate::pi::Pi::with_rom(rom);
    }

    /// Load PIF boot ROM bytes (e.g. from a dump). Must be at least [`crate::pif::PIF_ROM_LEN`].
    pub fn set_pif_rom(&mut self, data: &[u8]) -> Result<(), PifRomLoadError> {
        self.bus.pif.replace_rom(data)
    }

    /// Cold reset with PC at the PIF ROM entry in kseg1 ([`PIF_KSEG1_RESET_PC`]).
    /// Does not preload the cart into RDRAM; real firmware is expected to drive PI DMA and transfer control.
    pub fn bootstrap_from_pif_reset(&mut self) {
        self.cpu.reset(PIF_KSEG1_RESET_PC);
    }

    /// Reset CPU and load the IPL3 ROM region into RDRAM via **PI DMA** (see [`crate::boot::ipl3_load_via_pi_dma`]).
    /// Sets GPR/COP0 state to match what IPL3 would leave (HLE boot), per other emulator conventions.
    pub fn bootstrap_cart_from_rom(&mut self) {
        let Some(pc) =
            ipl3_load_via_pi_dma(&mut self.bus.pi, &mut self.bus.rdram, &mut self.bus.mi) else {
            return;
        };
        // Verify boot DMA worked by checking first word at entry point
        let entry_phys = crate::bus::virt_to_phys(pc).unwrap_or(0x400);
        let first_word = self.bus.rdram.read_u32(entry_phys).unwrap_or(0);
        eprintln!("[BOOT] entry_pc=0x{:08X} entry_phys=0x{:06X} first_word=0x{:08X}",
            pc as u32, entry_phys, first_word);
        self.cpu.reset(pc);
        self.cpu.regs[29] = DEFAULT_GAME_SP; // $sp

        // --- HLE boot state: match what IPL3 leaves for the game ---

        // PI: real IPL3 acknowledges the boot DMA interrupt before handing off.
        // Clear PI status interrupt flag and MI_INTR_PI so games don't see stale state.
        self.bus.pi.status = 0;
        self.bus.mi.clear(crate::mi::MI_INTR_PI);

        // COP0: after IPL3 finishes, ERL and BEV are cleared; CU0/CU1 enabled, FR=1 (64-bit FP).
        // CU1 (bit 29) | CU0 (bit 28) | FR (bit 26) = 0x3400_0000
        // BEV (bit 22) must be cleared so exceptions go to RDRAM handlers, not PIF ROM.
        // IE (bit 0) stays off — the game's osInitialize enables it when ready.
        self.cpu.cop0.status = 0x3400_0000;

        // CIC seed: games (via libultra __osPiInit) read a seed from the CIC and use it for checksums.
        // SM64 uses CIC-6102: seed = 0x3F. IPL3 puts this in s3 and also writes it to PIF RAM byte 0x24.
        // For CIC-6102: s3 = seed word, s4 = osVersion from ROM header, s6 = IPL3 first word.
        let cic_seed = self.detect_cic_seed();
        self.cpu.regs[19] = cic_seed as u64;         // $s3 = CIC seed
        self.cpu.regs[20] = 0x0000_0001;              // $s4 = TV system (1 = NTSC)
        self.cpu.regs[22] = if self.bus.pi.rom.len() >= 0x44 { // $s6 = first word of IPL3
            crate::boot::rom_u32_be(&self.bus.pi.rom, 0x40).unwrap_or(0) as u64
        } else { 0 };
        self.cpu.regs[23] = pc;                        // $s7 = entry PC

        // Memory size in $s4 area: osInitialize reads osMemSize from a known location.
        // Write RDRAM size (4 MiB = 0x0040_0000) to the conventional address 0x80000318.
        let rdram_size = self.bus.rdram.data.len() as u32;
        self.bus.rdram.write_u32(0x318, rdram_size);

        // osTvType at 0x80000300: 0=PAL, 1=NTSC, 2=MPAL. SM64 US is NTSC.
        // Detect from ROM header byte 0x3E (country code): 'E'=USA/NTSC, 'J'=Japan/NTSC, 'P'=PAL
        let tv_type: u32 = if self.bus.pi.rom.len() > 0x3E {
            match self.bus.pi.rom[0x3E] {
                b'E' | b'J' | b'U' => 1, // NTSC
                b'P' | b'D' | b'F' | b'S' | b'I' => 0, // PAL
                _ => 1, // default NTSC
            }
        } else {
            1 // default NTSC
        };
        self.bus.rdram.write_u32(0x300, tv_type);

        // osRomBase at 0x80000304: cartridge ROM base address in PI space
        self.bus.rdram.write_u32(0x304, crate::pi::CART_DOM1_ADDR2_BASE);

        // 0x80000308: This address is read during boot and added to base addresses.
        // Must be 0 for cold boot; value 1 would cause misaligned address computations.
        self.bus.rdram.write_u32(0x308, 0);

        // osResetType at 0x8000030C: 0 = cold boot, 1 = NMI reset
        self.bus.rdram.write_u32(0x30C, 0);

        // osCicId at 0x80000310: CIC chip ID (varies by game, used for anti-piracy)
        // CIC-6102 = 2
        self.bus.rdram.write_u32(0x310, 2);

        // osVersion at 0x80000314: libultra version
        // This should match what IPL3 would leave; typically 0x0000_0001 or as set by ROM
        self.bus.rdram.write_u32(0x314, 0x0000_0001);

        // PIF: write CIC seed byte into PIF RAM at offset 0x24 (byte 36), where osInitialize reads it.
        // CIC-6102 seed value = 0x3F → written as {0x00, 0x06, seed, seed}.
        self.bus.pif.ram[0x24] = 0x00;
        self.bus.pif.ram[0x25] = 0x06;
        self.bus.pif.ram[0x26] = cic_seed as u8;
        self.bus.pif.ram[0x27] = cic_seed as u8;

        // PIF: write IPL3 completion status at offset 0x3C (last word of PIF RAM).
        // osInitialize reads 0x1FC007FC to verify IPL3 ran successfully.
        // Byte 0x3F (offset 63) should be 0x08 to indicate successful completion.
        self.bus.pif.ram[0x3C] = 0x00;
        self.bus.pif.ram[0x3D] = 0x00;
        self.bus.pif.ram[0x3E] = 0x00;
        self.bus.pif.ram[0x3F] = 0x08;
    }

    /// Detect CIC chip type from ROM header and return the IPL seed.
    fn detect_cic_seed(&self) -> u32 {
        // CIC detection heuristic: hash the IPL3 region (ROM[0x40..0x1000]) like other emulators.
        // For simplicity, check the first word of IPL3 and a few known CRC patterns.
        if self.bus.pi.rom.len() < 0x1000 {
            return 0x3F; // default CIC-6102
        }
        // Compute a simple checksum of ROM[0x40..0x1000]
        let mut sum: u32 = 0;
        for chunk in self.bus.pi.rom[0x40..0x1000].chunks(4) {
            if chunk.len() == 4 {
                let w = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                sum = sum.wrapping_add(w);
            }
        }
        match sum {
            0x587E_40FF => 0x3F, // CIC-6101 (Star Fox 64)
            0x6170_A4A1 => 0x3F, // CIC-6102 (SM64, most games)
            0x009E_9EA3 => 0x78, // CIC-6103 (Banjo-Kazooie)
            0x90BB_6CB5 => 0x91, // CIC-6105 (Zelda OoT)
            0x0B05_0EE0 => 0x85, // CIC-6106 (F-Zero X)
            _ => 0x3F,           // fallback CIC-6102
        }
    }

    /// RCP cycles accounted for the last finished [`step`](Self::step) or [`run`](Self::run) iteration
    /// (CPU retirements + [`SystemBus::drain_deferred_cycles`] debt). Zero before the first step.
    #[inline]
    pub fn last_step_rcp_cycles(&self) -> MasterCycles {
        self.last_retired_cycles
    }

    pub fn step(&mut self) -> Result<(), CpuHalt> {
        // CP0 Count increments at half the CPU pipeline clock (93.75 MHz / 2 = 46.875 MHz).
        // Accumulate master cycles and advance Count by the integer number of half-cycles.
        self.count_cycle_accum += self.last_retired_cycles;
        let count_delta = self.count_cycle_accum / 2;
        self.count_cycle_accum %= 2;
        self.cpu.cop0.advance_count_wrapped(count_delta);
        self.last_retired_cycles = 0;

        let irq = self.bus.mi.cpu_irq_pending();
        let prev_int = crate::cpu::cop0::INT_TAKEN_COUNT
            .load(std::sync::atomic::Ordering::Relaxed);
        let mi_intr_before = self.bus.mi.intr;
        let pc_before = self.cpu.pc as u32;
        let c = self.cpu.step(&mut self.bus, irq)?;
        let now_int = crate::cpu::cop0::INT_TAKEN_COUNT
            .load(std::sync::atomic::Ordering::Relaxed);
        if now_int != prev_int {
            // An interrupt was taken in this step. Record which MI bits were set.
            for b in 0..6u8 {
                if (mi_intr_before >> b) & 1 != 0 {
                    crate::cpu::cop0::INT_TAKEN_MI_HISTOGRAM[b as usize]
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            }
        }
        // Detect MI_INTR ack (VI bit cleared) and record PC of writer.
        if (mi_intr_before & 0x08) != 0 && (self.bus.mi.intr & 0x08) == 0 {
            MI_VI_ACK_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let n = MI_VI_ACK_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n < 10 {
                eprintln!("[MI_VI_ACK #{}] PC=0x{:08X}", n, pc_before);
            }
        }
        let def = self.bus.drain_deferred_cycles();
        let delta = c.saturating_add(def);
        self.bus.rcp_advance_dma_in_flight(delta);
        self.master_cycles = self.master_cycles.wrapping_add(delta);
        self.last_retired_cycles = delta;
        self.bus.advance_vi_frame_timing(delta);
        Ok(())
    }

    /// Run up to `max_steps` CPU steps (each may include a branch delay slot).
    pub fn run(&mut self, max_steps: u64) -> Result<(), CpuHalt> {
        for _ in 0..max_steps {
            // CP0 Count increments at half the CPU pipeline clock.
            self.count_cycle_accum += self.last_retired_cycles;
            let count_delta = self.count_cycle_accum / 2;
            self.count_cycle_accum %= 2;
            self.cpu.cop0.advance_count_wrapped(count_delta);
            self.last_retired_cycles = 0;

            let irq = self.bus.mi.cpu_irq_pending();
            let c = self.cpu.step(&mut self.bus, irq)?;
            let def = self.bus.drain_deferred_cycles();
            let delta = c.saturating_add(def);
            self.bus.rcp_advance_dma_in_flight(delta);
            self.master_cycles = self.master_cycles.wrapping_add(delta);
            self.last_retired_cycles = delta;
            self.bus.advance_vi_frame_timing(delta);
        }
        Ok(())
    }
}

impl Default for Machine {
    fn default() -> Self {
        Self::new()
    }
}

pub static MI_VI_ACK_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static MI_VI_ACK_LOG: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bus::Bus;
    use crate::cpu::cop0::{
        CAUSE_BD, CAUSE_EXCCODE_MASK, CAUSE_EXCCODE_SHIFT, EXCCODE_INT, GENERAL_EXCEPTION_OFFSET,
        KSEG0_INTERRUPT_VECTOR_PC, MIPS_OPCODE_BREAK, STATUS_EXL, STATUS_IE, STATUS_IM2,
    };
    use crate::ai::{AI_REG_LEN, AI_REGS_BASE};
    use crate::timing::ai_pcm_buffer_cycles;
    use crate::si::SI_DMA_CYCLES;

    /// Test RDRAM destinations for PI/SI DMA integration tests (physical).
    const RDRAM_TEST_PI_DST: u32 = 0x100;
    const RDRAM_TEST_SI_DST: u32 = 0x200;
    use crate::mi::{
        MI_INTR_AI, MI_INTR_DP, MI_INTR_PI, MI_INTR_SI, MI_INTR_SP, MI_INTR_VI,
    };
    use crate::rcp::{DPC_REG_END, DPC_REGS_BASE, SP_REG_STATUS, SP_REGS_BASE};

    #[test]
    fn last_step_rcp_cycles_matches_master_delta() {
        let mut m = Machine::new();
        assert_eq!(m.last_step_rcp_cycles(), 0);
        m.bus
            .rdram
            .write_u32(0, MIPS_OPCODE_BREAK);
        m.cpu.reset(0x8000_0000);
        m.step().unwrap();
        let last = m.last_step_rcp_cycles();
        assert!(last > 0);
        assert_eq!(m.master_cycles, last);
    }

    #[test]
    fn mi_interrupt_delivers_to_handler_vector() {
        let mut m = Machine::new();
        // Placeholder handler at KSEG0 general vector (physical GENERAL_EXCEPTION_OFFSET): `break`.
        m.bus
            .rdram
            .write_u32(GENERAL_EXCEPTION_OFFSET, MIPS_OPCODE_BREAK);

        m.cpu.reset(0x8000_2000);
        // Fresh `reset` COP0 still has ERL/BEV bits that block external interrupts.
        m.cpu.cop0.status = STATUS_IE | STATUS_IM2;
        m.cpu.cop0.cause |= CAUSE_BD;

        m.bus.mi.mask = 0xFF;
        m.bus.mi.raise(MI_INTR_VI);

        m.step().unwrap();

        assert_eq!(m.cpu.pc, KSEG0_INTERRUPT_VECTOR_PC);
        assert_eq!(m.cpu.cop0.epc, 0x8000_2000);
        assert!((m.cpu.cop0.status & STATUS_EXL) != 0);
        assert!((m.cpu.cop0.cause & CAUSE_BD) == 0);
    }

    #[test]
    fn bootstrap_from_pif_reset_sets_entry_pc() {
        use crate::pif::PIF_ROM_LEN;

        let mut m = Machine::new();
        m.set_pif_rom(&vec![0x5Au8; PIF_ROM_LEN]).unwrap();
        m.bootstrap_from_pif_reset();
        assert_eq!(m.cpu.pc, PIF_KSEG1_RESET_PC);
    }

    #[test]
    fn pi_dma_sets_mi_and_interrupt_delivers() {
        use crate::pi::{
            CART_DOM1_ADDR2_BASE, CART_ROM_TEST_DWORD_OFF, PI_REG_CART_ADDR, PI_REG_DRAM_ADDR,
            PI_REG_WR_LEN, PI_REGS_BASE,
        };

        let mut m = Machine::new();
        let mut rom = vec![0u8; 0x200];
        rom[CART_ROM_TEST_DWORD_OFF..CART_ROM_TEST_DWORD_OFF + 4]
            .copy_from_slice(&0x1122_3344u32.to_be_bytes());
        m.set_cartridge_rom(rom);

        m.bus
            .rdram
            .write_u32(GENERAL_EXCEPTION_OFFSET, MIPS_OPCODE_BREAK);
        m.cpu.reset(0x8000_4000);
        m.cpu.cop0.status = STATUS_IE | STATUS_IM2;
        m.bus.mi.mask = MI_INTR_PI;
        m.bus.mi.intr = 0;

        m.bus.write_u32(PI_REGS_BASE + PI_REG_DRAM_ADDR, RDRAM_TEST_PI_DST);
        m.bus.write_u32(
            PI_REGS_BASE + PI_REG_CART_ADDR,
            CART_DOM1_ADDR2_BASE + CART_ROM_TEST_DWORD_OFF as u32,
        );
        // PI_WR_LEN triggers Cart→RDRAM DMA (named from PI perspective: PI writes to RDRAM)
        m.bus.write_u32(PI_REGS_BASE + PI_REG_WR_LEN, 3);

        assert_eq!(m.bus.mi.intr & MI_INTR_PI, 0);
        let pi_cost = m.bus.pi.cart_dma_cost_cycles(4);
        m.bus.rcp_advance_dma_in_flight(pi_cost);
        assert_ne!(m.bus.mi.intr & MI_INTR_PI, 0);
        assert_eq!(
            m.bus.rdram.read_u32(RDRAM_TEST_PI_DST).unwrap(),
            0x1122_3344
        );

        m.step().unwrap();
        assert_eq!(m.cpu.pc, KSEG0_INTERRUPT_VECTOR_PC);
        assert_eq!(m.cpu.cop0.epc, 0x8000_4000);
        let exc = (m.cpu.cop0.cause >> CAUSE_EXCCODE_SHIFT) & CAUSE_EXCCODE_MASK;
        assert_eq!(exc, EXCCODE_INT);
    }

    #[test]
    fn si_dma_sets_mi_and_interrupt_delivers() {
        use crate::pif::{PIF_RAM_START, PIF_ROM_LEN};
        use crate::si::{SI_REG_DRAM_ADDR, SI_REG_PIF_ADDR_RD64B, SI_REGS_BASE};

        let mut m = Machine::new();
        m.set_pif_rom(&vec![0x3Cu8; PIF_ROM_LEN]).unwrap();
        m.bus.pif.ram[0..4].copy_from_slice(&0x99AA_BBCCu32.to_be_bytes());

        m.bus
            .rdram
            .write_u32(GENERAL_EXCEPTION_OFFSET, MIPS_OPCODE_BREAK);
        m.cpu.reset(0x8000_5000);
        m.cpu.cop0.status = STATUS_IE | STATUS_IM2;
        m.bus.mi.mask = MI_INTR_SI;
        m.bus.mi.intr = 0;

        m.bus.write_u32(SI_REGS_BASE + SI_REG_DRAM_ADDR, RDRAM_TEST_SI_DST);
        m.bus.write_u32(SI_REGS_BASE + SI_REG_PIF_ADDR_RD64B, PIF_RAM_START);

        assert_eq!(m.bus.mi.intr & MI_INTR_SI, 0);
        m.bus.rcp_advance_dma_in_flight(SI_DMA_CYCLES);
        assert_ne!(m.bus.mi.intr & MI_INTR_SI, 0);
        assert_eq!(
            m.bus.rdram.read_u32(RDRAM_TEST_SI_DST).unwrap(),
            0x99AA_BBCC
        );

        m.step().unwrap();
        assert_eq!(m.cpu.pc, KSEG0_INTERRUPT_VECTOR_PC);
        assert_eq!(m.cpu.cop0.epc, 0x8000_5000);
        let exc = (m.cpu.cop0.cause >> CAUSE_EXCCODE_SHIFT) & CAUSE_EXCCODE_MASK;
        assert_eq!(exc, EXCCODE_INT);
    }

    #[test]
    fn ai_len_write_sets_mi_and_interrupt_delivers() {
        let mut m = Machine::new();
        m.bus
            .rdram
            .write_u32(GENERAL_EXCEPTION_OFFSET, MIPS_OPCODE_BREAK);
        m.cpu.reset(0x8000_6000);
        m.cpu.cop0.status = STATUS_IE | STATUS_IM2;
        m.bus.mi.mask = MI_INTR_AI;
        m.bus.mi.intr = 0;

        m.bus.write_u32(AI_REGS_BASE + AI_REG_LEN, 0x400);

        assert_eq!(m.bus.mi.intr & MI_INTR_AI, 0);
        m.bus
            .rcp_advance_dma_in_flight(ai_pcm_buffer_cycles(0x400));
        assert_ne!(m.bus.mi.intr & MI_INTR_AI, 0);
        m.step().unwrap();
        assert_eq!(m.cpu.pc, KSEG0_INTERRUPT_VECTOR_PC);
        assert_eq!(m.cpu.cop0.epc, 0x8000_6000);
        let exc = (m.cpu.cop0.cause >> CAUSE_EXCCODE_SHIFT) & CAUSE_EXCCODE_MASK;
        assert_eq!(exc, EXCCODE_INT);
    }

    #[test]
    fn sp_set_intr_via_status_delivers_interrupt() {
        // SP DMA completion does not raise MI_INTR_SP on real hardware. The
        // SP interrupt path is exercised via a SET_INTR write to SP_STATUS
        // (bit 4) instead, and that should dispatch to the CPU exception
        // vector when STATUS_IM2 + IE are set.
        let mut m = Machine::new();
        m.bus
            .rdram
            .write_u32(GENERAL_EXCEPTION_OFFSET, MIPS_OPCODE_BREAK);
        m.cpu.reset(0x8000_7000);
        m.cpu.cop0.status = STATUS_IE | STATUS_IM2;
        m.bus.mi.mask = MI_INTR_SP;
        m.bus.mi.intr = 0;

        // SP_STATUS write: bit 4 = set MI SP intr.
        m.bus.write_u32(SP_REGS_BASE + SP_REG_STATUS, 1 << 4);

        assert_ne!(m.bus.mi.intr & MI_INTR_SP, 0);
        m.step().unwrap();
        assert_eq!(m.cpu.pc, KSEG0_INTERRUPT_VECTOR_PC);
        assert_eq!(m.cpu.cop0.epc, 0x8000_7000);
        let exc = (m.cpu.cop0.cause >> CAUSE_EXCCODE_SHIFT) & CAUSE_EXCCODE_MASK;
        assert_eq!(exc, EXCCODE_INT);
    }

    #[test]
    fn dpc_end_sets_mi_and_interrupt_delivers() {
        let mut m = Machine::new();
        m.bus
            .rdram
            .write_u32(GENERAL_EXCEPTION_OFFSET, MIPS_OPCODE_BREAK);
        m.cpu.reset(0x8000_8000);
        m.cpu.cop0.status = STATUS_IE | STATUS_IM2;
        m.bus.mi.mask = MI_INTR_DP;
        m.bus.mi.intr = 0;

        // Real hardware only raises MI_INTR_DP when the command stream
        // executes OP_SYNC_FULL. Place a SyncFull opcode at the start of the
        // display list so the deferred RDP processor sees it and raises.
        m.bus.rdram.write_u32(0, 0xE900_0000);
        m.bus.rdram.write_u32(4, 0x0000_0000);

        m.bus.write_u32(DPC_REGS_BASE + DPC_REG_END, 0x0000_0008);

        assert_eq!(m.bus.mi.intr & MI_INTR_DP, 0);
        let est = crate::rdp::Rdp::estimate_display_list_cycles(0, 0x0000_0008);
        m.bus.rcp_advance_dma_in_flight(est);
        assert_ne!(m.bus.mi.intr & MI_INTR_DP, 0);
        m.step().unwrap();
        assert_eq!(m.cpu.pc, KSEG0_INTERRUPT_VECTOR_PC);
        assert_eq!(m.cpu.cop0.epc, 0x8000_8000);
        let exc = (m.cpu.cop0.cause >> CAUSE_EXCCODE_SHIFT) & CAUSE_EXCCODE_MASK;
        assert_eq!(exc, EXCCODE_INT);
    }
}
