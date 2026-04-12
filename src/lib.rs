//! Ultrareality: Nintendo 64 emulation core.
//!
//! The long-term goal is **cycle-accurate** behavior: the CPU, caches, system
//! bus, and Reality Co-Processor (RSP/RDP) advance in lockstep on a shared
//! master clock. The current code implements a **cycle-counted** R4300i
//! interpreter and memory system as the first building block; pipeline
//! interlocks, cache simulation, and RCP timing are explicit extension points.

pub mod ai;
pub mod boot;
pub mod bus;
pub mod cpu;
pub mod cycles;
pub mod machine;
pub mod mi;
pub mod pi;
pub mod pif;
pub mod rcp;
pub mod si;
pub mod vi;
pub mod video;

pub use ai::Ai;
pub use boot::{cart_boot_pc, ipl3_load_via_pi_dma, sign_extend_word32};
pub use bus::{Bus, PhysicalMemory, SystemBus, virt_to_phys, virt_to_phys_rdram};
pub use cpu::R4300i;
pub use machine::Machine;
pub use mi::{Mi, MI_INTR_AI, MI_INTR_DP, MI_INTR_PI, MI_INTR_SI, MI_INTR_SP};
pub use pi::Pi;
pub use pif::{Pif, PifRomLoadError, PIF_KSEG1_RESET_PC, PIF_ROM_LEN};
pub use rcp::{
    sp_dma_decode, sp_dma_end_addresses, DpcRegs, SpRegs, DPC_REG_CURRENT, DPC_REG_END,
    DPC_REG_START, DPC_REGS_BASE, SP_PC_REGS_BASE, SP_REG_RD_LEN, SP_REGS_BASE,
};
pub use si::Si;
pub use vi::{
    Vi, VI_NTSC_CYCLES_PER_FRAME, VI_REG_ORIGIN, VI_REG_WIDTH,
};
pub use video::{blit_rgba5551, pixel_rgba5551_to_argb};
