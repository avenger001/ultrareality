//! Ultrareality: Nintendo 64 emulation core.
//!
//! The long-term goal is **cycle-accurate** behavior: the CPU, caches, system
//! bus, and Reality Co-Processor (RSP/RDP) advance in lockstep on a shared
//! master clock. The current code implements a **cycle-counted** R4300i
//! interpreter and memory system as the first building block; pipeline
//! interlocks, cache simulation, and RCP timing are explicit extension points.
//! Shared NTSC **93.75 MHz**-scaled costs live in [`timing`].

pub mod ai;
pub mod boot;
pub mod bus;
pub mod cpu;
pub mod cycles;
pub mod machine;
pub mod mi;
pub mod pi;
pub mod pif;
pub mod present;
pub mod rcp;
pub mod rdp;
pub mod ri;
pub mod rsp;
pub mod si;
pub mod timing;
pub mod vi;
pub mod video;

pub use ai::{Ai, AI_REG_DRAM_ADDR, AI_REG_LEN, AI_REGS_BASE, AI_REGS_LEN};
pub use boot::{cart_boot_pc, ipl3_load_via_pi_dma, sign_extend_word32, ROM_OFF_BOOT_ADDRESS};
pub use bus::{Bus, PhysicalMemory, SystemBus, virt_to_phys, virt_to_phys_rdram};
pub use cpu::R4300i;
pub use cpu::cop0::{
    GENERAL_EXCEPTION_OFFSET, KSEG0_INTERRUPT_VECTOR_PC, KSEG1_BEV_INTERRUPT_VECTOR_PC,
    MIPS_OPCODE_BREAK,
};
pub use machine::Machine;
pub use mi::{
    Mi, MI_INTR_AI, MI_INTR_DP, MI_INTR_PI, MI_INTR_SI, MI_INTR_SP, MI_INTR_VI, MI_REG_INTR,
    MI_REG_INTR_MASK, MI_REG_MODE, MI_REG_VERSION, MI_REGS_BASE, MI_REGS_LEN, MI_VERSION_DEFAULT,
};
pub use pi::{
    Pi, CART_DOM1_ADDR2_BASE, CART_ROM_TEST_DWORD_OFF, PI_REG_BSD_DOM1_LAT, PI_REG_CART_ADDR,
    PI_REG_DRAM_ADDR, PI_REG_RD_LEN, PI_REG_WR_LEN, PI_REGS_BASE, PI_REGS_LEN,
};
pub use pif::{Pif, PifRomLoadError, PIF_KSEG1_RESET_PC, PIF_ROM_LEN};
pub use ri::{
    Ri, RI_MODE_DEFAULT, RI_REG_CONFIG, RI_REG_CURRENT_LOAD, RI_REG_LATENCY, RI_REG_MODE,
    RI_REG_REFRESH, RI_REG_SELECT, RI_REGS_BASE, RI_REGS_LEN,
};
pub use rcp::{
    sp_dma_decode, sp_dma_end_addresses, DpcEndKick, DpcRegs, SpRegs, DPC_REG_CURRENT, DPC_REG_END,
    DPC_REG_START, DPC_REG_STATUS, DPC_REGS_BASE, DPC_REGS_LEN, SP_PC_REGS_BASE, SP_PC_REGS_LEN,
    SP_REG_DRAM_ADDR, SP_REG_MEM_ADDR, SP_REG_RD_LEN, SP_REG_SEMAPHORE, SP_REG_STATUS,
    SP_REG_WR_LEN, SP_REGS_BASE, SP_REGS_LEN,
};
pub use rdp::{Rdp, RDRAM_CYCLES_PER_BYTE, RDP_CYCLES_PER_CMD, TMEM_SIZE};
pub use si::{
    Si, SI_REG_DRAM_ADDR, SI_REG_PIF_ADDR_RD64B, SI_REG_PIF_ADDR_WR64B, SI_REG_STATUS, SI_REGS_BASE,
    SI_REGS_LEN,
};
pub use vi::{
    vi_reg_byte_off, Vi, VI_NTSC_CYCLES_PER_FRAME, VI_OFF_BURST, VI_OFF_CONTROL, VI_OFF_H_SYNC,
    VI_OFF_H_VIDEO, VI_OFF_LEAP, VI_OFF_ORIGIN, VI_OFF_V_BURST, VI_OFF_V_CURRENT, VI_OFF_V_INTR,
    VI_OFF_V_SYNC, VI_OFF_V_VIDEO, VI_OFF_WIDTH, VI_OFF_X_SCALE, VI_OFF_Y_SCALE, VI_REG_ORIGIN,
    VI_REG_WIDTH, VI_REGS_BASE, VI_REGS_LEN, VI_RDRAM_CYCLES_PER_BYTE,
};
pub use timing::{
    ai_pcm_buffer_cycles, pi_cart_dma_total_cycles, sp_rsp_dma_total_cycles, RCP_MASTER_HZ_NTSC,
};
pub use present::{
    graphics_phase_reached, run_wgpu_loop, GraphicsPhase, PresentError, WgpuPresenter,
};
pub use video::{blit_rgba5551, blit_rgba5551_to_rgba8, pixel_rgba5551_to_argb};
pub use rsp::{Rsp, RspState};
