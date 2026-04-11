//! Ultrareality: Nintendo 64 emulation core.
//!
//! The long-term goal is **cycle-accurate** behavior: the CPU, caches, system
//! bus, and Reality Co-Processor (RSP/RDP) advance in lockstep on a shared
//! master clock. The current code implements a **cycle-counted** R4300i
//! interpreter and memory system as the first building block; pipeline
//! interlocks, cache simulation, and RCP timing are explicit extension points.

pub mod bus;
pub mod cpu;
pub mod cycles;
pub mod machine;

pub use bus::{Bus, PhysicalMemory};
pub use cpu::R4300i;
pub use machine::Machine;
