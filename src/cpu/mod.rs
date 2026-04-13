pub mod cache;
pub mod cop0;
pub mod cop1;
pub mod scoreboard;
pub mod tlb;
mod r4300i;

pub use cache::{CacheOp, DCache, ICache};
pub use r4300i::{CpuHalt, R4300i, MATRIX_WATCH_ARMED};
pub use scoreboard::Scoreboard;
