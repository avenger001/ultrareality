//! VR4300 pipeline scoreboard for register hazard detection.
//!
//! The VR4300 is a 5-stage pipeline (IF-RD-ALU-MEM-WB) with interlocks for:
//! - **Load-use hazard**: 1 cycle stall when an instruction uses a load result
//! - **FP latencies**: Variable latency FP operations block dependent instructions
//! - **MDU interlock**: Handled separately via `mdu_issue_remain`
//!
//! This scoreboard tracks when each GPR and FPR will have its result available,
//! allowing the CPU to compute stall cycles for data hazards.

/// Number of GPR registers.
const GPR_COUNT: usize = 32;
/// Number of FPR registers.
const FPR_COUNT: usize = 32;

/// VR4300 FP operation latencies (cycles until result is available).
pub mod fp_latency {
    /// ADD.S, SUB.S, MUL.S
    pub const ADD_MUL_S: u8 = 3;
    /// ADD.D, SUB.D, MUL.D
    pub const ADD_MUL_D: u8 = 4;
    /// DIV.S
    pub const DIV_S: u8 = 23;
    /// DIV.D
    pub const DIV_D: u8 = 36;
    /// SQRT.S
    pub const SQRT_S: u8 = 28;
    /// SQRT.D
    pub const SQRT_D: u8 = 58;
    /// CVT.*.*, ABS, NEG, MOV
    pub const CVT: u8 = 2;
    /// Compare operations (C.cond.fmt)
    pub const CMP: u8 = 1;
}

/// Load-use stall (1 cycle for integer loads).
pub const LOAD_USE_STALL: u8 = 1;

/// Pipeline scoreboard tracking register writeback availability.
#[derive(Clone, Debug)]
pub struct Scoreboard {
    /// Cycle count when each GPR value will be available (0 = already available).
    /// Value is cycles remaining until writeback completes.
    gpr_ready: [u8; GPR_COUNT],
    /// Cycle count when each FPR value will be available.
    fpr_ready: [u8; FPR_COUNT],
    /// FCSR condition code (CC) availability for BC1F/BC1T.
    fcsr_cc_ready: u8,
}

impl Default for Scoreboard {
    fn default() -> Self {
        Self::new()
    }
}

impl Scoreboard {
    pub fn new() -> Self {
        Self {
            gpr_ready: [0; GPR_COUNT],
            fpr_ready: [0; FPR_COUNT],
            fcsr_cc_ready: 0,
        }
    }

    /// Reset all registers to immediately available.
    pub fn reset(&mut self) {
        self.gpr_ready.fill(0);
        self.fpr_ready.fill(0);
        self.fcsr_cc_ready = 0;
    }

    /// Advance the scoreboard by `cycles` (after instruction retirement).
    /// Decrements all pending counters, saturating at 0.
    pub fn advance(&mut self, cycles: u64) {
        let c = cycles.min(255) as u8;
        for r in &mut self.gpr_ready {
            *r = r.saturating_sub(c);
        }
        for r in &mut self.fpr_ready {
            *r = r.saturating_sub(c);
        }
        self.fcsr_cc_ready = self.fcsr_cc_ready.saturating_sub(c);
    }

    /// Mark a GPR as having a pending writeback in `latency` cycles.
    /// r0 is always 0, so we skip it.
    #[inline]
    pub fn set_gpr_latency(&mut self, reg: usize, latency: u8) {
        if reg != 0 && reg < GPR_COUNT {
            self.gpr_ready[reg] = latency;
        }
    }

    /// Mark an FPR as having a pending writeback in `latency` cycles.
    #[inline]
    pub fn set_fpr_latency(&mut self, reg: usize, latency: u8) {
        if reg < FPR_COUNT {
            self.fpr_ready[reg] = latency;
        }
    }

    /// Mark FCSR condition code as pending for `latency` cycles.
    #[inline]
    pub fn set_fcsr_cc_latency(&mut self, latency: u8) {
        self.fcsr_cc_ready = latency;
    }

    /// Get stall cycles needed before reading GPR `reg`.
    /// Returns 0 if the value is already available.
    #[inline]
    pub fn gpr_stall(&self, reg: usize) -> u64 {
        if reg == 0 || reg >= GPR_COUNT {
            0
        } else {
            u64::from(self.gpr_ready[reg])
        }
    }

    /// Get stall cycles needed before reading FPR `reg`.
    #[inline]
    pub fn fpr_stall(&self, reg: usize) -> u64 {
        if reg >= FPR_COUNT {
            0
        } else {
            u64::from(self.fpr_ready[reg])
        }
    }

    /// Get stall cycles needed before reading FCSR condition code.
    #[inline]
    pub fn fcsr_cc_stall(&self) -> u64 {
        u64::from(self.fcsr_cc_ready)
    }

    /// Compute total stall for reading two GPRs (common I-type and R-type).
    #[inline]
    pub fn gpr_stall_2(&self, rs: usize, rt: usize) -> u64 {
        self.gpr_stall(rs).max(self.gpr_stall(rt))
    }

    /// Compute total stall for reading three GPRs.
    #[inline]
    pub fn gpr_stall_3(&self, rs: usize, rt: usize, rd: usize) -> u64 {
        self.gpr_stall(rs)
            .max(self.gpr_stall(rt))
            .max(self.gpr_stall(rd))
    }

    /// Compute total stall for reading two FPRs.
    #[inline]
    pub fn fpr_stall_2(&self, fs: usize, ft: usize) -> u64 {
        self.fpr_stall(fs).max(self.fpr_stall(ft))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpr_latency_and_stall() {
        let mut sb = Scoreboard::new();
        sb.set_gpr_latency(5, 3);
        assert_eq!(sb.gpr_stall(5), 3);
        assert_eq!(sb.gpr_stall(6), 0);

        sb.advance(2);
        assert_eq!(sb.gpr_stall(5), 1);

        sb.advance(1);
        assert_eq!(sb.gpr_stall(5), 0);
    }

    #[test]
    fn r0_always_zero_stall() {
        let mut sb = Scoreboard::new();
        sb.set_gpr_latency(0, 10);
        assert_eq!(sb.gpr_stall(0), 0);
    }

    #[test]
    fn fpr_latency_and_stall() {
        let mut sb = Scoreboard::new();
        sb.set_fpr_latency(10, fp_latency::DIV_S);
        assert_eq!(sb.fpr_stall(10), 23);

        sb.advance(20);
        assert_eq!(sb.fpr_stall(10), 3);
    }

    #[test]
    fn fcsr_cc_stall() {
        let mut sb = Scoreboard::new();
        sb.set_fcsr_cc_latency(5);
        assert_eq!(sb.fcsr_cc_stall(), 5);

        sb.advance(5);
        assert_eq!(sb.fcsr_cc_stall(), 0);
    }
}
