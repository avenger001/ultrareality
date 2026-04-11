//! COP1 (FPU) — minimal state for commercial ROM bring-up.
//!
//! VR4300 exposes 32 × 64-bit FPRs and control registers (FCR0, FCSR/FCR31, …).

/// Typical read value for FCR0 (implementation / revision) on VR4300-class CPUs.
pub const FCR0_IMP_REV: u32 = 0x511;

#[derive(Clone, Debug)]
pub struct Cop1 {
    pub fpr: [u64; 32],
    /// FCSR (also FCR31): rounding mode, exception enables, compare flags (CC).
    pub fcsr: u32,
}

impl Cop1 {
    pub fn new() -> Self {
        Self {
            fpr: [0; 32],
            fcsr: 0,
        }
    }

    pub fn reset(&mut self) {
        self.fpr = [0; 32];
        self.fcsr = 0;
    }

    #[inline]
    pub fn read_fcr(&self, rd: usize) -> u32 {
        match rd {
            0 => FCR0_IMP_REV,
            31 => self.fcsr,
            _ => 0,
        }
    }

    #[inline]
    pub fn write_fcr(&mut self, rd: usize, v: u32) {
        if rd == 31 {
            // Only a subset is writable; good enough for bring-up.
            self.fcsr = v & 0x0183_FFFF;
        }
    }

    #[inline]
    pub fn fpr_u32(&self, i: usize) -> u32 {
        self.fpr[i] as u32
    }

    #[inline]
    pub fn set_fpr_u32(&mut self, i: usize, v: u32) {
        self.fpr[i] = u64::from(v);
    }

    #[inline]
    pub fn fpr_f32(&self, i: usize) -> f32 {
        f32::from_bits(self.fpr_u32(i))
    }

    #[inline]
    pub fn set_fpr_f32(&mut self, i: usize, v: f32) {
        self.set_fpr_u32(i, v.to_bits());
    }

    #[inline]
    pub fn fpr_f64(&self, i: usize) -> f64 {
        f64::from_bits(self.fpr[i])
    }

    #[inline]
    pub fn set_fpr_f64(&mut self, i: usize, v: f64) {
        self.fpr[i] = v.to_bits();
    }

    /// Condition code 0 (FCSR bit 23), used by `BC1F` / `BC1T`.
    #[inline]
    pub fn set_cc0(&mut self, v: bool) {
        if v {
            self.fcsr |= 1 << 23;
        } else {
            self.fcsr &= !(1 << 23);
        }
    }
}

/// `C.cond.fmt` — MIPS III `funct` 0x30–0x3F (VR4300 COP1). Ignores signaling-NaN policy.
#[inline]
pub fn cond_f32(a: f32, b: f32, funct: u32) -> bool {
    let u = a.is_nan() || b.is_nan();
    let eq = !u && a == b;
    let lt = !u && a < b;
    let le = !u && a <= b;
    match funct {
        0x30 => false, // F
        0x31 => u,    // UN
        0x32 => eq,   // EQ
        0x33 => u || eq, // UEQ
        0x34 => lt,   // OLT
        0x35 => u || lt, // ULT
        0x36 => le,   // OLE
        0x37 => u || le, // ULE
        0x38 => false, // SF
        0x39 => u || a > b, // NGLE
        0x3A => eq,   // SEQ
        0x3B => u || a == b, // NGL (unordered or equal)
        0x3C => lt,   // LT
        0x3D => u || a < b, // NGE
        0x3E => le,   // LE
        0x3F => u || a <= b, // NGT
        _ => false,
    }
}

/// Same as [`cond_f32`] for double precision.
#[inline]
pub fn cond_f64(a: f64, b: f64, funct: u32) -> bool {
    let u = a.is_nan() || b.is_nan();
    let eq = !u && a == b;
    let lt = !u && a < b;
    let le = !u && a <= b;
    match funct {
        0x30 => false,
        0x31 => u,
        0x32 => eq,
        0x33 => u || eq,
        0x34 => lt,
        0x35 => u || lt,
        0x36 => le,
        0x37 => u || le,
        0x38 => false,
        0x39 => u || a > b,
        0x3A => eq,
        0x3B => u || a == b,
        0x3C => lt,
        0x3D => u || a < b,
        0x3E => le,
        0x3F => u || a <= b,
        _ => false,
    }
}

/// FCSR rounding mode in bits `[1:0]` (0 = nearest/even, 1 = toward 0, 2 = toward +inf, 3 = toward -inf).
#[inline]
pub const fn fcsr_rm(fcsr: u32) -> u32 {
    fcsr & 3
}

#[inline]
fn clamp_f64_to_i32(r: f64) -> i32 {
    if !r.is_finite() {
        return 0;
    }
    if r >= f64::from(i32::MAX) {
        return i32::MAX;
    }
    if r <= f64::from(i32::MIN) {
        return i32::MIN;
    }
    r as i32
}

#[inline]
fn clamp_f64_to_i64(r: f64) -> i64 {
    if !r.is_finite() {
        return 0;
    }
    if r >= i64::MAX as f64 {
        return i64::MAX;
    }
    if r <= i64::MIN as f64 {
        return i64::MIN;
    }
    r as i64
}

/// Nearest integer, ties to even (MIPS RN, mode 0). For `|x|` ≥ 2^53 returns `x` (already integral).
fn f64_round_nearest_ties_even(x: f64) -> f64 {
    if !x.is_finite() {
        return x;
    }
    if x.abs() >= (1u64 << 53) as f64 {
        return x;
    }
    let i = x as i64;
    let fi = i as f64;
    let d = x - fi;
    if d > 0.5 {
        return (i + 1) as f64;
    }
    if d < -0.5 {
        return (i - 1) as f64;
    }
    if d == 0.5 {
        return if i.rem_euclid(2) == 0 {
            fi
        } else {
            (i + 1) as f64
        };
    }
    if d == -0.5 {
        return if i.rem_euclid(2) == 0 {
            fi
        } else {
            (i - 1) as f64
        };
    }
    fi
}

/// `f32` → signed 32-bit using FCSR RM (for `CVT.W.S`, `ROUND.W.S`, …).
#[inline]
pub fn f32_to_i32_rm(x: f32, rm: u32) -> i32 {
    f64_to_i32_rm(f64::from(x), rm)
}

/// `f64` → signed 32-bit using FCSR RM (`CVT.W.D`, …).
#[inline]
pub fn f64_to_i32_rm(x: f64, rm: u32) -> i32 {
    if !x.is_finite() {
        return 0;
    }
    let r = match rm & 3 {
        0 => f64_round_nearest_ties_even(x),
        1 => x.trunc(),
        2 => x.ceil(),
        3 => x.floor(),
        _ => x.trunc(),
    };
    clamp_f64_to_i32(r)
}

/// `f32` → signed 64-bit using FCSR RM (`CVT.L.S`, …).
#[inline]
pub fn f32_to_i64_rm(x: f32, rm: u32) -> i64 {
    f64_to_i64_rm(f64::from(x), rm)
}

/// `f64` → signed 64-bit using FCSR RM (`CVT.L.D`, …).
#[inline]
pub fn f64_to_i64_rm(x: f64, rm: u32) -> i64 {
    if !x.is_finite() {
        return 0;
    }
    let r = match rm & 3 {
        0 => f64_round_nearest_ties_even(x),
        1 => x.trunc(),
        2 => x.ceil(),
        3 => x.floor(),
        _ => x.trunc(),
    };
    clamp_f64_to_i64(r)
}

#[inline]
pub fn f32_to_i32_trunc(x: f32) -> i32 {
    f64_to_i32_trunc(f64::from(x))
}

#[inline]
pub fn f64_to_i32_trunc(x: f64) -> i32 {
    if !x.is_finite() {
        return 0;
    }
    clamp_f64_to_i32(x.trunc())
}

#[inline]
pub fn f32_to_i32_ceil(x: f32) -> i32 {
    f64_to_i32_ceil(f64::from(x))
}

#[inline]
pub fn f64_to_i32_ceil(x: f64) -> i32 {
    if !x.is_finite() {
        return 0;
    }
    clamp_f64_to_i32(x.ceil())
}

#[inline]
pub fn f32_to_i32_floor(x: f32) -> i32 {
    f64_to_i32_floor(f64::from(x))
}

#[inline]
pub fn f64_to_i32_floor(x: f64) -> i32 {
    if !x.is_finite() {
        return 0;
    }
    clamp_f64_to_i32(x.floor())
}

impl Default for Cop1 {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        cond_f32, cond_f64, f32_to_i32_rm, f64_to_i32_rm, f64_round_nearest_ties_even,
    };

    #[test]
    fn round_nearest_ties_even_half() {
        assert_eq!(f64_round_nearest_ties_even(2.5), 2.0);
        assert_eq!(f64_round_nearest_ties_even(3.5), 4.0);
        assert_eq!(f64_round_nearest_ties_even(-1.5), -2.0);
    }

    #[test]
    fn cvt_w_respects_rm() {
        assert_eq!(f32_to_i32_rm(1.2, 1), 1);
        assert_eq!(f32_to_i32_rm(-1.2, 1), -1);
        assert_eq!(f64_to_i32_rm(1.7, 3), 1);
        assert_eq!(f64_to_i32_rm(1.2, 2), 2);
    }

    #[test]
    fn cond_eq_and_olt_f32() {
        assert!(cond_f32(1.0, 1.0, 0x32));
        assert!(!cond_f32(1.0, 2.0, 0x32));
        assert!(cond_f32(1.0, 2.0, 0x34));
        assert!(!cond_f32(2.0, 1.0, 0x34));
    }

    #[test]
    fn cond_unordered_f64() {
        let nan = f64::NAN;
        assert!(cond_f64(1.0, nan, 0x31));
        assert!(!cond_f64(1.0, nan, 0x32));
    }
}
