//! RSP Vector Unit (COP2): register file, accumulator, and all vector operations.
//!
//! The RSP VU has 32 128-bit vector registers (each holding 8 x i16 elements),
//! a 48-bit accumulator per element (split into hi/md/lo), comparison flags (VCC/VCO/VCE),
//! and reciprocal/rsq latches.
//!
//! Reference: [n64brew RSP](https://n64brew.dev/wiki/Reality_Signal_Processor/CPU_Core)

// ---------------------------------------------------------------------------
// Vector register state
// ---------------------------------------------------------------------------

/// 32 vector registers, each 128 bits = 8 x i16 elements.
/// Element 0 is the most-significant (big-endian order matching hardware byte layout).
pub type Vreg = [u16; 8];

/// RSP Vector Unit state.
#[derive(Clone, Debug)]
pub struct VectorUnit {
    /// 32 vector registers.
    pub vr: [Vreg; 32],
    /// 48-bit accumulator high (bits 47–32 per element).
    pub acc_hi: [u16; 8],
    /// 48-bit accumulator middle (bits 31–16 per element).
    pub acc_md: [u16; 8],
    /// 48-bit accumulator low (bits 15–0 per element).
    pub acc_lo: [u16; 8],
    /// VCC — vector compare code (16 bits: high 8 = clip, low 8 = compare).
    pub vcc: u16,
    /// VCO — vector carry out (16 bits: high 8 = carry, low 8 = not-equal).
    pub vco: u16,
    /// VCE — vector compare extension (8 bits).
    pub vce: u8,
    /// Reciprocal input latch (for VRCP/VRSQ sequencing).
    pub div_in: i32,
    /// Reciprocal output latch.
    pub div_out: i32,
    /// Whether div_in has been loaded (for high/low split ops).
    pub div_in_loaded: bool,
}

impl Default for VectorUnit {
    fn default() -> Self {
        Self::new()
    }
}

impl VectorUnit {
    pub fn new() -> Self {
        Self {
            vr: [[0u16; 8]; 32],
            acc_hi: [0; 8],
            acc_md: [0; 8],
            acc_lo: [0; 8],
            vcc: 0,
            vco: 0,
            vce: 0,
            div_in: 0,
            div_out: 0,
            div_in_loaded: false,
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

// ---------------------------------------------------------------------------
// Element broadcast decode
// ---------------------------------------------------------------------------

/// Decode the `e` field (bits 24:21 of a COP2 instruction) into an element
/// index mapping. Returns an array where `result[i]` is the element index of
/// vt to use for lane `i`.
///
/// - e=0–1: no broadcast (identity)
/// - e=2–3: pairs (0,0,2,2,4,4,6,6) or (1,1,3,3,5,5,7,7)
/// - e=4–7: quads
/// - e=8–15: scalar broadcast
#[inline]
pub fn element_map(e: u32) -> [usize; 8] {
    if e < 2 {
        [0, 1, 2, 3, 4, 5, 6, 7]
    } else if e < 4 {
        // Pairs: e & 1 selects which pair half
        let base = (e & 1) as usize;
        [
            base,
            base,
            base + 2,
            base + 2,
            base + 4,
            base + 4,
            base + 6,
            base + 6,
        ]
    } else if e < 8 {
        // Quads
        let base = (e & 3) as usize;
        [base, base, base, base, base + 4, base + 4, base + 4, base + 4]
    } else {
        // Scalar broadcast: e & 7 selects the element
        let s = (e & 7) as usize;
        [s, s, s, s, s, s, s, s]
    }
}

/// Get the broadcast elements from vt given element field `e`.
#[inline]
pub fn broadcast(vt: &Vreg, e: u32) -> Vreg {
    let map = element_map(e);
    [
        vt[map[0]], vt[map[1]], vt[map[2]], vt[map[3]],
        vt[map[4]], vt[map[5]], vt[map[6]], vt[map[7]],
    ]
}

// ---------------------------------------------------------------------------
// Clamp helpers
// ---------------------------------------------------------------------------

/// Signed clamp to i16 range (for accumulator read-back).
#[inline]
fn clamp_signed(val: i32) -> u16 {
    val.clamp(-32768, 32767) as i16 as u16
}

/// Unsigned clamp to u16 range.
#[inline]
fn clamp_unsigned(val: i32) -> u16 {
    val.clamp(0, 65535) as u16
}

// ---------------------------------------------------------------------------
// RCP/RSQ lookup tables (matching hardware ROM)
// ---------------------------------------------------------------------------

use std::sync::LazyLock;

/// Reciprocal lookup table (512 entries).
static RCP_TABLE: LazyLock<[u16; 512]> = LazyLock::new(|| {
    let mut t = [0u16; 512];
    for i in 0..512 {
        let x = (0x200u64 + i as u64) as u64;
        let r = ((1u64 << 34) + x - 1) / x;
        t[i] = (r & 0xFFFF) as u16;
    }
    t
});

/// Reciprocal square root lookup table (512 entries).
static RSQ_TABLE: LazyLock<[u16; 512]> = LazyLock::new(|| {
    let mut t = [0u16; 512];
    for i in 0..512usize {
        let x = 0x200u64 + i as u64;
        let shifted = x << 17;
        let s = (shifted as f64).sqrt() as u64;
        let r = if s == 0 { 0xFFFF } else { ((1u64 << 31) + s - 1) / s };
        t[i] = (r & 0xFFFF) as u16;
    }
    t
});

/// Compute reciprocal using the hardware lookup table.
pub fn rcp_lookup(input: i32) -> i32 {
    if input == 0 {
        return 0x7FFF_FFFFu32 as i32;
    }
    let negative = input < 0;
    let abs_input = if negative { -(input as i64) } else { input as i64 } as u32;

    // Find the leading one position
    let shift = abs_input.leading_zeros();
    // Normalize: shift so bit 31 is set, then take bits 30:22 as index
    let normalized = abs_input << shift;
    let index = ((normalized >> 22) & 0x1FF) as usize;

    let rom_val = RCP_TABLE[index] as u32;
    // Result: (0x10000 | rom_val) << 14, then shift back
    let result = ((0x10000u32 | rom_val) << 14) >> (31 - (31 - shift));

    if negative {
        -(result as i32)
    } else {
        result as i32
    }
}

/// Compute reciprocal square root using the hardware lookup table.
pub fn rsq_lookup(input: i32) -> i32 {
    if input == 0 {
        return 0x7FFF_FFFFu32 as i32;
    }
    let negative = input < 0;
    let abs_input = if negative { -(input as i64) } else { input as i64 } as u32;

    let shift = abs_input.leading_zeros();
    let normalized = abs_input << shift;
    // For RSQ, the index also depends on whether shift is odd
    let idx_base = ((normalized >> 22) & 0x1FF) as usize;
    let _odd = shift & 1;

    let rom_val = RSQ_TABLE[idx_base] as u32;
    let half_shift = (31 - shift) / 2;
    let result = ((0x10000u32 | rom_val) << 14) >> (14u32.wrapping_sub(half_shift));

    if negative {
        -(result as i32)
    } else {
        result as i32
    }
}

// ---------------------------------------------------------------------------
// Vector ALU operations
// ---------------------------------------------------------------------------

impl VectorUnit {
    // -----------------------------------------------------------------------
    // Multiply family
    // -----------------------------------------------------------------------

    /// VMULF: signed fraction multiply (round) — result = clamp_signed(acc >> 16)
    pub fn vmulf(&mut self, vd: usize, vs: usize, vt_broadcast: &Vreg) {
        for i in 0..8 {
            let s = self.vr[vs][i] as i16 as i32;
            let t = vt_broadcast[i] as i16 as i32;
            let prod = (s * t) as i64 * 2 + 0x8000;
            self.acc_hi[i] = (prod >> 32) as u16;
            self.acc_md[i] = (prod >> 16) as u16;
            self.acc_lo[i] = prod as u16;
            self.vr[vd][i] = clamp_signed((prod >> 16) as i32);
        }
    }

    /// VMULU: unsigned fraction multiply (round) — result = clamp_unsigned(acc >> 16)
    pub fn vmulu(&mut self, vd: usize, vs: usize, vt_broadcast: &Vreg) {
        for i in 0..8 {
            let s = self.vr[vs][i] as i16 as i32;
            let t = vt_broadcast[i] as i16 as i32;
            let prod = (s * t) as i64 * 2 + 0x8000;
            self.acc_hi[i] = (prod >> 32) as u16;
            self.acc_md[i] = (prod >> 16) as u16;
            self.acc_lo[i] = prod as u16;
            self.vr[vd][i] = clamp_unsigned((prod >> 16) as i32);
        }
    }

    /// VMUDL: unsigned low partial product
    pub fn vmudl(&mut self, vd: usize, vs: usize, vt_broadcast: &Vreg) {
        for i in 0..8 {
            let s = self.vr[vs][i] as u32;
            let t = vt_broadcast[i] as u32;
            let prod = (s * t) as u64;
            self.acc_hi[i] = 0;
            self.acc_md[i] = 0;
            self.acc_lo[i] = (prod >> 16) as u16;
            self.vr[vd][i] = self.acc_lo[i];
        }
    }

    /// VMUDM: signed*unsigned mid partial product
    pub fn vmudm(&mut self, vd: usize, vs: usize, vt_broadcast: &Vreg) {
        for i in 0..8 {
            let s = self.vr[vs][i] as i16 as i32;
            let t = vt_broadcast[i] as u32;
            let prod = (s as i64) * (t as i64);
            self.acc_hi[i] = (prod >> 32) as u16;
            self.acc_md[i] = (prod >> 16) as u16;
            self.acc_lo[i] = prod as u16;
            self.vr[vd][i] = self.acc_md[i];
        }
    }

    /// VMUDN: unsigned*signed mid partial product
    pub fn vmudn(&mut self, vd: usize, vs: usize, vt_broadcast: &Vreg) {
        for i in 0..8 {
            let s = self.vr[vs][i] as u32;
            let t = vt_broadcast[i] as i16 as i32;
            let prod = (s as i64) * (t as i64);
            self.acc_hi[i] = (prod >> 32) as u16;
            self.acc_md[i] = (prod >> 16) as u16;
            self.acc_lo[i] = prod as u16;
            self.vr[vd][i] = self.acc_lo[i];
        }
    }

    /// VMUDH: signed high partial product
    pub fn vmudh(&mut self, vd: usize, vs: usize, vt_broadcast: &Vreg) {
        for i in 0..8 {
            let s = self.vr[vs][i] as i16 as i32;
            let t = vt_broadcast[i] as i16 as i32;
            let prod = (s * t) as i64;
            self.acc_hi[i] = (prod >> 16) as u16;
            self.acc_md[i] = prod as u16;
            self.acc_lo[i] = 0;
            self.vr[vd][i] = clamp_signed(prod as i32);
        }
    }

    // --- Multiply-accumulate family ---

    /// Helper: read the 48-bit accumulator for element `i` as a sign-extended i64.
    #[inline]
    fn acc48(&self, i: usize) -> i64 {
        let hi = self.acc_hi[i] as i16 as i64;
        let md = self.acc_md[i] as i64;
        let lo = self.acc_lo[i] as i64;
        (hi << 32) | ((md & 0xFFFF) << 16) | (lo & 0xFFFF)
    }

    /// Helper: write a 48-bit value into the accumulator for element `i`.
    #[inline]
    fn set_acc48(&mut self, i: usize, val: i64) {
        self.acc_hi[i] = (val >> 32) as u16;
        self.acc_md[i] = (val >> 16) as u16;
        self.acc_lo[i] = val as u16;
    }

    /// VMACF: signed fraction multiply-accumulate
    pub fn vmacf(&mut self, vd: usize, vs: usize, vt_broadcast: &Vreg) {
        for i in 0..8 {
            let s = self.vr[vs][i] as i16 as i32;
            let t = vt_broadcast[i] as i16 as i32;
            let prod = (s * t) as i64 * 2;
            let acc = self.acc48(i).wrapping_add(prod);
            self.set_acc48(i, acc);
            self.vr[vd][i] = clamp_signed((acc >> 16) as i32);
        }
    }

    /// VMACU: unsigned fraction multiply-accumulate
    pub fn vmacu(&mut self, vd: usize, vs: usize, vt_broadcast: &Vreg) {
        for i in 0..8 {
            let s = self.vr[vs][i] as i16 as i32;
            let t = vt_broadcast[i] as i16 as i32;
            let prod = (s * t) as i64 * 2;
            let acc = self.acc48(i).wrapping_add(prod);
            self.set_acc48(i, acc);
            self.vr[vd][i] = clamp_unsigned((acc >> 16) as i32);
        }
    }

    /// VMADL: unsigned low partial accumulate
    pub fn vmadl(&mut self, vd: usize, vs: usize, vt_broadcast: &Vreg) {
        for i in 0..8 {
            let s = self.vr[vs][i] as u32;
            let t = vt_broadcast[i] as u32;
            let prod = ((s * t) >> 16) as u64;
            let acc = (self.acc48(i) as u64).wrapping_add(prod);
            self.set_acc48(i, acc as i64);
            // Result is clamped based on accumulator sign
            let sign = self.acc_hi[i] as i16;
            if sign < 0 {
                self.vr[vd][i] = if sign != -1 || self.acc_md[i] >= 0x8000 { 0 } else { self.acc_lo[i] };
            } else if sign != 0 || self.acc_md[i] >= 0x8000 {
                self.vr[vd][i] = 0xFFFF;
            } else {
                self.vr[vd][i] = self.acc_lo[i];
            }
        }
    }

    /// VMADM: signed*unsigned mid accumulate
    pub fn vmadm(&mut self, vd: usize, vs: usize, vt_broadcast: &Vreg) {
        for i in 0..8 {
            let s = self.vr[vs][i] as i16 as i32;
            let t = vt_broadcast[i] as u32;
            let prod = (s as i64) * (t as i64);
            let acc = self.acc48(i).wrapping_add(prod);
            self.set_acc48(i, acc);
            self.vr[vd][i] = clamp_signed((acc >> 16) as i32);
        }
    }

    /// VMADN: unsigned*signed mid accumulate
    pub fn vmadn(&mut self, vd: usize, vs: usize, vt_broadcast: &Vreg) {
        for i in 0..8 {
            let s = self.vr[vs][i] as u32;
            let t = vt_broadcast[i] as i16 as i32;
            let prod = (s as i64) * (t as i64);
            let acc = self.acc48(i).wrapping_add(prod);
            self.set_acc48(i, acc);
            // Result from acc low with clamping based on hi/md sign
            let sign = self.acc_hi[i] as i16;
            if sign < 0 {
                self.vr[vd][i] = if sign != -1 || self.acc_md[i] >= 0x8000 { 0 } else { self.acc_lo[i] };
            } else if sign != 0 || self.acc_md[i] >= 0x8000 {
                self.vr[vd][i] = 0xFFFF;
            } else {
                self.vr[vd][i] = self.acc_lo[i];
            }
        }
    }

    /// VMADH: signed high accumulate
    pub fn vmadh(&mut self, vd: usize, vs: usize, vt_broadcast: &Vreg) {
        for i in 0..8 {
            let s = self.vr[vs][i] as i16 as i32;
            let t = vt_broadcast[i] as i16 as i32;
            let prod = (s * t) as i64;
            let acc = self.acc48(i).wrapping_add(prod << 16);
            self.set_acc48(i, acc);
            self.vr[vd][i] = clamp_signed((acc >> 16) as i32);
        }
    }

    // -----------------------------------------------------------------------
    // Arithmetic
    // -----------------------------------------------------------------------

    /// VADD: vector add with carry-in from VCO low bits
    pub fn vadd(&mut self, vd: usize, vs: usize, vt_broadcast: &Vreg) {
        for i in 0..8 {
            let s = self.vr[vs][i] as i16 as i32;
            let t = vt_broadcast[i] as i16 as i32;
            let carry = ((self.vco >> i) & 1) as i32;
            let sum = s + t + carry;
            self.acc_lo[i] = sum as u16;
            self.vr[vd][i] = clamp_signed(sum);
        }
        self.vco = 0;
    }

    /// VSUB: vector subtract with borrow from VCO low bits
    pub fn vsub(&mut self, vd: usize, vs: usize, vt_broadcast: &Vreg) {
        for i in 0..8 {
            let s = self.vr[vs][i] as i16 as i32;
            let t = vt_broadcast[i] as i16 as i32;
            let borrow = ((self.vco >> i) & 1) as i32;
            let diff = s - t - borrow;
            self.acc_lo[i] = diff as u16;
            self.vr[vd][i] = clamp_signed(diff);
        }
        self.vco = 0;
    }

    /// VABS: absolute value
    pub fn vabs(&mut self, vd: usize, vs: usize, vt_broadcast: &Vreg) {
        for i in 0..8 {
            let s = self.vr[vs][i] as i16;
            let t = vt_broadcast[i] as i16;
            let result = if s < 0 {
                if t == -32768 { 32767i16 } else { -t }
            } else if s > 0 {
                t
            } else {
                0
            };
            self.acc_lo[i] = result as u16;
            self.vr[vd][i] = result as u16;
        }
    }

    /// VADDC: add with carry output
    pub fn vaddc(&mut self, vd: usize, vs: usize, vt_broadcast: &Vreg) {
        self.vco = 0;
        for i in 0..8 {
            let s = self.vr[vs][i] as u32;
            let t = vt_broadcast[i] as u32;
            let sum = s + t;
            self.acc_lo[i] = sum as u16;
            self.vr[vd][i] = sum as u16;
            if sum > 0xFFFF {
                self.vco |= 1 << i;
            }
        }
    }

    /// VSUBC: subtract with carry/borrow output
    pub fn vsubc(&mut self, vd: usize, vs: usize, vt_broadcast: &Vreg) {
        self.vco = 0;
        for i in 0..8 {
            let s = self.vr[vs][i] as u32;
            let t = vt_broadcast[i] as u32;
            let diff = s.wrapping_sub(t);
            self.acc_lo[i] = diff as u16;
            self.vr[vd][i] = diff as u16;
            // Low bit: borrow (s < t)
            if s < t {
                self.vco |= 1 << i;
            }
            // High bit: not-equal
            if diff != 0 {
                self.vco |= 1 << (i + 8);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Logic
    // -----------------------------------------------------------------------

    pub fn vand(&mut self, vd: usize, vs: usize, vt_broadcast: &Vreg) {
        for i in 0..8 {
            let r = self.vr[vs][i] & vt_broadcast[i];
            self.acc_lo[i] = r;
            self.vr[vd][i] = r;
        }
    }

    pub fn vnand(&mut self, vd: usize, vs: usize, vt_broadcast: &Vreg) {
        for i in 0..8 {
            let r = !(self.vr[vs][i] & vt_broadcast[i]);
            self.acc_lo[i] = r;
            self.vr[vd][i] = r;
        }
    }

    pub fn vor(&mut self, vd: usize, vs: usize, vt_broadcast: &Vreg) {
        for i in 0..8 {
            let r = self.vr[vs][i] | vt_broadcast[i];
            self.acc_lo[i] = r;
            self.vr[vd][i] = r;
        }
    }

    pub fn vnor(&mut self, vd: usize, vs: usize, vt_broadcast: &Vreg) {
        for i in 0..8 {
            let r = !(self.vr[vs][i] | vt_broadcast[i]);
            self.acc_lo[i] = r;
            self.vr[vd][i] = r;
        }
    }

    pub fn vxor(&mut self, vd: usize, vs: usize, vt_broadcast: &Vreg) {
        for i in 0..8 {
            let r = self.vr[vs][i] ^ vt_broadcast[i];
            self.acc_lo[i] = r;
            self.vr[vd][i] = r;
        }
    }

    pub fn vnxor(&mut self, vd: usize, vs: usize, vt_broadcast: &Vreg) {
        for i in 0..8 {
            let r = !(self.vr[vs][i] ^ vt_broadcast[i]);
            self.acc_lo[i] = r;
            self.vr[vd][i] = r;
        }
    }

    // -----------------------------------------------------------------------
    // Compare / select
    // -----------------------------------------------------------------------

    /// VLT: vector less-than
    pub fn vlt(&mut self, vd: usize, vs: usize, vt_broadcast: &Vreg) {
        self.vcc = 0;
        for i in 0..8 {
            let s = self.vr[vs][i] as i16;
            let t = vt_broadcast[i] as i16;
            let ne = (self.vco >> (i + 8)) & 1;
            let co = (self.vco >> i) & 1;
            let lt = (s < t) || (s == t && ne != 0 && co != 0);
            if lt {
                self.vcc |= 1 << i;
                self.vr[vd][i] = self.vr[vs][i];
            } else {
                self.vr[vd][i] = vt_broadcast[i];
            }
            self.acc_lo[i] = self.vr[vd][i];
        }
        self.vco = 0;
    }

    /// VEQ: vector equal
    pub fn veq(&mut self, vd: usize, vs: usize, vt_broadcast: &Vreg) {
        self.vcc = 0;
        for i in 0..8 {
            let ne = (self.vco >> (i + 8)) & 1;
            let eq = self.vr[vs][i] == vt_broadcast[i] && ne == 0;
            if eq {
                self.vcc |= 1 << i;
                self.vr[vd][i] = self.vr[vs][i];
            } else {
                self.vr[vd][i] = vt_broadcast[i];
            }
            self.acc_lo[i] = self.vr[vd][i];
        }
        self.vco = 0;
    }

    /// VNE: vector not-equal
    pub fn vne(&mut self, vd: usize, vs: usize, vt_broadcast: &Vreg) {
        self.vcc = 0;
        for i in 0..8 {
            let ne_flag = (self.vco >> (i + 8)) & 1;
            let neq = self.vr[vs][i] != vt_broadcast[i] || ne_flag != 0;
            if neq {
                self.vcc |= 1 << i;
                self.vr[vd][i] = self.vr[vs][i];
            } else {
                self.vr[vd][i] = vt_broadcast[i];
            }
            self.acc_lo[i] = self.vr[vd][i];
        }
        self.vco = 0;
    }

    /// VGE: vector greater-than-or-equal
    pub fn vge(&mut self, vd: usize, vs: usize, vt_broadcast: &Vreg) {
        self.vcc = 0;
        for i in 0..8 {
            let s = self.vr[vs][i] as i16;
            let t = vt_broadcast[i] as i16;
            let ne = (self.vco >> (i + 8)) & 1;
            let co = (self.vco >> i) & 1;
            let ge = (s > t) || (s == t && (ne == 0 || co == 0));
            if ge {
                self.vcc |= 1 << i;
                self.vr[vd][i] = self.vr[vs][i];
            } else {
                self.vr[vd][i] = vt_broadcast[i];
            }
            self.acc_lo[i] = self.vr[vd][i];
        }
        self.vco = 0;
    }

    /// VCH: vector clip half (sets VCC, VCO, VCE for subsequent VCL/VCR)
    pub fn vch(&mut self, vd: usize, vs: usize, vt_broadcast: &Vreg) {
        self.vcc = 0;
        self.vco = 0;
        self.vce = 0;
        for i in 0..8 {
            let s = self.vr[vs][i] as i16;
            let t = vt_broadcast[i] as i16;
            if (s ^ t) < 0 {
                // Different signs
                let sum = s.wrapping_add(t);
                let le = sum <= 0;
                if le {
                    self.vcc |= 1 << i; // clip low
                }
                let ge = (-t as i32) <= (s as i32);
                if ge {
                    self.vcc |= 1 << (i + 8); // clip high
                }
                if sum == -1 {
                    self.vce |= 1 << i;
                }
                self.vco |= 1 << i; // sign bit
                self.vco |= 1 << (i + 8); // ne
                self.vr[vd][i] = if le { (!t) as u16 } else { self.vr[vs][i] };
            } else {
                // Same sign
                let diff = s.wrapping_sub(t);
                let ge = diff >= 0;
                if ge {
                    self.vcc |= 1 << (i + 8);
                }
                let le = (s as i32) <= (t as i32);
                if le {
                    self.vcc |= 1 << i;
                }
                if diff != 0 {
                    self.vco |= 1 << (i + 8);
                }
                self.vr[vd][i] = if ge { self.vr[vs][i] } else { vt_broadcast[i] };
            }
            self.acc_lo[i] = self.vr[vd][i];
        }
    }

    /// VCL: vector clip low (uses state from VCH)
    pub fn vcl(&mut self, vd: usize, vs: usize, vt_broadcast: &Vreg) {
        for i in 0..8 {
            let s = self.vr[vs][i] as u16;
            let t = vt_broadcast[i] as u16;
            let sign = (self.vco >> i) & 1 != 0;
            let ne = (self.vco >> (i + 8)) & 1 != 0;
            let ce = (self.vce >> i) & 1 != 0;

            if sign {
                if ne {
                    let le = if ce {
                        (s as u32).wrapping_add(t as u32) <= 0xFFFF
                    } else {
                        (s as i16 as i32) + (t as i16 as i32) <= 0
                    };
                    if le {
                        self.vcc |= 1 << i;
                    } else {
                        self.vcc &= !(1 << i);
                    }
                }
                let clip = (self.vcc >> i) & 1 != 0;
                self.vr[vd][i] = if clip { (!t).wrapping_add(0) } else { s };
            } else {
                if ne {
                    let ge = (s as i16 as i32) >= (t as i16 as i32);
                    if ge {
                        self.vcc |= 1 << (i + 8);
                    } else {
                        self.vcc &= !(1 << (i + 8));
                    }
                }
                let clip = (self.vcc >> (i + 8)) & 1 != 0;
                self.vr[vd][i] = if clip { s } else { t };
            }
            self.acc_lo[i] = self.vr[vd][i];
        }
        self.vco = 0;
        self.vce = 0;
    }

    /// VCR: vector clip reciprocal (single-pass clip)
    pub fn vcr(&mut self, vd: usize, vs: usize, vt_broadcast: &Vreg) {
        self.vcc = 0;
        self.vco = 0;
        self.vce = 0;
        for i in 0..8 {
            let s = self.vr[vs][i] as i16;
            let t = vt_broadcast[i] as i16;
            if (s ^ t) < 0 {
                // Different signs
                let le = (s as i32) + (t as i32) + 1 <= 0;
                if le {
                    self.vcc |= 1 << i;
                }
                self.vr[vd][i] = if le { !vt_broadcast[i] } else { self.vr[vs][i] };
                let ge = (s as i32) + (t as i32) + 1 >= 0;
                if ge {
                    self.vcc |= 1 << (i + 8);
                }
            } else {
                // Same sign
                let ge = (s as i32) - (t as i32) >= 0;
                if ge {
                    self.vcc |= 1 << (i + 8);
                }
                self.vr[vd][i] = if ge { self.vr[vs][i] } else { vt_broadcast[i] };
                let le = (s as i32) - (t as i32) <= 0;
                if le {
                    self.vcc |= 1 << i;
                }
            }
            self.acc_lo[i] = self.vr[vd][i];
        }
    }

    /// VMRG: vector merge based on VCC
    pub fn vmrg(&mut self, vd: usize, vs: usize, vt_broadcast: &Vreg) {
        for i in 0..8 {
            self.vr[vd][i] = if (self.vcc >> i) & 1 != 0 {
                self.vr[vs][i]
            } else {
                vt_broadcast[i]
            };
            self.acc_lo[i] = self.vr[vd][i];
        }
        self.vco = 0;
    }

    // -----------------------------------------------------------------------
    // Accumulator read
    // -----------------------------------------------------------------------

    /// VSAR: vector accumulator read (e selects hi/md/lo)
    pub fn vsar(&mut self, vd: usize, e: u32) {
        match e {
            8 => self.vr[vd] = self.acc_hi,
            9 => self.vr[vd] = self.acc_md,
            10 => self.vr[vd] = self.acc_lo,
            _ => self.vr[vd] = [0; 8],
        }
    }

    // -----------------------------------------------------------------------
    // Reciprocal / RSQ
    // -----------------------------------------------------------------------

    /// VRCP: single-precision reciprocal
    pub fn vrcp(&mut self, vd: usize, de: usize, vt_elem: i16) {
        self.div_in = vt_elem as i32;
        self.div_out = rcp_lookup(self.div_in);
        self.vr[vd][de & 7] = self.div_out as u16;
        self.div_in_loaded = false;
    }

    /// VRCPL: reciprocal low (uses previously loaded high via VRCPH)
    pub fn vrcpl(&mut self, vd: usize, de: usize, vt_elem: i16) {
        if self.div_in_loaded {
            self.div_in = (self.div_in & !0xFFFF) | (vt_elem as u16 as i32);
        } else {
            self.div_in = vt_elem as i32;
        }
        self.div_out = rcp_lookup(self.div_in);
        self.vr[vd][de & 7] = self.div_out as u16;
        self.div_in_loaded = false;
    }

    /// VRCPH: reciprocal high — loads high 16 bits, returns div_out high
    pub fn vrcph(&mut self, vd: usize, de: usize, vt_elem: i16) {
        self.div_in = (vt_elem as i32) << 16;
        self.div_in_loaded = true;
        self.vr[vd][de & 7] = (self.div_out >> 16) as u16;
    }

    /// VRSQ: single-precision reciprocal square root
    pub fn vrsq(&mut self, vd: usize, de: usize, vt_elem: i16) {
        self.div_in = vt_elem as i32;
        self.div_out = rsq_lookup(self.div_in);
        self.vr[vd][de & 7] = self.div_out as u16;
        self.div_in_loaded = false;
    }

    /// VRSQL: RSQ low
    pub fn vrsql(&mut self, vd: usize, de: usize, vt_elem: i16) {
        if self.div_in_loaded {
            self.div_in = (self.div_in & !0xFFFF) | (vt_elem as u16 as i32);
        } else {
            self.div_in = vt_elem as i32;
        }
        self.div_out = rsq_lookup(self.div_in);
        self.vr[vd][de & 7] = self.div_out as u16;
        self.div_in_loaded = false;
    }

    /// VRSQH: RSQ high — loads high 16 bits, returns div_out high
    pub fn vrsqh(&mut self, vd: usize, de: usize, vt_elem: i16) {
        self.div_in = (vt_elem as i32) << 16;
        self.div_in_loaded = true;
        self.vr[vd][de & 7] = (self.div_out >> 16) as u16;
    }

    /// VMOV: move element
    pub fn vmov(&mut self, vd: usize, de: usize, vt_elem: u16) {
        self.vr[vd][de & 7] = vt_elem;
    }
}

// ---------------------------------------------------------------------------
// Vector load/store
// ---------------------------------------------------------------------------

/// Load bytes from DMEM into a vector register.
/// `dmem` is the 4KB DMEM, `base_addr` is the computed byte address (base+offset*scale).
pub fn vector_load(vu: &mut VectorUnit, dmem: &[u8; 4096], vt: usize, element: usize, addr: usize, opcode: usize) {
    match opcode {
        0 => {
            // LBV: load 1 byte
            let a = addr & 0xFFF;
            let byte_idx = element & 15;
            vu.vr[vt][byte_idx / 2] = if byte_idx & 1 == 0 {
                (dmem[a] as u16) << 8 | (vu.vr[vt][byte_idx / 2] & 0xFF)
            } else {
                (vu.vr[vt][byte_idx / 2] & 0xFF00) | dmem[a] as u16
            };
        }
        1 => {
            // LSV: load 2 bytes (halfword)
            let a = addr & 0xFFF;
            let start = element & 14;
            for i in 0..2usize {
                let byte_pos = (start + i) & 15;
                let b = dmem[(a + i) & 0xFFF];
                set_vr_byte(&mut vu.vr[vt], byte_pos, b);
            }
        }
        2 => {
            // LLV: load 4 bytes (word)
            let a = addr & 0xFFF;
            let start = element;
            for i in 0..4usize {
                let byte_pos = (start + i) & 15;
                let b = dmem[(a + i) & 0xFFF];
                set_vr_byte(&mut vu.vr[vt], byte_pos, b);
            }
        }
        3 => {
            // LDV: load 8 bytes (doubleword)
            let a = addr & 0xFFF;
            let start = element;
            for i in 0..8usize {
                let byte_pos = (start + i) & 15;
                let b = dmem[(a + i) & 0xFFF];
                set_vr_byte(&mut vu.vr[vt], byte_pos, b);
            }
        }
        4 => {
            // LQV: load up to 16 bytes (quad), stopping at 16-byte boundary
            let a = addr & 0xFFF;
            let end = (a | 15) + 1; // next 16-byte boundary
            let count = (end - a).min(16 - element);
            for i in 0..count {
                let byte_pos = (element + i) & 15;
                let b = dmem[(a + i) & 0xFFF];
                set_vr_byte(&mut vu.vr[vt], byte_pos, b);
            }
        }
        5 => {
            // LRV: load remainder (bytes before alignment)
            let a = addr & 0xFFF;
            let offset = a & 15;
            let aligned = a & !15;
            let start = 16 - offset;
            for i in start..16usize {
                let byte_pos = i;
                let b = dmem[(aligned + i) & 0xFFF];
                set_vr_byte(&mut vu.vr[vt], byte_pos, b);
            }
        }
        6 => {
            // LPV: load packed (8 upper bytes from unsigned)
            let a = addr & 0xFFF;
            for i in 0..8usize {
                let b = dmem[(a + ((element + i) & 0xF)) & 0xFFF] as u16;
                vu.vr[vt][i] = b << 8;
            }
        }
        7 => {
            // LUV: load unpacked (8 upper bytes, shifted)
            let a = addr & 0xFFF;
            for i in 0..8usize {
                let b = dmem[(a + ((element + i) & 0xF)) & 0xFFF] as u16;
                vu.vr[vt][i] = b << 7;
            }
        }
        11 => {
            // LTV: load transposed (loads elements across 8 consecutive vector registers)
            let a = addr & 0xFFF;
            let base_vt = vt & !7; // round down to group of 8
            let start = (element >> 1) & 7; // which element lane
            for i in 0..8usize {
                let vreg = base_vt + ((start + i) & 7);
                if vreg < 32 {
                    let hi = dmem[(a + i * 2) & 0xFFF];
                    let lo = dmem[(a + i * 2 + 1) & 0xFFF];
                    vu.vr[vreg][i] = (hi as u16) << 8 | lo as u16;
                }
            }
        }
        _ => {
            let n = LWC2_UNKNOWN_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n < 20 {
                eprintln!("[RSP LWC2 UNK] vt={} e={} addr=0x{:03X} sub=0x{:02X}", vt, element, addr & 0xFFF, opcode);
            }
        }
    }
}

/// Store bytes from a vector register to DMEM.
pub fn vector_store(vu: &VectorUnit, dmem: &mut [u8; 4096], vt: usize, element: usize, addr: usize, opcode: usize) {
    match opcode {
        0 => {
            // SBV: store 1 byte
            let a = addr & 0xFFF;
            dmem[a] = get_vr_byte(&vu.vr[vt], element & 15);
        }
        1 => {
            // SSV: store 2 bytes
            let a = addr & 0xFFF;
            let start = element & 14;
            for i in 0..2usize {
                let byte_pos = (start + i) & 15;
                dmem[(a + i) & 0xFFF] = get_vr_byte(&vu.vr[vt], byte_pos);
            }
        }
        2 => {
            // SLV: store 4 bytes
            let a = addr & 0xFFF;
            let start = element;
            for i in 0..4usize {
                let byte_pos = (start + i) & 15;
                dmem[(a + i) & 0xFFF] = get_vr_byte(&vu.vr[vt], byte_pos);
            }
        }
        3 => {
            // SDV: store 8 bytes
            let a = addr & 0xFFF;
            let start = element;
            for i in 0..8usize {
                let byte_pos = (start + i) & 15;
                dmem[(a + i) & 0xFFF] = get_vr_byte(&vu.vr[vt], byte_pos);
            }
        }
        4 => {
            // SQV: store up to 16 bytes, stopping at 16-byte boundary
            let a = addr & 0xFFF;
            let end = (a | 15) + 1;
            let count = (end - a).min(16 - element);
            for i in 0..count {
                let byte_pos = (element + i) & 15;
                dmem[(a + i) & 0xFFF] = get_vr_byte(&vu.vr[vt], byte_pos);
            }
        }
        5 => {
            // SRV: store remainder
            let a = addr & 0xFFF;
            let offset = a & 15;
            let aligned = a & !15;
            let start = 16 - offset;
            for i in start..16usize {
                dmem[(aligned + i) & 0xFFF] = get_vr_byte(&vu.vr[vt], i);
            }
        }
        6 => {
            // SPV: store packed
            let a = addr & 0xFFF;
            for i in 0..8usize {
                dmem[(a + i) & 0xFFF] = (vu.vr[vt][i] >> 8) as u8;
            }
        }
        7 => {
            // SUV: store unpacked
            let a = addr & 0xFFF;
            for i in 0..8usize {
                dmem[(a + i) & 0xFFF] = (vu.vr[vt][i] >> 7) as u8;
            }
        }
        11 => {
            // STV: store transposed
            let a = addr & 0xFFF;
            let base_vt = vt & !7;
            let start = (element >> 1) & 7;
            for i in 0..8usize {
                let vreg = base_vt + ((start + i) & 7);
                if vreg < 32 {
                    dmem[(a + i * 2) & 0xFFF] = (vu.vr[vreg][i] >> 8) as u8;
                    dmem[(a + i * 2 + 1) & 0xFFF] = vu.vr[vreg][i] as u8;
                }
            }
        }
        _ => {
            let n = SWC2_UNKNOWN_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n < 20 {
                eprintln!("[RSP SWC2 UNK] vt={} e={} addr=0x{:03X} sub=0x{:02X}", vt, element, addr & 0xFFF, opcode);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Byte-level access into vector registers (big-endian byte 0 = MSB of element 0)
// ---------------------------------------------------------------------------

/// Get byte `n` (0–15) from a vector register (big-endian).
#[inline]
pub fn get_vr_byte(vr: &Vreg, n: usize) -> u8 {
    let elem = (n & 15) / 2;
    if n & 1 == 0 {
        (vr[elem] >> 8) as u8
    } else {
        vr[elem] as u8
    }
}

/// Set byte `n` (0–15) in a vector register (big-endian).
#[inline]
pub fn set_vr_byte(vr: &mut Vreg, n: usize, val: u8) {
    let elem = (n & 15) / 2;
    if n & 1 == 0 {
        vr[elem] = (val as u16) << 8 | (vr[elem] & 0xFF);
    } else {
        vr[elem] = (vr[elem] & 0xFF00) | val as u16;
    }
}

// ---------------------------------------------------------------------------
// COP2 instruction dispatch (called from RSP interpreter)
// ---------------------------------------------------------------------------

/// Counter of unknown COP2 functs (rs >= 16 with funct not in our table).
pub static COP2_UNKNOWN_FUNCT_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
/// Counter of unknown COP2 rs values (rs < 16 not in {0,2,4,6}).
pub static COP2_UNKNOWN_RS_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
/// Counter of unknown LWC2 sub-opcodes.
pub static LWC2_UNKNOWN_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
/// Counter of unknown SWC2 sub-opcodes.
pub static SWC2_UNKNOWN_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
/// Last unknown COP2 funct seen (low 6 bits) | (rs<<8) for context.
pub static COP2_LAST_UNKNOWN: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
/// Histogram of COP2 functs (0..64) executed. Helps spot the dominant funct in a stuck loop.
pub static COP2_FUNCT_HIST: [std::sync::atomic::AtomicU32; 64] = {
    const Z: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    [Z; 64]
};

/// Execute a COP2 instruction. `iw` is the full 32-bit instruction word.
/// `scalar_regs` is the RSP scalar GPR file for MFC2/MTC2/CFC2/CTC2.
pub fn execute_cop2(vu: &mut VectorUnit, scalar_regs: &mut [u32; 32], _dmem: &mut [u8; 4096], iw: u32) {
    let rs = ((iw >> 21) & 31) as usize;
    let rt = ((iw >> 16) & 31) as usize;
    let rd = ((iw >> 11) & 31) as usize;
    let e = (iw >> 7) & 0xF; // element field for MFC2/MTC2

    match rs {
        0 => {
            // MFC2: move from vector register element to scalar GPR
            // rt = scalar dest, rd = vector src, e = byte element (bits 10:7)
            let byte_hi = (e as usize) & 15;
            let hi = get_vr_byte(&vu.vr[rd], byte_hi) as u16;
            let lo = get_vr_byte(&vu.vr[rd], (byte_hi + 1) & 15) as u16;
            let val = ((hi << 8) | lo) as i16 as i32 as u32;
            if rt != 0 {
                scalar_regs[rt] = val;
            }
        }
        4 => {
            // MTC2: move from scalar GPR to vector register element
            let val = if rt == 0 { 0u32 } else { scalar_regs[rt] };
            let byte_hi = (e as usize) & 15;
            set_vr_byte(&mut vu.vr[rd], byte_hi, (val >> 8) as u8);
            set_vr_byte(&mut vu.vr[rd], (byte_hi + 1) & 15, val as u8);
        }
        2 => {
            // CFC2: move from VCC/VCO/VCE to scalar GPR
            let val = match rd & 3 {
                0 => vu.vco as u32,
                1 => vu.vcc as u32,
                2 => vu.vce as u32,
                _ => 0,
            };
            if rt != 0 {
                scalar_regs[rt] = val as i16 as i32 as u32;
            }
        }
        6 => {
            // CTC2: move from scalar GPR to VCC/VCO/VCE
            let val = if rt == 0 { 0u32 } else { scalar_regs[rt] };
            match rd & 3 {
                0 => vu.vco = val as u16,
                1 => vu.vcc = val as u16,
                2 => vu.vce = (val & 0xFF) as u8,
                _ => {}
            }
        }
        _ if rs >= 16 => {
            // Vector compute: funct determines operation
            let funct = iw & 0x3F;
            COP2_FUNCT_HIST[funct as usize].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let vd = ((iw >> 6) & 31) as usize;
            let vs = rd; // rd field is vs for vector ops
            let vt = rt;
            let e_field = (iw >> 21) & 0xF;
            let vt_b = broadcast(&vu.vr[vt], e_field);

            match funct {
                // Multiply family
                0x00 => vu.vmulf(vd, vs, &vt_b),
                0x01 => vu.vmulu(vd, vs, &vt_b),
                0x04 => vu.vmudl(vd, vs, &vt_b),
                0x05 => vu.vmudm(vd, vs, &vt_b),
                0x06 => vu.vmudn(vd, vs, &vt_b),
                0x07 => vu.vmudh(vd, vs, &vt_b),
                0x08 => vu.vmacf(vd, vs, &vt_b),
                0x09 => vu.vmacu(vd, vs, &vt_b),
                0x0C => vu.vmadl(vd, vs, &vt_b),
                0x0D => vu.vmadm(vd, vs, &vt_b),
                0x0E => vu.vmadn(vd, vs, &vt_b),
                0x0F => vu.vmadh(vd, vs, &vt_b),

                // Arithmetic
                0x10 => vu.vadd(vd, vs, &vt_b),
                0x11 => vu.vsub(vd, vs, &vt_b),
                0x13 => vu.vabs(vd, vs, &vt_b),
                0x14 => vu.vaddc(vd, vs, &vt_b),
                0x15 => vu.vsubc(vd, vs, &vt_b),

                // Compare/select
                0x20 => vu.vlt(vd, vs, &vt_b),
                0x21 => vu.veq(vd, vs, &vt_b),
                0x22 => vu.vne(vd, vs, &vt_b),
                0x23 => vu.vge(vd, vs, &vt_b),
                0x24 => vu.vch(vd, vs, &vt_b),
                0x25 => vu.vcl(vd, vs, &vt_b),
                0x26 => vu.vcr(vd, vs, &vt_b),
                0x27 => vu.vmrg(vd, vs, &vt_b),

                // Logic
                0x28 => vu.vand(vd, vs, &vt_b),
                0x29 => vu.vnand(vd, vs, &vt_b),
                0x2A => vu.vor(vd, vs, &vt_b),
                0x2B => vu.vnor(vd, vs, &vt_b),
                0x2C => vu.vxor(vd, vs, &vt_b),
                0x2D => vu.vnxor(vd, vs, &vt_b),

                // Accumulator read
                0x1D => vu.vsar(vd, e_field),

                // Reciprocal family
                0x30 => {
                    // VRCP
                    let se = (e_field & 7) as usize;
                    let de = (iw >> 6) & 31;
                    vu.vrcp(vd, de as usize, vt_b[se] as i16);
                    // Also write acc_lo from vt
                    vu.acc_lo = vt_b;
                }
                0x31 => {
                    // VRCPL
                    let se = (e_field & 7) as usize;
                    let de = (iw >> 6) & 31;
                    vu.vrcpl(vd, de as usize, vt_b[se] as i16);
                    vu.acc_lo = vt_b;
                }
                0x32 => {
                    // VRCPH
                    let se = (e_field & 7) as usize;
                    let de = (iw >> 6) & 31;
                    vu.vrcph(vd, de as usize, vt_b[se] as i16);
                    vu.acc_lo = vt_b;
                }
                0x34 => {
                    // VRSQ
                    let se = (e_field & 7) as usize;
                    let de = (iw >> 6) & 31;
                    vu.vrsq(vd, de as usize, vt_b[se] as i16);
                    vu.acc_lo = vt_b;
                }
                0x35 => {
                    // VRSQL
                    let se = (e_field & 7) as usize;
                    let de = (iw >> 6) & 31;
                    vu.vrsql(vd, de as usize, vt_b[se] as i16);
                    vu.acc_lo = vt_b;
                }
                0x36 => {
                    // VRSQH
                    let se = (e_field & 7) as usize;
                    let de = (iw >> 6) & 31;
                    vu.vrsqh(vd, de as usize, vt_b[se] as i16);
                    vu.acc_lo = vt_b;
                }

                // Move
                0x33 => {
                    // VMOV
                    let se = (e_field & 7) as usize;
                    let de = ((iw >> 6) & 31) as usize;
                    vu.vmov(vd, de, vt_b[se]);
                    vu.acc_lo = vt_b;
                }

                0x37 => {} // VNOP

                _ => {
                    let n = COP2_UNKNOWN_FUNCT_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    COP2_LAST_UNKNOWN.store((rs as u32) << 8 | funct, std::sync::atomic::Ordering::Relaxed);
                    if n < 20 {
                        eprintln!("[RSP COP2 UNK funct] iw=0x{:08X} rs={} funct=0x{:02X}", iw, rs, funct);
                    }
                }
            }
        }
        _ => {
            let n = COP2_UNKNOWN_RS_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            COP2_LAST_UNKNOWN.store((rs as u32) << 8, std::sync::atomic::Ordering::Relaxed);
            if n < 20 {
                eprintln!("[RSP COP2 UNK rs] iw=0x{:08X} rs={}", iw, rs);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn element_broadcast_identity() {
        let map = element_map(0);
        assert_eq!(map, [0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn element_broadcast_scalar() {
        let map = element_map(10); // 8 + 2 => broadcast element 2
        assert_eq!(map, [2, 2, 2, 2, 2, 2, 2, 2]);
    }

    #[test]
    fn element_broadcast_pairs() {
        let map = element_map(2);
        assert_eq!(map, [0, 0, 2, 2, 4, 4, 6, 6]);
        let map = element_map(3);
        assert_eq!(map, [1, 1, 3, 3, 5, 5, 7, 7]);
    }

    #[test]
    fn element_broadcast_quads() {
        let map = element_map(4);
        assert_eq!(map, [0, 0, 0, 0, 4, 4, 4, 4]);
        let map = element_map(5);
        assert_eq!(map, [1, 1, 1, 1, 5, 5, 5, 5]);
    }

    #[test]
    fn vadd_basic() {
        let mut vu = VectorUnit::new();
        vu.vr[1] = [100, 200, 300, 400, 500, 600, 700, 800];
        vu.vr[2] = [10, 20, 30, 40, 50, 60, 70, 80];
        let vt_b = broadcast(&vu.vr[2], 0);
        vu.vadd(3, 1, &vt_b);
        assert_eq!(vu.vr[3], [110, 220, 330, 440, 550, 660, 770, 880]);
    }

    #[test]
    fn vsub_basic() {
        let mut vu = VectorUnit::new();
        vu.vr[1] = [100, 200, 300, 400, 500, 600, 700, 800];
        vu.vr[2] = [10, 20, 30, 40, 50, 60, 70, 80];
        let vt_b = broadcast(&vu.vr[2], 0);
        vu.vsub(3, 1, &vt_b);
        assert_eq!(vu.vr[3], [90, 180, 270, 360, 450, 540, 630, 720]);
    }

    #[test]
    fn vmudh_basic() {
        let mut vu = VectorUnit::new();
        vu.vr[1] = [3, 0, 0, 0, 0, 0, 0, 0];
        vu.vr[2] = [7, 0, 0, 0, 0, 0, 0, 0];
        let vt_b = broadcast(&vu.vr[2], 0);
        vu.vmudh(3, 1, &vt_b);
        assert_eq!(vu.vr[3][0], 21);
    }

    #[test]
    fn vand_vor_vxor() {
        let mut vu = VectorUnit::new();
        vu.vr[1] = [0xFF00, 0, 0, 0, 0, 0, 0, 0];
        vu.vr[2] = [0x0F0F, 0, 0, 0, 0, 0, 0, 0];
        let vt_b = broadcast(&vu.vr[2], 0);
        vu.vand(3, 1, &vt_b);
        assert_eq!(vu.vr[3][0], 0x0F00);
        vu.vor(4, 1, &vt_b);
        assert_eq!(vu.vr[4][0], 0xFF0F);
        vu.vxor(5, 1, &vt_b);
        assert_eq!(vu.vr[5][0], 0xF00F);
    }

    #[test]
    fn vmrg_basic() {
        let mut vu = VectorUnit::new();
        vu.vr[1] = [1, 2, 3, 4, 5, 6, 7, 8];
        vu.vr[2] = [10, 20, 30, 40, 50, 60, 70, 80];
        vu.vcc = 0b1010_1010; // elements 1,3,5,7 from vs; rest from vt
        let vt_b = broadcast(&vu.vr[2], 0);
        vu.vmrg(3, 1, &vt_b);
        assert_eq!(vu.vr[3], [10, 2, 30, 4, 50, 6, 70, 8]);
    }

    #[test]
    fn vsar_reads_accumulator() {
        let mut vu = VectorUnit::new();
        vu.acc_hi = [0x11; 8];
        vu.acc_md = [0x22; 8];
        vu.acc_lo = [0x33; 8];
        vu.vsar(1, 8); // hi
        assert_eq!(vu.vr[1], [0x11; 8]);
        vu.vsar(2, 9); // md
        assert_eq!(vu.vr[2], [0x22; 8]);
        vu.vsar(3, 10); // lo
        assert_eq!(vu.vr[3], [0x33; 8]);
    }

    #[test]
    fn vr_byte_access() {
        let mut vr: Vreg = [0xABCD, 0x1234, 0, 0, 0, 0, 0, 0];
        assert_eq!(get_vr_byte(&vr, 0), 0xAB);
        assert_eq!(get_vr_byte(&vr, 1), 0xCD);
        assert_eq!(get_vr_byte(&vr, 2), 0x12);
        assert_eq!(get_vr_byte(&vr, 3), 0x34);
        set_vr_byte(&mut vr, 0, 0xFF);
        assert_eq!(vr[0], 0xFFCD);
        set_vr_byte(&mut vr, 1, 0xEE);
        assert_eq!(vr[0], 0xFFEE);
    }

    #[test]
    fn lqv_sqv_round_trip() {
        let mut vu = VectorUnit::new();
        let mut dmem = [0u8; 4096];
        // Write 16 bytes starting at aligned address 0x100
        for i in 0..16u8 {
            dmem[0x100 + i as usize] = i + 1;
        }
        vector_load(&mut vu, &dmem, 5, 0, 0x100, 4); // LQV vr[5], 0, 0x100
        assert_eq!(get_vr_byte(&vu.vr[5], 0), 1);
        assert_eq!(get_vr_byte(&vu.vr[5], 15), 16);

        let mut dmem2 = [0u8; 4096];
        vector_store(&vu, &mut dmem2, 5, 0, 0x200, 4); // SQV vr[5], 0, 0x200
        for i in 0..16 {
            assert_eq!(dmem2[0x200 + i], dmem[0x100 + i]);
        }
    }

    #[test]
    fn mfc2_mtc2_round_trip() {
        let mut vu = VectorUnit::new();
        let mut regs = [0u32; 32];
        let mut dmem = [0u8; 4096];

        // MTC2: write 0x1234 to vr[5], element 0
        regs[1] = 0x1234;
        // MTC2 rt=1, rd=5, e=0: op=18, rs=4, rt=1, rd=5, e=0
        let iw = (18 << 26) | (4 << 21) | (1 << 16) | (5 << 11) | (0 << 7);
        execute_cop2(&mut vu, &mut regs, &mut dmem, iw);
        assert_eq!(vu.vr[5][0], 0x1234);

        // MFC2: read it back to r2
        let iw2 = (18 << 26) | (0 << 21) | (2 << 16) | (5 << 11) | (0 << 7);
        execute_cop2(&mut vu, &mut regs, &mut dmem, iw2);
        assert_eq!(regs[2], 0x1234);
    }

    #[test]
    fn vaddc_carry_out() {
        let mut vu = VectorUnit::new();
        vu.vr[1] = [0xFFFF, 0, 0, 0, 0, 0, 0, 0];
        vu.vr[2] = [1, 0, 0, 0, 0, 0, 0, 0];
        let vt_b = broadcast(&vu.vr[2], 0);
        vu.vaddc(3, 1, &vt_b);
        assert_eq!(vu.vr[3][0], 0); // wraps
        assert_ne!(vu.vco & 1, 0, "carry should be set for element 0");
    }
}
