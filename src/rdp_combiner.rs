//! RDP Color Combiner: mux decode and `(a - b) * c + d` per-channel evaluation.
//!
//! The N64 RDP has a two-cycle color combiner. Each cycle selects inputs a/b/c/d
//! for both color (RGB) and alpha channels from a set of sources. The combiner
//! equation is: `result = (a - b) * c + d`, evaluated per channel, clamped to [0,255].
//!
//! The `SetCombine` command (0xFC) packs selectors into an irregular 64-bit layout.

/// Multiply two RGBA5551 colors channel-wise (5/5/5/1), matching common `MODULATE` behavior.
#[inline]
pub fn rgba5551_modulate(tex: u16, prim: u16) -> u16 {
    let tr = (tex >> 11) & 0x1F;
    let tg = (tex >> 6) & 0x1F;
    let tb = (tex >> 1) & 0x1F;
    let ta = tex & 1;

    let pr = (prim >> 11) & 0x1F;
    let pg = (prim >> 6) & 0x1F;
    let pb = (prim >> 1) & 0x1F;
    let pa = prim & 1;

    let r = (tr * pr + 15) / 31;
    let g = (tg * pg + 15) / 31;
    let b = (tb * pb + 15) / 31;
    let a = ta & pa;

    ((r & 0x1F) << 11) | ((g & 0x1F) << 6) | ((b & 0x1F) << 1) | a
}

// ---------------------------------------------------------------------------
// Combiner mux selectors
// ---------------------------------------------------------------------------

/// Color combiner inputs for one cycle (a, b, c, d selectors for RGB).
#[derive(Clone, Copy, Debug, Default)]
pub struct CombinerCycle {
    pub a_rgb: u8,
    pub b_rgb: u8,
    pub c_rgb: u8,
    pub d_rgb: u8,
    pub a_alpha: u8,
    pub b_alpha: u8,
    pub c_alpha: u8,
    pub d_alpha: u8,
}

/// Decoded combiner state (two cycles).
#[derive(Clone, Copy, Debug, Default)]
pub struct CombinerMux {
    pub cycle: [CombinerCycle; 2],
}

/// Decode `SetCombine` mux word into per-cycle selectors.
///
/// The 64-bit mux is packed as follows (bits numbered 63..0):
///
/// ```text
/// Cycle 0:                          Cycle 1:
///   a_rgb:   bits 52-55 (4 bits)      a_rgb:   bits 28-31 (4 bits)  [w1 bits 28-31]
///   c_rgb:   bits 47-51 (5 bits)      c_rgb:   bits 23-27 (5 bits)  [w1 bits 23-27]
///   a_alpha:  bits 44-46 (3 bits)      a_alpha:  bits 21-23 (3 bits) [w1 bits 21-23]
///   c_alpha:  bits 41-43 (3 bits)      c_alpha:  bits 18-20 (3 bits) [w1 bits 18-20]
///   b_rgb:   bits 37-40 (4 bits) [across word boundary!]
///             ... complicated interleave ...
/// ```
///
/// The actual N64 bit layout (from n64brew / Angrylion):
///   w0 (bits 63-32 of the 64-bit word):
///     bits 20-23 (4): cycle 0 a_rgb
///     bits 15-19 (5): cycle 0 c_rgb
///     bits 12-14 (3): cycle 0 a_alpha
///     bits  9-11 (3): cycle 0 c_alpha
///     bits  5-8  (4): cycle 1 a_rgb
///     bits  0-4  (5): cycle 1 c_rgb
///
///   w1 (bits 31-0):
///     bits 28-31 (4): cycle 0 b_rgb
///     bits 24-27 (4): cycle 1 b_rgb
///     bits 21-23 (3): cycle 0 d_rgb  (actually from different field)
///     ... the exact layout is complex. Let's use the standard decode.
///
/// Standard decode per Angrylion / ares:
pub fn decode_combine(w0: u32, w1: u32) -> CombinerMux {
    let mut mux = CombinerMux::default();

    // Cycle 0 color (RGB)
    mux.cycle[0].a_rgb = ((w0 >> 20) & 0xF) as u8;
    mux.cycle[0].c_rgb = ((w0 >> 15) & 0x1F) as u8;
    mux.cycle[0].b_rgb = ((w1 >> 28) & 0xF) as u8;
    mux.cycle[0].d_rgb = ((w1 >> 15) & 0x7) as u8;

    // Cycle 0 alpha
    mux.cycle[0].a_alpha = ((w0 >> 12) & 0x7) as u8;
    mux.cycle[0].c_alpha = ((w0 >> 9) & 0x7) as u8;
    mux.cycle[0].b_alpha = ((w1 >> 12) & 0x7) as u8;
    mux.cycle[0].d_alpha = ((w1 >> 9) & 0x7) as u8;

    // Cycle 1 color (RGB)
    mux.cycle[1].a_rgb = ((w0 >> 5) & 0xF) as u8;
    mux.cycle[1].c_rgb = (w0 & 0x1F) as u8;
    mux.cycle[1].b_rgb = ((w1 >> 24) & 0xF) as u8;
    mux.cycle[1].d_rgb = ((w1 >> 6) & 0x7) as u8;

    // Cycle 1 alpha
    mux.cycle[1].a_alpha = ((w1 >> 21) & 0x7) as u8;
    mux.cycle[1].c_alpha = ((w1 >> 18) & 0x7) as u8;
    mux.cycle[1].b_alpha = ((w1 >> 3) & 0x7) as u8;
    mux.cycle[1].d_alpha = (w1 & 0x7) as u8;

    mux
}

// ---------------------------------------------------------------------------
// Combiner source lookup
// ---------------------------------------------------------------------------

/// RGBA8 color (0-255 per channel).
pub type Rgba8 = [u8; 4];

/// Color combiner RGB source (a/b/d selector, 4-bit field):
///   0: COMBINED   1: TEXEL0    2: TEXEL1    3: PRIM
///   4: SHADE      5: ENV       6: 1.0       7: NOISE (stub: 0)
///   8+: 0.0
#[inline]
fn cc_rgb_abd(sel: u8, texel0: &Rgba8, shade: &Rgba8, prim: &Rgba8, env: &Rgba8, combined: &Rgba8) -> [i16; 3] {
    match sel {
        0 => [combined[0] as i16, combined[1] as i16, combined[2] as i16],
        1 => [texel0[0] as i16, texel0[1] as i16, texel0[2] as i16],
        2 => [0, 0, 0], // TEXEL1 stub
        3 => [prim[0] as i16, prim[1] as i16, prim[2] as i16],
        4 => [shade[0] as i16, shade[1] as i16, shade[2] as i16],
        5 => [env[0] as i16, env[1] as i16, env[2] as i16],
        6 => [255, 255, 255], // 1.0
        _ => [0, 0, 0],
    }
}

/// Color combiner RGB 'c' source (multiply factor, 5-bit field):
///   0: COMBINED_A  1: TEXEL0_A    2: TEXEL1_A    3: PRIM_A
///   4: SHADE_A     5: ENV_A       6: unused       7: unused
///   8: TEXEL0     9: TEXEL1      10: PRIM        11: SHADE
///   12: ENV       13: LOD_FRAC   14: PRIM_LOD    15: K5 (stub)
///   16+: 0
#[inline]
fn cc_rgb_c(sel: u8, texel0: &Rgba8, shade: &Rgba8, prim: &Rgba8, env: &Rgba8, combined: &Rgba8) -> [i16; 3] {
    match sel {
        0 => { let a = combined[3] as i16; [a, a, a] }
        1 => { let a = texel0[3] as i16; [a, a, a] }
        2 => [0, 0, 0], // TEXEL1_A
        3 => { let a = prim[3] as i16; [a, a, a] }
        4 => { let a = shade[3] as i16; [a, a, a] }
        5 => { let a = env[3] as i16; [a, a, a] }
        8 => [texel0[0] as i16, texel0[1] as i16, texel0[2] as i16],
        9 => [0, 0, 0], // TEXEL1
        10 => [prim[0] as i16, prim[1] as i16, prim[2] as i16],
        11 => [shade[0] as i16, shade[1] as i16, shade[2] as i16],
        12 => [env[0] as i16, env[1] as i16, env[2] as i16],
        _ => [0, 0, 0],
    }
}

/// Alpha combiner source (a/b/d selector, 3-bit):
///   0: COMBINED_A  1: TEXEL0_A  2: TEXEL1_A  3: PRIM_A
///   4: SHADE_A     5: ENV_A     6: 1.0        7: 0.0
#[inline]
fn cc_alpha_abd(sel: u8, texel0: &Rgba8, shade: &Rgba8, prim: &Rgba8, env: &Rgba8, combined: &Rgba8) -> i16 {
    match sel {
        0 => combined[3] as i16,
        1 => texel0[3] as i16,
        2 => 0, // TEXEL1_A
        3 => prim[3] as i16,
        4 => shade[3] as i16,
        5 => env[3] as i16,
        6 => 255,
        _ => 0,
    }
}

/// Alpha combiner 'c' source (3-bit):
///   0: LOD_FRAC  1: TEXEL0_A  2: TEXEL1_A  3: PRIM_A
///   4: SHADE_A   5: ENV_A     6: PRIM_LOD  7: 0.0
#[inline]
fn cc_alpha_c(sel: u8, texel0: &Rgba8, shade: &Rgba8, prim: &Rgba8, env: &Rgba8) -> i16 {
    match sel {
        0 => 0, // LOD_FRAC stub
        1 => texel0[3] as i16,
        2 => 0, // TEXEL1_A
        3 => prim[3] as i16,
        4 => shade[3] as i16,
        5 => env[3] as i16,
        _ => 0,
    }
}

// ---------------------------------------------------------------------------
// Full combiner evaluation
// ---------------------------------------------------------------------------

/// Evaluate one combiner cycle: `(a - b) * c + d`, clamped to [0, 255].
pub fn evaluate_cycle(
    cycle: &CombinerCycle,
    texel0: &Rgba8,
    shade: &Rgba8,
    prim: &Rgba8,
    env: &Rgba8,
    combined: &Rgba8,
) -> Rgba8 {
    let a = cc_rgb_abd(cycle.a_rgb, texel0, shade, prim, env, combined);
    let b = cc_rgb_abd(cycle.b_rgb, texel0, shade, prim, env, combined);
    let c = cc_rgb_c(cycle.c_rgb, texel0, shade, prim, env, combined);
    let d = cc_rgb_abd(cycle.d_rgb, texel0, shade, prim, env, combined);

    let r = (((a[0] as i32 - b[0] as i32) * c[0] as i32) / 256 + d[0] as i32).clamp(0, 255) as u8;
    let g = (((a[1] as i32 - b[1] as i32) * c[1] as i32) / 256 + d[1] as i32).clamp(0, 255) as u8;
    let bl = (((a[2] as i32 - b[2] as i32) * c[2] as i32) / 256 + d[2] as i32).clamp(0, 255) as u8;

    let aa = cc_alpha_abd(cycle.a_alpha, texel0, shade, prim, env, combined) as i32;
    let ab = cc_alpha_abd(cycle.b_alpha, texel0, shade, prim, env, combined) as i32;
    let ac = cc_alpha_c(cycle.c_alpha, texel0, shade, prim, env) as i32;
    let ad = cc_alpha_abd(cycle.d_alpha, texel0, shade, prim, env, combined) as i32;
    let alpha = ((aa - ab) * ac / 256 + ad).clamp(0, 255) as u8;

    [r, g, bl, alpha]
}

/// Evaluate the full two-cycle combiner. For 1-cycle mode, only cycle 0 is used.
/// `cycle_type`: 0 = 1-cycle, 1 = 2-cycle, 2 = copy, 3 = fill.
pub fn evaluate_combiner(
    mux: &CombinerMux,
    cycle_type: u8,
    texel0: &Rgba8,
    shade: &Rgba8,
    prim: &Rgba8,
    env: &Rgba8,
) -> Rgba8 {
    match cycle_type {
        0 => {
            // 1-cycle: only cycle 0
            let combined = [0u8; 4]; // COMBINED input is 0 for first cycle
            evaluate_cycle(&mux.cycle[0], texel0, shade, prim, env, &combined)
        }
        1 => {
            // 2-cycle: cycle 0 feeds into cycle 1 as COMBINED
            let combined0 = [0u8; 4];
            let result0 = evaluate_cycle(&mux.cycle[0], texel0, shade, prim, env, &combined0);
            evaluate_cycle(&mux.cycle[1], texel0, shade, prim, env, &result0)
        }
        _ => {
            // Copy/fill: pass through texel
            *texel0
        }
    }
}

/// Convert RGBA8 to RGBA5551 (16-bit).
#[inline]
pub fn rgba8_to_5551(c: &Rgba8) -> u16 {
    let r = (c[0] as u16 >> 3) & 0x1F;
    let g = (c[1] as u16 >> 3) & 0x1F;
    let b = (c[2] as u16 >> 3) & 0x1F;
    let a = u16::from(c[3] >= 128);
    (r << 11) | (g << 6) | (b << 1) | a
}

/// Convert RGBA5551 to RGBA8.
#[inline]
pub fn rgba5551_to_rgba8(c: u16) -> Rgba8 {
    let r5 = (c >> 11) & 0x1F;
    let g5 = (c >> 6) & 0x1F;
    let b5 = (c >> 1) & 0x1F;
    let a1 = c & 1;
    [
        ((r5 << 3) | (r5 >> 2)) as u8,
        ((g5 << 3) | (g5 >> 2)) as u8,
        ((b5 << 3) | (b5 >> 2)) as u8,
        if a1 != 0 { 255 } else { 0 },
    ]
}

/// Convert u32 color (RGBA8888 as stored by SetPrimColor etc.) to Rgba8.
#[inline]
pub fn u32_to_rgba8(c: u32) -> Rgba8 {
    [
        (c >> 24) as u8,
        (c >> 16) as u8,
        (c >> 8) as u8,
        c as u8,
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modulate_white_is_identity() {
        let white = 0xFFFF;
        let color = 0b10101_01010_11111_1u16; // some color
        assert_eq!(rgba5551_modulate(color, white), color);
    }

    #[test]
    fn combiner_texel0_times_shade() {
        // SetCombine for TEXEL0 * SHADE:
        // Cycle 0: a=TEXEL0(1), b=0(8+), c=SHADE(11), d=0(8+)
        let cycle = CombinerCycle {
            a_rgb: 1,  // TEXEL0
            b_rgb: 8,  // 0
            c_rgb: 11, // SHADE
            d_rgb: 8,  // 0
            a_alpha: 1, // TEXEL0_A
            b_alpha: 7, // 0
            c_alpha: 4, // SHADE_A
            d_alpha: 7, // 0
        };
        let texel0: Rgba8 = [200, 100, 50, 255];
        let shade: Rgba8 = [128, 128, 128, 255];
        let prim: Rgba8 = [0; 4];
        let env: Rgba8 = [0; 4];
        let combined: Rgba8 = [0; 4];
        let result = evaluate_cycle(&cycle, &texel0, &shade, &prim, &env, &combined);
        // (200 - 0) * 128 / 256 + 0 = 100
        assert_eq!(result[0], 100);
        // (100 - 0) * 128 / 256 + 0 = 50
        assert_eq!(result[1], 50);
        // (50 - 0) * 128 / 256 + 0 = 25
        assert_eq!(result[2], 25);
    }

    #[test]
    fn combiner_prim_passthrough() {
        // a=0, b=0, c=0, d=PRIM => result = PRIM
        let cycle = CombinerCycle {
            a_rgb: 8, b_rgb: 8, c_rgb: 16, d_rgb: 3, // d=PRIM
            a_alpha: 7, b_alpha: 7, c_alpha: 7, d_alpha: 3, // d=PRIM_A
        };
        let texel0: Rgba8 = [0; 4];
        let shade: Rgba8 = [0; 4];
        let prim: Rgba8 = [255, 128, 64, 200];
        let env: Rgba8 = [0; 4];
        let combined: Rgba8 = [0; 4];
        let result = evaluate_cycle(&cycle, &texel0, &shade, &prim, &env, &combined);
        assert_eq!(result, [255, 128, 64, 200]);
    }

    #[test]
    fn rgba8_5551_round_trip() {
        let orig: Rgba8 = [248, 128, 64, 255]; // values that survive 5-bit truncation
        let packed = rgba8_to_5551(&orig);
        let back = rgba5551_to_rgba8(packed);
        // Allow ±7 due to 5-bit precision
        for i in 0..3 {
            assert!((back[i] as i16 - orig[i] as i16).unsigned_abs() <= 7);
        }
        assert_eq!(back[3], 255); // alpha 1
    }

    #[test]
    fn decode_combine_basic() {
        // Encode a known combiner: TEXEL0 * SHADE for cycle 0
        // a_rgb=1(TEXEL0), b_rgb=0, c_rgb=11(SHADE), d_rgb=0
        // Pack into w0/w1 per the layout:
        // w0 bits 20-23: a_rgb0=1  => w0 |= 1 << 20
        // w0 bits 15-19: c_rgb0=11 => w0 |= 11 << 15
        // w1 bits 28-31: b_rgb0=0
        // w1 bits 15-17: d_rgb0=0
        let w0 = (1u32 << 20) | (11u32 << 15);
        let w1 = 0u32;
        let mux = decode_combine(w0, w1);
        assert_eq!(mux.cycle[0].a_rgb, 1);
        assert_eq!(mux.cycle[0].c_rgb, 11);
        assert_eq!(mux.cycle[0].b_rgb, 0);
        assert_eq!(mux.cycle[0].d_rgb, 0);
    }
}
