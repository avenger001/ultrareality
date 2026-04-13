//! Hardware-style triangle decode, flat fill, and shaded/textured rasterization.
//!
//! - Fill triangles: `fill_color` flat fill
//! - Shaded triangles: interpolated RGBA vertex colors → combiner → framebuffer
//! - Textured triangles: TMEM sampling (CI4, CI8, RGBA16, IA, I) → combiner
//! - Shaded + textured: both shade interpolation and texture sampling
//! - Z-buffer: per-pixel depth compare and update

use crate::rdp::{TileSlot, TMEM_SIZE};
use crate::rdp_combiner::{
    CombinerMux, Rgba8, evaluate_combiner, rgba8_to_5551, rgba5551_to_rgba8,
};
use crate::timing::RDRAM_BUS_CYCLES_PER_BYTE;

/// Bytes consumed by an RDP command, indexed by `(w0 >> 24) & 0x3F` (see GLideN64 `CmdLength`).
pub const CMD_LENGTH: [usize; 64] = [
    8, 8, 8, 8, 8, 8, 8, 8, // 0x00–0x07
    32,
    32 + 16,
    32 + 64,
    32 + 64 + 16,
    32 + 64,
    32 + 64 + 16,
    32 + 64 + 64,
    32 + 64 + 64 + 16, // 0x08–0x0f triangles
    8, 8, 8, 8, 8, 8, 8, 8, // 0x10–0x17
    8, 8, 8, 8, 8, 8, 8, 8, // 0x18–0x1f
    8, 8, 8, 8, 16, 16, 8, 8, // 0x20–0x27 (texrect etc.)
    8, 8, 8, 8, 8, 8, 8, 8, // 0x28–0x2f
    8, 8, 8, 8, 8, 8, 8, 8, // 0x30–0x37
    8, 8, 8, 8, 8, 8, 8, 8, // 0x38–0x3f
];

#[inline]
pub fn command_bytes(w0: u32) -> usize {
    CMD_LENGTH[((w0 >> 24) & 0x3F) as usize]
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

#[inline]
fn sign_extend(w: i32, bits: u32) -> i32 {
    let shift = 32 - bits;
    w.wrapping_shl(shift).wrapping_shr(shift)
}

/// `_FIXED2FLOAT((x & !1), 16)` — convert s15.16-style fixed to float (GLideN64).
#[inline]
fn fixed_to_f64(x: i32) -> f64 {
    let v = ((x as i64) & !1) as f64;
    v / 65536.0
}

/// Slope term: `_FIXED2FLOAT(((dx >> 2) & !1), 16)`.
#[inline]
fn slope_to_f64(dx: i32) -> f64 {
    let t = (((dx as i64) >> 2) & !1) as f64;
    t / 65536.0
}

#[inline]
fn texel_bytes(size: u8) -> usize {
    match size & 3 {
        0 => 1,
        1 => 2,
        2 => 2,
        3 => 4,
        _ => 2,
    }
}

// ---------------------------------------------------------------------------
// Triangle coefficient types and parsing
// ---------------------------------------------------------------------------

/// Per-channel coefficients: initial value and derivatives for edge/X/Y stepping.
#[derive(Clone, Copy, Debug, Default)]
pub struct ChannelCoeffs {
    /// Initial value at the triangle's starting position.
    pub val: f64,
    /// Per-pixel horizontal derivative (dC/dx).
    pub dx: f64,
    /// Per-scanline edge derivative (dC/de).
    pub de: f64,
    /// Per-scanline Y derivative (dC/dy).
    pub dy: f64,
}

/// Parsed shade/texture/Z coefficients from a triangle command.
#[derive(Clone, Debug, Default)]
pub struct TriCoeffs {
    /// Shade RGBA coefficients [R, G, B, A].
    pub shade: Option<[ChannelCoeffs; 4]>,
    /// Texture S/T/W coefficients [S, T, W, unused].
    pub tex: Option<[ChannelCoeffs; 4]>,
    /// Z depth coefficient.
    pub z: Option<ChannelCoeffs>,
}

/// Combine high and low 16-bit halves into an s15.16 fixed-point value, returned as f64.
#[inline]
fn assemble_fixed(hi: u16, lo: u16) -> f64 {
    let val = ((hi as i16 as i32) << 16) | (lo as u32 as i32);
    val as f64 / 65536.0
}

/// Parse a 4-channel coefficient block (16 words) → `[ch0, ch1, ch2, ch3]`.
///
/// Layout per N64 hardware (channel pairs packed per word):
/// - Words 0–1: initial values (hi), Words 4–5: initial values (lo)
/// - Words 2–3: dCdx (hi), Words 6–7: dCdx (lo)
/// - Words 8–9: dCde (hi), Words 12–13: dCde (lo)
/// - Words 10–11: dCdy (hi), Words 14–15: dCdy (lo)
fn parse_4ch_coeffs(w: &[u32]) -> [ChannelCoeffs; 4] {
    let u = |word: u32| (word >> 16) as u16;
    let l = |word: u32| word as u16;

    [
        // Channel 0 (R / S): upper halves of even-indexed word pairs
        ChannelCoeffs {
            val: assemble_fixed(u(w[0]), u(w[4])),
            dx:  assemble_fixed(u(w[2]), u(w[6])),
            de:  assemble_fixed(u(w[8]), u(w[12])),
            dy:  assemble_fixed(u(w[10]), u(w[14])),
        },
        // Channel 1 (G / T): lower halves of even-indexed word pairs
        ChannelCoeffs {
            val: assemble_fixed(l(w[0]), l(w[4])),
            dx:  assemble_fixed(l(w[2]), l(w[6])),
            de:  assemble_fixed(l(w[8]), l(w[12])),
            dy:  assemble_fixed(l(w[10]), l(w[14])),
        },
        // Channel 2 (B / W): upper halves of odd-indexed word pairs
        ChannelCoeffs {
            val: assemble_fixed(u(w[1]), u(w[5])),
            dx:  assemble_fixed(u(w[3]), u(w[7])),
            de:  assemble_fixed(u(w[9]), u(w[13])),
            dy:  assemble_fixed(u(w[11]), u(w[15])),
        },
        // Channel 3 (A / unused): lower halves of odd-indexed word pairs
        ChannelCoeffs {
            val: assemble_fixed(l(w[1]), l(w[5])),
            dx:  assemble_fixed(l(w[3]), l(w[7])),
            de:  assemble_fixed(l(w[9]), l(w[13])),
            dy:  assemble_fixed(l(w[11]), l(w[15])),
        },
    ]
}

/// Parse Z coefficient block (4 words).
fn parse_z_coeffs(w: &[u32]) -> ChannelCoeffs {
    ChannelCoeffs {
        val: assemble_fixed((w[0] >> 16) as u16, (w[2] >> 16) as u16),
        dx:  assemble_fixed(w[0] as u16, w[2] as u16),
        de:  assemble_fixed((w[1] >> 16) as u16, (w[3] >> 16) as u16),
        dy:  assemble_fixed(w[1] as u16, w[3] as u16),
    }
}

/// Parse triangle coefficients from the command words based on opcode feature bits.
///
/// Feature bits (low 3 bits of opcode):
/// - bit 2: shade (64 bytes)
/// - bit 1: texture (64 bytes)
/// - bit 0: Z depth (16 bytes)
///
/// Data order after the 8-word base geometry: shade, then texture, then Z.
pub fn parse_tri_coeffs(words: &[u32], op: u8) -> TriCoeffs {
    let features = op & 7;
    let has_shade = features & 4 != 0;
    let has_tex = features & 2 != 0;
    let has_z = features & 1 != 0;

    let mut off = 8;

    let shade = if has_shade && words.len() >= off + 16 {
        let s = parse_4ch_coeffs(&words[off..off + 16]);
        off += 16;
        Some(s)
    } else {
        None
    };

    let tex = if has_tex && words.len() >= off + 16 {
        let t = parse_4ch_coeffs(&words[off..off + 16]);
        off += 16;
        Some(t)
    } else {
        None
    };

    let z = if has_z && words.len() >= off + 4 {
        Some(parse_z_coeffs(&words[off..off + 4]))
    } else {
        None
    };

    TriCoeffs { shade, tex, z }
}

// ---------------------------------------------------------------------------
// TMEM texture sampling
// ---------------------------------------------------------------------------

/// Convert IA16 (intensity + alpha, 8 bits each) to RGBA8.
#[inline]
fn ia16_to_rgba8(c: u16) -> Rgba8 {
    let i = (c >> 8) as u8;
    let a = c as u8;
    [i, i, i, a]
}

/// Sample a texel from TMEM at integer coordinates `(s, t)`.
///
/// Supports CI4, CI8, RGBA16, RGBA32, IA4, IA8, IA16, I4, I8 formats.
/// TLUT (palette) lookups use the upper TMEM half (offset 0x800).
pub fn sample_tmem(tmem: &[u8], tile: &TileSlot, s: i32, t: i32, tlut_type: u8) -> Rgba8 {
    let tmem_base = tile.tmem_qwords as usize * 8;
    let stride = (tile.line_qwords as usize * 8).max(1);
    let s = s.max(0) as usize;
    let t = t.max(0) as usize;

    match (tile.fmt, tile.siz) {
        // CI4: 4-bit color index → TLUT
        (2, 0) => {
            let addr = tmem_base + t * stride + s / 2;
            if addr >= TMEM_SIZE { return [0; 4]; }
            let byte = tmem[addr];
            let index = if s & 1 == 0 { byte >> 4 } else { byte & 0xF } as usize;
            let pal = tile.palette as usize;
            let tlut_addr = 0x800 + (pal * 16 + index) * 2;
            if tlut_addr + 1 >= TMEM_SIZE { return [0; 4]; }
            let c16 = u16::from_be_bytes([tmem[tlut_addr], tmem[tlut_addr + 1]]);
            if tlut_type == 0 { rgba5551_to_rgba8(c16) } else { ia16_to_rgba8(c16) }
        }
        // CI8: 8-bit color index → TLUT
        (2, 1) => {
            let addr = tmem_base + t * stride + s;
            if addr >= TMEM_SIZE { return [0; 4]; }
            let index = tmem[addr] as usize;
            let tlut_addr = 0x800 + index * 2;
            if tlut_addr + 1 >= TMEM_SIZE { return [0; 4]; }
            let c16 = u16::from_be_bytes([tmem[tlut_addr], tmem[tlut_addr + 1]]);
            if tlut_type == 0 { rgba5551_to_rgba8(c16) } else { ia16_to_rgba8(c16) }
        }
        // RGBA16
        (0, 2) => {
            let addr = tmem_base + t * stride + s * 2;
            if addr + 1 >= TMEM_SIZE { return [0; 4]; }
            rgba5551_to_rgba8(u16::from_be_bytes([tmem[addr], tmem[addr + 1]]))
        }
        // RGBA32
        (0, 3) => {
            let addr = tmem_base + t * stride + s * 4;
            if addr + 3 >= TMEM_SIZE { return [0; 4]; }
            [tmem[addr], tmem[addr + 1], tmem[addr + 2], tmem[addr + 3]]
        }
        // IA4: 3-bit intensity + 1-bit alpha (packed nibbles)
        (3, 0) => {
            let addr = tmem_base + t * stride + s / 2;
            if addr >= TMEM_SIZE { return [0; 4]; }
            let nib = if s & 1 == 0 { tmem[addr] >> 4 } else { tmem[addr] & 0xF };
            let i3 = (nib >> 1) & 7;
            let i = (i3 << 5) | (i3 << 2) | (i3 >> 1);
            let a = if nib & 1 != 0 { 255 } else { 0 };
            [i, i, i, a]
        }
        // IA8: 4-bit intensity + 4-bit alpha
        (3, 1) => {
            let addr = tmem_base + t * stride + s;
            if addr >= TMEM_SIZE { return [0; 4]; }
            let byte = tmem[addr];
            let i4 = byte >> 4;
            let a4 = byte & 0xF;
            let i = i4 | (i4 << 4);
            let a = a4 | (a4 << 4);
            [i, i, i, a]
        }
        // IA16: 8-bit intensity + 8-bit alpha
        (3, 2) => {
            let addr = tmem_base + t * stride + s * 2;
            if addr + 1 >= TMEM_SIZE { return [0; 4]; }
            let i = tmem[addr];
            let a = tmem[addr + 1];
            [i, i, i, a]
        }
        // I4: 4-bit intensity
        (4, 0) => {
            let addr = tmem_base + t * stride + s / 2;
            if addr >= TMEM_SIZE { return [0; 4]; }
            let nib = if s & 1 == 0 { tmem[addr] >> 4 } else { tmem[addr] & 0xF };
            let i = nib | (nib << 4);
            [i, i, i, 255]
        }
        // I8: 8-bit intensity
        (4, 1) => {
            let addr = tmem_base + t * stride + s;
            if addr >= TMEM_SIZE { return [0; 4]; }
            let i = tmem[addr];
            [i, i, i, 255]
        }
        // Fallback: treat as RGBA16
        _ => {
            let bpp = texel_bytes(tile.siz);
            let addr = tmem_base + t * stride + s * bpp;
            if addr + 1 >= TMEM_SIZE { return [0; 4]; }
            rgba5551_to_rgba8(u16::from_be_bytes([tmem[addr], tmem[addr + 1]]))
        }
    }
}

// ---------------------------------------------------------------------------
// Rasterization context
// ---------------------------------------------------------------------------

/// RDP state needed for shaded/textured triangle rasterization.
pub struct RasterCtx<'a> {
    pub tmem: &'a [u8],
    pub tiles: &'a [TileSlot; 8],
    pub combiner: &'a CombinerMux,
    pub cycle_type: u8,
    pub prim_color: Rgba8,
    pub env_color: Rgba8,
    pub blend_color: Rgba8,
    pub fog_color: Rgba8,
    pub fill_color: u32,
    pub ci_addr: u32,
    pub ci_width: u16,
    pub ci_size: u8,
    pub z_image_addr: u32,
    pub z_compare_en: bool,
    pub z_update_en: bool,
    pub z_source_sel: bool,
    pub prim_z: u16,
    pub tlut_type: u8,
    pub force_blend: bool,
    /// Blender mux cycle 0: P(m1a), A(m1b), Q(m2a), B(m2b) — each 2 bits.
    pub blend_m1a_0: u8,
    pub blend_m1b_0: u8,
    pub blend_m2a_0: u8,
    pub blend_m2b_0: u8,
    /// Blender mux cycle 1.
    pub blend_m1a_1: u8,
    pub blend_m1b_1: u8,
    pub blend_m2a_1: u8,
    pub blend_m2b_1: u8,
    pub clip: Option<(i32, i32, i32, i32)>,
}

// ---------------------------------------------------------------------------
// Blender
// ---------------------------------------------------------------------------

/// Read an existing framebuffer pixel as RGBA8.
#[inline]
fn read_fb_rgba8(rdram: &[u8], off: usize, bpp: usize) -> Rgba8 {
    if bpp == 2 && off + 1 < rdram.len() {
        rgba5551_to_rgba8(u16::from_be_bytes([rdram[off], rdram[off + 1]]))
    } else if bpp == 4 && off + 3 < rdram.len() {
        [rdram[off], rdram[off + 1], rdram[off + 2], rdram[off + 3]]
    } else {
        [0, 0, 0, 0]
    }
}

/// Evaluate one blender cycle: `(P * A + Q * B) / (A + B)`.
///
/// Mux selectors (2 bits each):
/// - **m1a** (P): 0 = IN_COLOR, 1 = MEM_COLOR, 2 = BLEND_COLOR, 3 = FOG_COLOR
/// - **m1b** (A): 0 = IN_ALPHA, 1 = FOG_ALPHA, 2 = SHADE_ALPHA, 3 = 0
/// - **m2a** (Q): 0 = IN_COLOR, 1 = MEM_COLOR, 2 = BLEND_COLOR, 3 = FOG_COLOR
/// - **m2b** (B): 0 = 1−A, 1 = MEM_ALPHA, 2 = 1, 3 = 0
fn evaluate_blender(
    in_color: &Rgba8,
    shade_alpha: u8,
    mem_color: &Rgba8,
    blend_color: &Rgba8,
    fog_color: &Rgba8,
    m1a: u8,
    m1b: u8,
    m2a: u8,
    m2b: u8,
) -> Rgba8 {
    let p = match m1a {
        0 => in_color,
        1 => mem_color,
        2 => blend_color,
        _ => fog_color,
    };
    let a: u16 = match m1b {
        0 => in_color[3] as u16,
        1 => fog_color[3] as u16,
        2 => shade_alpha as u16,
        _ => 0,
    };
    let q = match m2a {
        0 => in_color,
        1 => mem_color,
        2 => blend_color,
        _ => fog_color,
    };
    let b: u16 = match m2b {
        0 => 255u16.saturating_sub(a),
        1 => mem_color[3] as u16,
        2 => 255,
        _ => 0,
    };

    let denom = (a + b).max(1) as u32;
    let ch = |i: usize| -> u8 {
        ((p[i] as u32 * a as u32 + q[i] as u32 * b as u32) / denom).min(255) as u8
    };
    [ch(0), ch(1), ch(2), in_color[3]]
}

// ---------------------------------------------------------------------------
// Full shaded/textured triangle rasterizer
// ---------------------------------------------------------------------------

/// Rasterize a shaded/textured/Z triangle. Returns `(RDRAM write cycles, pixels written)`.
///
/// Walks the triangle scanlines using the same edge-stepping geometry as [`raster_fill_hw`],
/// then per-pixel: interpolates shade RGBA and texture S/T/W from [`TriCoeffs`], samples
/// TMEM, evaluates the color combiner, performs Z-buffer compare/update, and writes the
/// framebuffer.
pub fn raster_shaded_hw(
    words: &[u32],
    coeffs: &TriCoeffs,
    ctx: &RasterCtx,
    rdram: &mut [u8],
) -> (u64, u64) {
    if words.len() < 8 {
        return (0, 0);
    }

    let s = |w: i32, bits: u32| sign_extend(w, bits);

    let mut yh = s(words[1] as i32, 14);
    let ym = s((words[1] as i32) >> 16, 14);
    let yl = s(words[0] as i32, 14);
    yh &= !3;

    let xl = s(words[2] as i32, 28);
    let dxldy = s(words[3] as i32, 30);
    let xh = s(words[4] as i32, 28);
    let dxhdy = s(words[5] as i32, 30);
    let xm = s(words[6] as i32, 28);
    let dxmdy = s(words[7] as i32, 30);

    let xhf = fixed_to_f64(xh);
    let xmf = fixed_to_f64(xm);
    let xlf = fixed_to_f64(xl);

    let hk = slope_to_f64(dxhdy);
    let mk = slope_to_f64(dxmdy);
    let lk = slope_to_f64(dxldy);

    let yhf = yh as f64;
    let ymf = ym as f64;
    let ylf = yl as f64;

    let hc = xhf - hk * yhf;
    let mc = xmf - mk * yhf;
    let lc = xlf - lk * ymf;

    let tile_idx = ((words[0] >> 18) & 7) as usize;
    let tile = &ctx.tiles[tile_idx.min(7)];

    let bpp = texel_bytes(ctx.ci_size);
    let cw = ctx.ci_width.max(1) as i32;
    let base = (ctx.ci_addr & 0x00FF_FFFF) as usize;

    let py0 = (yh + 2) >> 2;
    let py1 = (yl + 2) >> 2;

    let mut pix = 0u64;

    for py in py0..=py1 {
        let y_sub = ((py as i64) << 2) as f64;
        if y_sub < yhf || y_sub > ylf {
            continue;
        }

        // Compute left/right edge X positions at this scanline
        let xa = hk * y_sub + hc;
        let xb = if y_sub < ymf {
            mk * y_sub + mc
        } else {
            lk * y_sub + lc
        };
        let mut x0 = xa.min(xb).floor() as i32;
        let mut x1 = xa.max(xb).ceil() as i32;

        if let Some((sxa, sxb, sya, syb)) = ctx.clip {
            if py < sya || py > syb {
                continue;
            }
            x0 = x0.max(sxa);
            x1 = x1.min(sxb);
        }
        if x1 < x0 || py < 0 {
            continue;
        }

        // Scanline step factor: number of scanlines from the triangle top (yh)
        let scan_steps = (y_sub - yhf) / 4.0;

        for x in x0..=x1 {
            if x < 0 {
                continue;
            }

            // Pixel offset from the h-edge X position (reference for dCdx)
            let x_off = x as f64 - xa;

            // --- Interpolate shade RGBA ---
            let shade = if let Some(ref sc) = coeffs.shade {
                [
                    (sc[0].val + sc[0].de * scan_steps + sc[0].dx * x_off).clamp(0.0, 255.0) as u8,
                    (sc[1].val + sc[1].de * scan_steps + sc[1].dx * x_off).clamp(0.0, 255.0) as u8,
                    (sc[2].val + sc[2].de * scan_steps + sc[2].dx * x_off).clamp(0.0, 255.0) as u8,
                    (sc[3].val + sc[3].de * scan_steps + sc[3].dx * x_off).clamp(0.0, 255.0) as u8,
                ]
            } else {
                [255, 255, 255, 255]
            };

            // --- Interpolate texture S/T/W and sample TMEM ---
            let texel = if let Some(ref tc) = coeffs.tex {
                let s_raw = tc[0].val + tc[0].de * scan_steps + tc[0].dx * x_off;
                let t_raw = tc[1].val + tc[1].de * scan_steps + tc[1].dx * x_off;
                let w_raw = tc[2].val + tc[2].de * scan_steps + tc[2].dx * x_off;

                // Perspective correction: S and T are stored as S/W and T/W;
                // dividing by 1/W (the W channel) recovers the true coordinate.
                let (s_f, t_f) = if w_raw.abs() > 1e-6 {
                    (s_raw / w_raw, t_raw / w_raw)
                } else {
                    (s_raw, t_raw)
                };

                let si = s_f.floor() as i32;
                let ti = t_f.floor() as i32;
                sample_tmem(ctx.tmem, tile, si, ti, ctx.tlut_type)
            } else {
                [255, 255, 255, 255]
            };

            // --- Z-buffer ---
            let z_needed = ctx.z_compare_en || ctx.z_update_en;
            if z_needed {
                let z_val = if ctx.z_source_sel {
                    ctx.prim_z as f64
                } else if let Some(ref zc) = coeffs.z {
                    zc.val + zc.de * scan_steps + zc.dx * x_off
                } else {
                    0.0
                };
                let new_z = z_val.clamp(0.0, 65535.0) as u16;
                let z_off = (ctx.z_image_addr & 0x00FF_FFFF) as usize
                    + ((py as usize) * cw as usize + (x as usize)) * 2;
                if z_off + 1 < rdram.len() {
                    if ctx.z_compare_en {
                        let stored_z =
                            u16::from_be_bytes([rdram[z_off], rdram[z_off + 1]]);
                        // Pixel passes if closer (smaller or equal Z)
                        if new_z > stored_z {
                            continue;
                        }
                    }
                    if ctx.z_update_en {
                        rdram[z_off..z_off + 2].copy_from_slice(&new_z.to_be_bytes());
                    }
                }
            }

            // --- Color combiner ---
            let combined = evaluate_combiner(
                ctx.combiner,
                ctx.cycle_type,
                &texel,
                &shade,
                &ctx.prim_color,
                &ctx.env_color,
            );

            // --- Blender ---
            let fb_off = base + ((py as usize) * cw as usize + (x as usize)) * bpp;
            let pixel = if ctx.force_blend && ctx.cycle_type < 2 {
                let mem = read_fb_rgba8(rdram, fb_off, bpp);
                if ctx.cycle_type == 1 {
                    // 2-cycle: run blender cycle 0 then cycle 1
                    let blend0 = evaluate_blender(
                        &combined, shade[3], &mem,
                        &ctx.blend_color, &ctx.fog_color,
                        ctx.blend_m1a_0, ctx.blend_m1b_0,
                        ctx.blend_m2a_0, ctx.blend_m2b_0,
                    );
                    evaluate_blender(
                        &blend0, shade[3], &mem,
                        &ctx.blend_color, &ctx.fog_color,
                        ctx.blend_m1a_1, ctx.blend_m1b_1,
                        ctx.blend_m2a_1, ctx.blend_m2b_1,
                    )
                } else {
                    evaluate_blender(
                        &combined, shade[3], &mem,
                        &ctx.blend_color, &ctx.fog_color,
                        ctx.blend_m1a_0, ctx.blend_m1b_0,
                        ctx.blend_m2a_0, ctx.blend_m2b_0,
                    )
                }
            } else {
                combined
            };

            // --- Write framebuffer ---
            if bpp == 2 && fb_off + 2 <= rdram.len() {
                let c16 = rgba8_to_5551(&pixel);
                rdram[fb_off..fb_off + 2].copy_from_slice(&c16.to_be_bytes());
                pix += 1;
            } else if bpp == 4 && fb_off + 4 <= rdram.len() {
                rdram[fb_off..fb_off + 4].copy_from_slice(&pixel);
                pix += 1;
            }
        }
    }

    (
        pix.saturating_mul(bpp as u64 * RDRAM_BUS_CYCLES_PER_BYTE),
        pix,
    )
}

// ---------------------------------------------------------------------------
// Flat fill rasterizer (fill_color only)
// ---------------------------------------------------------------------------

/// Rasterize a **flat** triangle (`fill_color` only) from hardware base words; ignores shade/tex/Z suffixes.
/// Returns `(RDRAM-style write cycles, pixels written)`.
///
/// When `clip` is `Some((xmin, xmax, ymin, ymax))`, spans are clipped to **inclusive** pixel bounds (same
/// convention as [`crate::rdp::Rdp::fill_rect`]).
pub fn raster_fill_hw(
    words: &[u32],
    fill_color: u32,
    color_image_addr: u32,
    color_width: u16,
    color_size: u8,
    clip: Option<(i32, i32, i32, i32)>,
    rdram: &mut [u8],
) -> (u64, u64) {
    if words.len() < 8 {
        return (0, 0);
    }
    let s = |w: i32, bits: u32| sign_extend(w, bits);

    let mut yh = s(words[1] as i32, 14);
    let ym = s((words[1] as i32) >> 16, 14);
    let yl = s(words[0] as i32, 14);
    yh &= !3;

    let xl = s(words[2] as i32, 28);
    let dxldy = s(words[3] as i32, 30);
    let xh = s(words[4] as i32, 28);
    let dxhdy = s(words[5] as i32, 30);
    let xm = s(words[6] as i32, 28);
    let dxmdy = s(words[7] as i32, 30);

    let xhf = fixed_to_f64(xh);
    let xmf = fixed_to_f64(xm);
    let xlf = fixed_to_f64(xl);

    let hk = slope_to_f64(dxhdy);
    let mk = slope_to_f64(dxmdy);
    let lk = slope_to_f64(dxldy);

    let yhf = yh as f64;
    let ymf = ym as f64;

    let hc = xhf - hk * yhf;
    let mc = xmf - mk * yhf;
    let lc = xlf - lk * ymf;

    let bpp = texel_bytes(color_size);
    let cw = color_width.max(1) as i32;
    let base = (color_image_addr & 0x00FF_FFFF) as usize;
    let px16 = fill_color as u16;

    let mut pix = 0u64;

    let py0 = (yh + 2) >> 2;
    let py1 = (yl + 2) >> 2;
    for py in py0..=py1 {
        let y_sub = ((py as i64) << 2) as f64;
        if y_sub < yhf || y_sub > yl as f64 {
            continue;
        }

        let xa = hk * y_sub + hc;
        let xb = if y_sub < ymf {
            mk * y_sub + mc
        } else {
            lk * y_sub + lc
        };
        let mut x0 = (xa.min(xb)).floor() as i32;
        let mut x1 = (xa.max(xb)).ceil() as i32;

        if let Some((sxa, sxb, sya, syb)) = clip {
            if py < sya || py > syb {
                continue;
            }
            x0 = x0.max(sxa);
            x1 = x1.min(sxb);
        }

        if x1 < x0 {
            continue;
        }
        if py < 0 {
            continue;
        }

        for x in x0..=x1 {
            if x < 0 {
                continue;
            }
            let off = base.saturating_add(((py as usize) * cw as usize + (x as usize)) * bpp);
            if bpp == 2 && off + 2 <= rdram.len() {
                rdram[off..off + 2].copy_from_slice(&px16.to_be_bytes());
                pix = pix.saturating_add(1);
            } else if bpp == 4 && off + 4 <= rdram.len() {
                rdram[off..off + 4].copy_from_slice(&fill_color.to_be_bytes());
                pix = pix.saturating_add(1);
            }
        }
    }

    (
        pix.saturating_mul(bpp as u64 * RDRAM_BUS_CYCLES_PER_BYTE),
        pix,
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rdp::TileSlot;

    #[test]
    fn assemble_fixed_positive() {
        // 3.5 in s15.16 = 0x00038000: hi = 0x0003, lo = 0x8000
        let v = assemble_fixed(0x0003, 0x8000);
        assert!((v - 3.5).abs() < 0.001);
    }

    #[test]
    fn assemble_fixed_negative() {
        // -1.0 in s15.16 = 0xFFFF0000: hi = 0xFFFF, lo = 0x0000
        let v = assemble_fixed(0xFFFF, 0x0000);
        assert!((v - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn assemble_fixed_small_negative() {
        // -0.5 in s15.16 = 0xFFFF8000: hi = 0xFFFF, lo = 0x8000
        let v = assemble_fixed(0xFFFF, 0x8000);
        assert!((v - (-0.5)).abs() < 0.001);
    }

    #[test]
    fn parse_coeffs_roundtrip() {
        // Build a 16-word block with known values:
        // Channel 0 (R): val=100.0, dx=1.0, de=0.5, dy=0.25
        let mut w = [0u32; 16];
        let encode = |v: f64| -> i32 { (v * 65536.0) as i32 };
        // val hi/lo
        let r_val = encode(100.0);
        w[0] = (w[0] & 0x0000FFFF) | ((r_val >> 16) as u32 & 0xFFFF) << 16;
        w[4] = (w[4] & 0x0000FFFF) | ((r_val & 0xFFFF) as u32) << 16;
        // dx hi/lo
        let r_dx = encode(1.0);
        w[2] = (w[2] & 0x0000FFFF) | ((r_dx >> 16) as u32 & 0xFFFF) << 16;
        w[6] = (w[6] & 0x0000FFFF) | ((r_dx & 0xFFFF) as u32) << 16;
        // de hi/lo
        let r_de = encode(0.5);
        w[8] = (w[8] & 0x0000FFFF) | ((r_de >> 16) as u32 & 0xFFFF) << 16;
        w[12] = (w[12] & 0x0000FFFF) | ((r_de & 0xFFFF) as u32) << 16;
        // dy hi/lo
        let r_dy = encode(0.25);
        w[10] = (w[10] & 0x0000FFFF) | ((r_dy >> 16) as u32 & 0xFFFF) << 16;
        w[14] = (w[14] & 0x0000FFFF) | ((r_dy & 0xFFFF) as u32) << 16;

        let ch = parse_4ch_coeffs(&w);
        assert!((ch[0].val - 100.0).abs() < 0.01);
        assert!((ch[0].dx - 1.0).abs() < 0.01);
        assert!((ch[0].de - 0.5).abs() < 0.01);
        assert!((ch[0].dy - 0.25).abs() < 0.01);
    }

    #[test]
    fn sample_tmem_rgba16() {
        let mut tmem = [0u8; TMEM_SIZE];
        // RGBA5551 at texel (1, 0): R=31, G=0, B=21, A=1
        let val: u16 = (31 << 11) | (0 << 6) | (21 << 1) | 1;
        tmem[2] = (val >> 8) as u8;
        tmem[3] = val as u8;
        let tile = TileSlot {
            fmt: 0, siz: 2, line_qwords: 1, tmem_qwords: 0, palette: 0,
            ..TileSlot::default()
        };
        let c = sample_tmem(&tmem, &tile, 1, 0, 0);
        assert_eq!(c[0], 255); // R: 31 → (31<<3)|(31>>2) = 255
        assert_eq!(c[1], 0);   // G: 0
        assert_eq!(c[3], 255); // A: 1 → 255
    }

    #[test]
    fn sample_tmem_ci4_with_palette() {
        let mut tmem = [0u8; TMEM_SIZE];
        // CI4 texel at (0,0): high nibble = index 3
        tmem[0] = 0x30;
        // Palette 2, index 3 → TLUT at 0x800 + (2*16 + 3)*2 = 0x800 + 70 = 0x846
        let tlut_val: u16 = 0xFFFF; // white, alpha=1
        tmem[0x846] = (tlut_val >> 8) as u8;
        tmem[0x847] = tlut_val as u8;
        let tile = TileSlot {
            fmt: 2, siz: 0, line_qwords: 1, tmem_qwords: 0, palette: 2,
            ..TileSlot::default()
        };
        let c = sample_tmem(&tmem, &tile, 0, 0, 0);
        assert_eq!(c, [255, 255, 255, 255]);
    }

    #[test]
    fn sample_tmem_ci8() {
        let mut tmem = [0u8; TMEM_SIZE];
        // CI8 texel at (5, 0): index = 0x0A
        tmem[5] = 0x0A;
        // TLUT entry 10 at 0x800 + 10*2 = 0x814
        let val: u16 = (16 << 11) | (8 << 6) | (4 << 1) | 1;
        tmem[0x814] = (val >> 8) as u8;
        tmem[0x815] = val as u8;
        let tile = TileSlot {
            fmt: 2, siz: 1, line_qwords: 1, tmem_qwords: 0, palette: 0,
            ..TileSlot::default()
        };
        let c = sample_tmem(&tmem, &tile, 5, 0, 0);
        let expected = rgba5551_to_rgba8(val);
        assert_eq!(c, expected);
    }

    #[test]
    fn sample_tmem_i8() {
        let mut tmem = [0u8; TMEM_SIZE];
        tmem[3] = 0x80;
        let tile = TileSlot {
            fmt: 4, siz: 1, line_qwords: 1, tmem_qwords: 0, palette: 0,
            ..TileSlot::default()
        };
        let c = sample_tmem(&tmem, &tile, 3, 0, 0);
        assert_eq!(c, [0x80, 0x80, 0x80, 255]);
    }

    #[test]
    fn sample_tmem_ia8() {
        let mut tmem = [0u8; TMEM_SIZE];
        tmem[0] = 0xA5; // I=0xA(10), A=0x5
        let tile = TileSlot {
            fmt: 3, siz: 1, line_qwords: 1, tmem_qwords: 0, palette: 0,
            ..TileSlot::default()
        };
        let c = sample_tmem(&tmem, &tile, 0, 0, 0);
        // I4=10 → 0xAA=170, A4=5 → 0x55=85
        assert_eq!(c[0], 0xAA);
        assert_eq!(c[3], 0x55);
    }

    #[test]
    fn parse_tri_coeffs_shade_only() {
        // Op 0x0C = shade (bit 2 set), 24 words total
        let mut words = [0u32; 24];
        words[0] = 0x0C << 24; // opcode
        // Set shade R initial to 200.0
        let r_raw = (200.0 * 65536.0) as i32;
        words[8] = ((r_raw >> 16) as u32 & 0xFFFF) << 16;
        words[12] = ((r_raw & 0xFFFF) as u32) << 16;
        let coeffs = parse_tri_coeffs(&words, 0x0C);
        assert!(coeffs.shade.is_some());
        assert!(coeffs.tex.is_none());
        assert!(coeffs.z.is_none());
        let sc = coeffs.shade.unwrap();
        assert!((sc[0].val - 200.0).abs() < 0.01);
    }

    #[test]
    fn parse_tri_coeffs_shade_tex_z() {
        // Op 0x0F = shade + tex + Z (bits 2+1+0), 44 words total
        let words = [0u32; 44];
        let coeffs = parse_tri_coeffs(&words, 0x0F);
        assert!(coeffs.shade.is_some());
        assert!(coeffs.tex.is_some());
        assert!(coeffs.z.is_some());
    }

    #[test]
    fn raster_shaded_flat_shade() {
        // Build a shaded triangle (op 0x0C) with flat shade color R=128, G=64, B=32, A=255
        // and combiner set to SHADE passthrough: a=0, b=0, c=0, d=SHADE(4)
        let mut words = [0u32; 24];
        // Opcode 0x0C, tile=0, yl=40 (10 pixels << 2)
        words[0] = (0x0C << 24) | 40;
        // yh=0, ym=20
        words[1] = (20 << 16) | 0;
        // xl = 10.0 in s11.16 → 10 * 65536 = 655360 = 0x000A0000
        words[2] = 0x000A_0000u32 as i32 as u32;
        words[3] = 0; // dxldy = 0
        // xh = 0.0
        words[4] = 0;
        words[5] = 0x0004_0000u32 as i32 as u32; // dxhdy = 4.0 * 65536 (slope right)
        // xm = 0.0
        words[6] = 0;
        words[7] = 0x0004_0000u32 as i32 as u32; // dxmdy = same

        // Shade coefficients at words[8..24]: R=128, G=64, B=32, A=255
        let encode = |v: f64| -> i32 { (v * 65536.0) as i32 };
        let set_ch = |w: &mut [u32], ch: usize, val: f64| {
            let raw = encode(val);
            let (w_hi, w_lo, upper) = match ch {
                0 => (0, 4, true),  // R: upper of w[0]/w[4]
                1 => (0, 4, false), // G: lower of w[0]/w[4]
                2 => (1, 5, true),  // B: upper of w[1]/w[5]
                3 => (1, 5, false), // A: lower of w[1]/w[5]
                _ => return,
            };
            let base = 8; // shade starts at word 8
            if upper {
                w[base + w_hi] = (w[base + w_hi] & 0x0000FFFF) | (((raw >> 16) as u32 & 0xFFFF) << 16);
                w[base + w_lo] = (w[base + w_lo] & 0x0000FFFF) | (((raw & 0xFFFF) as u32) << 16);
            } else {
                w[base + w_hi] = (w[base + w_hi] & 0xFFFF0000) | ((raw >> 16) as u32 & 0xFFFF);
                w[base + w_lo] = (w[base + w_lo] & 0xFFFF0000) | ((raw & 0xFFFF) as u32);
            }
        };
        set_ch(&mut words, 0, 128.0); // R
        set_ch(&mut words, 1, 64.0);  // G
        set_ch(&mut words, 2, 32.0);  // B
        set_ch(&mut words, 3, 255.0); // A

        let coeffs = parse_tri_coeffs(&words, 0x0C);
        assert!(coeffs.shade.is_some());
        let sc = coeffs.shade.as_ref().unwrap();
        assert!((sc[0].val - 128.0).abs() < 0.01);

        // Set up combiner for SHADE passthrough: d_rgb=4(SHADE), d_alpha=4(SHADE_A)
        let combiner = CombinerMux {
            cycle: [
                crate::rdp_combiner::CombinerCycle {
                    a_rgb: 8, b_rgb: 8, c_rgb: 16, d_rgb: 4, // d=SHADE
                    a_alpha: 7, b_alpha: 7, c_alpha: 7, d_alpha: 4, // d=SHADE_A
                },
                Default::default(),
            ],
        };

        let tiles = [TileSlot::default(); 8];
        let tmem = [0u8; TMEM_SIZE];
        let ctx = RasterCtx {
            tmem: &tmem,
            tiles: &tiles,
            combiner: &combiner,
            cycle_type: 0, // 1-cycle
            prim_color: [0; 4],
            env_color: [0; 4],
            blend_color: [0; 4],
            fog_color: [0; 4],
            fill_color: 0,
            ci_addr: 0x10_0000,
            ci_width: 320,
            ci_size: 2, // RGBA16
            z_image_addr: 0,
            z_compare_en: false,
            z_update_en: false,
            z_source_sel: false,
            prim_z: 0,
            tlut_type: 0,
            force_blend: false,
            blend_m1a_0: 0, blend_m1b_0: 0, blend_m2a_0: 0, blend_m2b_0: 0,
            blend_m1a_1: 0, blend_m1b_1: 0, blend_m2a_1: 0, blend_m2b_1: 0,
            clip: None,
        };

        let mut rdram = vec![0u8; 0x20_0000];
        let (cycles, pixels) = raster_shaded_hw(&words, &coeffs, &ctx, &mut rdram);
        assert!(pixels > 0, "should have drawn some pixels");
        assert!(cycles > 0);
    }

    #[test]
    fn z_buffer_rejects_behind() {
        // Two triangles at the same position: first with Z=100, second with Z=200.
        // With z_compare + z_update, the second triangle (farther) should be rejected.
        let mut rdram = vec![0u8; 0x20_0000];
        let z_image = 0x18_0000u32;

        // Pre-fill Z-buffer with Z=100 at one pixel position
        let z_off = z_image as usize + (5 * 320 + 5) * 2;
        rdram[z_off..z_off + 2].copy_from_slice(&100u16.to_be_bytes());

        // A 1-pixel "triangle" covering (5,5): set Z=200 (behind Z=100)
        let combiner = CombinerMux::default();
        let tiles = [TileSlot::default(); 8];
        let tmem = [0u8; TMEM_SIZE];
        let z_coeffs = ChannelCoeffs { val: 200.0, dx: 0.0, de: 0.0, dy: 0.0 };
        let coeffs = TriCoeffs {
            shade: Some([
                ChannelCoeffs { val: 255.0, ..Default::default() },
                ChannelCoeffs { val: 0.0, ..Default::default() },
                ChannelCoeffs { val: 0.0, ..Default::default() },
                ChannelCoeffs { val: 255.0, ..Default::default() },
            ]),
            tex: None,
            z: Some(z_coeffs),
        };

        // Tiny triangle around (5,5)
        let mut words = [0u32; 12]; // shade+Z = 24+16=48 bytes? No, 0x0D = shade+Z = 32+64+16 = 28 words
        // Actually we just need base geometry (8 words). Coeffs are passed separately.
        words[0] = (0x0D << 24) | 24; // yl = 24 (6 pixels)
        words[1] = (20 << 16) | 16; // ym = 20, yh = 16
        // xl = 6 * 65536
        words[2] = 6 * 65536;
        words[4] = 4 * 65536; // xh = 4
        words[6] = 4 * 65536; // xm = 4

        let ctx = RasterCtx {
            tmem: &tmem, tiles: &tiles, combiner: &combiner,
            cycle_type: 0, prim_color: [0;4], env_color: [0;4],
            blend_color: [0; 4], fog_color: [0; 4],
            fill_color: 0, ci_addr: 0x10_0000, ci_width: 320, ci_size: 2,
            z_image_addr: z_image, z_compare_en: true, z_update_en: false,
            z_source_sel: false, prim_z: 0, tlut_type: 0,
            force_blend: false,
            blend_m1a_0: 0, blend_m1b_0: 0, blend_m2a_0: 0, blend_m2b_0: 0,
            blend_m1a_1: 0, blend_m1b_1: 0, blend_m2a_1: 0, blend_m2b_1: 0,
            clip: None,
        };

        let fb_off = 0x10_0000 + (5 * 320 + 5) * 2;
        rdram[fb_off] = 0;
        rdram[fb_off + 1] = 0;

        raster_shaded_hw(&words, &coeffs, &ctx, &mut rdram);
        // Pixel at (5,5) should NOT have been written (Z=200 > stored Z=100)
        assert_eq!(rdram[fb_off], 0);
        assert_eq!(rdram[fb_off + 1], 0);
    }
}
