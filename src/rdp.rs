//! Reality Display Processor: display list parsing, minimal raster paths, and timing hooks.
//!
//! Full hardware includes triangle rasterization, TMEM, texture filter, and the color combiner
//! pipeline. This module implements **command decode**, **RDRAM-backed SetColorImage + FillRectangle**
//! (RGBA16), **TextureRectangle** (nearest-neighbor from TMEM), **texture image / tile / load**
//! scaffolding for TMEM, **combiner register** storage, and **cycle estimates** for list DMA and
//! framebuffer writes. Triangles are recognized but not rasterized yet.

use crate::rcp::DPC_STATUS_XBUS_DMEM;
use crate::timing::RDRAM_BUS_CYCLES_PER_BYTE;

/// 4 KiB TMEM ([n64brew](https://n64brew.dev/wiki/Reality_Display_Processor/Texture_Memory)).
pub const TMEM_SIZE: usize = 4096;

// --- RDP command opcodes: high byte of word 0 (big-endian), matching libultra `gsDP*` macros. ---
pub const OP_SET_COLOR_IMAGE: u8 = 0xFF;
pub const OP_SET_Z_IMAGE: u8 = 0xFE;
pub const OP_SET_TEXTURE_IMAGE: u8 = 0xFD;
pub const OP_SET_COMBINE: u8 = 0xFC;
pub const OP_SET_ENV_COLOR: u8 = 0xFB;
pub const OP_SET_PRIM_COLOR: u8 = 0xFA;
pub const OP_SET_BLEND_COLOR: u8 = 0xF9;
pub const OP_SET_FOG_COLOR: u8 = 0xF8;
pub const OP_SET_FILL_COLOR: u8 = 0xF7;
pub const OP_FILL_RECT: u8 = 0xF6;
pub const OP_SET_TILE: u8 = 0xF5;
pub const OP_LOAD_TILE: u8 = 0xF4;
pub const OP_LOAD_BLOCK: u8 = 0xF3;
pub const OP_SET_TILE_SIZE: u8 = 0xF2;
pub const OP_SET_SCISSOR: u8 = 0xED;
pub const OP_SYNC_FULL: u8 = 0xE9;
pub const OP_SYNC_PIPE: u8 = 0xE8;
pub const OP_SYNC_TILE: u8 = 0xE7;
pub const OP_TEXRECT: u8 = 0xE4;
pub const OP_TEXRECT_FLIP: u8 = 0xE5;

/// First byte of triangle commands (shaded / textured / Z variants).
pub const TRI_CMD_MIN: u8 = 0x08;
pub const TRI_CMD_MAX: u8 = 0x0F;
/// Stub: real triangle packets are longer and vary by type; skip this many bytes so the list stays aligned.
pub const TRIANGLE_CMD_STUB_BYTES: usize = 32;

/// Estimated RCP master cycles to decode and dispatch one 64-bit RDP command.
pub const RDP_CYCLES_PER_CMD: u64 = 8;
/// RDRAM access cost used when charging DP list DMA and pixel writes ([`crate::timing::RDRAM_BUS_CYCLES_PER_BYTE`]).
pub const RDRAM_CYCLES_PER_BYTE: u64 = RDRAM_BUS_CYCLES_PER_BYTE;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColorImage {
    pub fmt: u8,
    pub size: u8,
    /// Line width in pixels (from `SetColorImage`; 0 treated as 320).
    pub width: u16,
    /// RDRAM byte address (`& 0x00FF_FFFF`).
    pub addr: u32,
}

impl Default for ColorImage {
    fn default() -> Self {
        Self {
            fmt: 0,
            size: 2,
            width: 320,
            addr: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TextureImage {
    pub fmt: u8,
    pub size: u8,
    pub width: u16,
    pub addr: u32,
}

/// Raw `SetCombine` mux lines — full combiner evaluation is not implemented; kept for future use.
#[derive(Clone, Copy, Debug, Default)]
pub struct CombinerState {
    pub mux0: u64,
    pub mux1: u64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Scissor {
    pub xh: u16,
    pub yh: u16,
    pub xl: u16,
    pub yl: u16,
    pub enabled: bool,
}

/// RDP processor state updated by the display list.
#[derive(Debug)]
pub struct Rdp {
    pub color_image: ColorImage,
    pub texture_image: TextureImage,
    pub combine: CombinerState,
    pub fill_color: u32,
    pub scissor: Scissor,
    pub tmem: Box<[u8; TMEM_SIZE]>,
    /// Commands that were triangles (not rasterized).
    pub triangles_skipped: u64,
    /// Texels drawn by the last TextureRectangle (debug / tuning).
    pub texrect_texels: u64,
    /// Other opcodes not handled in software yet.
    pub other_unhandled: u64,
    /// Cycles accumulated for the last `process_display_list` (also returned).
    pub last_list_cycles: u64,
}

impl Default for Rdp {
    fn default() -> Self {
        Self::new()
    }
}

impl Rdp {
    pub fn new() -> Self {
        Self {
            color_image: ColorImage::default(),
            texture_image: TextureImage::default(),
            combine: CombinerState::default(),
            fill_color: 0,
            scissor: Scissor::default(),
            tmem: Box::new([0u8; TMEM_SIZE]),
            triangles_skipped: 0,
            texrect_texels: 0,
            other_unhandled: 0,
            last_list_cycles: 0,
        }
    }

    /// Run the display list in `[start, end)` (physical byte addresses, 8-byte aligned).
    /// `dpc_status` supplies [`DPC_STATUS_XBUS_DMEM`] for command source (RDRAM vs RSP DMEM+IMEM).
    pub fn process_display_list(
        &mut self,
        rdram: &mut [u8],
        rsp_dmem: &[u8],
        rsp_imem: &[u8],
        start: u32,
        end: u32,
        dpc_status: u32,
    ) -> u64 {
        self.last_list_cycles = 0;
        self.texrect_texels = 0;
        let from_xbus = (dpc_status & DPC_STATUS_XBUS_DMEM) != 0;
        let s = (start & 0x00FF_FFF8) as usize;
        let e = (end & 0x00FF_FFF8) as usize;
        if e <= s {
            return 0;
        }
        let len = e - s;
        let mut cycles = len as u64 * RDRAM_CYCLES_PER_BYTE;

        let mut off = 0usize;
        while off + 8 <= len {
            let read_u32 = |rel_off: usize| {
                let abs = s + rel_off;
                if from_xbus {
                    rsp_read_be_u32(rsp_dmem, rsp_imem, abs)
                } else {
                    read_be_u32_rdram(rdram, abs)
                }
            };
            let w0 = read_u32(off);
            let w1 = read_u32(off + 4);
            let op = (w0 >> 24) as u8;
            cycles = cycles.saturating_add(RDP_CYCLES_PER_CMD);
            let advance = if (TRI_CMD_MIN..=TRI_CMD_MAX).contains(&op) {
                self.triangles_skipped = self.triangles_skipped.saturating_add(1);
                TRIANGLE_CMD_STUB_BYTES
            } else if (op == OP_TEXRECT || op == OP_TEXRECT_FLIP) && off + 16 <= len {
                let w2 = read_u32(off + 8);
                let w3 = read_u32(off + 12);
                let flip = op == OP_TEXRECT_FLIP;
                cycles = cycles.saturating_add(self.texture_rectangle(rdram, w0, w1, w2, w3, flip));
                16
            } else {
                match op {
                    OP_SET_COLOR_IMAGE => self.set_color_image(w0, w1),
                    OP_SET_TEXTURE_IMAGE => self.set_texture_image(w0, w1),
                    OP_SET_COMBINE => self.set_combine(w0, w1),
                    OP_SET_FILL_COLOR => self.fill_color = w0 & 0xFFFF_FFFF,
                    OP_FILL_RECT => {
                        cycles = cycles.saturating_add(self.fill_rect(rdram, w0, w1));
                    }
                    OP_SET_SCISSOR => self.set_scissor(w0, w1),
                    OP_LOAD_TILE => {
                        cycles = cycles.saturating_add(self.load_tile(rdram, w0, w1));
                    }
                    OP_LOAD_BLOCK => {
                        cycles = cycles.saturating_add(self.load_block(rdram, w0, w1));
                    }
                    OP_SYNC_FULL | OP_SYNC_PIPE | OP_SYNC_TILE => {}
                    OP_SET_TILE | OP_SET_TILE_SIZE => {}
                    OP_TEXRECT | OP_TEXRECT_FLIP => {
                        self.other_unhandled = self.other_unhandled.saturating_add(1);
                    }
                    _ => {
                        self.other_unhandled = self.other_unhandled.saturating_add(1);
                    }
                }
                8
            };
            off = off.saturating_add(advance).min(len);
        }

        self.last_list_cycles = cycles;
        cycles
    }

    /// Lower-bound RCP cycles before the list can finish (used to schedule deferred `DPC_END` completion).
    pub fn estimate_display_list_cycles(start: u32, end: u32) -> u64 {
        let s = (start & 0x00FF_FFF8) as u64;
        let e = (end & 0x00FF_FFF8) as u64;
        if e <= s {
            return 40;
        }
        let len = e - s;
        let cmds = (len / 8).max(1);
        len
            .saturating_mul(RDRAM_CYCLES_PER_BYTE)
            .saturating_add(cmds.saturating_mul(RDP_CYCLES_PER_CMD))
            .max(1)
    }
}

#[inline]
fn read_be_u32_rdram(mem: &[u8], i: usize) -> u32 {
    if i + 4 > mem.len() {
        return 0;
    }
    u32::from_be_bytes(mem[i..i + 4].try_into().unwrap())
}

#[inline]
fn rsp_read_u8(dmem: &[u8], imem: &[u8], flat: usize) -> u8 {
    let i = flat & 0x1FFF;
    if i < 0x1000 {
        dmem.get(i).copied().unwrap_or(0)
    } else {
        imem.get(i - 0x1000).copied().unwrap_or(0)
    }
}

fn rsp_read_be_u32(dmem: &[u8], imem: &[u8], flat: usize) -> u32 {
    let b = [
        rsp_read_u8(dmem, imem, flat),
        rsp_read_u8(dmem, imem, flat + 1),
        rsp_read_u8(dmem, imem, flat + 2),
        rsp_read_u8(dmem, imem, flat + 3),
    ];
    u32::from_be_bytes(b)
}

impl Rdp {
    fn set_color_image(&mut self, w0: u32, w1: u32) {
        let fmt = ((w0 >> 21) & 7) as u8;
        let size = ((w0 >> 19) & 3) as u8;
        let width = (w0 & 0xFFF) as u16;
        self.color_image = ColorImage {
            fmt,
            size,
            width,
            addr: w1 & 0x00FF_FFFF,
        };
    }

    fn set_texture_image(&mut self, w0: u32, w1: u32) {
        let fmt = ((w0 >> 21) & 7) as u8;
        let size = ((w0 >> 19) & 3) as u8;
        let width = (w0 & 0xFFF) as u16;
        self.texture_image = TextureImage {
            fmt,
            size,
            width,
            addr: w1 & 0x00FF_FFFF,
        };
    }

    fn set_combine(&mut self, w0: u32, w1: u32) {
        let hi = (w0 as u64) & 0x00FF_FFFF;
        let lo = w1 as u64;
        self.combine.mux0 = (hi << 32) | lo;
        // Many games pack both halves in two commands; second half often follows — single 64-bit stores low line only for now.
        self.combine.mux1 = 0;
    }

    fn set_scissor(&mut self, w0: u32, w1: u32) {
        // `gsDPSetScissor`: xh/yh in w0 bits, xl/yl in w1 (simplified 12-bit fields).
        self.scissor.xh = ((w0 >> 12) & 0xFFF) as u16;
        self.scissor.yh = (w0 & 0xFFF) as u16;
        self.scissor.xl = ((w1 >> 12) & 0xFFF) as u16;
        self.scissor.yl = (w1 & 0xFFF) as u16;
        self.scissor.enabled = true;
    }

    /// Copy a tile from RDRAM texture memory into TMEM (stride / format handling is incomplete).
    fn load_tile(&mut self, rdram: &[u8], w0: u32, w1: u32) -> u64 {
        let sl = ((w0 >> 12) & 0xFFF) as usize;
        let tl = (w0 & 0xFFF) as usize;
        let sh = ((w1 >> 12) & 0xFFF) as usize;
        let th = (w1 & 0xFFF) as usize;
        let addr = self.texture_image.addr as usize;
        let width = self.texture_image.width.max(1) as usize;
        let bpp = texel_bytes(self.texture_image.size);
        let mut c = 0u64;
        let mut dst = 0usize;
        for ty in tl..=th {
            for tx in sl..=sh {
                for b in 0..bpp {
                    let rd_i = addr.saturating_add(ty * width * bpp + tx * bpp + b);
                    if rd_i < rdram.len() && dst < TMEM_SIZE {
                        self.tmem[dst] = rdram[rd_i];
                        dst += 1;
                    }
                    c = c.saturating_add(RDRAM_CYCLES_PER_BYTE);
                }
            }
        }
        c
    }

    /// `LoadBlock`: fast linear copy into TMEM (used for texture loads).
    fn load_block(&mut self, rdram: &[u8], w0: u32, w1: u32) -> u64 {
        let tmem_offset = (w0 & 0xFFF) as usize;
        let rdram_offset = (w1 & 0x00FF_FFFF) as usize;
        let words = (((w0 >> 12) & 0xFFF) as usize).saturating_add(1);
        let mut c = 0u64;
        for i in 0..words.min(TMEM_SIZE / 4) {
            let ri = rdram_offset.saturating_add(i * 4);
            let ti = (tmem_offset + i * 4) & (TMEM_SIZE - 4);
            if ri + 4 <= rdram.len() {
                self.tmem[ti..ti + 4].copy_from_slice(&rdram[ri..ri + 4]);
            }
            c = c.saturating_add(4 * RDRAM_CYCLES_PER_BYTE);
        }
        c
    }

    fn fill_rect(&mut self, rdram: &mut [u8], w0: u32, w1: u32) -> u64 {
        let lrx = ((w0 >> 12) & 0xFFF) as i32;
        let lry = (w0 & 0xFFF) as i32;
        let ulx = ((w1 >> 12) & 0xFFF) as i32;
        let uly = (w1 & 0xFFF) as i32;
        // 10.2 fixed-point → pixel indices (same convention as common HLE cores).
        let x0 = ulx >> 2;
        let y0 = uly >> 2;
        let x1 = lrx >> 2;
        let y1 = lry >> 2;
        let (x0, x1) = if x0 <= x1 { (x0, x1) } else { (x1, x0) };
        let (y0, y1) = if y0 <= y1 { (y0, y1) } else { (y1, y0) };

        let mut x_start = x0;
        let mut x_end = x1.saturating_add(1);
        let mut y_start = y0;
        let mut y_end = y1.saturating_add(1);
        if self.scissor.enabled {
            let sx0 = (self.scissor.xh as i32) >> 2;
            let sy0 = (self.scissor.yh as i32) >> 2;
            let sx1 = (self.scissor.xl as i32) >> 2;
            let sy1 = (self.scissor.yl as i32) >> 2;
            x_start = x_start.max(sx0.min(sx1));
            y_start = y_start.max(sy0.min(sy1));
            x_end = x_end.min(sx1.max(sx0).saturating_add(1));
            y_end = y_end.min(sy1.max(sy0).saturating_add(1));
        }

        let cw = self.color_image.width.max(1) as i32;
        let bpp = texel_bytes(self.color_image.size);
        let base = (self.color_image.addr & 0x00FF_FFFF) as usize;
        let px = self.fill_color as u16;
        let mut pix = 0u64;
        for y in y_start..y_end {
            for x in x_start..x_end {
                if x < 0 || y < 0 {
                    continue;
                }
                let x = x as usize;
                let y = y as usize;
                let off = base.saturating_add((y * cw as usize + x) * bpp);
                if bpp == 2 && off + 2 <= rdram.len() {
                    rdram[off..off + 2].copy_from_slice(&px.to_be_bytes());
                } else if bpp == 4 && off + 4 <= rdram.len() {
                    let c = self.fill_color;
                    rdram[off..off + 4].copy_from_slice(&c.to_be_bytes());
                }
                pix = pix.saturating_add(1);
            }
        }
        pix.saturating_mul(bpp as u64 * RDRAM_CYCLES_PER_BYTE)
    }

    /// `TextureRectangle` / `TextureRectangleFlip`: two 64-bit words (screen + tile), then s/t/dsdx/dtdy.
    /// Nearest-neighbor sample from TMEM; derivatives are in the same fixed-point scale as hardware s/t (s10.5).
    fn texture_rectangle(
        &mut self,
        rdram: &mut [u8],
        w0: u32,
        w1: u32,
        w2: u32,
        w3: u32,
        flip: bool,
    ) -> u64 {
        let xh_fp = ((w0 >> 12) & 0xFFF) as i32;
        let yh_fp = (w0 & 0xFFF) as i32;
        let xl_fp = ((w1 >> 12) & 0xFFF) as i32;
        let yl_fp = (w1 & 0xFFF) as i32;
        let px0 = (xh_fp >> 2).min(xl_fp >> 2);
        let px1 = (xh_fp >> 2).max(xl_fp >> 2);
        let py0 = (yh_fp >> 2).min(yl_fp >> 2);
        let py1 = (yh_fp >> 2).max(yl_fp >> 2);

        let mut x_start = px0;
        let mut x_end = px1.saturating_add(1);
        let mut y_start = py0;
        let mut y_end = py1.saturating_add(1);

        if self.scissor.enabled {
            let sx0 = (self.scissor.xh as i32) >> 2;
            let sy0 = (self.scissor.yh as i32) >> 2;
            let sx1 = (self.scissor.xl as i32) >> 2;
            let sy1 = (self.scissor.yl as i32) >> 2;
            let sxa = sx0.min(sx1);
            let sxb = sx0.max(sx1).saturating_add(1);
            let sya = sy0.min(sy1);
            let syb = sy0.max(sy1).saturating_add(1);
            x_start = x_start.max(sxa);
            y_start = y_start.max(sya);
            x_end = x_end.min(sxb);
            y_end = y_end.min(syb);
        }

        let tile = ((w1 >> 24) & 7) as usize;
        let tmem_base = tile.saturating_mul(512).min(TMEM_SIZE.saturating_sub(64));

        let s0 = ((w2 >> 16) as i16) as i64;
        let t0 = ((w2 & 0xFFFF) as i16) as i64;
        let dsdx = ((w3 >> 16) as i16) as i64;
        let dtdy = ((w3 & 0xFFFF) as i16) as i64;

        let tw = self.texture_image.width.max(1) as usize;
        let bpp_tex = texel_bytes(self.texture_image.size);
        let bpp_fb = texel_bytes(self.color_image.size);
        let cw = self.color_image.width.max(1) as i32;
        let base = (self.color_image.addr & 0x00FF_FFFF) as usize;

        let mut pix = 0u64;
        for y in y_start..y_end {
            for x in x_start..x_end {
                if x < 0 || y < 0 {
                    continue;
                }
                let xu = x as usize;
                let yu = y as usize;
                let (s_acc, t_acc) = if !flip {
                    let s_acc = s0 + (x as i64 - px0 as i64) * dsdx;
                    let t_acc = t0 + (y as i64 - py0 as i64) * dtdy;
                    (s_acc, t_acc)
                } else {
                    // Transpose mapping (approximation for sprite flips; refine against hardware later).
                    let s_acc = s0 + (y as i64 - py0 as i64) * dsdx;
                    let t_acc = t0 + (x as i64 - px0 as i64) * dtdy;
                    (s_acc, t_acc)
                };
                let ts = (s_acc >> 5).max(0) as usize;
                let tt = (t_acc >> 5).max(0) as usize;
                let off_tm = tmem_base.saturating_add((tt * tw + ts).saturating_mul(bpp_tex));
                if off_tm + bpp_tex > TMEM_SIZE {
                    continue;
                }
                let texel16 = if bpp_tex >= 2 {
                    u16::from_be_bytes([self.tmem[off_tm], self.tmem[off_tm + 1]])
                } else {
                    continue;
                };

                let off_fb = base.saturating_add((yu * cw as usize + xu) * bpp_fb);
                if bpp_fb == 2 && off_fb + 2 <= rdram.len() {
                    rdram[off_fb..off_fb + 2].copy_from_slice(&texel16.to_be_bytes());
                } else if bpp_fb == 4 && off_fb + 4 <= rdram.len() {
                    let w = (texel16 as u32) << 16 | texel16 as u32;
                    rdram[off_fb..off_fb + 4].copy_from_slice(&w.to_be_bytes());
                }
                pix = pix.saturating_add(1);
            }
        }
        self.texrect_texels = self.texrect_texels.saturating_add(pix);
        pix.saturating_mul(bpp_fb as u64 * RDRAM_CYCLES_PER_BYTE)
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fill_rect_rgba16_writes_pixels() {
        let mut rdp = Rdp::new();
        let mut rdram = vec![0u8; 0x20_0000];
        rdp.color_image = ColorImage {
            fmt: 0,
            size: 2,
            width: 320,
            addr: 0x10_0000,
        };
        rdp.fill_color = 0xFFFF_FFFF;
        // ul=(4,4) lr=(8,8) in 10.2 → pixels 1..2 in x/y after >>2
        let w0 = ((OP_FILL_RECT as u32) << 24) | ((8 & 0xFFF) << 12) | (8 & 0xFFF);
        let w1 = ((4 & 0xFFF) << 12) | (4 & 0xFFF);
        rdp.fill_rect(&mut rdram, w0, w1);
        let base = 0x10_0000usize;
        let w = 320usize;
        let o = base + (1 * w + 1) * 2;
        assert_eq!(u16::from_be_bytes([rdram[o], rdram[o + 1]]), 0xFFFF);
    }

    #[test]
    fn texture_rectangle_nearest_two_texels() {
        let mut rdp = Rdp::new();
        let mut rdram = vec![0u8; 0x20_0000];
        rdp.color_image = ColorImage {
            fmt: 0,
            size: 2,
            width: 320,
            addr: 0x10_0000,
        };
        rdp.texture_image = TextureImage {
            fmt: 0,
            size: 2,
            width: 4,
            addr: 0,
        };
        rdp.tmem[0..2].copy_from_slice(&0xABCDu16.to_be_bytes());
        rdp.tmem[2..4].copy_from_slice(&0x1234u16.to_be_bytes());

        let list_base = 0x1000usize;
        // xh,yh = 0; xl,yl = 4,0 → 10.2 corners map to pixels x=0 and x=1 (two columns).
        let w0 = ((OP_TEXRECT as u32) << 24) | ((0 & 0xFFF) << 12) | (0 & 0xFFF);
        let w1 = ((0u32 & 7) << 24) | ((4 & 0xFFF) << 12) | (0 & 0xFFF);
        let w2 = 0u32;
        let w3 = (32i16 as u16 as u32) << 16 | 32i16 as u16 as u32;
        for (i, w) in [w0, w1, w2, w3].iter().enumerate() {
            rdram[list_base + i * 4..list_base + i * 4 + 4].copy_from_slice(&w.to_be_bytes());
        }

        let c = rdp.process_display_list(&mut rdram, &[], &[], list_base as u32, (list_base + 16) as u32, 0);
        assert!(c > 0);
        assert_eq!(rdp.texrect_texels, 2);

        let base = 0x10_0000usize;
        let w = 320usize;
        let o0 = base + (0 * w + 0) * 2;
        let o1 = base + (0 * w + 1) * 2;
        assert_eq!(u16::from_be_bytes([rdram[o0], rdram[o0 + 1]]), 0xABCD);
        assert_eq!(u16::from_be_bytes([rdram[o1], rdram[o1 + 1]]), 0x1234);
    }
}
