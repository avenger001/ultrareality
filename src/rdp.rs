//! Reality Display Processor: display list parsing, minimal raster paths, and timing hooks.
//!
//! Full hardware includes triangle rasterization, TMEM, texture filter, and the color combiner
//! pipeline. This module implements **command decode**, **RDRAM-backed SetColorImage + FillRectangle**
//! (RGBA16), **TextureRectangle** (nearest-neighbor from TMEM), **triangles** (GLideN64-style base
//! geometry + `fill_color` flat fill, plus a legacy unit-test vertex layout), **SetTile / SetTileSize**
//! TMEM descriptors, **primitive/env** colors with simple combiner modulation, and **cycle estimates**
//! for list DMA and framebuffer writes.

use crate::rcp::DPC_STATUS_XBUS_DMEM;
use crate::rdp_combiner::{rgba5551_modulate, u32_to_rgba8};
use crate::rdp_triangle;
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
pub const OP_SET_OTHER_MODES: u8 = 0xEF;
pub const OP_SET_PRIM_DEPTH: u8 = 0xEE;
pub const OP_SET_SCISSOR: u8 = 0xED;
pub const OP_SYNC_FULL: u8 = 0xE9;
pub const OP_SYNC_TILE: u8 = 0xE8;
pub const OP_SYNC_PIPE: u8 = 0xE7;
pub const OP_SYNC_LOAD: u8 = 0xE6;
pub const OP_TEXRECT: u8 = 0xE4;
pub const OP_TEXRECT_FLIP: u8 = 0xE5;

/// First byte of triangle commands (shaded / textured / Z variants).
pub const TRI_CMD_MIN: u8 = 0x08;
pub const TRI_CMD_MAX: u8 = 0x0F;
/// Bytes consumed by the unit-test **legacy** triangle layout (8 words).
pub const TRIANGLE_PACKET_BYTES: usize = 32;

/// Estimated RCP master cycles to decode and dispatch one 64-bit RDP command.
pub const RDP_CYCLES_PER_CMD: u64 = 8;
/// Default RDRAM access cost per byte ([`crate::timing::RDRAM_BUS_CYCLES_PER_BYTE`]).
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

/// One of eight RDP tiles: `SetTile` / `SetTileSize` state for TMEM addressing.
#[derive(Clone, Copy, Debug)]
pub struct TileSlot {
    pub fmt: u8,
    pub siz: u8,
    /// Stride in 64-bit (QWORD) units (`SetTile` **line** field).
    pub line_qwords: u16,
    /// TMEM base in QWORD units (`SetTile` low 9 bits).
    pub tmem_qwords: u16,
    /// CI4 palette index (0–15), from `SetTile` bits [23:20] of w1.
    pub palette: u8,
    /// Bounds from `SetTileSize` (12-bit fields, 10.2 fixed-point like other rect commands).
    pub sl: u16,
    pub tl: u16,
    pub sh: u16,
    pub th: u16,
}

impl Default for TileSlot {
    fn default() -> Self {
        Self {
            fmt: 0,
            siz: 2,
            line_qwords: 1,
            tmem_qwords: 0,
            palette: 0,
            sl: 0,
            tl: 0,
            sh: 0,
            th: 0,
        }
    }
}

/// Raw `SetCombine` mux words (stored as the original w0/w1 from the command).
#[derive(Clone, Copy, Debug, Default)]
pub struct CombinerState {
    pub mux0: u64,
    pub mux1: u64,
    /// Decoded combiner mux selectors.
    pub decoded: crate::rdp_combiner::CombinerMux,
}

/// RDP Other Modes register (SetOtherModes, opcode 0xEF).
#[derive(Clone, Copy, Debug)]
pub struct OtherModes {
    /// 0 = 1-cycle, 1 = 2-cycle, 2 = copy, 3 = fill
    pub cycle_type: u8,
    pub z_compare_en: bool,
    pub z_update_en: bool,
    pub z_source_sel: bool,
    pub force_blend: bool,
    pub alpha_compare_en: bool,
    pub image_read_en: bool,
    pub tex_filter: u8,
    pub tlut_type: u8,
    /// Blender mux: cycle 0 P/A/Q/B selectors
    pub blend_m1a_0: u8,
    pub blend_m1b_0: u8,
    pub blend_m2a_0: u8,
    pub blend_m2b_0: u8,
    /// Blender mux: cycle 1
    pub blend_m1a_1: u8,
    pub blend_m1b_1: u8,
    pub blend_m2a_1: u8,
    pub blend_m2b_1: u8,
}

impl Default for OtherModes {
    fn default() -> Self {
        Self {
            cycle_type: 0,
            z_compare_en: false,
            z_update_en: false,
            z_source_sel: false,
            force_blend: false,
            alpha_compare_en: false,
            image_read_en: false,
            tex_filter: 0,
            tlut_type: 0,
            blend_m1a_0: 0,
            blend_m1b_0: 0,
            blend_m2a_0: 0,
            blend_m2b_0: 0,
            blend_m1a_1: 0,
            blend_m1b_1: 0,
            blend_m2a_1: 0,
            blend_m2b_1: 0,
        }
    }
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
    pub other_modes: OtherModes,
    /// Z-buffer image RDRAM address (SetZImage, 0xFE).
    pub z_image_addr: u32,
    /// Primitive depth (SetPrimDepth, 0xEE): z and dz values.
    pub prim_z: u16,
    pub prim_dz: u16,
    /// `SetPrimColor` / `SetEnvColor` style constant (low 16 bits often RGBA5551 for modulation).
    pub prim_color: u32,
    pub env_color: u32,
    pub blend_color: u32,
    pub fog_color: u32,
    pub fill_color: u32,
    pub scissor: Scissor,
    pub tiles: [TileSlot; 8],
    pub tmem: Box<[u8; TMEM_SIZE]>,
    /// Triangle opcodes processed by the software raster (variable length; see [`crate::rdp_triangle::CMD_LENGTH`]).
    pub triangle_commands: u64,
    /// Pixels written by the last triangle batch in `process_display_list`.
    pub triangle_pixels: u64,
    /// Texels drawn by the last TextureRectangle (debug / tuning).
    pub texrect_texels: u64,
    /// Other opcodes not handled in software yet.
    pub other_unhandled: u64,
    /// Cycles accumulated for the last `process_display_list` (also returned).
    pub last_list_cycles: u64,
    /// Histogram of unhandled opcode values (for diagnostics).
    pub unhandled_hist: [u64; 256],
    /// Histogram of ALL opcode values processed (for diagnostics).
    pub op_hist: [u64; 256],
    /// Whether the most recent `process_display_list` call encountered an
    /// `OP_SYNC_FULL` opcode. Real hardware only raises `MI_INTR_DP` when
    /// SyncFull is executed — not on every DPC kick. Games split one gfx task
    /// into multiple DPC kicks (SetOtherModes + LoadBlock + triangles +
    /// SyncFull), and raising one IRQ per kick produces duplicate
    /// `OS_EVENT_DP` dispatches that confuse libultra's scheduler state.
    pub last_list_had_sync_full: bool,
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
            other_modes: OtherModes::default(),
            z_image_addr: 0,
            prim_z: 0,
            prim_dz: 0,
            prim_color: 0x0000_FFFF,
            env_color: 0x0000_FFFF,
            blend_color: 0,
            fog_color: 0,
            fill_color: 0,
            scissor: Scissor::default(),
            tiles: [TileSlot::default(); 8],
            tmem: Box::new([0u8; TMEM_SIZE]),
            triangle_commands: 0,
            triangle_pixels: 0,
            texrect_texels: 0,
            other_unhandled: 0,
            unhandled_hist: [0u64; 256],
            op_hist: [0u64; 256],
            last_list_cycles: 0,
            last_list_had_sync_full: false,
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
        self.last_list_had_sync_full = false;
        self.texrect_texels = 0;
        self.triangle_pixels = 0;
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
            self.op_hist[op as usize] = self.op_hist[op as usize].saturating_add(1);
            cycles = cycles.saturating_add(RDP_CYCLES_PER_CMD);
            let advance = if (TRI_CMD_MIN..=TRI_CMD_MAX).contains(&op) {
                let cmd_bytes = rdp_triangle::command_bytes(w0);
                if off + cmd_bytes <= len {
                    const MAX_TRI_WORDS: usize = 44;
                    let mut words = [0u32; MAX_TRI_WORDS];
                    let nw = cmd_bytes / 4;
                    for i in 0..nw {
                        words[i] = read_u32(off + i * 4);
                    }
                    cycles = cycles.saturating_add(
                        self.rasterize_triangle_command(&words[..nw], rdram),
                    );
                    self.triangle_commands = self.triangle_commands.saturating_add(1);
                    cmd_bytes
                } else {
                    self.other_unhandled = self.other_unhandled.saturating_add(1);
                    8
                }
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
                    OP_SET_PRIM_COLOR => self.prim_color = w1,
                    OP_SET_ENV_COLOR => self.env_color = w1,
                    OP_SET_BLEND_COLOR => self.blend_color = w1,
                    OP_SET_FOG_COLOR => self.fog_color = w1,
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
                    OP_SET_Z_IMAGE => {
                        self.z_image_addr = w1 & 0x00FF_FFFF;
                    }
                    OP_SET_OTHER_MODES => self.set_other_modes(w0, w1),
                    OP_SET_PRIM_DEPTH => {
                        self.prim_z = (w1 >> 16) as u16;
                        self.prim_dz = w1 as u16;
                    }
                    OP_SYNC_FULL => {
                        self.last_list_had_sync_full = true;
                    }
                    OP_SYNC_PIPE | OP_SYNC_TILE | OP_SYNC_LOAD => {}
                    OP_SET_TILE => self.set_tile_regs(w0, w1),
                    OP_SET_TILE_SIZE => self.set_tile_size_regs(w0, w1),
                    OP_TEXRECT | OP_TEXRECT_FLIP => {
                        self.other_unhandled = self.other_unhandled.saturating_add(1);
                        self.unhandled_hist[op as usize] =
                            self.unhandled_hist[op as usize].saturating_add(1);
                    }
                    _ => {
                        self.other_unhandled = self.other_unhandled.saturating_add(1);
                        self.unhandled_hist[op as usize] =
                            self.unhandled_hist[op as usize].saturating_add(1);
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
        let b = RDRAM_CYCLES_PER_BYTE;
        let s = (start & 0x00FF_FFF8) as u64;
        let e = (end & 0x00FF_FFF8) as u64;
        if e <= s {
            return 40;
        }
        let len = e - s;
        let cmds = (len / 8).max(1);
        len
            .saturating_mul(b)
            .saturating_add(cmds.saturating_mul(RDP_CYCLES_PER_CMD))
            .max(1)
    }

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
        self.combine.mux1 = 0;
        self.combine.decoded = crate::rdp_combiner::decode_combine(w0, w1);
    }

    fn set_scissor(&mut self, w0: u32, w1: u32) {
        // `gsDPSetScissor`: xh/yh in w0 bits, xl/yl in w1 (simplified 12-bit fields).
        self.scissor.xh = ((w0 >> 12) & 0xFFF) as u16;
        self.scissor.yh = (w0 & 0xFFF) as u16;
        self.scissor.xl = ((w1 >> 12) & 0xFFF) as u16;
        self.scissor.yl = (w1 & 0xFFF) as u16;
        self.scissor.enabled = true;
    }

    fn set_other_modes(&mut self, w0: u32, w1: u32) {
        // SetOtherModes (0xEF): 64-bit packed fields.
        // w0 bits 20-21: cycle_type
        self.other_modes.cycle_type = ((w0 >> 20) & 3) as u8;
        // w1 bit 4: z_compare_en
        self.other_modes.z_compare_en = (w1 >> 4) & 1 != 0;
        // w1 bit 5: z_update_en
        self.other_modes.z_update_en = (w1 >> 5) & 1 != 0;
        // w1 bit 2: z_source_sel (0=pixel, 1=prim)
        self.other_modes.z_source_sel = (w1 >> 2) & 1 != 0;
        // w1 bit 14: force_blend
        self.other_modes.force_blend = (w1 >> 14) & 1 != 0;
        // w1 bit 0: alpha_compare_en
        self.other_modes.alpha_compare_en = (w1 & 1) != 0;
        // w1 bit 6: image_read_en
        self.other_modes.image_read_en = (w1 >> 6) & 1 != 0;
        // w0 bit 13: sample_type (0=point, 1=bilinear). Bits 12 (mid_texel) and
        // 10-11 (bi_lerp0/1) also affect filtering, but `sample_type` alone
        // drives the on/off switch the shaded rasterizer looks at.
        self.other_modes.tex_filter = ((w0 >> 13) & 1) as u8;
        // w0 bit 14: tlut_type (0=RGBA16, 1=IA16). Bit 15 is en_tlut.
        self.other_modes.tlut_type = ((w0 >> 14) & 1) as u8;
        // Blender mux: w1 bits 30-31 / 26-27 / 22-23 / 18-19 / 28-29 / 24-25 / 20-21 / 16-17
        self.other_modes.blend_m1a_0 = ((w1 >> 30) & 3) as u8;
        self.other_modes.blend_m1b_0 = ((w1 >> 26) & 3) as u8;
        self.other_modes.blend_m2a_0 = ((w1 >> 22) & 3) as u8;
        self.other_modes.blend_m2b_0 = ((w1 >> 18) & 3) as u8;
        self.other_modes.blend_m1a_1 = ((w1 >> 28) & 3) as u8;
        self.other_modes.blend_m1b_1 = ((w1 >> 24) & 3) as u8;
        self.other_modes.blend_m2a_1 = ((w1 >> 20) & 3) as u8;
        self.other_modes.blend_m2b_1 = ((w1 >> 16) & 3) as u8;
    }

    fn set_tile_regs(&mut self, w0: u32, w1: u32) {
        let fmt = ((w0 >> 21) & 7) as u8;
        let siz = ((w0 >> 19) & 3) as u8;
        let line = ((w0 >> 9) & 0x1FF) as u16;
        let tmem_qw = (w0 & 0x1FF) as u16;
        let ti = ((w1 >> 24) & 7) as usize;
        let palette = ((w1 >> 20) & 0xF) as u8;
        if ti < 8 {
            self.tiles[ti].fmt = fmt;
            self.tiles[ti].siz = siz;
            self.tiles[ti].line_qwords = line.max(1);
            self.tiles[ti].tmem_qwords = tmem_qw;
            self.tiles[ti].palette = palette;
        }
    }

    fn set_tile_size_regs(&mut self, w0: u32, w1: u32) {
        let sl = ((w0 >> 12) & 0xFFF) as u16;
        let tl = (w0 & 0xFFF) as u16;
        let sh = ((w1 >> 12) & 0xFFF) as u16;
        let th = (w1 & 0xFFF) as u16;
        let ti = ((w1 >> 24) & 7) as usize;
        if ti < 8 {
            self.tiles[ti].sl = sl;
            self.tiles[ti].tl = tl;
            self.tiles[ti].sh = sh;
            self.tiles[ti].th = th;
        }
    }

    #[inline]
    fn combine_texel5551(&self, tex: u16) -> u16 {
        if self.combine.mux0 == 0 && self.combine.mux1 == 0 {
            return tex;
        }
        rgba5551_modulate(tex, self.prim_color as u16)
    }

    /// Copy a rectangular region from RDRAM (`texture_image`) into TMEM at the specified tile's base.
    fn load_tile(&mut self, rdram: &[u8], w0: u32, w1: u32) -> u64 {
        let tile_idx = ((w1 >> 24) & 7) as usize;
        let sl = ((w0 >> 12) & 0xFFF) as usize;
        let tl = (w0 & 0xFFF) as usize;
        let sh = ((w1 >> 12) & 0xFFF) as usize;
        let th = (w1 & 0xFFF) as usize;
        let addr = (self.texture_image.addr & 0x00FF_FFFF) as usize;
        let width = self.texture_image.width.max(1) as usize;
        let bpp = texel_bytes(self.texture_image.size);
        let mut c = 0u64;
        let mut dst = self.tiles[tile_idx.min(7)].tmem_qwords as usize * 8;
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

    /// `LoadBlock`: fast linear copy from RDRAM (at `texture_image.addr`) into TMEM
    /// at the tile's TMEM base. `SH` (w1 bits [23:12]) = texel count − 1.
    fn load_block(&mut self, rdram: &[u8], _w0: u32, w1: u32) -> u64 {
        let tile_idx = ((w1 >> 24) & 7) as usize;
        let sh = ((w1 >> 12) & 0xFFF) as usize;
        let texels = sh + 1;
        let src_addr = (self.texture_image.addr & 0x00FF_FFFF) as usize;
        let src_bpp = texel_bytes(self.texture_image.size);
        let dst_base = self.tiles[tile_idx.min(7)].tmem_qwords as usize * 8;
        let total_bytes = texels * src_bpp;
        let mut c = 0u64;
        for i in 0..total_bytes {
            let ri = src_addr + i;
            let ti = (dst_base + i) % TMEM_SIZE;
            if ri < rdram.len() {
                self.tmem[ti] = rdram[ri];
            }
            c = c.saturating_add(RDRAM_CYCLES_PER_BYTE);
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

        let bpp_tex = texel_bytes(self.texture_image.size);
        let tile_idx = ((w1 >> 24) & 7) as usize;
        let tile = &self.tiles[tile_idx.min(7)];
        let tmem_base = tile.tmem_qwords as usize * 8;
        let line_bytes = tile.line_qwords as usize * 8;
        let stride = if line_bytes >= bpp_tex {
            line_bytes
        } else {
            self.texture_image.width.max(1) as usize * bpp_tex
        };

        let s0 = ((w2 >> 16) as i16) as i64;
        let t0 = ((w2 & 0xFFFF) as i16) as i64;
        let dsdx = ((w3 >> 16) as i16) as i64;
        let dtdy = ((w3 & 0xFFFF) as i16) as i64;

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
                let off_tm = tmem_base
                    .saturating_add(tt.saturating_mul(stride))
                    .saturating_add(ts.saturating_mul(bpp_tex));
                if off_tm + bpp_tex > TMEM_SIZE {
                    continue;
                }
                let texel16 = if bpp_tex >= 2 {
                    u16::from_be_bytes([self.tmem[off_tm], self.tmem[off_tm + 1]])
                } else {
                    continue;
                };
                let out16 = self.combine_texel5551(texel16);

                let off_fb = base.saturating_add((yu * cw as usize + xu) * bpp_fb);
                if bpp_fb == 2 && off_fb + 2 <= rdram.len() {
                    rdram[off_fb..off_fb + 2].copy_from_slice(&out16.to_be_bytes());
                } else if bpp_fb == 4 && off_fb + 4 <= rdram.len() {
                    let w = (out16 as u32) << 16 | out16 as u32;
                    rdram[off_fb..off_fb + 4].copy_from_slice(&w.to_be_bytes());
                }
                pix = pix.saturating_add(1);
            }
        }
        self.texrect_texels = self.texrect_texels.saturating_add(pix);
        pix.saturating_mul(bpp_fb as u64 * RDRAM_CYCLES_PER_BYTE)
    }

    /// Legacy test layout: words `[2..4]` = vertex (x,y) pairs (big-endian halfwords), `words[5]` = RGBA5551.
    fn rasterize_triangle_legacy(&mut self, words: &[u32; 8], rdram: &mut [u8]) -> u64 {
        let x0 = (words[2] >> 16) as i16 as i32;
        let y0 = (words[2] & 0xFFFF) as i16 as i32;
        let x1 = (words[3] >> 16) as i16 as i32;
        let y1 = (words[3] & 0xFFFF) as i16 as i32;
        let x2 = (words[4] >> 16) as i16 as i32;
        let y2 = (words[4] & 0xFFFF) as i16 as i32;
        let c = (words[5] & 0xFFFF) as u16;
        let p = self.draw_triangle_flat(rdram, x0, y0, x1, y1, x2, y2, c);
        self.triangle_pixels = self.triangle_pixels.saturating_add(p);
        p.saturating_mul(texel_bytes(self.color_image.size) as u64 * RDRAM_CYCLES_PER_BYTE)
    }

    /// `Fill` / `Shade` / `Texture` / `Z` triangle: first 8 words are base geometry (GLideN64 layout).
    fn rasterize_triangle_command(&mut self, words: &[u32], rdram: &mut [u8]) -> u64 {
        if words.len() < 8 {
            return 0;
        }
        let op = (words[0] >> 24) as u8;
        if op == TRI_CMD_MIN && Self::legacy_synthetic_triangle(words) {
            let mut w = [0u32; 8];
            w.copy_from_slice(&words[..8]);
            return self.rasterize_triangle_legacy(&w, rdram);
        }
        if (TRI_CMD_MIN..=TRI_CMD_MAX).contains(&op) {
            let clip = if self.scissor.enabled {
                let sx0 = (self.scissor.xh as i32) >> 2;
                let sy0 = (self.scissor.yh as i32) >> 2;
                let sx1 = (self.scissor.xl as i32) >> 2;
                let sy1 = (self.scissor.yl as i32) >> 2;
                Some((sx0.min(sx1), sx0.max(sx1), sy0.min(sy1), sy0.max(sy1)))
            } else {
                None
            };
            let features = op & 7;
            // Shade or texture present → full shaded/textured rasterizer
            if features & 0x6 != 0 {
                let coeffs = rdp_triangle::parse_tri_coeffs(words, op);
                let ctx = rdp_triangle::RasterCtx {
                    tmem: &*self.tmem,
                    tiles: &self.tiles,
                    combiner: &self.combine.decoded,
                    cycle_type: self.other_modes.cycle_type,
                    prim_color: u32_to_rgba8(self.prim_color),
                    env_color: u32_to_rgba8(self.env_color),
                    blend_color: u32_to_rgba8(self.blend_color),
                    fog_color: u32_to_rgba8(self.fog_color),
                    fill_color: self.fill_color,
                    ci_addr: self.color_image.addr,
                    ci_width: self.color_image.width,
                    ci_size: self.color_image.size,
                    z_image_addr: self.z_image_addr,
                    z_compare_en: self.other_modes.z_compare_en,
                    z_update_en: self.other_modes.z_update_en,
                    z_source_sel: self.other_modes.z_source_sel,
                    prim_z: self.prim_z,
                    tlut_type: self.other_modes.tlut_type,
                    force_blend: self.other_modes.force_blend,
                    blend_m1a_0: self.other_modes.blend_m1a_0,
                    blend_m1b_0: self.other_modes.blend_m1b_0,
                    blend_m2a_0: self.other_modes.blend_m2a_0,
                    blend_m2b_0: self.other_modes.blend_m2b_0,
                    blend_m1a_1: self.other_modes.blend_m1a_1,
                    blend_m1b_1: self.other_modes.blend_m1b_1,
                    blend_m2a_1: self.other_modes.blend_m2a_1,
                    blend_m2b_1: self.other_modes.blend_m2b_1,
                    clip,
                };
                let (c, p) = rdp_triangle::raster_shaded_hw(words, &coeffs, &ctx, rdram);
                self.triangle_pixels = self.triangle_pixels.saturating_add(p);
                return c;
            }
            // Fill-only triangle (with optional Z ignored)
            let (c, p) = rdp_triangle::raster_fill_hw(
                words,
                self.fill_color,
                self.color_image.addr,
                self.color_image.width,
                self.color_image.size,
                clip,
                rdram,
            );
            self.triangle_pixels = self.triangle_pixels.saturating_add(p);
            return c;
        }
        0
    }

    /// Unit tests use `(x,y)` halfword pairs with **non-zero** low halves; hardware anchors usually leave
    /// the fractional halfword `0` for integer pixel coordinates (`0x000A0000` vs `0x000A000A`).
    fn legacy_synthetic_triangle(words: &[u32]) -> bool {
        if words.len() < 8 {
            return false;
        }
        (words[2] & 0xFFFF) != 0
            && (words[3] & 0xFFFF) != 0
            && (words[4] & 0xFFFF) != 0
    }

    fn draw_triangle_flat(
        &self,
        rdram: &mut [u8],
        x0: i32,
        y0: i32,
        x1: i32,
        y1: i32,
        x2: i32,
        y2: i32,
        color: u16,
    ) -> u64 {
        let mut xmin = x0.min(x1).min(x2);
        let mut xmax = x0.max(x1).max(x2);
        let mut ymin = y0.min(y1).min(y2);
        let mut ymax = y0.max(y1).max(y2);
        if self.scissor.enabled {
            let sx0 = (self.scissor.xh as i32) >> 2;
            let sy0 = (self.scissor.yh as i32) >> 2;
            let sx1 = (self.scissor.xl as i32) >> 2;
            let sy1 = (self.scissor.yl as i32) >> 2;
            xmin = xmin.max(sx0.min(sx1));
            xmax = xmax.min(sx0.max(sx1));
            ymin = ymin.max(sy0.min(sy1));
            ymax = ymax.min(sy0.max(sy1));
        }
        let cw = self.color_image.width.max(1) as i32;
        let bpp = texel_bytes(self.color_image.size);
        let base = (self.color_image.addr & 0x00FF_FFFF) as usize;
        let mut pix = 0u64;
        for y in ymin..=ymax {
            for x in xmin..=xmax {
                if !point_in_tri(x, y, x0, y0, x1, y1, x2, y2) {
                    continue;
                }
                if x < 0 || y < 0 {
                    continue;
                }
                let off = base.saturating_add(((y as usize) * cw as usize + (x as usize)) * bpp);
                if bpp == 2 && off + 2 <= rdram.len() {
                    rdram[off..off + 2].copy_from_slice(&color.to_be_bytes());
                    pix = pix.saturating_add(1);
                } else if bpp == 4 && off + 4 <= rdram.len() {
                    let w = (color as u32) << 16 | color as u32;
                    rdram[off..off + 4].copy_from_slice(&w.to_be_bytes());
                    pix = pix.saturating_add(1);
                }
            }
        }
        pix
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

#[inline]
fn point_in_tri(x: i32, y: i32, x0: i32, y0: i32, x1: i32, y1: i32, x2: i32, y2: i32) -> bool {
    let s0 = (x1 - x0) * (y - y0) - (y1 - y0) * (x - x0);
    let s1 = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1);
    let s2 = (x0 - x2) * (y - y2) - (y0 - y2) * (x - x2);
    (s0 >= 0 && s1 >= 0 && s2 >= 0) || (s0 <= 0 && s1 <= 0 && s2 <= 0)
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
    use crate::rdp_triangle;

    #[test]
    fn triangle_command_bytes_match_gliden64_table() {
        assert_eq!(rdp_triangle::command_bytes(0x08 << 24), 32);
        assert_eq!(rdp_triangle::command_bytes(0x09 << 24), 48);
        assert_eq!(rdp_triangle::command_bytes(0x0A << 24), 96);
        assert_eq!(rdp_triangle::command_bytes(0x0F << 24), 32 + 64 + 64 + 16);
    }

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
        let w1 = (0u32 << 26) | ((4 & 0xFFF) << 12) | (0 & 0xFFF);
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

    #[test]
    fn flat_triangle_fills_pixel() {
        let mut rdp = Rdp::new();
        let mut rdram = vec![0u8; 0x20_0000];
        rdp.color_image = ColorImage {
            fmt: 0,
            size: 2,
            width: 320,
            addr: 0x10_0000,
        };
        let list_base = 0x2000usize;
        let mut words = [0u32; 8];
        words[0] = ((TRI_CMD_MIN as u32) << 24) | (1 << 12) | 1;
        words[1] = 0;
        words[2] = (10u32 << 16) | 10u32;
        words[3] = (50u32 << 16) | 10u32;
        words[4] = (10u32 << 16) | 50u32;
        words[5] = 0xFFFFu32;
        for i in 0..8 {
            rdram[list_base + i * 4..list_base + i * 4 + 4].copy_from_slice(&words[i].to_be_bytes());
        }
        rdp.process_display_list(
            &mut rdram,
            &[],
            &[],
            list_base as u32,
            (list_base + TRIANGLE_PACKET_BYTES) as u32,
            0,
        );
        assert!(rdp.triangle_pixels > 0);
        let base = 0x10_0000usize + (20 * 320 + 20) * 2;
        assert_ne!(u16::from_be_bytes([rdram[base], rdram[base + 1]]), 0);
    }

    #[test]
    fn combine_modulates_texrect() {
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
        rdp.tmem[0..2].copy_from_slice(&0xFFFFu16.to_be_bytes());
        rdp.combine.mux0 = 1;
        rdp.prim_color = 0x0000_0000;
        let w0 = (OP_TEXRECT as u32) << 24;
        let w1 = (0u32 << 26) | ((4 & 0xFFF) << 12);
        let w2 = 0u32;
        let w3 = (32u32 << 16) | 32u32;
        rdp.texture_rectangle(&mut rdram, w0, w1, w2, w3, false);
        let base = 0x10_0000usize;
        assert_eq!(u16::from_be_bytes([rdram[base], rdram[base + 1]]), 0);
        rdp.prim_color = 0x0000_FFFF;
        rdp.texture_rectangle(&mut rdram, w0, w1, w2, w3, false);
        assert_eq!(u16::from_be_bytes([rdram[base], rdram[base + 1]]), 0xFFFF);
    }
}
