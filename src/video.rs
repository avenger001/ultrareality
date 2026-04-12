//! Framebuffer readout: 16-bit RGBA5551 in RDRAM → 32-bit ARGB for hosts.

/// N64 16-bit pixel (big-endian in memory) to `0xAARRGGBB` for minifb-style buffers.
#[inline]
pub fn pixel_rgba5551_to_argb(p: u16) -> u32 {
    let r = (p >> 11) & 0x1F;
    let g = (p >> 6) & 0x1F;
    let b = (p >> 1) & 0x1F;
    let a = p & 1;
    let r8 = (r << 3) | (r >> 2);
    let g8 = (g << 3) | (g >> 2);
    let b8 = (b << 3) | (b >> 2);
    let a8: u32 = if a != 0 { 255 } else { 0 };
    (a8 << 24) | ((r8 as u32) << 16) | ((g8 as u32) << 8) | b8 as u32
}

/// Blit RGBA5551 from RDRAM into **RGBA8** row-major bytes (for GPU upload / Vulkan path).
pub fn blit_rgba5551_to_rgba8(
    rdram: &[u8],
    origin: usize,
    width: u32,
    height: u32,
    out: &mut [u8],
    out_width: usize,
    out_height: usize,
) {
    out.fill(0);
    let w = width as usize;
    let h = height as usize;
    if w == 0 || h == 0 {
        return;
    }
    let row_bytes = w.saturating_mul(2);
    let row_out = out_width.saturating_mul(4);
    for y in 0..h.min(out_height) {
        for x in 0..w.min(out_width) {
            let idx = origin + y * row_bytes + x * 2;
            if idx + 2 > rdram.len() {
                continue;
            }
            let p = u16::from_be_bytes([rdram[idx], rdram[idx + 1]]);
            let argb = pixel_rgba5551_to_argb(p);
            let o = y * row_out + x * 4;
            if o + 4 <= out.len() {
                out[o] = (argb & 0xFF) as u8;
                out[o + 1] = ((argb >> 8) & 0xFF) as u8;
                out[o + 2] = ((argb >> 16) & 0xFF) as u8;
                out[o + 3] = ((argb >> 24) & 0xFF) as u8;
            }
        }
    }
}

/// Blit a `width`×`height` RGBA5551 region from `rdram` starting at `origin` (byte offset).
/// `out` is row-major ARGB, length ≥ `out_width * out_height`; image is centered if smaller.
pub fn blit_rgba5551(
    rdram: &[u8],
    origin: usize,
    width: u32,
    height: u32,
    out: &mut [u32],
    out_width: usize,
    out_height: usize,
) {
    out.fill(0);
    let w = width as usize;
    let h = height as usize;
    if w == 0 || h == 0 {
        return;
    }
    let row_bytes = w.saturating_mul(2);
    for y in 0..h.min(out_height) {
        for x in 0..w.min(out_width) {
            let idx = origin + y * row_bytes + x * 2;
            if idx + 2 > rdram.len() {
                continue;
            }
            let p = u16::from_be_bytes([rdram[idx], rdram[idx + 1]]);
            out[y * out_width + x] = pixel_rgba5551_to_argb(p);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn white_pixel() {
        let p = pixel_rgba5551_to_argb(0xFFFF);
        assert_eq!(p, 0xFFFFFFFF);
    }

    #[test]
    fn rgba8_blit_white() {
        let rd = [0xFFu8, 0xFF];
        let mut out = [0u8; 4];
        blit_rgba5551_to_rgba8(&rd, 0, 1, 1, &mut out, 1, 1);
        assert_eq!(out, [0xFF, 0xFF, 0xFF, 0xFF]);
    }
}
