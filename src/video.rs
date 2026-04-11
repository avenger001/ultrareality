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
}
