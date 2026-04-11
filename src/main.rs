//! UltraReality — run with a `.z64` ROM or `--demo` to verify video output.

use std::env;
use std::fs;
use std::path::Path;
use std::time::Instant;

use minifb::{Key, Window, WindowOptions};

use ultrareality::{
    blit_rgba5551, Machine, VI_NTSC_CYCLES_PER_FRAME, VI_REG_ORIGIN, VI_REG_WIDTH,
};

const OUT_W: usize = 320;
const OUT_H: usize = 240;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() >= 2 && args[1] == "--demo" {
        run_demo();
        return;
    }
    if args.len() < 2 {
        eprintln!("UltraReality — N64 emulator (early)\n");
        eprintln!("  ultrareality <game.z64>   — boot ROM and show VI framebuffer");
        eprintln!("  ultrareality --demo       — synthetic pattern (no ROM)");
        std::process::exit(1);
    }
    let path = Path::new(&args[1]);
    run_rom(path);
}

fn run_demo() {
    let mut m = Machine::new();
    let origin = 0x0010_0000usize;
    m.bus.vi.regs[VI_REG_ORIGIN] = origin as u32;
    m.bus.vi.regs[VI_REG_WIDTH] = OUT_W as u32;

    let w = OUT_W;
    let h = OUT_H;
    for y in 0..h {
        for x in 0..w {
            let idx = origin + (y * w + x) * 2;
            if idx + 2 > m.bus.rdram.data.len() {
                break;
            }
            let cx = (x / 32) & 1;
            let cy = (y / 32) & 1;
            let p: u16 = if cx ^ cy != 0 { 0xFFFF } else { 0x001F };
            let be = p.to_be_bytes();
            m.bus.rdram.data[idx] = be[0];
            m.bus.rdram.data[idx + 1] = be[1];
        }
    }

    let mut buffer = vec![0u32; OUT_W * OUT_H];
    let mut window = Window::new(
        "UltraReality — demo",
        OUT_W,
        OUT_H,
        WindowOptions::default(),
    )
    .expect("minifb window");

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let o = m.bus.vi.framebuffer_rdram_offset();
        let fw = m.bus.vi.display_width();
        let fh = m.bus.vi.display_height();
        blit_rgba5551(&m.bus.rdram.data, o, fw, fh, &mut buffer, OUT_W, OUT_H);
        window
            .update_with_buffer(&buffer, OUT_W, OUT_H)
            .expect("update");
    }
}

fn run_rom(path: &Path) {
    let rom = match fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("failed to read {:?}: {}", path, e);
            std::process::exit(1);
        }
    };

    let mut m = Machine::new();
    m.set_cartridge_rom(rom);
    m.bootstrap_hle_cart_entry();

    let title = format!("UltraReality — {}", path.file_name().unwrap_or_default().to_string_lossy());
    let mut window = Window::new(&title, OUT_W, OUT_H, WindowOptions::default()).expect("minifb");
    let mut buffer = vec![0u32; OUT_W * OUT_H];

    let mut last_status = Instant::now();
    let mut frames = 0u64;

    let mut cpu_halted = false;
    while window.is_open() && !window.is_key_down(Key::Escape) {
        let before = m.bus.vi.frame_counter;
        let mut steps = 0u64;
        const MAX_STEPS_PER_VI_FRAME: u64 = 50_000_000;
        while m.bus.vi.frame_counter == before && steps < MAX_STEPS_PER_VI_FRAME && !cpu_halted {
            match m.step() {
                Ok(()) => steps += 1,
                Err(e) => {
                    eprintln!("CPU halted: {:?}", e);
                    cpu_halted = true;
                    break;
                }
            }
        }
        // If the CPU stopped before VI advanced, still tick VI once per displayed frame
        // so the window loop does not spin forever.
        if m.bus.vi.frame_counter == before {
            m.bus.advance_vi_frame_timing(VI_NTSC_CYCLES_PER_FRAME);
        }
        frames += 1;
        if last_status.elapsed().as_secs() >= 1 {
            eprintln!(
                "VI frame {} | ~{:.0} VI fps | last CPU steps {}",
                m.bus.vi.frame_counter,
                frames as f64 / last_status.elapsed().as_secs_f64().max(0.001),
                steps
            );
            frames = 0;
            last_status = Instant::now();
        }

        let o = m.bus.vi.framebuffer_rdram_offset();
        let fw = m.bus.vi.display_width();
        let fh = m.bus.vi.display_height();
        blit_rgba5551(&m.bus.rdram.data, o, fw, fh, &mut buffer, OUT_W, OUT_H);

        window
            .update_with_buffer(&buffer, OUT_W, OUT_H)
            .expect("update");
    }
}
