//! UltraReality — run with a `.z64` ROM or `--demo` to verify video output.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
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
        print_usage();
        std::process::exit(1);
    }
    let tail: Vec<String> = args[1..].to_vec();
    let (pif_path, fast_boot, rom_path) = match parse_rom_args(&tail) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("{e}");
            print_usage();
            std::process::exit(1);
        }
    };
    run_rom(&rom_path, pif_path.as_deref(), fast_boot);
}

fn print_usage() {
    eprintln!("UltraReality — N64 emulator (early)\n");
    eprintln!("  ultrareality <game.z64>              — cart header PC + IPL3 region via PI DMA");
    eprintln!("  ultrareality --pif <pif.bin> <game>   — boot CPU from PIF reset (>= {} bytes)", ultrareality::PIF_ROM_LEN);
    eprintln!("  ultrareality --pif <pif> --fast-boot <game>");
    eprintln!("    load PIF into memory but use cart-header + IPL3 PI DMA (dev shortcut)");
    eprintln!("    (or set ULTRAREALITY_PIF; overridden by --pif)");
    eprintln!("  ultrareality --demo                  — synthetic pattern (no ROM)");
}

/// After `--demo` is handled, parse optional leading `--pif PATH`, `--fast-boot`, then one ROM path.
fn parse_rom_args(args: &[String]) -> Result<(Option<PathBuf>, bool, PathBuf), String> {
    let mut i = 0usize;
    let mut pif_path = None;
    let mut fast_boot = false;
    while i < args.len() {
        if args[i] == "--pif" {
            let p = args
                .get(i + 1)
                .ok_or_else(|| "missing path after --pif".to_string())?;
            pif_path = Some(PathBuf::from(p));
            i += 2;
            continue;
        }
        if args[i] == "--fast-boot" {
            fast_boot = true;
            i += 1;
            continue;
        }
        break;
    }
    let rest = &args[i..];
    if rest.len() != 1 {
        return Err(if rest.is_empty() {
            "missing ROM path".into()
        } else {
            "expected exactly one game ROM path (after optional flags)".into()
        });
    }
    if pif_path.is_none() {
        if let Ok(p) = env::var("ULTRAREALITY_PIF") {
            if !p.is_empty() {
                pif_path = Some(PathBuf::from(p));
            }
        }
    }
    Ok((pif_path, fast_boot, PathBuf::from(&rest[0])))
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
        m.bus.schedule_vi_frame_fetch();
        window
            .update_with_buffer(&buffer, OUT_W, OUT_H)
            .expect("update");
    }
}

fn run_rom(path: &Path, pif_path: Option<&Path>, fast_boot: bool) {
    let rom = match fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("failed to read {:?}: {}", path, e);
            std::process::exit(1);
        }
    };

    let mut m = Machine::new();
    let mut pif_ok = false;
    if let Some(pp) = pif_path {
        match fs::read(pp) {
            Ok(pif_bytes) => {
                if let Err(e) = m.set_pif_rom(&pif_bytes) {
                    eprintln!("invalid PIF ROM {:?}: {:?}", pp, e);
                    std::process::exit(1);
                }
                pif_ok = true;
            }
            Err(e) => {
                eprintln!("failed to read PIF {:?}: {}", pp, e);
                std::process::exit(1);
            }
        }
    }
    m.set_cartridge_rom(rom);
    if pif_ok && !fast_boot {
        m.bootstrap_from_pif_reset();
    } else {
        m.bootstrap_cart_from_rom();
    }

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
        m.bus.schedule_vi_frame_fetch();

        window
            .update_with_buffer(&buffer, OUT_W, OUT_H)
            .expect("update");
    }
}
