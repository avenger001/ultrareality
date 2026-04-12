//! UltraReality — run with a `.z64` ROM, `--demo`, or `--vk-test` (solid Vulkan frame).

use std::cell::RefCell;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::Instant;

use minifb::{Key, Window, WindowOptions};

use ultrareality::{
    blit_rgba5551, blit_rgba5551_to_rgba8, graphics_phase_reached, run_wgpu_loop, Machine,
    PresentError, VI_NTSC_CYCLES_PER_FRAME, VI_REG_ORIGIN, VI_REG_WIDTH,
};

const OUT_W: u32 = 320;
const OUT_H: u32 = 240;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() >= 2 && args[1] == "--demo" {
        if use_minifb() {
            run_demo_minifb();
        } else {
            run_demo_wgpu();
        }
        return;
    }
    if args.len() >= 2 && args[1] == "--vk-test" {
        run_vk_test_pattern();
        return;
    }
    if args.len() < 2 {
        print_usage();
        std::process::exit(1);
    }
    let tail: Vec<String> = args[1..].to_vec();
    let (pif_path, fast_boot, rom_path, minifb) = match parse_rom_args(&tail) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("{e}");
            print_usage();
            std::process::exit(1);
        }
    };
    if minifb {
        run_rom_minifb(&rom_path, pif_path.as_deref(), fast_boot);
    } else {
        run_rom_wgpu(&rom_path, pif_path.as_deref(), fast_boot);
    }
}

fn use_minifb() -> bool {
    env::var("ULTRAREALITY_MINIFB").ok().as_deref() == Some("1")
}

fn print_usage() {
    eprintln!("UltraReality — N64 emulator (early)\n");
    eprintln!("Graphics roadmap phase: {:?}\n", graphics_phase_reached());
    eprintln!("  ultrareality <game.z64>              — Vulkan (ULTRAREALITY_MINIFB=1 or --minifb for minifb)");
    eprintln!("  ultrareality --pif <pif.bin> <game>   — boot CPU from PIF reset (>= {} bytes)", ultrareality::PIF_ROM_LEN);
    eprintln!("  ultrareality --pif <pif> --fast-boot <game>");
    eprintln!("    (or set ULTRAREALITY_PIF; overridden by --pif)");
    eprintln!("  ultrareality --demo                  — synthetic pattern");
    eprintln!("  ultrareality --vk-test               — Phase 0: solid frame through Vulkan");
}

fn parse_rom_args(args: &[String]) -> Result<(Option<PathBuf>, bool, PathBuf, bool), String> {
    let mut i = 0usize;
    let mut pif_path = None;
    let mut fast_boot = false;
    let mut minifb = false;
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
        if args[i] == "--minifb" {
            minifb = true;
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
    Ok((pif_path, fast_boot, PathBuf::from(&rest[0]), minifb))
}

fn run_vk_test_pattern() {
    let n = (OUT_W * OUT_H * 4) as usize;
    let mut buf = vec![0u8; n];
    for px in buf.chunks_exact_mut(4) {
        px.copy_from_slice(&[0xFF, 0x00, 0xFF, 0xFF]);
    }
    let title = "UltraReality — vk test (Phase 0)";
    if let Err(e) = run_wgpu_loop(title, OUT_W, OUT_H, move || (buf.clone(), true)) {
        eprintln!("Vulkan/wgpu init failed: {e:?}");
        std::process::exit(1);
    }
}

fn run_demo_minifb() {
    let mut m = Machine::new();
    let origin = 0x0010_0000usize;
    m.bus.vi.regs[VI_REG_ORIGIN] = origin as u32;
    m.bus.vi.regs[VI_REG_WIDTH] = OUT_W;

    let w = OUT_W as usize;
    let h = OUT_H as usize;
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

    let mut buffer = vec![0u32; (OUT_W * OUT_H) as usize];
    let mut window = Window::new(
        "UltraReality — demo (minifb)",
        OUT_W as usize,
        OUT_H as usize,
        WindowOptions::default(),
    )
    .expect("minifb window");

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let o = m.bus.vi.framebuffer_rdram_offset();
        let fw = m.bus.vi.display_width();
        let fh = m.bus.vi.display_height();
        blit_rgba5551(
            &m.bus.rdram.data,
            o,
            fw,
            fh,
            &mut buffer,
            OUT_W as usize,
            OUT_H as usize,
        );
        m.bus.schedule_vi_frame_fetch();
        window
            .update_with_buffer(&buffer, OUT_W as usize, OUT_H as usize)
            .expect("update");
    }
}

fn run_demo_wgpu() {
    let m = Rc::new(RefCell::new(Machine::new()));
    {
        let mut g = m.borrow_mut();
        let origin = 0x0010_0000usize;
        g.bus.vi.regs[VI_REG_ORIGIN] = origin as u32;
        g.bus.vi.regs[VI_REG_WIDTH] = OUT_W;
        let w = OUT_W as usize;
        let h = OUT_H as usize;
        for y in 0..h {
            for x in 0..w {
                let idx = origin + (y * w + x) * 2;
                if idx + 2 > g.bus.rdram.data.len() {
                    break;
                }
                let cx = (x / 32) & 1;
                let cy = (y / 32) & 1;
                let p: u16 = if cx ^ cy != 0 { 0xFFFF } else { 0x001F };
                let be = p.to_be_bytes();
                g.bus.rdram.data[idx] = be[0];
                g.bus.rdram.data[idx + 1] = be[1];
            }
        }
    }

    let rgba = Rc::new(RefCell::new(vec![0u8; (OUT_W * OUT_H * 4) as usize]));
    let title = "UltraReality — demo (Vulkan)";
    let m2 = Rc::clone(&m);
    let rgba2 = Rc::clone(&rgba);
    if let Err(e) = run_wgpu_loop(title, OUT_W, OUT_H, move || {
        let mut g = m2.borrow_mut();
        let o = g.bus.vi.framebuffer_rdram_offset();
        let fw = g.bus.vi.display_width();
        let fh = g.bus.vi.display_height();
        let mut buf = rgba2.borrow_mut();
        blit_rgba5551_to_rgba8(
            &g.bus.rdram.data,
            o,
            fw,
            fh,
            &mut buf,
            OUT_W as usize,
            OUT_H as usize,
        );
        g.bus.schedule_vi_frame_fetch();
        (buf.clone(), true)
    }) {
        eprintln!("Vulkan/wgpu: {e:?}");
        std::process::exit(1);
    }
}

fn run_rom_minifb(path: &Path, pif_path: Option<&Path>, fast_boot: bool) {
    let rom = match fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("failed to read {:?}: {}", path, e);
            std::process::exit(1);
        }
    };

    let mut m = Machine::new();
    load_rom_and_boot(&mut m, rom, pif_path, fast_boot);

    let title = format!(
        "UltraReality — {}",
        path.file_name().unwrap_or_default().to_string_lossy()
    );
    let mut window =
        Window::new(&title, OUT_W as usize, OUT_H as usize, WindowOptions::default()).expect("minifb");
    let mut buffer = vec![0u32; (OUT_W * OUT_H) as usize];

    let mut last_status = Instant::now();
    let mut frames = 0u64;

    let mut cpu_halted = false;
    while window.is_open() && !window.is_key_down(Key::Escape) {
        step_until_vi_frame(&mut m, &mut cpu_halted);
        frames += 1;
        log_fps(&mut last_status, &mut frames, m.bus.vi.frame_counter);

        let o = m.bus.vi.framebuffer_rdram_offset();
        let fw = m.bus.vi.display_width();
        let fh = m.bus.vi.display_height();
        blit_rgba5551(
            &m.bus.rdram.data,
            o,
            fw,
            fh,
            &mut buffer,
            OUT_W as usize,
            OUT_H as usize,
        );
        m.bus.schedule_vi_frame_fetch();

        window
            .update_with_buffer(&buffer, OUT_W as usize, OUT_H as usize)
            .expect("update");
    }
}

struct WgpuRomCtx {
    machine: Machine,
    cpu_halted: bool,
    rgba: Vec<u8>,
    last_status: Instant,
    frames: u64,
}

impl WgpuRomCtx {
    fn step_vi(&mut self) {
        step_until_vi_frame(&mut self.machine, &mut self.cpu_halted);
    }

    fn blit_to_rgba(&mut self) {
        let o = self.machine.bus.vi.framebuffer_rdram_offset();
        let fw = self.machine.bus.vi.display_width();
        let fh = self.machine.bus.vi.display_height();
        blit_rgba5551_to_rgba8(
            &self.machine.bus.rdram.data,
            o,
            fw,
            fh,
            &mut self.rgba,
            OUT_W as usize,
            OUT_H as usize,
        );
    }

    fn log_fps_line(&mut self, vi_fc: u64) {
        if self.last_status.elapsed().as_secs() >= 1 {
            eprintln!(
                "VI frame {} | ~{:.0} displayed fps | phase {:?}",
                vi_fc,
                self.frames as f64 / self.last_status.elapsed().as_secs_f64().max(0.001),
                graphics_phase_reached()
            );
            self.frames = 0;
            self.last_status = Instant::now();
        }
    }
}

fn run_rom_wgpu(path: &Path, pif_path: Option<&Path>, fast_boot: bool) {
    let rom = match fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("failed to read {:?}: {}", path, e);
            std::process::exit(1);
        }
    };

    let mut machine = Machine::new();
    load_rom_and_boot(&mut machine, rom, pif_path, fast_boot);

    let ctx = Rc::new(RefCell::new(WgpuRomCtx {
        machine,
        cpu_halted: false,
        rgba: vec![0u8; (OUT_W * OUT_H * 4) as usize],
        last_status: Instant::now(),
        frames: 0,
    }));

    let title = format!(
        "UltraReality — {}",
        path.file_name().unwrap_or_default().to_string_lossy()
    );
    let c = Rc::clone(&ctx);
    if let Err(e) = run_wgpu_loop(&title, OUT_W, OUT_H, move || {
        let mut st = c.borrow_mut();
        st.step_vi();
        st.frames += 1;
        let vi_fc = st.machine.bus.vi.frame_counter;
        st.log_fps_line(vi_fc);

        st.blit_to_rgba();
        st.machine.bus.schedule_vi_frame_fetch();
        (st.rgba.clone(), true)
    }) {
        match e {
            PresentError::WgpuRequest(s) | PresentError::Surface(s) => {
                eprintln!("Vulkan/wgpu failed ({s}). Try --minifb or ULTRAREALITY_MINIFB=1.");
            }
        }
        std::process::exit(1);
    }
}

fn load_rom_and_boot(m: &mut Machine, rom: Vec<u8>, pif_path: Option<&Path>, fast_boot: bool) {
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
}

fn step_until_vi_frame(m: &mut Machine, cpu_halted: &mut bool) {
    let before = m.bus.vi.frame_counter;
    let mut steps = 0u64;
    const MAX_STEPS_PER_VI_FRAME: u64 = 50_000_000;
    while m.bus.vi.frame_counter == before && steps < MAX_STEPS_PER_VI_FRAME && !*cpu_halted {
        match m.step() {
            Ok(()) => steps += 1,
            Err(e) => {
                eprintln!("CPU halted: {:?}", e);
                *cpu_halted = true;
                break;
            }
        }
    }
    if m.bus.vi.frame_counter == before {
        m.bus
            .advance_vi_frame_timing(VI_NTSC_CYCLES_PER_FRAME);
    }
}

fn log_fps(last_status: &mut Instant, frames: &mut u64, vi_frame: u64) {
    if last_status.elapsed().as_secs() >= 1 {
        eprintln!(
            "VI frame {} | ~{:.0} displayed fps | phase {:?}",
            vi_frame,
            *frames as f64 / last_status.elapsed().as_secs_f64().max(0.001),
            graphics_phase_reached()
        );
        *frames = 0;
        *last_status = Instant::now();
    }
}
