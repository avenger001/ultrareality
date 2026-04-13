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
    PresentError, WgpuFrame, VI_NTSC_CYCLES_PER_FRAME, VI_REG_ORIGIN, VI_REG_WIDTH,
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
    let (pif_path, fast_boot, rom_path, minifb, headless_frames) = match parse_rom_args(&tail) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("{e}");
            print_usage();
            std::process::exit(1);
        }
    };
    if let Some(n) = headless_frames {
        run_rom_headless(&rom_path, pif_path.as_deref(), fast_boot, n);
    } else if minifb {
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

fn parse_rom_args(args: &[String]) -> Result<(Option<PathBuf>, bool, PathBuf, bool, Option<u64>), String> {
    let mut i = 0usize;
    let mut pif_path = None;
    let mut fast_boot = false;
    let mut minifb = false;
    let mut headless_frames: Option<u64> = None;
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
        if args[i] == "--headless-frames" {
            let n = args
                .get(i + 1)
                .ok_or_else(|| "missing N after --headless-frames".to_string())?
                .parse::<u64>()
                .map_err(|_| "invalid N for --headless-frames".to_string())?;
            headless_frames = Some(n);
            i += 2;
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
    Ok((pif_path, fast_boot, PathBuf::from(&rest[0]), minifb, headless_frames))
}

fn run_rom_headless(path: &Path, pif_path: Option<&Path>, fast_boot: bool, frames: u64) {
    let rom = match fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("failed to read {:?}: {}", path, e);
            std::process::exit(1);
        }
    };
    let mut m = Machine::new();
    load_rom_and_boot(&mut m, rom, pif_path, fast_boot);
    let mut cpu_halted = false;
    for fi in 0..frames {
        // Arm matrix-watch only for the final couple of frames around the Goddard panic.
        if fi == 896 {
            ultrareality::cpu::MATRIX_WATCH_ARMED
                .store(true, std::sync::atomic::Ordering::Relaxed);
        }
        step_until_vi_frame(&mut m, &mut cpu_halted);
        if cpu_halted {
            break;
        }
    }
    eprintln!("[headless] done after {} frames", frames);
    eprintln!("[headless] FINAL VI_INT_RAISES={} VI_INT_ACKS={} TOTAL_INT_TAKEN={} ERET_EXEC={}",
        ultrareality::vi::VI_INT_RAISE_COUNT.load(std::sync::atomic::Ordering::Relaxed),
        ultrareality::vi::VI_INT_ACK_COUNT.load(std::sync::atomic::Ordering::Relaxed),
        ultrareality::cpu::cop0::INT_TAKEN_COUNT.load(std::sync::atomic::Ordering::Relaxed),
        ultrareality::cpu::cop0::ERET_EXEC_COUNT.load(std::sync::atomic::Ordering::Relaxed));
    eprintln!("[headless] FINAL COMPARE_WRITE_TOTAL={}",
        ultrareality::cpu::cop0::COMPARE_WRITE_COUNT.load(std::sync::atomic::Ordering::Relaxed));
    // Print interrupt IP histogram
    eprint!("[headless] INT_TAKEN per IP bit: ");
    for b in 0..8 {
        let c = ultrareality::cpu::cop0::INT_TAKEN_IP_HISTOGRAM[b]
            .load(std::sync::atomic::Ordering::Relaxed);
        eprint!("IP{}={} ", b, c);
    }
    eprintln!();
    let mi_names = ["SP", "SI", "AI", "VI", "PI", "DP"];
    eprint!("[headless] INT_TAKEN per MI bit: ");
    for b in 0..6 {
        let c = ultrareality::cpu::cop0::INT_TAKEN_MI_HISTOGRAM[b]
            .load(std::sync::atomic::Ordering::Relaxed);
        eprint!("{}={} ", mi_names[b], c);
    }
    eprintln!();
    eprintln!("[headless] MI: intr=0x{:02X} mask=0x{:02X}", m.bus.mi.intr, m.bus.mi.mask);
    eprintln!(
        "[headless] FINAL RDP: tris={} unhandled={} tex_texels={} last_cycles={} last_sync_full={} CI(fmt={} size={} w={} addr=0x{:06X}) Z=0x{:06X} cyc_type={} combine_mux0=0x{:016X}",
        m.bus.rdp.triangle_commands,
        m.bus.rdp.other_unhandled,
        m.bus.rdp.texrect_texels,
        m.bus.rdp.last_list_cycles,
        m.bus.rdp.last_list_had_sync_full,
        m.bus.rdp.color_image.fmt,
        m.bus.rdp.color_image.size,
        m.bus.rdp.color_image.width,
        m.bus.rdp.color_image.addr,
        m.bus.rdp.z_image_addr,
        m.bus.rdp.other_modes.cycle_type,
        m.bus.rdp.combine.mux0,
    );
    eprintln!("[headless] FINAL VI_ORIGIN=0x{:08X} VI_WIDTH={}",
        m.bus.vi.regs[ultrareality::VI_REG_ORIGIN],
        m.bus.vi.regs[ultrareality::VI_REG_WIDTH]);
    {
        let mut top: Vec<(u8, u64)> = m.bus.rdp.unhandled_hist.iter().enumerate()
            .filter(|(_, &c)| c > 0)
            .map(|(i, &c)| (i as u8, c))
            .collect();
        top.sort_by_key(|&(_, c)| std::cmp::Reverse(c));
        let s: Vec<String> = top.iter().take(16)
            .map(|(op, c)| format!("0x{:02X}={}", op, c))
            .collect();
        eprintln!("[headless] FINAL RDP unhandled ops: {}", s.join(" "));

        let mut all: Vec<(u8, u64)> = m.bus.rdp.op_hist.iter().enumerate()
            .filter(|(_, &c)| c > 0)
            .map(|(i, &c)| (i as u8, c))
            .collect();
        all.sort_by_key(|&(_, c)| std::cmp::Reverse(c));
        let s2: Vec<String> = all.iter().take(20)
            .map(|(op, c)| format!("0x{:02X}={}", op, c))
            .collect();
        eprintln!("[headless] FINAL RDP all ops: {}", s2.join(" "));
    }
    // Final RSP state — to diagnose stuck microcode after the last START
    eprintln!(
        "[headless] FINAL RSP: pc=0x{:03X} halted={} broke={} sig=0x{:02X} dma_busy={}",
        m.bus.rsp_pc, m.bus.sp_halted, m.bus.sp_broke, m.bus.sp_signal,
        m.bus.sp_dma_busy(),
    );
    eprintln!(
        "[headless] FINAL RSP unimpl_count={} last_unimpl=0x{:08X}",
        ultrareality::rsp::RSP_UNIMPL_COUNT.load(std::sync::atomic::Ordering::Relaxed),
        ultrareality::rsp::RSP_LAST_UNIMPL.load(std::sync::atomic::Ordering::Relaxed),
    );
    eprintln!(
        "[headless] FINAL COP2 unk_funct={} unk_rs={} lwc2_unk={} swc2_unk={} last=0x{:08X}",
        ultrareality::rsp_vu::COP2_UNKNOWN_FUNCT_COUNT.load(std::sync::atomic::Ordering::Relaxed),
        ultrareality::rsp_vu::COP2_UNKNOWN_RS_COUNT.load(std::sync::atomic::Ordering::Relaxed),
        ultrareality::rsp_vu::LWC2_UNKNOWN_COUNT.load(std::sync::atomic::Ordering::Relaxed),
        ultrareality::rsp_vu::SWC2_UNKNOWN_COUNT.load(std::sync::atomic::Ordering::Relaxed),
        ultrareality::rsp_vu::COP2_LAST_UNKNOWN.load(std::sync::atomic::Ordering::Relaxed),
    );
    {
        let mut pairs: Vec<(usize, u32)> = (0..64)
            .map(|f| (f, ultrareality::rsp_vu::COP2_FUNCT_HIST[f].load(std::sync::atomic::Ordering::Relaxed)))
            .filter(|(_, c)| *c > 0)
            .collect();
        pairs.sort_by_key(|p| std::cmp::Reverse(p.1));
        let head: Vec<String> = pairs.iter().take(20)
            .map(|(f, c)| format!("0x{:02X}:{}", f, c))
            .collect();
        eprintln!("[headless] FINAL COP2 funct top: {}", head.join(" "));
    }
    if let Ok(t) = ultrareality::rsp::GFX_TRACE.lock() {
        if t.total > 0 {
            t.dump_head(65536, "END-OF-RUN HEAD (no BREAK)");
            t.dump_tail(65536, "END-OF-RUN TAIL (no BREAK)");
        }
    }
    // Top 16 hottest RSP IMEM PCs
    {
        let mut pairs: Vec<(usize, u32)> = (0..1024)
            .map(|i| (i << 2, ultrareality::rsp::RSP_PC_HIST[i].load(std::sync::atomic::Ordering::Relaxed)))
            .filter(|(_, c)| *c > 0)
            .collect();
        pairs.sort_by_key(|p| std::cmp::Reverse(p.1));
        let head: Vec<String> = pairs.iter().take(16)
            .map(|(pc, c)| format!("{:03X}:{}", pc, c))
            .collect();
        eprintln!("[headless] FINAL RSP PC top: {}", head.join(" "));
    }
    // OSTask in DMEM at end of run
    {
        let osk: Vec<String> = m.bus.rsp_dmem[0xFC0..0xFE0]
            .chunks(4)
            .map(|c| format!("{:02X}{:02X}{:02X}{:02X}", c[0], c[1], c[2], c[3]))
            .collect();
        eprintln!("[headless] FINAL DMEM[FC0..FE0]: {}", osk.join(" "));
    }
    // First 16 instructions of IMEM around current PC
    {
        let pc = m.bus.rsp_pc as usize & !0x3;
        let lo = pc.saturating_sub(0x10);
        let hi = (pc + 0x30).min(m.bus.rsp_imem.len());
        let mut chunks: Vec<String> = Vec::new();
        for off in (lo..hi).step_by(4) {
            let c = &m.bus.rsp_imem[off..off+4];
            let marker = if off == pc { "*" } else { " " };
            chunks.push(format!("{}{:03X}:{:02X}{:02X}{:02X}{:02X}", marker, off, c[0], c[1], c[2], c[3]));
        }
        eprintln!("[headless] FINAL IMEM near PC: {}", chunks.join(" "));
    }
    // Dump the hottest loop region: 0x000..0x060 and 0x3C0..0x3E0
    {
        let dump = |lo: usize, hi: usize, label: &str| {
            let mut chunks: Vec<String> = Vec::new();
            for off in (lo..hi).step_by(4) {
                let c = &m.bus.rsp_imem[off..off+4];
                chunks.push(format!("{:03X}:{:02X}{:02X}{:02X}{:02X}", off, c[0], c[1], c[2], c[3]));
            }
            eprintln!("[headless] {}: {}", label, chunks.join(" "));
        };
        dump(0x000, 0x060, "IMEM loop body");
        dump(0x3C0, 0x3E0, "IMEM back-branch");
        dump(0x160, 0x1C0, "IMEM dispatch fn");
        dump(0x380, 0x3C0, "IMEM dispatch top");
        dump(0x060, 0x100, "IMEM 060-100 (entry+poll)");
        dump(0x740, 0x790, "IMEM 740-790 (BREAK area)");
        // Full IMEM dump in 16-instruction chunks
        for base in (0x000..0x1000).step_by(0x40) {
            dump(base, base + 0x40, &format!("IMEM[{:03X}]", base));
        }
    }
    // Vector register file
    {
        eprintln!("[headless] FINAL RSP vec regs:");
        for vr in 0..32usize {
            let v = &m.bus.rsp_vu.vr[vr];
            eprintln!("  v{:02} = {:04X} {:04X} {:04X} {:04X} {:04X} {:04X} {:04X} {:04X}",
                vr, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
        }
        eprintln!("  vco=0x{:04X} vcc=0x{:04X} vce=0x{:02X}",
            m.bus.rsp_vu.vco, m.bus.rsp_vu.vcc, m.bus.rsp_vu.vce);
        eprintln!("  acc_hi: {:04X} {:04X} {:04X} {:04X} {:04X} {:04X} {:04X} {:04X}",
            m.bus.rsp_vu.acc_hi[0], m.bus.rsp_vu.acc_hi[1], m.bus.rsp_vu.acc_hi[2], m.bus.rsp_vu.acc_hi[3],
            m.bus.rsp_vu.acc_hi[4], m.bus.rsp_vu.acc_hi[5], m.bus.rsp_vu.acc_hi[6], m.bus.rsp_vu.acc_hi[7]);
        eprintln!("  acc_md: {:04X} {:04X} {:04X} {:04X} {:04X} {:04X} {:04X} {:04X}",
            m.bus.rsp_vu.acc_md[0], m.bus.rsp_vu.acc_md[1], m.bus.rsp_vu.acc_md[2], m.bus.rsp_vu.acc_md[3],
            m.bus.rsp_vu.acc_md[4], m.bus.rsp_vu.acc_md[5], m.bus.rsp_vu.acc_md[6], m.bus.rsp_vu.acc_md[7]);
        eprintln!("  acc_lo: {:04X} {:04X} {:04X} {:04X} {:04X} {:04X} {:04X} {:04X}",
            m.bus.rsp_vu.acc_lo[0], m.bus.rsp_vu.acc_lo[1], m.bus.rsp_vu.acc_lo[2], m.bus.rsp_vu.acc_lo[3],
            m.bus.rsp_vu.acc_lo[4], m.bus.rsp_vu.acc_lo[5], m.bus.rsp_vu.acc_lo[6], m.bus.rsp_vu.acc_lo[7]);
    }
    // Full DMEM dump 0x000..0x1000
    {
        for base in (0x000..0x1000).step_by(0x20) {
            let mut s = String::new();
            for i in 0..0x20 {
                if i % 4 == 0 && i > 0 { s.push(' '); }
                s.push_str(&format!("{:02X}", m.bus.rsp_dmem[base + i]));
            }
            eprintln!("[headless] DMEM[{:03X}]: {}", base, s);
        }
    }
    // Full IMEM dump 0x000..0x1000 at end-of-run
    {
        for base in (0x000..0x1000).step_by(0x20) {
            let mut s = String::new();
            for i in (0..0x20).step_by(4) {
                if i > 0 { s.push(' '); }
                s.push_str(&format!("{:08X}",
                    u32::from_be_bytes([m.bus.rsp_imem[base+i], m.bus.rsp_imem[base+i+1],
                                         m.bus.rsp_imem[base+i+2], m.bus.rsp_imem[base+i+3]])));
            }
            eprintln!("[headless] IMEM[{:03X}]: {}", base, s);
        }
    }
    // Final scalar regs of the RSP
    {
        let names = ["zr","at","v0","v1","a0","a1","a2","a3",
                     "t0","t1","t2","t3","t4","t5","t6","t7",
                     "s0","s1","s2","s3","s4","s5","s6","s7",
                     "t8","t9","k0","k1","gp","sp","fp","ra"];
        let mut parts = Vec::new();
        for i in 0..32usize {
            parts.push(format!("{}=0x{:08X}", names[i], m.bus.rsp_scalar_regs[i]));
        }
        eprintln!("[headless] FINAL RSP regs: {}", parts.join(" "));
    }
    // DMEM dispatch table region (where lh $v0, 0xC4($v0) reads handler addrs)
    {
        let mut chunks = Vec::new();
        for off in (0xC0..0x150).step_by(2) {
            let v = u16::from_be_bytes([m.bus.rsp_dmem[off], m.bus.rsp_dmem[off+1]]);
            chunks.push(format!("{:03X}:{:04X}", off, v));
        }
        eprintln!("[headless] FINAL DMEM[C0..150] hw: {}", chunks.join(" "));
    }
    // Search for code referencing gameThread queue 0x80206D60 across full RDRAM.
    // Pattern: lui $reg, 0x8020 (3C..8020) followed within ~16 bytes by an instruction
    // with low immediate 0x6D60. Also search for the literal pointer 0x80206D60 in data.
    eprintln!("[headless] searching for refs to 0x80206D60...");
    let mut found = 0;
    for a in (0x80000000u32..0x80400000).step_by(4) {
        let pa = (a & 0x1FFF_FFFF) as usize;
        if pa + 20 > m.bus.rdram.data.len() { break; }
        let w = u32::from_be_bytes([m.bus.rdram.data[pa], m.bus.rdram.data[pa+1], m.bus.rdram.data[pa+2], m.bus.rdram.data[pa+3]]);
        if (w & 0xFC00_0000) == 0x3C00_0000 {
            let imm = (w & 0xFFFF) as u16;
            if imm == 0x8020 {
                for j in 1..8 {
                    let pa2 = pa + j*4;
                    if pa2 + 4 > m.bus.rdram.data.len() { break; }
                    let w2 = u32::from_be_bytes([m.bus.rdram.data[pa2], m.bus.rdram.data[pa2+1], m.bus.rdram.data[pa2+2], m.bus.rdram.data[pa2+3]]);
                    let imm2 = (w2 & 0xFFFF) as u16;
                    if imm2 == 0x6D60 {
                        eprintln!("  ref @0x{:08X}: lui {:08X} ; +{} {:08X}", a, w, j*4, w2);
                        found += 1;
                        if found > 30 { break; }
                    }
                }
            }
        }
        if found > 30 { break; }
    }
    eprintln!("[headless] {} code refs found", found);
    // Search RDRAM for literal pointer 0x80206D60 stored as a word (e.g. in tables/structs)
    eprintln!("[headless] searching for literal pointer 0x80206D60 in RDRAM...");
    let needle: [u8; 4] = 0x80206D60u32.to_be_bytes();
    let mut lit = 0;
    let lim = m.bus.rdram.data.len().min(0x00400000) - 4;
    let mut i = 0usize;
    while i < lim {
        if m.bus.rdram.data[i] == needle[0]
            && m.bus.rdram.data[i+1] == needle[1]
            && m.bus.rdram.data[i+2] == needle[2]
            && m.bus.rdram.data[i+3] == needle[3]
        {
            eprintln!("  literal @0x{:08X}", 0x8000_0000 + i);
            lit += 1;
            if lit > 30 { break; }
        }
        i += 4;
    }
    eprintln!("[headless] {} literal refs found", lit);
    // Dump 0x80206D00..0x80206E00 to see what surrounds the queue
    // Dump 0x80322B00..0x80322E00 to inspect the function calling osSendMesg
    // Dump __osDispatchEvent at 0x80327B98
    eprintln!("[headless] dump 0x80327B80..0x80327D00 (__osDispatchEvent):");
    for off in (0x00327B80usize..0x00327D00).step_by(16) {
        if off + 16 > m.bus.rdram.data.len() { break; }
        let mut bs = String::new();
        for k in 0..4 {
            let w = u32::from_be_bytes([
                m.bus.rdram.data[off+k*4],
                m.bus.rdram.data[off+k*4+1],
                m.bus.rdram.data[off+k*4+2],
                m.bus.rdram.data[off+k*4+3],
            ]);
            bs.push_str(&format!("{:08X} ", w));
        }
        eprintln!("  0x{:08X}: {}", 0x8000_0000 + off, bs);
    }
    // Read __osViNext = *(0x80335A20), then dump the struct it points to
    let ptr_paddr = 0x00335A20usize;
    if ptr_paddr + 4 <= m.bus.rdram.data.len() {
        let vinext = u32::from_be_bytes([
            m.bus.rdram.data[ptr_paddr], m.bus.rdram.data[ptr_paddr+1],
            m.bus.rdram.data[ptr_paddr+2], m.bus.rdram.data[ptr_paddr+3],
        ]);
        eprintln!("[headless] __osViNext (*0x80335A20) = 0x{:08X}", vinext);
        // Also read __osViCurr at 0x80335A24
        let curr = u32::from_be_bytes([
            m.bus.rdram.data[ptr_paddr+4], m.bus.rdram.data[ptr_paddr+5],
            m.bus.rdram.data[ptr_paddr+6], m.bus.rdram.data[ptr_paddr+7],
        ]);
        eprintln!("[headless] __osViCurr (*0x80335A24) = 0x{:08X}", curr);
        // Read retraceCount at 0x80365E6C
        if 0x00365E6C + 2 <= m.bus.rdram.data.len() {
            let rc = u16::from_be_bytes([m.bus.rdram.data[0x00365E6C], m.bus.rdram.data[0x00365E6C+1]]);
            eprintln!("[headless] retraceCount @0x80365E6C = {}", rc);
        }
        // Dump the struct __osViNext points to (assume <= 0x40 bytes)
        if vinext >= 0x80000000 && vinext < 0x80400000 {
            let svp = (vinext & 0x1FFF_FFFF) as usize;
            eprintln!("[headless] dump __osViNext struct at 0x{:08X}:", vinext);
            for off in (svp..(svp+0x40)).step_by(16) {
                if off + 16 > m.bus.rdram.data.len() { break; }
                let mut bs = String::new();
                for k in 0..4 {
                    let w = u32::from_be_bytes([
                        m.bus.rdram.data[off+k*4],
                        m.bus.rdram.data[off+k*4+1],
                        m.bus.rdram.data[off+k*4+2],
                        m.bus.rdram.data[off+k*4+3],
                    ]);
                    bs.push_str(&format!("{:08X} ", w));
                }
                eprintln!("  0x{:08X}: {}", 0x8000_0000 + off, bs);
            }
            // mq is at offset 0x10
            let mq_off = svp + 0x10;
            if mq_off + 8 <= m.bus.rdram.data.len() {
                let mq = u32::from_be_bytes([
                    m.bus.rdram.data[mq_off], m.bus.rdram.data[mq_off+1],
                    m.bus.rdram.data[mq_off+2], m.bus.rdram.data[mq_off+3],
                ]);
                let msg = u32::from_be_bytes([
                    m.bus.rdram.data[mq_off+4], m.bus.rdram.data[mq_off+5],
                    m.bus.rdram.data[mq_off+6], m.bus.rdram.data[mq_off+7],
                ]);
                eprintln!("[headless] __osViNext->mq = 0x{:08X}, ->msg = 0x{:08X}", mq, msg);
            }
        }
    }
    // Dump gameThread @0x8033AA90 context (look for saved PC/RA)
    eprintln!("[headless] dump gameThread @0x8033AA90 (thread + context):");
    for off in (0x0033AA90usize..0x0033ABE0).step_by(16) {
        if off + 16 > m.bus.rdram.data.len() { break; }
        let mut bs = String::new();
        for k in 0..4 {
            let w = u32::from_be_bytes([
                m.bus.rdram.data[off+k*4],
                m.bus.rdram.data[off+k*4+1],
                m.bus.rdram.data[off+k*4+2],
                m.bus.rdram.data[off+k*4+3],
            ]);
            bs.push_str(&format!("{:08X} ", w));
        }
        eprintln!("  0x{:08X}: {}", 0x8000_0000 + off, bs);
    }
    // Decode gameThread context: at offset 0x20 starts __OSThreadContext.
    // Layout: at1,v0..ra (4 bytes each in 32-bit, 8 in 64-bit). For libultra 64-bit:
    // 0x20 = at, then v0..a3 (7 regs), then t0..t9 (10), then s0..s7 (8), then t8 t9 (2),
    // gp sp s8 ra (4), lo hi (2) = 33 regs of u64; at offset 0x128 = sr u32, 0x12C = pc u32
    let game_pc_addr = 0x0033AA90usize + 0x12C;
    let game_ra_addr = 0x0033AA90usize + 0x108 + 0x14; // approx
    if game_pc_addr + 4 <= m.bus.rdram.data.len() {
        let pc = u32::from_be_bytes([
            m.bus.rdram.data[game_pc_addr], m.bus.rdram.data[game_pc_addr+1],
            m.bus.rdram.data[game_pc_addr+2], m.bus.rdram.data[game_pc_addr+3],
        ]);
        eprintln!("[headless] gameThread saved PC at 0x{:08X} = 0x{:08X}", 0x8000_0000 + game_pc_addr, pc);
    }
    // Scan gameThread context for valid code addresses (0x80200000-0x80400000)
    eprintln!("[headless] scanning gameThread context for code addresses:");
    for off in (0x20usize..0x140).step_by(4) {
        let a = 0x0033AA90 + off;
        if a + 4 <= m.bus.rdram.data.len() {
            let v = u32::from_be_bytes([
                m.bus.rdram.data[a], m.bus.rdram.data[a+1],
                m.bus.rdram.data[a+2], m.bus.rdram.data[a+3],
            ]);
            if v >= 0x80200000 && v < 0x80400000 {
                eprintln!("  offset +0x{:03X}: 0x{:08X}", off, v);
            }
        }
    }
    // Dump scheduler thread struct at 0x80365E70 (first 0x40 bytes)
    eprintln!("[headless] dump scheduler thread @0x80365E70:");
    for off in (0x00365E70usize..0x00365EB0).step_by(16) {
        if off + 16 > m.bus.rdram.data.len() { break; }
        let mut bs = String::new();
        for k in 0..4 {
            let w = u32::from_be_bytes([
                m.bus.rdram.data[off+k*4],
                m.bus.rdram.data[off+k*4+1],
                m.bus.rdram.data[off+k*4+2],
                m.bus.rdram.data[off+k*4+3],
            ]);
            bs.push_str(&format!("{:08X} ", w));
        }
        eprintln!("  0x{:08X}: {}", 0x8000_0000 + off, bs);
    }
    // Dump thread5 stack region (saved sp = 0x80206CB0) - look for real caller RA
    eprintln!("[headless] dump thread5 stack 0x80206C00..0x80206E00:");
    for off in (0x00206C00usize..0x00206E00).step_by(16) {
        if off + 16 > m.bus.rdram.data.len() { break; }
        let mut bs = String::new();
        for k in 0..4 {
            let w = u32::from_be_bytes([
                m.bus.rdram.data[off+k*4],
                m.bus.rdram.data[off+k*4+1],
                m.bus.rdram.data[off+k*4+2],
                m.bus.rdram.data[off+k*4+3],
            ]);
            bs.push_str(&format!("{:08X} ", w));
        }
        eprintln!("  0x{:08X}: {}", 0x8000_0000 + off, bs);
    }
    // Dump 0x8033A8E0..0x8033AA10 (full thread struct + context)
    eprintln!("[headless] dump 0x8033A8E0..0x8033AA10 (thread #3 prio 100):");
    for off in (0x0033A8E0usize..0x0033AA10).step_by(16) {
        if off + 16 > m.bus.rdram.data.len() { break; }
        let mut bs = String::new();
        for k in 0..4 {
            let w = u32::from_be_bytes([
                m.bus.rdram.data[off+k*4],
                m.bus.rdram.data[off+k*4+1],
                m.bus.rdram.data[off+k*4+2],
                m.bus.rdram.data[off+k*4+3],
            ]);
            bs.push_str(&format!("{:08X} ", w));
        }
        eprintln!("  0x{:08X}: {}", 0x8000_0000 + off, bs);
    }
    // Dump 0x8033AC40..0x8033AD70 (full thread struct + context) — REAL scheduler thread (id=4)
    eprintln!("[headless] dump 0x8033AC40..0x8033AD70 (thread id=4 — SM64 scheduler):");
    for off in (0x0033AC40usize..0x0033AD70).step_by(16) {
        if off + 16 > m.bus.rdram.data.len() { break; }
        let mut bs = String::new();
        for k in 0..4 {
            let w = u32::from_be_bytes([
                m.bus.rdram.data[off+k*4],
                m.bus.rdram.data[off+k*4+1],
                m.bus.rdram.data[off+k*4+2],
                m.bus.rdram.data[off+k*4+3],
            ]);
            bs.push_str(&format!("{:08X} ", w));
        }
        eprintln!("  0x{:08X}: {}", 0x8000_0000 + off, bs);
    }
    // Decode each candidate scheduler/game thread (OSThread layout):
    //   0x00 next, 0x04 prio, 0x08 queue, 0x0C tlnext, 0x10/0x12 state/flags,
    //   0x14 id, 0x18 fp, 0x1C thprof, 0x20 OSThreadContext
    //   ctx layout: 31 u64 regs (at..ra,lo,hi) = 248B, then sr u32, pc u32 ...
    //   so saved-pc offset = 0x20 + 248 + 4 = 0x11C; sp = 0x20 + 24*8 = 0xB0 (sp is in slot index 24)
    let decode_thread = |m: &Machine, base_va: u32, label: &str| {
        let pa = (base_va & 0x1FFF_FFFF) as usize;
        if pa + 0x130 > m.bus.rdram.data.len() { return; }
        let r32 = |o: usize| u32::from_be_bytes([
            m.bus.rdram.data[pa+o], m.bus.rdram.data[pa+o+1],
            m.bus.rdram.data[pa+o+2], m.bus.rdram.data[pa+o+3],
        ]);
        let prio = r32(0x04);
        let queue = r32(0x08);
        let state_flags = r32(0x10);
        let id = r32(0x14);
        let saved_sr = r32(0x118);
        let saved_pc = r32(0x11C);
        // OSThreadContext starts at OSThread+0x20. Reg layout (each u64):
        //   at,v0,v1,a0,a1,a2,a3 | t0..t7 | s0..s7 | t8,t9 | gp,sp,s8 | ra | lo,hi | sr,pc,...
        // Context offsets: gp=0xC8, sp=0xD0, s8=0xD8, ra=0xE0, lo=0xE8, hi=0xF0, sr=0xF8, pc=0xFC.
        // For BE u64, the meaningful u32 lo half is at offset+4. Add 0x20 for OSThread base.
        let saved_sp = r32(0x20 + 0xD0 + 4);  // 0xF4
        let saved_ra = r32(0x20 + 0xE0 + 4);  // 0x104
        eprintln!(
            "  THREAD[{}] @0x{:08X}: id={} prio=0x{:08X} state/flg=0x{:08X} queue=0x{:08X} pc=0x{:08X} sp=0x{:08X} ra=0x{:08X} sr=0x{:08X}",
            label, base_va, id, prio, state_flags, queue, saved_pc, saved_sp, saved_ra, saved_sr,
        );
    };
    decode_thread(&m, 0x8033A8E0, "T#3 audio");
    decode_thread(&m, 0x8033AC40, "T#4 scheduler");
    decode_thread(&m, 0x80365E70, "T@65E70 (SCHEDQ waiter, prio 150)");
    decode_thread(&m, 0x8033AA90, "gameThread @AA90");
    // Dump panic-related strings at 0x801B8460, 0x801B9150, 0x801B89B0 (from stack trace).
    eprintln!("[headless] related panic strings:");
    for addr in [0x801B846Au32, 0x801B915Bu32, 0x801B89B2u32, 0x801B8400u32, 0x801B9100u32, 0x801B8980u32] {
        let pa = (addr & 0x1FFFFFFF) as usize;
        if pa + 128 <= m.bus.rdram.data.len() {
            let slice = &m.bus.rdram.data[pa..pa + 128];
            let end = slice.iter().position(|&b| b == 0 || b == 0xA).unwrap_or(128);
            let s = String::from_utf8_lossy(&slice[..end]);
            eprintln!("  0x{:08X}: \"{}\"", addr, s);
        }
    }
    // Dump additional code regions for tracing the panic call chain.
    // fatal_printf wrapper prologue is before 0x8018D200.
    // Upstream caller addresses from fatal_printf stack frame: 0x8019504C, 0x8017F30C, 0x80181CE8
    for (start, end, label) in [
        (0x8018D180u32, 0x8018D200u32, "fatal_printf wrapper prologue"),
        (0x80194F00u32, 0x80195100u32, "caller fn containing 0x80195048"),
        (0x8017F280u32, 0x8017F360u32, "upstream caller 0x8017F30C area"),
        (0x80181C80u32, 0x80181D20u32, "upstream caller 0x80181CE8 area"),
        (0x8017E580u32, 0x8017E6A0u32, "L2 caller around 0x8017E674"),
        (0x80196300u32, 0x80196600u32, "matrix writer (0x801963CC/801964A8)"),
        (0x80191900u32, 0x80191980u32, "copy caller A (ra=0x80191928)"),
        (0x80197080u32, 0x801970E0u32, "copy caller B (ra=0x801970B4)"),
        (0x80197000u32, 0x80197080u32, "copy caller B prologue area"),
        (0x80196600u32, 0x80196800u32, "code 0x80196600..0x80196800"),
        (0x80194B00u32, 0x80194D00u32, "caller of mul (ra=0x80194CC0)"),
        (0x80196100u32, 0x80196200u32, "quat_to_matrix at 0x80196114"),
        (0x80196200u32, 0x80196400u32, "rot-builder at 0x80196334"),
        (0x80196000u32, 0x80196100u32, "before quat_to_matrix"),
        (0x8019B400u32, 0x8019B600u32, "sin/cos at 0x8019B41C"),
        (0x80194900u32, 0x80194B00u32, "caller of mul prologue area"),
        (0x80196800u32, 0x80196A00u32, "mul func prologue"),
        (0x80196A00u32, 0x80196C00u32, "mul func body 1"),
        (0x80196C00u32, 0x80196E00u32, "mul func body 2"),
        (0x80196E00u32, 0x80197000u32, "mul func body 3"),
    ] {
        eprintln!("[headless] dump 0x{:08X}..0x{:08X} ({}):", start, end, label);
        for off_va in (start..end).step_by(16) {
            let pa = (off_va & 0x1FFFFFFF) as usize;
            if pa + 16 > m.bus.rdram.data.len() { break; }
            let w0 = u32::from_be_bytes([m.bus.rdram.data[pa], m.bus.rdram.data[pa+1], m.bus.rdram.data[pa+2], m.bus.rdram.data[pa+3]]);
            let w1 = u32::from_be_bytes([m.bus.rdram.data[pa+4], m.bus.rdram.data[pa+5], m.bus.rdram.data[pa+6], m.bus.rdram.data[pa+7]]);
            let w2 = u32::from_be_bytes([m.bus.rdram.data[pa+8], m.bus.rdram.data[pa+9], m.bus.rdram.data[pa+10], m.bus.rdram.data[pa+11]]);
            let w3 = u32::from_be_bytes([m.bus.rdram.data[pa+12], m.bus.rdram.data[pa+13], m.bus.rdram.data[pa+14], m.bus.rdram.data[pa+15]]);
            eprintln!("  0x{:08X}: {:08X} {:08X} {:08X} {:08X}", off_va, w0, w1, w2, w3);
        }
    }
    // Dump code around 0x8018D200..0x8018D560 (caller of exit() + gd_error).
    eprintln!("[headless] dump 0x8018D200..0x8018D560 (exit caller):");
    for off_va in (0x8018D200u32..0x8018D560).step_by(16) {
        let pa = (off_va & 0x1FFFFFFF) as usize;
        if pa + 16 > m.bus.rdram.data.len() { break; }
        let w0 = u32::from_be_bytes([m.bus.rdram.data[pa], m.bus.rdram.data[pa+1], m.bus.rdram.data[pa+2], m.bus.rdram.data[pa+3]]);
        let w1 = u32::from_be_bytes([m.bus.rdram.data[pa+4], m.bus.rdram.data[pa+5], m.bus.rdram.data[pa+6], m.bus.rdram.data[pa+7]]);
        let w2 = u32::from_be_bytes([m.bus.rdram.data[pa+8], m.bus.rdram.data[pa+9], m.bus.rdram.data[pa+10], m.bus.rdram.data[pa+11]]);
        let w3 = u32::from_be_bytes([m.bus.rdram.data[pa+12], m.bus.rdram.data[pa+13], m.bus.rdram.data[pa+14], m.bus.rdram.data[pa+15]]);
        eprintln!("  0x{:08X}: {:08X} {:08X} {:08X} {:08X}", off_va, w0, w1, w2, w3);
    }
    // Dump the format string region around 0x801B8990 (arg to wrapper at jal 0x18D1F8).
    eprintln!("[headless] format string region 0x801B8980..0x801B8A60:");
    {
        let pa = 0x001B8980usize;
        if pa + 0xE0 <= m.bus.rdram.data.len() {
            for row in 0..0xE0/16 {
                let base = pa + row * 16;
                let bytes = &m.bus.rdram.data[base..base + 16];
                let hex: Vec<String> = bytes.iter().map(|b| format!("{:02X}", b)).collect();
                let ascii: String = bytes.iter()
                    .map(|&b| if (32..127).contains(&b) { b as char } else { '.' })
                    .collect();
                eprintln!("  0x{:08X}: {} | {}", 0x801B8980 + row*16, hex.join(" "), ascii);
            }
        }
    }
    // Dump the panic string at 0x801B8ED8 (passed to the panic function by 0x8019BB18+).
    eprintln!("[headless] panic string region 0x801B8ED8..0x801B8F80:");
    {
        let pa = 0x001B8ED8usize;
        if pa + 0xA8 <= m.bus.rdram.data.len() {
            for row in 0..0xA8/16 {
                let base = pa + row * 16;
                let bytes = &m.bus.rdram.data[base..base + 16];
                let hex: Vec<String> = bytes.iter().map(|b| format!("{:02X}", b)).collect();
                let ascii: String = bytes.iter()
                    .map(|&b| if (32..127).contains(&b) { b as char } else { '.' })
                    .collect();
                eprintln!("  0x{:08X}: {} | {}", 0x80000000u32 + base as u32, hex.join(" "), ascii);
            }
        }
    }
    // Dump 0x8019BB00..0x8019BB80 — gameThread's current PC area (likely the stall site).
    eprintln!("[headless] dump 0x8019BB00..0x8019BB80 (gameThread PC area):");
    for off_va in (0x8019BB00u32..0x8019BB80).step_by(16) {
        let pa = (off_va & 0x1FFF_FFFF) as usize;
        if pa + 16 > m.bus.rdram.data.len() { break; }
        let w0 = u32::from_be_bytes([m.bus.rdram.data[pa], m.bus.rdram.data[pa+1], m.bus.rdram.data[pa+2], m.bus.rdram.data[pa+3]]);
        let w1 = u32::from_be_bytes([m.bus.rdram.data[pa+4], m.bus.rdram.data[pa+5], m.bus.rdram.data[pa+6], m.bus.rdram.data[pa+7]]);
        let w2 = u32::from_be_bytes([m.bus.rdram.data[pa+8], m.bus.rdram.data[pa+9], m.bus.rdram.data[pa+10], m.bus.rdram.data[pa+11]]);
        let w3 = u32::from_be_bytes([m.bus.rdram.data[pa+12], m.bus.rdram.data[pa+13], m.bus.rdram.data[pa+14], m.bus.rdram.data[pa+15]]);
        eprintln!("  0x{:08X}: {:08X} {:08X} {:08X} {:08X}", off_va, w0, w1, w2, w3);
    }
    // Dump context window around 0x80365E70 raw to inspect
    eprintln!("[headless] dump 0x80365E00..0x80366000 (thread @0x80365E70):");
    for off2 in (0x00365E00usize..0x00366000).step_by(16) {
        if off2 + 16 > m.bus.rdram.data.len() { break; }
        let mut bs = String::new();
        for k in 0..4 {
            let w = u32::from_be_bytes([
                m.bus.rdram.data[off2+k*4],
                m.bus.rdram.data[off2+k*4+1],
                m.bus.rdram.data[off2+k*4+2],
                m.bus.rdram.data[off2+k*4+3],
            ]);
            bs.push_str(&format!("{:08X} ", w));
        }
        eprintln!("  0x{:08X}: {}", 0x8000_0000 + off2, bs);
    }
    // Scan 0x8033A000..0x8033C000 for OSThread structs by signature: state in {0x0001,0x0002,0x0004,0x0008}
    // (RUNNABLE/RUNNING/WAITING/STOPPED) and a plausible id (1..32) at +0x14.
    eprintln!("[headless] scan 0x8033A000..0x8033C000 for OSThread structs:");
    let mut off = 0x0033A000usize;
    while off + 0x130 <= m.bus.rdram.data.len() && off < 0x0033C000 {
        let r32 = |o: usize| u32::from_be_bytes([
            m.bus.rdram.data[o], m.bus.rdram.data[o+1],
            m.bus.rdram.data[o+2], m.bus.rdram.data[o+3],
        ]);
        let state_flg = r32(off + 0x10);
        let id = r32(off + 0x14);
        let prio = r32(off + 0x04);
        let state = state_flg >> 16; // upper half is state field
        // Heuristic: state in {1,2,4,8}, id in 1..32, prio in 0..255
        if matches!(state, 1|2|4|8) && id >= 1 && id <= 32 && prio < 256 {
            let queue = r32(off + 0x08);
            let saved_pc = r32(off + 0x11C);
            let saved_sp = r32(off + 0x20 + 0xD0 + 4);
            let saved_ra = r32(off + 0x20 + 0xE0 + 4);
            eprintln!(
                "  T@0x{:08X}: id={} prio={} state=0x{:04X} queue=0x{:08X} pc=0x{:08X} sp=0x{:08X} ra=0x{:08X}",
                0x8000_0000 + off, id, prio, state, queue, saved_pc, saved_sp, saved_ra,
            );
            off += 0x120;
            continue;
        }
        off += 8;
    }
    // Walk each blocked thread's stack from saved sp, dump 0x80 bytes,
    // and highlight any words that look like return addresses (0x802xxxxx
    // / 0x803xxxxx that aren't libultra's osRecv/Send region 0x80322800-0x80322900).
    for (label, sp_va) in [
        ("T#3 audio sp", 0x80202DA0u32),
        ("T#4 sched sp", 0x80204DA8u32),
        ("T#5 game sp", 0x80206D88u32),
    ] {
        let pa = (sp_va & 0x1FFF_FFFF) as usize;
        if pa + 0x100 > m.bus.rdram.data.len() { continue; }
        eprintln!("[headless] {} stack @0x{:08X}:", label, sp_va);
        let mut ras = Vec::new();
        for off3 in (pa..pa + 0x100).step_by(4) {
            let w = u32::from_be_bytes([
                m.bus.rdram.data[off3], m.bus.rdram.data[off3+1],
                m.bus.rdram.data[off3+2], m.bus.rdram.data[off3+3],
            ]);
            // Plausible code address in game/libultra space
            if (0x8020_0000..=0x803F_FFFF).contains(&w) {
                ras.push((0x8000_0000 + (off3 - 0) as u32 - (0 as u32) + ((sp_va & 0x1FFF_FFFF) - pa as u32) , w));
            }
        }
        // simpler: re-walk and just print positions/values
        for off3 in (0..0x100usize).step_by(4) {
            let abs = pa + off3;
            let w = u32::from_be_bytes([
                m.bus.rdram.data[abs], m.bus.rdram.data[abs+1],
                m.bus.rdram.data[abs+2], m.bus.rdram.data[abs+3],
            ]);
            if (0x8020_0000..=0x803F_FFFF).contains(&w) {
                let tag = if (0x8032_2800..=0x8032_2900).contains(&w) { "(osRecv/Send)" }
                          else if (0x8032_0000..=0x8032_FFFF).contains(&w) { "(libultra)" }
                          else { "(game/other)" };
                eprintln!("  +0x{:02X} @0x{:08X} = 0x{:08X} {}", off3, sp_va + off3 as u32, w, tag);
            }
        }
        let _ = ras;
    }
    // Disassemble (raw bytes) the code regions around the saved RAs we found
    let dump_region = |m: &Machine, base: u32, len: u32, label: &str| {
        eprintln!("[headless] code @0x{:08X}..0x{:08X} ({}):", base, base+len, label);
        for off3 in (base..base+len).step_by(16) {
            let pa = (off3 & 0x1FFF_FFFF) as usize;
            if pa + 16 > m.bus.rdram.data.len() { break; }
            let mut bs = String::new();
            for k in 0..4 {
                let w = u32::from_be_bytes([
                    m.bus.rdram.data[pa+k*4], m.bus.rdram.data[pa+k*4+1],
                    m.bus.rdram.data[pa+k*4+2], m.bus.rdram.data[pa+k*4+3],
                ]);
                bs.push_str(&format!("{:08X} ", w));
            }
            eprintln!("  0x{:08X}: {}", off3, bs);
        }
    };
    // Dump the SM64 scheduler dispatch jump table at 0x80335B60..0x80335B80.
    // The 5 entries map (msg-100) to one of 0x80246A9C..0x80246ADC.
    dump_region(&m, 0x80335B60, 0x40, "scheduler dispatch jump table");
    // Dump scheduler globals around 0x8032D560..0x8032D580
    dump_region(&m, 0x8032D560, 0x40, "scheduler globals 0x8032D560");
    dump_region(&m, 0x8033D560, 0x40, "scheduler globals 0x8033D560");
    dump_region(&m, 0x80323100, 0x100, "code around 0x803231C4 (SEND caller)");
    dump_region(&m, 0x80246300, 0x200, "thread3 sched loop 0x80246300..0x80246500");
    dump_region(&m, 0x80246500, 0x200, "thread3 sched loop 0x80246500..0x80246700");
    dump_region(&m, 0x80246700, 0x200, "thread3 sched loop 0x80246700..0x80246900");
    dump_region(&m, 0x80246900, 0x200, "thread3 sched loop 0x80246900..0x80246B00");
    dump_region(&m, 0x80246B00, 0x200, "thread3 sched loop 0x80246B00..0x80246D00");
    dump_region(&m, 0x80214700, 0x80, "T#5 game RA 0x80214738");
    dump_region(&m, 0x80248080, 0x80, "T#5 game RA 0x802480B8/0xA0");
    dump_region(&m, 0x80248BA0, 0x80, "T#5 game RA 0x80248BCC/0xE0");
    dump_region(&m, 0x80246700, 0xA0, "T#3 audio RA 0x80246714/A70/AD4");
    dump_region(&m, 0x80249560, 0x40, "T#4 sched RA 0x80249578");
    dump_region(&m, 0x80226BA0, 0x40, "T#4 sched RA 0x80226BA0/C4C");
    // Dump the area around 0x8033B140 (scheduler's wait queue)
    eprintln!("[headless] dump 0x8033B100..0x8033B200 (sched queue 0x8033B140):");
    for off2 in (0x0033B100usize..0x0033B200).step_by(16) {
        if off2 + 16 > m.bus.rdram.data.len() { break; }
        let mut bs = String::new();
        for k in 0..4 {
            let w = u32::from_be_bytes([
                m.bus.rdram.data[off2+k*4],
                m.bus.rdram.data[off2+k*4+1],
                m.bus.rdram.data[off2+k*4+2],
                m.bus.rdram.data[off2+k*4+3],
            ]);
            bs.push_str(&format!("{:08X} ", w));
        }
        eprintln!("  0x{:08X}: {}", 0x8000_0000 + off2, bs);
    }
    // Decode queue 0x8033B140 specifically: OSMesgQueue layout is
    //   0x00 mtqueue  0x04 fullqueue  0x08 valid  0x0C first  0x10 msgCount  0x14 msg
    {
        let qbase = 0x0033B140usize;
        if qbase + 24 <= m.bus.rdram.data.len() {
            let r32 = |o: usize| u32::from_be_bytes([m.bus.rdram.data[o], m.bus.rdram.data[o+1], m.bus.rdram.data[o+2], m.bus.rdram.data[o+3]]);
            eprintln!(
                "  Q@0x8033B140 (sched input): mt=0x{:08X} full=0x{:08X} valid={} first={} count={} msg*=0x{:08X}",
                r32(qbase), r32(qbase+4), r32(qbase+8), r32(qbase+12), r32(qbase+16), r32(qbase+20),
            );
        }
    }
    // Also decode 0x8033B028 (gameThread's done queue from prior notes)
    {
        let qbase = 0x0033B028usize;
        if qbase + 24 <= m.bus.rdram.data.len() {
            let r32 = |o: usize| u32::from_be_bytes([m.bus.rdram.data[o], m.bus.rdram.data[o+1], m.bus.rdram.data[o+2], m.bus.rdram.data[o+3]]);
            eprintln!(
                "  Q@0x8033B028 (gameThread done?): mt=0x{:08X} full=0x{:08X} valid={} first={} count={} msg*=0x{:08X}",
                r32(qbase), r32(qbase+4), r32(qbase+8), r32(qbase+12), r32(qbase+16), r32(qbase+20),
            );
        }
    }
    // Walk active thread list from __osActiveQueue or __osRunningThread chain
    // by following the `next` pointer at offset 0x00 starting from a known thread.
    eprintln!("[headless] walk thread list starting at 0x8033A8E0:");
    let mut cur = 0x8033A8E0u32;
    for _ in 0..16 {
        let pa = (cur & 0x1FFF_FFFF) as usize;
        if pa + 0x18 > m.bus.rdram.data.len() { break; }
        let next = u32::from_be_bytes([
            m.bus.rdram.data[pa], m.bus.rdram.data[pa+1],
            m.bus.rdram.data[pa+2], m.bus.rdram.data[pa+3],
        ]);
        let prio = u32::from_be_bytes([
            m.bus.rdram.data[pa+4], m.bus.rdram.data[pa+5],
            m.bus.rdram.data[pa+6], m.bus.rdram.data[pa+7],
        ]);
        let id = u32::from_be_bytes([
            m.bus.rdram.data[pa+0x14], m.bus.rdram.data[pa+0x15],
            m.bus.rdram.data[pa+0x16], m.bus.rdram.data[pa+0x17],
        ]);
        eprintln!("  thread @0x{:08X}: id={} prio=0x{:08X} next=0x{:08X}", cur, id, prio, next);
        if next == 0 || next == cur { break; }
        cur = next;
    }
    // Dump 0x8033ADE0..0x8033AF00 (scheduler queues area)
    eprintln!("[headless] dump 0x8033ADE0..0x8033AF00 (scheduler queues):");
    for off in (0x0033ADE0usize..0x0033AF00).step_by(16) {
        if off + 16 > m.bus.rdram.data.len() { break; }
        let mut bs = String::new();
        for k in 0..4 {
            let w = u32::from_be_bytes([
                m.bus.rdram.data[off+k*4],
                m.bus.rdram.data[off+k*4+1],
                m.bus.rdram.data[off+k*4+2],
                m.bus.rdram.data[off+k*4+3],
            ]);
            bs.push_str(&format!("{:08X} ", w));
        }
        eprintln!("  0x{:08X}: {}", 0x8000_0000 + off, bs);
    }
    // Decode each potential queue header
    for qbase in [0x0033ADF0usize, 0x0033AE08, 0x0033AE20] {
        if qbase + 24 <= m.bus.rdram.data.len() {
            let read32 = |o: usize| u32::from_be_bytes([m.bus.rdram.data[o], m.bus.rdram.data[o+1], m.bus.rdram.data[o+2], m.bus.rdram.data[o+3]]);
            let mt = read32(qbase);
            let full = read32(qbase+4);
            let valid = read32(qbase+8);
            let first = read32(qbase+12);
            let count = read32(qbase+16);
            let msgp = read32(qbase+20);
            eprintln!("  Q@0x{:08X}: mt=0x{:08X} full=0x{:08X} valid={} first={} count={} msg*=0x{:08X}",
                0x8000_0000 + qbase, mt, full, valid, first, count, msgp);
        }
    }
    // Dump 0x80328060..0x80328080 (helper that returns __osViNext or similar)
    eprintln!("[headless] dump 0x80328060..0x80328080:");
    for off in (0x00328060usize..0x00328080).step_by(16) {
        if off + 16 > m.bus.rdram.data.len() { break; }
        let mut bs = String::new();
        for k in 0..4 {
            let w = u32::from_be_bytes([
                m.bus.rdram.data[off+k*4],
                m.bus.rdram.data[off+k*4+1],
                m.bus.rdram.data[off+k*4+2],
                m.bus.rdram.data[off+k*4+3],
            ]);
            bs.push_str(&format!("{:08X} ", w));
        }
        eprintln!("  0x{:08X}: {}", 0x8000_0000 + off, bs);
    }
    // Dump 0x80323150..0x80323200 (__osViMgrThread main loop)
    eprintln!("[headless] dump 0x80323140..0x80323200 (__osViMgrThread main loop):");
    for off in (0x00323140usize..0x00323200).step_by(16) {
        if off + 16 > m.bus.rdram.data.len() { break; }
        let mut bs = String::new();
        for k in 0..4 {
            let w = u32::from_be_bytes([
                m.bus.rdram.data[off+k*4],
                m.bus.rdram.data[off+k*4+1],
                m.bus.rdram.data[off+k*4+2],
                m.bus.rdram.data[off+k*4+3],
            ]);
            bs.push_str(&format!("{:08X} ", w));
        }
        eprintln!("  0x{:08X}: {}", 0x8000_0000 + off, bs);
    }
    // Dump __OSEventTable at 0x80364BA0 (15 entries x 8 bytes = 0x78)
    eprintln!("[headless] dump 0x80364BA0..0x80364C20 (__OSEventTable):");
    for off in (0x00364BA0usize..0x00364C20).step_by(16) {
        if off + 16 > m.bus.rdram.data.len() { break; }
        let mut bs = String::new();
        for k in 0..4 {
            let w = u32::from_be_bytes([
                m.bus.rdram.data[off+k*4],
                m.bus.rdram.data[off+k*4+1],
                m.bus.rdram.data[off+k*4+2],
                m.bus.rdram.data[off+k*4+3],
            ]);
            bs.push_str(&format!("{:08X} ", w));
        }
        eprintln!("  0x{:08X}: {}", 0x8000_0000 + off, bs);
    }
    // Decode each event entry: idx -> (mq, msg)
    let event_names = ["SW1","SW2","CART","COUNTER","SP","SI","AI","VI","PI","DP","CPU_BREAK","SP_BREAK","FAULT","THREADSTATUS","PRENMI"];
    for idx in 0..15usize {
        let off = 0x00364BA0 + idx*8;
        if off + 8 > m.bus.rdram.data.len() { break; }
        let mq = u32::from_be_bytes([m.bus.rdram.data[off], m.bus.rdram.data[off+1], m.bus.rdram.data[off+2], m.bus.rdram.data[off+3]]);
        let msg = u32::from_be_bytes([m.bus.rdram.data[off+4], m.bus.rdram.data[off+5], m.bus.rdram.data[off+6], m.bus.rdram.data[off+7]]);
        let name = event_names.get(idx).copied().unwrap_or("?");
        eprintln!("  EVENT[{}]={}: mq=0x{:08X} msg=0x{:08X}", idx, name, mq, msg);
    }
    // Dump __osViIntr (acks VI at 0x803279A8 — function probably starts ~0x80327900)
    eprintln!("[headless] dump 0x80327900..0x80327B00 (__osViIntr region):");
    for off in (0x00327900usize..0x00327B00).step_by(16) {
        if off + 16 > m.bus.rdram.data.len() { break; }
        let mut bs = String::new();
        for k in 0..4 {
            let w = u32::from_be_bytes([
                m.bus.rdram.data[off+k*4],
                m.bus.rdram.data[off+k*4+1],
                m.bus.rdram.data[off+k*4+2],
                m.bus.rdram.data[off+k*4+3],
            ]);
            bs.push_str(&format!("{:08X} ", w));
        }
        eprintln!("  0x{:08X}: {}", 0x8000_0000 + off, bs);
    }
    eprintln!("[headless] MI_VI_ACK total = {}",
        ultrareality::machine::MI_VI_ACK_COUNT.load(std::sync::atomic::Ordering::Relaxed));
    // Find any JAL to 0x80324460 (the function we identified)
    eprintln!("[headless] searching for JAL to 0x80324460 (caller of timer-or-VI dispatch)...");
    // jal target = 0x0C000000 | ((0x80324460 >> 2) & 0x03FFFFFF) = 0x0C0C9118
    let jal_target = 0x0C0C9118u32;
    let mut jal_count = 0;
    for a in (0x80000000u32..0x80400000).step_by(4) {
        let pa = (a & 0x1FFF_FFFF) as usize;
        if pa + 4 > m.bus.rdram.data.len() { break; }
        let w = u32::from_be_bytes([
            m.bus.rdram.data[pa], m.bus.rdram.data[pa+1],
            m.bus.rdram.data[pa+2], m.bus.rdram.data[pa+3],
        ]);
        if w == jal_target {
            eprintln!("  jal @0x{:08X}", a);
            jal_count += 1;
            if jal_count > 20 { break; }
        }
    }
    eprintln!("[headless] {} jal-callers found", jal_count);
    // Dump 0x80324400..0x80324600 to find the function that calls osSendMesg with scheduler queue
    eprintln!("[headless] dump 0x80324400..0x80324600 (containing 0x80324548):");
    for off in (0x00324400usize..0x00324600).step_by(16) {
        if off + 16 > m.bus.rdram.data.len() { break; }
        let mut bs = String::new();
        for k in 0..4 {
            let w = u32::from_be_bytes([
                m.bus.rdram.data[off+k*4],
                m.bus.rdram.data[off+k*4+1],
                m.bus.rdram.data[off+k*4+2],
                m.bus.rdram.data[off+k*4+3],
            ]);
            bs.push_str(&format!("{:08X} ", w));
        }
        eprintln!("  0x{:08X}: {}", 0x8000_0000 + off, bs);
    }
    eprintln!("[headless] dump 0x80322B00..0x80322E00 (caller of osSendMesg):");
    for off in (0x00322B00usize..0x00322E00).step_by(16) {
        if off + 16 > m.bus.rdram.data.len() { break; }
        let mut bs = String::new();
        for k in 0..4 {
            let w = u32::from_be_bytes([
                m.bus.rdram.data[off+k*4],
                m.bus.rdram.data[off+k*4+1],
                m.bus.rdram.data[off+k*4+2],
                m.bus.rdram.data[off+k*4+3],
            ]);
            bs.push_str(&format!("{:08X} ", w));
        }
        eprintln!("  0x{:08X}: {}", 0x8000_0000 + off, bs);
    }
    eprintln!("[headless] dump 0x80206D00..0x80206E00:");
    for off in (0x00206D00usize..0x00206E00).step_by(16) {
        if off + 16 > m.bus.rdram.data.len() { break; }
        let mut bs = String::new();
        for k in 0..4 {
            let w = u32::from_be_bytes([
                m.bus.rdram.data[off+k*4],
                m.bus.rdram.data[off+k*4+1],
                m.bus.rdram.data[off+k*4+2],
                m.bus.rdram.data[off+k*4+3],
            ]);
            bs.push_str(&format!("{:08X} ", w));
        }
        eprintln!("  0x{:08X}: {}", 0x8000_0000 + off, bs);
    }

    // Dump 0x80323CC0..0x80323DC0 (function entry - the function gameThread is stuck inside)
    eprintln!("[headless] dump 0x80323CC0..0x80323DC0:");
    for off in (0x00323CC0usize..0x00323DC0).step_by(16) {
        if off + 16 > m.bus.rdram.data.len() { break; }
        let mut bs = String::new();
        for k in 0..4 {
            let w = u32::from_be_bytes([
                m.bus.rdram.data[off+k*4], m.bus.rdram.data[off+k*4+1],
                m.bus.rdram.data[off+k*4+2], m.bus.rdram.data[off+k*4+3],
            ]);
            bs.push_str(&format!("{:08X} ", w));
        }
        eprintln!("  0x{:08X}: {}", 0x8000_0000 + off, bs);
    }
    // Also dump 0x80322800..0x80322900 (the function called by gameThread, possibly osRecvMesg)
    eprintln!("[headless] dump 0x80322800..0x80322900:");
    for off in (0x00322800usize..0x00322900).step_by(16) {
        if off + 16 > m.bus.rdram.data.len() { break; }
        let mut bs = String::new();
        for k in 0..4 {
            let w = u32::from_be_bytes([
                m.bus.rdram.data[off+k*4], m.bus.rdram.data[off+k*4+1],
                m.bus.rdram.data[off+k*4+2], m.bus.rdram.data[off+k*4+3],
            ]);
            bs.push_str(&format!("{:08X} ", w));
        }
        eprintln!("  0x{:08X}: {}", 0x8000_0000 + off, bs);
    }
    // Dump 0x80323DC0..0x80323E60 (caller of osRecvMesg from gameThread stack: ra=0x80323E18)
    eprintln!("[headless] dump 0x80323DC0..0x80323E80 (libultra caller ra=0x80323E18):");
    for off in (0x00323DC0usize..0x00323E80).step_by(16) {
        if off + 16 > m.bus.rdram.data.len() { break; }
        let mut bs = String::new();
        for k in 0..4 {
            let w = u32::from_be_bytes([
                m.bus.rdram.data[off+k*4], m.bus.rdram.data[off+k*4+1],
                m.bus.rdram.data[off+k*4+2], m.bus.rdram.data[off+k*4+3],
            ]);
            bs.push_str(&format!("{:08X} ", w));
        }
        eprintln!("  0x{:08X}: {}", 0x8000_0000 + off, bs);
    }
    // Dump 0x80248800..0x802488C0 (sm64 caller ra=0x80248870)
    eprintln!("[headless] dump 0x80248800..0x802488C0 (sm64 caller ra=0x80248870):");
    for off in (0x00248800usize..0x002488C0).step_by(16) {
        if off + 16 > m.bus.rdram.data.len() { break; }
        let mut bs = String::new();
        for k in 0..4 {
            let w = u32::from_be_bytes([
                m.bus.rdram.data[off+k*4], m.bus.rdram.data[off+k*4+1],
                m.bus.rdram.data[off+k*4+2], m.bus.rdram.data[off+k*4+3],
            ]);
            bs.push_str(&format!("{:08X} ", w));
        }
        eprintln!("  0x{:08X}: {}", 0x8000_0000 + off, bs);
    }
    // Scan loaded code for references to address 0x80206D60 (gameThread queue)
    // Pattern: lui $X, 0x8020 followed within ~8 instructions by an immediate
    // instruction with low halfword 0x6D60 using same $X.
    eprintln!("[headless] scanning code for refs to 0x80206D60 (gameThread queue)...");
    {
        let read_w = |a: u32| -> u32 {
            let pa = (a & 0x1FFF_FFFF) as usize;
            if pa + 4 > m.bus.rdram.data.len() { return 0; }
            u32::from_be_bytes([
                m.bus.rdram.data[pa], m.bus.rdram.data[pa+1],
                m.bus.rdram.data[pa+2], m.bus.rdram.data[pa+3],
            ])
        };
        let mut hits = 0;
        let mut a = 0x80000000u32;
        while a < 0x80400000 {
            let w = read_w(a);
            // lui = opcode 0x0F (top 6 bits = 0b001111 -> 0x3C in top byte)
            if (w >> 26) == 0x0F && (w & 0xFFFF) == 0x8020 {
                let rt = (w >> 16) & 0x1F;
                // Look at next 8 instructions for any imm-form using same reg with imm 0x6D60
                for k in 1..=8u32 {
                    let w2 = read_w(a + k*4);
                    if (w2 & 0xFFFF) == 0x6D60 {
                        let rt2 = (w2 >> 16) & 0x1F;
                        let rs2 = (w2 >> 21) & 0x1F;
                        // matches if either rs2 or rt2 == rt (e.g., addiu/sw/lw use rs as base)
                        if rs2 == rt || rt2 == rt {
                            let op = w2 >> 26;
                            eprintln!("  ref @0x{:08X}: lui ${}, 0x8020 ; +0x{:X}: op={:02X} w={:08X}",
                                a, rt, k*4, op, w2);
                            hits += 1;
                            if hits > 30 { break; }
                            break;
                        }
                    }
                }
                if hits > 30 { break; }
            }
            a += 4;
        }
        eprintln!("[headless] found {} candidate refs to 0x80206D60", hits);
    }
    // Scan RDRAM for the data value 0x80206D60 (pointer stored somewhere)
    eprintln!("[headless] scanning RDRAM for stored value 0x80206D60...");
    {
        let mut hits = 0;
        let mut off = 0usize;
        while off + 4 <= m.bus.rdram.data.len() {
            let w = u32::from_be_bytes([
                m.bus.rdram.data[off], m.bus.rdram.data[off+1],
                m.bus.rdram.data[off+2], m.bus.rdram.data[off+3],
            ]);
            if w == 0x80206D60 {
                eprintln!("  found 0x80206D60 stored @0x{:08X}", 0x8000_0000 + off);
                hits += 1;
                if hits > 30 { break; }
            }
            off += 4;
        }
        eprintln!("[headless] {} stored copies of 0x80206D60", hits);
    }

    eprintln!("[headless] cop0.count={} compare={} cause=0x{:08X} status=0x{:08X}",
        m.cpu.cop0.count, m.cpu.cop0.compare, m.cpu.cop0.cause, m.cpu.cop0.status);
    eprintln!("[headless] CPU PC=0x{:016X}", m.cpu.pc);
    // Search code for MTC0 to Compare (r11): pattern (w & 0xFFE007FF) == 0x40805800
    eprintln!("[headless] scanning code for MTC0 to CP0 r11 (Compare)...");
    {
        let mut hits = 0;
        let mut a = 0x80000000u32;
        while a < 0x80400000 {
            let pa = (a & 0x1FFF_FFFF) as usize;
            if pa + 4 > m.bus.rdram.data.len() { break; }
            let w = u32::from_be_bytes([
                m.bus.rdram.data[pa], m.bus.rdram.data[pa+1],
                m.bus.rdram.data[pa+2], m.bus.rdram.data[pa+3],
            ]);
            if (w & 0xFFE0_F800) == 0x4080_5800 {
                let rt = (w >> 16) & 0x1F;
                eprintln!("  mtc0 ${}, $11 @0x{:08X} (w={:08X})", rt, a, w);
                hits += 1;
                if hits > 30 { break; }
            }
            a += 4;
        }
        eprintln!("[headless] {} MTC0-Compare sites", hits);
    }
    // Find callers of __osSetCompare at 0x80329780
    // jal target encoding: 0x0C000000 | ((0x80329780 >> 2) & 0x03FFFFFF) = 0x0C0CA5E0
    eprintln!("[headless] scanning callers of __osSetCompare (jal 0x0C0CA5E0)...");
    {
        let mut hits = 0;
        let mut a = 0x80000000u32;
        while a < 0x80400000 {
            let pa = (a & 0x1FFF_FFFF) as usize;
            if pa + 4 > m.bus.rdram.data.len() { break; }
            let w = u32::from_be_bytes([
                m.bus.rdram.data[pa], m.bus.rdram.data[pa+1],
                m.bus.rdram.data[pa+2], m.bus.rdram.data[pa+3],
            ]);
            if w == 0x0C0CA5E0 {
                eprintln!("  jal __osSetCompare @0x{:08X}", a);
                hits += 1;
                if hits > 30 { break; }
            }
            a += 4;
        }
        eprintln!("[headless] {} callers", hits);
    }
    // Find callers of __osSetTimerIntr at 0x80326064 (real entry, NOP at 0x80326060)
    // jal target = 0x0C000000 | (0x80326064 >> 2 & 0x03FFFFFF) = 0x0C0C9819
    eprintln!("[headless] scanning callers of __osSetTimerIntr (jal 0x0C0C9819)...");
    {
        let mut hits = 0;
        let mut a = 0x80000000u32;
        while a < 0x80400000 {
            let pa = (a & 0x1FFF_FFFF) as usize;
            if pa + 4 > m.bus.rdram.data.len() { break; }
            let w = u32::from_be_bytes([
                m.bus.rdram.data[pa], m.bus.rdram.data[pa+1],
                m.bus.rdram.data[pa+2], m.bus.rdram.data[pa+3],
            ]);
            if w == 0x0C0C9819 {
                eprintln!("  jal __osSetTimerIntr @0x{:08X}", a);
                hits += 1;
                if hits > 30 { break; }
            }
            a += 4;
        }
        eprintln!("[headless] {} __osSetTimerIntr callers", hits);
    }
    // Find callers of 0x80328A10 (the timer-setting function that calls __osSetTimerIntr)
    // jal target = 0x0C000000 | (0x80328A10 >> 2 & 0x03FFFFFF) = 0x0C0CA284
    eprintln!("[headless] scanning callers of osSetTimer@0x80328A10 (jal 0x0C0CA284)...");
    {
        let mut hits = 0;
        let mut a = 0x80000000u32;
        while a < 0x80400000 {
            let pa = (a & 0x1FFF_FFFF) as usize;
            if pa + 4 > m.bus.rdram.data.len() { break; }
            let w = u32::from_be_bytes([
                m.bus.rdram.data[pa], m.bus.rdram.data[pa+1],
                m.bus.rdram.data[pa+2], m.bus.rdram.data[pa+3],
            ]);
            if w == 0x0C0CA284 {
                eprintln!("  jal osSetTimer @0x{:08X}", a);
                hits += 1;
                if hits > 30 { break; }
            }
            a += 4;
        }
        eprintln!("[headless] {} osSetTimer callers", hits);
    }
    // Dump 0x80325EEC..0x80326100 (timer dispatch + first caller)
    eprintln!("[headless] dump 0x80325EE0..0x80326100:");
    for off in (0x00325EE0usize..0x00326100).step_by(16) {
        if off + 16 > m.bus.rdram.data.len() { break; }
        let mut bs = String::new();
        for k in 0..4 {
            let w = u32::from_be_bytes([
                m.bus.rdram.data[off+k*4], m.bus.rdram.data[off+k*4+1],
                m.bus.rdram.data[off+k*4+2], m.bus.rdram.data[off+k*4+3],
            ]);
            bs.push_str(&format!("{:08X} ", w));
        }
        eprintln!("  0x{:08X}: {}", 0x8000_0000 + off, bs);
    }
    // Dump 0x80326080..0x80326200 (second caller area)
    eprintln!("[headless] dump 0x80326080..0x80326200:");
    for off in (0x00326080usize..0x00326200).step_by(16) {
        if off + 16 > m.bus.rdram.data.len() { break; }
        let mut bs = String::new();
        for k in 0..4 {
            let w = u32::from_be_bytes([
                m.bus.rdram.data[off+k*4], m.bus.rdram.data[off+k*4+1],
                m.bus.rdram.data[off+k*4+2], m.bus.rdram.data[off+k*4+3],
            ]);
            bs.push_str(&format!("{:08X} ", w));
        }
        eprintln!("  0x{:08X}: {}", 0x8000_0000 + off, bs);
    }
    // Dump the three MTC0-Compare contexts
    for (label, base) in [("0x803278C0", 0x002778C0usize), ("0x80329760", 0x00329760), ("0x8032BA80", 0x0032BA80)] {
        eprintln!("[headless] dump {}..+0x40:", label);
        for off in (base..(base+0x40)).step_by(16) {
            if off + 16 > m.bus.rdram.data.len() { break; }
            let mut bs = String::new();
            for k in 0..4 {
                let w = u32::from_be_bytes([
                    m.bus.rdram.data[off+k*4], m.bus.rdram.data[off+k*4+1],
                    m.bus.rdram.data[off+k*4+2], m.bus.rdram.data[off+k*4+3],
                ]);
                bs.push_str(&format!("{:08X} ", w));
            }
            eprintln!("  0x{:08X}: {}", 0x8000_0000 + off, bs);
        }
    }
    // Dump 0x80323280..0x80323380 (type=0x0E handler — likely timer dispatch)
    eprintln!("[headless] dump 0x80323280..0x80323380 (type 0x0E handler):");
    for off in (0x00323280usize..0x00323380).step_by(16) {
        if off + 16 > m.bus.rdram.data.len() { break; }
        let mut bs = String::new();
        for k in 0..4 {
            let w = u32::from_be_bytes([
                m.bus.rdram.data[off+k*4], m.bus.rdram.data[off+k*4+1],
                m.bus.rdram.data[off+k*4+2], m.bus.rdram.data[off+k*4+3],
            ]);
            bs.push_str(&format!("{:08X} ", w));
        }
        eprintln!("  0x{:08X}: {}", 0x8000_0000 + off, bs);
    }
    // Dump timer-related globals at 0x80335910
    eprintln!("[headless] timer globals at 0x80335910:");
    for addr in [0x80335910u32, 0x80335914, 0x80335918, 0x8033591C, 0x80335920] {
        let pa = (addr & 0x1FFF_FFFF) as usize;
        if pa + 4 <= m.bus.rdram.data.len() {
            let val = u32::from_be_bytes([
                m.bus.rdram.data[pa], m.bus.rdram.data[pa+1],
                m.bus.rdram.data[pa+2], m.bus.rdram.data[pa+3],
            ]);
            eprintln!("  0x{:08X}: 0x{:08X} ({})", addr, val, val);
        }
    }
    // Dump gameThread entry area at 0x802469B8 (from ERET #7)
    eprintln!("[headless] dump 0x80246980..0x80246A00 (gameThread entry area):");
    for off in (0x00246980usize..0x00246A00).step_by(16) {
        if off + 16 > m.bus.rdram.data.len() { break; }
        let mut bs = String::new();
        for k in 0..4 {
            let w = u32::from_be_bytes([
                m.bus.rdram.data[off+k*4], m.bus.rdram.data[off+k*4+1],
                m.bus.rdram.data[off+k*4+2], m.bus.rdram.data[off+k*4+3],
            ]);
            bs.push_str(&format!("{:08X} ", w));
        }
        eprintln!("  0x{:08X}: {}", 0x8000_0000 + off, bs);
    }
    // Dump game code area at 0x80370000 to check if loaded
    eprintln!("[headless] dump 0x80370000..0x80370040 (game code area):");
    for off in (0x00370000usize..0x00370040).step_by(16) {
        if off + 16 > m.bus.rdram.data.len() { break; }
        let mut bs = String::new();
        for k in 0..4 {
            let w = u32::from_be_bytes([
                m.bus.rdram.data[off+k*4], m.bus.rdram.data[off+k*4+1],
                m.bus.rdram.data[off+k*4+2], m.bus.rdram.data[off+k*4+3],
            ]);
            bs.push_str(&format!("{:08X} ", w));
        }
        let all_zero = (0..4).all(|k| {
            u32::from_be_bytes([
                m.bus.rdram.data[off+k*4], m.bus.rdram.data[off+k*4+1],
                m.bus.rdram.data[off+k*4+2], m.bus.rdram.data[off+k*4+3],
            ]) == 0
        });
        eprintln!("  0x{:08X}: {}{}", 0x8000_0000 + off, bs, if all_zero { "<-- NOT LOADED" } else { "" });
    }
    // Dump COUNTER msg struct at 0x80365E58 and VI msg at 0x80365E40
    for addr in [0x80365E40u32, 0x80365E58] {
        let pa = (addr & 0x1FFF_FFFF) as usize;
        if pa + 16 <= m.bus.rdram.data.len() {
            let mut bs = String::new();
            for k in 0..4 {
                let w = u32::from_be_bytes([
                    m.bus.rdram.data[pa+k*4], m.bus.rdram.data[pa+k*4+1],
                    m.bus.rdram.data[pa+k*4+2], m.bus.rdram.data[pa+k*4+3],
                ]);
                bs.push_str(&format!("{:08X} ", w));
            }
            eprintln!("[headless] msg@0x{:08X}: {}", addr, bs);
        }
    }

    // Dump the caller of __osSetTimerIntr at 0x80328AC0..0x80328B00
    eprintln!("[headless] dump 0x80328A80..0x80328B40 (caller of __osSetTimerIntr):");
    for off in (0x00328A80usize..0x00328B40).step_by(16) {
        if off + 16 > m.bus.rdram.data.len() { break; }
        let mut bs = String::new();
        for k in 0..4 {
            let w = u32::from_be_bytes([
                m.bus.rdram.data[off+k*4], m.bus.rdram.data[off+k*4+1],
                m.bus.rdram.data[off+k*4+2], m.bus.rdram.data[off+k*4+3],
            ]);
            bs.push_str(&format!("{:08X} ", w));
        }
        eprintln!("  0x{:08X}: {}", 0x8000_0000 + off, bs);
    }
    // Also dump the function entry point (look for addiu sp, sp, -N pattern)
    eprintln!("[headless] dump 0x80328A00..0x80328A80 (function start area):");
    for off in (0x00328A00usize..0x00328A80).step_by(16) {
        if off + 16 > m.bus.rdram.data.len() { break; }
        let mut bs = String::new();
        for k in 0..4 {
            let w = u32::from_be_bytes([
                m.bus.rdram.data[off+k*4], m.bus.rdram.data[off+k*4+1],
                m.bus.rdram.data[off+k*4+2], m.bus.rdram.data[off+k*4+3],
            ]);
            bs.push_str(&format!("{:08X} ", w));
        }
        eprintln!("  0x{:08X}: {}", 0x8000_0000 + off, bs);
    }

    // Decode gameThread queue pointer from struct
    let gt_pa = 0x0033AA90usize;
    let gt_queue = if gt_pa + 12 <= m.bus.rdram.data.len() {
        u32::from_be_bytes([m.bus.rdram.data[gt_pa+8], m.bus.rdram.data[gt_pa+9], m.bus.rdram.data[gt_pa+10], m.bus.rdram.data[gt_pa+11]])
    } else { 0 };
    eprintln!("[headless] gameThread.queue = 0x{:08X}", gt_queue);
    // Dump that queue
    let pa = (gt_queue & 0x1FFF_FFFF) as usize;
    if pa + 0x18 <= m.bus.rdram.data.len() && gt_queue >= 0x80000000 {
        let mtqueue = u32::from_be_bytes([m.bus.rdram.data[pa], m.bus.rdram.data[pa+1], m.bus.rdram.data[pa+2], m.bus.rdram.data[pa+3]]);
        let valid = u32::from_be_bytes([m.bus.rdram.data[pa+8], m.bus.rdram.data[pa+9], m.bus.rdram.data[pa+10], m.bus.rdram.data[pa+11]]);
        let first = u32::from_be_bytes([m.bus.rdram.data[pa+12], m.bus.rdram.data[pa+13], m.bus.rdram.data[pa+14], m.bus.rdram.data[pa+15]]);
        let msgcount = u32::from_be_bytes([m.bus.rdram.data[pa+16], m.bus.rdram.data[pa+17], m.bus.rdram.data[pa+18], m.bus.rdram.data[pa+19]]);
        let msgptr = u32::from_be_bytes([m.bus.rdram.data[pa+20], m.bus.rdram.data[pa+21], m.bus.rdram.data[pa+22], m.bus.rdram.data[pa+23]]);
        eprintln!("[headless] Q@0x{:08X} mtqueue=0x{:08X} valid={} first={} capacity={} msg=0x{:08X}",
            gt_queue, mtqueue, valid as i32, first, msgcount, msgptr);
    }
    let pa = 0x00365E10usize;
    if pa + 16 <= m.bus.rdram.data.len() {
        let valid = u32::from_be_bytes([m.bus.rdram.data[pa+8], m.bus.rdram.data[pa+9], m.bus.rdram.data[pa+10], m.bus.rdram.data[pa+11]]);
        let first = u32::from_be_bytes([m.bus.rdram.data[pa+12], m.bus.rdram.data[pa+13], m.bus.rdram.data[pa+14], m.bus.rdram.data[pa+15]]);
        eprintln!("[headless] Q@0x80365E10 valid={} first={}", valid, first);
    }
}

fn run_vk_test_pattern() {
    let n = (OUT_W * OUT_H * 4) as usize;
    let mut buf = vec![0u8; n];
    for px in buf.chunks_exact_mut(4) {
        px.copy_from_slice(&[0xFF, 0x00, 0xFF, 0xFF]);
    }
    let title = "UltraReality — vk test (Phase 0)";
    if let Err(e) = run_wgpu_loop(title, OUT_W, OUT_H, move || WgpuFrame::new(buf.clone(), true)) {
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
        WgpuFrame::new(buf.clone(), true)
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

        let o = m.bus.vi.framebuffer_rdram_offset();
        let fw = m.bus.vi.display_width();
        let fh = m.bus.vi.display_height();
        log_fps_with_fb(&mut last_status, &mut frames, m.bus.vi.frame_counter, o, fw as usize, fh as usize, m.bus.mi.intr, m.bus.mi.mask);
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
            let fb = self.machine.bus.vi.framebuffer_rdram_offset();
            let mi_intr = self.machine.bus.mi.intr;
            let mi_mask = self.machine.bus.mi.mask;
            eprintln!(
                "VI frame {} | ~{:.0} fps | FB={:06X} | MI=0x{:02X}/0x{:02X} | phase {:?}",
                vi_fc,
                self.frames as f64 / self.last_status.elapsed().as_secs_f64().max(0.001),
                fb,
                mi_intr, mi_mask,
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
        WgpuFrame::new(st.rgba.clone(), true)
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
    static TRACED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    static BOOT_CHECKED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

    // Check code at boot time (first call, frame 0)
    if !BOOT_CHECKED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        eprintln!("\n=== BOOT CODE CHECK ===");
        eprintln!("Caller region (0x80249520-0x80249540):");
        for addr in (0x80249520u32..0x80249540).step_by(4) {
            let paddr = (addr & 0x1FFF_FFFF) as usize;
            let word = if paddr + 4 <= m.bus.rdram.data.len() {
                u32::from_be_bytes([
                    m.bus.rdram.data[paddr],
                    m.bus.rdram.data[paddr + 1],
                    m.bus.rdram.data[paddr + 2],
                    m.bus.rdram.data[paddr + 3],
                ])
            } else { 0 };
            eprintln!("  0x{:08X}: {:08X}{}", addr, word, if word == 0 { " <-- ZERO!" } else { "" });
        }
        eprintln!("\nTarget region (0x80378800-0x80378820) - JAL target:");
        for addr in (0x80378800u32..0x80378820).step_by(4) {
            let paddr = (addr & 0x1FFF_FFFF) as usize;
            let word = if paddr + 4 <= m.bus.rdram.data.len() {
                u32::from_be_bytes([
                    m.bus.rdram.data[paddr],
                    m.bus.rdram.data[paddr + 1],
                    m.bus.rdram.data[paddr + 2],
                    m.bus.rdram.data[paddr + 3],
                ])
            } else { 0 };
            eprintln!("  0x{:08X}: {:08X}{}", addr, word, if word == 0 { " <-- ZERO!" } else { "" });
        }
        // Also check what's in ROM at the corresponding offset
        let rom_offset = 0x1000 + (0x378800 - 0x246000);
        eprintln!("\nROM at offset 0x{:06X} (maps to 0x80378800):", rom_offset);
        for i in 0..8 {
            let off = rom_offset + i * 4;
            if off + 4 <= m.bus.pi.rom.len() {
                let word = u32::from_be_bytes([
                    m.bus.pi.rom[off],
                    m.bus.pi.rom[off + 1],
                    m.bus.pi.rom[off + 2],
                    m.bus.pi.rom[off + 3],
                ]);
                eprintln!("  ROM[0x{:06X}]: {:08X}{}", off, word, if word == 0 { " <-- ZERO!" } else { "" });
            }
        }
        // Check ROM at caller location (0x80249528)
        let caller_rom_offset = 0x1000 + (0x249528 - 0x246000);
        eprintln!("\nROM at offset 0x{:06X} (maps to caller 0x80249528):", caller_rom_offset);
        for i in 0..4 {
            let off = caller_rom_offset + i * 4;
            if off + 4 <= m.bus.pi.rom.len() {
                let word = u32::from_be_bytes([
                    m.bus.pi.rom[off],
                    m.bus.pi.rom[off + 1],
                    m.bus.pi.rom[off + 2],
                    m.bus.pi.rom[off + 3],
                ]);
                eprintln!("  ROM[0x{:06X}]: {:08X}", off, word);
            }
        }
        eprintln!("=======================\n");
    }

    let before = m.bus.vi.frame_counter;
    let mut steps = 0u64;
    const MAX_STEPS_PER_VI_FRAME: u64 = 50_000_000;

    // Look for any game code execution across all frames
    static GAME_CODE_FOUND: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    static GAME_CODE_PC: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    static GAME_CODE_FRAME: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    // Trace exception handler flow on frame 10
    let do_trace = before == 10 && !TRACED.load(std::sync::atomic::Ordering::Relaxed);
    let mut in_handler = false;
    let mut handler_pcs: Vec<(u32, u32, u64)> = Vec::new(); // (pc, instruction, ra)
    let mut handler_trace_count = 0u32;
    let mut eret_pc: Option<u32> = None;
    let mut post_eret_pcs: Vec<u32> = Vec::new();
    let mut capture_post_eret = false;

    // Watch for the code at 0x80378800 being overwritten (the JAL target)
    static CODE_WIPE_LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    static PREV_VALUE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0xFFFFFFFF);

    // Watch gameThread queue (0x80206D60..0x80206D80) for changes; report PC of writer.
    static GTQ_PREV: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0xFFFF_FFFF_FFFF_FFFF);
    static GTQ_LOG_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    // Watch SM64 scheduler input queue Q@0x8033ADF0 — should receive a message every VI.
    static SCHEDQ_PREV: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0xFFFF_FFFF_FFFF_FFFF);
    static SCHEDQ_LOG_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    // Watch __osViMgrThread input queue Q@0x80365E10 — should tick on every VI ISR.
    static VIMGRQ_PREV: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0xFFFF_FFFF);
    static VIMGRQ_LOG_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    // Watch gameThread's ACTUAL queue Q@0x80367158 for messages
    static GTACTQ_PREV: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0xFFFF_FFFF);
    static GTACTQ_LOG_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    // After a VIMGRQ recv (valid 1->0), capture next N PCs (these are __osViMgrThread code).
    static VIMGR_CAPTURE_LEFT: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);
    static VIMGR_CAPTURE_DONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

    // Trace osSetTimer entry to find who sets the 5-second timer
    static OSSETTIMER_LOG_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    // Trace __osSetTimerIntr entry to find who sets the 5-second timer
    static SETTIMERINTR_LOG_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

    while m.bus.vi.frame_counter == before && steps < MAX_STEPS_PER_VI_FRAME && !*cpu_halted {
        let pc = m.cpu.pc as u32;

        // Catch first execution of game code at 0x80378800
        static GAME_CODE_EXEC_LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if pc == 0x80378800 && !GAME_CODE_EXEC_LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            let instr_pa = 0x00378800usize;
            let instr = if instr_pa + 4 <= m.bus.rdram.data.len() {
                u32::from_be_bytes([
                    m.bus.rdram.data[instr_pa],
                    m.bus.rdram.data[instr_pa+1],
                    m.bus.rdram.data[instr_pa+2],
                    m.bus.rdram.data[instr_pa+3],
                ])
            } else { 0 };
            let ra = m.cpu.regs[31] as u32;
            eprintln!("[GAME_CODE@0x80378800] f={} s={} RA=0x{:08X} instr=0x{:08X} ({})",
                before, steps, ra, instr,
                if instr == 0 { "NOP - code not loaded!" } else { "loaded" });
        }

        // Trace osCreateThread calls - entry point is a2 (argument 2 = entry function)
        // osCreateThread is typically at a known libultra address, let's catch it by looking for
        // the pattern of writing to thread struct followed by thread list manipulation
        // Actually, let's trace when PC enters the 0x80370xxx range (unloaded code)
        static UNLOADED_CODE_LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if pc >= 0x80370000 && pc < 0x80378800 && !UNLOADED_CODE_LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            let instr_pa = (pc & 0x1FFF_FFFF) as usize;
            let instr = if instr_pa + 4 <= m.bus.rdram.data.len() {
                u32::from_be_bytes([
                    m.bus.rdram.data[instr_pa],
                    m.bus.rdram.data[instr_pa+1],
                    m.bus.rdram.data[instr_pa+2],
                    m.bus.rdram.data[instr_pa+3],
                ])
            } else { 0 };
            let ra = m.cpu.regs[31] as u32;
            eprintln!("[UNLOADED_REGION] f={} s={} PC=0x{:08X} RA=0x{:08X} instr=0x{:08X} ({})",
                before, steps, pc, ra, instr,
                if instr == 0 { "ZERO - code not loaded!" } else { "has data" });
        }

        // Trace ERET execution to see thread dispatch targets
        static ERET_DISPATCH_LOG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let instr_pa = (pc & 0x1FFF_FFFF) as usize;
        let instr_word = if instr_pa + 4 <= m.bus.rdram.data.len() {
            u32::from_be_bytes([
                m.bus.rdram.data[instr_pa],
                m.bus.rdram.data[instr_pa+1],
                m.bus.rdram.data[instr_pa+2],
                m.bus.rdram.data[instr_pa+3],
            ])
        } else { 0 };
        // ERET = 0x42000018
        if instr_word == 0x42000018 {
            let n = ERET_DISPATCH_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n < 20 {
                let epc = m.cpu.cop0.read_xpr64(14) as u32; // EPC register
                eprintln!("[ERET #{}] f={} s={} PC=0x{:08X} -> EPC=0x{:08X}", n, before, steps, pc, epc);
            }
        }

        // Catch entry to osSetTimer at 0x80328A10
        // osSetTimer(OSTimer *t, OSTime countdown, OSTime interval, OSMesgQueue *mq, OSMesg msg)
        // For O32 ABI with 64-bit args:
        // a0 = timer, (a1 skipped), a2-a3 = countdown (64-bit), stack: interval, mq, msg
        if pc == 0x80328A10 {
            let n = OSSETTIMER_LOG_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n < 5 {
                let ra = m.cpu.regs[31] as u32;
                let a0 = m.cpu.regs[4] as u32; // timer struct pointer
                let a2 = m.cpu.regs[6] as u32; // countdown lo
                let a3 = m.cpu.regs[7] as u32; // countdown hi
                let sp = m.cpu.regs[29] as u32;
                // Read interval from stack (sp+0x10 and sp+0x14)
                let read_stack = |off: u32| -> u32 {
                    let pa = ((sp + off) & 0x1FFF_FFFF) as usize;
                    if pa + 4 <= m.bus.rdram.data.len() {
                        u32::from_be_bytes([
                            m.bus.rdram.data[pa], m.bus.rdram.data[pa+1],
                            m.bus.rdram.data[pa+2], m.bus.rdram.data[pa+3],
                        ])
                    } else { 0 }
                };
                // Stack layout for 64-bit arg in big-endian: HIGH word at lower address
                let interval_hi = read_stack(0x10);
                let interval_lo = read_stack(0x14);
                let mq = read_stack(0x18);
                let msg = read_stack(0x1C);
                // O32 big-endian: first reg = HIGH half, second reg = LOW half
                let countdown = ((a2 as u64) << 32) | (a3 as u64);
                let interval = ((interval_hi as u64) << 32) | (interval_lo as u64);
                let countdown_sec = countdown as f64 / 46_875_000.0;
                let interval_sec = interval as f64 / 46_875_000.0;
                eprintln!("[osSetTimer #{} ENTRY] f={} s={} RA=0x{:08X}", n, before, steps, ra);
                eprintln!("  timer=0x{:08X} mq=0x{:08X} msg=0x{:08X}", a0, mq, msg);
                eprintln!("  countdown={:.3}s (0x{:X}) interval={:.3}s (0x{:X})",
                    countdown_sec, countdown, interval_sec, interval);
            }
        }

        // Catch entry to __osSetTimerIntr (real entry at 0x80326064 after NOP at 0x80326060)
        if pc == 0x80326064 {
            let n = SETTIMERINTR_LOG_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n < 5 {
                let ra = m.cpu.regs[31] as u32;
                let a0 = m.cpu.regs[4] as u32; // first arg (timer interval lo)
                let a1 = m.cpu.regs[5] as u32; // second arg (timer interval hi)
                let sp = m.cpu.regs[29] as u32;
                let count = m.cpu.cop0.read_32(9);
                eprintln!("[__osSetTimerIntr #{} ENTRY] f={} s={} RA=0x{:08X} a0=0x{:08X} a1=0x{:08X} Count=0x{:08X} SP=0x{:08X}",
                    n, before, steps, ra, a0, a1, count, sp);
                // Read up the call stack - parent's saved RA is at current sp + frame_size (estimate 0x20-0x40)
                for offset in [0x14u32, 0x24, 0x34, 0x44, 0x54].iter() {
                    let parent_ra_pa = ((sp + offset) & 0x1FFF_FFFF) as usize;
                    if parent_ra_pa + 4 <= m.bus.rdram.data.len() {
                        let saved = u32::from_be_bytes([
                            m.bus.rdram.data[parent_ra_pa],
                            m.bus.rdram.data[parent_ra_pa+1],
                            m.bus.rdram.data[parent_ra_pa+2],
                            m.bus.rdram.data[parent_ra_pa+3],
                        ]);
                        if saved >= 0x80000000 && saved < 0x80800000 {
                            eprintln!("  SP+0x{:02X}: 0x{:08X}", offset, saved);
                        }
                    }
                }
            }
        }

        // Snapshot mt(+0) and validCount(+8) of Q@0x80206D60 prior to step
        let pa_q = 0x00206D60usize;
        let snap_now: u64 = if pa_q + 16 <= m.bus.rdram.data.len() {
            let v = u32::from_be_bytes([
                m.bus.rdram.data[pa_q+8], m.bus.rdram.data[pa_q+9],
                m.bus.rdram.data[pa_q+10], m.bus.rdram.data[pa_q+11],
            ]);
            let mt = u32::from_be_bytes([
                m.bus.rdram.data[pa_q], m.bus.rdram.data[pa_q+1],
                m.bus.rdram.data[pa_q+2], m.bus.rdram.data[pa_q+3],
            ]);
            ((mt as u64) << 32) | (v as u64)
        } else { 0 };
        let prev_snap = GTQ_PREV.load(std::sync::atomic::Ordering::Relaxed);
        if prev_snap != 0xFFFF_FFFF_FFFF_FFFF && snap_now != prev_snap {
            let n = GTQ_LOG_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n < 30 {
                eprintln!("[GTQ@0x80206D60 changed] frame={} step={} PC=0x{:08X} prev=mt/v=0x{:08X}/0x{:08X} now=0x{:08X}/0x{:08X}",
                    before, steps, pc,
                    (prev_snap >> 32) as u32, (prev_snap & 0xFFFF_FFFF) as u32,
                    (snap_now >> 32) as u32, (snap_now & 0xFFFF_FFFF) as u32);
            }
        }
        GTQ_PREV.store(snap_now, std::sync::atomic::Ordering::Relaxed);
        // Snapshot validCount(+8) and first(+12) of Q@0x8033AE08 (scheduler work queue - receives VI from __osViNext)
        let pa_sq = 0x0033AE08usize;
        let snap_sq: u64 = if pa_sq + 16 <= m.bus.rdram.data.len() {
            let v = u32::from_be_bytes([
                m.bus.rdram.data[pa_sq+8], m.bus.rdram.data[pa_sq+9],
                m.bus.rdram.data[pa_sq+10], m.bus.rdram.data[pa_sq+11],
            ]);
            let f = u32::from_be_bytes([
                m.bus.rdram.data[pa_sq+12], m.bus.rdram.data[pa_sq+13],
                m.bus.rdram.data[pa_sq+14], m.bus.rdram.data[pa_sq+15],
            ]);
            ((v as u64) << 32) | (f as u64)
        } else { 0 };
        let prev_sq = SCHEDQ_PREV.load(std::sync::atomic::Ordering::Relaxed);
        if prev_sq != 0xFFFF_FFFF_FFFF_FFFF && snap_sq != prev_sq {
            let n = SCHEDQ_LOG_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n < 60 || before >= 316 {
                let cause = m.cpu.cop0.cause;
                let status = m.cpu.cop0.status;
                let ra = m.cpu.regs[31] as u32;
                let sp = m.cpu.regs[29] as u32;
                // Read saved RA at sp+0x24 (osSendMesg saves $ra there)
                let saved_ra_pa = ((sp + 0x24) & 0x1FFF_FFFF) as usize;
                let saved_ra = if saved_ra_pa + 4 <= m.bus.rdram.data.len() {
                    u32::from_be_bytes([
                        m.bus.rdram.data[saved_ra_pa],
                        m.bus.rdram.data[saved_ra_pa+1],
                        m.bus.rdram.data[saved_ra_pa+2],
                        m.bus.rdram.data[saved_ra_pa+3],
                    ])
                } else { 0 };
                // Read saved arg1 (the OSMesg) at sp+0x3C
                let saved_msg_pa = ((sp + 0x3C) & 0x1FFF_FFFF) as usize;
                let saved_msg = if saved_msg_pa + 4 <= m.bus.rdram.data.len() {
                    u32::from_be_bytes([
                        m.bus.rdram.data[saved_msg_pa],
                        m.bus.rdram.data[saved_msg_pa+1],
                        m.bus.rdram.data[saved_msg_pa+2],
                        m.bus.rdram.data[saved_msg_pa+3],
                    ])
                } else { 0 };
                eprintln!("[SCHEDQ] f={} s={} PC=0x{:08X} SAVED_RA=0x{:08X} MSG=0x{:08X} v={}->{} f={}->{} status=0x{:08X}",
                    before, steps, pc, saved_ra, saved_msg,
                    (prev_sq >> 32) as u32, (snap_sq >> 32) as u32,
                    (prev_sq & 0xFFFF_FFFF) as u32, (snap_sq & 0xFFFF_FFFF) as u32,
                    status);
                let _ = cause;
                let _ = ra;
            }
        }
        SCHEDQ_PREV.store(snap_sq, std::sync::atomic::Ordering::Relaxed);

        // Snapshot validCount of Q@0x80365E10 (__osViMgrThread input queue)
        let pa_vmq = 0x00365E10usize;
        let snap_vmq: u32 = if pa_vmq + 12 <= m.bus.rdram.data.len() {
            u32::from_be_bytes([
                m.bus.rdram.data[pa_vmq+8], m.bus.rdram.data[pa_vmq+9],
                m.bus.rdram.data[pa_vmq+10], m.bus.rdram.data[pa_vmq+11],
            ])
        } else { 0 };
        let prev_vmq = VIMGRQ_PREV.load(std::sync::atomic::Ordering::Relaxed);
        if prev_vmq != 0xFFFF_FFFF && snap_vmq != prev_vmq {
            let n = VIMGRQ_LOG_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n < 60 {
                eprintln!("[VIMGRQ@0x80365E10] f={} s={} PC=0x{:08X} valid {}->{}",
                    before, steps, pc, prev_vmq, snap_vmq);
            }
            // On the second decrement (valid 1->0), arm capture of next 200 PCs (skip 1st frame to ensure full setup)
            if prev_vmq == 1 && snap_vmq == 0 && before >= 5 && !VIMGR_CAPTURE_DONE.load(std::sync::atomic::Ordering::Relaxed) {
                VIMGR_CAPTURE_LEFT.store(800, std::sync::atomic::Ordering::Relaxed);
                VIMGR_CAPTURE_DONE.store(true, std::sync::atomic::Ordering::Relaxed);
                eprintln!("[VIMGR_TRACE] arming capture at f={} s={}", before, steps);
            }
        }
        VIMGRQ_PREV.store(snap_vmq, std::sync::atomic::Ordering::Relaxed);

        // Monitor gameThread's actual queue Q@0x80367158
        let pa_gaq = 0x00367158usize;
        let snap_gaq: u32 = if pa_gaq + 12 <= m.bus.rdram.data.len() {
            u32::from_be_bytes([
                m.bus.rdram.data[pa_gaq+8], m.bus.rdram.data[pa_gaq+9],
                m.bus.rdram.data[pa_gaq+10], m.bus.rdram.data[pa_gaq+11],
            ])
        } else { 0 };
        let prev_gaq = GTACTQ_PREV.load(std::sync::atomic::Ordering::Relaxed);
        if prev_gaq != 0xFFFF_FFFF && snap_gaq != prev_gaq {
            let n = GTACTQ_LOG_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n < 30 {
                let ra = m.cpu.regs[31] as u32;
                eprintln!("[GTACTQ@0x80367158] f={} s={} PC=0x{:08X} RA=0x{:08X} valid {}->{}",
                    before, steps, pc, ra, prev_gaq as i32, snap_gaq as i32);
            }
        }
        GTACTQ_PREV.store(snap_gaq, std::sync::atomic::Ordering::Relaxed);

        // Hook __osSiGetAccess (0x803288F0), __osSiRelAccess (0x80328934), and the leak branch at
        // 0x803291C0 to trace SI access sequencing. Queue 0x80367158 is __osSiAccessQueue; the
        // function at 0x80329150 (osEepromRead) has a path that skips REL when __osEepStatus
        // returns non-zero or reports a non-EEPROM type.
        {
            static SIGETREL_LOG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if pc == 0x803288F0 {
                // __osSiGetAccess entry — $ra is the direct caller.
                let n = SIGETREL_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if n < 60 {
                    let ra = m.cpu.regs[31] as u32;
                    let sp = m.cpu.regs[29] as u32;
                    eprintln!("[SI_GET#{}] f={} s={} RA=0x{:08X} SP=0x{:08X}",
                        n, before, steps, ra, sp);
                }
            } else if pc == 0x80328934 {
                let n = SIGETREL_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if n < 60 {
                    let ra = m.cpu.regs[31] as u32;
                    eprintln!("[SI_REL#{}] f={} s={} RA=0x{:08X}", n, before, steps, ra);
                }
            } else if pc == 0x803291C0 {
                // Leak branch taken: `b 0x80329330` — reports back to caller WITHOUT REL.
                let n = SIGETREL_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if n < 60 {
                    let ra = m.cpu.regs[31] as u32;
                    let sp = m.cpu.regs[29] as u32;
                    // Read saved $v0-like values / $t8 that gated the branch.
                    let read_w = |addr: u32| -> u32 {
                        let pa = (addr & 0x1FFF_FFFF) as usize;
                        if pa + 4 <= m.bus.rdram.data.len() {
                            u32::from_be_bytes([
                                m.bus.rdram.data[pa], m.bus.rdram.data[pa+1],
                                m.bus.rdram.data[pa+2], m.bus.rdram.data[pa+3],
                            ])
                        } else { 0 }
                    };
                    let saved_v0 = read_w(sp + 0x34);
                    let saved_a0 = read_w(sp + 0x38);
                    let sp28_hw = read_w(sp + 0x28);
                    eprintln!("[SI_LEAK@0x803291C0] #{} f={} s={} RA=0x{:08X} SP=0x{:08X} saved_v0=0x{:08X} mq=0x{:08X} sp28={:08X}",
                        n, before, steps, ra, sp, saved_v0, saved_a0, sp28_hw);
                }
            }
        }

        // Hook osSendMesg @0x80322C20 — log every call so we can see which
        // queue is being posted to during the post-EEPROM phase. Limit to a
        // few hundred entries to avoid runaway.
        if pc == 0x8032_2C20 {
            static SENDMESG_LOG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            let n = SENDMESG_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            // Always log if we're past frame 310, otherwise sample early.
            if before >= 305 || n < 30 {
                let mq = m.cpu.regs[4] as u32;
                let msg = m.cpu.regs[5] as u32;
                let flag = m.cpu.regs[6] as u32;
                let ra = m.cpu.regs[31] as u32;
                let running = {
                    let pa = 0x3359B0usize;
                    if pa + 4 <= m.bus.rdram.data.len() {
                        u32::from_be_bytes([
                            m.bus.rdram.data[pa], m.bus.rdram.data[pa+1],
                            m.bus.rdram.data[pa+2], m.bus.rdram.data[pa+3],
                        ])
                    } else { 0 }
                };
                let tid = if running >= 0x80000000 {
                    let pa = (running & 0x1FFFFFFF) as usize + 0x14;
                    if pa + 4 <= m.bus.rdram.data.len() {
                        u32::from_be_bytes([
                            m.bus.rdram.data[pa], m.bus.rdram.data[pa+1],
                            m.bus.rdram.data[pa+2], m.bus.rdram.data[pa+3],
                        ])
                    } else { 0 }
                } else { 0 };
                eprintln!(
                    "[SEND #{}] f={} s={} thread={} mq=0x{:08X} msg=0x{:08X} flag={} ra=0x{:08X}",
                    n, before, steps, tid, mq, msg, flag, ra,
                );
            }
        }
        // Hook osRecvMesg @0x80322800 in same way to see who blocks where.
        let _ = ();
        // Hook entry to the printf-like function at 0x8019B53C (called by gd_error path)
        // to capture the actual format string and args. This should show what the
        // gd_error printed just before calling exit.
        // Hook entry to gd_inverse_mat4f at 0x80194FB8. Dump the 4x4 input matrix at a0.
        // Hook entry to caller of gd_inverse_mat4f at 0x80181CC8 to find its caller.
        if pc == 0x8018_1CC8 {
            static C1: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            let n = C1.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n < 5 {
                let a0 = m.cpu.regs[4] as u32;
                let ra = m.cpu.regs[31] as u32;
                eprintln!("[CALLER_0x80181CC4 #{}] f={} s={} a0=0x{:08X} ra=0x{:08X}", n, before, steps, a0, ra);
                // Dump the struct at a0 (0x100 bytes) — the matrix is at a0+0xE8.
                let pa = (a0 & 0x1FFFFFFF) as usize;
                if pa + 0x108 <= m.bus.rdram.data.len() {
                    for row in 0..0x108/16 {
                        let base = pa + row*16;
                        let mut words = [0u32; 4];
                        let mut floats = String::new();
                        for i in 0..4 {
                            words[i] = u32::from_be_bytes([
                                m.bus.rdram.data[base + i*4],
                                m.bus.rdram.data[base + i*4 + 1],
                                m.bus.rdram.data[base + i*4 + 2],
                                m.bus.rdram.data[base + i*4 + 3],
                            ]);
                            let fv = f32::from_bits(words[i]);
                            // Heuristic: show as float if it looks float-like, else hex
                            if fv.is_finite() && fv.abs() < 1e9 && fv != 0.0 {
                                floats.push_str(&format!(" {:+.3e}", fv));
                            } else {
                                floats.push_str(&format!(" {:08X}     ", words[i]));
                            }
                        }
                        eprintln!("  +{:03X}: {:08X} {:08X} {:08X} {:08X} |{}",
                            row*16, words[0], words[1], words[2], words[3], floats);
                    }
                }
            }
        }
        // Hook entry to the function one level higher at 0x8017E674 to find ITS caller.
        // First need to find its prologue; probe 0x8017E600..0x8017E680 for prologue.
        if pc == 0x8017_E630 || pc == 0x8017_E640 || pc == 0x8017_E650 || pc == 0x8017_E660 {
            static C2: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            let n = C2.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n < 5 {
                let a0 = m.cpu.regs[4] as u32;
                let sp = m.cpu.regs[29] as u32;
                let ra = m.cpu.regs[31] as u32;
                eprintln!("[C2_{:08X} #{}] f={} s={} a0=0x{:08X} sp=0x{:08X} ra=0x{:08X}",
                    pc, n, before, steps, a0, sp, ra);
            }
        }
        // Diagnostic: count visits to the function entry AND the jal site.
        if pc == 0x8019_4FBC || pc == 0x8019_5044 || pc == 0x8019_5048 {
            static VISIT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            let v = VISIT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if v < 20 {
                eprintln!("[VISIT {}] pc=0x{:08X} f={} s={} ra=0x{:08X}", v, pc, before, steps, m.cpu.regs[31] as u32);
            }
        }
        // Trace abs(det) sequence: entry to c.lt.s (0x80195000), bc1f (0x80195008),
        // b + neg.s (0x80195010), lwc1 path (0x80195018), merge point (0x80195020).
        if (0x8019_5000..=0x8019_5020).contains(&pc) && (pc & 3) == 0 {
            static ABS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            let n = ABS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n < 30 {
                let f4 = m.cpu.cop1.fpr[4] as u32;
                let f6 = m.cpu.cop1.fpr[6] as u32;
                let f20 = m.cpu.cop1.fpr[20] as u32;
                let fcsr = m.cpu.cop1.fcsr;
                let cc0 = (fcsr >> 23) & 1;
                eprintln!("[ABS_TRACE #{}] pc=0x{:08X} s={} f4=0x{:08X}({:+e}) f6=0x{:08X}({:+e}) f20=0x{:08X}({:+e}) fcsr=0x{:08X} cc0={}",
                    n, pc, steps,
                    f4, f32::from_bits(f4),
                    f6, f32::from_bits(f6),
                    f20, f32::from_bits(f20),
                    fcsr, cc0);
            }
        }
        // Dump det-check operands right before the C.LT.D at 0x80195030 in gd_inverse_mat4f.
        // pair (f8,f9) = cvt.d.s of abs(det); pair (f10,f11) = epsilon double loaded from 0x801B8A48.
        if pc == 0x8019_5030 {
            static DET: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            let n = DET.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n < 4 {
                let f8 = m.cpu.cop1.fpr[8];
                let f9 = m.cpu.cop1.fpr[9];
                let f10 = m.cpu.cop1.fpr[10];
                let f11 = m.cpu.cop1.fpr[11];
                let f20 = m.cpu.cop1.fpr[20];
                let status = m.cpu.cop0.status;
                let fr = (status >> 26) & 1;
                // FR=0 double read: even=lo, odd=hi
                let det_bits = if fr == 0 {
                    ((f9 as u32 as u64) << 32) | (f8 as u32 as u64)
                } else { f8 };
                let eps_bits = if fr == 0 {
                    ((f11 as u32 as u64) << 32) | (f10 as u32 as u64)
                } else { f10 };
                eprintln!("[DET_CLTD #{}] f={} s={} FR={} status=0x{:08X}", n, before, steps, fr, status);
                eprintln!("  f20 (|det| single) = 0x{:08X} ({:+e})", f20 as u32, f32::from_bits(f20 as u32));
                eprintln!("  f8  = 0x{:016X}  f9  = 0x{:016X}", f8, f9);
                eprintln!("  f10 = 0x{:016X}  f11 = 0x{:016X}", f10, f11);
                eprintln!("  det_double  = {:+e}  (bits=0x{:016X})", f64::from_bits(det_bits), det_bits);
                eprintln!("  eps_double  = {:+e}  (bits=0x{:016X})", f64::from_bits(eps_bits), eps_bits);
                eprintln!("  compare det<eps => {}", f64::from_bits(det_bits) < f64::from_bits(eps_bits));
            }
        }
        // Hook quat_to_matrix at 0x80196114 — dump FPU state and status register.
        if pc == 0x8019_6114 {
            static QC: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if before >= 895 && before <= 898 {
                let n = QC.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if n < 3 {
                    let a0 = m.cpu.regs[4] as u32;
                    let a1 = m.cpu.regs[5] as u32;
                    let a2 = m.cpu.regs[6] as u32;
                    let a3 = m.cpu.regs[7] as u32;
                    let status = m.cpu.cop0.status;
                    let fr = (status >> 26) & 1;
                    eprintln!("[QTM #{}] f={} s={} a0(dst)=0x{:08X} a1(axis)=0x{:08X} a2=0x{:08X} a3=0x{:08X} Status=0x{:08X} FR={}",
                        n, before, steps, a0, a1, a2, a3, status, fr);
                    // Dump axis at a1
                    let pa = (a1 & 0x1FFFFFFF) as usize;
                    if pa + 12 <= m.bus.rdram.data.len() {
                        let x = u32::from_be_bytes([m.bus.rdram.data[pa], m.bus.rdram.data[pa+1], m.bus.rdram.data[pa+2], m.bus.rdram.data[pa+3]]);
                        let y = u32::from_be_bytes([m.bus.rdram.data[pa+4], m.bus.rdram.data[pa+5], m.bus.rdram.data[pa+6], m.bus.rdram.data[pa+7]]);
                        let z = u32::from_be_bytes([m.bus.rdram.data[pa+8], m.bus.rdram.data[pa+9], m.bus.rdram.data[pa+10], m.bus.rdram.data[pa+11]]);
                        eprintln!("  axis = ({:+e}, {:+e}, {:+e})", f32::from_bits(x), f32::from_bits(y), f32::from_bits(z));
                    }
                    eprintln!("  a2(sin?)={:+e} a3(cos?)={:+e}", f32::from_bits(a2), f32::from_bits(a3));
                }
            }
        }
        // Hook at 0x80196180 (final sw f6, 0(a0) of quat_to_matrix diagonal m[0][0] = cos + (1-cos)*x^2).
        if pc == 0x8019_6180 {
            static QD: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if before >= 895 && before <= 898 {
                let n = QD.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if n < 5 {
                    // Dump key FP registers to see what value is about to be stored.
                    let f4 = m.cpu.cop1.fpr[4];
                    let f6 = m.cpu.cop1.fpr[6];
                    let f8 = m.cpu.cop1.fpr[8];
                    let f10 = m.cpu.cop1.fpr[10];
                    let f11 = m.cpu.cop1.fpr[11];
                    let f16 = m.cpu.cop1.fpr[16];
                    let f18 = m.cpu.cop1.fpr[18];
                    let status = m.cpu.cop0.status;
                    let fr = (status >> 26) & 1;
                    eprintln!("[QTM_M00 #{}] f={} s={} FR={} status=0x{:08X}", n, before, steps, fr, status);
                    eprintln!("  f4  = 0x{:016X} as_f32=({:+e}) as_f64=({:+e})", f4, f32::from_bits(f4 as u32), f64::from_bits(f4));
                    eprintln!("  f6  = 0x{:016X} as_f32=({:+e})", f6, f32::from_bits(f6 as u32));
                    eprintln!("  f8  = 0x{:016X} as_f32=({:+e})", f8, f32::from_bits(f8 as u32));
                    eprintln!("  f10 = 0x{:016X} as_f32=({:+e}) as_f64=({:+e})", f10, f32::from_bits(f10 as u32), f64::from_bits(f10));
                    eprintln!("  f11 = 0x{:016X} as_f32=({:+e})", f11, f32::from_bits(f11 as u32));
                    eprintln!("  f16 = 0x{:016X} as_f32=({:+e})", f16, f32::from_bits(f16 as u32));
                    eprintln!("  f18 = 0x{:016X} as_f64=({:+e})", f18, f64::from_bits(f18));
                }
            }
        }
        // Hook rot_builder at 0x80196334 — log angle_half (a2) and trig-divisor constants at 0x801B8A50..0x801B8A60.
        if pc == 0x8019_6334 {
            static RC: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if before >= 895 && before <= 898 {
                let n = RC.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if n < 5 {
                    let a0 = m.cpu.regs[4] as u32;
                    let a1 = m.cpu.regs[5] as u32;
                    let a2 = m.cpu.regs[6] as u32;
                    let f12 = m.cpu.cop1.fpr[12];
                    let f13 = m.cpu.cop1.fpr[13];
                    // Load the double constants at 0x801B8A50, 0x801B8A58
                    let mut dcs = [0u64; 2];
                    for (idx, off) in [(0usize, 0x001B_8A50usize), (1, 0x001B_8A58)].iter() {
                        if off + 8 <= m.bus.rdram.data.len() {
                            let hi = u32::from_be_bytes([m.bus.rdram.data[*off], m.bus.rdram.data[*off+1], m.bus.rdram.data[*off+2], m.bus.rdram.data[*off+3]]);
                            let lo = u32::from_be_bytes([m.bus.rdram.data[*off+4], m.bus.rdram.data[*off+5], m.bus.rdram.data[*off+6], m.bus.rdram.data[*off+7]]);
                            // MIPS big-endian double: first word = high 32 bits
                            dcs[*idx] = ((hi as u64) << 32) | (lo as u64);
                        }
                    }
                    eprintln!("[ROT_BLD #{}] f={} s={} dst=0x{:08X} axis=0x{:08X} angle_half=0x{:08X} ({:+e})",
                        n, before, steps, a0, a1, a2, f32::from_bits(a2));
                    eprintln!("  const1@0x801B8A50 = 0x{:016X} ({:+e})", dcs[0], f64::from_bits(dcs[0]));
                    eprintln!("  const2@0x801B8A58 = 0x{:016X} ({:+e})", dcs[1], f64::from_bits(dcs[1]));
                    eprintln!("  f12=0x{:016X} f13=0x{:016X}", f12, f13);
                }
            }
        }
        // Hook mul_mat4 at 0x80196754 — logs A, B pointers + matrices.
        // Filter to only show the call whose output eventually becomes the
        // panic source (dst=0x800B77F8 via gd_copy_mat4). That call's a2
        // (out mat pointer) will be the same stack slot as the copy's src.
        if pc == 0x8019_6754 {
            static MC: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            // Target the specific mul call that overwrites the GdObj matrix (out=0x800B77F8).
            let out_ptr = m.cpu.regs[6] as u32;
            if before >= 895 && before <= 898 && (out_ptr == 0x800B_77F8 || out_ptr == 0x800B_7878 || out_ptr == 0x800B_7838) {
                let n = MC.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if n < 40 {
                    let a0 = m.cpu.regs[4] as u32;
                    let a1 = m.cpu.regs[5] as u32;
                    let a2 = m.cpu.regs[6] as u32;
                    let ra = m.cpu.regs[31] as u32;
                    eprintln!("[MUL_MAT4 #{}] f={} s={} matA=0x{:08X} matB=0x{:08X} out=0x{:08X} ra=0x{:08X}",
                        n, before, steps, a0, a1, a2, ra);
                    for (lbl, ptr) in [("A", a0), ("B", a1)] {
                        let pa = (ptr & 0x1FFFFFFF) as usize;
                        if pa + 64 <= m.bus.rdram.data.len() {
                            let mut rows = String::new();
                            for i in 0..16 {
                                let w = u32::from_be_bytes([
                                    m.bus.rdram.data[pa + i*4],
                                    m.bus.rdram.data[pa + i*4 + 1],
                                    m.bus.rdram.data[pa + i*4 + 2],
                                    m.bus.rdram.data[pa + i*4 + 3],
                                ]);
                                rows.push_str(&format!(" {:+.3e}", f32::from_bits(w)));
                                if i % 4 == 3 { rows.push('\n'); rows.push_str("      "); }
                            }
                            eprintln!("  mat{}: {}", lbl, rows.trim_end());
                        }
                    }
                }
            }
        }
        // Hook gd_copy_mat4 at 0x801964A0 — when dst=0x800B77F8, log source matrix + caller.
        // This is the function that copies the degenerate matrix into the GdObj.
        if pc == 0x8019_64A0 {
            let a0_src = m.cpu.regs[4] as u32;
            let a1_dst = m.cpu.regs[5] as u32;
            if before >= 895 && before <= 898 && (a1_dst == 0x800B_77F8 || a1_dst == 0x800B_7878 || a1_dst == 0x800B_7838) {
                static CPC: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                let n = CPC.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if n < 40 {
                    let ra = m.cpu.regs[31] as u32;
                    let pa = (a0_src & 0x1FFFFFFF) as usize;
                    let mut rows = String::new();
                    if pa + 64 <= m.bus.rdram.data.len() {
                        for i in 0..16 {
                            let w = u32::from_be_bytes([
                                m.bus.rdram.data[pa + i*4],
                                m.bus.rdram.data[pa + i*4 + 1],
                                m.bus.rdram.data[pa + i*4 + 2],
                                m.bus.rdram.data[pa + i*4 + 3],
                            ]);
                            rows.push_str(&format!(" {:+.3e}", f32::from_bits(w)));
                            if i % 4 == 3 { rows.push('\n'); rows.push_str("     "); }
                        }
                    }
                    eprintln!("[COPY_MAT4 #{}] f={} s={} src=0x{:08X} dst=0x{:08X} ra=0x{:08X}", n, before, steps, a0_src, a1_dst, ra);
                    eprintln!("     {}", rows.trim_end());
                }
            }
        }
        if pc == 0x8019_4FBC {
            static INV_HOOK: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            let n = INV_HOOK.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            // Log only the last few calls before the panic frame (897).
            // Log all calls; restrict volume via n.
            if n < 5 || (before >= 895 && before <= 898 && n < 2000) {
                let a0 = m.cpu.regs[4] as u32;
                let a1 = m.cpu.regs[5] as u32;
                let ra = m.cpu.regs[31] as u32;
                let pa = (a0 & 0x1FFFFFFF) as usize;
                if pa + 64 <= m.bus.rdram.data.len() {
                    let mut rows = String::new();
                    let mut all_words = [0u32; 16];
                    for i in 0..16 {
                        let w = u32::from_be_bytes([
                            m.bus.rdram.data[pa + i*4],
                            m.bus.rdram.data[pa + i*4 + 1],
                            m.bus.rdram.data[pa + i*4 + 2],
                            m.bus.rdram.data[pa + i*4 + 3],
                        ]);
                        all_words[i] = w;
                        let fv = f32::from_bits(w);
                        rows.push_str(&format!(" {:+.6e}", fv));
                        if i % 4 == 3 { rows.push('\n'); rows.push_str("     "); }
                    }
                    // Compute native Rust determinant for cross-check
                    let m4 = [
                        [f32::from_bits(all_words[0]), f32::from_bits(all_words[1]), f32::from_bits(all_words[2]), f32::from_bits(all_words[3])],
                        [f32::from_bits(all_words[4]), f32::from_bits(all_words[5]), f32::from_bits(all_words[6]), f32::from_bits(all_words[7])],
                        [f32::from_bits(all_words[8]), f32::from_bits(all_words[9]), f32::from_bits(all_words[10]), f32::from_bits(all_words[11])],
                        [f32::from_bits(all_words[12]), f32::from_bits(all_words[13]), f32::from_bits(all_words[14]), f32::from_bits(all_words[15])],
                    ];
                    // Native 4x4 determinant via cofactor expansion on row 0 (f64 for precision)
                    fn det3(a: &[[f64; 3]; 3]) -> f64 {
                        a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])
                         - a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])
                         + a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0])
                    }
                    let mut det = 0.0f64;
                    for col in 0..4 {
                        let mut sub = [[0.0f64; 3]; 3];
                        for r in 1..4 {
                            let mut sc = 0;
                            for c in 0..4 {
                                if c == col { continue; }
                                sub[r-1][sc] = m4[r][c] as f64;
                                sc += 1;
                            }
                        }
                        let sign = if col % 2 == 0 { 1.0 } else { -1.0 };
                        det += sign * (m4[0][col] as f64) * det3(&sub);
                    }
                    eprintln!("[gd_inverse_mat4f #{}] f={} s={} a0=0x{:08X} a1=0x{:08X} ra=0x{:08X}", n, before, steps, a0, a1, ra);
                    eprintln!("     {}", rows.trim_end());
                    eprintln!("  native_det(f64) = {:+.6e}", det);
                }
            }
        }
        if pc == 0x8019_B53C {
            static PRINTF_HOOK: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            let n = PRINTF_HOOK.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            // Only log calls NEAR the panic frame (f=897) to avoid flooding
            if before >= 895 && before <= 898 && n < 200 {
                let a0 = m.cpu.regs[4] as u32;
                let a1 = m.cpu.regs[5] as u32;
                let a2 = m.cpu.regs[6] as u32;
                let a3 = m.cpu.regs[7] as u32;
                let ra = m.cpu.regs[31] as u32;
                // Try to read format string from a0
                let fmt: String = {
                    let pa = (a0 & 0x1FFFFFFF) as usize;
                    if pa < m.bus.rdram.data.len().saturating_sub(200) {
                        let slice = &m.bus.rdram.data[pa..pa + 200];
                        let end = slice.iter().position(|&b| b == 0).unwrap_or(200);
                        String::from_utf8_lossy(&slice[..end]).to_string()
                    } else {
                        String::new()
                    }
                };
                eprintln!("[PRINTF #{}] f={} s={} ra=0x{:08X} a0=0x{:08X} \"{}\" a1=0x{:08X} a2=0x{:08X} a3=0x{:08X}",
                    n, before, steps, ra, a0, fmt.replace('\n', "\\n"), a1, a2, a3);
                // On the FIRST printf call near the panic frame, dump fatal_printf's
                // stack frame to find its caller (= gd_inverse_mat4f return address).
                if n == 0 {
                    let sp = m.cpu.regs[29] as u32;
                    let pa = (sp & 0x1FFFFFFF) as usize;
                    if pa + 128 <= m.bus.rdram.data.len() {
                        let mut stk = String::new();
                        for i in 0..32 {
                            let w = u32::from_be_bytes([
                                m.bus.rdram.data[pa + i*4],
                                m.bus.rdram.data[pa + i*4 + 1],
                                m.bus.rdram.data[pa + i*4 + 2],
                                m.bus.rdram.data[pa + i*4 + 3],
                            ]);
                            stk.push_str(&format!(" {:08X}", w));
                            if i % 8 == 7 { stk.push('\n'); }
                        }
                        eprintln!("[PRINTF #0] fatal_printf sp=0x{:08X} frame:\n{}", sp, stk);
                    }
                }
            }
        }
        // Hook entry to the `exit(...)` panic function at 0x8019BB0C.
        // Log caller RA the first few times so we can identify the panic path.
        if pc == 0x8019_BB0C {
            static EXIT_HOOK: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            let n = EXIT_HOOK.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n < 8 {
                let ra = m.cpu.regs[31] as u32;
                let a0 = m.cpu.regs[4] as u32;
                let a1 = m.cpu.regs[5] as u32;
                let a2 = m.cpu.regs[6] as u32;
                let sp = m.cpu.regs[29] as u32;
                eprintln!("[EXIT #{}] f={} s={} pc=0x{:08X} ra=0x{:08X} a0=0x{:08X} a1=0x{:08X} a2=0x{:08X} sp=0x{:08X}",
                    n, before, steps, pc, ra, a0, a1, a2, sp);
                // Dump 32 bytes from sp for stack trace
                let pa = (sp & 0x1FFFFFFF) as usize;
                if pa + 64 <= m.bus.rdram.data.len() {
                    let mut stk = String::new();
                    for i in 0..16 {
                        let w = u32::from_be_bytes([
                            m.bus.rdram.data[pa + i*4],
                            m.bus.rdram.data[pa + i*4 + 1],
                            m.bus.rdram.data[pa + i*4 + 2],
                            m.bus.rdram.data[pa + i*4 + 3],
                        ]);
                        stk.push_str(&format!(" {:08X}", w));
                    }
                    eprintln!("[EXIT #{}] stack@sp:{}", n, stk);
                }
            }
        }
        if pc == 0x8032_2800 {
            static RECVMESG_LOG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            let n = RECVMESG_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if before >= 305 || n < 30 {
                let mq = m.cpu.regs[4] as u32;
                let msg = m.cpu.regs[5] as u32;
                let flag = m.cpu.regs[6] as u32;
                let ra = m.cpu.regs[31] as u32;
                let running = {
                    let pa = 0x3359B0usize;
                    if pa + 4 <= m.bus.rdram.data.len() {
                        u32::from_be_bytes([
                            m.bus.rdram.data[pa], m.bus.rdram.data[pa+1],
                            m.bus.rdram.data[pa+2], m.bus.rdram.data[pa+3],
                        ])
                    } else { 0 }
                };
                let tid = if running >= 0x80000000 {
                    let pa = (running & 0x1FFFFFFF) as usize + 0x14;
                    if pa + 4 <= m.bus.rdram.data.len() {
                        u32::from_be_bytes([
                            m.bus.rdram.data[pa], m.bus.rdram.data[pa+1],
                            m.bus.rdram.data[pa+2], m.bus.rdram.data[pa+3],
                        ])
                    } else { 0 }
                } else { 0 };
                eprintln!(
                    "[RECV #{}] f={} s={} thread={} mq=0x{:08X} msg=0x{:08X} flag={} ra=0x{:08X}",
                    n, before, steps, tid, mq, msg, flag, ra,
                );
            }
        }

        // Hook entry to the SM64 SP-done handler (0x802467FC) and DP-done
        // handler (0x80246_94C) to verify whether the scheduler ever sees a
        // task-completion message after the first audio task runs.
        if pc == 0x8024_67FC || pc == 0x8024_694C {
            static SPDONE_LOG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            let n = SPDONE_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n < 20 {
                eprintln!("[HANDLER] f={} s={} pc=0x{:08X} ra=0x{:08X}",
                    before, steps, pc, m.cpu.regs[31] as u32);
            }
        }

        // Hook entry to __osDispatchEvent (0x80327B98) so we can confirm that
        // the libultra exception path is reached at all after an SP IRQ.
        if pc == 0x8032_7B98 {
            static DISPEV_LOG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            let n = DISPEV_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n < 30 {
                let a0 = m.cpu.regs[4] as u32;
                let mi_intr = m.bus.mi.intr;
                eprintln!("[DISPEV #{}] f={} s={} a0=0x{:X} mi.intr=0x{:02X}",
                    n, before, steps, a0, mi_intr);
            }
        }

        // Hook the SM64 scheduler dispatch jr at 0x80246A94. By the time we
        // reach this instruction, $t2 (reg 10) holds the message just received
        // (loaded at 0x80246A70 as `lw t2, 0x24(sp)`) and $t3 (reg 11) holds
        // the jump-table target (loaded at 0x80246A90 as `lw t3, 0x5B60(at)`).
        // We log the (msg → handler) mapping every time the dispatcher fires.
        if pc == 0x8024_6A94 {
            static DISP_LOG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            let n = DISP_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if before >= 315 || n < 6 {
                let msg = m.cpu.regs[10] as u32;
                let target = m.cpu.regs[11] as u32;
                eprintln!("[DISPATCH #{}] f={} s={} msg=0x{:X} -> 0x{:08X}", n, before, steps, msg, target);
            }
        }

        // Capture PCs after __osViMgrThread recv
        let cap_left = VIMGR_CAPTURE_LEFT.load(std::sync::atomic::Ordering::Relaxed);
        if cap_left > 0 {
            let paddr = (pc & 0x1FFF_FFFF) as usize;
            let word = if paddr + 4 <= m.bus.rdram.data.len() {
                u32::from_be_bytes([
                    m.bus.rdram.data[paddr], m.bus.rdram.data[paddr+1],
                    m.bus.rdram.data[paddr+2], m.bus.rdram.data[paddr+3],
                ])
            } else { 0 };
            eprintln!("[VIMGR_TRACE {}] PC=0x{:08X} ({:08X}) ra=0x{:08X}",
                800 - cap_left, pc, word, m.cpu.regs[31] as u32);
            VIMGR_CAPTURE_LEFT.store(cap_left - 1, std::sync::atomic::Ordering::Relaxed);
        }

        // Track ERET destinations across ALL frames for thread analysis
        {
            let paddr = (pc & 0x1FFF_FFFF) as usize;
            if paddr + 4 <= m.bus.rdram.data.len() {
                let word = u32::from_be_bytes([
                    m.bus.rdram.data[paddr],
                    m.bus.rdram.data[paddr + 1],
                    m.bus.rdram.data[paddr + 2],
                    m.bus.rdram.data[paddr + 3],
                ]);
                if word == 0x4200_0018 {  // ERET
                    let epc = m.cpu.cop0.epc as u32;
                    if let Ok(mut targets) = ERET_TARGETS.try_lock() {
                        *targets.entry(epc).or_insert(0) += 1;
                    }
                }
            }
        }

        // Track transition to idle loop — log the last few PCs before entering idle
        {
            static IDLE_LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            static PREV_PC: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            static PREV_PREV_PC: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if pc == 0x8024_6DD8 && !IDLE_LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                let p1 = PREV_PC.load(std::sync::atomic::Ordering::Relaxed);
                let p2 = PREV_PREV_PC.load(std::sync::atomic::Ordering::Relaxed);
                eprintln!("[IDLE] First entry to idle at frame {}, step {}", before, steps);
                eprintln!("  Previous PCs: 0x{:08X} -> 0x{:08X} -> idle", p2, p1);
                eprintln!("  RA=0x{:08X} SP=0x{:08X}", m.cpu.regs[31] as u32, m.cpu.regs[29] as u32);
                eprintln!("  COP0.EPC=0x{:08X} Status=0x{:08X}", m.cpu.cop0.epc as u32, m.cpu.cop0.status);
            }
            PREV_PREV_PC.store(PREV_PC.load(std::sync::atomic::Ordering::Relaxed), std::sync::atomic::Ordering::Relaxed);
            PREV_PC.store(pc, std::sync::atomic::Ordering::Relaxed);
        }

        // Track game code execution
        static GAME_PC_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if pc >= 0x8037_0000 && pc < 0x8040_0000 {
            let count = GAME_PC_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if count < 20 {
                let paddr = (pc & 0x1FFF_FFFF) as usize;
                let word = if paddr + 4 <= m.bus.rdram.data.len() {
                    u32::from_be_bytes([
                        m.bus.rdram.data[paddr],
                        m.bus.rdram.data[paddr + 1],
                        m.bus.rdram.data[paddr + 2],
                        m.bus.rdram.data[paddr + 3],
                    ])
                } else { 0 };
                eprintln!("[GAME #{:5}] PC=0x{:08X} ({:08X}) RA=0x{:08X}",
                    count, pc, word, m.cpu.regs[31] as u32);
            }
            if !GAME_CODE_FOUND.swap(true, std::sync::atomic::Ordering::Relaxed) {
                GAME_CODE_PC.store(pc, std::sync::atomic::Ordering::Relaxed);
                GAME_CODE_FRAME.store(before, std::sync::atomic::Ordering::Relaxed);
            }
        }

        // Log game thread PCs after frame 316 to see what it's doing post-timer
        static POST_TIMER_LOG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if before >= 316 && before <= 320 {
            // Log first 30 PCs from each of frames 316-320 in game code range
            if pc >= 0x8024_0000 && pc < 0x8040_0000 {
                let count = POST_TIMER_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if count < 100 && (count < 30 || count % 10 == 0) {
                    eprintln!("[POST_TIMER f={} s={}] PC=0x{:08X} RA=0x{:08X}",
                        before, steps, pc, m.cpu.regs[31] as u32);
                }
            }
        }

        // Monitor scheduler client list at 0x8033ADE0 to see if osScAddClient is called
        static CLIENT_LIST_PREV: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0xFFFF_FFFF);
        static CLIENT_LIST_LOG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let pa_cl = 0x0033ADE0usize;
        let snap_cl: u32 = if pa_cl + 4 <= m.bus.rdram.data.len() {
            u32::from_be_bytes([
                m.bus.rdram.data[pa_cl], m.bus.rdram.data[pa_cl+1],
                m.bus.rdram.data[pa_cl+2], m.bus.rdram.data[pa_cl+3],
            ])
        } else { 0 };
        let prev_cl = CLIENT_LIST_PREV.load(std::sync::atomic::Ordering::Relaxed);
        if prev_cl != 0xFFFF_FFFF && snap_cl != prev_cl {
            let n = CLIENT_LIST_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n < 10 {
                eprintln!("[SCHED_CLIENT_LIST] f={} s={} PC=0x{:08X} prev=0x{:08X} now=0x{:08X}",
                    before, steps, pc, prev_cl, snap_cl);
            }
        }
        CLIENT_LIST_PREV.store(snap_cl, std::sync::atomic::Ordering::Relaxed);

        // Monitor __osRunQueue at 0x8033A730 AND 0x803359A8 (game's actual queue)
        static RUNQ_PREV: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0xFFFF_FFFF);
        static RUNQ_LOG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        static RUNQ59A8_PREV: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0xFFFF_FFFF);
        static RUNQ59A8_LOG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

        let pa_rq = 0x0033A730usize;
        let snap_rq: u32 = if pa_rq + 4 <= m.bus.rdram.data.len() {
            u32::from_be_bytes([
                m.bus.rdram.data[pa_rq], m.bus.rdram.data[pa_rq+1],
                m.bus.rdram.data[pa_rq+2], m.bus.rdram.data[pa_rq+3],
            ])
        } else { 0 };
        let prev_rq = RUNQ_PREV.load(std::sync::atomic::Ordering::Relaxed);
        if prev_rq != 0xFFFF_FFFF && snap_rq != prev_rq {
            let n = RUNQ_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n < 30 {
                eprintln!("[RUNQUEUE@A730] f={} s={} PC=0x{:08X} prev=0x{:08X} now=0x{:08X}",
                    before, steps, pc, prev_rq, snap_rq);
            }
        }
        RUNQ_PREV.store(snap_rq, std::sync::atomic::Ordering::Relaxed);

        // Also monitor the game's actual queue at 0x803359A8
        let pa_rq2 = 0x003359A8usize;
        let snap_rq2: u32 = if pa_rq2 + 4 <= m.bus.rdram.data.len() {
            u32::from_be_bytes([
                m.bus.rdram.data[pa_rq2], m.bus.rdram.data[pa_rq2+1],
                m.bus.rdram.data[pa_rq2+2], m.bus.rdram.data[pa_rq2+3],
            ])
        } else { 0 };
        let prev_rq2 = RUNQ59A8_PREV.load(std::sync::atomic::Ordering::Relaxed);
        if prev_rq2 != 0xFFFF_FFFF && snap_rq2 != prev_rq2 && (before >= 315 && before <= 320 || before <= 15) {
            let n = RUNQ59A8_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n < 50 {
                let label = match snap_rq2 {
                    0x8033AA90 => "gameThread",
                    0x803359A0 => "idle",
                    0x8033A730 => "__osRunQueue_addr",
                    _ => "?"
                };
                eprintln!("[RUNQUEUE@59A8] f={} s={} PC=0x{:08X} prev=0x{:08X} now=0x{:08X}({})",
                    before, steps, pc, prev_rq2, snap_rq2, label);
            }
        }
        RUNQ59A8_PREV.store(snap_rq2, std::sync::atomic::Ordering::Relaxed);

        // At frame 316 step ~150505, dump code at 0x80322E30-0x80322E60 (osSendMesg wake path)
        // AND dump __osEnqueueThread at 0x80327D10
        static FRAME316_DUMP_CODE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if before == 316 && steps == 150505 && !FRAME316_DUMP_CODE.swap(true, std::sync::atomic::Ordering::Relaxed) {
            eprintln!("\n=== CODE DUMP: osSendMesg wake path (0x80322E30-0x80322E70) ===");
            for addr in (0x80322E30u32..0x80322E70).step_by(4) {
                let pa = (addr & 0x1FFF_FFFF) as usize;
                let word = if pa + 4 <= m.bus.rdram.data.len() {
                    u32::from_be_bytes([m.bus.rdram.data[pa], m.bus.rdram.data[pa+1], m.bus.rdram.data[pa+2], m.bus.rdram.data[pa+3]])
                } else { 0 };
                eprintln!("  0x{:08X}: {:08X}", addr, word);
            }
            eprintln!("\n=== CODE DUMP: function at 0x80327D10 (called with a0=0x803359A8) ===");
            for addr in (0x80327D10u32..0x80327D80).step_by(4) {
                let pa = (addr & 0x1FFF_FFFF) as usize;
                let word = if pa + 4 <= m.bus.rdram.data.len() {
                    u32::from_be_bytes([m.bus.rdram.data[pa], m.bus.rdram.data[pa+1], m.bus.rdram.data[pa+2], m.bus.rdram.data[pa+3]])
                } else { 0 };
                eprintln!("  0x{:08X}: {:08X}", addr, word);
            }
            eprintln!("=================================\n");
        }

        // At frame 316 step ~150510, dump run queue CORRECTLY
        static FRAME316_DUMP3: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if before == 316 && steps >= 150508 && steps <= 150515 && !FRAME316_DUMP3.swap(true, std::sync::atomic::Ordering::Relaxed) {
            let read_w = |addr: u32| -> u32 {
                let pa = (addr & 0x1FFF_FFFF) as usize;
                if pa + 4 <= m.bus.rdram.data.len() {
                    u32::from_be_bytes([m.bus.rdram.data[pa], m.bus.rdram.data[pa+1], m.bus.rdram.data[pa+2], m.bus.rdram.data[pa+3]])
                } else { 0 }
            };
            let read_h = |addr: u32| -> u16 {
                let pa = (addr & 0x1FFF_FFFF) as usize;
                if pa + 2 <= m.bus.rdram.data.len() {
                    u16::from_be_bytes([m.bus.rdram.data[pa], m.bus.rdram.data[pa+1]])
                } else { 0 }
            };
            eprintln!("\n=== FRAME 316 RUN QUEUE ANALYSIS ===");
            eprintln!("  step={} PC=0x{:08X}", steps, pc);
            eprintln!("  Checking TWO possible run queue addresses:");
            eprintln!("  ");
            eprintln!("  Option A: __osRunQueue at 0x8033A730:");
            eprintln!("    [0x8033A730] = 0x{:08X}", read_w(0x8033A730));
            eprintln!("    [0x8033A734] = 0x{:08X}", read_w(0x8033A734));
            eprintln!("    [0x8033A738] = 0x{:08X}", read_w(0x8033A738));
            eprintln!("  ");
            eprintln!("  Option B: game uses 0x803359A8 as queue:");
            eprintln!("    [0x803359A8] = 0x{:08X} (read as queue head)", read_w(0x803359A8));
            // What __osEnqueueThread sees:
            let a0 = 0x803359A8u32;
            let t8 = read_w(a0);  // first read: *a0
            eprintln!("    After lw t8, 0(a0):  t8 = 0x{:08X}", t8);
            let t6 = read_w(t8 + 4);  // read "priority"
            eprintln!("    After lw t6, 4(t8):  t6 = 0x{:08X} (treated as priority)", t6);
            eprintln!("  ");
            // gameThread info
            let gt = 0x8033AA90u32;
            let gt_prio = read_w(gt + 4) as i32;
            eprintln!("  gameThread @0x{:08X}: priority={}", gt, gt_prio);
            eprintln!("  Comparison: is {} < {} ?  {}", t6 as i32, gt_prio, (t6 as i32) < gt_prio);
            eprintln!("  ");
            eprintln!("  AFTER __osEnqueueThread returns, check both locations:");
        }

        // Dump state AFTER __osEnqueueThread (around step 150550)
        static FRAME316_DUMP4: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if before == 316 && steps >= 150545 && steps <= 150555 && !FRAME316_DUMP4.swap(true, std::sync::atomic::Ordering::Relaxed) {
            let read_w = |addr: u32| -> u32 {
                let pa = (addr & 0x1FFF_FFFF) as usize;
                if pa + 4 <= m.bus.rdram.data.len() {
                    u32::from_be_bytes([m.bus.rdram.data[pa], m.bus.rdram.data[pa+1], m.bus.rdram.data[pa+2], m.bus.rdram.data[pa+3]])
                } else { 0 }
            };
            eprintln!("  step={} PC=0x{:08X} (after __osEnqueueThread)", steps, pc);
            eprintln!("    [0x8033A730] = 0x{:08X} (real __osRunQueue)", read_w(0x8033A730));
            eprintln!("    [0x803359A8] = 0x{:08X} (where game adds threads)", read_w(0x803359A8));
            let gt = 0x8033AA90u32;
            eprintln!("    gameThread->next  = 0x{:08X}", read_w(gt));
            eprintln!("    gameThread->queue = 0x{:08X}", read_w(gt + 8));
            eprintln!("    ");
            eprintln!("    Double indirection check:");
            eprintln!("    [0x8033A738] = 0x{:08X}", read_w(0x8033A738));
            let ptr1 = read_w(0x8033A738);
            if ptr1 >= 0x80000000 && ptr1 < 0x80800000 {
                eprintln!("    [[0x8033A738]] = [0x{:08X}] = 0x{:08X}", ptr1, read_w(ptr1));
            }
            // Dump code and relevant data
            eprintln!("    ");
            eprintln!("    Relevant scheduler data:");
            eprintln!("    [0x803359B0] = 0x{:08X} (__osRunningThread?)", read_w(0x803359B0));
            eprintln!("    [0x803359B4] = 0x{:08X}", read_w(0x803359B4));
            eprintln!("    [0x803359B8] = 0x{:08X}", read_w(0x803359B8));
            eprintln!("=================================\n");
        }

        // At frame 316 step ~151090, dump queue 0x803359A8 that gameThread blocks on
        static FRAME316_DUMP2: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if before == 316 && steps >= 151080 && steps <= 151090 && !FRAME316_DUMP2.swap(true, std::sync::atomic::Ordering::Relaxed) {
            let read_w = |addr: u32| -> u32 {
                let pa = (addr & 0x1FFF_FFFF) as usize;
                if pa + 4 <= m.bus.rdram.data.len() {
                    u32::from_be_bytes([m.bus.rdram.data[pa], m.bus.rdram.data[pa+1], m.bus.rdram.data[pa+2], m.bus.rdram.data[pa+3]])
                } else { 0 }
            };
            eprintln!("\n=== FRAME 316 QUEUE 0x803359A8 DUMP ===");
            let q = 0x803359A8u32;
            eprintln!("  OSMesgQueue @0x{:08X}:", q);
            eprintln!("    mtqueue    = 0x{:08X}", read_w(q + 0x00));
            eprintln!("    fullqueue  = 0x{:08X}", read_w(q + 0x04));
            eprintln!("    validCount = {}", read_w(q + 0x08) as i32);
            eprintln!("    first      = {}", read_w(q + 0x0C) as i32);
            eprintln!("    msgCount   = {}", read_w(q + 0x10) as i32);
            eprintln!("    msg*       = 0x{:08X}", read_w(q + 0x14));
            // Also check what this queue is - look at nearby structures
            eprintln!("  Nearby memory (0x803359A0-0x803359C0):");
            for off in (0x003359A0usize..0x003359C0).step_by(4) {
                if off + 4 <= m.bus.rdram.data.len() {
                    let w = u32::from_be_bytes([
                        m.bus.rdram.data[off], m.bus.rdram.data[off+1],
                        m.bus.rdram.data[off+2], m.bus.rdram.data[off+3]
                    ]);
                    eprintln!("    [0x{:08X}] = 0x{:08X}", 0x8000_0000 + off, w);
                }
            }
            eprintln!("=================================\n");
        }

        // At frame 316 step ~150480, dump gameThread state
        static FRAME316_DUMP: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if before == 316 && steps >= 150475 && steps <= 150485 && !FRAME316_DUMP.swap(true, std::sync::atomic::Ordering::Relaxed) {
            let read_w = |addr: u32| -> u32 {
                let pa = (addr & 0x1FFF_FFFF) as usize;
                if pa + 4 <= m.bus.rdram.data.len() {
                    u32::from_be_bytes([
                        m.bus.rdram.data[pa], m.bus.rdram.data[pa+1],
                        m.bus.rdram.data[pa+2], m.bus.rdram.data[pa+3],
                    ])
                } else { 0 }
            };
            let read_h = |addr: u32| -> u16 {
                let pa = (addr & 0x1FFF_FFFF) as usize;
                if pa + 2 <= m.bus.rdram.data.len() {
                    u16::from_be_bytes([m.bus.rdram.data[pa], m.bus.rdram.data[pa+1]])
                } else { 0 }
            };
            eprintln!("\n=== FRAME 316 GAMETHREAD DUMP ===");
            eprintln!("  PC=0x{:08X} step={}", pc, steps);
            let gt = 0x8033AA90u32;
            eprintln!("  gameThread @0x{:08X}:", gt);
            eprintln!("    next     = 0x{:08X}", read_w(gt + 0x00));
            eprintln!("    priority = {}", read_w(gt + 0x04) as i32);
            eprintln!("    queue    = 0x{:08X}", read_w(gt + 0x08));
            eprintln!("    state    = {} ({})", read_h(gt + 0x10),
                match read_h(gt + 0x10) { 1 => "RUNNING", 2 => "RUNNABLE", 4 => "STOPPED", 8 => "WAITING", _ => "?" });
            eprintln!("    id       = {}", read_w(gt + 0x14));
            eprintln!("  __osRunQueue = 0x{:08X}", read_w(0x8033A730));
            eprintln!("=================================\n");
        }

        // Monitor gameThread (0x8033AA90) state field at +0x10 (state is u16)
        static GT_STATE_PREV: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0xFFFF_FFFF);
        static GT_STATE_LOG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let pa_gt_state = 0x0033AA90usize + 0x10;
        let snap_gt_state: u32 = if pa_gt_state + 2 <= m.bus.rdram.data.len() {
            u16::from_be_bytes([
                m.bus.rdram.data[pa_gt_state], m.bus.rdram.data[pa_gt_state+1],
            ]) as u32
        } else { 0 };
        let prev_gt_state = GT_STATE_PREV.load(std::sync::atomic::Ordering::Relaxed);
        if prev_gt_state != 0xFFFF_FFFF && snap_gt_state != prev_gt_state {
            let n = GT_STATE_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let state_str = match snap_gt_state {
                1 => "RUNNING", 2 => "RUNNABLE", 4 => "STOPPED", 8 => "WAITING", _ => "?"
            };
            if n < 100 || before >= 310 {
                // Also read gameThread's queue field at +0x08
                let pa_gt_queue = 0x0033AA90usize + 0x08;
                let gt_queue: u32 = if pa_gt_queue + 4 <= m.bus.rdram.data.len() {
                    u32::from_be_bytes([
                        m.bus.rdram.data[pa_gt_queue], m.bus.rdram.data[pa_gt_queue+1],
                        m.bus.rdram.data[pa_gt_queue+2], m.bus.rdram.data[pa_gt_queue+3],
                    ])
                } else { 0 };
                // If blocking on an invalid queue, dump more context
                let queue_label = match gt_queue {
                    0x80206D60 => "timer_queue",
                    0x80367158 => "gameThread_work_queue",
                    0x8033ADF0 => "scheduler_queue",
                    0x803359A8 => "INVALID(idle_thread_field)",
                    _ => "unknown"
                };
                let ra = m.cpu.regs[31] as u32;
                let a0 = m.cpu.regs[4] as u32;
                let a1 = m.cpu.regs[5] as u32;
                // When going to WAITING, also show $a0 which is often the osRecvMesg queue arg
                if snap_gt_state == 8 {
                    eprintln!("[GAMETHREAD_STATE] f={} s={} PC=0x{:08X} RA=0x{:08X} state {}->{}({}) queue=0x{:08X}({}) recvmq_arg=$a0=0x{:08X} $a1=0x{:08X}",
                        before, steps, pc, ra, prev_gt_state, snap_gt_state, state_str, gt_queue, queue_label, a0, a1);
                } else {
                    eprintln!("[GAMETHREAD_STATE] f={} s={} PC=0x{:08X} RA=0x{:08X} state {}->{}({}) queue=0x{:08X}({})",
                        before, steps, pc, ra, prev_gt_state, snap_gt_state, state_str, gt_queue, queue_label);
                }
            }
        }
        GT_STATE_PREV.store(snap_gt_state, std::sync::atomic::Ordering::Relaxed);

        match m.step() {
            Ok(()) => steps += 1,
            Err(e) => {
                eprintln!("CPU halted: {:?}", e);
                *cpu_halted = true;
                break;
            }
        }
    }

    // Track unique thread dispatch targets (PC after ERET)
    static ERET_TARGETS: std::sync::Mutex<std::collections::BTreeMap<u32, u32>> =
        std::sync::Mutex::new(std::collections::BTreeMap::new());
    // Print summary at frame 100
    if before == 100 && !TRACED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        if let Ok(targets) = ERET_TARGETS.try_lock() {
            let total: u32 = targets.values().sum();
            eprintln!("\n--- Thread dispatch targets (ERET destinations, total={}) ---", total);
            for (pc, count) in targets.iter() {
                let label = if *pc >= 0x8024_6DD0 && *pc <= 0x8024_6DE0 {
                    " (idle thread)"
                } else if *pc >= 0x8032_0000 && *pc < 0x8033_0000 {
                    " (libultra)"
                } else if *pc >= 0x8037_0000 && *pc < 0x8040_0000 {
                    " (game code)"
                } else {
                    ""
                };
                eprintln!("  0x{:08X}: {} times{}", pc, count, label);
            }
        }
        eprintln!("\n=== FRAME 100 SUMMARY ===");
        if GAME_CODE_FOUND.load(std::sync::atomic::Ordering::Relaxed) {
            let pc = GAME_CODE_PC.load(std::sync::atomic::Ordering::Relaxed);
            let frame = GAME_CODE_FRAME.load(std::sync::atomic::Ordering::Relaxed);
            eprintln!("  Game code was reached at frame {} (PC=0x{:08X})", frame, pc);
        } else {
            eprintln!("  Game code (0x8037xxxx) was NEVER reached!");
        }
        eprintln!("  VI_ORIGIN: 0x{:08X}", m.bus.vi.regs[ultrareality::VI_REG_ORIGIN]);
        eprintln!("  SI DMA busy: {}", m.bus.si.dma_busy());
        eprintln!("  Current PC: 0x{:08X}", m.cpu.pc as u32);
        eprintln!("  SP halted: {}, SP broke: {}", m.bus.sp_halted, m.bus.sp_broke);
        eprintln!("  MI intr: 0x{:02X}, mask: 0x{:02X}", m.bus.mi.intr, m.bus.mi.mask);

        // Dump idle loop code to understand what it does
        eprintln!("\n--- Idle loop code (0x80246DD0-0x80246DF0) ---");
        for addr in (0x80246DD0u32..0x80246DF4).step_by(4) {
            let paddr = (addr & 0x1FFF_FFFF) as usize;
            let word = if paddr + 4 <= m.bus.rdram.data.len() {
                u32::from_be_bytes([
                    m.bus.rdram.data[paddr],
                    m.bus.rdram.data[paddr + 1],
                    m.bus.rdram.data[paddr + 2],
                    m.bus.rdram.data[paddr + 3],
                ])
            } else { 0 };
            eprintln!("  0x{:08X}: {:08X}", addr, word);
        }

        // Dump __osRunQueue and nearby thread queue pointers
        // SM64 typical addresses: __osRunQueue ~0x8033A730
        // Try to find thread queue by scanning for known patterns
        eprintln!("\n--- Thread queue scan ---");
        // The idle thread PC 0x80246DD8 should be stored in a thread struct's context.
        // COP0 status reg at frame 100:
        eprintln!("  COP0 Status: 0x{:08X}", m.cpu.cop0.status);
        eprintln!("  COP0 Cause:  0x{:08X}", m.cpu.cop0.cause);
        eprintln!("  COP0 EPC:    0x{:08X}", m.cpu.cop0.epc as u32);
        eprintln!("  COP0 Count:  0x{:08X}", m.cpu.cop0.count);
        eprintln!("  COP0 Compare:0x{:08X}", m.cpu.cop0.compare);

        // Check for pending Count/Compare timer interrupt
        let count_cmp = if m.cpu.cop0.compare != 0 && m.cpu.cop0.count >= m.cpu.cop0.compare {
            "EXPIRED"
        } else {
            "not expired"
        };
        eprintln!("  Timer: count={} compare={} ({})", m.cpu.cop0.count, m.cpu.cop0.compare, count_cmp);
        eprintln!("  VI_V_INTR: {} (0=disabled)", m.bus.vi.regs[ultrareality::VI_REG_V_INTR] & 0x3FF);
        eprintln!("  VI_INT_RAISES: {}",
            ultrareality::vi::VI_INT_RAISE_COUNT.load(std::sync::atomic::Ordering::Relaxed));
        eprintln!("  TOTAL_INT_TAKEN: {}",
            ultrareality::cpu::cop0::INT_TAKEN_COUNT.load(std::sync::atomic::Ordering::Relaxed));
        eprintln!("  ERET_EXEC: {}",
            ultrareality::cpu::cop0::ERET_EXEC_COUNT.load(std::sync::atomic::Ordering::Relaxed));
        let exc_names = ["Int","Mod","TLBL","TLBS","AdEL","AdES","IBE","DBE","Sys","Bp","RI","CpU","Ov","Tr","FPE","WATCH"];
        for (i, name) in exc_names.iter().enumerate() {
            let n = ultrareality::cpu::cop0::GEN_EXC_COUNT[i].load(std::sync::atomic::Ordering::Relaxed);
            if n > 0 {
                eprintln!("  GEN_EXC[{}/{}]: {}", i, name, n);
            }
        }

        // Check osRecvMesg queue state by looking at thread context
        // Scan for thread structs: look for known PC values in memory
        eprintln!("\n--- Scanning for blocked thread contexts ---");
        // libultra OSThread has state at offset 0x10 relative to thread pointer
        // state: OS_STATE_RUNNING=1, OS_STATE_RUNNABLE=2, OS_STATE_WAITING=8, OS_STATE_STOPPED=4
        // We look for words matching known thread PCs
        let known_pcs: &[(u32, &str)] = &[
            (0x80248AF0, "startup"),
            (0x802781A0, "piMgr?"),
            (0x802785A4, "viMgr?"),
            (0x8027F528, "gameThread?"),
            (0x80317088, "game1"),
            (0x803195A8, "game2"),
            (0x8031AC50, "game3"),
            (0x8031E284, "game4"),
            (0x80322868, "libultra_main"),
        ];
        // Search RDRAM for thread queue pointers
        // __osRunQueue should be a pointer to first thread on run queue
        // In SM64, it's around 0x8033A730
        let read_w = |addr: u32| -> u32 {
            let pa = (addr & 0x1FFF_FFFF) as usize;
            if pa + 4 <= m.bus.rdram.data.len() {
                u32::from_be_bytes([
                    m.bus.rdram.data[pa],
                    m.bus.rdram.data[pa+1],
                    m.bus.rdram.data[pa+2],
                    m.bus.rdram.data[pa+3],
                ])
            } else { 0 }
        };
        let read_h = |addr: u32| -> u16 {
            let pa = (addr & 0x1FFF_FFFF) as usize;
            if pa + 2 <= m.bus.rdram.data.len() {
                u16::from_be_bytes([m.bus.rdram.data[pa], m.bus.rdram.data[pa+1]])
            } else { 0 }
        };
        for test_addr in [0x8033A730u32, 0x8033A734, 0x80339ED0u32, 0x80339EC0u32] {
            eprintln!("  [0x{:08X}] = 0x{:08X}", test_addr, read_w(test_addr));
        }

        // Walk run queue: each OSThread has next ptr at 0
        // Fields: 0x00 next, 0x04 priority, 0x08 queue, 0x0C tlnext, 0x10 state(u16), 0x12 flags(u16), 0x14 id, 0x20+ context
        // Context: 0x18 context.pc (offset 0x20+0x118 in OSThread) -- complex; just dump first 0x40 bytes
        eprintln!("\n--- Walk __osRunQueue (chase next pointers) ---");
        let mut t = read_w(0x8033A730);
        for i in 0..10 {
            if t == 0 || t == 0xFFFFFFFF { break; }
            let next = read_w(t + 0x00);
            let prio = read_w(t + 0x04) as i32;
            let queue = read_w(t + 0x08);
            let state = read_h(t + 0x10);
            let flags = read_h(t + 0x12);
            let id = read_w(t + 0x14);
            let state_str = match state {
                1 => "RUNNING",
                2 => "RUNNABLE",
                4 => "STOPPED",
                8 => "WAITING",
                _ => "?",
            };
            eprintln!("  [{}] thread@0x{:08X} next=0x{:08X} prio={} queue=0x{:08X} state={}({}) flags=0x{:04X} id={}",
                i, t, next, prio, queue, state, state_str, flags, id);
            if next == t { break; } // self-loop
            t = next;
        }

        // Dump exception handler at 0x80000180
        eprintln!("\n--- Exception handler at 0x80000180 ---");
        for addr in (0x80000180u32..0x800001A0).step_by(4) {
            eprintln!("  0x{:08X}: {:08X}", addr, read_w(addr));
        }
        // Dump real handler at 0x80327650
        eprintln!("\n--- Real handler at 0x80327650 ---");
        for addr in (0x80327650u32..0x80327750).step_by(4) {
            eprintln!("  0x{:08X}: {:08X}", addr, read_w(addr));
        }

        // Dump key libultra code regions
        eprintln!("\n--- Code around 0x80322868 (wider window) ---");
        for addr in (0x80322820u32..0x803228C0).step_by(4) {
            let marker = if addr == 0x80322868 { " <-- ERET LANDING" } else { "" };
            eprintln!("  0x{:08X}: {:08X}{}", addr, read_w(addr), marker);
        }
        eprintln!("\n--- Code at idle JAL target 0x803236F0 ---");
        for addr in (0x803236F0u32..0x80323730).step_by(4) {
            eprintln!("  0x{:08X}: {:08X}", addr, read_w(addr));
        }
        eprintln!("\n--- Code at 0x80322F24 (2nd most dispatched) ---");
        for addr in (0x80322F24u32..0x80322F64).step_by(4) {
            eprintln!("  0x{:08X}: {:08X}", addr, read_w(addr));
        }

        // Find IDLE thread struct: scan for fields containing PC=0x80246DD8
        eprintln!("\n--- Scan for OSThread containing pc=0x80246DD8 ---");
        let target_pc = 0x80246DD8u32;
        let mut hits = 0;
        for a in (0x80300000u32..0x80400000u32).step_by(4) {
            if read_w(a) == target_pc {
                eprintln!("  found 0x{:08X} at addr 0x{:08X}", target_pc, a);
                hits += 1;
                if hits > 20 { break; }
            }
        }

        // Dump 0x803359A0 (the "sentinel" all waiting threads point at)
        eprintln!("\n--- Dump 0x803359A0 (shared sentinel?) ---");
        for off in (0..0x40u32).step_by(4) {
            eprintln!("  0x{:08X}: {:08X}", 0x803359A0u32 + off, read_w(0x803359A0u32 + off));
        }

        // Dump full OSThread struct for each WAITING candidate
        eprintln!("\n--- Full OSThread struct dumps for WAITING threads ---");
        for thr in [0x8033AA90u32, 0x80364C60, 0x80365E70] {
            eprintln!("  Thread @0x{:08X}:", thr);
            eprintln!("    next=0x{:08X}  prio={}  queue=0x{:08X}  tlnext=0x{:08X}",
                read_w(thr+0x00), read_w(thr+0x04) as i32, read_w(thr+0x08), read_w(thr+0x0C));
            eprintln!("    state={}  flags=0x{:04X}  id={}",
                read_h(thr+0x10), read_h(thr+0x12), read_w(thr+0x14));
            // Try to find PC: try a few common offsets
            for off in [0x118u32, 0x11C, 0x120, 0x13C, 0x140] {
                let v = read_w(thr + 0x20 + off);
                if v >= 0x80000000 && v < 0x80400000 {
                    eprintln!("    pc-candidate@+0x20+0x{:03X}=0x{:08X}", off, v);
                }
            }
        }

        // Dump message queues that WAITING threads are blocked on.
        // OSMesgQueue: 0x00 mtqueue, 0x04 fullqueue, 0x08 validCount, 0x0C first, 0x10 msgCount, 0x14 msg*
        eprintln!("\n--- Dump message queues that block threads ---");
        for q_addr in [0x80206D60u32, 0x80365E10, 0x8033ADF0] {
            let mt = read_w(q_addr + 0x00);
            let full = read_w(q_addr + 0x04);
            let valid = read_w(q_addr + 0x08) as i32;
            let first = read_w(q_addr + 0x0C) as i32;
            let mcount = read_w(q_addr + 0x10) as i32;
            let mptr = read_w(q_addr + 0x14);
            eprintln!("  Q@0x{:08X}: mt=0x{:08X} full=0x{:08X} valid={} first={} count={} msg*=0x{:08X}",
                q_addr, mt, full, valid, first, mcount, mptr);
        }

        // Check viMgr message queue (typical SM64 location)
        // __osViMesgQueue is in libultra; we can search by looking at known structure
        // OSMesgQueue: 0x00 mtqueue, 0x04 fullqueue, 0x08 validCount, 0x0C first, 0x10 msgCount, 0x14 msg ptr
        // Look at common addresses for queue structures
        eprintln!("\n--- Search for waiting threads (state==8) ---");
        // Scan low RDRAM for OSThread structs by looking for state==WAITING(8) at +0x10
        let mut found = 0;
        let mut a = 0x80300000u32;
        while a < 0x80400000 && found < 20 {
            let s = read_h(a + 0x10);
            let f = read_h(a + 0x12);
            // Heuristic: state in {1,2,4,8} and flags small and id reasonable
            if (s == 1 || s == 2 || s == 4 || s == 8) && f < 0x100 {
                let prio = read_w(a + 0x04) as i32;
                let id = read_w(a + 0x14);
                if prio >= 0 && prio < 256 && id < 1000 {
                    let state_str = match s { 1=>"RUNNING",2=>"RUNNABLE",4=>"STOPPED",8=>"WAITING",_=>"?"};
                    let queue = read_w(a + 0x08);
                    eprintln!("  candidate@0x{:08X} prio={} state={} queue=0x{:08X} id={}", a, prio, state_str, queue, id);
                    found += 1;
                    a += 0x200; // skip past this struct
                    continue;
                }
            }
            a += 8;
        }

        // Dump the idle loop region to see if it's a tight WAIT or a spin
        // Also check: does the idle thread enable interrupts?
        eprintln!("=========================\n");
    }

    if m.bus.vi.frame_counter == before {
        m.bus
            .advance_vi_frame_timing(VI_NTSC_CYCLES_PER_FRAME);
    }
}

fn log_fps_with_fb(last_status: &mut Instant, frames: &mut u64, vi_frame: u64, fb_offset: usize, width: usize, height: usize, mi_intr: u32, mi_mask: u32) {
    if last_status.elapsed().as_secs() >= 1 {
        eprintln!(
            "VI frame {} | ~{:.0} fps | FB={:06X} {}x{} | MI=0x{:02X}/0x{:02X} | phase {:?}",
            vi_frame,
            *frames as f64 / last_status.elapsed().as_secs_f64().max(0.001),
            fb_offset, width, height,
            mi_intr, mi_mask,
            graphics_phase_reached()
        );
        *frames = 0;
        *last_status = Instant::now();
    }
}
