//! Ultrareality host binary — placeholder until ROM loading and video/audio exist.

fn main() {
    println!(
        "ultrareality {} — N64 core (development)",
        env!("CARGO_PKG_VERSION")
    );
    let mut m = ultrareality::Machine::new();
    println!(
        "master_cycles after reset: {} (CPU PC = 0x{:016X})",
        m.master_cycles, m.cpu.pc
    );
    if let Err(e) = m.run(1) {
        eprintln!("first step: {:?}", e);
    }
}
