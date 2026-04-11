//! Physical memory and address decoding for early bootstrap / test harness.
//!
//! Maps common segments (kseg0/kseg1/useg low) onto a single linear RDRAM
//! window. A full MMU/TLB and PI/SI/VI/AI/RDP register decoding will replace
//! this piecemeal.

/// Default retail RDRAM size (4 MiB). Expansion Pak (8 MiB) can be enabled later.
pub const DEFAULT_RDRAM_SIZE: usize = 4 * 1024 * 1024;

pub trait Bus {
    fn read_u32(&mut self, paddr: u32) -> Option<u32>;
    fn write_u32(&mut self, paddr: u32, value: u32);
    fn read_u8(&mut self, paddr: u32) -> Option<u8>;
    fn write_u8(&mut self, paddr: u32, value: u8);
}

/// Contiguous big-endian RAM backing store.
pub struct PhysicalMemory {
    pub data: Box<[u8]>,
}

impl PhysicalMemory {
    pub fn new(size: usize) -> Self {
        Self {
            data: vec![0u8; size].into_boxed_slice(),
        }
    }

    #[inline]
    fn in_bounds(&self, paddr: u32, len: u32) -> bool {
        (paddr as u64) + (len as u64) <= self.data.len() as u64
    }
}

impl Bus for PhysicalMemory {
    fn read_u32(&mut self, paddr: u32) -> Option<u32> {
        if !self.in_bounds(paddr, 4) {
            return None;
        }
        let i = paddr as usize;
        Some(u32::from_be_bytes(
            self.data[i..i + 4].try_into().unwrap(),
        ))
    }

    fn write_u32(&mut self, paddr: u32, value: u32) {
        if !self.in_bounds(paddr, 4) {
            return;
        }
        let i = paddr as usize;
        self.data[i..i + 4].copy_from_slice(&value.to_be_bytes());
    }

    fn read_u8(&mut self, paddr: u32) -> Option<u8> {
        if !self.in_bounds(paddr, 1) {
            return None;
        }
        Some(self.data[paddr as usize])
    }

    fn write_u8(&mut self, paddr: u32, value: u8) {
        if !self.in_bounds(paddr, 1) {
            return;
        }
        self.data[paddr as usize] = value;
    }
}

/// Translate virtual CPU addresses to physical RAM offsets for the simple
/// flat map used in tests and early IPL bring-up.
pub fn virt_to_phys_rdram(vaddr: u64, rdram_size: usize) -> Option<u32> {
    let mask = (rdram_size as u64) - 1;
    let paddr = match vaddr {
        0x8000_0000..=0x803F_FFFF | 0xA000_0000..=0xA03F_FFFF => vaddr & mask,
        0x0000_0000..=0x003F_FFFF => vaddr & mask,
        _ => return None,
    };
    Some(paddr as u32)
}
