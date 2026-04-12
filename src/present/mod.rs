//! **Phase 0–1:** Host display output. Default path uses **wgpu** (Vulkan on Windows) to present
//! RGBA8 frames. See [`GraphicsPhase`] for roadmap stages (RSP → software RDP → GPU path → games).

mod wgpu_present;

pub use wgpu_present::{run_wgpu_loop, PresentError, WgpuPresenter};

/// Roadmap milestones from the graphics plan (core work tracks these over time).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum GraphicsPhase {
    /// Swapchain + fullscreen textured draw (wgpu).
    VulkanPresent = 0,
    /// Stable frame pump: VI-timed blit through one presenter API.
    FramePump = 1,
    /// RSP task / interpreter produces RDP lists.
    RspTask = 2,
    /// Software RDP: triangles, textures, combiner coverage.
    SoftwareRdp = 3,
    /// RDP → GPU draws / textures (not only CPU upload of final FB).
    VulkanRdp = 4,
    /// First real title booting to recognizable picture.
    TargetGame = 5,
}

/// Reported progress toward on-screen output (extend as features land).
pub fn graphics_phase_reached() -> GraphicsPhase {
    GraphicsPhase::RspTask
}
