//! wgpu surface + fullscreen textured triangle. Uses **Vulkan** (`Backends::VULKAN`).

use std::sync::Arc;

use winit::dpi::LogicalSize;
use winit::event::{ElementState, Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowBuilder};

const SHADER: &str = r#"
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var o: VertexOutput;
    o.clip_position = vec4<f32>(positions[vi], 0.0, 1.0);
    o.uv = positions[vi] * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
    return o;
}

@group(0) @binding(0)
var frame_tex: texture_2d<f32>;
@group(0) @binding(1)
var frame_samp: sampler;
@group(0) @binding(2)
var rdp_aux_tex: texture_2d<f32>;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let fb = textureSample(frame_tex, frame_samp, in.uv);
    let aux_uv = (in.uv - vec2<f32>(0.62, 0.08)) / vec2<f32>(0.35, 0.35);
    let aux = textureSample(rdp_aux_tex, frame_samp, aux_uv);
    let pip = aux_uv.x >= 0.0 && aux_uv.x <= 1.0 && aux_uv.y >= 0.0 && aux_uv.y <= 1.0;
    if (pip) {
        return mix(fb, aux, 0.5);
    }
    return fb;
}
"#;

/// Frame payload for [`run_wgpu_loop`]: main RGBA8 framebuffer plus optional RDP/TMEM overlay for [`WgpuPresenter::upload_and_present`].
pub struct WgpuFrame {
    pub rgba: Vec<u8>,
    /// RGBA8 auxiliary image (e.g. software RDP / TMEM view); width/height must match the presenter `aux_w` / `aux_h` (128×128).
    pub aux_rgba: Option<(Vec<u8>, u32, u32)>,
    pub keep_open: bool,
}

impl WgpuFrame {
    pub fn new(rgba: Vec<u8>, keep_open: bool) -> Self {
        Self {
            rgba,
            aux_rgba: None,
            keep_open,
        }
    }
}

#[derive(Debug)]
pub enum PresentError {
    WgpuRequest(String),
    Surface(String),
}

pub struct WgpuPresenter {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    texture: wgpu::Texture,
    rdp_aux_texture: wgpu::Texture,
    bind_group: wgpu::BindGroup,
    // Kept so the sampler outlives the bind group (wgpu lifetime).
    #[allow(dead_code)]
    sampler: wgpu::Sampler,
    tex_w: u32,
    tex_h: u32,
    pub aux_w: u32,
    pub aux_h: u32,
}

impl WgpuPresenter {
    pub fn new(
        event_loop: &EventLoop<()>,
        title: &str,
        tex_w: u32,
        tex_h: u32,
    ) -> Result<Self, PresentError> {
        let window = Arc::new(
            WindowBuilder::new()
                .with_title(title)
                .with_inner_size(LogicalSize::new(tex_w as f64, tex_h as f64))
                .build(event_loop)
                .map_err(|e| PresentError::WgpuRequest(e.to_string()))?,
        );

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });

        let surface = instance
            .create_surface(window.clone())
            .map_err(|e| PresentError::Surface(e.to_string()))?;

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .ok_or_else(|| PresentError::WgpuRequest("no suitable adapter".into()))?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("ultrareality"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        ))
        .map_err(|e| PresentError::WgpuRequest(e.to_string()))?;

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .find(|f| **f == wgpu::TextureFormat::Bgra8Unorm || **f == wgpu::TextureFormat::Rgba8Unorm)
            .copied()
            .unwrap_or(caps.formats[0]);

        let size = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fullscreen"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        const AUX_W: u32 = 128;
        const AUX_H: u32 = 128;

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tex_bind"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("fullscreen"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("nearest"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("frame"),
            size: wgpu::Extent3d {
                width: tex_w,
                height: tex_h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let rdp_aux_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("rdp_aux"),
            size: wgpu::Extent3d {
                width: AUX_W,
                height: AUX_H,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let rdp_aux_view = rdp_aux_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frame_bg"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&rdp_aux_view),
                },
            ],
        });

        Ok(Self {
            window,
            surface,
            device,
            queue,
            config,
            pipeline,
            texture,
            rdp_aux_texture,
            bind_group,
            sampler,
            tex_w,
            tex_h,
            aux_w: AUX_W,
            aux_h: AUX_H,
        })
    }

    pub fn window_id(&self) -> winit::window::WindowId {
        self.window.id()
    }

    pub fn request_redraw(&self) {
        self.window.request_redraw();
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
    }

    /// Upload the main framebuffer; optional `aux` is composited (PIP) when dimensions match [`Self::aux_w`] / [`Self::aux_h`].
    pub fn upload_and_present(
        &mut self,
        rgba: &[u8],
        width: u32,
        height: u32,
        aux: Option<(&[u8], u32, u32)>,
    ) -> Result<(), PresentError> {
        if width != self.tex_w || height != self.tex_h {
            return Err(PresentError::Surface(format!(
                "frame size {width}x{height} != presenter {}x{}",
                self.tex_w, self.tex_h
            )));
        }
        let expected = (width * height * 4) as usize;
        if rgba.len() < expected {
            return Err(PresentError::Surface(format!(
                "rgba buffer too short: {} < {}",
                rgba.len(),
                expected
            )));
        }

        if let Some((adata, aw, ah)) = aux {
            if aw != self.aux_w || ah != self.aux_h {
                return Err(PresentError::Surface(format!(
                    "aux size {aw}x{ah} != presenter aux {}x{}",
                    self.aux_w, self.aux_h
                )));
            }
            let need = (aw * ah * 4) as usize;
            if adata.len() < need {
                return Err(PresentError::Surface(format!(
                    "aux rgba too short: {} < {}",
                    adata.len(),
                    need
                )));
            }
            self.queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &self.rdp_aux_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &adata[..need],
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * aw),
                    rows_per_image: Some(ah),
                },
                wgpu::Extent3d {
                    width: aw,
                    height: ah,
                    depth_or_array_layers: 1,
                },
            );
        }

        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &rgba[..expected],
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        let output = self
            .surface
            .get_current_texture()
            .map_err(|e| PresentError::Surface(e.to_string()))?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("present"),
            });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

pub fn run_wgpu_loop(
    title: &str,
    tex_w: u32,
    tex_h: u32,
    mut on_frame: impl FnMut() -> WgpuFrame,
) -> Result<(), PresentError> {
    let event_loop = EventLoop::new().map_err(|e| PresentError::WgpuRequest(e.to_string()))?;
    let mut gpu = WgpuPresenter::new(&event_loop, title, tex_w, tex_h)?;
    let window_id = gpu.window_id();

    event_loop
        .run(move |event, elwt| {
            match event {
                Event::WindowEvent {
                    window_id: id,
                    event,
                } if id == window_id => match event {
                    WindowEvent::CloseRequested | WindowEvent::Destroyed => elwt.exit(),
                    WindowEvent::Resized(size) => gpu.resize(size.width, size.height),
                    WindowEvent::RedrawRequested => {
                        let frame = on_frame();
                        if !frame.keep_open {
                            elwt.exit();
                            return;
                        }
                        let aux = frame
                            .aux_rgba
                            .as_ref()
                            .map(|(buf, w, h)| (buf.as_slice(), *w, *h));
                        if let Err(e) = gpu.upload_and_present(&frame.rgba, tex_w, tex_h, aux) {
                            eprintln!("wgpu present: {e:?}");
                            elwt.exit();
                        }
                    }
                    WindowEvent::KeyboardInput { event, .. } => {
                        if event.state == ElementState::Pressed
                            && event.physical_key == PhysicalKey::Code(KeyCode::Escape)
                        {
                            elwt.exit();
                        }
                    }
                    _ => {}
                },
                Event::AboutToWait => {
                    gpu.request_redraw();
                }
                _ => {}
            }
        })
        .map_err(|e| PresentError::WgpuRequest(e.to_string()))?;

    Ok(())
}
