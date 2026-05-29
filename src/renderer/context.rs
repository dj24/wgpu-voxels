use std::sync::Arc;

use wgpu::{CurrentSurfaceTexture, SurfaceTexture};
use winit::{
    dpi::PhysicalSize,
    window::{Window, WindowId},
};

pub(crate) struct GpuContext {
    pub(crate) window: Arc<Window>,
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
}

impl GpuContext {
    pub(crate) async fn new(window: Arc<Window>) -> Result<Self, String> {
        let mut instance_desc = wgpu::InstanceDescriptor::new_without_display_handle();
        instance_desc.backends = wgpu::Backends::VULKAN;
        let instance = wgpu::Instance::new(instance_desc);

        let surface = instance
            .create_surface(window.clone())
            .map_err(|error| format!("create surface: {error}"))?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .map_err(|error| format!("request adapter: {error}"))?;

        let required_features = wgpu::Features::EXPERIMENTAL_RAY_QUERY;
        if !adapter.features().contains(required_features) {
            return Err(String::from(
                "the selected Vulkan adapter does not support wgpu experimental ray queries",
            ));
        }

        let required_limits =
            wgpu::Limits::default().using_acceleration_structure_values(adapter.limits());

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("device"),
                required_features,
                required_limits,
                experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|error| format!("request device: {error}"))?;

        let size = window.inner_size();
        let mut surface_config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .ok_or_else(|| String::from("surface is not supported by the selected adapter"))?;

        let capabilities = surface.get_capabilities(&adapter);
        if let Some(srgb_format) = capabilities
            .formats
            .iter()
            .copied()
            .find(|format| format.is_srgb())
        {
            surface_config.format = srgb_format;
        }
        if capabilities
            .present_modes
            .contains(&wgpu::PresentMode::Immediate)
        {
            surface_config.present_mode = wgpu::PresentMode::Immediate;
        } else if capabilities
            .present_modes
            .contains(&wgpu::PresentMode::Mailbox)
        {
            surface_config.present_mode = wgpu::PresentMode::Mailbox;
        }

        surface.configure(&device, &surface_config);

        Ok(Self {
            window,
            device,
            queue,
            surface,
            surface_config,
        })
    }

    pub(crate) fn window_id(&self) -> WindowId {
        self.window.id()
    }

    pub(crate) fn request_redraw(&self) {
        self.window.request_redraw();
    }

    pub(crate) fn current_size(&self) -> PhysicalSize<u32> {
        PhysicalSize::new(self.surface_config.width, self.surface_config.height)
    }

    pub(crate) fn surface_format(&self) -> wgpu::TextureFormat {
        self.surface_config.format
    }

    pub(crate) fn surface_config(&self) -> &wgpu::SurfaceConfiguration {
        &self.surface_config
    }

    pub(crate) fn resize(&mut self, new_size: PhysicalSize<u32>) {
        self.surface_config.width = new_size.width;
        self.surface_config.height = new_size.height;

        if new_size.width == 0 || new_size.height == 0 {
            return;
        }

        self.surface.configure(&self.device, &self.surface_config);
    }

    pub(crate) fn acquire_frame(&mut self) -> Result<Option<SurfaceTexture>, String> {
        if self.surface_config.width == 0 || self.surface_config.height == 0 {
            return Ok(None);
        }

        let frame = match self.surface.get_current_texture() {
            CurrentSurfaceTexture::Success(frame) => frame,
            CurrentSurfaceTexture::Suboptimal(frame) => {
                self.surface.configure(&self.device, &self.surface_config);
                frame
            }
            CurrentSurfaceTexture::Timeout | CurrentSurfaceTexture::Occluded => return Ok(None),
            CurrentSurfaceTexture::Outdated | CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.surface_config);
                return Ok(None);
            }
            CurrentSurfaceTexture::Validation => {
                return Err(String::from("surface returned a validation error"));
            }
        };

        Ok(Some(frame))
    }
}
