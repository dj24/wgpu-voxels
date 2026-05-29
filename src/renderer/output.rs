pub(crate) const OUTPUT_TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

pub(crate) struct OutputTarget {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
}

impl OutputTarget {
    pub(crate) fn new(device: &wgpu::Device, surface_config: &wgpu::SurfaceConfiguration) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("compute output texture"),
            size: wgpu::Extent3d {
                width: surface_config.width.max(1),
                height: surface_config.height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: OUTPUT_TEXTURE_FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            _texture: texture,
            view,
        }
    }

    pub(crate) fn recreate(
        &mut self,
        device: &wgpu::Device,
        surface_config: &wgpu::SurfaceConfiguration,
    ) {
        *self = Self::new(device, surface_config);
    }

    pub(crate) fn view(&self) -> &wgpu::TextureView {
        &self.view
    }
}
