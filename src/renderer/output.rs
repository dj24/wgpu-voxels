pub(crate) const OUTPUT_TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
pub(crate) const COARSE_DEPTH_TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;
pub(crate) const COARSE_DEPTH_DIVISOR: u32 = 8;

pub(crate) struct OutputTarget {
    _output_texture: wgpu::Texture,
    output_view: wgpu::TextureView,
    _coarse_depth_texture: wgpu::Texture,
    coarse_depth_view: wgpu::TextureView,
    coarse_depth_size: (u32, u32),
}

impl OutputTarget {
    pub(crate) fn new(device: &wgpu::Device, surface_config: &wgpu::SurfaceConfiguration) -> Self {
        let output_texture = device.create_texture(&wgpu::TextureDescriptor {
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
        let output_view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let coarse_depth_size =
            coarse_depth_dimensions(surface_config.width, surface_config.height);
        let coarse_depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("coarse depth texture"),
            size: wgpu::Extent3d {
                width: coarse_depth_size.0,
                height: coarse_depth_size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: COARSE_DEPTH_TEXTURE_FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let coarse_depth_view =
            coarse_depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            _output_texture: output_texture,
            output_view,
            _coarse_depth_texture: coarse_depth_texture,
            coarse_depth_view,
            coarse_depth_size,
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
        &self.output_view
    }

    pub(crate) fn coarse_depth_view(&self) -> &wgpu::TextureView {
        &self.coarse_depth_view
    }

    pub(crate) fn coarse_depth_size(&self) -> (u32, u32) {
        self.coarse_depth_size
    }
}

fn coarse_depth_dimensions(width: u32, height: u32) -> (u32, u32) {
    (
        width.max(1).div_ceil(COARSE_DEPTH_DIVISOR),
        height.max(1).div_ceil(COARSE_DEPTH_DIVISOR),
    )
}

#[cfg(test)]
mod tests {
    use super::coarse_depth_dimensions;

    #[test]
    fn coarse_depth_dimensions_round_up_by_eighths() {
        assert_eq!(coarse_depth_dimensions(1920, 1080), (240, 135));
        assert_eq!(coarse_depth_dimensions(1919, 1079), (240, 135));
        assert_eq!(coarse_depth_dimensions(1, 1), (1, 1));
        assert_eq!(coarse_depth_dimensions(0, 0), (1, 1));
    }
}
