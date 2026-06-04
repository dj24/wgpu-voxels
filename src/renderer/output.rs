use std::{path::Path, sync::mpsc};

pub(crate) const OUTPUT_TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
pub(crate) const COARSE_DEPTH_TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;
pub(crate) const WORLD_POSITION_TEXTURE_FORMAT: wgpu::TextureFormat =
    wgpu::TextureFormat::Rgba32Float;
pub(crate) const SHADING_INPUT_TEXTURE_FORMAT: wgpu::TextureFormat =
    wgpu::TextureFormat::Rgba32Float;
pub(crate) const COARSE_DEPTH_DIVISOR: u32 = 8;

pub(crate) struct OutputTarget {
    output_texture: wgpu::Texture,
    output_view: wgpu::TextureView,
    _world_position_texture: wgpu::Texture,
    world_position_view: wgpu::TextureView,
    _shading_input_texture: wgpu::Texture,
    shading_input_view: wgpu::TextureView,
    _coarse_depth_texture: wgpu::Texture,
    coarse_depth_view: wgpu::TextureView,
    coarse_depth_size: (u32, u32),
    size: (u32, u32),
}

impl OutputTarget {
    pub(crate) fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let width = width.max(1);
        let height = height.max(1);
        let output_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("compute output texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: OUTPUT_TEXTURE_FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let output_view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let world_position_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("world position texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: WORLD_POSITION_TEXTURE_FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let world_position_view =
            world_position_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let shading_input_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shading input texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: SHADING_INPUT_TEXTURE_FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shading_input_view =
            shading_input_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let coarse_depth_size = coarse_depth_dimensions(width, height);
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
            output_texture,
            output_view,
            _world_position_texture: world_position_texture,
            world_position_view,
            _shading_input_texture: shading_input_texture,
            shading_input_view,
            _coarse_depth_texture: coarse_depth_texture,
            coarse_depth_view,
            coarse_depth_size,
            size: (width, height),
        }
    }

    pub(crate) fn recreate(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        *self = Self::new(device, width, height);
    }

    pub(crate) fn view(&self) -> &wgpu::TextureView {
        &self.output_view
    }

    pub(crate) fn coarse_depth_view(&self) -> &wgpu::TextureView {
        &self.coarse_depth_view
    }

    pub(crate) fn world_position_view(&self) -> &wgpu::TextureView {
        &self.world_position_view
    }

    pub(crate) fn shading_input_view(&self) -> &wgpu::TextureView {
        &self.shading_input_view
    }

    pub(crate) fn coarse_depth_size(&self) -> (u32, u32) {
        self.coarse_depth_size
    }

    pub(crate) fn save_png(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        path: &Path,
    ) -> Result<(), String> {
        let bytes_per_pixel = 4;
        let unpadded_bytes_per_row = self.size.0 * bytes_per_pixel;
        let padded_bytes_per_row =
            wgpu::util::align_to(unpadded_bytes_per_row, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("headless readback buffer"),
            size: padded_bytes_per_row as u64 * self.size.1 as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("headless readback encoder"),
        });
        encoder.copy_texture_to_buffer(
            self.output_texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(self.size.1),
                },
            },
            wgpu::Extent3d {
                width: self.size.0,
                height: self.size.1,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(encoder.finish()));

        let slice = readback_buffer.slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result.map_err(|error| error.to_string()));
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|error| format!("poll device: {error}"))?;
        receiver
            .recv()
            .map_err(|error| format!("receive readback result: {error}"))??;

        let mapped = slice.get_mapped_range();
        let mut pixels = Vec::with_capacity((self.size.0 * self.size.1 * bytes_per_pixel) as usize);
        for padded_row in mapped.chunks(padded_bytes_per_row as usize) {
            pixels.extend_from_slice(&padded_row[..unpadded_bytes_per_row as usize]);
        }
        drop(mapped);
        readback_buffer.unmap();

        let file = std::fs::File::create(path).map_err(|error| format!("create png: {error}"))?;
        let mut encoder = png::Encoder::new(file, self.size.0, self.size.1);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder
            .write_header()
            .map_err(|error| format!("encode png header: {error}"))?;
        writer
            .write_image_data(&pixels)
            .map_err(|error| format!("encode png data: {error}"))?;

        Ok(())
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
