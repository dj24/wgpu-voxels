use std::{path::Path, sync::mpsc};

pub(crate) const OUTPUT_TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
pub(crate) const COARSE_DEPTH_TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;
pub(crate) const WORLD_POSITION_TEXTURE_FORMAT: wgpu::TextureFormat =
    wgpu::TextureFormat::Rgba32Float;
pub(crate) const SHADING_INPUT_TEXTURE_FORMAT: wgpu::TextureFormat =
    wgpu::TextureFormat::Rgba32Float;
pub(crate) const SURFACE_COLOR_TEXTURE_FORMAT: wgpu::TextureFormat =
    wgpu::TextureFormat::Rgba8Unorm;
pub(crate) const MOTION_VECTOR_TEXTURE_FORMAT: wgpu::TextureFormat =
    wgpu::TextureFormat::Rgba16Float;
pub(crate) const COARSE_DEPTH_DIVISOR: u32 = 2;

pub(crate) struct OutputTarget {
    output_texture: wgpu::Texture,
    output_view: wgpu::TextureView,
    history_textures: [wgpu::Texture; 2],
    history_views: [wgpu::TextureView; 2],
    _world_position_texture: wgpu::Texture,
    world_position_view: wgpu::TextureView,
    history_world_position_textures: [wgpu::Texture; 2],
    history_world_position_views: [wgpu::TextureView; 2],
    _shading_input_texture: wgpu::Texture,
    shading_input_view: wgpu::TextureView,
    _surface_color_texture: wgpu::Texture,
    surface_color_view: wgpu::TextureView,
    _motion_vector_texture: wgpu::Texture,
    motion_vector_view: wgpu::TextureView,
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
        let history_textures = core::array::from_fn(|index| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(if index == 0 {
                    "temporal history texture a"
                } else {
                    "temporal history texture b"
                }),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: OUTPUT_TEXTURE_FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            })
        });
        let history_views = history_textures
            .each_ref()
            .map(|texture| texture.create_view(&Default::default()));
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
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let world_position_view =
            world_position_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let history_world_position_textures = core::array::from_fn(|index| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(if index == 0 {
                    "temporal history world position texture a"
                } else {
                    "temporal history world position texture b"
                }),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: WORLD_POSITION_TEXTURE_FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            })
        });
        let history_world_position_views = history_world_position_textures
            .each_ref()
            .map(|texture| texture.create_view(&Default::default()));
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
        let surface_color_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("surface color texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: SURFACE_COLOR_TEXTURE_FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let surface_color_view =
            surface_color_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let motion_vector_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("motion vector texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: MOTION_VECTOR_TEXTURE_FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let motion_vector_view =
            motion_vector_texture.create_view(&wgpu::TextureViewDescriptor::default());
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
            history_textures,
            history_views,
            _world_position_texture: world_position_texture,
            world_position_view,
            history_world_position_textures,
            history_world_position_views,
            _shading_input_texture: shading_input_texture,
            shading_input_view,
            _surface_color_texture: surface_color_texture,
            surface_color_view,
            _motion_vector_texture: motion_vector_texture,
            motion_vector_view,
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

    pub(crate) fn history_view(&self, index: usize) -> &wgpu::TextureView {
        &self.history_views[index]
    }

    pub(crate) fn copy_output_to_history(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        history_index: usize,
    ) {
        encoder.copy_texture_to_texture(
            self.output_texture.as_image_copy(),
            self.history_textures[history_index].as_image_copy(),
            wgpu::Extent3d {
                width: self.size.0,
                height: self.size.1,
                depth_or_array_layers: 1,
            },
        );
    }

    pub(crate) fn world_position_view(&self) -> &wgpu::TextureView {
        &self.world_position_view
    }

    pub(crate) fn history_world_position_view(&self, index: usize) -> &wgpu::TextureView {
        &self.history_world_position_views[index]
    }

    pub(crate) fn copy_world_position_to_history(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        history_index: usize,
    ) {
        encoder.copy_texture_to_texture(
            self._world_position_texture.as_image_copy(),
            self.history_world_position_textures[history_index].as_image_copy(),
            wgpu::Extent3d {
                width: self.size.0,
                height: self.size.1,
                depth_or_array_layers: 1,
            },
        );
    }

    pub(crate) fn shading_input_view(&self) -> &wgpu::TextureView {
        &self.shading_input_view
    }

    pub(crate) fn motion_vector_view(&self) -> &wgpu::TextureView {
        &self.motion_vector_view
    }

    pub(crate) fn surface_color_view(&self) -> &wgpu::TextureView {
        &self.surface_color_view
    }

    pub(crate) fn coarse_depth_size(&self) -> (u32, u32) {
        self.coarse_depth_size
    }

    pub(crate) fn save_png(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        path: &Path,
        history_index: usize,
    ) -> Result<(), String> {
        self.save_texture_png(device, queue, &self.history_textures[history_index], path)
    }

    pub(crate) fn save_output_png(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        path: &Path,
    ) -> Result<(), String> {
        self.save_texture_png(device, queue, &self.output_texture, path)
    }

    fn save_texture_png(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
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
            texture.as_image_copy(),
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
    fn coarse_depth_dimensions_round_up_by_divisor() {
        assert_eq!(coarse_depth_dimensions(1920, 1080), (960, 540));
        assert_eq!(coarse_depth_dimensions(1919, 1079), (960, 540));
        assert_eq!(coarse_depth_dimensions(1, 1), (1, 1));
        assert_eq!(coarse_depth_dimensions(0, 0), (1, 1));
    }
}
