use std::time::{Duration, Instant};

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use winit::dpi::PhysicalSize;

const LABEL: &str = "FPS";
const FONT_WIDTH: usize = 3;
const FONT_HEIGHT: usize = 5;
const CELL_SIZE: f32 = 5.0;
const CELL_GAP: f32 = 1.0;
const CHARACTER_SPACING: f32 = 3.0;
const PANEL_PADDING_X: f32 = 8.0;
const PANEL_PADDING_Y: f32 = 6.0;
const VIEWPORT_MARGIN: f32 = 12.0;
const PANEL_COLOR: [f32; 4] = [0.0, 0.0, 0.0, 0.55];
const TEXT_COLOR: [f32; 4] = [0.95, 0.97, 1.0, 1.0];
const FPS_UPDATE_INTERVAL: Duration = Duration::from_millis(250);

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct OverlayVertex {
    position: [f32; 2],
    color: [f32; 4],
}

struct FpsCounter {
    last_frame_instant: Option<Instant>,
    accumulated_time: Duration,
    accumulated_frames: u32,
    displayed_fps: u32,
}

impl FpsCounter {
    fn new() -> Self {
        Self {
            last_frame_instant: None,
            accumulated_time: Duration::ZERO,
            accumulated_frames: 0,
            displayed_fps: 0,
        }
    }

    fn tick(&mut self) -> u32 {
        let now = Instant::now();
        let Some(previous_frame) = self.last_frame_instant.replace(now) else {
            return self.displayed_fps;
        };

        self.accumulated_time += now.saturating_duration_since(previous_frame);
        self.accumulated_frames += 1;

        if self.accumulated_time >= FPS_UPDATE_INTERVAL {
            self.displayed_fps = (self.accumulated_frames as f32
                / self.accumulated_time.as_secs_f32())
            .round() as u32;
            self.accumulated_time = Duration::ZERO;
            self.accumulated_frames = 0;
        }

        self.displayed_fps
    }
}

pub(crate) struct FpsOverlay {
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    vertex_capacity: usize,
    vertex_count: u32,
    vertices: Vec<OverlayVertex>,
    fps_counter: FpsCounter,
}

impl FpsOverlay {
    pub(crate) fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fps overlay shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("fps_overlay.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fps overlay pipeline layout"),
            bind_group_layouts: &[],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("fps overlay pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<OverlayVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: std::mem::size_of::<[f32; 2]>() as u64,
                            shader_location: 1,
                        },
                    ],
                }],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let vertex_capacity = 256;
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fps overlay vertex buffer"),
            contents: &vec![0_u8; vertex_capacity * std::mem::size_of::<OverlayVertex>()],
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            pipeline,
            vertex_buffer,
            vertex_capacity,
            vertex_count: 0,
            vertices: Vec::with_capacity(vertex_capacity),
            fps_counter: FpsCounter::new(),
        }
    }

    pub(crate) fn update(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_size: PhysicalSize<u32>,
    ) {
        if surface_size.width == 0 || surface_size.height == 0 {
            self.vertex_count = 0;
            return;
        }

        let fps = self.fps_counter.tick();
        let text = format!("{LABEL} {fps}");
        let text_size = measure_text(&text);
        let panel_width = text_size[0] + PANEL_PADDING_X * 2.0;
        let panel_height = text_size[1] + PANEL_PADDING_Y * 2.0;
        let origin_x = surface_size.width as f32 - VIEWPORT_MARGIN - panel_width;
        let origin_y = VIEWPORT_MARGIN;
        let text_x = origin_x + PANEL_PADDING_X;
        let text_y = origin_y + PANEL_PADDING_Y;

        self.vertices.clear();
        self.push_rect(
            surface_size,
            origin_x,
            origin_y,
            origin_x + panel_width,
            origin_y + panel_height,
            PANEL_COLOR,
        );
        self.push_text(surface_size, text_x, text_y, &text, TEXT_COLOR);

        self.vertex_count = self.vertices.len() as u32;
        self.ensure_capacity(device);
        queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&self.vertices));
    }

    pub(crate) fn draw<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        if self.vertex_count == 0 {
            return;
        }

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.draw(0..self.vertex_count, 0..1);
    }

    fn ensure_capacity(&mut self, device: &wgpu::Device) {
        if self.vertices.len() <= self.vertex_capacity {
            return;
        }

        self.vertex_capacity = self.vertices.len().next_power_of_two();
        self.vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fps overlay vertex buffer"),
            size: (self.vertex_capacity * std::mem::size_of::<OverlayVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
    }

    fn push_text(
        &mut self,
        surface_size: PhysicalSize<u32>,
        mut x: f32,
        y: f32,
        text: &str,
        color: [f32; 4],
    ) {
        for ch in text.chars() {
            if ch == ' ' {
                x += glyph_advance();
                continue;
            }

            if let Some(rows) = glyph_rows(ch) {
                for (row_index, row_bits) in rows.into_iter().enumerate() {
                    for column_index in 0..FONT_WIDTH {
                        if (row_bits & (1 << (FONT_WIDTH - column_index - 1))) == 0 {
                            continue;
                        }

                        let cell_x = x + column_index as f32 * (CELL_SIZE + CELL_GAP);
                        let cell_y = y + row_index as f32 * (CELL_SIZE + CELL_GAP);
                        self.push_rect(
                            surface_size,
                            cell_x,
                            cell_y,
                            cell_x + CELL_SIZE,
                            cell_y + CELL_SIZE,
                            color,
                        );
                    }
                }
            }

            x += glyph_advance();
        }
    }

    fn push_rect(
        &mut self,
        surface_size: PhysicalSize<u32>,
        x0: f32,
        y0: f32,
        x1: f32,
        y1: f32,
        color: [f32; 4],
    ) {
        let top_left = pixels_to_ndc(surface_size, x0, y0);
        let top_right = pixels_to_ndc(surface_size, x1, y0);
        let bottom_left = pixels_to_ndc(surface_size, x0, y1);
        let bottom_right = pixels_to_ndc(surface_size, x1, y1);

        self.vertices.extend_from_slice(&[
            OverlayVertex {
                position: top_left,
                color,
            },
            OverlayVertex {
                position: bottom_left,
                color,
            },
            OverlayVertex {
                position: top_right,
                color,
            },
            OverlayVertex {
                position: top_right,
                color,
            },
            OverlayVertex {
                position: bottom_left,
                color,
            },
            OverlayVertex {
                position: bottom_right,
                color,
            },
        ]);
    }
}

fn pixels_to_ndc(surface_size: PhysicalSize<u32>, x: f32, y: f32) -> [f32; 2] {
    let width = surface_size.width.max(1) as f32;
    let height = surface_size.height.max(1) as f32;
    [(x / width) * 2.0 - 1.0, 1.0 - (y / height) * 2.0]
}

fn measure_text(text: &str) -> [f32; 2] {
    let width = text.chars().count() as f32 * glyph_advance() - CHARACTER_SPACING;
    let height = FONT_HEIGHT as f32 * CELL_SIZE + (FONT_HEIGHT - 1) as f32 * CELL_GAP;
    [width.max(0.0), height]
}

fn glyph_advance() -> f32 {
    FONT_WIDTH as f32 * CELL_SIZE + (FONT_WIDTH - 1) as f32 * CELL_GAP + CHARACTER_SPACING
}

fn glyph_rows(ch: char) -> Option<[u8; FONT_HEIGHT]> {
    match ch {
        '0' => Some([0b111, 0b101, 0b101, 0b101, 0b111]),
        '1' => Some([0b010, 0b110, 0b010, 0b010, 0b111]),
        '2' => Some([0b111, 0b001, 0b111, 0b100, 0b111]),
        '3' => Some([0b111, 0b001, 0b111, 0b001, 0b111]),
        '4' => Some([0b101, 0b101, 0b111, 0b001, 0b001]),
        '5' => Some([0b111, 0b100, 0b111, 0b001, 0b111]),
        '6' => Some([0b111, 0b100, 0b111, 0b101, 0b111]),
        '7' => Some([0b111, 0b001, 0b001, 0b001, 0b001]),
        '8' => Some([0b111, 0b101, 0b111, 0b101, 0b111]),
        '9' => Some([0b111, 0b101, 0b111, 0b001, 0b111]),
        'F' => Some([0b111, 0b100, 0b111, 0b100, 0b100]),
        'P' => Some([0b110, 0b101, 0b110, 0b100, 0b100]),
        'S' => Some([0b111, 0b100, 0b111, 0b001, 0b111]),
        _ => None,
    }
}
