use std::{f32::consts::FRAC_PI_4, mem, sync::Arc};

use bytemuck::{Pod, Zeroable};
use wgpu::{CurrentSurfaceTexture, util::DeviceExt};
use winit::{
    dpi::PhysicalSize,
    window::{Window, WindowId},
};

const WORKGROUP_SIZE: u32 = 8;
const OUTPUT_TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

const SCENE_VERTEX_POSITIONS: [[f32; 3]; 8] = [
    [-0.75, -0.75, -0.75],
    [0.75, -0.75, -0.75],
    [0.75, 0.75, -0.75],
    [-0.75, 0.75, -0.75],
    [-0.75, -0.75, 0.75],
    [0.75, -0.75, 0.75],
    [0.75, 0.75, 0.75],
    [-0.75, 0.75, 0.75],
];

const SCENE_INDICES: [u16; 36] = [
    0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 0, 4, 5, 0, 5, 1, 3, 2, 6, 3, 6, 7, 1, 5, 6, 1, 6, 2, 0, 3,
    7, 0, 7, 4,
];

const INSTANCE_POSITIONS: [[f32; 3]; 4] = [
    [-1.8, 0.0, 0.0],
    [-0.2, 0.3, -0.8],
    [1.4, -0.1, 0.4],
    [0.7, 0.8, -1.6],
];

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CameraUniform {
    position: [f32; 4],
    forward: [f32; 4],
    right: [f32; 4],
    up: [f32; 4],
    viewport: [f32; 4],
}

struct Camera {
    position: [f32; 3],
    target: [f32; 3],
    up: [f32; 3],
    vertical_fov_radians: f32,
}

impl Camera {
    fn new() -> Self {
        Self {
            position: [0.0, 1.2, 5.5],
            target: [0.0, 0.3, 0.0],
            up: [0.0, 1.0, 0.0],
            vertical_fov_radians: FRAC_PI_4,
        }
    }

    fn to_uniform(&self, size: PhysicalSize<u32>) -> CameraUniform {
        let aspect = (size.width.max(1) as f32) / (size.height.max(1) as f32);
        let forward = normalize3(sub3(self.target, self.position));
        let right = normalize3(cross3(forward, self.up));
        let up = normalize3(cross3(right, forward));
        let tan_half_fov = (self.vertical_fov_radians * 0.5).tan();

        CameraUniform {
            position: [self.position[0], self.position[1], self.position[2], 0.0],
            forward: [forward[0], forward[1], forward[2], 0.0],
            right: [right[0], right[1], right[2], 0.0],
            up: [up[0], up[1], up[2], 0.0],
            viewport: [tan_half_fov * aspect, tan_half_fov, aspect, 0.0],
        }
    }
}

pub(crate) struct Renderer {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface_config: wgpu::SurfaceConfiguration,
    camera: Camera,
    camera_buffer: wgpu::Buffer,
    _scene_vertex_buffer: wgpu::Buffer,
    _scene_index_buffer: wgpu::Buffer,
    _blas: wgpu::Blas,
    tlas: wgpu::Tlas,
    compute_bind_group_layout: wgpu::BindGroupLayout,
    blit_bind_group_layout: wgpu::BindGroupLayout,
    compute_pipeline: wgpu::ComputePipeline,
    blit_pipeline: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
    output_texture: wgpu::Texture,
    output_view: wgpu::TextureView,
    compute_bind_group: wgpu::BindGroup,
    blit_bind_group: wgpu::BindGroup,
}

impl Renderer {
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
                // Ray queries are still behind wgpu's experimental feature gate.
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

        surface.configure(&device, &surface_config);

        let camera = Camera::new();
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera buffer"),
            contents: bytemuck::bytes_of(&camera.to_uniform(size)),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let (scene_vertex_buffer, scene_index_buffer, blas, tlas) =
            Self::create_acceleration_scene(&device, &queue);

        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("uv compute shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("compute.wgsl").into()),
        });

        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("uv blit shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("blit.wgsl").into()),
        });

        let compute_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("compute bind group layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: OUTPUT_TEXTURE_FORMAT,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::AccelerationStructure {
                            vertex_return: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let blit_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("blit bind group layout"),
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
                ],
            });

        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("compute pipeline layout"),
                bind_group_layouts: &[Some(&compute_bind_group_layout)],
                immediate_size: 0,
            });

        let blit_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("blit pipeline layout"),
            bind_group_layouts: &[Some(&blit_bind_group_layout)],
            immediate_size: 0,
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("uv compute pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: Some("compute_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blit pipeline"),
            layout: Some(&blit_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &blit_shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &blit_shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("output sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let (output_texture, output_view, compute_bind_group, blit_bind_group) =
            Self::create_output_resources(
                &device,
                &surface_config,
                &compute_bind_group_layout,
                &blit_bind_group_layout,
                &sampler,
                &tlas,
                &camera_buffer,
            );

        Ok(Self {
            window,
            surface,
            device,
            queue,
            surface_config,
            camera,
            camera_buffer,
            _scene_vertex_buffer: scene_vertex_buffer,
            _scene_index_buffer: scene_index_buffer,
            _blas: blas,
            tlas,
            compute_bind_group_layout,
            blit_bind_group_layout,
            compute_pipeline,
            blit_pipeline,
            sampler,
            output_texture,
            output_view,
            compute_bind_group,
            blit_bind_group,
        })
    }

    pub(crate) fn window_id(&self) -> WindowId {
        self.window.id()
    }

    pub(crate) fn request_redraw(&self) {
        self.window.request_redraw();
    }

    pub(crate) fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            self.surface_config.width = new_size.width;
            self.surface_config.height = new_size.height;
            return;
        }

        self.surface_config.width = new_size.width;
        self.surface_config.height = new_size.height;
        self.surface.configure(&self.device, &self.surface_config);
        self.update_camera_buffer();
        self.recreate_output_resources();
    }

    pub(crate) fn render(&mut self) -> Result<(), String> {
        if self.surface_config.width == 0 || self.surface_config.height == 0 {
            return Ok(());
        }

        let frame = match self.surface.get_current_texture() {
            CurrentSurfaceTexture::Success(frame) => frame,
            CurrentSurfaceTexture::Suboptimal(frame) => {
                self.surface.configure(&self.device, &self.surface_config);
                frame
            }
            CurrentSurfaceTexture::Timeout | CurrentSurfaceTexture::Occluded => return Ok(()),
            CurrentSurfaceTexture::Outdated | CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.surface_config);
                return Ok(());
            }
            CurrentSurfaceTexture::Validation => {
                return Err(String::from("surface returned a validation error"));
            }
        };

        let surface_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ray query compute pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &self.compute_bind_group, &[]);
            compute_pass.dispatch_workgroups(
                self.surface_config.width.div_ceil(WORKGROUP_SIZE),
                self.surface_config.height.div_ceil(WORKGROUP_SIZE),
                1,
            );
        }

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("present pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &surface_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            render_pass.set_pipeline(&self.blit_pipeline);
            render_pass.set_bind_group(0, &self.blit_bind_group, &[]);
            render_pass.draw(0..3, 0..1);
        }

        self.queue.submit(Some(encoder.finish()));
        self.window.pre_present_notify();
        frame.present();
        Ok(())
    }

    fn recreate_output_resources(&mut self) {
        let (output_texture, output_view, compute_bind_group, blit_bind_group) =
            Self::create_output_resources(
                &self.device,
                &self.surface_config,
                &self.compute_bind_group_layout,
                &self.blit_bind_group_layout,
                &self.sampler,
                &self.tlas,
                &self.camera_buffer,
            );

        self.output_texture = output_texture;
        self.output_view = output_view;
        self.compute_bind_group = compute_bind_group;
        self.blit_bind_group = blit_bind_group;
    }

    fn update_camera_buffer(&self) {
        let uniform = self.camera.to_uniform(PhysicalSize::new(
            self.surface_config.width,
            self.surface_config.height,
        ));
        self.queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&uniform));
    }

    fn create_acceleration_scene(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> (wgpu::Buffer, wgpu::Buffer, wgpu::Blas, wgpu::Tlas) {
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("scene vertex buffer"),
            contents: bytemuck::cast_slice(&SCENE_VERTEX_POSITIONS),
            usage: wgpu::BufferUsages::BLAS_INPUT,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("scene index buffer"),
            contents: bytemuck::cast_slice(&SCENE_INDICES),
            usage: wgpu::BufferUsages::BLAS_INPUT,
        });

        let geometry_size = wgpu::BlasTriangleGeometrySizeDescriptor {
            vertex_format: wgpu::VertexFormat::Float32x3,
            vertex_count: SCENE_VERTEX_POSITIONS.len() as u32,
            index_format: Some(wgpu::IndexFormat::Uint16),
            index_count: Some(SCENE_INDICES.len() as u32),
            flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
        };

        let blas = device.create_blas(
            &wgpu::CreateBlasDescriptor {
                label: Some("scene cube blas"),
                flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                update_mode: wgpu::AccelerationStructureUpdateMode::Build,
            },
            wgpu::BlasGeometrySizeDescriptors::Triangles {
                descriptors: vec![geometry_size.clone()],
            },
        );

        let mut tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
            label: Some("scene tlas"),
            max_instances: INSTANCE_POSITIONS.len() as u32,
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        });

        for (index, position) in INSTANCE_POSITIONS.iter().enumerate() {
            tlas[index] = Some(wgpu::TlasInstance::new(
                &blas,
                translation_transform(*position),
                index as u32,
                0xff,
            ));
        }

        let geometry = wgpu::BlasTriangleGeometry {
            size: &geometry_size,
            vertex_buffer: &vertex_buffer,
            first_vertex: 0,
            vertex_stride: mem::size_of::<[f32; 3]>() as u64,
            index_buffer: Some(&index_buffer),
            first_index: Some(0),
            transform_buffer: None,
            transform_buffer_offset: None,
        };

        let blas_build_entry = wgpu::BlasBuildEntry {
            blas: &blas,
            geometry: wgpu::BlasGeometries::TriangleGeometries(vec![geometry]),
        };

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("acceleration structure build encoder"),
        });
        encoder.build_acceleration_structures([&blas_build_entry], [&tlas]);
        queue.submit(Some(encoder.finish()));

        (vertex_buffer, index_buffer, blas, tlas)
    }

    fn create_output_resources(
        device: &wgpu::Device,
        surface_config: &wgpu::SurfaceConfiguration,
        compute_bind_group_layout: &wgpu::BindGroupLayout,
        blit_bind_group_layout: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
        tlas: &wgpu::Tlas,
        camera_buffer: &wgpu::Buffer,
    ) -> (
        wgpu::Texture,
        wgpu::TextureView,
        wgpu::BindGroup,
        wgpu::BindGroup,
    ) {
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

        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compute bind group"),
            layout: compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: tlas.as_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: camera_buffer.as_entire_binding(),
                },
            ],
        });

        let blit_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blit bind group"),
            layout: blit_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });

        (
            output_texture,
            output_view,
            compute_bind_group,
            blit_bind_group,
        )
    }
}

fn translation_transform(position: [f32; 3]) -> [f32; 12] {
    [
        1.0,
        0.0,
        0.0,
        position[0],
        0.0,
        1.0,
        0.0,
        position[1],
        0.0,
        0.0,
        1.0,
        position[2],
    ]
}

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-6);
    [v[0] / len, v[1] / len, v[2] / len]
}
