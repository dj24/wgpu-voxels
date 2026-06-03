use crate::renderer::output::{
    COARSE_DEPTH_TEXTURE_FORMAT, OUTPUT_TEXTURE_FORMAT, WORLD_POSITION_TEXTURE_FORMAT,
};

const WORKGROUP_SIZE: u32 = 8;

#[allow(dead_code)]
#[derive(Clone, Copy)]
enum VisualizationMode {
    WorldPosition,
    TileGroups,
}

const DEFAULT_VISUALIZATION_MODE: VisualizationMode = VisualizationMode::TileGroups;

pub(crate) struct ComputeVoxelsPass {
    coarse_depth_bind_group_layout: wgpu::BindGroupLayout,
    trace_bind_group_layout: wgpu::BindGroupLayout,
    visualize_bind_group_layout: wgpu::BindGroupLayout,
    coarse_depth_pipeline: wgpu::ComputePipeline,
    trace_pipeline: wgpu::ComputePipeline,
    visualize_world_position_pipeline: wgpu::ComputePipeline,
    visualize_tiles_pipeline: wgpu::ComputePipeline,
    coarse_depth_bind_group: wgpu::BindGroup,
    trace_bind_group: wgpu::BindGroup,
    visualize_bind_group: wgpu::BindGroup,
}

impl ComputeVoxelsPass {
    pub(crate) fn new(
        device: &wgpu::Device,
        output_view: &wgpu::TextureView,
        world_position_view: &wgpu::TextureView,
        coarse_depth_view: &wgpu::TextureView,
        tlas: &wgpu::Tlas,
        camera_buffer: &wgpu::Buffer,
        debug_visualization_buffer: &wgpu::Buffer,
        voxel_mask_buffer: &wgpu::Buffer,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("voxel compute shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../compute.wgsl").into()),
        });

        let coarse_depth_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("coarse depth bind group layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::AccelerationStructure {
                            vertex_return: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: COARSE_DEPTH_TEXTURE_FORMAT,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });

        let trace_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("trace bind group layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::AccelerationStructure {
                            vertex_return: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: WORLD_POSITION_TEXTURE_FORMAT,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });

        let visualize_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("visualize bind group layout"),
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
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
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

        let coarse_depth_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("coarse depth pipeline layout"),
                bind_group_layouts: &[Some(&coarse_depth_bind_group_layout)],
                immediate_size: 0,
            });
        let trace_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("trace pipeline layout"),
                bind_group_layouts: &[Some(&trace_bind_group_layout)],
                immediate_size: 0,
            });
        let visualize_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("visualize pipeline layout"),
                bind_group_layouts: &[None, Some(&visualize_bind_group_layout)],
                immediate_size: 0,
            });

        let coarse_depth_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("coarse depth compute pipeline"),
                layout: Some(&coarse_depth_pipeline_layout),
                module: &shader,
                entry_point: Some("coarse_depth_prepass_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let trace_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("trace world position pipeline"),
            layout: Some(&trace_pipeline_layout),
            module: &shader,
            entry_point: Some("trace_world_position_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let visualize_world_position_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("visualize world position pipeline"),
                layout: Some(&visualize_pipeline_layout),
                module: &shader,
                entry_point: Some("visualize_world_position_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let visualize_tiles_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("visualize tile groups pipeline"),
                layout: Some(&visualize_pipeline_layout),
                module: &shader,
                entry_point: Some("visualize_tile_groups_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let coarse_depth_bind_group = Self::create_coarse_depth_bind_group(
            device,
            &coarse_depth_bind_group_layout,
            coarse_depth_view,
            tlas,
            camera_buffer,
            voxel_mask_buffer,
        );
        let trace_bind_group = Self::create_trace_bind_group(
            device,
            &trace_bind_group_layout,
            world_position_view,
            coarse_depth_view,
            tlas,
            camera_buffer,
            voxel_mask_buffer,
        );
        let visualize_bind_group = Self::create_visualize_bind_group(
            device,
            &visualize_bind_group_layout,
            output_view,
            world_position_view,
            debug_visualization_buffer,
        );

        Self {
            coarse_depth_bind_group_layout,
            trace_bind_group_layout,
            visualize_bind_group_layout,
            coarse_depth_pipeline,
            trace_pipeline,
            visualize_world_position_pipeline,
            visualize_tiles_pipeline,
            coarse_depth_bind_group,
            trace_bind_group,
            visualize_bind_group,
        }
    }

    pub(crate) fn rebind(
        &mut self,
        device: &wgpu::Device,
        output_view: &wgpu::TextureView,
        world_position_view: &wgpu::TextureView,
        coarse_depth_view: &wgpu::TextureView,
        tlas: &wgpu::Tlas,
        camera_buffer: &wgpu::Buffer,
        debug_visualization_buffer: &wgpu::Buffer,
        voxel_mask_buffer: &wgpu::Buffer,
    ) {
        self.coarse_depth_bind_group = Self::create_coarse_depth_bind_group(
            device,
            &self.coarse_depth_bind_group_layout,
            coarse_depth_view,
            tlas,
            camera_buffer,
            voxel_mask_buffer,
        );
        self.trace_bind_group = Self::create_trace_bind_group(
            device,
            &self.trace_bind_group_layout,
            world_position_view,
            coarse_depth_view,
            tlas,
            camera_buffer,
            voxel_mask_buffer,
        );
        self.visualize_bind_group = Self::create_visualize_bind_group(
            device,
            &self.visualize_bind_group_layout,
            output_view,
            world_position_view,
            debug_visualization_buffer,
        );
    }

    pub(crate) fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        width: u32,
        height: u32,
        coarse_width: u32,
        coarse_height: u32,
    ) {
        {
            let mut coarse_depth_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("coarse depth compute pass"),
                timestamp_writes: None,
            });
            coarse_depth_pass.set_pipeline(&self.coarse_depth_pipeline);
            coarse_depth_pass.set_bind_group(0, &self.coarse_depth_bind_group, &[]);
            coarse_depth_pass.dispatch_workgroups(
                coarse_width.div_ceil(WORKGROUP_SIZE),
                coarse_height.div_ceil(WORKGROUP_SIZE),
                1,
            );
        }

        {
            let mut trace_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("world position trace pass"),
                timestamp_writes: None,
            });
            trace_pass.set_pipeline(&self.trace_pipeline);
            trace_pass.set_bind_group(0, &self.trace_bind_group, &[]);
            trace_pass.dispatch_workgroups(
                width.div_ceil(WORKGROUP_SIZE),
                height.div_ceil(WORKGROUP_SIZE),
                1,
            );
        }

        let mut visualize_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("world position visualize pass"),
            timestamp_writes: None,
        });
        visualize_pass.set_pipeline(match DEFAULT_VISUALIZATION_MODE {
            VisualizationMode::WorldPosition => &self.visualize_world_position_pipeline,
            VisualizationMode::TileGroups => &self.visualize_tiles_pipeline,
        });
        visualize_pass.set_bind_group(1, &self.visualize_bind_group, &[]);
        visualize_pass.dispatch_workgroups(
            width.div_ceil(WORKGROUP_SIZE),
            height.div_ceil(WORKGROUP_SIZE),
            1,
        );
    }

    fn create_coarse_depth_bind_group(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        coarse_depth_view: &wgpu::TextureView,
        tlas: &wgpu::Tlas,
        camera_buffer: &wgpu::Buffer,
        voxel_mask_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("coarse depth bind group"),
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: tlas.as_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: voxel_mask_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(coarse_depth_view),
                },
            ],
        })
    }

    fn create_trace_bind_group(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        world_position_view: &wgpu::TextureView,
        coarse_depth_view: &wgpu::TextureView,
        tlas: &wgpu::Tlas,
        camera_buffer: &wgpu::Buffer,
        voxel_mask_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("trace bind group"),
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: tlas.as_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: voxel_mask_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(coarse_depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(world_position_view),
                },
            ],
        })
    }

    fn create_visualize_bind_group(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        output_view: &wgpu::TextureView,
        world_position_view: &wgpu::TextureView,
        debug_visualization_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("visualize bind group"),
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(world_position_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: debug_visualization_buffer.as_entire_binding(),
                },
            ],
        })
    }
}
