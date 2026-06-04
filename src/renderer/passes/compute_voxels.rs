use crate::renderer::{
    DebugView,
    output::{
        COARSE_DEPTH_TEXTURE_FORMAT, OUTPUT_TEXTURE_FORMAT, SHADING_INPUT_TEXTURE_FORMAT,
        WORLD_POSITION_TEXTURE_FORMAT,
    },
};

const TILE_WORKGROUP_SIZE: u32 = 8;
#[repr(C)]
#[derive(Clone, Copy)]
struct PackedShadeCommand {
    words: [u32; 2],
}

struct ShadeCommandResources {
    count_buffer: wgpu::Buffer,
    command_buffer: wgpu::Buffer,
    dispatch_args_buffer: wgpu::Buffer,
}

pub(crate) struct ComputeVoxelsPass {
    coarse_depth_bind_group_layout: wgpu::BindGroupLayout,
    trace_bind_group_layout: wgpu::BindGroupLayout,
    visualize_scene_bind_group_layout: wgpu::BindGroupLayout,
    visualize_bind_group_layout: wgpu::BindGroupLayout,
    prepare_bind_group_layout: wgpu::BindGroupLayout,
    coarse_depth_pipeline: wgpu::ComputePipeline,
    trace_pipeline: wgpu::ComputePipeline,
    emit_shade_commands_pipeline: wgpu::ComputePipeline,
    prepare_shade_dispatch_args_pipeline: wgpu::ComputePipeline,
    consume_shade_commands_pipeline: wgpu::ComputePipeline,
    debug_visualization_pipeline: wgpu::ComputePipeline,
    shade_command_count_buffer: wgpu::Buffer,
    shade_command_buffer: wgpu::Buffer,
    shade_dispatch_args_buffer: wgpu::Buffer,
    coarse_depth_bind_group: wgpu::BindGroup,
    trace_bind_group: wgpu::BindGroup,
    visualize_scene_bind_group: wgpu::BindGroup,
    visualize_bind_group: wgpu::BindGroup,
    prepare_bind_group: wgpu::BindGroup,
}

impl ComputeVoxelsPass {
    pub(crate) fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        output_view: &wgpu::TextureView,
        world_position_view: &wgpu::TextureView,
        shading_input_view: &wgpu::TextureView,
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
        let shade_command_resources = Self::create_shade_command_resources(device, width, height);

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
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: SHADING_INPUT_TEXTURE_FORMAT,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });

        let visualize_scene_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("visualize scene bind group layout"),
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
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
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
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let prepare_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("prepare bind group layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
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
                bind_group_layouts: &[
                    Some(&visualize_scene_bind_group_layout),
                    Some(&visualize_bind_group_layout),
                ],
                immediate_size: 0,
            });
        let prepare_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("prepare pipeline layout"),
                bind_group_layouts: &[
                    Some(&visualize_scene_bind_group_layout),
                    Some(&prepare_bind_group_layout),
                ],
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

        let emit_shade_commands_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("emit shade commands pipeline"),
                layout: Some(&visualize_pipeline_layout),
                module: &shader,
                entry_point: Some("emit_shade_commands_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let prepare_shade_dispatch_args_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("prepare shade dispatch args pipeline"),
                layout: Some(&prepare_pipeline_layout),
                module: &shader,
                entry_point: Some("prepare_shade_dispatch_args_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let consume_shade_commands_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("consume shade commands pipeline"),
                layout: Some(&visualize_pipeline_layout),
                module: &shader,
                entry_point: Some("consume_shade_commands_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let debug_visualization_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("debug visualization pipeline"),
                layout: Some(&visualize_pipeline_layout),
                module: &shader,
                entry_point: Some("debug_visualization_main"),
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
            shading_input_view,
            coarse_depth_view,
            tlas,
            camera_buffer,
            voxel_mask_buffer,
        );
        let visualize_scene_bind_group = Self::create_visualize_scene_bind_group(
            device,
            &visualize_scene_bind_group_layout,
            tlas,
            voxel_mask_buffer,
        );
        let visualize_bind_group = Self::create_visualize_bind_group(
            device,
            &visualize_bind_group_layout,
            output_view,
            world_position_view,
            shading_input_view,
            debug_visualization_buffer,
            &shade_command_resources.count_buffer,
            &shade_command_resources.command_buffer,
        );
        let prepare_bind_group = Self::create_prepare_bind_group(
            device,
            &prepare_bind_group_layout,
            &shade_command_resources.count_buffer,
            &shade_command_resources.dispatch_args_buffer,
        );

        Self {
            coarse_depth_bind_group_layout,
            trace_bind_group_layout,
            visualize_scene_bind_group_layout,
            visualize_bind_group_layout,
            prepare_bind_group_layout,
            coarse_depth_pipeline,
            trace_pipeline,
            emit_shade_commands_pipeline,
            prepare_shade_dispatch_args_pipeline,
            consume_shade_commands_pipeline,
            debug_visualization_pipeline,
            shade_command_count_buffer: shade_command_resources.count_buffer,
            shade_command_buffer: shade_command_resources.command_buffer,
            shade_dispatch_args_buffer: shade_command_resources.dispatch_args_buffer,
            coarse_depth_bind_group,
            trace_bind_group,
            visualize_scene_bind_group,
            visualize_bind_group,
            prepare_bind_group,
        }
    }

    pub(crate) fn rebind(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
        output_view: &wgpu::TextureView,
        world_position_view: &wgpu::TextureView,
        shading_input_view: &wgpu::TextureView,
        coarse_depth_view: &wgpu::TextureView,
        tlas: &wgpu::Tlas,
        camera_buffer: &wgpu::Buffer,
        debug_visualization_buffer: &wgpu::Buffer,
        voxel_mask_buffer: &wgpu::Buffer,
    ) {
        let shade_command_resources = Self::create_shade_command_resources(device, width, height);
        self.shade_command_count_buffer = shade_command_resources.count_buffer;
        self.shade_command_buffer = shade_command_resources.command_buffer;
        self.shade_dispatch_args_buffer = shade_command_resources.dispatch_args_buffer;
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
            shading_input_view,
            coarse_depth_view,
            tlas,
            camera_buffer,
            voxel_mask_buffer,
        );
        self.visualize_scene_bind_group = Self::create_visualize_scene_bind_group(
            device,
            &self.visualize_scene_bind_group_layout,
            tlas,
            voxel_mask_buffer,
        );
        self.visualize_bind_group = Self::create_visualize_bind_group(
            device,
            &self.visualize_bind_group_layout,
            output_view,
            world_position_view,
            shading_input_view,
            debug_visualization_buffer,
            &self.shade_command_count_buffer,
            &self.shade_command_buffer,
        );
        self.prepare_bind_group = Self::create_prepare_bind_group(
            device,
            &self.prepare_bind_group_layout,
            &self.shade_command_count_buffer,
            &self.shade_dispatch_args_buffer,
        );
    }

    pub(crate) fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        width: u32,
        height: u32,
        coarse_width: u32,
        coarse_height: u32,
        debug_view: DebugView,
    ) {
        {
            let mut coarse_depth_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("coarse depth compute pass"),
                timestamp_writes: None,
            });
            coarse_depth_pass.set_pipeline(&self.coarse_depth_pipeline);
            coarse_depth_pass.set_bind_group(0, &self.coarse_depth_bind_group, &[]);
            coarse_depth_pass.dispatch_workgroups(
                coarse_width.div_ceil(TILE_WORKGROUP_SIZE),
                coarse_height.div_ceil(TILE_WORKGROUP_SIZE),
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
                width.div_ceil(TILE_WORKGROUP_SIZE),
                height.div_ceil(TILE_WORKGROUP_SIZE),
                1,
            );
        }

        encoder.clear_buffer(&self.shade_command_count_buffer, 0, None);

        {
            let mut emit_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("emit shade commands pass"),
                timestamp_writes: None,
            });
            emit_pass.set_pipeline(&self.emit_shade_commands_pipeline);
            emit_pass.set_bind_group(0, &self.visualize_scene_bind_group, &[]);
            emit_pass.set_bind_group(1, &self.visualize_bind_group, &[]);
            emit_pass.dispatch_workgroups(
                width.div_ceil(TILE_WORKGROUP_SIZE),
                height.div_ceil(TILE_WORKGROUP_SIZE),
                1,
            );
        }

        {
            let mut prepare_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("prepare shade dispatch args pass"),
                timestamp_writes: None,
            });
            prepare_pass.set_pipeline(&self.prepare_shade_dispatch_args_pipeline);
            prepare_pass.set_bind_group(0, &self.visualize_scene_bind_group, &[]);
            prepare_pass.set_bind_group(1, &self.prepare_bind_group, &[]);
            prepare_pass.dispatch_workgroups(1, 1, 1);
        }

        {
            let mut consume_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("consume shade commands pass"),
                timestamp_writes: None,
            });
            consume_pass.set_pipeline(&self.consume_shade_commands_pipeline);
            consume_pass.set_bind_group(0, &self.visualize_scene_bind_group, &[]);
            consume_pass.set_bind_group(1, &self.visualize_bind_group, &[]);
            consume_pass.dispatch_workgroups_indirect(&self.shade_dispatch_args_buffer, 0);
        }

        if matches!(debug_view, DebugView::Default) {
            return;
        }

        let mut debug_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("debug visualization pass"),
            timestamp_writes: None,
        });
        debug_pass.set_pipeline(&self.debug_visualization_pipeline);
        debug_pass.set_bind_group(0, &self.visualize_scene_bind_group, &[]);
        debug_pass.set_bind_group(1, &self.visualize_bind_group, &[]);
        debug_pass.dispatch_workgroups(
            width.div_ceil(TILE_WORKGROUP_SIZE),
            height.div_ceil(TILE_WORKGROUP_SIZE),
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
        shading_input_view: &wgpu::TextureView,
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
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(shading_input_view),
                },
            ],
        })
    }

    fn create_visualize_bind_group(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        output_view: &wgpu::TextureView,
        world_position_view: &wgpu::TextureView,
        shading_input_view: &wgpu::TextureView,
        debug_visualization_buffer: &wgpu::Buffer,
        shade_command_count_buffer: &wgpu::Buffer,
        shade_command_buffer: &wgpu::Buffer,
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
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(shading_input_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: shade_command_count_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: shade_command_buffer.as_entire_binding(),
                },
            ],
        })
    }

    fn create_prepare_bind_group(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        shade_command_count_buffer: &wgpu::Buffer,
        shade_dispatch_args_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("prepare bind group"),
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: shade_command_count_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: shade_dispatch_args_buffer.as_entire_binding(),
                },
            ],
        })
    }

    fn create_visualize_scene_bind_group(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        tlas: &wgpu::Tlas,
        voxel_mask_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("visualize scene bind group"),
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: tlas.as_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: voxel_mask_buffer.as_entire_binding(),
                },
            ],
        })
    }

    fn create_shade_command_resources(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> ShadeCommandResources {
        let max_commands = u64::from(width.max(1)) * u64::from(height.max(1));
        let count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shade command count buffer"),
            size: core::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let command_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shade command buffer"),
            size: max_commands * core::mem::size_of::<PackedShadeCommand>() as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let dispatch_args_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shade dispatch args buffer"),
            size: core::mem::size_of::<wgpu::util::DispatchIndirectArgs>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
            mapped_at_creation: false,
        });

        ShadeCommandResources {
            count_buffer,
            command_buffer,
            dispatch_args_buffer,
        }
    }
}
