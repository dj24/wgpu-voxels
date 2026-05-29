use crate::renderer::output::OUTPUT_TEXTURE_FORMAT;

const WORKGROUP_SIZE: u32 = 8;

pub(crate) struct ComputeVoxelsPass {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
}

impl ComputeVoxelsPass {
    pub(crate) fn new(
        device: &wgpu::Device,
        output_view: &wgpu::TextureView,
        tlas: &wgpu::Tlas,
        camera_buffer: &wgpu::Buffer,
        voxel_mask_buffer: &wgpu::Buffer,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("uv compute shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../compute.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("compute pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("uv compute pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("compute_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let bind_group = Self::create_bind_group(
            device,
            &bind_group_layout,
            output_view,
            tlas,
            camera_buffer,
            voxel_mask_buffer,
        );

        Self {
            bind_group_layout,
            pipeline,
            bind_group,
        }
    }

    pub(crate) fn rebind(
        &mut self,
        device: &wgpu::Device,
        output_view: &wgpu::TextureView,
        tlas: &wgpu::Tlas,
        camera_buffer: &wgpu::Buffer,
        voxel_mask_buffer: &wgpu::Buffer,
    ) {
        self.bind_group = Self::create_bind_group(
            device,
            &self.bind_group_layout,
            output_view,
            tlas,
            camera_buffer,
            voxel_mask_buffer,
        );
    }

    pub(crate) fn dispatch(&self, encoder: &mut wgpu::CommandEncoder, width: u32, height: u32) {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ray query compute pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &self.bind_group, &[]);
        compute_pass.dispatch_workgroups(
            width.div_ceil(WORKGROUP_SIZE),
            height.div_ceil(WORKGROUP_SIZE),
            1,
        );
    }

    fn create_bind_group(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        output_view: &wgpu::TextureView,
        tlas: &wgpu::Tlas,
        camera_buffer: &wgpu::Buffer,
        voxel_mask_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compute bind group"),
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: tlas.as_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: voxel_mask_buffer.as_entire_binding(),
                },
            ],
        })
    }
}
