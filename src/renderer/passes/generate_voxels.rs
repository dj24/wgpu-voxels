use crate::{
    renderer::FrameParams,
    scene::{OCCUPANCY_WORD_COUNT, VOXEL_GRID_DIM},
};

const CLEAR_WORKGROUP_SIZE: u32 = 256;
const POPULATE_WORKGROUP_SIZE_XY: u32 = 8;
const POPULATE_WORKGROUP_SIZE_Z: u32 = 2;

pub(crate) struct GenerateVoxelsPass {
    bind_group: wgpu::BindGroup,
    clear_pipeline: wgpu::ComputePipeline,
    populate_pipeline: wgpu::ComputePipeline,
    total_mask_words: u32,
    object_count: u32,
}

impl GenerateVoxelsPass {
    pub(crate) fn new(
        device: &wgpu::Device,
        voxel_mask_buffer: &wgpu::Buffer,
        frame_params_buffer: &wgpu::Buffer,
        object_count: u32,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("voxel generation shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../voxel_generation.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("voxel generation bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(
                            wgpu::BufferSize::new(core::mem::size_of::<FrameParams>() as u64)
                                .expect("frame params size"),
                        ),
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("voxel generation pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let clear_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("clear voxel masks pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("clear_occupancy_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let populate_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("populate debug sdf voxel masks pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("populate_debug_sdf_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("voxel generation bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: voxel_mask_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: frame_params_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            bind_group,
            clear_pipeline,
            populate_pipeline,
            total_mask_words: object_count * OCCUPANCY_WORD_COUNT as u32,
            object_count,
        }
    }

    pub(crate) fn dispatch(&self, encoder: &mut wgpu::CommandEncoder) {
        {
            let mut clear_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("clear voxel masks pass"),
                timestamp_writes: None,
            });
            clear_pass.set_pipeline(&self.clear_pipeline);
            clear_pass.set_bind_group(0, &self.bind_group, &[]);
            clear_pass.dispatch_workgroups(
                self.total_mask_words.div_ceil(CLEAR_WORKGROUP_SIZE),
                1,
                1,
            );
        }

        let total_object_slices = self.object_count * VOXEL_GRID_DIM;
        let mut populate_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("populate debug sdf voxel masks pass"),
            timestamp_writes: None,
        });
        populate_pass.set_pipeline(&self.populate_pipeline);
        populate_pass.set_bind_group(0, &self.bind_group, &[]);
        populate_pass.dispatch_workgroups(
            VOXEL_GRID_DIM.div_ceil(POPULATE_WORKGROUP_SIZE_XY),
            VOXEL_GRID_DIM.div_ceil(POPULATE_WORKGROUP_SIZE_XY),
            total_object_slices.div_ceil(POPULATE_WORKGROUP_SIZE_Z),
        );
    }
}
