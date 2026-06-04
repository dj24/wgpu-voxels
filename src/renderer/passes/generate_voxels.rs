use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::scene::OCCUPANCY_WORD_COUNT;

const CLEAR_WORKGROUP_SIZE: u32 = 256;
const POPULATE_WORKGROUP_SIZE_XY: u32 = 8;
const POPULATE_WORKGROUP_SIZE_Z: u32 = 2;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GenerationParams {
    active_object_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ChunkGenerationObject {
    chunk_origin: [f32; 4],
    object_index: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

pub(crate) struct GenerateVoxelsPass {
    bind_group: wgpu::BindGroup,
    clear_pipeline: wgpu::ComputePipeline,
    populate_pipeline: wgpu::ComputePipeline,
    total_mask_words: u32,
    active_object_count: u32,
    _params_buffer: wgpu::Buffer,
    _object_buffer: wgpu::Buffer,
}

impl GenerateVoxelsPass {
    pub(crate) fn new(
        device: &wgpu::Device,
        voxel_mask_buffer: &wgpu::Buffer,
        objects: &[crate::scene::RenderObject],
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(
                            wgpu::BufferSize::new(core::mem::size_of::<GenerationParams>() as u64)
                                .expect("generation params size"),
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
            label: Some("populate chunk voxel masks pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("populate_chunk_noise_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("voxel generation params buffer"),
            contents: bytemuck::bytes_of(&GenerationParams {
                active_object_count: objects.len() as u32,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let generation_objects: Vec<_> = objects
            .iter()
            .map(|object| ChunkGenerationObject {
                chunk_origin: [
                    object.position[0],
                    object.position[1],
                    object.position[2],
                    0.0,
                ],
                object_index: object.object_index,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            })
            .collect();
        let object_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("voxel generation chunk metadata buffer"),
            contents: bytemuck::cast_slice(&generation_objects),
            usage: wgpu::BufferUsages::STORAGE,
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
                    resource: object_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            bind_group,
            clear_pipeline,
            populate_pipeline,
            total_mask_words: max_object_count(objects) * OCCUPANCY_WORD_COUNT as u32,
            active_object_count: objects.len() as u32,
            _params_buffer: params_buffer,
            _object_buffer: object_buffer,
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

        let total_object_slices = self.active_object_count * 64;
        let mut populate_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("populate chunk voxel masks pass"),
            timestamp_writes: None,
        });
        populate_pass.set_pipeline(&self.populate_pipeline);
        populate_pass.set_bind_group(0, &self.bind_group, &[]);
        populate_pass.dispatch_workgroups(
            64u32.div_ceil(POPULATE_WORKGROUP_SIZE_XY),
            64u32.div_ceil(POPULATE_WORKGROUP_SIZE_XY),
            total_object_slices.div_ceil(POPULATE_WORKGROUP_SIZE_Z),
        );
    }
}

fn max_object_count(objects: &[crate::scene::RenderObject]) -> u32 {
    objects
        .iter()
        .map(|object| object.object_index + 1)
        .max()
        .unwrap_or(0)
}
