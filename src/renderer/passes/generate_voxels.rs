use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::scene::{
    LEAF_VOXEL_WORD_COUNT, OCCUPANCY_WORD_COUNT, RenderObject, VoxelGenerationKind,
};

const CLEAR_WORKGROUP_SIZE: u32 = 256;
const POPULATE_WORKGROUP_SIZE_XY: u32 = 8;
const POPULATE_WORKGROUP_SIZE_Z: u32 = 2;
const VOXEL_GRID_DIM: u32 = 64;

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

struct PopulatePass {
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::ComputePipeline,
    active_object_count: u32,
    _params_buffer: wgpu::Buffer,
    _object_buffer: wgpu::Buffer,
}

pub(crate) struct GenerateVoxelsPass {
    clear_bind_group: wgpu::BindGroup,
    clear_pipeline: wgpu::ComputePipeline,
    clear_leaf_pipeline: wgpu::ComputePipeline,
    terrain_populate_pass: Option<PopulatePass>,
    cornell_populate_pass: Option<PopulatePass>,
    total_mask_words: u32,
    total_leaf_words: u32,
    _clear_params_buffer: wgpu::Buffer,
    _clear_object_buffer: wgpu::Buffer,
}

impl GenerateVoxelsPass {
    pub(crate) fn new(
        device: &wgpu::Device,
        voxel_mask_buffer: &wgpu::Buffer,
        leaf_voxel_buffer: &wgpu::Buffer,
        objects: &[RenderObject],
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
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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
        let clear_leaf_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("clear leaf voxels pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("clear_leaf_voxels_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let clear_object_buffer = create_generation_object_buffer(
            device,
            "voxel generation clear metadata buffer",
            &[ChunkGenerationObject::zeroed()],
        );
        let clear_params_buffer = create_params_buffer(
            device,
            "voxel generation clear params buffer",
            max_object_count(objects),
        );
        let clear_bind_group = create_bind_group(
            device,
            "voxel generation clear bind group",
            &bind_group_layout,
            voxel_mask_buffer,
            &clear_object_buffer,
            &clear_params_buffer,
            leaf_voxel_buffer,
        );

        let (terrain_objects, cornell_objects) = partition_generation_objects(objects);
        let terrain_populate_pass = create_populate_pass(
            device,
            &shader,
            &pipeline_layout,
            &bind_group_layout,
            voxel_mask_buffer,
            leaf_voxel_buffer,
            &terrain_objects,
            "terrain voxel generation",
            "populate_chunk_terrain_main",
        );
        let cornell_populate_pass = create_populate_pass(
            device,
            &shader,
            &pipeline_layout,
            &bind_group_layout,
            voxel_mask_buffer,
            leaf_voxel_buffer,
            &cornell_objects,
            "cornell voxel generation",
            "populate_chunk_cornell_main",
        );

        Self {
            clear_bind_group,
            clear_pipeline,
            clear_leaf_pipeline,
            terrain_populate_pass,
            cornell_populate_pass,
            total_mask_words: max_object_count(objects) * OCCUPANCY_WORD_COUNT as u32,
            total_leaf_words: max_object_count(objects) * LEAF_VOXEL_WORD_COUNT as u32,
            _clear_params_buffer: clear_params_buffer,
            _clear_object_buffer: clear_object_buffer,
        }
    }

    pub(crate) fn dispatch(&self, encoder: &mut wgpu::CommandEncoder) {
        {
            let mut clear_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("clear voxel masks pass"),
                timestamp_writes: None,
            });
            clear_pass.set_pipeline(&self.clear_pipeline);
            clear_pass.set_bind_group(0, &self.clear_bind_group, &[]);
            clear_pass.dispatch_workgroups(
                self.total_mask_words.div_ceil(CLEAR_WORKGROUP_SIZE),
                1,
                1,
            );
        }
        {
            let mut clear_leaf_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("clear leaf voxels pass"),
                timestamp_writes: None,
            });
            clear_leaf_pass.set_pipeline(&self.clear_leaf_pipeline);
            clear_leaf_pass.set_bind_group(0, &self.clear_bind_group, &[]);
            let total_leaf_workgroups = self.total_leaf_words.div_ceil(CLEAR_WORKGROUP_SIZE);
            let clear_leaf_dispatch_x = total_leaf_workgroups.min(u16::MAX as u32);
            let clear_leaf_dispatch_y =
                total_leaf_workgroups.div_ceil(clear_leaf_dispatch_x.max(1));
            clear_leaf_pass.dispatch_workgroups(
                clear_leaf_dispatch_x.max(1),
                clear_leaf_dispatch_y,
                1,
            );
        }

        dispatch_populate_pass(
            encoder,
            self.terrain_populate_pass.as_ref(),
            "populate terrain voxel masks pass",
        );
        dispatch_populate_pass(
            encoder,
            self.cornell_populate_pass.as_ref(),
            "populate cornell voxel masks pass",
        );
    }
}

fn create_populate_pass(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    pipeline_layout: &wgpu::PipelineLayout,
    bind_group_layout: &wgpu::BindGroupLayout,
    voxel_mask_buffer: &wgpu::Buffer,
    leaf_voxel_buffer: &wgpu::Buffer,
    objects: &[ChunkGenerationObject],
    label_prefix: &str,
    entry_point: &str,
) -> Option<PopulatePass> {
    if objects.is_empty() {
        return None;
    }

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(&format!("{label_prefix} pipeline")),
        layout: Some(pipeline_layout),
        module: shader,
        entry_point: Some(entry_point),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let object_buffer = create_generation_object_buffer(
        device,
        &format!("{label_prefix} metadata buffer"),
        objects,
    );
    let params_buffer = create_params_buffer(
        device,
        &format!("{label_prefix} params buffer"),
        objects.len() as u32,
    );
    let bind_group = create_bind_group(
        device,
        &format!("{label_prefix} bind group"),
        bind_group_layout,
        voxel_mask_buffer,
        &object_buffer,
        &params_buffer,
        leaf_voxel_buffer,
    );

    Some(PopulatePass {
        bind_group,
        pipeline,
        active_object_count: objects.len() as u32,
        _params_buffer: params_buffer,
        _object_buffer: object_buffer,
    })
}

fn create_params_buffer(
    device: &wgpu::Device,
    label: &str,
    active_object_count: u32,
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::bytes_of(&GenerationParams {
            active_object_count,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        }),
        usage: wgpu::BufferUsages::UNIFORM,
    })
}

fn create_generation_object_buffer(
    device: &wgpu::Device,
    label: &str,
    objects: &[ChunkGenerationObject],
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(objects),
        usage: wgpu::BufferUsages::STORAGE,
    })
}

fn create_bind_group(
    device: &wgpu::Device,
    label: &str,
    bind_group_layout: &wgpu::BindGroupLayout,
    voxel_mask_buffer: &wgpu::Buffer,
    object_buffer: &wgpu::Buffer,
    params_buffer: &wgpu::Buffer,
    leaf_voxel_buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout: bind_group_layout,
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
            wgpu::BindGroupEntry {
                binding: 3,
                resource: leaf_voxel_buffer.as_entire_binding(),
            },
        ],
    })
}

fn dispatch_populate_pass(
    encoder: &mut wgpu::CommandEncoder,
    populate_pass: Option<&PopulatePass>,
    label: &str,
) {
    let Some(populate_pass) = populate_pass else {
        return;
    };

    let total_object_slices = populate_pass.active_object_count * VOXEL_GRID_DIM;
    let mut populate_stage = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(label),
        timestamp_writes: None,
    });
    populate_stage.set_pipeline(&populate_pass.pipeline);
    populate_stage.set_bind_group(0, &populate_pass.bind_group, &[]);
    populate_stage.dispatch_workgroups(
        VOXEL_GRID_DIM.div_ceil(POPULATE_WORKGROUP_SIZE_XY),
        VOXEL_GRID_DIM.div_ceil(POPULATE_WORKGROUP_SIZE_XY),
        total_object_slices.div_ceil(POPULATE_WORKGROUP_SIZE_Z),
    );
}

fn partition_generation_objects(
    objects: &[RenderObject],
) -> (Vec<ChunkGenerationObject>, Vec<ChunkGenerationObject>) {
    let mut terrain_objects = Vec::new();
    let mut cornell_objects = Vec::new();

    for object in objects {
        let generation_object = ChunkGenerationObject {
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
        };

        match object.generation_kind {
            VoxelGenerationKind::Terrain => terrain_objects.push(generation_object),
            VoxelGenerationKind::Cornell => cornell_objects.push(generation_object),
        }
    }

    (terrain_objects, cornell_objects)
}

fn max_object_count(objects: &[RenderObject]) -> u32 {
    objects
        .iter()
        .map(|object| object.object_index + 1)
        .max()
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::partition_generation_objects;
    use crate::scene::{RenderObject, VoxelGenerationKind};

    #[test]
    fn partitions_generation_objects_by_kind() {
        let objects = vec![
            RenderObject {
                position: [0.0, 0.0, 0.0],
                object_index: 0,
                generation_kind: VoxelGenerationKind::Terrain,
            },
            RenderObject {
                position: [1.0, 0.0, 0.0],
                object_index: 1,
                generation_kind: VoxelGenerationKind::Cornell,
            },
            RenderObject {
                position: [2.0, 0.0, 0.0],
                object_index: 2,
                generation_kind: VoxelGenerationKind::Terrain,
            },
        ];

        let (terrain, cornell) = partition_generation_objects(&objects);

        assert_eq!(terrain.len(), 2);
        assert_eq!(cornell.len(), 1);
        assert_eq!(terrain[0].object_index, 0);
        assert_eq!(cornell[0].object_index, 1);
        assert_eq!(terrain[1].object_index, 2);
    }
}
