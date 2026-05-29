use std::mem::size_of;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use wgpu::{hal::CommandEncoder as _, hal::Device as _};

const PLACEHOLDER_TRIANGLE_COUNT: u32 = 128;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AabbPositions {
    min_x: f32,
    min_y: f32,
    min_z: f32,
    max_x: f32,
    max_y: f32,
    max_z: f32,
}

pub(crate) struct ProceduralAccelerationScene {
    _aabb_input: wgpu::Buffer,
    _instance_input: wgpu::Buffer,
    _scratch_buffer: wgpu::Buffer,
    pub(crate) _blas: wgpu::Blas,
    pub(crate) _tlas: wgpu::Tlas,
}

impl ProceduralAccelerationScene {
    pub(crate) fn build(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        instance_positions: &[[f32; 3]],
        bounds_min: [f32; 3],
        bounds_max: [f32; 3],
    ) -> Result<Self, String> {
        let mut blas = create_placeholder_blas(device);
        let mut tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
            label: Some("procedural interop tlas"),
            max_instances: instance_positions.len() as u32,
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        });

        for (index, position) in instance_positions.iter().enumerate() {
            tlas[index] = Some(wgpu::TlasInstance::new(
                &blas,
                translation_transform(*position),
                index as u32,
                0xff,
            ));
        }

        let aabb_input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("procedural interop aabb input"),
            contents: bytemuck::bytes_of(&AabbPositions {
                min_x: bounds_min[0],
                min_y: bounds_min[1],
                min_z: bounds_min[2],
                max_x: bounds_max[0],
                max_y: bounds_max[1],
                max_z: bounds_max[2],
            }),
            usage: wgpu::BufferUsages::BLAS_INPUT | wgpu::BufferUsages::COPY_DST,
        });

        let instance_input = {
            let raw_device = unsafe { device.as_hal::<wgpu::hal::api::Vulkan>() }
                .ok_or_else(|| String::from("Vulkan HAL device access is unavailable"))?;
            let instance_bytes = encode_tlas_instances(&raw_device, &blas, instance_positions)?;

            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("procedural interop instance input"),
                contents: &instance_bytes,
                usage: wgpu::BufferUsages::TLAS_INPUT | wgpu::BufferUsages::COPY_DST,
            })
        };

        let scratch_size = {
            let raw_device = unsafe { device.as_hal::<wgpu::hal::api::Vulkan>() }
                .ok_or_else(|| String::from("Vulkan HAL device access is unavailable"))?;
            let raw_aabb_input = unsafe { aabb_input.as_hal::<wgpu::hal::api::Vulkan>() }
                .ok_or_else(|| String::from("Vulkan HAL AABB input access is unavailable"))?;
            let raw_instance_input = unsafe { instance_input.as_hal::<wgpu::hal::api::Vulkan>() }
                .ok_or_else(|| {
                String::from("Vulkan HAL instance input access is unavailable")
            })?;

            let blas_sizes = unsafe {
                raw_device.get_acceleration_structure_build_sizes(
                    &wgpu::hal::GetAccelerationStructureBuildSizesDescriptor {
                        entries: &wgpu::hal::AccelerationStructureEntries::AABBs(vec![
                            wgpu::hal::AccelerationStructureAABBs {
                                buffer: Some(&*raw_aabb_input),
                                offset: 0,
                                count: 1,
                                stride: size_of::<AabbPositions>() as u64,
                                flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
                            },
                        ]),
                        flags: wgpu::hal::AccelerationStructureBuildFlags::PREFER_FAST_TRACE,
                    },
                )
            };

            let tlas_sizes = unsafe {
                raw_device.get_acceleration_structure_build_sizes(
                    &wgpu::hal::GetAccelerationStructureBuildSizesDescriptor {
                        entries: &wgpu::hal::AccelerationStructureEntries::Instances(
                            wgpu::hal::AccelerationStructureInstances {
                                buffer: Some(&*raw_instance_input),
                                offset: 0,
                                count: instance_positions.len() as u32,
                            },
                        ),
                        flags: wgpu::hal::AccelerationStructureBuildFlags::PREFER_FAST_TRACE,
                    },
                )
            };

            blas_sizes
                .build_scratch_size
                .max(tlas_sizes.build_scratch_size)
        };

        let scratch_buffer = {
            let raw_device = unsafe { device.as_hal::<wgpu::hal::api::Vulkan>() }
                .ok_or_else(|| String::from("Vulkan HAL device access is unavailable"))?;
            let raw_scratch = unsafe {
                raw_device.create_buffer(&wgpu::hal::BufferDescriptor {
                    label: Some("procedural interop scratch"),
                    size: scratch_size,
                    usage: wgpu::BufferUses::ACCELERATION_STRUCTURE_SCRATCH,
                    memory_flags: wgpu::hal::MemoryFlags::empty(),
                })
            }
            .map_err(|error| format!("create scratch buffer: {error}"))?;

            unsafe {
                device.create_buffer_from_hal::<wgpu::hal::api::Vulkan>(
                    raw_scratch,
                    &wgpu::BufferDescriptor {
                        label: Some("procedural interop scratch"),
                        size: scratch_size,
                        usage: wgpu::BufferUsages::STORAGE,
                        mapped_at_creation: false,
                    },
                )
            }
        };

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("procedural interop build encoder"),
        });

        {
            let raw_aabb_input = unsafe { aabb_input.as_hal::<wgpu::hal::api::Vulkan>() }
                .ok_or_else(|| String::from("Vulkan HAL AABB input access is unavailable"))?;
            let raw_instance_input = unsafe { instance_input.as_hal::<wgpu::hal::api::Vulkan>() }
                .ok_or_else(|| {
                String::from("Vulkan HAL instance input access is unavailable")
            })?;
            let raw_scratch = unsafe { scratch_buffer.as_hal::<wgpu::hal::api::Vulkan>() }
                .ok_or_else(|| String::from("Vulkan HAL scratch access is unavailable"))?;
            let raw_blas = unsafe { blas.as_hal::<wgpu::hal::api::Vulkan>() }
                .ok_or_else(|| String::from("Vulkan HAL BLAS access is unavailable"))?;
            let raw_tlas = unsafe { tlas.as_hal::<wgpu::hal::api::Vulkan>() }
                .ok_or_else(|| String::from("Vulkan HAL TLAS access is unavailable"))?;

            let build_result = unsafe {
                encoder.as_hal_mut::<wgpu::hal::api::Vulkan, _, _>(|maybe_raw_encoder| {
                    let raw_encoder = maybe_raw_encoder.ok_or_else(|| {
                        String::from("Vulkan HAL command encoder access is unavailable")
                    })?;

                    let blas_entries = wgpu::hal::AccelerationStructureEntries::AABBs(vec![
                        wgpu::hal::AccelerationStructureAABBs {
                            buffer: Some(&*raw_aabb_input),
                            offset: 0,
                            count: 1,
                            stride: size_of::<AabbPositions>() as u64,
                            flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
                        },
                    ]);

                    raw_encoder.place_acceleration_structure_barrier(
                        wgpu::hal::AccelerationStructureBarrier {
                            usage: wgpu::hal::StateTransition {
                                from: wgpu::hal::AccelerationStructureUses::empty(),
                                to: wgpu::hal::AccelerationStructureUses::BUILD_OUTPUT,
                            },
                        },
                    );

                    raw_encoder.build_acceleration_structures(
                        1,
                        [wgpu::hal::BuildAccelerationStructureDescriptor {
                            entries: &blas_entries,
                            mode: wgpu::hal::AccelerationStructureBuildMode::Build,
                            flags: wgpu::hal::AccelerationStructureBuildFlags::PREFER_FAST_TRACE,
                            source_acceleration_structure: None,
                            destination_acceleration_structure: &*raw_blas,
                            scratch_buffer: &*raw_scratch,
                            scratch_buffer_offset: 0,
                        }],
                    );

                    raw_encoder.transition_buffers(
                        [wgpu::hal::BufferBarrier {
                            buffer: &*raw_scratch,
                            usage: wgpu::hal::StateTransition {
                                from: wgpu::BufferUses::BOTTOM_LEVEL_ACCELERATION_STRUCTURE_INPUT,
                                to: wgpu::BufferUses::TOP_LEVEL_ACCELERATION_STRUCTURE_INPUT,
                            },
                        }]
                        .into_iter(),
                    );

                    raw_encoder.place_acceleration_structure_barrier(
                        wgpu::hal::AccelerationStructureBarrier {
                            usage: wgpu::hal::StateTransition {
                                from: wgpu::hal::AccelerationStructureUses::BUILD_OUTPUT,
                                to: wgpu::hal::AccelerationStructureUses::BUILD_INPUT,
                            },
                        },
                    );

                    let tlas_entries = wgpu::hal::AccelerationStructureEntries::Instances(
                        wgpu::hal::AccelerationStructureInstances {
                            buffer: Some(&*raw_instance_input),
                            offset: 0,
                            count: instance_positions.len() as u32,
                        },
                    );

                    raw_encoder.build_acceleration_structures(
                        1,
                        [wgpu::hal::BuildAccelerationStructureDescriptor {
                            entries: &tlas_entries,
                            mode: wgpu::hal::AccelerationStructureBuildMode::Build,
                            flags: wgpu::hal::AccelerationStructureBuildFlags::PREFER_FAST_TRACE,
                            source_acceleration_structure: None,
                            destination_acceleration_structure: &*raw_tlas,
                            scratch_buffer: &*raw_scratch,
                            scratch_buffer_offset: 0,
                        }],
                    );

                    raw_encoder.place_acceleration_structure_barrier(
                        wgpu::hal::AccelerationStructureBarrier {
                            usage: wgpu::hal::StateTransition {
                                from: wgpu::hal::AccelerationStructureUses::BUILD_OUTPUT,
                                to: wgpu::hal::AccelerationStructureUses::SHADER_INPUT,
                            },
                        },
                    );

                    Ok::<(), String>(())
                })
            };

            build_result?;
        }

        unsafe {
            encoder.mark_acceleration_structures_built([&blas], [&tlas]);
        }
        queue.submit(Some(encoder.finish()));

        Ok(Self {
            _aabb_input: aabb_input,
            _instance_input: instance_input,
            _scratch_buffer: scratch_buffer,
            _blas: blas,
            _tlas: tlas,
        })
    }
}

fn create_placeholder_blas(device: &wgpu::Device) -> wgpu::Blas {
    // The public wgpu BLAS API still sizes only triangle geometry. We intentionally
    // over-allocate with a conservative triangle placeholder, then build procedural
    // AABBs into that native allocation through the HAL/Vulkan interop path.
    device.create_blas(
        &wgpu::CreateBlasDescriptor {
            label: Some("procedural interop placeholder blas"),
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        },
        wgpu::BlasGeometrySizeDescriptors::Triangles {
            descriptors: vec![wgpu::BlasTriangleGeometrySizeDescriptor {
                vertex_format: wgpu::VertexFormat::Float32x3,
                vertex_count: PLACEHOLDER_TRIANGLE_COUNT * 3,
                index_format: None,
                index_count: None,
                flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
            }],
        },
    )
}

fn encode_tlas_instances(
    raw_device: &wgpu::hal::vulkan::Device,
    blas: &wgpu::Blas,
    instance_positions: &[[f32; 3]],
) -> Result<Vec<u8>, String> {
    let blas_address = blas
        .handle()
        .ok_or_else(|| String::from("placeholder BLAS did not expose a raw handle"))?;

    let mut bytes = Vec::new();
    for (index, position) in instance_positions.iter().enumerate() {
        bytes.extend(raw_device.tlas_instance_to_bytes(wgpu::hal::TlasInstance {
            transform: translation_transform(*position),
            custom_data: index as u32,
            mask: 0xff,
            blas_address,
        }));
    }

    Ok(bytes)
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
