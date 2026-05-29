use std::{mem::size_of, ptr};

use bytemuck::{Pod, Zeroable};
use wgpu::{hal::CommandEncoder as _, hal::Device as _};

const PLACEHOLDER_TRIANGLE_COUNT: u32 = 128;

pub(crate) const OBJECT_BOUNDS_MIN: [f32; 3] = [-0.75, -0.75, -0.75];
pub(crate) const OBJECT_BOUNDS_MAX: [f32; 3] = [0.75, 0.75, 0.75];

pub(crate) const INSTANCE_POSITIONS: [[f32; 3]; 4] = [
    [-1.8, 0.0, 0.0],
    [-0.2, 0.3, -0.8],
    [1.4, -0.1, 0.4],
    [0.7, 0.8, -1.6],
];

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
    device: wgpu::Device,
    _aabb_input: wgpu::Buffer,
    _instance_input: wgpu::Buffer,
    scratch_buffer: Option<wgpu::hal::vulkan::Buffer>,
    _blas: wgpu::Blas,
    tlas: wgpu::Tlas,
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

        let aabb = AabbPositions {
            min_x: bounds_min[0],
            min_y: bounds_min[1],
            min_z: bounds_min[2],
            max_x: bounds_max[0],
            max_y: bounds_max[1],
            max_z: bounds_max[2],
        };
        let aabb_contents = bytemuck::bytes_of(&aabb);

        let (aabb_input, instance_input, scratch_buffer, instance_input_size) = {
            let raw_device = unsafe { device.as_hal::<wgpu::hal::api::Vulkan>() }
                .ok_or_else(|| String::from("Vulkan HAL device access is unavailable"))?;

            let aabb_input = create_raw_buffer_with_contents(
                &raw_device,
                "procedural interop aabb input",
                wgpu::BufferUses::MAP_WRITE
                    | wgpu::BufferUses::BOTTOM_LEVEL_ACCELERATION_STRUCTURE_INPUT,
                wgpu::hal::MemoryFlags::TRANSIENT | wgpu::hal::MemoryFlags::PREFER_COHERENT,
                aabb_contents,
            )?;

            let instance_bytes = encode_tlas_instances(&raw_device, &blas, instance_positions)?;
            let instance_input_size = instance_bytes.len() as u64;
            let instance_input = create_raw_buffer_with_contents(
                &raw_device,
                "procedural interop instance input",
                wgpu::BufferUses::MAP_WRITE
                    | wgpu::BufferUses::TOP_LEVEL_ACCELERATION_STRUCTURE_INPUT,
                wgpu::hal::MemoryFlags::TRANSIENT | wgpu::hal::MemoryFlags::PREFER_COHERENT,
                &instance_bytes,
            )?;

            let blas_entries = wgpu::hal::AccelerationStructureEntries::AABBs(vec![
                wgpu::hal::AccelerationStructureAABBs {
                    buffer: Some(&aabb_input),
                    offset: 0,
                    count: 1,
                    stride: size_of::<AabbPositions>() as u64,
                    flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
                },
            ]);

            let tlas_entries = wgpu::hal::AccelerationStructureEntries::Instances(
                wgpu::hal::AccelerationStructureInstances {
                    buffer: Some(&instance_input),
                    offset: 0,
                    count: instance_positions.len() as u32,
                },
            );

            let blas_sizes = unsafe {
                raw_device.get_acceleration_structure_build_sizes(
                    &wgpu::hal::GetAccelerationStructureBuildSizesDescriptor {
                        entries: &blas_entries,
                        flags: wgpu::hal::AccelerationStructureBuildFlags::PREFER_FAST_TRACE,
                    },
                )
            };

            let tlas_sizes = unsafe {
                raw_device.get_acceleration_structure_build_sizes(
                    &wgpu::hal::GetAccelerationStructureBuildSizesDescriptor {
                        entries: &tlas_entries,
                        flags: wgpu::hal::AccelerationStructureBuildFlags::PREFER_FAST_TRACE,
                    },
                )
            };

            let scratch_buffer = create_raw_buffer(
                &raw_device,
                "procedural interop scratch",
                blas_sizes
                    .build_scratch_size
                    .max(tlas_sizes.build_scratch_size),
                wgpu::BufferUses::ACCELERATION_STRUCTURE_SCRATCH,
                wgpu::hal::MemoryFlags::empty(),
            )?;

            (
                aabb_input,
                instance_input,
                scratch_buffer,
                instance_input_size,
            )
        };

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("procedural interop build encoder"),
        });

        {
            let raw_blas = unsafe { blas.as_hal::<wgpu::hal::api::Vulkan>() }
                .ok_or_else(|| String::from("Vulkan HAL BLAS access is unavailable"))?;

            unsafe {
                encoder.as_hal_mut::<wgpu::hal::api::Vulkan, _, _>(|maybe_raw_encoder| {
                    let raw_encoder = maybe_raw_encoder.ok_or_else(|| {
                        String::from("Vulkan HAL command encoder access is unavailable")
                    })?;

                    let blas_entries = wgpu::hal::AccelerationStructureEntries::AABBs(vec![
                        wgpu::hal::AccelerationStructureAABBs {
                            buffer: Some(&aabb_input),
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
                            scratch_buffer: &scratch_buffer,
                            scratch_buffer_offset: 0,
                        }],
                    );

                    Ok::<(), String>(())
                })
            }?;
        }

        {
            let raw_tlas = unsafe { tlas.as_hal::<wgpu::hal::api::Vulkan>() }
                .ok_or_else(|| String::from("Vulkan HAL TLAS access is unavailable"))?;

            unsafe {
                encoder.as_hal_mut::<wgpu::hal::api::Vulkan, _, _>(|maybe_raw_encoder| {
                    let raw_encoder = maybe_raw_encoder.ok_or_else(|| {
                        String::from("Vulkan HAL command encoder access is unavailable")
                    })?;

                    let tlas_entries = wgpu::hal::AccelerationStructureEntries::Instances(
                        wgpu::hal::AccelerationStructureInstances {
                            buffer: Some(&instance_input),
                            offset: 0,
                            count: instance_positions.len() as u32,
                        },
                    );

                    raw_encoder.transition_buffers(
                        [wgpu::hal::BufferBarrier {
                            buffer: &scratch_buffer,
                            usage: wgpu::hal::StateTransition {
                                from: wgpu::BufferUses::ACCELERATION_STRUCTURE_SCRATCH,
                                to: wgpu::BufferUses::ACCELERATION_STRUCTURE_SCRATCH,
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

                    raw_encoder.build_acceleration_structures(
                        1,
                        [wgpu::hal::BuildAccelerationStructureDescriptor {
                            entries: &tlas_entries,
                            mode: wgpu::hal::AccelerationStructureBuildMode::Build,
                            flags: wgpu::hal::AccelerationStructureBuildFlags::PREFER_FAST_TRACE,
                            source_acceleration_structure: None,
                            destination_acceleration_structure: &*raw_tlas,
                            scratch_buffer: &scratch_buffer,
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
            }?;
        }

        let aabb_input = unsafe {
            device.create_buffer_from_hal::<wgpu::hal::api::Vulkan>(
                aabb_input,
                &wgpu::BufferDescriptor {
                    label: Some("procedural interop aabb input"),
                    size: size_of::<AabbPositions>() as u64,
                    usage: wgpu::BufferUsages::BLAS_INPUT,
                    mapped_at_creation: false,
                },
            )
        };

        let instance_input = unsafe {
            device.create_buffer_from_hal::<wgpu::hal::api::Vulkan>(
                instance_input,
                &wgpu::BufferDescriptor {
                    label: Some("procedural interop instance input"),
                    size: instance_input_size,
                    usage: wgpu::BufferUsages::TLAS_INPUT,
                    mapped_at_creation: false,
                },
            )
        };

        unsafe {
            encoder.mark_acceleration_structures_built([&blas], [&tlas]);
        }
        queue.submit(Some(encoder.finish()));

        Ok(Self {
            device: device.clone(),
            _aabb_input: aabb_input,
            _instance_input: instance_input,
            scratch_buffer: Some(scratch_buffer),
            _blas: blas,
            tlas,
        })
    }

    pub(crate) fn tlas(&self) -> &wgpu::Tlas {
        &self.tlas
    }
}

impl Drop for ProceduralAccelerationScene {
    fn drop(&mut self) {
        let Some(scratch_buffer) = self.scratch_buffer.take() else {
            return;
        };

        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());

        if let Some(raw_device) = unsafe { self.device.as_hal::<wgpu::hal::api::Vulkan>() } {
            unsafe {
                raw_device.destroy_buffer(scratch_buffer);
            }
        }
    }
}

fn create_placeholder_blas(device: &wgpu::Device) -> wgpu::Blas {
    // The public wgpu BLAS API still sizes only triangle geometry. We over-allocate
    // a conservative placeholder and then build procedural AABBs into it through HAL.
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

fn create_raw_buffer(
    raw_device: &wgpu::hal::vulkan::Device,
    label: &'static str,
    size: u64,
    usage: wgpu::BufferUses,
    memory_flags: wgpu::hal::MemoryFlags,
) -> Result<wgpu::hal::vulkan::Buffer, String> {
    unsafe {
        raw_device.create_buffer(&wgpu::hal::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            memory_flags,
        })
    }
    .map_err(|error| format!("create {label}: {error}"))
}

fn create_raw_buffer_with_contents(
    raw_device: &wgpu::hal::vulkan::Device,
    label: &'static str,
    usage: wgpu::BufferUses,
    memory_flags: wgpu::hal::MemoryFlags,
    contents: &[u8],
) -> Result<wgpu::hal::vulkan::Buffer, String> {
    let buffer = create_raw_buffer(
        raw_device,
        label,
        contents.len() as u64,
        usage,
        memory_flags,
    )?;

    if !contents.is_empty() {
        let mapping = unsafe { raw_device.map_buffer(&buffer, 0..contents.len() as u64) }
            .map_err(|error| format!("map {label}: {error}"))?;

        unsafe {
            ptr::copy_nonoverlapping(contents.as_ptr(), mapping.ptr.as_ptr(), contents.len());
            raw_device.unmap_buffer(&buffer);
        }
    }

    Ok(buffer)
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
