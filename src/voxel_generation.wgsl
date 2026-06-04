struct GenerationParams {
    active_object_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct ChunkGenerationObject {
    chunk_origin: vec4<f32>,
    object_index: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0)
var<storage, read_write> voxel_occupancy: array<atomic<u32>>;

@group(0) @binding(1)
var<storage, read> chunk_objects: array<ChunkGenerationObject>;

@group(0) @binding(2)
var<uniform> generation_params: GenerationParams;

const VOXEL_GRID_DIM_U32: u32 = 64u;
const REGION_AXIS_U32: u32 = 8u;
const COARSE_REGION_AXIS_U32: u32 = 4u;
const MASK_WORD_BITS_U32: u32 = 32u;
const MASK_WORD_COUNT_U32: u32 = 16u;
const COARSE_MASK_WORD_OFFSET_U32: u32 = 16u;
const LEAF_MASK_WORD_OFFSET_U32: u32 = 18u;
const OCCUPANCY_WORD_COUNT_U32: u32 = 8210u;
const OBJECT_BOUNDS_MIN: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);

fn voxel_size() -> f32 {
    return 1.0 / f32(VOXEL_GRID_DIM_U32);
}

fn occupancy_word_index(bit_index: u32) -> u32 {
    return bit_index / MASK_WORD_BITS_U32;
}

fn occupancy_bit_mask(bit_index: u32) -> u32 {
    return 1u << (bit_index & 31u);
}

fn flatten_region_index(region_position: vec3<u32>) -> u32 {
    return region_position.x
        + REGION_AXIS_U32 * (region_position.y + REGION_AXIS_U32 * region_position.z);
}

fn flatten_coarse_index(region_position: vec3<u32>) -> u32 {
    return region_position.x
        + COARSE_REGION_AXIS_U32
            * (region_position.y + COARSE_REGION_AXIS_U32 * region_position.z);
}

fn flatten_leaf_index(local_position: vec3<u32>) -> u32 {
    return local_position.x
        + REGION_AXIS_U32 * (local_position.y + REGION_AXIS_U32 * local_position.z);
}

fn leaf_mask_word_offset(region_index: u32) -> u32 {
    return LEAF_MASK_WORD_OFFSET_U32 + region_index * MASK_WORD_COUNT_U32;
}

fn object_mask_word_index(object_index: u32, word_index: u32) -> u32 {
    return object_index * OCCUPANCY_WORD_COUNT_U32 + word_index;
}

fn set_mask_bit(object_index: u32, word_index: u32, bit_index: u32) {
    atomicOr(
        &voxel_occupancy[object_mask_word_index(object_index, word_index)],
        occupancy_bit_mask(bit_index),
    );
}

fn world_position(chunk_origin: vec3<f32>, voxel: vec3<u32>) -> vec3<f32> {
    let size = voxel_size();
    return chunk_origin + OBJECT_BOUNDS_MIN + (vec3<f32>(voxel) + vec3<f32>(0.5)) * size;
}

fn hash31(p: vec3<f32>) -> f32 {
  var p3 = fract(p * vec3<f32>(0.1031, 0.1030, 0.0973));
  p3 += dot(p3, p3.yxz + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

fn noise3D(x: vec3<f32>) -> f32 {
  let p = floor(x);
  let f = fract(x);

  return mix(
    mix(
      mix(hash31(p),
          hash31(p + vec3<f32>(1.0, 0.0, 0.0)),
          f.x),
      mix(hash31(p + vec3<f32>(0.0, 1.0, 0.0)),
          hash31(p + vec3<f32>(1.0, 1.0, 0.0)),
          f.x),
      f.y),
    mix(
      mix(hash31(p + vec3<f32>(0.0, 0.0, 1.0)),
          hash31(p + vec3<f32>(1.0, 0.0, 1.0)),
          f.x),
      mix(hash31(p + vec3<f32>(0.0, 1.0, 1.0)),
          hash31(p + vec3<f32>(1.0, 1.0, 1.0)),
          f.x),
      f.y),
    f.z);
}

const MAX_HEIGHT = 3;

fn chunk_density_filled(chunk_origin: vec3<f32>, voxel: vec3<u32>) -> bool {
    let position = world_position(chunk_origin, voxel);
    let squish_factor = f32(position.y) / f32(MAX_HEIGHT);
    let noise = noise3D(position);
    let squished_noise = noise - squish_factor;
    return squished_noise > 0.5;
}

@compute @workgroup_size(256, 1, 1)
fn clear_occupancy_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let total_word_count = generation_params.active_object_count * OCCUPANCY_WORD_COUNT_U32;
    if (id.x >= total_word_count) {
        return;
    }

    atomicStore(&voxel_occupancy[id.x], 0u);
}

@compute @workgroup_size(8, 8, 2)
fn populate_chunk_noise_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let total_object_slices = generation_params.active_object_count * VOXEL_GRID_DIM_U32;
    if (id.x >= VOXEL_GRID_DIM_U32 || id.y >= VOXEL_GRID_DIM_U32 || id.z >= total_object_slices) {
        return;
    }

    let object_slot = id.z / VOXEL_GRID_DIM_U32;
    let voxel_z = id.z % VOXEL_GRID_DIM_U32;
    let object = chunk_objects[object_slot];
    let voxel = vec3<u32>(id.x, id.y, voxel_z);

    if (!chunk_density_filled(object.chunk_origin.xyz, voxel)) {
        return;
    }

    let object_index = object.object_index;
    let region = voxel >> vec3<u32>(3u);
    let coarse_region = voxel >> vec3<u32>(4u);
    let region_index = flatten_region_index(region);
    let coarse_index = flatten_coarse_index(coarse_region);
    let leaf_index = flatten_leaf_index(voxel & vec3<u32>(7u));

    set_mask_bit(object_index, occupancy_word_index(region_index), region_index);
    set_mask_bit(
        object_index,
        COARSE_MASK_WORD_OFFSET_U32 + occupancy_word_index(coarse_index),
        coarse_index,
    );
    set_mask_bit(
        object_index,
        leaf_mask_word_offset(region_index) + occupancy_word_index(leaf_index),
        leaf_index,
    );
}
