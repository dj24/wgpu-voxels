struct FrameParams {
    time_seconds: f32,
    object_count: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0)
var<storage, read_write> voxel_occupancy: array<atomic<u32>>;

@group(0) @binding(1)
var<uniform> frame_params: FrameParams;

const VOXEL_GRID_DIM_U32: u32 = 64u;
const REGION_AXIS_U32: u32 = 8u;
const MASK_WORD_BITS_U32: u32 = 32u;
const MASK_WORD_COUNT_U32: u32 = 16u;
const COARSE_MASK_WORD_OFFSET_U32: u32 = 16u;
const LEAF_MASK_WORD_OFFSET_U32: u32 = 18u;
const OCCUPANCY_WORD_COUNT_U32: u32 = 8210u;
const OBJECT_BOUNDS_MIN: vec3<f32> = vec3<f32>(-0.75, -0.75, -0.75);
const OBJECT_BOUNDS_MAX: vec3<f32> = vec3<f32>(0.75, 0.75, 0.75);
const HALF_PI: f32 = 1.5707963;

fn sd_torus(point: vec3<f32>, radii: vec2<f32>) -> f32 {
    let q = vec2<f32>(length(point.xz) - radii.x, point.y);
    return length(q) - radii.y;
}

fn rotate_y(point: vec3<f32>, angle: f32) -> vec3<f32> {
    let s = sin(angle);
    let c = cos(angle);
    return vec3<f32>(c * point.x + s * point.z, point.y, -s * point.x + c * point.z);
}

fn rotate_z(point: vec3<f32>, angle: f32) -> vec3<f32> {
    let s = sin(angle);
    let c = cos(angle);
    return vec3<f32>(c * point.x - s * point.y, s * point.x + c * point.y, point.z);
}

fn debug_sdf(point: vec3<f32>) -> f32 {
    let rotated_point = rotate_y(point, -frame_params.time_seconds * 0.3);
    return sd_torus(rotate_z(rotated_point, HALF_PI), vec2<f32>(0.38, 0.16));
}

fn voxel_size() -> f32 {
    return (OBJECT_BOUNDS_MAX.x - OBJECT_BOUNDS_MIN.x) / f32(VOXEL_GRID_DIM_U32);
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
    return region_position.x + 4u * (region_position.y + 4u * region_position.z);
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

@compute @workgroup_size(256, 1, 1)
fn clear_occupancy_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let total_word_count = frame_params.object_count * OCCUPANCY_WORD_COUNT_U32;
    if (id.x >= total_word_count) {
        return;
    }

    atomicStore(&voxel_occupancy[id.x], 0u);
}

@compute @workgroup_size(8, 8, 2)
fn populate_debug_sdf_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let total_object_slices = frame_params.object_count * VOXEL_GRID_DIM_U32;
    if (id.x >= VOXEL_GRID_DIM_U32 || id.y >= VOXEL_GRID_DIM_U32 || id.z >= total_object_slices) {
        return;
    }

    let object_index = id.z / VOXEL_GRID_DIM_U32;
    let voxel_z = id.z % VOXEL_GRID_DIM_U32;
    let voxel = vec3<u32>(id.x, id.y, voxel_z);
    let size = voxel_size();
    let center = OBJECT_BOUNDS_MIN + (vec3<f32>(voxel) + vec3<f32>(0.5)) * size;

    if (debug_sdf(center) > 0.0) {
        return;
    }

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
