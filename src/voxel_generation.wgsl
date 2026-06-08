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

@group(0) @binding(3)
var<storage, read_write> leaf_voxels: array<u32>;

const VOXEL_GRID_DIM_U32: u32 = 64u;
const LEAF_VOXEL_WORD_COUNT_U32: u32 = 262144u;
const REGION_AXIS_U32: u32 = 8u;
const COARSE_REGION_AXIS_U32: u32 = 4u;
const MASK_WORD_BITS_U32: u32 = 32u;
const MASK_WORD_COUNT_U32: u32 = 16u;
const COARSE_MASK_WORD_OFFSET_U32: u32 = 16u;
const LEAF_MASK_WORD_OFFSET_U32: u32 = 18u;
const OCCUPANCY_WORD_COUNT_U32: u32 = 8210u;
const OBJECT_BOUNDS_MIN: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
const MAX_HEIGHT: f32 = 3.0;
// Larger radii communicate broader implied form at the cost of local detail.
const DSS_NORMAL_KERNEL_RADIUS: i32 = 2;
const CORNELL_WALL_THICKNESS: f32 = 0.055;
const CORNELL_WHITE: vec3<f32> = vec3<f32>(0.82, 0.80, 0.76);
const CORNELL_RED: vec3<f32> = vec3<f32>(0.74, 0.12, 0.10);
const CORNELL_GREEN: vec3<f32> = vec3<f32>(0.16, 0.58, 0.16);

struct SceneSdfSample {
    distance: f32,
    color: vec3<f32>,
}

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

fn flatten_dense_voxel_index(voxel: vec3<u32>) -> u32 {
    return voxel.x
        + VOXEL_GRID_DIM_U32 * (voxel.y + VOXEL_GRID_DIM_U32 * voxel.z);
}

fn leaf_mask_word_offset(region_index: u32) -> u32 {
    return LEAF_MASK_WORD_OFFSET_U32 + region_index * MASK_WORD_COUNT_U32;
}

fn object_mask_word_index(object_index: u32, word_index: u32) -> u32 {
    return object_index * OCCUPANCY_WORD_COUNT_U32 + word_index;
}

fn object_leaf_word_index(object_index: u32, voxel_index: u32) -> u32 {
    return object_index * LEAF_VOXEL_WORD_COUNT_U32 + voxel_index;
}

fn set_mask_bit(object_index: u32, word_index: u32, bit_index: u32) {
    atomicOr(
        &voxel_occupancy[object_mask_word_index(object_index, word_index)],
        occupancy_bit_mask(bit_index),
    );
}

fn palette(index: u32) -> vec3<f32> {
    switch index % 4u {
        case 0u: {
            return vec3<f32>(0.96, 0.28, 0.24);
        }
        case 1u: {
            return vec3<f32>(0.16, 0.72, 0.98);
        }
        case 2u: {
            return vec3<f32>(0.98, 0.78, 0.24);
        }
        default: {
            return vec3<f32>(0.30, 0.92, 0.46);
        }
    }
}

fn world_position_from_voxel(chunk_origin: vec3<f32>, voxel: vec3<u32>) -> vec3<f32> {
    let size = voxel_size();
    return chunk_origin + OBJECT_BOUNDS_MIN + (vec3<f32>(voxel) + vec3<f32>(0.5)) * size;
}

fn world_position_from_voxel_i32(chunk_origin: vec3<f32>, voxel: vec3<i32>) -> vec3<f32> {
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
            mix(hash31(p), hash31(p + vec3<f32>(1.0, 0.0, 0.0)), f.x),
            mix(
                hash31(p + vec3<f32>(0.0, 1.0, 0.0)),
                hash31(p + vec3<f32>(1.0, 1.0, 0.0)),
                f.x,
            ),
            f.y,
        ),
        mix(
            mix(
                hash31(p + vec3<f32>(0.0, 0.0, 1.0)),
                hash31(p + vec3<f32>(1.0, 0.0, 1.0)),
                f.x,
            ),
            mix(
                hash31(p + vec3<f32>(0.0, 1.0, 1.0)),
                hash31(p + vec3<f32>(1.0, 1.0, 1.0)),
                f.x,
            ),
            f.y,
        ),
        f.z,
    );
}

fn terrain_density_value(position: vec3<f32>) -> f32 {
    let squish_factor = position.y / MAX_HEIGHT;
    let noise = noise3D(position);
    return noise - squish_factor - 0.5;
}

fn terrain_density_filled(chunk_origin: vec3<f32>, voxel: vec3<u32>) -> bool {
    return terrain_density_value(world_position_from_voxel(chunk_origin, voxel)) > 0.0;
}

fn terrain_density_filled_i32(chunk_origin: vec3<f32>, voxel: vec3<i32>) -> bool {
    return terrain_density_value(world_position_from_voxel_i32(chunk_origin, voxel)) > 0.0;
}

fn terrain_voxel_has_exposed_side(chunk_origin: vec3<f32>, voxel: vec3<i32>) -> bool {
    return !terrain_density_filled_i32(chunk_origin, voxel + vec3<i32>(-1, 0, 0))
        || !terrain_density_filled_i32(chunk_origin, voxel + vec3<i32>(1, 0, 0))
        || !terrain_density_filled_i32(chunk_origin, voxel + vec3<i32>(0, -1, 0))
        || !terrain_density_filled_i32(chunk_origin, voxel + vec3<i32>(0, 1, 0))
        || !terrain_density_filled_i32(chunk_origin, voxel + vec3<i32>(0, 0, -1))
        || !terrain_density_filled_i32(chunk_origin, voxel + vec3<i32>(0, 0, 1));
}

fn gaussian_weight(dx: i32, dy: i32, dz: i32, radius: i32) -> f32 {
    let d2 = f32(dx * dx + dy * dy + dz * dz);
    let sigma = max(0.75, f32(radius) * 0.65);
    return exp(-d2 / (2.0 * sigma * sigma));
}

fn terrain_continuous_density_gradient(chunk_origin: vec3<f32>, voxel: vec3<i32>) -> vec3<f32> {
    return vec3<f32>(
        terrain_density_value(world_position_from_voxel_i32(chunk_origin, voxel + vec3<i32>(-1, 0, 0)))
            - terrain_density_value(world_position_from_voxel_i32(chunk_origin, voxel + vec3<i32>(1, 0, 0))),
        terrain_density_value(world_position_from_voxel_i32(chunk_origin, voxel + vec3<i32>(0, -1, 0)))
            - terrain_density_value(world_position_from_voxel_i32(chunk_origin, voxel + vec3<i32>(0, 1, 0))),
        terrain_density_value(world_position_from_voxel_i32(chunk_origin, voxel + vec3<i32>(0, 0, -1)))
            - terrain_density_value(world_position_from_voxel_i32(chunk_origin, voxel + vec3<i32>(0, 0, 1))),
    );
}

fn normalize_or(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let length_squared = dot(v, v);
    if (length_squared > 1e-8) {
        return v * inverseSqrt(length_squared);
    }
    return fallback;
}

fn terrain_smoothed_density(chunk_origin: vec3<f32>, voxel: vec3<i32>, axis_to_ignore: i32) -> f32 {
    var sum = 0.0;
    var total = 0.0;

    for (var dx = -DSS_NORMAL_KERNEL_RADIUS; dx <= DSS_NORMAL_KERNEL_RADIUS; dx = dx + 1) {
        for (var dy = -DSS_NORMAL_KERNEL_RADIUS; dy <= DSS_NORMAL_KERNEL_RADIUS; dy = dy + 1) {
            for (var dz = -DSS_NORMAL_KERNEL_RADIUS; dz <= DSS_NORMAL_KERNEL_RADIUS; dz = dz + 1) {
                let wx = select(dx, 0, axis_to_ignore == 0);
                let wy = select(dy, 0, axis_to_ignore == 1);
                let wz = select(dz, 0, axis_to_ignore == 2);
                let weight = gaussian_weight(wx, wy, wz, DSS_NORMAL_KERNEL_RADIUS);
                let sample_voxel = voxel + vec3<i32>(dx, dy, dz);
                sum += select(0.0, 1.0, terrain_density_filled_i32(chunk_origin, sample_voxel))
                    * weight;
                total += weight;
            }
        }
    }

    if (total <= 1e-4) {
        return 0.0;
    }

    return sum / total;
}

fn terrain_dss_gradient_normal(chunk_origin: vec3<f32>, voxel: vec3<i32>) -> vec3<f32> {
    let r = DSS_NORMAL_KERNEL_RADIUS;
    return vec3<f32>(
        terrain_smoothed_density(chunk_origin, voxel + vec3<i32>(-r, 0, 0), 0)
            - terrain_smoothed_density(chunk_origin, voxel + vec3<i32>(r, 0, 0), 0),
        terrain_smoothed_density(chunk_origin, voxel + vec3<i32>(0, -r, 0), 1)
            - terrain_smoothed_density(chunk_origin, voxel + vec3<i32>(0, r, 0), 1),
        terrain_smoothed_density(chunk_origin, voxel + vec3<i32>(0, 0, -r), 2)
            - terrain_smoothed_density(chunk_origin, voxel + vec3<i32>(0, 0, r), 2),
    );
}

fn terrain_dss_centroid_normal(chunk_origin: vec3<f32>, voxel: vec3<i32>) -> vec3<f32> {
    var weighted_mass_offset = vec3<f32>(0.0);
    var total_weight = 0.0;

    for (var dx = -DSS_NORMAL_KERNEL_RADIUS; dx <= DSS_NORMAL_KERNEL_RADIUS; dx = dx + 1) {
        for (var dy = -DSS_NORMAL_KERNEL_RADIUS; dy <= DSS_NORMAL_KERNEL_RADIUS; dy = dy + 1) {
            for (var dz = -DSS_NORMAL_KERNEL_RADIUS; dz <= DSS_NORMAL_KERNEL_RADIUS; dz = dz + 1) {
                if (dx == 0 && dy == 0 && dz == 0) {
                    continue;
                }

                let sample_voxel = voxel + vec3<i32>(dx, dy, dz);
                if (!terrain_density_filled_i32(chunk_origin, sample_voxel)) {
                    continue;
                }

                let weight = gaussian_weight(dx, dy, dz, DSS_NORMAL_KERNEL_RADIUS);
                weighted_mass_offset += vec3<f32>(f32(dx), f32(dy), f32(dz)) * weight;
                total_weight += weight;
            }
        }
    }

    if (total_weight <= 1e-4) {
        return vec3<f32>(0.0);
    }

    return -weighted_mass_offset / total_weight;
}

fn quantize_unorm(value: f32, max_value: u32) -> u32 {
    return u32(round(clamp(value, 0.0, 1.0) * f32(max_value)));
}

fn quantize_normal_component(value: f32) -> u32 {
    return quantize_unorm(value * 0.5 + 0.5, 15u);
}

fn pack_leaf_voxel(
    material_type: u32,
    normal: vec3<f32>,
    color: vec3<f32>,
) -> u32 {
    let packed_normal = vec3<u32>(
        quantize_normal_component(normal.x),
        quantize_normal_component(normal.y),
        quantize_normal_component(normal.z),
    );
    let packed_normal_x = packed_normal.x;
    let packed_normal_y = packed_normal.y;
    let packed_normal_z = packed_normal.z;
    let packed_color_r = quantize_unorm(color.r, 63u);
    let packed_color_g = quantize_unorm(color.g, 63u);
    let packed_color_b = quantize_unorm(color.b, 63u);

    return (material_type & 0x3u)
        | ((packed_normal_x & 0xfu) << 2u)
        | ((packed_normal_y & 0xfu) << 6u)
        | ((packed_normal_z & 0xfu) << 10u)
        | ((packed_color_r & 0x3fu) << 14u)
        | ((packed_color_g & 0x3fu) << 20u)
        | ((packed_color_b & 0x3fu) << 26u);
}

fn terrain_voxel_payload(chunk_origin: vec3<f32>, object_index: u32, voxel: vec3<u32>) -> u32 {
    let voxel_i32 = vec3<i32>(voxel);
    if (!terrain_voxel_has_exposed_side(chunk_origin, voxel_i32)) {
        return pack_leaf_voxel(0u, vec3<f32>(0.0, 0.0, 0.0), palette(object_index));
    }
    let dss_gradient = terrain_dss_gradient_normal(chunk_origin, voxel_i32);
    let dss_centroid = terrain_dss_centroid_normal(chunk_origin, voxel_i32);
    let continuous_gradient = terrain_continuous_density_gradient(chunk_origin, voxel_i32);
    let local_normal = normalize_or(
        dss_gradient,
        normalize_or(
            dss_centroid,
            normalize_or(continuous_gradient, vec3<f32>(0.0, 1.0, 0.0)),
        ),
    );
    let color = palette(object_index);
    return pack_leaf_voxel(0u, local_normal, color);
}

fn sdf_box(point: vec3<f32>, half_extent: vec3<f32>) -> f32 {
    let q = abs(point) - half_extent;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn sdf_box_sample(
    point: vec3<f32>,
    center: vec3<f32>,
    half_extent: vec3<f32>,
    color: vec3<f32>,
) -> SceneSdfSample {
    return SceneSdfSample(sdf_box(point - center, half_extent), color);
}

fn rotate_y(point: vec3<f32>, angle: f32) -> vec3<f32> {
    let c = cos(angle);
    let s = sin(angle);

    return vec3<f32>(
        c * point.x - s * point.z,
        point.y,
        s * point.x + c * point.z,
    );
}

fn sdf_rotated_y_box_sample(
    point: vec3<f32>,
    center: vec3<f32>,
    half_extent: vec3<f32>,
    angle: f32,
    color: vec3<f32>,
) -> SceneSdfSample {
    let local_point = rotate_y(point - center, -angle);
    return SceneSdfSample(sdf_box(local_point, half_extent), color);
}

fn union_sample(a: SceneSdfSample, b: SceneSdfSample) -> SceneSdfSample {
    if (b.distance < a.distance) {
        return b;
    }
    return a;
}

fn cornell_scene_sample(local_position: vec3<f32>) -> SceneSdfSample {
    let wall_half = CORNELL_WALL_THICKNESS * 0.5;
    var sample = sdf_box_sample(
        local_position,
        vec3<f32>(0.5, wall_half, 0.5),
        vec3<f32>(0.5, wall_half, 0.5),
        CORNELL_WHITE,
    );
    sample = union_sample(
        sample,
        sdf_box_sample(
            local_position,
            vec3<f32>(0.5, 1.0 - wall_half, 0.5),
            vec3<f32>(0.5, wall_half, 0.5),
            CORNELL_WHITE,
        ),
    );
    sample = union_sample(
        sample,
        sdf_box_sample(
            local_position,
            vec3<f32>(0.5, 0.5, wall_half),
            vec3<f32>(0.5, 0.5, wall_half),
            CORNELL_WHITE,
        ),
    );
    sample = union_sample(
        sample,
        sdf_box_sample(
            local_position,
            vec3<f32>(wall_half, 0.5, 0.5),
            vec3<f32>(wall_half, 0.5, 0.5),
            CORNELL_RED,
        ),
    );
    sample = union_sample(
        sample,
        sdf_box_sample(
            local_position,
            vec3<f32>(1.0 - wall_half, 0.5, 0.5),
            vec3<f32>(wall_half, 0.5, 0.5),
            CORNELL_GREEN,
        ),
    );
    sample = union_sample(
        sample,
        sdf_rotated_y_box_sample(
            local_position,
            vec3<f32>(0.33, 0.17, 0.34),
            vec3<f32>(0.12, 0.17, 0.12),
            0.22,
            CORNELL_WHITE,
        ),
    );
    sample = union_sample(
        sample,
        sdf_rotated_y_box_sample(
            local_position,
            vec3<f32>(0.68, 0.31, 0.62),
            vec3<f32>(0.13, 0.31, 0.13),
            -0.28,
            CORNELL_WHITE,
        ),
    );
    return sample;
}

fn cornell_scene_distance(local_position: vec3<f32>) -> f32 {
    return cornell_scene_sample(local_position).distance;
}

fn cornell_scene_normal(local_position: vec3<f32>) -> vec3<f32> {
    let epsilon = voxel_size() * 0.75;
    let gradient = vec3<f32>(
        cornell_scene_distance(local_position + vec3<f32>(epsilon, 0.0, 0.0))
            - cornell_scene_distance(local_position - vec3<f32>(epsilon, 0.0, 0.0)),
        cornell_scene_distance(local_position + vec3<f32>(0.0, epsilon, 0.0))
            - cornell_scene_distance(local_position - vec3<f32>(0.0, epsilon, 0.0)),
        cornell_scene_distance(local_position + vec3<f32>(0.0, 0.0, epsilon))
            - cornell_scene_distance(local_position - vec3<f32>(0.0, 0.0, epsilon)),
    );
    return normalize_or(gradient, vec3<f32>(0.0, 1.0, 0.0));
}

fn cornell_voxel_payload(voxel: vec3<u32>) -> u32 {
    let local_position = (vec3<f32>(voxel) + vec3<f32>(0.5)) * voxel_size();
    let sample = cornell_scene_sample(local_position);
    return pack_leaf_voxel(0u, cornell_scene_normal(local_position), sample.color);
}

fn write_voxel_data(object_index: u32, voxel: vec3<u32>, payload: u32) {
    let region = voxel >> vec3<u32>(3u);
    let coarse_region = voxel >> vec3<u32>(4u);
    let region_index = flatten_region_index(region);
    let coarse_index = flatten_coarse_index(coarse_region);
    let leaf_index = flatten_leaf_index(voxel & vec3<u32>(7u));
    let dense_index = flatten_dense_voxel_index(voxel);

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
    leaf_voxels[object_leaf_word_index(object_index, dense_index)] = payload;
}

@compute @workgroup_size(256, 1, 1)
fn clear_occupancy_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let total_word_count = generation_params.active_object_count * OCCUPANCY_WORD_COUNT_U32;
    if (id.x >= total_word_count) {
        return;
    }

    atomicStore(&voxel_occupancy[id.x], 0u);
}

@compute @workgroup_size(256, 1, 1)
fn clear_leaf_voxels_main(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let total_voxel_word_count = generation_params.active_object_count * LEAF_VOXEL_WORD_COUNT_U32;
    let linear_index = id.x + id.y * num_workgroups.x * 256u;
    if (linear_index >= total_voxel_word_count) {
        return;
    }

    leaf_voxels[linear_index] = 0u;
}

@compute @workgroup_size(8, 8, 2)
fn populate_chunk_terrain_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let total_object_slices = generation_params.active_object_count * VOXEL_GRID_DIM_U32;
    if (id.x >= VOXEL_GRID_DIM_U32 || id.y >= VOXEL_GRID_DIM_U32 || id.z >= total_object_slices) {
        return;
    }

    let object_slot = id.z / VOXEL_GRID_DIM_U32;
    let voxel_z = id.z % VOXEL_GRID_DIM_U32;
    let object = chunk_objects[object_slot];
    let voxel = vec3<u32>(id.x, id.y, voxel_z);

    if (!terrain_density_filled(object.chunk_origin.xyz, voxel)) {
        return;
    }

    write_voxel_data(
        object.object_index,
        voxel,
        terrain_voxel_payload(object.chunk_origin.xyz, object.object_index, voxel),
    );
}

@compute @workgroup_size(8, 8, 2)
fn populate_chunk_cornell_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let total_object_slices = generation_params.active_object_count * VOXEL_GRID_DIM_U32;
    if (id.x >= VOXEL_GRID_DIM_U32 || id.y >= VOXEL_GRID_DIM_U32 || id.z >= total_object_slices) {
        return;
    }

    let object_slot = id.z / VOXEL_GRID_DIM_U32;
    let voxel_z = id.z % VOXEL_GRID_DIM_U32;
    let object = chunk_objects[object_slot];
    let voxel = vec3<u32>(id.x, id.y, voxel_z);
    let local_position = (vec3<f32>(voxel) + vec3<f32>(0.5)) * voxel_size();

    if (cornell_scene_distance(local_position) > 0.0) {
        return;
    }

    write_voxel_data(object.object_index, voxel, cornell_voxel_payload(voxel));
}
