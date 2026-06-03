enable wgpu_ray_query;

struct Camera {
    position: vec4<f32>,
    forward: vec4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
    viewport: vec4<f32>,
}

struct DebugVisualizationParams {
    world_min: vec4<f32>,
    world_extent: vec4<f32>,
}

struct RayMarch {
    hit: bool,
    t: f32,
    cell: vec3<i32>,
}

struct BoxIntersection {
    hit: bool,
    t_enter: f32,
    t_exit: f32,
}

@group(0) @binding(0)
var scene_tlas: acceleration_structure;

@group(0) @binding(1)
var<uniform> camera: Camera;

@group(0) @binding(2)
var<storage, read> voxel_occupancy: array<u32>;

@group(0) @binding(3)
var coarse_depth_texture: texture_2d<f32>;

@group(0) @binding(4)
var coarse_depth_output: texture_storage_2d<r32float, write>;

@group(0) @binding(5)
var world_position_output: texture_storage_2d<rgba32float, write>;

@group(1) @binding(0)
var output_texture: texture_storage_2d<rgba8unorm, write>;

@group(1) @binding(1)
var world_position_texture: texture_2d<f32>;

@group(1) @binding(2)
var<uniform> debug_visualization: DebugVisualizationParams;

const VOXEL_GRID_DIM: i32 = 64;
const REGION_AXIS: i32 = 8;
const REGION_AXIS_U32: u32 = 8u;
const COARSE_REGION_AXIS: i32 = 4;
const COARSE_REGION_AXIS_U32: u32 = 4u;
const COARSE_CELL_AXIS: i32 = 16;
const MASK_WORD_BITS_U32: u32 = 32u;
const MASK_WORD_COUNT_U32: u32 = 16u;
const COARSE_MASK_WORD_COUNT_U32: u32 = 2u;
const COARSE_MASK_WORD_OFFSET_U32: u32 = MASK_WORD_COUNT_U32;
const LEAF_MASK_WORD_OFFSET_U32: u32 = COARSE_MASK_WORD_OFFSET_U32 + COARSE_MASK_WORD_COUNT_U32;
const OCCUPANCY_WORD_COUNT_U32: u32 = 8210u;
const MAX_RAY_MARCH_STEPS: u32 = 512u;
const OBJECT_BOUNDS_MIN: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
const OBJECT_BOUNDS_MAX: vec3<f32> = vec3<f32>(1.0, 1.0, 1.0);
const COARSE_DEPTH_BIAS_SCALE: f32 = 1.7320508;

var<workgroup> shared_keys: array<vec4<f32>, 64>;
var<workgroup> shared_valid: array<u32, 64>;

fn saturate(value: f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

fn voxel_size() -> f32 {
    return (OBJECT_BOUNDS_MAX.x - OBJECT_BOUNDS_MIN.x) / f32(VOXEL_GRID_DIM);
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

fn flatten_leaf_index(local_position: vec3<u32>) -> u32 {
    return local_position.x
        + REGION_AXIS_U32 * (local_position.y + REGION_AXIS_U32 * local_position.z);
}

fn flatten_coarse_index(region_position: vec3<u32>) -> u32 {
    return region_position.x
        + COARSE_REGION_AXIS_U32
            * (region_position.y + COARSE_REGION_AXIS_U32 * region_position.z);
}

fn leaf_mask_word_offset(region_index: u32) -> u32 {
    return LEAF_MASK_WORD_OFFSET_U32 + region_index * MASK_WORD_COUNT_U32;
}

fn region_grid_dimensions() -> vec3<i32> {
    return vec3<i32>(VOXEL_GRID_DIM / REGION_AXIS);
}

fn coarse_region_grid_dimensions() -> vec3<i32> {
    return vec3<i32>(COARSE_REGION_AXIS);
}

fn object_mask_word(object_mask_offset: u32, word_index: u32) -> u32 {
    return voxel_occupancy[object_mask_offset + word_index];
}

fn coarse_region_mask_at_region_coord(
    object_mask_offset: u32,
    coarse_region_position: vec3<i32>,
) -> bool {
    let coarse_dims = coarse_region_grid_dimensions();
    if (any(coarse_region_position < vec3<i32>(0)) || any(coarse_region_position >= coarse_dims)) {
        return false;
    }

    let coarse_index = flatten_coarse_index(vec3<u32>(coarse_region_position));
    let coarse_word = object_mask_word(
        object_mask_offset,
        COARSE_MASK_WORD_OFFSET_U32 + occupancy_word_index(coarse_index),
    );
    return (coarse_word & occupancy_bit_mask(coarse_index)) != 0u;
}

fn region_occupancy_at(object_mask_offset: u32, cell: vec3<i32>) -> bool {
    if (any(cell < vec3<i32>(0)) || any(cell >= vec3<i32>(VOXEL_GRID_DIM))) {
        return false;
    }

    let voxel = vec3<u32>(cell);
    let region = vec3<u32>(voxel.x >> 3u, voxel.y >> 3u, voxel.z >> 3u);
    let region_index = flatten_region_index(region);
    let region_word = object_mask_word(object_mask_offset, occupancy_word_index(region_index));
    return (region_word & occupancy_bit_mask(region_index)) != 0u;
}

fn coarse_region_occupancy_at(object_mask_offset: u32, cell: vec3<i32>) -> bool {
    if (any(cell < vec3<i32>(0)) || any(cell >= vec3<i32>(VOXEL_GRID_DIM))) {
        return false;
    }

    let voxel = vec3<u32>(cell);
    let coarse_region = vec3<u32>(voxel.x >> 4u, voxel.y >> 4u, voxel.z >> 4u);
    return coarse_region_mask_at_region_coord(object_mask_offset, vec3<i32>(coarse_region));
}

fn voxel_filled(object_mask_offset: u32, cell: vec3<i32>) -> bool {
    if (!region_occupancy_at(object_mask_offset, cell)) {
        return false;
    }

    let voxel = vec3<u32>(cell);
    let region = vec3<u32>(voxel.x >> 3u, voxel.y >> 3u, voxel.z >> 3u);
    let region_index = flatten_region_index(region);
    let leaf_local = vec3<u32>(voxel.x & 7u, voxel.y & 7u, voxel.z & 7u);
    let leaf_index = flatten_leaf_index(leaf_local);
    let leaf_word = object_mask_word(
        object_mask_offset,
        leaf_mask_word_offset(region_index) + occupancy_word_index(leaf_index),
    );
    return (leaf_word & occupancy_bit_mask(leaf_index)) != 0u;
}

fn ray_box(
    origin: vec3<f32>,
    direction: vec3<f32>,
    bounds_min: vec3<f32>,
    bounds_max: vec3<f32>,
    ray_t_min: f32,
    ray_t_max: f32,
) -> BoxIntersection {
    let inv_dir = 1.0 / direction;
    let t0 = (bounds_min - origin) * inv_dir;
    let t1 = (bounds_max - origin) * inv_dir;
    let tmin = min(t0, t1);
    let tmax = max(t0, t1);
    let t_enter = max(max(tmin.x, tmin.y), max(tmin.z, ray_t_min));
    let t_exit = min(min(tmax.x, tmax.y), min(tmax.z, ray_t_max));
    return BoxIntersection(t_exit >= t_enter, t_enter, t_exit);
}

fn initial_grid_cell(
    origin: vec3<f32>,
    direction: vec3<f32>,
    t_enter: f32,
    bounds_min: vec3<f32>,
    bounds_max: vec3<f32>,
    grid_dims: vec3<i32>,
) -> vec3<i32> {
    let local_point = clamp(origin + direction * t_enter, bounds_min, bounds_max - vec3<f32>(1e-4));
    let relative = clamp(
        (local_point - bounds_min) / max(bounds_max - bounds_min, vec3<f32>(1e-5)),
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
    return min(vec3<i32>(relative * vec3<f32>(grid_dims)), grid_dims - vec3<i32>(1));
}

fn rebuild_dda_state(
    origin: vec3<f32>,
    direction: vec3<f32>,
    bounds_min: vec3<f32>,
    cell_size: f32,
    cell: vec3<i32>,
) -> vec3<f32> {
    let step_mask = vec3<f32>(
        select(0.0, 1.0, direction.x >= 0.0),
        select(0.0, 1.0, direction.y >= 0.0),
        select(0.0, 1.0, direction.z >= 0.0),
    );
    let next_boundary = bounds_min + (vec3<f32>(cell) + step_mask) * cell_size;
    return vec3<f32>(
        select(1e30, (next_boundary.x - origin.x) / direction.x, abs(direction.x) > 1e-5),
        select(1e30, (next_boundary.y - origin.y) / direction.y, abs(direction.y) > 1e-5),
        select(1e30, (next_boundary.z - origin.z) / direction.z, abs(direction.z) > 1e-5),
    );
}

fn intersect_voxel_object(
    local_origin: vec3<f32>,
    local_direction: vec3<f32>,
    ray_t_min: f32,
    ray_t_max: f32,
    instance_custom_data: u32,
) -> RayMarch {
    let object_mask_offset = instance_custom_data * OCCUPANCY_WORD_COUNT_U32;
    let box_hit = ray_box(
        local_origin,
        local_direction,
        OBJECT_BOUNDS_MIN,
        OBJECT_BOUNDS_MAX,
        ray_t_min,
        ray_t_max,
    );

    if (!box_hit.hit) {
        return RayMarch(false, ray_t_max, vec3<i32>(-1));
    }

    var t_enter = box_hit.t_enter;
    let t_exit = box_hit.t_exit;
    let grid_dims = vec3<i32>(VOXEL_GRID_DIM);
    let extent = OBJECT_BOUNDS_MAX - OBJECT_BOUNDS_MIN;
    var cell = initial_grid_cell(
        local_origin,
        local_direction,
        t_enter,
        OBJECT_BOUNDS_MIN,
        OBJECT_BOUNDS_MAX,
        grid_dims,
    );
    let step_dir = vec3<i32>(
        select(-1, 1, local_direction.x >= 0.0),
        select(-1, 1, local_direction.y >= 0.0),
        select(-1, 1, local_direction.z >= 0.0),
    );
    let size = voxel_size();
    var t_max = rebuild_dda_state(local_origin, local_direction, OBJECT_BOUNDS_MIN, size, cell);
    let t_delta = abs(vec3<f32>(size) / max(abs(local_direction), vec3<f32>(1e-5)));
    var last_axis = -1;
    var step_count = 0u;
    let advance_epsilon = size * 0.5;

    loop {
        if (any(cell < vec3<i32>(0)) || any(cell >= grid_dims)) {
            break;
        }

        step_count = step_count + 1u;
        if (step_count > MAX_RAY_MARCH_STEPS) {
            break;
        }

        if (!coarse_region_occupancy_at(object_mask_offset, cell)) {
            let coarse_region = vec3<i32>(cell.x >> 4, cell.y >> 4, cell.z >> 4);
            let coarse_local = vec3<i32>(cell.x & 15, cell.y & 15, cell.z & 15);
            let steps_to_coarse_exit = vec3<i32>(
                select(coarse_local.x + 1, COARSE_CELL_AXIS - coarse_local.x, step_dir.x > 0),
                select(coarse_local.y + 1, COARSE_CELL_AXIS - coarse_local.y, step_dir.y > 0),
                select(coarse_local.z + 1, COARSE_CELL_AXIS - coarse_local.z, step_dir.z > 0),
            );
            let t_coarse_exit = t_max + vec3<f32>(
                f32(steps_to_coarse_exit.x - 1) * t_delta.x,
                f32(steps_to_coarse_exit.y - 1) * t_delta.y,
                f32(steps_to_coarse_exit.z - 1) * t_delta.z,
            );

            if (t_coarse_exit.x < t_coarse_exit.y && t_coarse_exit.x < t_coarse_exit.z) {
                t_enter = t_coarse_exit.x;
                last_axis = 0;
            } else if (t_coarse_exit.y < t_coarse_exit.z) {
                t_enter = t_coarse_exit.y;
                last_axis = 1;
            } else {
                t_enter = t_coarse_exit.z;
                last_axis = 2;
            }

            if (t_enter > t_exit || t_enter > ray_t_max) {
                break;
            }

            let travel_t = min(t_enter + advance_epsilon, t_exit);
            let advanced_point = clamp(
                local_origin + local_direction * travel_t,
                OBJECT_BOUNDS_MIN,
                OBJECT_BOUNDS_MAX - vec3<f32>(1e-4),
            );
            let advanced_relative = clamp(
                (advanced_point - OBJECT_BOUNDS_MIN) / max(extent, vec3<f32>(1e-5)),
                vec3<f32>(0.0),
                vec3<f32>(1.0),
            );
            cell = min(vec3<i32>(advanced_relative * vec3<f32>(grid_dims)), grid_dims - vec3<i32>(1));
            if (all(coarse_region == vec3<i32>(cell.x >> 4, cell.y >> 4, cell.z >> 4))) {
                if (last_axis == 0) {
                    cell.x = cell.x + step_dir.x * steps_to_coarse_exit.x;
                } else if (last_axis == 1) {
                    cell.y = cell.y + step_dir.y * steps_to_coarse_exit.y;
                } else {
                    cell.z = cell.z + step_dir.z * steps_to_coarse_exit.z;
                }
            }
            t_max = rebuild_dda_state(local_origin, local_direction, OBJECT_BOUNDS_MIN, size, cell);
            continue;
        }

        if (!region_occupancy_at(object_mask_offset, cell)) {
            let region = vec3<i32>(cell.x >> 3, cell.y >> 3, cell.z >> 3);
            let region_local = vec3<i32>(cell.x & 7, cell.y & 7, cell.z & 7);
            let steps_to_region_exit = vec3<i32>(
                select(region_local.x + 1, REGION_AXIS - region_local.x, step_dir.x > 0),
                select(region_local.y + 1, REGION_AXIS - region_local.y, step_dir.y > 0),
                select(region_local.z + 1, REGION_AXIS - region_local.z, step_dir.z > 0),
            );
            let t_region_exit = t_max + vec3<f32>(
                f32(steps_to_region_exit.x - 1) * t_delta.x,
                f32(steps_to_region_exit.y - 1) * t_delta.y,
                f32(steps_to_region_exit.z - 1) * t_delta.z,
            );

            if (t_region_exit.x < t_region_exit.y && t_region_exit.x < t_region_exit.z) {
                t_enter = t_region_exit.x;
                last_axis = 0;
            } else if (t_region_exit.y < t_region_exit.z) {
                t_enter = t_region_exit.y;
                last_axis = 1;
            } else {
                t_enter = t_region_exit.z;
                last_axis = 2;
            }

            if (t_enter > t_exit || t_enter > ray_t_max) {
                break;
            }

            let travel_t = min(t_enter + advance_epsilon, t_exit);
            let advanced_point = clamp(
                local_origin + local_direction * travel_t,
                OBJECT_BOUNDS_MIN,
                OBJECT_BOUNDS_MAX - vec3<f32>(1e-4),
            );
            let advanced_relative = clamp(
                (advanced_point - OBJECT_BOUNDS_MIN) / max(extent, vec3<f32>(1e-5)),
                vec3<f32>(0.0),
                vec3<f32>(1.0),
            );
            cell = min(vec3<i32>(advanced_relative * vec3<f32>(grid_dims)), grid_dims - vec3<i32>(1));
            if (all(region == vec3<i32>(cell.x >> 3, cell.y >> 3, cell.z >> 3))) {
                if (last_axis == 0) {
                    cell.x = cell.x + step_dir.x * steps_to_region_exit.x;
                } else if (last_axis == 1) {
                    cell.y = cell.y + step_dir.y * steps_to_region_exit.y;
                } else {
                    cell.z = cell.z + step_dir.z * steps_to_region_exit.z;
                }
            }
            t_max = rebuild_dda_state(local_origin, local_direction, OBJECT_BOUNDS_MIN, size, cell);
            continue;
        }

        if (voxel_filled(object_mask_offset, cell)) {
            return RayMarch(true, max(t_enter, ray_t_min), cell);
        }

        if (t_max.x < t_max.y && t_max.x < t_max.z) {
            t_enter = t_max.x;
            t_max.x = t_max.x + t_delta.x;
            cell.x = cell.x + step_dir.x;
        } else if (t_max.y < t_max.z) {
            t_enter = t_max.y;
            t_max.y = t_max.y + t_delta.y;
            cell.y = cell.y + step_dir.y;
        } else {
            t_enter = t_max.z;
            t_max.z = t_max.z + t_delta.z;
            cell.z = cell.z + step_dir.z;
        }

        if (t_enter > t_exit || t_enter > ray_t_max) {
            break;
        }
    }

    return RayMarch(false, t_exit, vec3<i32>(-1));
}

fn compute_camera_ray_direction(uv: vec2<f32>) -> vec3<f32> {
    let screen = uv * 2.0 - 1.0;
    let view = camera.forward.xyz
        + screen.x * camera.viewport.x * camera.right.xyz
        - screen.y * camera.viewport.y * camera.up.xyz;
    return normalize(view);
}

fn debug_background(uv: vec2<f32>) -> vec3<f32> {
    let base = mix(vec3<f32>(0.04, 0.05, 0.07), vec3<f32>(0.10, 0.12, 0.16), uv.y);
    let band = 0.03 * vec3<f32>(uv.x, 1.0 - uv.y, 0.5 + 0.5 * uv.x);
    return base + band;
}

fn sample_min_coarse_depth(uv: vec2<f32>) -> f32 {
    let coarse_dims = textureDimensions(coarse_depth_texture);
    if (coarse_dims.x == 0u || coarse_dims.y == 0u) {
        return 0.0;
    }

    let coarse_size = vec2<i32>(coarse_dims);
    let center = clamp(
        vec2<i32>(uv * vec2<f32>(coarse_dims)),
        vec2<i32>(0),
        coarse_size - vec2<i32>(1),
    );
    var min_depth = 0.0;

    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let sample_coord = clamp(
                center + vec2<i32>(dx, dy),
                vec2<i32>(0),
                coarse_size - vec2<i32>(1),
            );
            let sample_depth = textureLoad(coarse_depth_texture, sample_coord, 0).x;
            min_depth = select(sample_depth, min(min_depth, sample_depth), min_depth > 0.0);
        }
    }

    return min_depth;
}

fn normalized_world_position(world_position: vec3<f32>) -> vec3<f32> {
    return clamp(
        (world_position - debug_visualization.world_min.xyz) / debug_visualization.world_extent.xyz,
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
}

fn hash01(coord: vec2<u32>) -> f32 {
    let value = sin(dot(vec2<f32>(coord), vec2<f32>(12.9898, 78.233))) * 43758.5453;
    return fract(value);
}

fn group_debug_color(base_color: vec3<f32>, group_coord: vec2<u32>) -> vec3<f32> {
    let noise = hash01(group_coord);
    let tint = mix(vec3<f32>(0.82, 0.88, 1.0), vec3<f32>(1.0, 0.86, 0.72), noise);
    let amplitude = mix(0.82, 1.08, noise);
    return clamp(base_color * amplitude + 0.14 * (tint - vec3<f32>(0.5)), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn per_pixel_debug_color(world_position: vec3<f32>, gid: vec2<u32>) -> vec3<f32> {
    let normalized = normalized_world_position(world_position);
    let noise = hash01(gid);
    let base_color = mix(vec3<f32>(0.96, 0.80, 0.20), normalized, 0.45);
    return clamp(base_color + (noise - 0.5) * 0.12, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn local_index(local_id: vec2<u32>) -> u32 {
    return local_id.y * 8u + local_id.x;
}

fn base_index_4x4(block_id: u32) -> u32 {
    let base_x = (block_id & 1u) * 4u;
    let base_y = (block_id >> 1u) * 4u;
    return base_y * 8u + base_x;
}

fn base_index_2x2(block_id: u32) -> u32 {
    let base_x = (block_id & 3u) * 2u;
    let base_y = (block_id >> 2u) * 2u;
    return base_y * 8u + base_x;
}

fn is_uniform_key(base_index: u32, width: u32, height: u32) -> bool {
    let base_key = shared_keys[base_index];
    if (shared_valid[base_index] == 0u || base_key.w < 0.5) {
        return false;
    }

    let base_x = base_index & 7u;
    let base_y = base_index >> 3u;
    for (var offset_y = 0u; offset_y < height; offset_y = offset_y + 1u) {
        for (var offset_x = 0u; offset_x < width; offset_x = offset_x + 1u) {
            let index = (base_y + offset_y) * 8u + (base_x + offset_x);
            if (shared_valid[index] == 0u) {
                return false;
            }
            let candidate = shared_keys[index];
            if (candidate.w < 0.5 || any(candidate != base_key)) {
                return false;
            }
        }
    }

    return true;
}

@compute @workgroup_size(8, 8, 1)
fn coarse_depth_prepass_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let coarse_dimensions = textureDimensions(coarse_depth_output);
    if (id.x >= coarse_dimensions.x || id.y >= coarse_dimensions.y) {
        return;
    }

    let uv = (vec2<f32>(id.xy) + vec2<f32>(0.5)) / vec2<f32>(coarse_dimensions);
    let ray_origin = camera.position.xyz;
    let ray_direction = compute_camera_ray_direction(uv);

    var query: ray_query;
    let ray = RayDesc(0u, 0xffu, 0.01, 100.0, ray_origin, ray_direction);
    rayQueryInitialize(&query, scene_tlas, ray);

    var coarse_depth = ray.tmax;

    while (rayQueryProceed(&query)) {
        let candidate = rayQueryGetCandidateIntersection(&query);
        if (candidate.kind != 3u) {
            continue;
        }

        let committed = rayQueryGetCommittedIntersection(&query);
        let ray_t_max = select(ray.tmax, committed.t, committed.kind != 0u);
        let local_origin = (candidate.world_to_object * vec4<f32>(ray_origin, 1.0)).xyz;
        let local_direction = normalize((candidate.world_to_object * vec4<f32>(ray_direction, 0.0)).xyz);
        let marched = intersect_voxel_object(
            local_origin,
            local_direction,
            ray.tmin,
            ray_t_max,
            candidate.instance_custom_data,
        );
        let hit_t = marched.t;

        if (marched.hit) {
            rayQueryGenerateIntersection(&query, hit_t);
            coarse_depth = hit_t;
        }
    }

    textureStore(
        coarse_depth_output,
        vec2<i32>(id.xy),
        vec4<f32>(coarse_depth, 0.0, 0.0, 0.0),
    );
}

@compute @workgroup_size(8, 8, 1)
fn trace_world_position_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dimensions = textureDimensions(world_position_output);
    if (id.x >= dimensions.x || id.y >= dimensions.y) {
        return;
    }

    let uv = (vec2<f32>(id.xy) + vec2<f32>(0.5)) / vec2<f32>(dimensions);
    let ray_origin = camera.position.xyz;
    let ray_direction = compute_camera_ray_direction(uv);
    let coarse_depth = sample_min_coarse_depth(uv);
    let coarse_depth_bias = voxel_size() * f32(REGION_AXIS) * COARSE_DEPTH_BIAS_SCALE;
    let ray_t_min = select(
        0.01,
        clamp(coarse_depth - coarse_depth_bias, 0.01, 99.999),
        coarse_depth > 0.0,
    );

    var query: ray_query;
    let ray = RayDesc(0u, 0xffu, ray_t_min, 100.0, ray_origin, ray_direction);
    rayQueryInitialize(&query, scene_tlas, ray);

    var stored_world_position = vec4<f32>(0.0);

    while (rayQueryProceed(&query)) {
        let candidate = rayQueryGetCandidateIntersection(&query);
        if (candidate.kind != 3u) {
            continue;
        }

        let committed = rayQueryGetCommittedIntersection(&query);
        let ray_t_max = select(ray.tmax, committed.t, committed.kind != 0u);
        let local_origin = (candidate.world_to_object * vec4<f32>(ray_origin, 1.0)).xyz;
        let marched = intersect_voxel_object(
            local_origin,
            ray_direction,
            ray.tmin,
            ray_t_max,
            candidate.instance_custom_data,
        );
        let hit_t = marched.t;

        if (!marched.hit) {
            continue;
        }

        let local_voxel_center = (vec3<f32>(marched.cell) + vec3<f32>(0.5)) / f32(VOXEL_GRID_DIM);
        let world_position = (candidate.object_to_world * vec4<f32>(local_voxel_center, 1.0)).xyz;
        stored_world_position = vec4<f32>(world_position, 1.0);

        rayQueryGenerateIntersection(&query, hit_t);
    }

    textureStore(world_position_output, vec2<i32>(id.xy), stored_world_position);
}

@compute @workgroup_size(8, 8, 1)
fn visualize_world_position_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dimensions = textureDimensions(world_position_texture);
    if (id.x >= dimensions.x || id.y >= dimensions.y) {
        return;
    }

    let key = textureLoad(world_position_texture, vec2<i32>(id.xy), 0);
    let uv = (vec2<f32>(id.xy) + vec2<f32>(0.5)) / vec2<f32>(dimensions);
    let color = select(
        debug_background(uv),
        normalized_world_position(key.xyz),
        key.w > 0.5,
    );
    textureStore(output_texture, vec2<i32>(id.xy), vec4<f32>(color, 1.0));
}

@compute @workgroup_size(8, 8, 1)
fn visualize_tile_groups_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let dimensions = textureDimensions(world_position_texture);
    let idx = local_index(lid.xy);
    let in_bounds = gid.x < dimensions.x && gid.y < dimensions.y;

    if (in_bounds) {
        shared_keys[idx] = textureLoad(world_position_texture, vec2<i32>(gid.xy), 0);
        shared_valid[idx] = 1u;
    } else {
        shared_keys[idx] = vec4<f32>(0.0, 0.0, 0.0, -1.0);
        shared_valid[idx] = 0u;
    }

    workgroupBarrier();

    let block4_id = (lid.y >> 2u) * 2u + (lid.x >> 2u);
    let block2_id = (lid.y >> 1u) * 4u + (lid.x >> 1u);

    if (!in_bounds) {
        return;
    }

    let block4_is_uniform = is_uniform_key(base_index_4x4(block4_id), 4u, 4u);
    let block2_is_uniform = is_uniform_key(base_index_2x2(block2_id), 2u, 2u);
    let key = shared_keys[idx];
    let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) / vec2<f32>(dimensions);
    let fallback_color = select(
        debug_background(uv),
        per_pixel_debug_color(key.xyz, gid.xy),
        key.w > 0.5,
    );
    var color = fallback_color;
    if (block4_is_uniform) {
        color = group_debug_color(
            vec3<f32>(0.16, 0.76, 0.32),
            vec2<u32>(gid.x >> 2u, gid.y >> 2u),
        );
    } else if (block2_is_uniform) {
        color = group_debug_color(
            vec3<f32>(0.18, 0.44, 0.96),
            vec2<u32>(gid.x >> 1u, gid.y >> 1u),
        );
    }

    textureStore(output_texture, vec2<i32>(gid.xy), vec4<f32>(color, 1.0));
}
