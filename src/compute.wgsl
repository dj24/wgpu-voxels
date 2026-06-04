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
    camera_position: vec4<f32>,
    debug_view: vec4<u32>,
}

struct RayMarch {
    hit: bool,
    t: f32,
    normal: vec3<f32>,
    cell: vec3<i32>,
    step_count: u32,
}

struct BoxIntersection {
    hit: bool,
    t_enter: f32,
    t_exit: f32,
}

struct ShadeCommandCounter {
    value: atomic<u32>,
}

struct DispatchIndirectArgsStorage {
    x: u32,
    y: u32,
    z: u32,
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

@group(0) @binding(6)
var shading_input_output: texture_storage_2d<rgba32float, write>;

@group(1) @binding(0)
var output_texture: texture_storage_2d<rgba8unorm, write>;

@group(1) @binding(1)
var world_position_texture: texture_2d<f32>;

@group(1) @binding(2)
var<uniform> debug_visualization: DebugVisualizationParams;

@group(1) @binding(3)
var shading_input_texture: texture_2d<f32>;

@group(1) @binding(4)
var<storage, read_write> shade_command_count: ShadeCommandCounter;

@group(1) @binding(5)
var<storage, read_write> shade_commands: array<vec2<u32>>;

@group(1) @binding(6)
var<storage, read_write> shade_dispatch_args: DispatchIndirectArgsStorage;

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
const COARSE_DEPTH_BIAS_SCALE: f32 = 0.0;
const PRIMARY_RAY_T_MAX: f32 = 100.0;
const SHADOW_RAY_T_MIN: f32 = 0.001;
const SHADOW_RAY_T_MAX: f32 = 100.0;
const SHADE_COMMAND_WORKGROUP_SIZE: u32 = 64u;
const DEBUG_VIEW_DEFAULT: u32 = 0u;
const DEBUG_VIEW_HEATMAP: u32 = 1u;
const DEBUG_VIEW_WORLD_POSITION: u32 = 2u;
const DEBUG_VIEW_DEPTH: u32 = 3u;

var<workgroup> shared_keys: array<vec4<f32>, 64>;
var<workgroup> shared_valid: array<u32, 64>;

fn saturate(value: f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

fn heatmap_ramp(t: f32) -> vec3<f32> {
    let cool = vec3<f32>(0.04, 0.05, 0.08);
    let blue = vec3<f32>(0.05, 0.32, 0.95);
    let cyan = vec3<f32>(0.05, 0.9, 0.95);
    let yellow = vec3<f32>(1.0, 0.9, 0.15);
    let hot = vec3<f32>(1.0, 0.18, 0.08);
    let ramp_t = saturate(t);

    if (ramp_t < 0.25) {
        return mix(cool, blue, ramp_t / 0.25);
    }
    if (ramp_t < 0.5) {
        return mix(blue, cyan, (ramp_t - 0.25) / 0.25);
    }
    if (ramp_t < 0.75) {
        return mix(cyan, yellow, (ramp_t - 0.5) / 0.25);
    }
    return mix(yellow, hot, (ramp_t - 0.75) / 0.25);
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

fn voxel_occupancy_value(object_mask_offset: u32, cell: vec3<i32>) -> f32 {
    if (voxel_filled(object_mask_offset, cell)) {
        return 1.0;
    }
    return 0.0;
}

fn fallback_normal(direction: vec3<f32>, last_axis: i32, step_dir: vec3<i32>) -> vec3<f32> {
    if (last_axis == 0) {
        return vec3<f32>(-f32(step_dir.x), 0.0, 0.0);
    }
    if (last_axis == 1) {
        return vec3<f32>(0.0, -f32(step_dir.y), 0.0);
    }
    if (last_axis == 2) {
        return vec3<f32>(0.0, 0.0, -f32(step_dir.z));
    }

    let axis = abs(direction);
    if (axis.x >= axis.y && axis.x >= axis.z) {
        return vec3<f32>(select(1.0, -1.0, direction.x >= 0.0), 0.0, 0.0);
    }
    if (axis.y >= axis.z) {
        return vec3<f32>(0.0, select(1.0, -1.0, direction.y >= 0.0), 0.0);
    }
    return vec3<f32>(0.0, 0.0, select(1.0, -1.0, direction.z >= 0.0));
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
        return RayMarch(false, ray_t_max, vec3<f32>(0.0), vec3<i32>(-1), 0u);
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
            let gradient = vec3<f32>(
                voxel_occupancy_value(object_mask_offset, cell + vec3<i32>(-1, 0, 0)) - voxel_occupancy_value(object_mask_offset, cell + vec3<i32>(1, 0, 0)),
                voxel_occupancy_value(object_mask_offset, cell + vec3<i32>(0, -1, 0)) - voxel_occupancy_value(object_mask_offset, cell + vec3<i32>(0, 1, 0)),
                voxel_occupancy_value(object_mask_offset, cell + vec3<i32>(0, 0, -1)) - voxel_occupancy_value(object_mask_offset, cell + vec3<i32>(0, 0, 1)),
            );
            let local_normal = select(
                fallback_normal(local_direction, last_axis, step_dir),
                normalize(gradient),
                length(gradient) > 1e-5,
            );
            return RayMarch(true, max(t_enter, ray_t_min), local_normal, cell, step_count);
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

    return RayMarch(false, t_exit, vec3<f32>(0.0), vec3<i32>(-1), step_count);
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

fn shade_ndotl(base_color: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    let light_direction = normalize(vec3<f32>(0.45, 0.8, 0.35));
    let ndotl = saturate(dot(normalize(normal), light_direction));
    let diffuse = 0.15 + ndotl * 0.85;
    return base_color * diffuse;
}

fn sun_direction() -> vec3<f32> {
    return normalize(vec3<f32>(0.45, 0.8, 0.35));
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

fn debug_view_mode() -> u32 {
    return debug_visualization.debug_view.x;
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

fn shade_from_input(shading_input: vec4<f32>) -> vec3<f32> {
    let object_index = u32(max(shading_input.w, 0.0));
    return shade_ndotl(palette(object_index), shading_input.xyz);
}

fn sun_visibility(world_position: vec3<f32>, world_normal: vec3<f32>) -> f32 {
    let ray_direction = sun_direction();
    let ray_origin =
        world_position + ray_direction * voxel_size();

    var query: ray_query;
    let ray = RayDesc(0u, 0xffu, SHADOW_RAY_T_MIN, SHADOW_RAY_T_MAX, ray_origin, ray_direction);
    rayQueryInitialize(&query, scene_tlas, ray);

    while (rayQueryProceed(&query)) {
        let candidate = rayQueryGetCandidateIntersection(&query);
        if (candidate.kind != 3u) {
            continue;
        }

        let committed = rayQueryGetCommittedIntersection(&query);
        let ray_t_max = select(ray.tmax, committed.t, committed.kind != 0u);
        let local_origin = (candidate.world_to_object * vec4<f32>(ray_origin, 1.0)).xyz;
        let local_direction =
            normalize((candidate.world_to_object * vec4<f32>(ray_direction, 0.0)).xyz);
        let marched = intersect_voxel_object(
            local_origin,
            local_direction,
            ray.tmin,
            ray_t_max,
            candidate.instance_custom_data,
        );

        if (marched.hit) {
            return 0.0;
        }
    }

    return 1.0;
}

fn shaded_color(world_position: vec3<f32>, shading_input: vec4<f32>) -> vec3<f32> {
    let visibility = sun_visibility(world_position, shading_input.xyz);
    return shade_from_input(shading_input) * mix(visibility, 1.0, 0.15);
}

fn encoded_step_count(world_position: vec4<f32>) -> u32 {
    return u32(max(abs(world_position.w) - 1.0, 0.0));
}

fn has_heatmap_debug(world_position: vec4<f32>) -> bool {
    return abs(world_position.w) > 0.5;
}

fn is_visible_surface(world_position: vec4<f32>) -> bool {
    return world_position.w > 0.5;
}

fn shade_ray_complexity(base_color: vec3<f32>, step_count: u32) -> vec3<f32> {
    let normalized_steps = saturate(log2(f32(step_count) + 1.0) / log2(f32(MAX_RAY_MARCH_STEPS) + 1.0));
    let heatmap = heatmap_ramp(normalized_steps);
    return mix(base_color * 0.18, heatmap, 0.88);
}

fn depth_debug_color(world_position: vec3<f32>) -> vec3<f32> {
    let depth = length(world_position - debug_visualization.camera_position.xyz);
    let normalized_depth = saturate(log2(depth + 1.0) / log2(PRIMARY_RAY_T_MAX + 1.0));
    let value = 1.0 - normalized_depth;
    return vec3<f32>(value);
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

fn block_origin_from_base_index(workgroup_origin: vec2<u32>, base_index: u32) -> vec2<u32> {
    let local_x = base_index & 7u;
    let local_y = base_index >> 3u;
    return workgroup_origin + vec2<u32>(local_x, local_y);
}

fn max_command_count() -> u32 {
    let dimensions = textureDimensions(world_position_texture);
    return dimensions.x * dimensions.y;
}

fn coverage_mode_for_size(coverage_size: vec2<u32>) -> u32 {
    if (coverage_size.x == 4u && coverage_size.y == 4u) {
        return 2u;
    }
    if (coverage_size.x == 2u && coverage_size.y == 2u) {
        return 1u;
    }
    return 0u;
}

fn coverage_size_from_mode(coverage_mode: u32) -> vec2<u32> {
    switch coverage_mode {
        case 2u: {
            return vec2<u32>(4u, 4u);
        }
        case 1u: {
            return vec2<u32>(2u, 2u);
        }
        default: {
            return vec2<u32>(1u, 1u);
        }
    }
}

fn pack_command_origin(origin_pixel: vec2<u32>) -> u32 {
    return (origin_pixel.x & 0xffffu) | ((origin_pixel.y & 0xffffu) << 16u);
}

fn unpack_command_origin(packed_origin: u32) -> vec2<u32> {
    return vec2<u32>(packed_origin & 0xffffu, packed_origin >> 16u);
}

fn append_shade_command(
    origin_pixel: vec2<u32>,
    coverage_size: vec2<u32>,
) {
    let command_index = atomicAdd(&shade_command_count.value, 1u);
    if (command_index >= max_command_count()) {
        return;
    }

    shade_commands[command_index] = vec2<u32>(
        pack_command_origin(origin_pixel),
        coverage_mode_for_size(coverage_size),
    );
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

fn coverage_debug_color(origin: vec2<u32>, coverage: vec2<u32>, world_position: vec4<f32>) -> vec3<f32> {
    let visible = is_visible_surface(world_position);
    if (!visible) {
        let dimensions = textureDimensions(world_position_texture);
        let uv = (vec2<f32>(origin) + vec2<f32>(0.5)) / vec2<f32>(dimensions);
        return debug_background(uv);
    }

    if (coverage.x == 4u && coverage.y == 4u) {
        return group_debug_color(vec3<f32>(0.22, 0.82, 0.36), origin / vec2<u32>(4u, 4u));
    }
    if (coverage.x == 2u && coverage.y == 2u) {
        return group_debug_color(vec3<f32>(0.22, 0.58, 0.96), origin / vec2<u32>(2u, 2u));
    }
    return per_pixel_debug_color(world_position.xyz, origin);
}

fn broadcast_command_color(origin: vec2<u32>, coverage: vec2<u32>, color: vec3<f32>) {
    for (var offset_y = 0u; offset_y < coverage.y; offset_y = offset_y + 1u) {
        for (var offset_x = 0u; offset_x < coverage.x; offset_x = offset_x + 1u) {
            textureStore(
                output_texture,
                vec2<i32>(origin + vec2<u32>(offset_x, offset_y)),
                vec4<f32>(color, 1.0),
            );
        }
    }
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
    let ray = RayDesc(0u, 0xffu, 0.01, PRIMARY_RAY_T_MAX, ray_origin, ray_direction);
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
    let ray_t_min = max(coarse_depth - coarse_depth_bias, 0.01);

    var query: ray_query;
    let ray = RayDesc(0u, 0xffu, ray_t_min, PRIMARY_RAY_T_MAX, ray_origin, ray_direction);
    rayQueryInitialize(&query, scene_tlas, ray);

    var stored_world_position = vec4<f32>(0.0);
    var stored_shading_input = vec4<f32>(0.0);
    var accumulated_step_count = 0u;
    var closest_debug_t = ray.tmax;
    var closest_debug_step_count = 0u;
    var closest_debug_object_index = 0u;

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
        accumulated_step_count = accumulated_step_count + marched.step_count;

        if (!marched.hit) {
            if (marched.step_count > 0u && marched.t < closest_debug_t) {
                closest_debug_t = marched.t;
                closest_debug_step_count = marched.step_count;
                closest_debug_object_index = candidate.instance_custom_data;
            }
            continue;
        }

        let local_voxel_center = (vec3<f32>(marched.cell) + vec3<f32>(0.5)) / f32(VOXEL_GRID_DIM);
        let world_position = (candidate.object_to_world * vec4<f32>(local_voxel_center, 1.0)).xyz;
        let world_normal = normalize((candidate.object_to_world * vec4<f32>(marched.normal, 0.0)).xyz);
        stored_world_position = vec4<f32>(world_position, f32(accumulated_step_count) + 1.0);
        stored_shading_input = vec4<f32>(world_normal, f32(candidate.instance_custom_data));

        rayQueryGenerateIntersection(&query, hit_t);
    }

    if (!is_visible_surface(stored_world_position) && closest_debug_step_count > 0u) {
        stored_world_position = vec4<f32>(0.0, 0.0, 0.0, -(f32(closest_debug_step_count) + 1.0));
        stored_shading_input = vec4<f32>(0.0, 0.0, 0.0, f32(closest_debug_object_index));
    }

    textureStore(world_position_output, vec2<i32>(id.xy), stored_world_position);
    textureStore(shading_input_output, vec2<i32>(id.xy), stored_shading_input);
}

@compute @workgroup_size(8, 8, 1)
fn emit_shade_commands_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let dimensions = textureDimensions(world_position_texture);
    let idx = local_index(lid.xy);
    let in_bounds = gid.x < dimensions.x && gid.y < dimensions.y;
    let workgroup_origin = gid.xy - lid.xy;
    let block4_id = (lid.y >> 2u) * 2u + (lid.x >> 2u);
    let block2_id = (lid.y >> 1u) * 4u + (lid.x >> 1u);
    let in4_index = (lid.y & 3u) * 4u + (lid.x & 3u);
    let in2_index = (lid.y & 1u) * 2u + (lid.x & 1u);

    if (in_bounds) {
        shared_keys[idx] = textureLoad(world_position_texture, vec2<i32>(gid.xy), 0);
        shared_valid[idx] = 1u;
    } else {
        shared_keys[idx] = vec4<f32>(0.0, 0.0, 0.0, -1.0);
        shared_valid[idx] = 0u;
    }

    workgroupBarrier();

    let block4_base = base_index_4x4(block4_id);
    let block2_base = base_index_2x2(block2_id);
    let block4_uniform = is_uniform_key(block4_base, 4u, 4u);
    let block2_uniform = !block4_uniform && is_uniform_key(block2_base, 2u, 2u);

    if (in4_index == 0u && block4_uniform) {
        let origin = block_origin_from_base_index(workgroup_origin, block4_base);
        append_shade_command(origin, vec2<u32>(4u, 4u));
        return;
    }

    if (in2_index == 0u && block2_uniform) {
        let origin = block_origin_from_base_index(workgroup_origin, block2_base);
        append_shade_command(origin, vec2<u32>(2u, 2u));
        return;
    }

    if (in_bounds && !block4_uniform && !block2_uniform) {
        append_shade_command(gid.xy, vec2<u32>(1u, 1u));
    }
}

@compute @workgroup_size(1, 1, 1)
fn prepare_shade_dispatch_args_main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (any(id != vec3<u32>(0u))) {
        return;
    }

    let command_count = atomicLoad(&shade_command_count.value);
    shade_dispatch_args.x = max(
        1u,
        (command_count + SHADE_COMMAND_WORKGROUP_SIZE - 1u) / SHADE_COMMAND_WORKGROUP_SIZE,
    );
    shade_dispatch_args.y = 1u;
    shade_dispatch_args.z = 1u;
}

fn consume_shade_command(command_index: u32) {
    let command_count = atomicLoad(&shade_command_count.value);
    if (command_index >= command_count) {
        return;
    }

    let command = shade_commands[command_index];
    let origin = unpack_command_origin(command.x);
    let coverage = coverage_size_from_mode(command.y);
    let world_position = textureLoad(world_position_texture, vec2<i32>(origin), 0);
    let shading_input = textureLoad(shading_input_texture, vec2<i32>(origin), 0);
    let color = select(
        shaded_color(world_position.xyz, shading_input),
        coverage_debug_color(origin, coverage, world_position),
        !is_visible_surface(world_position),
    );
    broadcast_command_color(origin, coverage, color);
}

@compute @workgroup_size(64, 1, 1)
fn consume_shade_commands_main(@builtin(global_invocation_id) id: vec3<u32>) {
    consume_shade_command(id.x);
}

@compute @workgroup_size(8, 8, 1)
fn debug_visualization_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dimensions = textureDimensions(world_position_texture);
    if (id.x >= dimensions.x || id.y >= dimensions.y) {
        return;
    }

    let pixel = vec2<i32>(id.xy);
    let world_position = textureLoad(world_position_texture, pixel, 0);
    let shading_input = textureLoad(shading_input_texture, pixel, 0);
    let uv = (vec2<f32>(id.xy) + vec2<f32>(0.5)) / vec2<f32>(dimensions);
    let background = debug_background(uv);
    let mode = debug_view_mode();

    var color = background;

    if (mode == DEBUG_VIEW_HEATMAP) {
        let base_color = palette(u32(max(shading_input.w, 0.0)));
        let heatmap_color = shade_ray_complexity(base_color, encoded_step_count(world_position));
        color = select(background, heatmap_color, has_heatmap_debug(world_position));
    } else if (mode == DEBUG_VIEW_WORLD_POSITION) {
        color = select(
            background,
            normalized_world_position(world_position.xyz),
            is_visible_surface(world_position),
        );
    } else if (mode == DEBUG_VIEW_DEPTH) {
        color = select(
            background,
            depth_debug_color(world_position.xyz),
            is_visible_surface(world_position),
        );
    } else if (mode == DEBUG_VIEW_DEFAULT) {
        return;
    }

    textureStore(output_texture, pixel, vec4<f32>(color, 1.0));
}
