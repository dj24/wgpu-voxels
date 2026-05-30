enable wgpu_ray_query;

struct Camera {
    position: vec4<f32>,
    forward: vec4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
    viewport: vec4<f32>,
}

struct RayShade {
    hit: bool,
    t: f32,
    normal: vec3<f32>,
    color: vec3<f32>,
    step_count: u32,
}

@group(0) @binding(0)
var output_texture: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(1)
var scene_tlas: acceleration_structure;

@group(0) @binding(2)
var<uniform> camera: Camera;

@group(0) @binding(3)
var<storage, read> voxel_occupancy: array<u32>;

@group(0) @binding(4)
var coarse_depth_texture: texture_2d<f32>;

@group(0) @binding(5)
var coarse_depth_output: texture_storage_2d<r32float, write>;

const VOXEL_GRID_DIM: i32 = 64;
const VOXEL_GRID_DIM_U32: u32 = 64u;
const REGION_AXIS: i32 = 8;
const REGION_AXIS_U32: u32 = 8u;
const REGION_COUNT_U32: u32 = 512u;
const MASK_WORD_BITS_U32: u32 = 32u;
const MASK_WORD_COUNT_U32: u32 = 16u;
const LEAF_MASK_WORD_OFFSET_U32: u32 = MASK_WORD_COUNT_U32;
const OCCUPANCY_WORD_COUNT_U32: u32 = 8208u;
const OBJECT_BOUNDS_MIN: vec3<f32> = vec3<f32>(-0.75, -0.75, -0.75);
const OBJECT_BOUNDS_MAX: vec3<f32> = vec3<f32>(0.75, 0.75, 0.75);
const COARSE_DEPTH_BIAS_SCALE: f32 = 1.7320508;

struct BoxIntersection {
    hit: bool,
    t_enter: f32,
    t_exit: f32,
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

fn leaf_mask_word_offset(region_index: u32) -> u32 {
    return LEAF_MASK_WORD_OFFSET_U32 + region_index * MASK_WORD_COUNT_U32;
}

fn region_grid_dimensions() -> vec3<i32> {
    return vec3<i32>(VOXEL_GRID_DIM / REGION_AXIS);
}

fn object_mask_word(object_mask_offset: u32, word_index: u32) -> u32 {
    return voxel_occupancy[object_mask_offset + word_index];
}

fn region_mask_at_region_coord(object_mask_offset: u32, region_position: vec3<i32>) -> bool {
    let region_dims = region_grid_dimensions();
    if (any(region_position < vec3<i32>(0)) || any(region_position >= region_dims)) {
        return false;
    }

    let region_index = flatten_region_index(vec3<u32>(region_position));
    let region_word = object_mask_word(object_mask_offset, occupancy_word_index(region_index));
    return (region_word & occupancy_bit_mask(region_index)) != 0u;
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
) -> RayShade {
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
        return RayShade(false, ray_t_max, vec3<f32>(0.0), vec3<f32>(0.0), 0u);
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
        if (step_count > 512u) {
            break;
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
            return RayShade(
                true,
                max(t_enter, ray_t_min),
                local_normal,
                palette(instance_custom_data),
                step_count,
            );
        }

        if (t_max.x < t_max.y && t_max.x < t_max.z) {
            t_enter = t_max.x;
            t_max.x = t_max.x + t_delta.x;
            cell.x = cell.x + step_dir.x;
            last_axis = 0;
        } else if (t_max.y < t_max.z) {
            t_enter = t_max.y;
            t_max.y = t_max.y + t_delta.y;
            cell.y = cell.y + step_dir.y;
            last_axis = 1;
        } else {
            t_enter = t_max.z;
            t_max.z = t_max.z + t_delta.z;
            cell.z = cell.z + step_dir.z;
            last_axis = 2;
        }

        if (t_enter > t_exit || t_enter > ray_t_max) {
            break;
        }
    }

    return RayShade(false, ray_t_max, vec3<f32>(0.0), vec3<f32>(0.0), step_count);
}

fn compute_camera_ray_direction(uv: vec2<f32>) -> vec3<f32> {
    let screen = uv * 2.0 - 1.0;
    let view = camera.forward.xyz
        + screen.x * camera.viewport.x * camera.right.xyz
        - screen.y * camera.viewport.y * camera.up.xyz;
    return normalize(view);
}

fn shade_background(direction: vec3<f32>, uv: vec2<f32>) -> vec3<f32> {
    let horizon = 0.5 * (direction.y + 1.0);
    let sky = mix(vec3<f32>(0.07, 0.09, 0.13), vec3<f32>(0.28, 0.42, 0.72), horizon);
    let grid = 0.04 * vec3<f32>(uv, 1.0 - uv.x);
    return sky + grid;
}

fn shade_ray_complexity(base_color: vec3<f32>, step_count: u32) -> vec3<f32> {
    let normalized_steps = saturate(log2(f32(step_count) + 1.0) / log2(513.0));
    let heatmap = heatmap_ramp(normalized_steps);
    return mix(base_color * 0.18, heatmap, 0.88);
}

fn shade_ndotl(base_color: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    let light_direction = normalize(vec3<f32>(0.45, 0.8, 0.35));
    let ndotl = saturate(dot(normalize(normal), light_direction));
    let diffuse = 0.15 + ndotl * 0.85;
    return base_color * diffuse;
}

fn sample_min_coarse_depth(uv: vec2<f32>) -> f32 {
    let coarse_dims = textureDimensions(coarse_depth_texture, 0);
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
            if (sample_depth <= 0.0) {
                continue;
            }

            min_depth = select(sample_depth, min(min_depth, sample_depth), min_depth > 0.0);
        }
    }

    return min_depth;
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

    var coarse_depth = 0.0;

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

        if (marched.hit) {
            let hit_t = marched.t;
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
fn compute_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dimensions = textureDimensions(output_texture);
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

    var color = shade_background(ray_direction, uv);
    var best_hit = RayShade(false, ray.tmax, vec3<f32>(0.0), vec3<f32>(0.0), 0u);

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

        best_hit.step_count = best_hit.step_count + marched.step_count;
        if (!marched.hit) {
            continue;
        }

        let hit_t = marched.t;
        let world_normal = normalize((candidate.object_to_world * vec4<f32>(marched.normal, 0.0)).xyz);
        rayQueryGenerateIntersection(&query, hit_t);
        best_hit.hit = true;
        best_hit.t = hit_t;
        best_hit.normal = world_normal;
        best_hit.color = marched.color;
    }

    if (best_hit.hit) {
        let heatmap_color = shade_ray_complexity(best_hit.color, best_hit.step_count);
        let shaded_color = shade_ndotl(best_hit.color, best_hit.normal);
        color = select(shaded_color, heatmap_color, uv.x < 0.5);
    }

    textureStore(output_texture, vec2<i32>(id.xy), vec4<f32>(color, 1.0));
}
