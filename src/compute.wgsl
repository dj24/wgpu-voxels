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

const VOXEL_GRID_DIM: i32 = 24;
const OBJECT_BOUNDS_MIN: vec3<f32> = vec3<f32>(-0.75, -0.75, -0.75);
const OBJECT_BOUNDS_MAX: vec3<f32> = vec3<f32>(0.75, 0.75, 0.75);

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

fn sdf_sphere(point: vec3<f32>) -> f32 {
    return length(point) - 0.55;
}

fn voxel_size() -> f32 {
    return (OBJECT_BOUNDS_MAX.x - OBJECT_BOUNDS_MIN.x) / f32(VOXEL_GRID_DIM);
}

fn voxel_center(cell: vec3<i32>) -> vec3<f32> {
    let size = voxel_size();
    return OBJECT_BOUNDS_MIN + (vec3<f32>(cell) + vec3<f32>(0.5)) * size;
}

fn voxel_filled(cell: vec3<i32>) -> bool {
    if (any(cell < vec3<i32>(0)) || any(cell >= vec3<i32>(VOXEL_GRID_DIM))) {
        return false;
    }

    return sdf_sphere(voxel_center(cell)) <= 0.0;
}

fn voxel_occupancy_value(cell: vec3<i32>) -> f32 {
    if (voxel_filled(cell)) {
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
) -> vec3<i32> {
    let local_point = clamp(origin + direction * t_enter, bounds_min, bounds_max - vec3<f32>(1e-4));
    let relative = clamp(
        (local_point - bounds_min) / max(bounds_max - bounds_min, vec3<f32>(1e-5)),
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
    return min(vec3<i32>(relative * f32(VOXEL_GRID_DIM)), vec3<i32>(VOXEL_GRID_DIM - 1));
}

fn rebuild_dda_state(
    origin: vec3<f32>,
    direction: vec3<f32>,
    bounds_min: vec3<f32>,
    cell: vec3<i32>,
) -> vec3<f32> {
    let size = voxel_size();
    let step_mask = vec3<f32>(
        select(0.0, 1.0, direction.x >= 0.0),
        select(0.0, 1.0, direction.y >= 0.0),
        select(0.0, 1.0, direction.z >= 0.0),
    );
    let next_boundary = bounds_min + (vec3<f32>(cell) + step_mask) * size;
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
    var cell = initial_grid_cell(
        local_origin,
        local_direction,
        t_enter,
        OBJECT_BOUNDS_MIN,
        OBJECT_BOUNDS_MAX,
    );
    let step_dir = vec3<i32>(
        select(-1, 1, local_direction.x >= 0.0),
        select(-1, 1, local_direction.y >= 0.0),
        select(-1, 1, local_direction.z >= 0.0),
    );
    var t_max = rebuild_dda_state(local_origin, local_direction, OBJECT_BOUNDS_MIN, cell);
    let size = voxel_size();
    let t_delta = abs(vec3<f32>(size) / max(abs(local_direction), vec3<f32>(1e-5)));
    var last_axis = -1;
    var step_count = 0u;

    loop {
        if (any(cell < vec3<i32>(0)) || any(cell >= vec3<i32>(VOXEL_GRID_DIM))) {
            break;
        }

        step_count = step_count + 1u;
        if (voxel_filled(cell)) {
            let gradient = vec3<f32>(
                voxel_occupancy_value(cell + vec3<i32>(-1, 0, 0)) - voxel_occupancy_value(cell + vec3<i32>(1, 0, 0)),
                voxel_occupancy_value(cell + vec3<i32>(0, -1, 0)) - voxel_occupancy_value(cell + vec3<i32>(0, 1, 0)),
                voxel_occupancy_value(cell + vec3<i32>(0, 0, -1)) - voxel_occupancy_value(cell + vec3<i32>(0, 0, 1)),
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

        if (t_enter > t_exit || t_enter > ray_t_max || step_count > 512u) {
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

@compute @workgroup_size(8, 8, 1)
fn compute_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dimensions = textureDimensions(output_texture);
    if (id.x >= dimensions.x || id.y >= dimensions.y) {
        return;
    }

    let uv = (vec2<f32>(id.xy) + vec2<f32>(0.5)) / vec2<f32>(dimensions);
    let ray_origin = camera.position.xyz;
    let ray_direction = compute_camera_ray_direction(uv);

    var query: ray_query;
    let ray = RayDesc(0u, 0xffu, 0.01, 100.0, ray_origin, ray_direction);
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
        let light_dir = normalize(vec3<f32>(0.4, 0.8, 0.3));
        let diffuse = max(dot(best_hit.normal, light_dir), 0.0);
        let ambient = 0.18;
        let fog = exp(-0.08 * best_hit.t);
        let lit = best_hit.color * (ambient + diffuse * 0.82);
        color = mix(vec3<f32>(0.03, 0.04, 0.06), lit, fog);
    }

    textureStore(output_texture, vec2<i32>(id.xy), vec4<f32>(color, 1.0));
}
