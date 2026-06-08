struct Camera {
    position: vec4<f32>,
    forward: vec4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
    viewport: vec4<f32>,
}

@group(0) @binding(0)
var current_texture: texture_2d<f32>;

@group(0) @binding(1)
var world_position_texture: texture_2d<f32>;

@group(0) @binding(2)
var shading_input_texture: texture_2d<f32>;

@group(0) @binding(3)
var<uniform> camera: Camera;

const VOXEL_GRID_DIM: f32 = 64.0;
const OBJECT_BOUNDS_MIN: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
const OBJECT_BOUNDS_MAX: vec3<f32> = vec3<f32>(1.0, 1.0, 1.0);
const FACE_PLANE_TOLERANCE_SCALE: f32 = 0.04;
const DEBUG_FACE_OUTPUT_NONE: u32 = 0u;
const DEBUG_FACE_OUTPUT_AXIS: u32 = 1u;
const DEBUG_FACE_OUTPUT_QUADRANT: u32 = 2u;
const DEBUG_FACE_OUTPUT_VALID_NEIGHBOURS: u32 = 3u;
const DEBUG_FACE_OUTPUT_REJECTED_SAMPLES: u32 = 4u;
override DEBUG_FACE_OUTPUT_MODE: u32 = 0u;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

struct SampleResult {
    valid: bool,
    color: vec3<f32>,
    rejected_samples: u32,
}

struct FaceInfo {
    voxel_coord: vec3<i32>,
    local_position: vec3<f32>,
    normal: vec3<i32>,
    axis: u32,
    sign: f32,
    tangent_axis_0: u32,
    tangent_axis_1: u32,
    face_uv: vec2<f32>,
    plane: f32,
}

fn voxel_size() -> f32 {
    return (OBJECT_BOUNDS_MAX.x - OBJECT_BOUNDS_MIN.x) / VOXEL_GRID_DIM;
}

fn is_visible_voxel(world_position: vec4<f32>) -> bool {
    return world_position.w > 0.5;
}

fn pixel_coord(uv: vec2<f32>, dimensions: vec2<u32>) -> vec2<i32> {
    let max_coord = vec2<i32>(dimensions) - vec2<i32>(1);
    return clamp(vec2<i32>(uv * vec2<f32>(dimensions)), vec2<i32>(0), max_coord);
}

fn pixel_in_bounds(pixel: vec2<i32>, dimensions: vec2<u32>) -> bool {
    return all(pixel >= vec2<i32>(0)) && all(pixel < vec2<i32>(dimensions));
}

fn compute_camera_ray_direction(uv: vec2<f32>) -> vec3<f32> {
    let screen = uv * 2.0 - vec2<f32>(1.0, 1.0);
    let view =
        camera.forward.xyz
        + screen.x * camera.viewport.x * camera.right.xyz
        - screen.y * camera.viewport.y * camera.up.xyz;
    return normalize(view);
}

fn axis_component(value: vec3<f32>, axis: u32) -> f32 {
    if (axis == 0u) {
        return value.x;
    }
    if (axis == 1u) {
        return value.y;
    }
    return value.z;
}

fn axis_step(axis: u32, direction: i32) -> vec3<i32> {
    if (axis == 0u) {
        return vec3<i32>(direction, 0, 0);
    }
    if (axis == 1u) {
        return vec3<i32>(0, direction, 0);
    }
    return vec3<i32>(0, 0, direction);
}

fn dominant_axis(value: vec3<f32>) -> u32 {
    let magnitude = abs(value);
    if (magnitude.x >= magnitude.y && magnitude.x >= magnitude.z) {
        return 0u;
    }
    if (magnitude.y >= magnitude.z) {
        return 1u;
    }
    return 2u;
}

fn voxel_coord_from_center(world_center: vec3<f32>) -> vec3<i32> {
    let size = voxel_size();
    return vec3<i32>(floor((world_center - OBJECT_BOUNDS_MIN) / size));
}

fn tangent_axis_0(face_axis: u32) -> u32 {
    if (face_axis == 0u) {
        return 1u;
    }
    return 0u;
}

fn tangent_axis_1(face_axis: u32) -> u32 {
    if (face_axis == 2u) {
        return 1u;
    }
    return 2u;
}

fn derive_face_info(
    world_center: vec3<f32>,
    hit_world_position: vec3<f32>,
    shading_normal: vec3<f32>,
) -> FaceInfo {
    let size = voxel_size();
    let center_delta = (hit_world_position - world_center) / size;
    let local_position = clamp(center_delta + vec3<f32>(0.5), vec3<f32>(0.0), vec3<f32>(1.0));
    let delta_axis = dominant_axis(center_delta);
    let normal_axis = dominant_axis(shading_normal);
    let delta_axis_distance = abs(axis_component(center_delta, delta_axis));
    let face_axis = select(delta_axis, normal_axis, delta_axis_distance < 1e-4);
    let axis_delta = axis_component(center_delta, face_axis);
    let normal_delta = axis_component(shading_normal, face_axis);
    let face_sign = select(-1.0, 1.0, select(axis_delta, normal_delta, abs(axis_delta) < 1e-4) >= 0.0);
    let tangent_0 = tangent_axis_0(face_axis);
    let tangent_1 = tangent_axis_1(face_axis);

    return FaceInfo(
        voxel_coord_from_center(world_center),
        local_position,
        axis_step(face_axis, i32(face_sign)),
        face_axis,
        face_sign,
        tangent_0,
        tangent_1,
        vec2<f32>(axis_component(local_position, tangent_0), axis_component(local_position, tangent_1)),
        axis_component(world_center, face_axis) + face_sign * size * 0.5,
    );
}

fn project_world_to_uv(world_position: vec3<f32>) -> vec3<f32> {
    let offset = world_position - camera.position.xyz;
    let view_x = dot(offset, camera.right.xyz);
    let view_y = dot(offset, camera.up.xyz);
    let view_z = dot(offset, camera.forward.xyz);
    if (view_z <= 1e-5) {
        return vec3<f32>(0.0);
    }

    let ndc = vec2<f32>(
        view_x / (view_z * max(camera.viewport.x, 1e-5)),
        -view_y / (view_z * max(camera.viewport.y, 1e-5)),
    );
    let uv = ndc * 0.5 + vec2<f32>(0.5, 0.5);
    let visible = select(0.0, 1.0, all(uv >= vec2<f32>(0.0)) && all(uv <= vec2<f32>(1.0)));
    return vec3<f32>(uv, visible);
}

fn sample_projected_face(
    expected_voxel_coord: vec3<i32>,
    expected_face_position: vec3<f32>,
    source_face_normal: vec3<i32>,
    source_face_axis: u32,
    source_face_plane: f32,
    current_color: vec3<f32>,
    dimensions: vec2<u32>,
) -> SampleResult {
    let projected = project_world_to_uv(expected_face_position);
    if (projected.z < 0.5) {
        return SampleResult(false, current_color, 1u);
    }

    let candidate_pixel = pixel_coord(projected.xy, dimensions);
    if (!pixel_in_bounds(candidate_pixel, dimensions)) {
        return SampleResult(false, current_color, 1u);
    }

    let sampled_center = textureLoad(world_position_texture, candidate_pixel, 0);
    if (!is_visible_voxel(sampled_center)) {
        return SampleResult(false, current_color, 1u);
    }

    let sampled_shading_input = textureLoad(shading_input_texture, candidate_pixel, 0);
    if (sampled_shading_input.w <= 0.0) {
        return SampleResult(false, current_color, 1u);
    }

    let candidate_uv = (vec2<f32>(candidate_pixel) + vec2<f32>(0.5)) / vec2<f32>(dimensions);
    let candidate_hit_world_position =
        camera.position.xyz + compute_camera_ray_direction(candidate_uv) * sampled_shading_input.w;
    let candidate_face =
        derive_face_info(sampled_center.xyz, candidate_hit_world_position, sampled_shading_input.xyz);
    let plane_tolerance = voxel_size() * FACE_PLANE_TOLERANCE_SCALE;
    let coord_matches = all(candidate_face.voxel_coord == expected_voxel_coord);
    let normal_matches = all(candidate_face.normal == source_face_normal);
    let coplanar =
        candidate_face.axis == source_face_axis
        && abs(candidate_face.plane - source_face_plane) <= plane_tolerance;

    if (!coord_matches || !normal_matches || !coplanar) {
        return SampleResult(false, current_color, 1u);
    }

    return SampleResult(true, textureLoad(current_texture, candidate_pixel, 0).xyz, 0u);
}

fn bilinear_weighted_sum(
    sample: SampleResult,
    weight: f32,
    accumulated_color: vec3<f32>,
    accumulated_weight: f32,
) -> vec4<f32> {
    if (!sample.valid || weight <= 0.0) {
        return vec4<f32>(accumulated_color, accumulated_weight);
    }
    return vec4<f32>(accumulated_color + sample.color * weight, accumulated_weight + weight);
}

fn face_axis_weight(offset: i32, uv: f32) -> f32 {
    if (offset < 0) {
        return 0.5 * (1.0 - uv) * (1.0 - uv);
    }
    if (offset > 0) {
        return 0.5 * uv * uv;
    }
    return 0.5 + uv * (1.0 - uv);
}

fn debug_face_axis_color(face_axis: u32, face_sign: f32) -> vec3<f32> {
    var color = vec3<f32>(0.0);
    if (face_axis == 0u) {
        color = vec3<f32>(1.0, 0.12, 0.08);
    } else if (face_axis == 1u) {
        color = vec3<f32>(0.10, 0.82, 0.20);
    } else {
        color = vec3<f32>(0.18, 0.36, 1.0);
    }
    return select(color * 0.45, color, face_sign > 0.0);
}

fn debug_quadrant_color(step_0: i32, step_1: i32) -> vec3<f32> {
    if (step_0 < 0 && step_1 < 0) {
        return vec3<f32>(0.95, 0.20, 0.25);
    }
    if (step_0 > 0 && step_1 < 0) {
        return vec3<f32>(0.98, 0.78, 0.15);
    }
    if (step_0 < 0 && step_1 > 0) {
        return vec3<f32>(0.10, 0.78, 0.88);
    }
    return vec3<f32>(0.30, 0.92, 0.42);
}

fn debug_count_color(count: u32, maximum: f32) -> vec3<f32> {
    let t = clamp(f32(count) / maximum, 0.0, 1.0);
    return mix(vec3<f32>(0.04, 0.04, 0.05), vec3<f32>(1.0, 0.25, 0.08), t);
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var positions = array(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(3.0, 1.0),
    );

    let position = positions[vertex_index];

    var output: VertexOutput;
    output.position = vec4<f32>(position, 0.0, 1.0);
    output.uv = position * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
    return output;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dimensions = textureDimensions(current_texture);
    let pixel = pixel_coord(in.uv, dimensions);
    let current_color = textureLoad(current_texture, pixel, 0).xyz;
    let current_center = textureLoad(world_position_texture, pixel, 0);
    let shading_input = textureLoad(shading_input_texture, pixel, 0);
    let hit_depth = shading_input.w;

    if (!is_visible_voxel(current_center) || hit_depth <= 0.0) {
        return vec4<f32>(current_color, 1.0);
    }

    let size = voxel_size();
    let hit_world_position = camera.position.xyz + compute_camera_ray_direction(in.uv) * hit_depth;
    let source_face = derive_face_info(current_center.xyz, hit_world_position, shading_input.xyz);

    var accumulated = vec4<f32>(0.0);
    var valid_neighbour_count = 0u;
    var rejected_sample_count = 0u;

    for (var tangent_y = -1; tangent_y <= 1; tangent_y = tangent_y + 1) {
        for (var tangent_x = -1; tangent_x <= 1; tangent_x = tangent_x + 1) {
            let face_offset =
                axis_step(source_face.tangent_axis_0, tangent_x)
                + axis_step(source_face.tangent_axis_1, tangent_y);
            let weight =
                face_axis_weight(tangent_x, source_face.face_uv.x)
                * face_axis_weight(tangent_y, source_face.face_uv.y);
            var sample = SampleResult(true, current_color, 0u);

            if (tangent_x != 0 || tangent_y != 0) {
                let expected_face_position = hit_world_position + vec3<f32>(face_offset) * size;
                sample = sample_projected_face(
                    source_face.voxel_coord + face_offset,
                    expected_face_position,
                    source_face.normal,
                    source_face.axis,
                    source_face.plane,
                    current_color,
                    dimensions,
                );
                valid_neighbour_count = valid_neighbour_count + select(0u, 1u, sample.valid);
                rejected_sample_count = rejected_sample_count + sample.rejected_samples;
            }

            accumulated = bilinear_weighted_sum(sample, weight, accumulated.xyz, accumulated.w);
        }
    }

    if (DEBUG_FACE_OUTPUT_MODE == DEBUG_FACE_OUTPUT_AXIS) {
        return vec4<f32>(debug_face_axis_color(source_face.axis, source_face.sign), 1.0);
    }
    if (DEBUG_FACE_OUTPUT_MODE == DEBUG_FACE_OUTPUT_QUADRANT) {
        return vec4<f32>(
            debug_quadrant_color(
                select(-1, 1, source_face.face_uv.x >= 0.5),
                select(-1, 1, source_face.face_uv.y >= 0.5),
            ),
            1.0,
        );
    }
    if (DEBUG_FACE_OUTPUT_MODE == DEBUG_FACE_OUTPUT_VALID_NEIGHBOURS) {
        return vec4<f32>(debug_count_color(valid_neighbour_count, 8.0), 1.0);
    }
    if (DEBUG_FACE_OUTPUT_MODE == DEBUG_FACE_OUTPUT_REJECTED_SAMPLES) {
        return vec4<f32>(debug_count_color(min(rejected_sample_count, 13u), 13.0), 1.0);
    }

    if (accumulated.w <= 1e-5) {
        return vec4<f32>(current_color, 1.0);
    }
    return vec4<f32>(accumulated.xyz / accumulated.w, 1.0);
}
