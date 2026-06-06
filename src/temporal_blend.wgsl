@group(0) @binding(0)
var current_texture: texture_2d<f32>;

@group(0) @binding(1)
var history_texture: texture_2d<f32>;

@group(0) @binding(2)
var motion_vector_texture: texture_2d<f32>;

@group(0) @binding(3)
var current_world_position_texture: texture_2d<f32>;

@group(0) @binding(4)
var history_world_position_texture: texture_2d<f32>;

struct Camera {
    position: vec4<f32>,
    forward: vec4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
    viewport: vec4<f32>,
}

@group(0) @binding(5)
var<uniform> previous_camera: Camera;

@group(0) @binding(6)
var current_sampler: sampler;

@group(0) @binding(7)
var motion_sampler: sampler;

@group(0) @binding(8)
var history_sampler: sampler;

const TEMPORAL_HISTORY_WEIGHT: f32 = 0.95;
const TEMPORAL_CURRENT_WEIGHT: f32 = 0.05;
const DEPTH_REJECTION_BIAS: f32 = 0.02;
const DEPTH_REJECTION_SCALE: f32 = 0.01;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

fn pixel_coord(uv: vec2<f32>, dimensions: vec2<u32>) -> vec2<i32> {
    let max_coord = vec2<i32>(dimensions) - vec2<i32>(1);
    return clamp(vec2<i32>(uv * vec2<f32>(dimensions)), vec2<i32>(0), max_coord);
}

fn previous_view_depth(world_position: vec3<f32>) -> f32 {
    return dot(world_position - previous_camera.position.xyz, previous_camera.forward.xyz);
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
    let current = textureSample(current_texture, current_sampler, in.uv);
    let motion_vector = textureSample(motion_vector_texture, motion_sampler, in.uv);
    if (motion_vector.z < 0.5) {
        return current;
    }

    let history_uv = in.uv - motion_vector.xy;
    if (any(history_uv < vec2<f32>(0.0)) || any(history_uv > vec2<f32>(1.0))) {
        return current;
    }

    let current_world_position = textureLoad(
        current_world_position_texture,
        pixel_coord(in.uv, textureDimensions(current_world_position_texture)),
        0,
    );
    if (current_world_position.w < 0.5) {
        return current;
    }

    let history_world_position = textureLoad(
        history_world_position_texture,
        pixel_coord(history_uv, textureDimensions(history_world_position_texture)),
        0,
    );
    if (history_world_position.w < 0.5) {
        return current;
    }

    let current_history_depth = previous_view_depth(current_world_position.xyz);
    let sampled_history_depth = previous_view_depth(history_world_position.xyz);
    if (current_history_depth <= 0.0 || sampled_history_depth <= 0.0) {
        return current;
    }

    let depth_delta = abs(current_history_depth - sampled_history_depth);
    let depth_tolerance =
        max(DEPTH_REJECTION_BIAS, max(current_history_depth, sampled_history_depth) * DEPTH_REJECTION_SCALE);
    if (depth_delta > depth_tolerance) {
        return current;
    }

    let history = textureSample(history_texture, history_sampler, history_uv);
    return history * TEMPORAL_HISTORY_WEIGHT + current * TEMPORAL_CURRENT_WEIGHT;
}
