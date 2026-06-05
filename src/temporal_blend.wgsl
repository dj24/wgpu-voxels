@group(0) @binding(0)
var current_texture: texture_2d<f32>;

@group(0) @binding(1)
var history_texture: texture_2d<f32>;

@group(0) @binding(2)
var history_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
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
    let current = textureSample(current_texture, history_sampler, in.uv);
    let history = textureSample(history_texture, history_sampler, in.uv);
    return history * 0.95 + current * 0.05;
}
