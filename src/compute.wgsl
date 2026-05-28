@group(0) @binding(0)
var output_texture: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8, 1)
fn compute_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dimensions = textureDimensions(output_texture);
    if (id.x >= dimensions.x || id.y >= dimensions.y) {
        return;
    }

    let uv = vec2<f32>(id.xy) / vec2<f32>(dimensions);
    textureStore(output_texture, vec2<i32>(id.xy), vec4<f32>(uv, 0.0, 1.0));
}
