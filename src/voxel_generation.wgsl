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

const VOXEL_GRID_DIM_U32: u32 = 64u;
const REGION_AXIS_U32: u32 = 8u;
const COARSE_REGION_AXIS_U32: u32 = 4u;
const MASK_WORD_BITS_U32: u32 = 32u;
const MASK_WORD_COUNT_U32: u32 = 16u;
const COARSE_MASK_WORD_OFFSET_U32: u32 = 16u;
const LEAF_MASK_WORD_OFFSET_U32: u32 = 18u;
const OCCUPANCY_WORD_COUNT_U32: u32 = 8210u;
const OBJECT_BOUNDS_MIN: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);

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

fn world_position(chunk_origin: vec3<f32>, voxel: vec3<u32>) -> vec3<f32> {
    let size = voxel_size();
    return chunk_origin + OBJECT_BOUNDS_MIN + (vec3<f32>(voxel) + vec3<f32>(0.5)) * size;
}

fn taylor_inv_sqrt4(r: vec4<f32>) -> vec4<f32> {
    return 1.79284291400159 - 0.85373472095314 * r;
}

fn perlin_noise3_permute4(x: vec4<f32>) -> vec4<f32> {
    return ((x * 34.0 + 1.0) * x) % vec4<f32>(289.0);
}

fn perlin_noise3_fade3(t: vec3<f32>) -> vec3<f32> {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn perlin_noise3(position: vec3<f32>) -> f32 {
    var cell0 = floor(position);
    var cell1 = cell0 + vec3<f32>(1.0);
    cell0 = cell0 % vec3<f32>(289.0);
    cell1 = cell1 % vec3<f32>(289.0);

    let local0 = fract(position);
    let local1 = local0 - vec3<f32>(1.0);

    let ix = vec4<f32>(cell0.x, cell1.x, cell0.x, cell1.x);
    let iy = vec4<f32>(cell0.yy, cell1.yy);
    let iz0 = cell0.zzzz;
    let iz1 = cell1.zzzz;

    let ixy = perlin_noise3_permute4(perlin_noise3_permute4(ix) + iy);
    let ixy0 = perlin_noise3_permute4(ixy + iz0);
    let ixy1 = perlin_noise3_permute4(ixy + iz1);

    var gx0 = ixy0 / 7.0;
    var gy0 = fract(floor(gx0) / 7.0) - 0.5;
    gx0 = fract(gx0);
    var gz0 = vec4<f32>(0.5) - abs(gx0) - abs(gy0);
    let sz0 = step(gz0, vec4<f32>(0.0));
    gx0 = gx0 + sz0 * (step(vec4<f32>(0.0), gx0) - 0.5);
    gy0 = gy0 + sz0 * (step(vec4<f32>(0.0), gy0) - 0.5);

    var gx1 = ixy1 / 7.0;
    var gy1 = fract(floor(gx1) / 7.0) - 0.5;
    gx1 = fract(gx1);
    var gz1 = vec4<f32>(0.5) - abs(gx1) - abs(gy1);
    let sz1 = step(gz1, vec4<f32>(0.0));
    gx1 = gx1 - sz1 * (step(vec4<f32>(0.0), gx1) - 0.5);
    gy1 = gy1 - sz1 * (step(vec4<f32>(0.0), gy1) - 0.5);

    var g000 = vec3<f32>(gx0.x, gy0.x, gz0.x);
    var g100 = vec3<f32>(gx0.y, gy0.y, gz0.y);
    var g010 = vec3<f32>(gx0.z, gy0.z, gz0.z);
    var g110 = vec3<f32>(gx0.w, gy0.w, gz0.w);
    var g001 = vec3<f32>(gx1.x, gy1.x, gz1.x);
    var g101 = vec3<f32>(gx1.y, gy1.y, gz1.y);
    var g011 = vec3<f32>(gx1.z, gy1.z, gz1.z);
    var g111 = vec3<f32>(gx1.w, gy1.w, gz1.w);

    let norm0 = taylor_inv_sqrt4(vec4<f32>(
        dot(g000, g000),
        dot(g010, g010),
        dot(g100, g100),
        dot(g110, g110),
    ));
    g000 *= norm0.x;
    g010 *= norm0.y;
    g100 *= norm0.z;
    g110 *= norm0.w;

    let norm1 = taylor_inv_sqrt4(vec4<f32>(
        dot(g001, g001),
        dot(g011, g011),
        dot(g101, g101),
        dot(g111, g111),
    ));
    g001 *= norm1.x;
    g011 *= norm1.y;
    g101 *= norm1.z;
    g111 *= norm1.w;

    let n000 = dot(g000, local0);
    let n100 = dot(g100, vec3<f32>(local1.x, local0.yz));
    let n010 = dot(g010, vec3<f32>(local0.x, local1.y, local0.z));
    let n110 = dot(g110, vec3<f32>(local1.xy, local0.z));
    let n001 = dot(g001, vec3<f32>(local0.xy, local1.z));
    let n101 = dot(g101, vec3<f32>(local1.x, local0.y, local1.z));
    let n011 = dot(g011, vec3<f32>(local0.x, local1.yz));
    let n111 = dot(g111, local1);

    let fade_xyz = perlin_noise3_fade3(local0);
    let n_z = mix(
        vec4<f32>(n000, n100, n010, n110),
        vec4<f32>(n001, n101, n011, n111),
        vec4<f32>(fade_xyz.z),
    );
    let n_yz = mix(n_z.xy, n_z.zw, vec2<f32>(fade_xyz.y));
    let n_xyz = mix(n_yz.x, n_yz.y, fade_xyz.x);
    return 2.2 * n_xyz;
}

fn fbm_noise3(position: vec3<f32>) -> f32 {
    var amplitude = 0.5;
    var frequency = 1.0;
    var value = 0.0;

    for (var octave = 0; octave < 4; octave++) {
        value += perlin_noise3(position * frequency) * amplitude;
        frequency *= 2.0;
        amplitude *= 0.5;
    }

    return value;
}

fn chunk_density_filled(chunk_origin: vec3<f32>, voxel: vec3<u32>) -> bool {
    let position = world_position(chunk_origin, voxel);

    return perlin_noise3(position) > 0.0;
}

@compute @workgroup_size(256, 1, 1)
fn clear_occupancy_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let total_word_count = generation_params.active_object_count * OCCUPANCY_WORD_COUNT_U32;
    if (id.x >= total_word_count) {
        return;
    }

    atomicStore(&voxel_occupancy[id.x], 0u);
}

@compute @workgroup_size(8, 8, 2)
fn populate_chunk_noise_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let total_object_slices = generation_params.active_object_count * VOXEL_GRID_DIM_U32;
    if (id.x >= VOXEL_GRID_DIM_U32 || id.y >= VOXEL_GRID_DIM_U32 || id.z >= total_object_slices) {
        return;
    }

    let object_slot = id.z / VOXEL_GRID_DIM_U32;
    let voxel_z = id.z % VOXEL_GRID_DIM_U32;
    let object = chunk_objects[object_slot];
    let voxel = vec3<u32>(id.x, id.y, voxel_z);

    if (!chunk_density_filled(object.chunk_origin.xyz, voxel)) {
        return;
    }

    let object_index = object.object_index;
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
