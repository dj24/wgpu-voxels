pub(crate) const VOXEL_GRID_DIM: u32 = 64;
pub(crate) const VOXEL_GRID_DIM_I32: i32 = VOXEL_GRID_DIM as i32;
pub(crate) const VOXEL_GRID_VOXEL_COUNT: usize =
    (VOXEL_GRID_DIM as usize) * (VOXEL_GRID_DIM as usize) * (VOXEL_GRID_DIM as usize);
pub(crate) const VOXEL_MASK_WORD_COUNT: usize = VOXEL_GRID_VOXEL_COUNT.div_ceil(32);

pub(crate) fn build_sphere_voxel_mask(bounds_min: [f32; 3], bounds_max: [f32; 3]) -> Vec<u32> {
    let mut words = vec![0u32; VOXEL_MASK_WORD_COUNT];
    let object_extent = bounds_max[0] - bounds_min[0];
    let voxel_size = object_extent / VOXEL_GRID_DIM as f32;
    let radius_sq = 0.55f32 * 0.55f32;

    for z in 0..VOXEL_GRID_DIM_I32 {
        for y in 0..VOXEL_GRID_DIM_I32 {
            for x in 0..VOXEL_GRID_DIM_I32 {
                let center = [
                    bounds_min[0] + (x as f32 + 0.5) * voxel_size,
                    bounds_min[1] + (y as f32 + 0.5) * voxel_size,
                    bounds_min[2] + (z as f32 + 0.5) * voxel_size,
                ];
                let distance_sq =
                    center[0] * center[0] + center[1] * center[1] + center[2] * center[2];

                if distance_sq > radius_sq {
                    continue;
                }

                let voxel_index = x as usize
                    + y as usize * VOXEL_GRID_DIM as usize
                    + z as usize * VOXEL_GRID_DIM as usize * VOXEL_GRID_DIM as usize;
                let word_index = voxel_index / 32;
                let bit_index = voxel_index % 32;
                words[word_index] |= 1u32 << bit_index;
            }
        }
    }

    words
}

#[cfg(test)]
mod tests {
    use super::{VOXEL_GRID_VOXEL_COUNT, VOXEL_MASK_WORD_COUNT, build_sphere_voxel_mask};

    #[test]
    fn voxel_mask_uses_one_bit_per_voxel() {
        assert_eq!(VOXEL_GRID_VOXEL_COUNT, 64 * 64 * 64);
        assert_eq!(
            VOXEL_MASK_WORD_COUNT * core::mem::size_of::<u32>(),
            32 * 1024
        );
    }

    #[test]
    fn sphere_mask_sets_some_voxels() {
        let words = build_sphere_voxel_mask([-0.75, -0.75, -0.75], [0.75, 0.75, 0.75]);
        assert!(words.iter().any(|word| *word != 0));
    }
}
