pub(crate) const VOXEL_GRID_DIM: u32 = 64;
pub(crate) const VOXEL_GRID_DIM_I32: i32 = VOXEL_GRID_DIM as i32;
pub(crate) const REGION_AXIS: u32 = 8;
pub(crate) const REGION_COUNT: usize =
    (REGION_AXIS as usize) * (REGION_AXIS as usize) * (REGION_AXIS as usize);
pub(crate) const COARSE_REGION_AXIS: u32 = 4;
pub(crate) const COARSE_REGION_COUNT: usize =
    (COARSE_REGION_AXIS as usize) * (COARSE_REGION_AXIS as usize) * (COARSE_REGION_AXIS as usize);
pub(crate) const MASK_WORD_BITS: usize = u32::BITS as usize;
pub(crate) const MASK_WORD_COUNT: usize = REGION_COUNT.div_ceil(MASK_WORD_BITS);
pub(crate) const COARSE_MASK_WORD_COUNT: usize = COARSE_REGION_COUNT.div_ceil(MASK_WORD_BITS);
pub(crate) const COARSE_MASK_WORD_OFFSET: usize = MASK_WORD_COUNT;
pub(crate) const LEAF_MASK_WORD_OFFSET: usize = COARSE_MASK_WORD_OFFSET + COARSE_MASK_WORD_COUNT;
pub(crate) const OCCUPANCY_WORD_COUNT: usize =
    LEAF_MASK_WORD_OFFSET + REGION_COUNT * MASK_WORD_COUNT;

pub(crate) fn build_sphere_voxel_mask(
    bounds_min: [f32; 3],
    bounds_max: [f32; 3],
    radius: f32,
) -> Vec<u32> {
    let mut words = vec![0u32; OCCUPANCY_WORD_COUNT];
    let object_extent = bounds_max[0] - bounds_min[0];
    let voxel_size = object_extent / VOXEL_GRID_DIM as f32;
    let radius_sq = radius * radius;

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

                set_occupancy_bit(&mut words, [x as u32, y as u32, z as u32]);
            }
        }
    }

    build_coarse_mask_from_regions(&mut words);
    words
}

fn flatten_region_index(region_position: [u32; 3]) -> usize {
    debug_assert!(region_position[0] < REGION_AXIS);
    debug_assert!(region_position[1] < REGION_AXIS);
    debug_assert!(region_position[2] < REGION_AXIS);

    region_position[0] as usize
        + REGION_AXIS as usize
            * (region_position[1] as usize + REGION_AXIS as usize * region_position[2] as usize)
}

fn flatten_coarse_index(region_position: [u32; 3]) -> usize {
    debug_assert!(region_position[0] < COARSE_REGION_AXIS);
    debug_assert!(region_position[1] < COARSE_REGION_AXIS);
    debug_assert!(region_position[2] < COARSE_REGION_AXIS);

    region_position[0] as usize
        + COARSE_REGION_AXIS as usize
            * (region_position[1] as usize
                + COARSE_REGION_AXIS as usize * region_position[2] as usize)
}

fn flatten_leaf_index(local_position: [u32; 3]) -> usize {
    debug_assert!(local_position[0] < REGION_AXIS);
    debug_assert!(local_position[1] < REGION_AXIS);
    debug_assert!(local_position[2] < REGION_AXIS);

    local_position[0] as usize
        + REGION_AXIS as usize
            * (local_position[1] as usize + REGION_AXIS as usize * local_position[2] as usize)
}

fn occupancy_word_and_mask(index: usize) -> (usize, u32) {
    let word_index = index / MASK_WORD_BITS;
    let bit_index = index % MASK_WORD_BITS;
    (word_index, 1u32 << bit_index)
}

fn region_leaf_word_offset(region_index: usize) -> usize {
    debug_assert!(region_index < REGION_COUNT);
    LEAF_MASK_WORD_OFFSET + region_index * MASK_WORD_COUNT
}

fn region_mask_bit_is_set(occupancy: &[u32], region_index: usize) -> bool {
    let (word_index, bit_mask) = occupancy_word_and_mask(region_index);
    occupancy[word_index] & bit_mask != 0
}

#[cfg(test)]
fn coarse_mask_bit_is_set(occupancy: &[u32], coarse_index: usize) -> bool {
    let (word_index, bit_mask) = occupancy_word_and_mask(coarse_index);
    occupancy[COARSE_MASK_WORD_OFFSET + word_index] & bit_mask != 0
}

#[cfg(test)]
fn occupancy_bit_is_set(occupancy: &[u32], position: [u32; 3]) -> bool {
    debug_assert!(position[0] < VOXEL_GRID_DIM);
    debug_assert!(position[1] < VOXEL_GRID_DIM);
    debug_assert!(position[2] < VOXEL_GRID_DIM);

    let region_position = [position[0] / 8, position[1] / 8, position[2] / 8];
    let region_index = flatten_region_index(region_position);
    if !region_mask_bit_is_set(occupancy, region_index) {
        return false;
    }

    let leaf_local = [position[0] & 7, position[1] & 7, position[2] & 7];
    let leaf_index = flatten_leaf_index(leaf_local);
    let (word_index, bit_mask) = occupancy_word_and_mask(leaf_index);
    occupancy[region_leaf_word_offset(region_index) + word_index] & bit_mask != 0
}

fn set_occupancy_bit(occupancy: &mut [u32], position: [u32; 3]) {
    debug_assert!(position[0] < VOXEL_GRID_DIM);
    debug_assert!(position[1] < VOXEL_GRID_DIM);
    debug_assert!(position[2] < VOXEL_GRID_DIM);

    let region_position = [position[0] / 8, position[1] / 8, position[2] / 8];
    let region_index = flatten_region_index(region_position);
    let (region_word_index, region_bit_mask) = occupancy_word_and_mask(region_index);
    occupancy[region_word_index] |= region_bit_mask;

    let leaf_local = [position[0] & 7, position[1] & 7, position[2] & 7];
    let leaf_index = flatten_leaf_index(leaf_local);
    let (leaf_word_index, leaf_bit_mask) = occupancy_word_and_mask(leaf_index);
    occupancy[region_leaf_word_offset(region_index) + leaf_word_index] |= leaf_bit_mask;
}

fn build_coarse_mask_from_regions(occupancy: &mut [u32]) {
    for coarse_z in 0..COARSE_REGION_AXIS {
        for coarse_y in 0..COARSE_REGION_AXIS {
            for coarse_x in 0..COARSE_REGION_AXIS {
                let coarse_index = flatten_coarse_index([coarse_x, coarse_y, coarse_z]);
                let has_occupied_region = (0..2).any(|dz| {
                    (0..2).any(|dy| {
                        (0..2).any(|dx| {
                            let region_index = flatten_region_index([
                                coarse_x * 2 + dx,
                                coarse_y * 2 + dy,
                                coarse_z * 2 + dz,
                            ]);
                            region_mask_bit_is_set(occupancy, region_index)
                        })
                    })
                });

                if has_occupied_region {
                    let (word_index, bit_mask) = occupancy_word_and_mask(coarse_index);
                    occupancy[COARSE_MASK_WORD_OFFSET + word_index] |= bit_mask;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        COARSE_MASK_WORD_COUNT, COARSE_MASK_WORD_OFFSET, COARSE_REGION_AXIS, COARSE_REGION_COUNT,
        LEAF_MASK_WORD_OFFSET, MASK_WORD_COUNT, OCCUPANCY_WORD_COUNT, REGION_AXIS, REGION_COUNT,
        VOXEL_GRID_DIM, build_coarse_mask_from_regions, build_sphere_voxel_mask,
        coarse_mask_bit_is_set, flatten_coarse_index, flatten_leaf_index, flatten_region_index,
        occupancy_bit_is_set, region_leaf_word_offset, region_mask_bit_is_set, set_occupancy_bit,
    };

    #[test]
    fn occupancy_layout_matches_ash_voxels() {
        assert_eq!(VOXEL_GRID_DIM, 64);
        assert_eq!(REGION_AXIS, 8);
        assert_eq!(REGION_COUNT, 512);
        assert_eq!(COARSE_REGION_AXIS, 4);
        assert_eq!(COARSE_REGION_COUNT, 64);
        assert_eq!(MASK_WORD_COUNT, 16);
        assert_eq!(COARSE_MASK_WORD_COUNT, 2);
        assert_eq!(COARSE_MASK_WORD_OFFSET, 16);
        assert_eq!(LEAF_MASK_WORD_OFFSET, 18);
        assert_eq!(OCCUPANCY_WORD_COUNT, 8_210);
        assert_eq!(OCCUPANCY_WORD_COUNT * core::mem::size_of::<u32>(), 32_840);
    }

    #[test]
    fn setting_one_voxel_marks_one_region_and_one_leaf_bit() {
        let mut occupancy = vec![0u32; OCCUPANCY_WORD_COUNT];
        let voxel = [9, 2, 17];

        set_occupancy_bit(&mut occupancy, voxel);
        build_coarse_mask_from_regions(&mut occupancy);

        let region_index = flatten_region_index([1, 0, 2]);
        assert!(region_mask_bit_is_set(&occupancy, region_index));
        assert!(coarse_mask_bit_is_set(
            &occupancy,
            flatten_coarse_index([0, 0, 1])
        ));
        assert_eq!(
            occupancy[..MASK_WORD_COUNT]
                .iter()
                .filter(|word| **word != 0)
                .count(),
            1
        );
        assert!(occupancy_bit_is_set(&occupancy, voxel));
        assert_eq!(
            occupancy[region_leaf_word_offset(region_index)
                ..region_leaf_word_offset(region_index) + MASK_WORD_COUNT]
                .iter()
                .filter(|word| **word != 0)
                .count(),
            1
        );
    }

    #[test]
    fn voxels_in_same_region_share_region_mask_but_use_distinct_leaf_bits() {
        let mut occupancy = vec![0u32; OCCUPANCY_WORD_COUNT];
        let first = [8, 8, 8];
        let second = [15, 15, 15];

        set_occupancy_bit(&mut occupancy, first);
        set_occupancy_bit(&mut occupancy, second);
        build_coarse_mask_from_regions(&mut occupancy);

        let region_index = flatten_region_index([1, 1, 1]);
        assert!(region_mask_bit_is_set(&occupancy, region_index));
        assert!(coarse_mask_bit_is_set(
            &occupancy,
            flatten_coarse_index([0, 0, 0])
        ));
        assert!(occupancy_bit_is_set(&occupancy, first));
        assert!(occupancy_bit_is_set(&occupancy, second));
        assert_eq!(
            occupancy[..MASK_WORD_COUNT]
                .iter()
                .filter(|word| **word != 0)
                .count(),
            1
        );
        assert_ne!(
            flatten_leaf_index([first[0] & 7, first[1] & 7, first[2] & 7]),
            flatten_leaf_index([second[0] & 7, second[1] & 7, second[2] & 7]),
        );
    }

    #[test]
    fn voxels_in_different_regions_use_distinct_region_and_leaf_offsets() {
        let mut occupancy = vec![0u32; OCCUPANCY_WORD_COUNT];
        let first = [0, 0, 0];
        let second = [63, 63, 63];

        set_occupancy_bit(&mut occupancy, first);
        set_occupancy_bit(&mut occupancy, second);
        build_coarse_mask_from_regions(&mut occupancy);

        let first_region = flatten_region_index([0, 0, 0]);
        let second_region = flatten_region_index([7, 7, 7]);
        assert_ne!(first_region, second_region);
        assert!(coarse_mask_bit_is_set(
            &occupancy,
            flatten_coarse_index([0, 0, 0])
        ));
        assert!(coarse_mask_bit_is_set(
            &occupancy,
            flatten_coarse_index([3, 3, 3])
        ));
        assert_ne!(
            region_leaf_word_offset(first_region),
            region_leaf_word_offset(second_region)
        );
        assert!(occupancy_bit_is_set(&occupancy, first));
        assert!(occupancy_bit_is_set(&occupancy, second));
        assert_eq!(
            occupancy[..MASK_WORD_COUNT]
                .iter()
                .filter(|word| **word != 0)
                .count(),
            2
        );
    }

    #[test]
    fn sphere_mask_sets_region_and_leaf_bits() {
        let occupancy = build_sphere_voxel_mask([-0.75, -0.75, -0.75], [0.75, 0.75, 0.75], 0.55);

        assert_eq!(occupancy.len(), OCCUPANCY_WORD_COUNT);
        assert!(occupancy[..MASK_WORD_COUNT].iter().any(|word| *word != 0));
        assert!(
            occupancy[COARSE_MASK_WORD_OFFSET..LEAF_MASK_WORD_OFFSET]
                .iter()
                .any(|word| *word != 0)
        );
        assert!(
            occupancy[LEAF_MASK_WORD_OFFSET..]
                .iter()
                .any(|word| *word != 0)
        );
        assert!(occupancy_bit_is_set(&occupancy, [32, 32, 32]));
    }

    #[test]
    fn coarse_mask_is_derived_from_region_mask_after_population() {
        let mut occupancy = vec![0u32; OCCUPANCY_WORD_COUNT];

        set_occupancy_bit(&mut occupancy, [0, 0, 0]);
        set_occupancy_bit(&mut occupancy, [16, 16, 16]);
        build_coarse_mask_from_regions(&mut occupancy);

        assert!(coarse_mask_bit_is_set(
            &occupancy,
            flatten_coarse_index([0, 0, 0])
        ));
        assert!(coarse_mask_bit_is_set(
            &occupancy,
            flatten_coarse_index([1, 1, 1])
        ));
        assert!(!coarse_mask_bit_is_set(
            &occupancy,
            flatten_coarse_index([3, 3, 3]),
        ));
    }

    #[test]
    fn larger_radius_fills_more_voxels() {
        let small = build_sphere_voxel_mask([-0.75, -0.75, -0.75], [0.75, 0.75, 0.75], 0.25);
        let large = build_sphere_voxel_mask([-0.75, -0.75, -0.75], [0.75, 0.75, 0.75], 0.70);

        let small_bits = small.iter().map(|word| word.count_ones()).sum::<u32>();
        let large_bits = large.iter().map(|word| word.count_ones()).sum::<u32>();

        assert!(large_bits > small_bits);
        assert!(occupancy_bit_is_set(&small, [32, 32, 32]));
        assert!(occupancy_bit_is_set(&large, [32, 32, 32]));
    }
}
