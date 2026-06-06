pub(crate) const VOXEL_GRID_DIM: u32 = 64;
pub(crate) const LEAF_VOXEL_WORD_COUNT: usize =
    (VOXEL_GRID_DIM as usize) * (VOXEL_GRID_DIM as usize) * (VOXEL_GRID_DIM as usize);
#[cfg(test)]
pub(crate) const REGION_AXIS: u32 = 8;
#[cfg(test)]
pub(crate) const REGION_COUNT: usize =
    (REGION_AXIS as usize) * (REGION_AXIS as usize) * (REGION_AXIS as usize);
#[cfg(test)]
pub(crate) const COARSE_REGION_AXIS: u32 = 4;
#[cfg(test)]
pub(crate) const COARSE_REGION_COUNT: usize =
    (COARSE_REGION_AXIS as usize) * (COARSE_REGION_AXIS as usize) * (COARSE_REGION_AXIS as usize);
#[cfg(test)]
pub(crate) const MASK_WORD_BITS: usize = u32::BITS as usize;
#[cfg(test)]
pub(crate) const MASK_WORD_COUNT: usize = REGION_COUNT.div_ceil(MASK_WORD_BITS);
#[cfg(test)]
pub(crate) const COARSE_MASK_WORD_COUNT: usize = COARSE_REGION_COUNT.div_ceil(MASK_WORD_BITS);
#[cfg(test)]
pub(crate) const COARSE_MASK_WORD_OFFSET: usize = MASK_WORD_COUNT;
#[cfg(test)]
pub(crate) const LEAF_MASK_WORD_OFFSET: usize = COARSE_MASK_WORD_OFFSET + COARSE_MASK_WORD_COUNT;
pub(crate) const OCCUPANCY_WORD_COUNT: usize = 8_210;

#[cfg(test)]
fn flatten_region_index(region_position: [u32; 3]) -> usize {
    debug_assert!(region_position[0] < REGION_AXIS);
    debug_assert!(region_position[1] < REGION_AXIS);
    debug_assert!(region_position[2] < REGION_AXIS);

    region_position[0] as usize
        + REGION_AXIS as usize
            * (region_position[1] as usize + REGION_AXIS as usize * region_position[2] as usize)
}

#[cfg(test)]
fn flatten_coarse_index(region_position: [u32; 3]) -> usize {
    debug_assert!(region_position[0] < COARSE_REGION_AXIS);
    debug_assert!(region_position[1] < COARSE_REGION_AXIS);
    debug_assert!(region_position[2] < COARSE_REGION_AXIS);

    region_position[0] as usize
        + COARSE_REGION_AXIS as usize
            * (region_position[1] as usize
                + COARSE_REGION_AXIS as usize * region_position[2] as usize)
}

#[cfg(test)]
fn flatten_leaf_index(local_position: [u32; 3]) -> usize {
    debug_assert!(local_position[0] < REGION_AXIS);
    debug_assert!(local_position[1] < REGION_AXIS);
    debug_assert!(local_position[2] < REGION_AXIS);

    local_position[0] as usize
        + REGION_AXIS as usize
            * (local_position[1] as usize + REGION_AXIS as usize * local_position[2] as usize)
}

#[cfg(test)]
fn occupancy_word_and_mask(index: usize) -> (usize, u32) {
    let word_index = index / MASK_WORD_BITS;
    let bit_index = index % MASK_WORD_BITS;
    (word_index, 1u32 << bit_index)
}

#[cfg(test)]
fn region_leaf_word_offset(region_index: usize) -> usize {
    debug_assert!(region_index < REGION_COUNT);
    LEAF_MASK_WORD_OFFSET + region_index * MASK_WORD_COUNT
}

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
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
        VOXEL_GRID_DIM, build_coarse_mask_from_regions, coarse_mask_bit_is_set,
        flatten_coarse_index, flatten_leaf_index, flatten_region_index, occupancy_bit_is_set,
        occupancy_word_and_mask, region_leaf_word_offset, region_mask_bit_is_set,
        set_occupancy_bit,
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
    fn occupancy_word_and_mask_maps_indices() {
        assert_eq!(occupancy_word_and_mask(0), (0, 1));
        assert_eq!(occupancy_word_and_mask(31), (0, 1u32 << 31));
        assert_eq!(occupancy_word_and_mask(32), (1, 1));
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
}
