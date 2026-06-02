# Voxel Storage Plan

The goal of the renderer is to compress voxel data so we can render more world
space on screen without making the data model unreasonably complex.

This plan keeps the useful parts of the `ash-voxels` storage design, but
describes them in renderer-agnostic terms so they can be implemented cleanly in
this `wgpu-voxels` prototype.

## Working Assumptions

These simplifications keep the first version manageable:

* All voxels have a consistent size.
* All voxels are axis aligned.
* Voxels are stored in world chunks.
* The first version should optimize for static or infrequently rebuilt chunks.

## Current Repo Status

The repo already has the first half of this layout in `src/scene/voxel_mask.rs`:

* A chunk-local `64x64x64` voxel grid
* `8x8x8` regions
* A chunk region occupancy mask
* Per-region `8x8x8` leaf occupancy masks

That existing bitmask layout is a good staging point, even if we later replace
the per-region leaf masks with a tighter palette-driven payload.

## Chunks

* The world is chunked into `64x64x64` voxel volumes.
* Each chunk contains `512` regions arranged as `8x8x8`.
* Chunk coordinates are world-space chunk indices, not voxel-space positions.

### Chunk Header

Byte-aligned layout:

| Byte Size | Stored                    | Notes                                               |
|-----------|---------------------------|-----------------------------------------------------|
| 12        | World position            | `i32 x, y, z` chunk coordinates                     |
| 64        | Region occupancy bitfield | One bit per `8x8x8` region                          |
| 2,048     | 512 region headers        | One fixed-size header per region                    |

Packed size: `2,124` bytes per chunk before region payload blobs.

## Regions

* Each chunk is divided into `512` `8x8x8` palette regions.
* Each region covers `512` voxels.
* Each voxel stores a palette index instead of a full material value.
* The index bit width depends on the palette size for that region.

Examples:

* A region with `2` distinct values needs `1` bit per voxel.
* A region with `15` distinct values needs `4` bits per voxel.
* A region with `255` distinct values needs `8` bits per voxel.

Initial constraint:

* Cap palette size at `255` entries for the first implementation.
* Treat larger material variety as an overflow case to design later if needed.

### Region Header

Byte-aligned layout:

| Byte Size | Stored         | Notes                                                   |
|-----------|----------------|---------------------------------------------------------|
| 1         | Palette length | `0` means empty region, `1..255` are valid sizes        |
| 3         | Blob pointer   | Offset into a chunk-local or arena allocation of blobs  |

Notes:

* Empty regions do not allocate blob storage.
* The pointer can be an offset rather than a raw address.
* For `wgpu`, this should map naturally to offsets inside one or more storage
  buffers rather than backend-specific resource handles.

## Region Blob

Each non-empty region owns a variable-sized blob containing:

| Byte Size                                      | Stored           | Notes                                      |
|------------------------------------------------|------------------|--------------------------------------------|
| `2 * palette_size`                             | Palette swatches | Fixed `2` bytes per swatch in the first pass |
| `ceil(voxel_count * palette_index_bits / 8)`   | Palette indices  | Bit-packed indices for the `512` voxels    |

Where:

* `voxel_count = 512`
* `palette_index_bits = ceil(log2(palette_size))`, with a minimum of `1` for a
  non-empty region

This is the main compression win: sparse or low-variety regions stay very
small, while dense high-variety regions still avoid paying for a full dense
material array.

## Example Chunk Sizes

Assumptions for the table below:

* Chunk dimensions are `64x64x64` (`262,144` voxels).
* Regions are `8x8x8` (`512` voxels per region, `512` regions per chunk).
* Exactly `2` voxel colors are used, so palette indices cost `1` bit per
  occupied voxel in each populated region.
* Region headers are already included in the fixed chunk cost.
* Blob payloads are only allocated for non-empty regions.
* `2` colors means each populated region stores `4` bytes of swatches.
* A populated `8x8x8` region stores `64` bytes of packed indices.
* Total bytes = `2,124 + populated_regions * 68`.

| Chunk Fill | Occupied Voxels | Populated Regions | Total Bytes | Total KiB | Reduction vs dense `u16` array |
|------------|-----------------|-------------------|-------------|-----------|-------------------------------|
| 0%         | 0               | 0                 | 2,124       | 2.07      | 99.59%                        |
| 25%        | 65,536          | 128               | 10,828      | 10.57     | 97.93%                        |
| 50%        | 131,072         | 256               | 19,532      | 19.07     | 96.27%                        |
| 100%       | 262,144         | 512               | 36,940      | 36.07     | 92.95%                        |

For comparison, a dense `u16` material array would use
`262,144 * 2 = 524,288` bytes (`512 KiB`) per chunk.

## Palette Swatch

First-pass swatches can stay at `2` bytes each.

This is enough for a compact prototype material model without overcommitting to
full PBR data too early.

Suggested bit split:

| Bit Size | Stored     |
|----------|------------|
| 3        | Voxel type |
| 4        | Red        |
| 5        | Green      |
| 4        | Blue       |

Notes:

* `3` bits for voxel type gives `8` coarse material classes.
* RGB `4:5:4` is intentionally low precision, but works well for a stylized
  look.
* If we want smoother gradients later, we can add shader-side dithering instead
  of inflating storage immediately.

## Editing Strategy

For the prototype, keep edits simple:

* Large-scale gameplay edits can be applied on the CPU, then rebuild the
  affected chunk from source voxel data.
* Small procedural animation that does not change palette cardinality could be
  handled in GPU code later, but it should not be part of the first storage
  implementation.

This favors correctness and iteration speed over fully dynamic in-place updates.

## Implementation Notes For This Repo

The most practical migration path in `wgpu-voxels` looks like this:

1. Keep the existing chunk and region dimensions from `src/scene/voxel_mask.rs`.
2. Split the current monolithic occupancy representation into:
   * a compact chunk header buffer
   * a region-header buffer
   * a blob buffer for palette data and packed indices
3. Start with CPU-built chunks uploaded into storage buffers.
4. Update traversal code to:
   * test the chunk region mask first
   * fetch the region header
   * decode palette indices from the region blob
   * map the palette swatch to shading data
5. Only revisit streaming, paging, and defragmentation once the static format is
   working end to end.

One likely transitional step is to keep the current per-region leaf masks during
debugging, then replace them with packed palette indices once the data upload and
shader decode path is stable.

## Outstanding Design Work

* Solid chunk fast-path handling
* Blob allocation strategy
* Defragmentation and recycling of freed blob space
* Chunk streaming and paging
* Material and voxel-type catalog
* Edit pipeline for persistent world changes
* Whether leaf occupancy should be explicit or inferred from palette indices
