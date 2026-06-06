# Voxel Storage Plan

The goal of the renderer is to store enough shading data per voxel that the GPU
can shade directly from voxel memory, while still skipping empty space at the
chunk and region level.

This version removes the palette idea entirely. Instead, every stored voxel owns
one fixed `u32` payload that packs material id, normal, and color together.

## Working Assumptions

These simplifications keep the first version manageable:

* All voxels have a consistent size.
* All voxels are axis aligned.
* Voxels are stored in world chunks.
* The first version should optimize for static or infrequently rebuilt chunks.
* The main storage win comes from eliding empty regions, not from per-region
  color deduplication.

## Current Repo Status

The repo already has the first half of this layout in `src/scene/voxel_mask.rs`:

* A chunk-local `64x64x64` voxel grid
* `8x8x8` regions
* A chunk region occupancy mask
* Per-region `8x8x8` leaf occupancy masks

That existing bitmask layout is a good staging point, even if we later replace
the per-region leaf masks with fixed packed-voxel payloads.

## Chunks

* The world is chunked into `64x64x64` voxel volumes.
* Each chunk contains `512` regions arranged as `8x8x8`.
* Chunk coordinates are world-space chunk indices, not voxel-space positions.

### Chunk Header

Byte-aligned layout:

| Byte Size | Stored                    | Notes                           |
|-----------|---------------------------|---------------------------------|
| 12        | World position            | `i32 x, y, z` chunk coordinates |
| 64        | Region occupancy bitfield | One bit per `8x8x8` region      |
| 2,048     | 512 region headers        | One fixed-size header per region |

Packed size: `2,124` bytes per chunk before region payload blobs.

## Regions

* Each chunk is divided into `512` `8x8x8` regions.
* Each region covers `512` voxels.
* Each stored voxel uses one fixed `32`-bit packed value.
* Empty regions do not allocate voxel payload storage.
* Non-empty regions allocate a dense `512`-voxel payload.

This is simpler than the palette design. It gives up low-variety compression,
but it makes decode trivial and keeps every voxel self-contained.

### Region Header

Byte-aligned layout:

| Byte Size | Stored       | Notes                                                  |
|-----------|--------------|--------------------------------------------------------|
| 1         | Region state | `0` empty, `1` packed voxel payload present            |
| 3         | Blob pointer | Offset into a chunk-local or arena allocation of blobs |

Notes:

* Empty regions do not allocate blob storage.
* The pointer can be an offset rather than a raw address.
* For `wgpu`, this maps naturally to offsets inside one or more storage
  buffers rather than backend-specific resource handles.

## Region Blob

Each non-empty region owns one fixed-size blob:

| Byte Size | Stored        | Notes                                |
|-----------|---------------|--------------------------------------|
| `2,048`   | Packed voxels | `512` voxels * `4` bytes per voxel   |

Where:

* `voxel_count = 512`
* `bytes_per_voxel = 4`

The region payload is always dense once a region is present. Compression now
comes from skipping empty regions entirely rather than shrinking individual
region payloads.

## Packed Voxel Word

Each stored voxel is one `u32` with this bit layout:

| Bit Range | Bit Size | Stored         |
|-----------|----------|----------------|
| `0..1`    | 2        | `material_type` |
| `2..5`    | 4        | `normal.x`     |
| `6..9`    | 4        | `normal.y`     |
| `10..13`  | 4        | `normal.z`     |
| `14..19`  | 6        | `color.r`      |
| `20..25`  | 6        | `color.g`      |
| `26..31`  | 6        | `color.b`      |

Total: `32` bits.

Notes:

* `material_type` supports `4` material classes.
* Normals are stored as `vec3` components with `4` bits per axis.
* Colors are stored as `RGB666`.
* Shader decode should unpack into normalized float ranges on load.
* If we keep explicit occupancy masks, all `4` material ids stay available.
* If we later infer occupancy from the voxel word, one material id may need to
  become the "empty" sentinel.

## Example Chunk Sizes

Assumptions for the table below:

* Chunk dimensions are `64x64x64` (`262,144` voxels).
* Regions are `8x8x8` (`512` voxels per region, `512` regions per chunk).
* Region headers are already included in the fixed chunk cost.
* Blob payloads are only allocated for non-empty regions.
* Each populated region costs `2,048` bytes.
* Total bytes = `2,124 + populated_regions * 2,048`.

| Chunk Fill | Occupied Voxels | Populated Regions | Total Bytes | Total KiB | Reduction vs dense `u32` array |
|------------|-----------------|-------------------|-------------|-----------|--------------------------------|
| 0%         | 0               | 0                 | 2,124       | 2.07      | 99.80%                         |
| 25%        | 65,536          | 128               | 264,268     | 258.07    | 74.80%                         |
| 50%        | 131,072         | 256               | 526,412     | 514.07    | 49.80%                         |
| 100%       | 262,144         | 512               | 1,050,700   | 1,026.07  | -0.20%                         |

For comparison, a dense `u32` voxel array would use
`262,144 * 4 = 1,048,576` bytes (`1,024 KiB`) per chunk.

At full occupancy this layout is slightly larger than a monolithic dense array
because of the chunk and region headers. That is acceptable for the prototype:
the point of the hierarchy is to avoid paying for empty space and to preserve a
clean traversal structure for later streaming work.

## Editing Strategy

For the prototype, keep edits simple:

* Large-scale gameplay edits can be applied on the CPU, then rebuild the
  affected chunk from source voxel data.
* Small procedural animation should be treated as a full chunk or region rebuild
  unless we later prove that partial repacking is worth the complexity.

This favors correctness and iteration speed over fully dynamic in-place updates.

## Implementation Notes For This Repo

The most practical migration path in `wgpu-voxels` looks like this:

1. Keep the existing chunk and region dimensions from `src/scene/voxel_mask.rs`.
2. Split the current monolithic occupancy representation into:
   * a compact chunk header buffer
   * a region-header buffer
   * a blob buffer containing packed `u32` voxel payloads
3. Start with CPU-built chunks uploaded into storage buffers.
4. Update traversal code to:
   * test the chunk region mask first
   * fetch the region header
   * compute the local voxel index inside the `8x8x8` region
   * unpack the `u32` voxel payload into material, normal, and color
5. Keep occupancy masks initially so shader-side migration stays easy.
6. Only revisit streaming, paging, and defragmentation once the static format is
   working end to end.

One likely transitional step is to keep the current per-region leaf masks during
debugging, then decide whether they stay as the long-term occupancy source or
whether occupancy can be inferred directly from the packed voxel word.

## Outstanding Design Work

* Solid chunk fast-path handling
* Blob allocation strategy
* Defragmentation and recycling of freed blob space
* Chunk streaming and paging
* Material-type catalog for the `2`-bit `material_type`
* Normal quantization and decode scheme for the `4`-bit axes
* Edit pipeline for persistent world changes
* Whether occupancy stays explicit or is inferred from packed voxels
