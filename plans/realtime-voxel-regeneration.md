# Realtime Voxel Regeneration Plan

This note captures the runtime SDF voxel regeneration path that was removed in
favor of one-time static chunk generation. The intent is to preserve the useful
implementation details so we can reintroduce chunk rebuilds later without
re-discovering the wiring from scratch.

## What Was Removed

The old path rebuilt voxel occupancy on the GPU during rendering:

1. `src/renderer/mod.rs`
   stored a `GenerateVoxelsPass`, a `FrameParams` uniform buffer, and a
   `last_voxel_update_at` timestamp.
2. `Renderer::render` and `Renderer::render_headless`
   called `should_update_voxels()` before tracing.
3. `should_update_voxels()`
   allowed a regeneration pass on the first frame and then throttled updates to
   `60 Hz` with:
   `TARGET_VOXEL_UPDATES_PER_SECOND = 60`
   `TARGET_VOXEL_UPDATE_INTERVAL = 1_000_000_000 ns / 60`
4. `src/renderer/passes/generate_voxels.rs`
   dispatched two compute kernels from `src/voxel_generation.wgsl`:
   `clear_occupancy_main`
   `populate_debug_sdf_main`

## Old GPU Data Flow

Bindings for the removed generation pass:

- `@group(0) @binding(0)` storage buffer of `atomic<u32>` occupancy words
- `@group(0) @binding(1)` uniform `FrameParams`

`FrameParams` contained:

- `time_seconds: f32`
- `object_count: u32`

The occupancy layout matched the current chunk mask format:

- `VOXEL_GRID_DIM = 64`
- `REGION_AXIS = 8`
- `REGION_COUNT = 512`
- `COARSE_REGION_AXIS = 4`
- `MASK_WORD_COUNT = 16`
- `COARSE_MASK_WORD_OFFSET = 16`
- `LEAF_MASK_WORD_OFFSET = 18`
- `OCCUPANCY_WORD_COUNT = 8210`

Workgroup sizing in the removed pass:

- clear: `@workgroup_size(256, 1, 1)`
- populate: `@workgroup_size(8, 8, 2)`

Dispatch dimensions:

- clear words: `ceil(object_count * OCCUPANCY_WORD_COUNT / 256)`
- populate X/Y: `ceil(64 / 8)`
- populate Z: `ceil(object_count * 64 / 2)`

## Old Shape Logic

The removed shader evaluated a debug torus SDF in object-local space:

- object bounds were `[0, 0, 0]` to `[1, 1, 1]`
- the sample point was the voxel center
- the point was rotated over time using `frame_params.time_seconds`
- filled voxels set:
  region bit
  coarse-region bit
  leaf bit

That gave us a nice proof-of-concept for per-frame dynamic population, but every
object shared the same animated local-space volume.

## Why We Switched

The current prototype does not need live voxel mutation yet, and the old path
had a few drawbacks for the next world-building phase:

- every chunk effectively shared one procedural model
- regeneration was tied to frame cadence rather than chunk dirtiness
- scene spawning/model updates forced us to keep rebuild orchestration live even
  when content was static

## Recommended Path Back To Dynamic Rebuilds

When we bring this back, the better shape is chunk-driven rebuilds rather than
per-frame global updates:

1. Keep chunk occupancy storage as one unique volume per chunk.
2. Track dirty chunk indices on the CPU.
3. Rebuild only dirty chunks, either:
   CPU-side into a staging vector with `queue.write_buffer`, or
   GPU-side with a compute pass that writes only the target chunk slice.
4. Pass chunk/world transform data to the generator so density is evaluated in
   world space, not shared local space.
5. If we return to GPU generation, bind at least:
   chunk metadata buffer
   occupancy storage buffer
   dirty chunk list / indirect dispatch args
6. Prefer explicit rebuild triggers:
   scene edits
   streaming in new chunks
   terrain brush operations
   authored model import
7. Keep the coarse and leaf mask packing exactly as it is now so tracing code
   does not need to change.

## Suggested Future Implementation

Phase A:
CPU rebuilds for dirty chunks only. This is the lowest-risk route and keeps the
render pass simple.

Phase B:
GPU rebuild kernel per dirty chunk, using the current occupancy format and a
world-space density function or SDF library.

Phase C:
Move from noise-only generation to authored chunk content and chunk streaming,
while preserving the same per-chunk buffer addressing model.
