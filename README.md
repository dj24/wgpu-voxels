# wgpu-voxels

A small Rust/WGPU voxel renderer. It opens a `winit` window, builds a procedural
scene acceleration structure, renders into an offscreen texture with a compute
shader, then blits the result to the swapchain with an FPS overlay.

Voxel occupancy is generated once on startup on the CPU. Each chunk gets its
own mask from a world-space sine-based density field, so neighboring chunks are
distinct while still reading as one continuous world.

## Running

```powershell
cargo run
```

```powershell
cargo run -- --headless-png screenshot-headless.png --delay-ms 1000
```

The renderer currently requests Vulkan and `wgpu` experimental ray query support,
so it needs a compatible GPU/driver.

## Repository Structure

- `Cargo.toml` - crate metadata and dependencies.
- `Cargo.lock` - pinned dependency versions for reproducible builds.
- `LICENSE` - project license.
- `src/main.rs` - application entry point, `winit` event loop, window lifecycle,
  keyboard input, CLI parsing, frame timing, redraw scheduling, and headless
  PNG capture entrypoint.
- `src/scene/` - CPU-side scene data.
- `src/scene/camera.rs` - movable camera state and uniform conversion.
- `src/scene/procedural.rs` - procedural instance data and acceleration
  structure construction.
- `src/renderer/` - GPU setup and frame orchestration.
- `src/renderer/context.rs` - WGPU instance, adapter, device, queue, surface,
  swapchain configuration, and resize/acquire handling.
- `src/renderer/output.rs` - offscreen storage texture used as the compute
  render target.
- `src/renderer/passes/` - individual GPU passes:
  compute voxel rendering, blitting to the surface, and FPS overlay drawing.
- `src/*.wgsl` - shader sources used by the render passes.
- `target/` - Cargo build output, ignored by git.
- `.idea/` - local IDE metadata, not part of the runtime.

## Render Flow

1. `main.rs` creates the window and initializes `Renderer`.
2. `Renderer` builds GPU context, camera resources, scene acceleration data, and
   a static per-chunk voxel mask buffer.
3. Each frame updates the camera from input, dispatches `compute.wgsl` into the
   offscreen output texture, blits that texture to the window, draws the FPS
   overlay, and presents the frame.

Headless mode skips the window and presentation pass, renders directly into the
offscreen output texture, and writes that texture to a PNG.

## Debug Views

The interactive renderer supports runtime debug view toggles:

- `1` switches back to the default shaded output.
- `2` shows the heatmap view.
- `3` shows normalized world-position output.
- `4` shows final visible-surface depth from the full-resolution trace.

The debug views are applied as a fullscreen compute pass after the normal shaded
frame is generated, so the default render path stays intact while debug output
can be enabled on demand.
