mod context;
mod output;
mod passes;

use std::{
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use winit::{
    dpi::PhysicalSize,
    window::{Window, WindowId},
};

use crate::{
    InputState,
    scene::{
        Camera, OCCUPANCY_WORD_COUNT, OBJECT_BOUNDS_MAX, OBJECT_BOUNDS_MIN,
        ProceduralAccelerationScene, RenderObject,
    },
};

use self::{
    context::GpuContext,
    output::OutputTarget,
    passes::{BlitPass, ComputeVoxelsPass, FpsOverlay, GenerateVoxelsPass},
};

const TARGET_VOXEL_UPDATES_PER_SECOND: u64 = 60;
const TARGET_VOXEL_UPDATE_INTERVAL: Duration =
    Duration::from_nanos(1_000_000_000 / TARGET_VOXEL_UPDATES_PER_SECOND);

struct PresentationPasses {
    blit_pass: BlitPass,
    fps_overlay: FpsOverlay,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DebugVisualizationParams {
    world_min: [f32; 4],
    world_extent: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct FrameParams {
    time_seconds: f32,
    object_count: u32,
    _pad0: u32,
    _pad1: u32,
}

pub(crate) struct Renderer {
    context: GpuContext,
    camera: Camera,
    camera_buffer: wgpu::Buffer,
    debug_visualization_buffer: wgpu::Buffer,
    frame_params_buffer: wgpu::Buffer,
    voxel_mask_buffer: wgpu::Buffer,
    object_count: u32,
    active_objects: usize,
    frame_started_at: Instant,
    last_voxel_update_at: Option<Instant>,
    procedural_scene: ProceduralAccelerationScene,
    output_target: OutputTarget,
    generate_voxels_pass: GenerateVoxelsPass,
    compute_pass: ComputeVoxelsPass,
    presentation: Option<PresentationPasses>,
}

impl Renderer {
    pub(crate) async fn new(
        window: Arc<Window>,
        all_objects: &[RenderObject],
        active_objects: &[RenderObject],
    ) -> Result<Self, String> {
        let context = GpuContext::new(window).await?;
        Self::new_with_context(context, all_objects, active_objects).await
    }

    pub(crate) async fn new_headless(
        size: PhysicalSize<u32>,
        all_objects: &[RenderObject],
        active_objects: &[RenderObject],
    ) -> Result<Self, String> {
        let context = GpuContext::new_headless(size).await?;
        Self::new_with_context(context, all_objects, active_objects).await
    }

    async fn new_with_context(
        context: GpuContext,
        all_objects: &[RenderObject],
        active_objects: &[RenderObject],
    ) -> Result<Self, String> {
        let camera = Camera::new();
        let object_count = all_objects.len() as u32;
        let size = context.current_size();
        let debug_visualization = debug_visualization_params(all_objects);
        let camera_buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("camera buffer"),
                contents: bytemuck::bytes_of(&camera.to_uniform(size)),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let debug_visualization_buffer =
            context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("debug visualization buffer"),
                    contents: bytemuck::bytes_of(&debug_visualization),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
        let frame_params_buffer =
            context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("frame params buffer"),
                    contents: bytemuck::bytes_of(&FrameParams {
                        time_seconds: 0.0,
                        object_count,
                        _pad0: 0,
                        _pad1: 0,
                    }),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });
        let voxel_mask_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel occupancy bitmask"),
            size: (object_count as u64) * (OCCUPANCY_WORD_COUNT * core::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let procedural_scene = ProceduralAccelerationScene::build(
            &context.device,
            &context.queue,
            active_objects,
            OBJECT_BOUNDS_MIN,
            OBJECT_BOUNDS_MAX,
        )?;

        let output_target = OutputTarget::new(&context.device, size.width, size.height);
        let generate_voxels_pass = GenerateVoxelsPass::new(
            &context.device,
            &voxel_mask_buffer,
            &frame_params_buffer,
            object_count,
        );
        let compute_pass = ComputeVoxelsPass::new(
            &context.device,
            output_target.view(),
            output_target.world_position_view(),
            output_target.coarse_depth_view(),
            procedural_scene.tlas(),
            &camera_buffer,
            &debug_visualization_buffer,
            &voxel_mask_buffer,
        );
        let presentation = context.window.as_ref().map(|_| PresentationPasses {
            blit_pass: BlitPass::new(&context.device, context.surface_format(), output_target.view()),
            fps_overlay: FpsOverlay::new(&context.device, context.surface_format()),
        });

        Ok(Self {
            context,
            camera,
            camera_buffer,
            debug_visualization_buffer,
            frame_params_buffer,
            voxel_mask_buffer,
            object_count,
            active_objects: active_objects.len(),
            frame_started_at: Instant::now(),
            last_voxel_update_at: None,
            procedural_scene,
            output_target,
            generate_voxels_pass,
            compute_pass,
            presentation,
        })
    }

    pub(crate) fn window_id(&self) -> WindowId {
        self.context.window_id()
    }

    pub(crate) fn request_redraw(&self) {
        self.context.request_redraw();
    }

    pub(crate) fn update_camera(&mut self, input: &InputState, delta_seconds: f32) {
        self.camera.update(input, delta_seconds);
        self.update_camera_buffer();
    }

    pub(crate) fn sync_scene(&mut self, active_objects: &[RenderObject]) -> Result<(), String> {
        if active_objects.len() == self.active_objects {
            return Ok(());
        }

        self.procedural_scene = ProceduralAccelerationScene::build(
            &self.context.device,
            &self.context.queue,
            active_objects,
            OBJECT_BOUNDS_MIN,
            OBJECT_BOUNDS_MAX,
        )?;
        self.active_objects = active_objects.len();
        self.compute_pass.rebind(
            &self.context.device,
            self.output_target.view(),
            self.output_target.world_position_view(),
            self.output_target.coarse_depth_view(),
            self.procedural_scene.tlas(),
            &self.camera_buffer,
            &self.debug_visualization_buffer,
            &self.voxel_mask_buffer,
        );
        Ok(())
    }

    pub(crate) fn resize(&mut self, new_size: PhysicalSize<u32>) {
        self.context.resize(new_size);

        if new_size.width == 0 || new_size.height == 0 {
            return;
        }

        self.update_camera_buffer();
        self.recreate_output_resources();
    }

    pub(crate) fn render(&mut self) -> Result<(), String> {
        let Some(frame) = self.context.acquire_frame()? else {
            return Ok(());
        };
        self.update_frame_params_buffer();
        let should_update_voxels = self.should_update_voxels();
        let presentation = self
            .presentation
            .as_mut()
            .ok_or_else(|| String::from("windowed render called on a headless renderer"))?;

        let surface_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        presentation.fps_overlay.update(
            &self.context.device,
            &self.context.queue,
            self.context.current_size(),
        );

        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("frame encoder"),
                });

        let (coarse_width, coarse_height) = self.output_target.coarse_depth_size();
        let size = self.context.current_size();
        if should_update_voxels {
            self.generate_voxels_pass.dispatch(&mut encoder);
        }
        self.compute_pass.dispatch(
            &mut encoder,
            size.width,
            size.height,
            coarse_width,
            coarse_height,
        );

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("present pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &surface_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            presentation.blit_pass.draw(&mut render_pass);
            presentation.fps_overlay.draw(&mut render_pass);
        }

        self.context.queue.submit(Some(encoder.finish()));
        self.context
            .window
            .as_ref()
            .expect("window missing for windowed renderer")
            .pre_present_notify();
        frame.present();
        Ok(())
    }

    pub(crate) fn render_headless(&mut self) -> Result<(), String> {
        self.update_frame_params_buffer();
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("headless frame encoder"),
                });

        let (coarse_width, coarse_height) = self.output_target.coarse_depth_size();
        let size = self.context.current_size();
        if self.should_update_voxels() {
            self.generate_voxels_pass.dispatch(&mut encoder);
        }
        self.compute_pass.dispatch(
            &mut encoder,
            size.width,
            size.height,
            coarse_width,
            coarse_height,
        );

        self.context.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    pub(crate) fn save_headless_png(&self, path: &Path) -> Result<(), String> {
        self.output_target
            .save_png(&self.context.device, &self.context.queue, path)
    }

    fn recreate_output_resources(&mut self) {
        let size = self.context.current_size();
        self.output_target
            .recreate(&self.context.device, size.width, size.height);
        self.compute_pass.rebind(
            &self.context.device,
            self.output_target.view(),
            self.output_target.world_position_view(),
            self.output_target.coarse_depth_view(),
            self.procedural_scene.tlas(),
            &self.camera_buffer,
            &self.debug_visualization_buffer,
            &self.voxel_mask_buffer,
        );
        if let Some(presentation) = self.presentation.as_mut() {
            presentation
                .blit_pass
                .rebind(&self.context.device, self.output_target.view());
        }
    }

    fn update_camera_buffer(&self) {
        let uniform = self.camera.to_uniform(self.context.current_size());
        self.context
            .queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&uniform));
    }

    fn update_frame_params_buffer(&self) {
        let frame_params = FrameParams {
            time_seconds: self.frame_started_at.elapsed().as_secs_f32(),
            object_count: self.object_count,
            _pad0: 0,
            _pad1: 0,
        };
        self.context.queue.write_buffer(
            &self.frame_params_buffer,
            0,
            bytemuck::bytes_of(&frame_params),
        );
    }

    fn should_update_voxels(&mut self) -> bool {
        let now = Instant::now();
        let Some(last_update_at) = self.last_voxel_update_at else {
            self.last_voxel_update_at = Some(now);
            return true;
        };

        if now.duration_since(last_update_at) < TARGET_VOXEL_UPDATE_INTERVAL {
            return false;
        }

        self.last_voxel_update_at = Some(now);
        true
    }
}

fn debug_visualization_params(all_objects: &[RenderObject]) -> DebugVisualizationParams {
    let mut world_min = [
        OBJECT_BOUNDS_MIN[0],
        OBJECT_BOUNDS_MIN[1],
        OBJECT_BOUNDS_MIN[2],
    ];
    let mut world_max = [
        OBJECT_BOUNDS_MAX[0],
        OBJECT_BOUNDS_MAX[1],
        OBJECT_BOUNDS_MAX[2],
    ];

    if let Some(first) = all_objects.first() {
        world_min = [
            first.position[0] + OBJECT_BOUNDS_MIN[0],
            first.position[1] + OBJECT_BOUNDS_MIN[1],
            first.position[2] + OBJECT_BOUNDS_MIN[2],
        ];
        world_max = [
            first.position[0] + OBJECT_BOUNDS_MAX[0],
            first.position[1] + OBJECT_BOUNDS_MAX[1],
            first.position[2] + OBJECT_BOUNDS_MAX[2],
        ];

        for object in &all_objects[1..] {
            let object_min = [
                object.position[0] + OBJECT_BOUNDS_MIN[0],
                object.position[1] + OBJECT_BOUNDS_MIN[1],
                object.position[2] + OBJECT_BOUNDS_MIN[2],
            ];
            let object_max = [
                object.position[0] + OBJECT_BOUNDS_MAX[0],
                object.position[1] + OBJECT_BOUNDS_MAX[1],
                object.position[2] + OBJECT_BOUNDS_MAX[2],
            ];

            for axis in 0..3 {
                world_min[axis] = world_min[axis].min(object_min[axis]);
                world_max[axis] = world_max[axis].max(object_max[axis]);
            }
        }
    }

    DebugVisualizationParams {
        world_min: [world_min[0], world_min[1], world_min[2], 0.0],
        world_extent: [
            (world_max[0] - world_min[0]).max(1e-5),
            (world_max[1] - world_min[1]).max(1e-5),
            (world_max[2] - world_min[2]).max(1e-5),
            0.0,
        ],
    }
}
