mod context;
mod output;
mod passes;

use std::{path::Path, sync::Arc};

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use winit::{
    dpi::PhysicalSize,
    window::{Window, WindowId},
};

use crate::{
    InputState,
    scene::{
        Camera, OBJECT_BOUNDS_MAX, OBJECT_BOUNDS_MIN, OCCUPANCY_WORD_COUNT,
        ProceduralAccelerationScene, RenderObject,
    },
};

use self::{
    context::GpuContext,
    output::OutputTarget,
    passes::{BlitPass, ComputeVoxelsPass, FpsOverlay, GenerateVoxelsPass},
};

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

pub(crate) struct Renderer {
    context: GpuContext,
    camera: Camera,
    camera_buffer: wgpu::Buffer,
    debug_visualization_buffer: wgpu::Buffer,
    voxel_mask_buffer: wgpu::Buffer,
    procedural_scene: ProceduralAccelerationScene,
    output_target: OutputTarget,
    generate_voxels_pass: GenerateVoxelsPass,
    compute_pass: ComputeVoxelsPass,
    presentation: Option<PresentationPasses>,
}

impl Renderer {
    pub(crate) async fn new(window: Arc<Window>, objects: &[RenderObject]) -> Result<Self, String> {
        let context = GpuContext::new(window).await?;
        Self::new_with_context(context, objects).await
    }

    pub(crate) async fn new_headless(
        size: PhysicalSize<u32>,
        objects: &[RenderObject],
    ) -> Result<Self, String> {
        let context = GpuContext::new_headless(size).await?;
        Self::new_with_context(context, objects).await
    }

    async fn new_with_context(
        context: GpuContext,
        objects: &[RenderObject],
    ) -> Result<Self, String> {
        let camera = Camera::new();
        let size = context.current_size();
        let camera_buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("camera buffer"),
                contents: bytemuck::bytes_of(&camera.to_uniform(size)),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let debug_visualization_buffer =
            Self::create_debug_visualization_buffer(&context.device, objects);
        let voxel_mask_buffer = Self::create_voxel_mask_buffer(&context.device, objects);
        let generate_voxels_pass =
            GenerateVoxelsPass::new(&context.device, &voxel_mask_buffer, objects);
        Self::dispatch_voxel_generation(&context.device, &context.queue, &generate_voxels_pass);

        let procedural_scene = ProceduralAccelerationScene::build(
            &context.device,
            &context.queue,
            objects,
            OBJECT_BOUNDS_MIN,
            OBJECT_BOUNDS_MAX,
        )?;

        let output_target = OutputTarget::new(&context.device, size.width, size.height);
        let compute_pass = ComputeVoxelsPass::new(
            &context.device,
            size.width,
            size.height,
            output_target.view(),
            output_target.world_position_view(),
            output_target.shading_input_view(),
            output_target.coarse_depth_view(),
            procedural_scene.tlas(),
            &camera_buffer,
            &debug_visualization_buffer,
            &voxel_mask_buffer,
        );
        let presentation = context.window.as_ref().map(|_| PresentationPasses {
            blit_pass: BlitPass::new(
                &context.device,
                context.surface_format(),
                output_target.view(),
            ),
            fps_overlay: FpsOverlay::new(&context.device, context.surface_format()),
        });

        Ok(Self {
            context,
            camera,
            camera_buffer,
            debug_visualization_buffer,
            voxel_mask_buffer,
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

    pub(crate) fn sync_scene(&mut self, objects: &[RenderObject]) -> Result<(), String> {
        self.debug_visualization_buffer =
            Self::create_debug_visualization_buffer(&self.context.device, objects);
        self.voxel_mask_buffer = Self::create_voxel_mask_buffer(&self.context.device, objects);
        self.generate_voxels_pass =
            GenerateVoxelsPass::new(&self.context.device, &self.voxel_mask_buffer, objects);
        Self::dispatch_voxel_generation(
            &self.context.device,
            &self.context.queue,
            &self.generate_voxels_pass,
        );
        self.procedural_scene = ProceduralAccelerationScene::build(
            &self.context.device,
            &self.context.queue,
            objects,
            OBJECT_BOUNDS_MIN,
            OBJECT_BOUNDS_MAX,
        )?;
        self.compute_pass.rebind(
            &self.context.device,
            self.context.current_size().width,
            self.context.current_size().height,
            self.output_target.view(),
            self.output_target.world_position_view(),
            self.output_target.shading_input_view(),
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
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("headless frame encoder"),
                });

        let (coarse_width, coarse_height) = self.output_target.coarse_depth_size();
        let size = self.context.current_size();
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
            size.width,
            size.height,
            self.output_target.view(),
            self.output_target.world_position_view(),
            self.output_target.shading_input_view(),
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

    fn create_debug_visualization_buffer(
        device: &wgpu::Device,
        objects: &[RenderObject],
    ) -> wgpu::Buffer {
        let debug_visualization = debug_visualization_params(objects);
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("debug visualization buffer"),
            contents: bytemuck::bytes_of(&debug_visualization),
            usage: wgpu::BufferUsages::UNIFORM,
        })
    }

    fn create_voxel_mask_buffer(device: &wgpu::Device, objects: &[RenderObject]) -> wgpu::Buffer {
        let object_count = objects
            .iter()
            .map(|object| object.object_index as u64 + 1)
            .max()
            .unwrap_or(0);
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel occupancy bitmask"),
            size: object_count * (OCCUPANCY_WORD_COUNT * core::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        })
    }

    fn dispatch_voxel_generation(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        generate_voxels_pass: &GenerateVoxelsPass,
    ) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("voxel generation encoder"),
        });
        generate_voxels_pass.dispatch(&mut encoder);
        queue.submit(Some(encoder.finish()));
    }
}

fn debug_visualization_params(objects: &[RenderObject]) -> DebugVisualizationParams {
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

    if let Some(first) = objects.first() {
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

        for object in &objects[1..] {
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
