mod context;
mod output;
mod passes;

use std::{path::Path, sync::Arc};

use wgpu::util::DeviceExt;
use winit::{
    dpi::PhysicalSize,
    window::{Window, WindowId},
};

use crate::{
    InputState,
    scene::{
        Camera, OCCUPANCY_WORD_COUNT, OBJECT_BOUNDS_MAX, OBJECT_BOUNDS_MIN,
        ProceduralAccelerationScene, RenderObject, build_sphere_voxel_mask,
    },
};

use self::{
    context::GpuContext,
    output::OutputTarget,
    passes::{BlitPass, ComputeVoxelsPass, FpsOverlay},
};

struct PresentationPasses {
    blit_pass: BlitPass,
    fps_overlay: FpsOverlay,
}

pub(crate) struct Renderer {
    context: GpuContext,
    camera: Camera,
    camera_buffer: wgpu::Buffer,
    voxel_mask_buffer: wgpu::Buffer,
    active_objects: usize,
    procedural_scene: ProceduralAccelerationScene,
    output_target: OutputTarget,
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
        let size = context.current_size();
        let camera_buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("camera buffer"),
                contents: bytemuck::bytes_of(&camera.to_uniform(size)),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let voxel_mask = build_scene_voxel_masks(all_objects);
        let voxel_mask_buffer =
            context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("voxel occupancy bitmask"),
                    contents: bytemuck::cast_slice(voxel_mask.as_slice()),
                    usage: wgpu::BufferUsages::STORAGE,
                });

        let procedural_scene = ProceduralAccelerationScene::build(
            &context.device,
            &context.queue,
            active_objects,
            OBJECT_BOUNDS_MIN,
            OBJECT_BOUNDS_MAX,
        )?;

        let output_target = OutputTarget::new(&context.device, size.width, size.height);
        let compute_pass = ComputeVoxelsPass::new(
            &context.device,
            output_target.view(),
            output_target.coarse_depth_view(),
            procedural_scene.tlas(),
            &camera_buffer,
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
            voxel_mask_buffer,
            active_objects: active_objects.len(),
            procedural_scene,
            output_target,
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
            self.output_target.coarse_depth_view(),
            self.procedural_scene.tlas(),
            &self.camera_buffer,
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
            self.output_target.view(),
            self.output_target.coarse_depth_view(),
            self.procedural_scene.tlas(),
            &self.camera_buffer,
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
}

fn build_scene_voxel_masks(objects: &[RenderObject]) -> Vec<u32> {
    let mut words = Vec::with_capacity(objects.len() * OCCUPANCY_WORD_COUNT);
    for object in objects {
        words.extend(build_sphere_voxel_mask(
            OBJECT_BOUNDS_MIN,
            OBJECT_BOUNDS_MAX,
            object.radius,
        ));
    }
    words
}
