mod context;
mod output;
mod passes;

use std::sync::Arc;

use wgpu::util::DeviceExt;
use winit::{
    dpi::PhysicalSize,
    window::{Window, WindowId},
};

use crate::{
    InputState,
    scene::{
        Camera, INSTANCE_POSITIONS, OBJECT_BOUNDS_MAX, OBJECT_BOUNDS_MIN,
        ProceduralAccelerationScene, build_sphere_voxel_mask,
    },
};

use self::{
    context::GpuContext,
    output::OutputTarget,
    passes::{BlitPass, ComputeVoxelsPass, FpsOverlay},
};

pub(crate) struct Renderer {
    context: GpuContext,
    camera: Camera,
    camera_buffer: wgpu::Buffer,
    voxel_mask_buffer: wgpu::Buffer,
    procedural_scene: ProceduralAccelerationScene,
    output_target: OutputTarget,
    compute_pass: ComputeVoxelsPass,
    blit_pass: BlitPass,
    fps_overlay: FpsOverlay,
}

impl Renderer {
    pub(crate) async fn new(window: Arc<Window>) -> Result<Self, String> {
        let context = GpuContext::new(window).await?;
        let camera = Camera::new();
        let camera_buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("camera buffer"),
                contents: bytemuck::bytes_of(&camera.to_uniform(context.current_size())),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let voxel_mask = build_sphere_voxel_mask(OBJECT_BOUNDS_MIN, OBJECT_BOUNDS_MAX);
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
            INSTANCE_POSITIONS.as_slice(),
            OBJECT_BOUNDS_MIN,
            OBJECT_BOUNDS_MAX,
        )?;

        let output_target = OutputTarget::new(&context.device, context.surface_config());
        let compute_pass = ComputeVoxelsPass::new(
            &context.device,
            output_target.view(),
            output_target.coarse_depth_view(),
            procedural_scene.tlas(),
            &camera_buffer,
            &voxel_mask_buffer,
        );
        let blit_pass = BlitPass::new(
            &context.device,
            context.surface_format(),
            output_target.view(),
        );
        let fps_overlay = FpsOverlay::new(&context.device, context.surface_format());

        Ok(Self {
            context,
            camera,
            camera_buffer,
            voxel_mask_buffer,
            procedural_scene,
            output_target,
            compute_pass,
            blit_pass,
            fps_overlay,
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

        let surface_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.fps_overlay.update(
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
        self.compute_pass.dispatch(
            &mut encoder,
            self.context.surface_config().width,
            self.context.surface_config().height,
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
            self.blit_pass.draw(&mut render_pass);
            self.fps_overlay.draw(&mut render_pass);
        }

        self.context.queue.submit(Some(encoder.finish()));
        self.context.window.pre_present_notify();
        frame.present();
        Ok(())
    }

    fn recreate_output_resources(&mut self) {
        self.output_target
            .recreate(&self.context.device, self.context.surface_config());
        self.compute_pass.rebind(
            &self.context.device,
            self.output_target.view(),
            self.output_target.coarse_depth_view(),
            self.procedural_scene.tlas(),
            &self.camera_buffer,
            &self.voxel_mask_buffer,
        );
        self.blit_pass
            .rebind(&self.context.device, self.output_target.view());
    }

    fn update_camera_buffer(&self) {
        let uniform = self.camera.to_uniform(self.context.current_size());
        self.context
            .queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&uniform));
    }
}
