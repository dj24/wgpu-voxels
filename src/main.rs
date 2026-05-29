use std::{sync::Arc, time::Instant};

use winit::{
    application::ApplicationHandler,
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, KeyCode, NamedKey, PhysicalKey},
    window::{Window, WindowId},
};

mod fps_overlay;
mod procedural_interop;
mod renderer;

use renderer::Renderer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::new()?;
    let mut app = App::default();
    event_loop.run_app(&mut app)?;
    Ok(())
}

#[derive(Default)]
struct App {
    renderer: Option<Renderer>,
    input: InputState,
    last_frame_at: Option<Instant>,
}

#[derive(Default)]
pub(crate) struct InputState {
    pub(crate) forward: bool,
    pub(crate) backward: bool,
    pub(crate) left: bool,
    pub(crate) right: bool,
    pub(crate) up: bool,
    pub(crate) down: bool,
    pub(crate) turn_left: bool,
    pub(crate) turn_right: bool,
    pub(crate) look_up: bool,
    pub(crate) look_down: bool,
}

impl InputState {
    fn set_key(&mut self, key_code: KeyCode, pressed: bool) {
        match key_code {
            KeyCode::KeyW => self.forward = pressed,
            KeyCode::KeyS => self.backward = pressed,
            KeyCode::KeyA => self.left = pressed,
            KeyCode::KeyD => self.right = pressed,
            KeyCode::Space => self.up = pressed,
            KeyCode::ShiftLeft => self.down = pressed,
            KeyCode::ArrowLeft => self.turn_left = pressed,
            KeyCode::ArrowRight => self.turn_right = pressed,
            KeyCode::ArrowUp => self.look_up = pressed,
            KeyCode::ArrowDown => self.look_down = pressed,
            _ => {}
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.renderer.is_some() {
            return;
        }

        let window_attributes = Window::default_attributes().with_title("wgpu UV Compute");
        let window = match event_loop.create_window(window_attributes) {
            Ok(window) => Arc::new(window),
            Err(error) => {
                eprintln!("failed to create window: {error}");
                event_loop.exit();
                return;
            }
        };

        match pollster::block_on(Renderer::new(window)) {
            Ok(renderer) => {
                self.renderer = Some(renderer);
                self.last_frame_at = Some(Instant::now());
            }
            Err(error) => {
                eprintln!("failed to initialize renderer: {error}");
                event_loop.exit();
            }
        }
    }

    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        self.renderer = None;
        self.input = InputState::default();
        self.last_frame_at = None;
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };

        if renderer.window_id() != window_id {
            return;
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed
                    && matches!(event.logical_key, Key::Named(NamedKey::Escape)) =>
            {
                event_loop.exit();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    self.input
                        .set_key(code, event.state == ElementState::Pressed);
                }
            }
            WindowEvent::Resized(new_size) => renderer.resize(new_size),
            WindowEvent::RedrawRequested => {
                if let Err(error) = renderer.render() {
                    eprintln!("render error: {error}");
                    event_loop.exit();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        event_loop.set_control_flow(ControlFlow::Poll);

        if let Some(renderer) = self.renderer.as_mut() {
            let now = Instant::now();
            let previous = self.last_frame_at.replace(now).unwrap_or(now);
            renderer.update_camera(&self.input, (now - previous).as_secs_f32());
            renderer.request_redraw();
        }
    }
}
