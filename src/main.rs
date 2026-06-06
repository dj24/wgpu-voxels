use std::{
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{DeviceEvent, ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, KeyCode, NamedKey, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};

mod renderer;
mod scene;

use renderer::{DebugView, Renderer};
use scene::{
    ActiveSceneSnapshot, SceneWorld, advance_chunk_loading, build_scene_world,
    collect_active_render_objects, load_max_active_chunks,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = RuntimeConfig::from_env_args()?;
    match config.launch_mode {
        LaunchMode::Windowed => run_interactive()?,
        LaunchMode::HeadlessCapture { output_path, delay } => run_headless(&output_path, delay)?,
    }
    Ok(())
}

fn run_interactive() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::new()?;
    let mut app = App::default();
    event_loop.run_app(&mut app)?;
    Ok(())
}

fn run_headless(output_path: &Path, delay: Duration) -> Result<(), Box<dyn std::error::Error>> {
    const DEFAULT_CAPTURE_SIZE: PhysicalSize<u32> = PhysicalSize::new(1280, 720);

    let mut world = build_scene_world();
    load_max_active_chunks(&mut world);
    let objects = collect_active_render_objects(&world);
    let mut renderer = pollster::block_on(Renderer::new_headless(DEFAULT_CAPTURE_SIZE, &objects))?;
    std::thread::sleep(delay);
    renderer.update_camera(&InputState::default(), delay.as_secs_f32());
    renderer.render_headless()?;
    renderer.save_headless_png(output_path)?;
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RuntimeConfig {
    launch_mode: LaunchMode,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum LaunchMode {
    Windowed,
    HeadlessCapture {
        output_path: PathBuf,
        delay: Duration,
    },
}

impl RuntimeConfig {
    fn from_env_args() -> Result<Self, String> {
        parse_runtime_config(std::env::args().skip(1))
    }
}

fn parse_runtime_config<I, S>(args: I) -> Result<RuntimeConfig, String>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut output_path = None;
    let mut delay = None;
    let mut args = args.into_iter().map(Into::into);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--headless-png" => {
                let Some(path) = args.next() else {
                    return Err(String::from("expected a file path after --headless-png"));
                };
                output_path = Some(PathBuf::from(path));
            }
            "--delay-ms" => {
                let Some(raw_delay) = args.next() else {
                    return Err(String::from(
                        "expected a millisecond value after --delay-ms",
                    ));
                };
                let parsed_delay = raw_delay
                    .parse::<u64>()
                    .map_err(|_| format!("invalid --delay-ms value: {raw_delay}"))?;
                delay = Some(Duration::from_millis(parsed_delay));
            }
            _ => return Err(format!("unrecognized argument: {arg}")),
        }
    }

    match (output_path, delay) {
        (None, None) => Ok(RuntimeConfig {
            launch_mode: LaunchMode::Windowed,
        }),
        (Some(output_path), Some(delay)) => Ok(RuntimeConfig {
            launch_mode: LaunchMode::HeadlessCapture { output_path, delay },
        }),
        (Some(_), None) => Err(String::from(
            "--delay-ms is required when using --headless-png",
        )),
        (None, Some(_)) => Err(String::from(
            "--headless-png is required when using --delay-ms",
        )),
    }
}

struct App {
    renderer: Option<Renderer>,
    debug_view: DebugView,
    world: SceneWorld,
    scene_snapshot: ActiveSceneSnapshot,
    input: InputState,
    last_frame_at: Option<Instant>,
}

impl Default for App {
    fn default() -> Self {
        let world = build_scene_world();
        Self {
            renderer: None,
            debug_view: DebugView::Default,
            scene_snapshot: ActiveSceneSnapshot {
                active_count: collect_active_render_objects(&world).len(),
            },
            world,
            input: InputState::default(),
            last_frame_at: None,
        }
    }
}

#[derive(Clone, Copy, Default)]
pub(crate) struct InputState {
    pub(crate) forward: bool,
    pub(crate) backward: bool,
    pub(crate) left: bool,
    pub(crate) right: bool,
    pub(crate) up: bool,
    pub(crate) down: bool,
    pub(crate) mouse_delta_x: f32,
    pub(crate) mouse_delta_y: f32,
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
            _ => {}
        }
    }

    fn add_mouse_delta(&mut self, delta_x: f64, delta_y: f64) {
        self.mouse_delta_x += delta_x as f32;
        self.mouse_delta_y += delta_y as f32;
    }

    fn take_mouse_delta(&mut self) -> (f32, f32) {
        let delta = (self.mouse_delta_x, self.mouse_delta_y);
        self.mouse_delta_x = 0.0;
        self.mouse_delta_y = 0.0;
        delta
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.renderer.is_some() {
            return;
        }

        let window_attributes = Window::default_attributes()
            .with_title("wgpu UV Compute")
            .with_inner_size(PhysicalSize::new(1920, 1080));
        let window = match event_loop.create_window(window_attributes) {
            Ok(window) => Arc::new(window),
            Err(error) => {
                eprintln!("failed to create window: {error}");
                event_loop.exit();
                return;
            }
        };

        let objects = collect_active_render_objects(&self.world);
        match pollster::block_on(Renderer::new(window, &objects)) {
            Ok(mut renderer) => {
                renderer.set_debug_view(self.debug_view);
                let window = renderer.window();
                window.set_cursor_visible(false);
                if let Err(error) = window.set_cursor_grab(CursorGrabMode::Locked) {
                    if let Err(confined_error) = window.set_cursor_grab(CursorGrabMode::Confined) {
                        eprintln!(
                            "failed to grab cursor: {error}; fallback failed: {confined_error}"
                        );
                    }
                }
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
        self.world = build_scene_world();
        self.scene_snapshot = ActiveSceneSnapshot {
            active_count: collect_active_render_objects(&self.world).len(),
        };
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
                    if event.state == ElementState::Pressed {
                        let debug_view = match code {
                            KeyCode::Digit1 => Some(DebugView::Default),
                            KeyCode::Digit2 => Some(DebugView::Heatmap),
                            KeyCode::Digit3 => Some(DebugView::WorldPosition),
                            KeyCode::Digit4 => Some(DebugView::Depth),
                            KeyCode::Digit5 => Some(DebugView::Normals),
                            KeyCode::Digit6 => Some(DebugView::SamplingRate),
                            KeyCode::Digit7 => Some(DebugView::MotionVectors),
                            _ => None,
                        };
                        if let Some(debug_view) = debug_view {
                            self.debug_view = debug_view;
                            renderer.set_debug_view(debug_view);
                        }
                    }
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

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        if self.renderer.is_none() {
            return;
        }

        if let DeviceEvent::MouseMotion { delta } = event {
            self.input.add_mouse_delta(delta.0, delta.1);
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        event_loop.set_control_flow(ControlFlow::Poll);

        if let Some(renderer) = self.renderer.as_mut() {
            let now = Instant::now();
            let previous = self.last_frame_at.replace(now).unwrap_or(now);
            let delta_seconds = (now - previous).as_secs_f32();
            let (mouse_delta_x, mouse_delta_y) = self.input.take_mouse_delta();
            let mut frame_input = self.input;
            frame_input.mouse_delta_x = mouse_delta_x;
            frame_input.mouse_delta_y = mouse_delta_y;
            renderer.update_camera(&frame_input, delta_seconds);
            let scene_snapshot = advance_chunk_loading(&mut self.world);
            if scene_snapshot != self.scene_snapshot {
                let active_objects = collect_active_render_objects(&self.world);
                if let Err(error) = renderer.sync_scene(&active_objects) {
                    eprintln!("scene sync error: {error}");
                    event_loop.exit();
                    return;
                }
                self.scene_snapshot = scene_snapshot;
            }
            renderer.request_redraw();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{LaunchMode, RuntimeConfig, parse_runtime_config};
    use std::{path::PathBuf, time::Duration};

    #[test]
    fn headless_mode_accepts_path_and_delay() {
        let config = parse_runtime_config(["--headless-png", "capture.png", "--delay-ms", "1500"])
            .expect("headless launch config");
        assert_eq!(
            config,
            RuntimeConfig {
                launch_mode: LaunchMode::HeadlessCapture {
                    output_path: PathBuf::from("capture.png"),
                    delay: Duration::from_millis(1500),
                },
            }
        );
    }

    #[test]
    fn headless_mode_requires_delay() {
        let error =
            parse_runtime_config(["--headless-png", "capture.png"]).expect_err("missing delay");
        assert!(error.contains("--delay-ms is required when using --headless-png"));
    }

    #[test]
    fn delay_requires_headless_mode() {
        let error = parse_runtime_config(["--delay-ms", "1000"]).expect_err("missing output path");
        assert!(error.contains("--headless-png is required when using --delay-ms"));
    }
}
