use std::f32::consts::FRAC_PI_3;

use bytemuck::{Pod, Zeroable};
use glam::{Quat, Vec3};
use winit::dpi::PhysicalSize;

use crate::InputState;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct CameraUniform {
    position: [f32; 4],
    forward: [f32; 4],
    right: [f32; 4],
    up: [f32; 4],
    viewport: [f32; 4],
}

pub(crate) struct Camera {
    position: Vec3,
    yaw: f32,
    pitch: f32,
    vertical_fov_radians: f32,
}

impl Camera {
    pub(crate) fn new() -> Self {
        Self {
            position: Vec3::new(0.5, 0.5, 2.2),
            yaw: 0.0,
            pitch: 0.0,
            vertical_fov_radians: FRAC_PI_3,
        }
    }

    pub(crate) fn to_uniform(&self, size: PhysicalSize<u32>) -> CameraUniform {
        let aspect = (size.width.max(1) as f32) / (size.height.max(1) as f32);
        let basis = self.basis();
        let tan_half_fov = (self.vertical_fov_radians * 0.5).tan();

        CameraUniform {
            position: self.position.extend(0.0).to_array(),
            forward: basis.forward.extend(0.0).to_array(),
            right: basis.right.extend(0.0).to_array(),
            up: basis.up.extend(0.0).to_array(),
            viewport: [tan_half_fov * aspect, tan_half_fov, aspect, 0.0],
        }
    }

    pub(crate) fn position_uniform(&self) -> [f32; 4] {
        self.position.extend(0.0).to_array()
    }

    pub(crate) fn update(&mut self, input: &InputState, delta_seconds: f32) {
        if delta_seconds <= 0.0 {
            return;
        }

        let basis = self.basis();
        let movement_speed = 4.5;
        let look_sensitivity = 0.0035;
        let mut movement = Vec3::ZERO;

        if input.forward {
            movement += basis.forward;
        }
        if input.backward {
            movement -= basis.forward;
        }
        if input.left {
            movement -= basis.right;
        }
        if input.right {
            movement += basis.right;
        }
        if input.up {
            movement += basis.up;
        }
        if input.down {
            movement -= basis.up;
        }

        if movement.length_squared() > 0.0 {
            self.position += movement.normalize() * movement_speed * delta_seconds;
        }

        self.yaw -= input.mouse_delta_x * look_sensitivity;
        self.pitch = (self.pitch - input.mouse_delta_y * look_sensitivity).clamp(-1.3, 1.3);
    }

    fn basis(&self) -> CameraBasis {
        let rotation = Quat::from_rotation_y(self.yaw) * Quat::from_rotation_x(self.pitch);
        let forward = rotation * -Vec3::Z;
        let right = rotation * Vec3::X;
        let up = rotation * Vec3::Y;

        CameraBasis {
            forward: forward.normalize_or_zero(),
            right: right.normalize_or_zero(),
            up: up.normalize_or_zero(),
        }
    }
}

struct CameraBasis {
    forward: Vec3,
    right: Vec3,
    up: Vec3,
}

#[cfg(test)]
mod tests {
    use super::Camera;
    use crate::InputState;

    #[test]
    fn forward_input_moves_camera_forward() {
        let mut camera = Camera::new();
        let start = camera.position;
        let input = InputState {
            forward: true,
            ..InputState::default()
        };

        camera.update(&input, 1.0);

        assert!(camera.position.z < start.z);
    }

    #[test]
    fn pitch_is_clamped() {
        let mut camera = Camera::new();
        let input = InputState {
            mouse_delta_y: -10_000.0,
            ..InputState::default()
        };

        camera.update(&input, 1.0);

        assert!(camera.pitch <= 1.3);
    }
}
