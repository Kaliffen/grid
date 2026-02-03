use bevy::prelude::*;
use bevy::window::{MonitorSelection, VideoModeSelection, WindowMode, PresentMode};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                mode: WindowMode::Fullscreen(
                    MonitorSelection::Primary,      // which monitor
                    VideoModeSelection::Current,    // keep current resolution/refresh
                ),
                present_mode: PresentMode::AutoVsync,
                ..default()
            }),
            ..default()
        }))
        .add_systems(Startup, setup)
        .add_systems(Update, (camera_pan_zoom, update_grid_lod, draw_grids))
        .run();
}


#[derive(Component, Clone)]
struct GridOverlay {
    enabled: bool,

    // Grid transform (2D rotation-only basis; should be orthonormal)
    origin: Vec2,
    basis_x: Vec2,
    basis_y: Vec2,

    // Screen-space density target
    target_px_spacing: f32, // desired minor spacing in pixels
    major_every: u32,       // major line every N minor lines

    // Colors
    minor_color: Color,
    major_color: Color,
    axis_x_color: Color,
    axis_y_color: Color,

    // Misc
    z_layer: f32,
    max_half_lines: u32, // clamp "lines to each side" per axis
    draw_axes: bool,
}

#[derive(Component)]
struct GridRuntime {
    minor_step_world: f32,
}

fn setup(mut commands: Commands) {
    commands.spawn(Camera2d);

    commands.spawn((
        GridOverlay {
            enabled: true,
            origin: Vec2::ZERO,
            basis_x: Vec2::X,
            basis_y: Vec2::Y,
            target_px_spacing: 32.0,
            major_every: 5,
            minor_color: Color::srgba(0.35, 0.35, 0.35, 0.35),
            major_color: Color::srgba(0.60, 0.60, 0.60, 0.55),
            axis_x_color: Color::srgba(1.00, 0.20, 0.20, 0.85),
            axis_y_color: Color::srgba(0.20, 1.00, 0.20, 0.85),
            z_layer: 0.0,
            max_half_lines: 900,
            draw_axes: true,
        },
        GridRuntime {
            minor_step_world: 0.0,
        },
    ));

    // Rotated grid
    let theta = std::f32::consts::FRAC_PI_6; // 30 degrees
    let (s, c) = theta.sin_cos();
    let bx = Vec2::new(c, s).normalize();
    let by = Vec2::new(-s, c).normalize();

    commands.spawn((
        GridOverlay {
            enabled: true,
            origin: Vec2::new(200.0, 120.0),
            basis_x: bx,
            basis_y: by,
            target_px_spacing: 40.0,
            major_every: 4,
            minor_color: Color::srgba(0.20, 0.45, 0.85, 0.20),
            major_color: Color::srgba(0.20, 0.65, 1.00, 0.35),
            axis_x_color: Color::srgba(0.90, 0.75, 0.20, 0.85),
            axis_y_color: Color::srgba(0.85, 0.25, 0.85, 0.85),
            z_layer: 0.01,
            max_half_lines: 900,
            draw_axes: true,
        },
        GridRuntime {
            minor_step_world: 0.0,
        },
    ));
}

fn camera_pan_zoom(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    mut q_cam: Query<(&mut Transform, &mut Projection), With<Camera2d>>,
) {
    let Ok((mut transform, mut projection)) = q_cam.single_mut() else {
        return;
    };

    let ortho = match &mut *projection {
        Projection::Orthographic(o) => o,
        _ => return,
    };

    let zoom = ortho.scale.max(0.0001);
    let pan_speed = 700.0 * zoom;

    let mut pan = Vec2::ZERO;
    if keys.pressed(KeyCode::KeyW) || keys.pressed(KeyCode::ArrowUp) {
        pan.y += 1.0;
    }
    if keys.pressed(KeyCode::KeyS) || keys.pressed(KeyCode::ArrowDown) {
        pan.y -= 1.0;
    }
    if keys.pressed(KeyCode::KeyA) || keys.pressed(KeyCode::ArrowLeft) {
        pan.x -= 1.0;
    }
    if keys.pressed(KeyCode::KeyD) || keys.pressed(KeyCode::ArrowRight) {
        pan.x += 1.0;
    }
    if pan.length_squared() > 0.0 {
        pan = pan.normalize();
        transform.translation.x += pan.x * pan_speed * time.delta_secs();
        transform.translation.y += pan.y * pan_speed * time.delta_secs();
    }

    let zoom_rate = 1.25_f32;
    if keys.pressed(KeyCode::KeyZ) {
        ortho.scale = (ortho.scale / zoom_rate).clamp(0.02, 1000.0);
    }
    if keys.pressed(KeyCode::KeyX) {
        ortho.scale = (ortho.scale * zoom_rate).clamp(0.02, 1000.0);
    }
    if keys.just_pressed(KeyCode::KeyR) {
        ortho.scale = 1.0;
        transform.translation = Vec3::new(0.0, 0.0, transform.translation.z);
    }
}

fn update_grid_lod(
    q_cam: Query<&Projection, With<Camera2d>>,
    mut q_grids: Query<(&GridOverlay, &mut GridRuntime)>,
) {
    let Ok(projection) = q_cam.single() else {
        return;
    };
    let ortho = match projection {
        Projection::Orthographic(o) => o,
        _ => return,
    };

    // Common 2D ortho behavior: scale ~= world-units-per-pixel
    let wup = ortho.scale.max(0.000001);

    for (grid, mut rt) in q_grids.iter_mut() {
        if !grid.enabled {
            continue;
        }

        let desired_step_world = grid.target_px_spacing.max(2.0) * wup;
        let candidate = nice_step(desired_step_world).max(0.000001);

        if rt.minor_step_world <= 0.0 || !rt.minor_step_world.is_finite() {
            rt.minor_step_world = candidate;
            continue;
        }

        // Hysteresis to prevent LOD thrash after zoom.
        let cur = rt.minor_step_world;
        let up_threshold = cur * 1.40;
        let down_threshold = cur / 1.40;

        if candidate > up_threshold || candidate < down_threshold {
            rt.minor_step_world = candidate;
        }
    }
}

fn draw_grids(
    mut gizmos: Gizmos,
    q_cam: Query<(&Camera, &GlobalTransform, &Projection), With<Camera2d>>,
    q_grids: Query<(&GridOverlay, &GridRuntime)>,
) {
    let Ok((camera, cam_xform, projection)) = q_cam.single() else {
        return;
    };

    let ortho = match projection {
        Projection::Orthographic(o) => o,
        _ => return,
    };

    let Some(vp) = camera.logical_viewport_size() else {
        return;
    };

    let wup = ortho.scale.max(0.000001);

    // Camera view half-extents in world space (axis-aligned)
    let half_w_world = vp.x * wup * 0.5;
    let half_h_world = vp.y * wup * 0.5;

    // Rotation-safe coverage: use half-diagonal so rotated grids cover the full screen.
    let half_diag = (half_w_world * half_w_world + half_h_world * half_h_world).sqrt();

    let cam_center = cam_xform.translation().truncate();

    for (grid, rt) in q_grids.iter() {
        if !grid.enabled {
            continue;
        }

        let bx = safe_normalize_or(grid.basis_x, Vec2::X);
        let by = safe_normalize_or(grid.basis_y, Vec2::Y);

        let minor_step = rt.minor_step_world.max(0.000001);
        let major_every_i32 = grid.major_every.max(1) as i32;

        // Camera center in grid-local coordinates (dot with basis)
        let d = cam_center - grid.origin;
        let cam_local = Vec2::new(d.dot(bx), d.dot(by));

        // Local extents for line endpoints (rotation-safe)
        let half_x_local = half_diag;
        let half_y_local = half_diag;

        let mut half_lines_x = (half_x_local / minor_step).ceil() as u32 + 2;
        let mut half_lines_y = (half_y_local / minor_step).ceil() as u32 + 2;
        half_lines_x = half_lines_x.min(grid.max_half_lines);
        half_lines_y = half_lines_y.min(grid.max_half_lines);

        let kx0 = fast_floor_div(cam_local.x, minor_step);
        let ky0 = fast_floor_div(cam_local.y, minor_step);

        let x_min = cam_local.x - half_x_local;
        let x_max = cam_local.x + half_x_local;
        let y_min = cam_local.y - half_y_local;
        let y_max = cam_local.y + half_y_local;

        // Vertical lines: local x = k * step
        for i in 0..=half_lines_x {
            for sign in [-1i32, 1i32] {
                if i == 0 && sign == -1 {
                    continue;
                }
                let k = kx0 + sign * i as i32;
                let x = k as f32 * minor_step;

                let is_axis = k == 0;
                let is_major = k.rem_euclid(major_every_i32) == 0;

                if is_axis && !grid.draw_axes {
                    continue;
                }

                let color = if is_axis && grid.draw_axes {
                    grid.axis_y_color // x=0 (local Y axis)
                } else if is_major {
                    grid.major_color
                } else {
                    grid.minor_color
                };

                let a_world = local_to_world(grid.origin, bx, by, Vec2::new(x, y_min), grid.z_layer);
                let b_world = local_to_world(grid.origin, bx, by, Vec2::new(x, y_max), grid.z_layer);
                gizmos.line(a_world, b_world, color);
            }
        }

        // Horizontal lines: local y = k * step
        for i in 0..=half_lines_y {
            for sign in [-1i32, 1i32] {
                if i == 0 && sign == -1 {
                    continue;
                }
                let k = ky0 + sign * i as i32;
                let y = k as f32 * minor_step;

                let is_axis = k == 0;
                let is_major = k.rem_euclid(major_every_i32) == 0;

                if is_axis && !grid.draw_axes {
                    continue;
                }

                let color = if is_axis && grid.draw_axes {
                    grid.axis_x_color // y=0 (local X axis)
                } else if is_major {
                    grid.major_color
                } else {
                    grid.minor_color
                };

                let a_world = local_to_world(grid.origin, bx, by, Vec2::new(x_min, y), grid.z_layer);
                let b_world = local_to_world(grid.origin, bx, by, Vec2::new(x_max, y), grid.z_layer);
                gizmos.line(a_world, b_world, color);
            }
        }
    }
}

fn local_to_world(origin: Vec2, bx: Vec2, by: Vec2, local: Vec2, z: f32) -> Vec3 {
    let w = origin + bx * local.x + by * local.y;
    Vec3::new(w.x, w.y, z)
}

fn safe_normalize_or(v: Vec2, fallback: Vec2) -> Vec2 {
    let len2 = v.length_squared();
    if len2 > 1e-12 && len2.is_finite() {
        v / len2.sqrt()
    } else {
        fallback
    }
}

fn nice_step(x: f32) -> f32 {
    if !x.is_finite() || x <= 0.0 {
        return 1.0;
    }
    let exp = x.abs().log10().floor();
    let base = 10.0_f32.powf(exp);
    let m = x / base;

    let nice_m = if m <= 1.0 {
        1.0
    } else if m <= 2.0 {
        2.0
    } else if m <= 5.0 {
        5.0
    } else {
        10.0
    };

    nice_m * base
}

fn fast_floor_div(value: f32, step: f32) -> i32 {
    if step <= 0.0 {
        return 0;
    }
    (value / step).floor() as i32
}
