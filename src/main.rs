use bevy::prelude::*;
use std::time::Duration;

mod render;
mod sim;

use render::PressureRenderPlugin;
use sim::PressureSimPlugin;

fn main() {
    // Pick a reasonable sim rate (e.g. 200 Hz). Change as needed.
    let sim_hz = 60.0;

    App::new()
        .insert_resource(Time::<Fixed>::from_duration(Duration::from_secs_f64(
            1.0 / sim_hz,
        )))
        .add_plugins(DefaultPlugins)
        .add_plugins((PressureSimPlugin, PressureRenderPlugin))
        .run();
}

// use bevy::prelude::*;
//
// mod coords_overlay;
//
// use coords_overlay::CoordOverlayPlugin;
//
// fn main() {
//     App::new()
//         .add_plugins(DefaultPlugins)
//         .add_plugins(CoordOverlayPlugin)
//         //.add_systems(Startup, setup)
//         .add_systems(Update, camera_pan_zoom)
//         .run();
// }
//
// fn setup(mut commands: Commands) {
//     commands.spawn(Camera2d);
// }
//
//
// fn camera_pan_zoom(
//     time: Res<Time>,
//     keys: Res<ButtonInput<KeyCode>>,
//     mut q_cam: Query<(&mut Transform, &mut Projection), With<Camera2d>>,
// ) {
//     let Ok((mut transform, mut projection)) = q_cam.single_mut() else {
//         return;
//     };
//
//     let ortho = match &mut *projection {
//         Projection::Orthographic(o) => o,
//         _ => return,
//     };
//
//     let zoom = ortho.scale.max(0.0001);
//     let pan_speed = 700.0 * zoom;
//
//     let mut pan = Vec2::ZERO;
//     if keys.pressed(KeyCode::KeyW) || keys.pressed(KeyCode::ArrowUp) {
//         pan.y += 1.0;
//     }
//     if keys.pressed(KeyCode::KeyS) || keys.pressed(KeyCode::ArrowDown) {
//         pan.y -= 1.0;
//     }
//     if keys.pressed(KeyCode::KeyA) || keys.pressed(KeyCode::ArrowLeft) {
//         pan.x -= 1.0;
//     }
//     if keys.pressed(KeyCode::KeyD) || keys.pressed(KeyCode::ArrowRight) {
//         pan.x += 1.0;
//     }
//     if pan.length_squared() > 0.0 {
//         pan = pan.normalize();
//         transform.translation.x += pan.x * pan_speed * time.delta_secs();
//         transform.translation.y += pan.y * pan_speed * time.delta_secs();
//     }
//
//     let zoom_rate = 1.25_f32;
//     if keys.pressed(KeyCode::KeyZ) {
//         ortho.scale = (ortho.scale / zoom_rate).clamp(0.02, 1000.0);
//     }
//     if keys.pressed(KeyCode::KeyX) {
//         ortho.scale = (ortho.scale * zoom_rate).clamp(0.02, 1000.0);
//     }
//     if keys.just_pressed(KeyCode::KeyR) {
//         ortho.scale = 1.0;
//         transform.translation = Vec3::new(0.0, 0.0, transform.translation.z);
//     }
// }