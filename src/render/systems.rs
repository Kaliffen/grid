use bevy::asset::RenderAssetUsages;
use bevy::image::ImageSampler;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

use bevy::window::PrimaryWindow;

use crate::sim::resources::{GasCatalog, GridSettings, PressurePresented, SelectedCell};

// ------------------------------------
// Diagnostics (Render)
// ------------------------------------

#[derive(Resource, Default)]
pub(crate) struct RenderDiag {
    wall_accum: f64,
    frames: u64,
}

// ------------------------------------
// Heatmap rendering
// ------------------------------------

#[derive(Resource)]
pub(crate) struct HeatmapViz {
    image: Handle<Image>,
    width: u32,
    height: u32,
    min_pressure: f32,
    max_pressure: f32,
    last_tick: u64,
}

#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OverlayMode {
    Pressure,
    Wind,
    Both,
}

impl OverlayMode {
    fn next(self) -> Self {
        match self {
            OverlayMode::Pressure => OverlayMode::Wind,
            OverlayMode::Wind => OverlayMode::Both,
            OverlayMode::Both => OverlayMode::Pressure,
        }
    }

    fn label(self) -> &'static str {
        match self {
            OverlayMode::Pressure => "pressure",
            OverlayMode::Wind => "wind",
            OverlayMode::Both => "both",
        }
    }
}

// ------------------------------------
// UI
// ------------------------------------

#[derive(Component)]
pub(crate) struct VizLabel;

#[derive(Component)]
pub(crate) struct HeatmapSprite;

#[derive(Default)]
struct UiState {
    last_tick: u64,
    last_selected: SelectedCell,
}

// ------------------------------------
// Startup: Render setup
// ------------------------------------

pub(crate) fn setup_render(
    mut commands: Commands,
    grid: Res<GridSettings>,
    mut images: ResMut<Assets<Image>>,
) {
    // Camera + UI
    commands.spawn(Camera2d);

    commands.spawn((
        Text::new("booting…"),
        TextFont {
            font_size: 18.0,
            ..default()
        },
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(12.0),
            top: Val::Px(12.0),
            ..default()
        },
        VizLabel,
    ));

    let max_dim = grid.width.max(grid.height) as f32;
    let target_screen_size = 512.0_f32;
    let scale = (target_screen_size / max_dim).clamp(0.1, 16.0);

    let mut image = Image::new_fill(
        Extent3d {
            width: grid.width,
            height: grid.height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 255],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    image.sampler = ImageSampler::nearest();
    let image_handle = images.add(image);

    commands.spawn((
        Sprite::from_image(image_handle.clone()),
        Transform::from_scale(Vec3::splat(scale)),
        HeatmapSprite,
    ));

    commands.insert_resource(HeatmapViz {
        image: image_handle,
        width: grid.width,
        height: grid.height,
        min_pressure: 0.0,
        max_pressure: 20.0,
        last_tick: 0,
    });
    commands.insert_resource(OverlayMode::Pressure);
    commands.insert_resource(RenderDiag::default());
}

// ------------------------------------
// UI: Update (vsync)
// ------------------------------------

pub(crate) fn update_ui(
    gases: Res<GasCatalog>,
    presented: Res<PressurePresented>,
    overlay: Res<OverlayMode>,
    selected: Res<SelectedCell>,
    mut q: Query<&mut Text, With<VizLabel>>,
    mut ui_state: Local<UiState>,
) {
    // Update UI when the sim tick changes (vsync may show repeated frames)
    if presented.tick == ui_state.last_tick && *selected == ui_state.last_selected {
        return;
    }
    ui_state.last_tick = presented.tick;
    ui_state.last_selected = *selected;

    // Build a short composition string for the selected cell: show top 3 gases by fraction
    let mut indices: Vec<usize> = (0..gases.gases.len()).collect();
    indices.sort_by(|&a, &b| {
        presented.selected_gas_fractions[b]
            .partial_cmp(&presented.selected_gas_fractions[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let top_n = indices.len().min(3);
    let mut comp = String::new();
    for i in 0..top_n {
        let gi = indices[i];
        let frac = presented.selected_gas_fractions[gi];
        if frac <= 0.0 {
            continue;
        }
        let name = gases.gases[gi].name;
        let pct = frac * 100.0;
        if !comp.is_empty() {
            comp.push_str(", ");
        }
        comp.push_str(&format!("{}: {:.1}%", name, pct));
    }
    if comp.is_empty() {
        comp.push_str("none");
    }

    if let Ok(mut text) = q.single_mut() {
        let cell_i = (selected.y * presented.width + selected.x) as usize;
        let wind = presented
            .wind
            .get(cell_i)
            .copied()
            .unwrap_or(Vec2::ZERO);
        let wind_speed = wind.length();
        *text = Text::new(format!(
            "tick: {}\nselected cell: ({}, {})\nselected total moles (pressure proxy): {:.6}\nselected mix: {}\nwind (m/s): [{:.3}, {:.3}] |v|={:.3}\noverlay: {} (space cycle, 1 pressure, 2 wind, 3 both)",
            presented.tick,
            selected.x,
            selected.y,
            presented.selected_total_moles,
            comp,
            wind.x,
            wind.y,
            wind_speed,
            overlay.label(),
        ));
    }
}

// ------------------------------------
// Heatmap rendering: Update
// ------------------------------------

pub(crate) fn update_overlay_mode_input(
    keys: Res<ButtonInput<KeyCode>>,
    mut overlay: ResMut<OverlayMode>,
) {
    if keys.just_pressed(KeyCode::Space) {
        *overlay = overlay.next();
    }
    if keys.just_pressed(KeyCode::Digit1) {
        *overlay = OverlayMode::Pressure;
    }
    if keys.just_pressed(KeyCode::Digit2) {
        *overlay = OverlayMode::Wind;
    }
    if keys.just_pressed(KeyCode::Digit3) {
        *overlay = OverlayMode::Both;
    }
}

pub(crate) fn update_selected_tile_input(
    buttons: Res<ButtonInput<MouseButton>>,
    window_query: Query<&Window, With<PrimaryWindow>>,
    camera_query: Query<(&Camera, &GlobalTransform)>,
    heatmap_query: Query<&GlobalTransform, With<HeatmapSprite>>,
    grid: Res<GridSettings>,
    mut selected: ResMut<SelectedCell>,
) {
    if !buttons.just_pressed(MouseButton::Left) {
        return;
    }

    let Ok(window) = window_query.single() else {
        return;
    };
    let Some(cursor_pos) = window.cursor_position() else {
        return;
    };

    let Ok((camera, camera_transform)) = camera_query.single() else {
        return;
    };
    let Ok(heatmap_transform) = heatmap_query.single() else {
        return;
    };

    let Ok(world_pos) = camera.viewport_to_world_2d(camera_transform, cursor_pos) else {
        return;
    };

    let local = heatmap_transform
        .affine()
        .inverse()
        .transform_point3(Vec3::new(world_pos.x, world_pos.y, 0.0));

    let half_width = grid.width as f32 * 0.5;
    let half_height = grid.height as f32 * 0.5;
    let pixel_x = local.x + half_width;
    let pixel_y = half_height - local.y;

    if pixel_x < 0.0
        || pixel_y < 0.0
        || pixel_x >= grid.width as f32
        || pixel_y >= grid.height as f32
    {
        return;
    }

    let x = pixel_x.floor() as u32;
    let y = pixel_y.floor() as u32;
    *selected = SelectedCell { x, y };
}

pub(crate) fn update_heatmap_texture(
    presented: Res<PressurePresented>,
    mut heatmap: ResMut<HeatmapViz>,
    overlay: Res<OverlayMode>,
    grid: Res<GridSettings>,
    mut images: ResMut<Assets<Image>>,
) {
    if presented.tick == heatmap.last_tick {
        return;
    }
    heatmap.last_tick = presented.tick;

    let Some(image) = images.get_mut(&heatmap.image) else {
        return;
    };

    let width = heatmap.width as usize;
    let height = heatmap.height as usize;
    let expected_len = width * height * 4;
    let data = image.data.get_or_insert_with(|| vec![0; expected_len]);
    if data.len() != expected_len {
        data.resize(expected_len, 0);
    }

    let wind_floor = grid.wind_visual_min_speed.max(0.0);
    let wind_scale = presented
        .max_wind_speed
        .max(wind_floor + f32::EPSILON);

    for y in 0..height {
        for x in 0..width {
            let cell_i = y * width + x;
            let pressure = presented.pressure[cell_i];

            let pressure_color = pressure_ramp_atm(pressure);
            let wind = presented.wind[cell_i];
            let wind_color = wind_ramp(wind, wind_floor, wind_scale);

            let rgb = match *overlay {
                OverlayMode::Pressure => pressure_color,
                OverlayMode::Wind => wind_color,
                OverlayMode::Both => pressure_color.lerp(wind_color, 0.5),
            };

            let rgb = linear_to_srgb(rgb);

            let base = cell_i * 4;
            data[base] = (rgb.x * 255.0) as u8;
            data[base + 1] = (rgb.y * 255.0) as u8;
            data[base + 2] = (rgb.z * 255.0) as u8;
            data[base + 3] = 255;
        }
    }
}

const ATM_PA: f32 = 101_325.0;

#[inline]
fn pressure_ramp_atm(pressure_pa: f32) -> Vec3 {
    let atm = (pressure_pa / ATM_PA).max(0.0);

    let vacuum = Vec3::new(0.0, 0.0, 0.0);
    let low = Vec3::new(0.05, 0.1, 0.6);
    let normal = Vec3::new(0.1, 0.8, 0.2);
    let high = Vec3::new(0.95, 0.5, 0.1);
    let danger = Vec3::new(0.9, 0.05, 0.05);

    let t = |value: f32, start: f32, end: f32| ((value - start) / (end - start)).clamp(0.0, 1.0);

    if atm <= 0.05 {
        vacuum.lerp(low, t(atm, 0.0, 0.05))
    } else if atm <= 1.0 {
        low.lerp(normal, t(atm, 0.05, 1.0))
    } else if atm <= 3.0 {
        normal.lerp(high, t(atm, 1.0, 3.0))
    } else if atm <= 10.0 {
        high.lerp(danger, t(atm, 3.0, 10.0))
    } else {
        danger
    }
}

#[inline]
fn wind_ramp(wind: Vec2, floor: f32, scale: f32) -> Vec3 {
    let mag = wind.length();
    if mag <= floor {
        return Vec3::ZERO;
    }

    let denom = (scale - floor).max(f32::EPSILON);
    let normalized = ((mag - floor) / denom).clamp(0.0, 1.0);
    if normalized <= 0.0 {
        return Vec3::ZERO;
    }

    let angle = wind.y.atan2(wind.x);
    let hue = ((angle / std::f32::consts::TAU) + 1.0) % 1.0;
    let sat = 0.85;
    let val = 0.1 + 0.9 * normalized;
    hsv_to_rgb(hue, sat, val)
}

#[inline]
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> Vec3 {
    let h = (h * 6.0).rem_euclid(6.0);
    let i = h.floor();
    let f = h - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));

    match i as i32 {
        0 => Vec3::new(v, t, p),
        1 => Vec3::new(q, v, p),
        2 => Vec3::new(p, v, t),
        3 => Vec3::new(p, q, v),
        4 => Vec3::new(t, p, v),
        _ => Vec3::new(v, p, q),
    }
}

#[inline]
fn linear_to_srgb(rgb: Vec3) -> Vec3 {
    Vec3::new(
        linear_channel_to_srgb(rgb.x),
        linear_channel_to_srgb(rgb.y),
        linear_channel_to_srgb(rgb.z),
    )
}

#[inline]
fn linear_channel_to_srgb(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    if x <= 0.003_130_8 {
        12.92 * x
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    }
}

// ------------------------------------
// Diagnostics (requested format)
// ------------------------------------

pub(crate) fn render_diagnostics(time: Res<Time>, mut diag: ResMut<RenderDiag>) {
    diag.wall_accum += time.delta_secs_f64();
    diag.frames += 1;

    if diag.wall_accum >= 1.0 {
        let actual_dt = diag.wall_accum / (diag.frames as f64);
        let fps = 1.0 / actual_dt;

        info!(
            "Render: actual dt={:.6}s, configured≈vsync (≈60 on this system), {:.1} FPS",
            actual_dt, fps
        );

        diag.wall_accum = 0.0;
        diag.frames = 0;
    }
}
