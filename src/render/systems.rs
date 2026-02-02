use bevy::asset::RenderAssetUsages;
use bevy::image::ImageSampler;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

use crate::sim::resources::{GasCatalog, GridSettings, PressurePresented};

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
    mut q: Query<&mut Text, With<VizLabel>>,
    mut last_tick: Local<u64>,
) {
    // Update UI when the sim tick changes (vsync may show repeated frames)
    if presented.tick == *last_tick {
        return;
    }
    *last_tick = presented.tick;

    // Build a short composition string for the center cell: show top 3 gases by fraction
    let mut indices: Vec<usize> = (0..gases.gases.len()).collect();
    indices.sort_by(|&a, &b| {
        presented.center_gas_fractions[b]
            .partial_cmp(&presented.center_gas_fractions[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let top_n = indices.len().min(3);
    let mut comp = String::new();
    for i in 0..top_n {
        let gi = indices[i];
        let frac = presented.center_gas_fractions[gi];
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
        *text = Text::new(format!(
            "tick: {}\ncenter total moles (pressure proxy): {:.6}\ncenter mix: {}\noverlay: {} (space cycle, 1 pressure, 2 wind, 3 both)",
            presented.tick,
            presented.center_total_moles,
            comp,
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

pub(crate) fn update_heatmap_texture(
    presented: Res<PressurePresented>,
    mut heatmap: ResMut<HeatmapViz>,
    overlay: Res<OverlayMode>,
    mut images: ResMut<Assets<Image>>,
) {
    if presented.tick == heatmap.last_tick {
        return;
    }
    heatmap.last_tick = presented.tick;

    let Some(image) = images.get_mut(&heatmap.image) else {
        return;
    };

    let min_p = heatmap.min_pressure;
    let max_p = heatmap.max_pressure.max(min_p + f32::EPSILON);
    let inv_range = 1.0 / (max_p - min_p);

    let width = heatmap.width as usize;
    let height = heatmap.height as usize;
    let expected_len = width * height * 4;
    let data = image
        .data
        .get_or_insert_with(|| vec![0; expected_len]);
    if data.len() != expected_len {
        data.resize(expected_len, 0);
    }

    let max_wind = presented.max_wind_speed.max(f32::EPSILON);
    let inv_wind = 1.0 / max_wind;

    for y in 0..height {
        for x in 0..width {
            let cell_i = y * width + x;
            let pressure = presented.pressure[cell_i];
            let mut t = (pressure - min_p) * inv_range;
            t = t.clamp(0.0, 1.0);

            let pressure_color = pressure_ramp(t);
            let wind = presented.wind[cell_i];
            let wind_color = wind_ramp(wind, inv_wind);

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

#[inline]
fn pressure_ramp(t: f32) -> Vec3 {
    let low = Vec3::new(0.05, 0.05, 0.3);
    let high = Vec3::new(1.0, 0.2, 0.1);
    low.lerp(high, t)
}

#[inline]
fn wind_ramp(wind: Vec2, inv_wind: f32) -> Vec3 {
    let mag = (wind.length() * inv_wind).clamp(0.0, 1.0);
    if mag <= 0.0 {
        return Vec3::ZERO;
    }

    let angle = wind.y.atan2(wind.x);
    let hue = ((angle / std::f32::consts::TAU) + 1.0) % 1.0;
    let sat = 0.85;
    let val = 0.15 + 0.85 * mag;
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
