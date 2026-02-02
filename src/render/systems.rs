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
    commands.insert_resource(RenderDiag::default());
}

// ------------------------------------
// UI: Update (vsync)
// ------------------------------------

pub(crate) fn update_ui(
    gases: Res<GasCatalog>,
    presented: Res<PressurePresented>,
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
            "tick: {}\ncenter total moles (pressure proxy): {:.6}\ncenter mix: {}",
            presented.tick, presented.center_total_moles, comp
        ));
    }
}

// ------------------------------------
// Heatmap rendering: Update
// ------------------------------------

pub(crate) fn update_heatmap_texture(
    presented: Res<PressurePresented>,
    mut heatmap: ResMut<HeatmapViz>,
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
    let data = &mut image.data;

    for y in 0..height {
        for x in 0..width {
            let cell_i = y * width + x;
            let pressure = presented.pressure[cell_i];
            let mut t = (pressure - min_p) * inv_range;
            t = t.clamp(0.0, 1.0);

            let low = Vec3::new(0.05, 0.05, 0.3);
            let high = Vec3::new(1.0, 0.2, 0.1);
            let rgb = low.lerp(high, t);

            let base = cell_i * 4;
            let data = data.get_or_insert_with(Vec::new);

            // ensure it is large enough (example; pick correct size)
            let needed = base + 4;
            if data.len() < needed {
                data.resize(needed, 0);
            }

            data[base] = (rgb.x * 255.0) as u8;
            data[base + 1] = (rgb.y * 255.0) as u8;
            data[base + 2] = (rgb.z * 255.0) as u8;
            data[base + 3] = 255;
        }
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
