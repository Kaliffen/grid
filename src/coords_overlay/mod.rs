use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, ShaderType};
use bevy::sprite_render::{AlphaMode2d, Material2d, Material2dPlugin};
use bevy::sprite_render::{Mesh2d, MeshMaterial2d};
use bevy::reflect::TypePath;

const GRID_SHADER_PATH: &str = "shaders/coords_overlay.wgsl";

pub struct CoordOverlayPlugin;

impl Plugin for CoordOverlayPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<GridBehaviorParams>()
            .init_resource::<GridGlobals>()
            .add_plugins(Material2dPlugin::<CoordGridMaterial>::default())
            .add_systems(Startup, setup)
            .add_systems(Update, (update_grid_globals, sync_materials));
    }
}

#[derive(Component, Clone, Copy)]
pub struct CoordFrame2D {
    pub half_extent_local: Vec2,
    pub style: GridStyle,
    pub flags: u32,
}

#[derive(Clone, Copy)]
pub struct GridStyle {
    pub axis_color_x: Vec4,
    pub axis_color_y: Vec4,
    pub major_color: Vec4,
    pub minor_color: Vec4,
    pub axis_thickness_px: f32,
    pub major_thickness_px: f32,
    pub minor_thickness_px: f32,
    pub axis_opacity: f32,
    pub major_opacity: f32,
    pub minor_opacity: f32,
}

impl Default for GridStyle {
    fn default() -> Self {
        Self {
            axis_color_x: Vec4::new(0.9, 0.2, 0.2, 1.0),
            axis_color_y: Vec4::new(0.2, 0.6, 0.95, 1.0),
            major_color: Vec4::new(0.72, 0.72, 0.78, 1.0),
            minor_color: Vec4::new(0.62, 0.62, 0.68, 1.0),
            axis_thickness_px: 2.6,
            major_thickness_px: 1.2,
            minor_thickness_px: 0.8,
            axis_opacity: 1.0,
            major_opacity: 0.55,
            minor_opacity: 0.35,
        }
    }
}

pub mod flags {
    pub const SHOW_GRID: u32 = 1 << 0;
    pub const SHOW_AXES: u32 = 1 << 1;
    pub const AXES_ONLY: u32 = 1 << 2;
    pub const SELECTED: u32 = 1 << 3;
}

#[derive(Resource, Clone, Copy)]
pub struct GridBehaviorParams {
    pub target_major_px: f32,
    pub major_band_min_px: f32,
    pub major_band_max_px: f32,
    pub minor_subdivisions: f32,
    pub minor_fade_out_px: f32,
    pub minor_fade_in_px: f32,
}

impl Default for GridBehaviorParams {
    fn default() -> Self {
        Self {
            target_major_px: 100.0,
            major_band_min_px: 80.0,
            major_band_max_px: 140.0,
            minor_subdivisions: 5.0,
            minor_fade_out_px: 4.0,
            minor_fade_in_px: 12.0,
        }
    }
}

#[derive(Resource, Clone, Copy, Default)]
pub struct GridGlobals {
    pub ppu: f32,
    pub major_step_world: f32,
    pub minor_step_world: f32,
    pub alpha_minor_global: f32,
}

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct CoordGridMaterial {
    #[uniform(0)]
    pub style: GridStyleUniform,
    #[uniform(1)]
    pub globals: GridGlobalsUniform,
}

impl Material2d for CoordGridMaterial {
    fn fragment_shader() -> bevy::shader::ShaderRef {
        GRID_SHADER_PATH.into()
    }

    fn alpha_mode(&self) -> AlphaMode2d {
        AlphaMode2d::Premultiplied
    }
}

#[derive(Clone, Copy, Debug, ShaderType)]
pub struct GridStyleUniform {
    pub half_extent: Vec2,
    pub axis_color_x: Vec4,
    pub axis_color_y: Vec4,
    pub major_color: Vec4,
    pub minor_color: Vec4,
    pub thickness_px: Vec3,
    pub opacities: Vec3,
    pub flags: u32,
    pub _padding: Vec3,
}

#[derive(Clone, Copy, Debug, ShaderType)]
pub struct GridGlobalsUniform {
    pub ppu: f32,
    pub major_step_world: f32,
    pub minor_step_world: f32,
    pub alpha_minor_global: f32,
}

impl CoordGridMaterial {
    fn from_frame(frame: &CoordFrame2D, globals: GridGlobals) -> Self {
        Self {
            style: GridStyleUniform {
                half_extent: frame.half_extent_local,
                axis_color_x: frame.style.axis_color_x,
                axis_color_y: frame.style.axis_color_y,
                major_color: frame.style.major_color,
                minor_color: frame.style.minor_color,
                thickness_px: Vec3::new(
                    frame.style.axis_thickness_px,
                    frame.style.major_thickness_px,
                    frame.style.minor_thickness_px,
                ),
                opacities: Vec3::new(
                    frame.style.axis_opacity,
                    frame.style.major_opacity,
                    frame.style.minor_opacity,
                ),
                flags: frame.flags,
                _padding: Vec3::ZERO,
            },
            globals: GridGlobalsUniform {
                ppu: globals.ppu,
                major_step_world: globals.major_step_world,
                minor_step_world: globals.minor_step_world,
                alpha_minor_global: globals.alpha_minor_global,
            },
        }
    }

    fn sync_globals(&mut self, globals: GridGlobals) {
        self.globals = GridGlobalsUniform {
            ppu: globals.ppu,
            major_step_world: globals.major_step_world,
            minor_step_world: globals.minor_step_world,
            alpha_minor_global: globals.alpha_minor_global,
        };
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<CoordGridMaterial>>,
) {
    commands.spawn(Camera2d);

    let quad = meshes.add(Rectangle::new(2.0, 2.0));

    let mut spawn_frame = |position: Vec3, rotation: f32, half_extent: Vec2, style: GridStyle| {
        let frame = CoordFrame2D {
            half_extent_local: half_extent,
            style,
            flags: flags::SHOW_GRID | flags::SHOW_AXES,
        };

        let material = materials.add(CoordGridMaterial::from_frame(&frame, GridGlobals::default()));

        commands.spawn((
            frame,
            Mesh2d(quad.clone()),
            MeshMaterial2d(material),
            Transform::from_translation(position)
                .with_rotation(Quat::from_rotation_z(rotation))
                .with_scale(Vec3::new(half_extent.x, half_extent.y, 1.0)),
        ));
    };

    spawn_frame(Vec3::new(-220.0, 0.0, 0.0), 0.2, Vec2::new(220.0, 160.0), GridStyle::default());
    spawn_frame(
        Vec3::new(240.0, 80.0, 0.0),
        -0.6,
        Vec2::new(160.0, 120.0),
        GridStyle {
            axis_color_x: Vec4::new(0.95, 0.35, 0.2, 1.0),
            axis_color_y: Vec4::new(0.2, 0.9, 0.5, 1.0),
            major_color: Vec4::new(0.7, 0.8, 0.9, 1.0),
            minor_color: Vec4::new(0.55, 0.65, 0.75, 1.0),
            axis_thickness_px: 2.3,
            major_thickness_px: 1.1,
            minor_thickness_px: 0.7,
            axis_opacity: 1.0,
            major_opacity: 0.45,
            minor_opacity: 0.3,
        },
    );
}

fn update_grid_globals(
    mut globals: ResMut<GridGlobals>,
    params: Res<GridBehaviorParams>,
    cameras: Query<(&Camera, &GlobalTransform), With<Camera2d>>,
) {
    let Ok((camera, camera_transform)) = cameras.get_single() else {
        return;
    };

    let Some(viewport_size) = camera.logical_viewport_size() else {
        return;
    };

    let center = viewport_size * 0.5;
    let Ok(center_world) = camera.viewport_to_world_2d(camera_transform, center) else {
        return;
    };
    let Ok(right_world) =
        camera.viewport_to_world_2d(camera_transform, center + Vec2::new(1.0, 0.0))
    else {
        return;
    };

    let world_per_px = (right_world - center_world).length().max(f32::EPSILON);
    let ppu = 1.0 / world_per_px;

    let desired_major = params.target_major_px / ppu;
    let mut step = closest_nice_step(desired_major);

    let mut step_px = step * ppu;
    while step_px < params.major_band_min_px {
        step = next_larger_nice(step);
        step_px = step * ppu;
    }
    while step_px > params.major_band_max_px {
        step = next_smaller_nice(step);
        step_px = step * ppu;
    }

    let substep = step / params.minor_subdivisions;
    let substep_px = substep * ppu;

    let alpha_minor_global = if substep_px <= params.minor_fade_out_px {
        0.0
    } else if substep_px >= params.minor_fade_in_px {
        1.0
    } else {
        (substep_px - params.minor_fade_out_px)
            / (params.minor_fade_in_px - params.minor_fade_out_px)
    };

    *globals = GridGlobals {
        ppu,
        major_step_world: step,
        minor_step_world: substep,
        alpha_minor_global,
    };
}

fn sync_materials(
    globals: Res<GridGlobals>,
    frames: Query<(&CoordFrame2D, &MeshMaterial2d<CoordGridMaterial>)>,
    mut materials: ResMut<Assets<CoordGridMaterial>>,
) {
    for (frame, material_handle) in &frames {
        if let Some(material) = materials.get_mut(&material_handle.0) {
            material.style.half_extent = frame.half_extent_local;
            material.style.axis_color_x = frame.style.axis_color_x;
            material.style.axis_color_y = frame.style.axis_color_y;
            material.style.major_color = frame.style.major_color;
            material.style.minor_color = frame.style.minor_color;
            material.style.thickness_px = Vec3::new(
                frame.style.axis_thickness_px,
                frame.style.major_thickness_px,
                frame.style.minor_thickness_px,
            );
            material.style.opacities = Vec3::new(
                frame.style.axis_opacity,
                frame.style.major_opacity,
                frame.style.minor_opacity,
            );
            material.style.flags = frame.flags;
            material.sync_globals(*globals);
        }
    }
}

fn closest_nice_step(desired: f32) -> f32 {
    if desired <= 0.0 {
        return 1.0;
    }
    let exp = desired.log10().floor() as i32;
    let base = 10_f32.powi(exp);
    let candidates = [1.0, 2.0, 5.0]
        .into_iter()
        .flat_map(|m| [m * base, m * base * 10.0]);

    candidates
        .min_by(|a, b| {
            let da = (a.ln() - desired.ln()).abs();
            let db = (b.ln() - desired.ln()).abs();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(1.0)
}

fn next_larger_nice(value: f32) -> f32 {
    let (mantissa, exp) = decompose(value);
    if mantissa < 1.0 {
        return 1.0 * 10_f32.powi(exp);
    }
    if mantissa < 2.0 {
        return 2.0 * 10_f32.powi(exp);
    }
    if mantissa < 5.0 {
        return 5.0 * 10_f32.powi(exp);
    }
    1.0 * 10_f32.powi(exp + 1)
}

fn next_smaller_nice(value: f32) -> f32 {
    let (mantissa, exp) = decompose(value);
    if mantissa > 5.0 {
        return 5.0 * 10_f32.powi(exp);
    }
    if mantissa > 2.0 {
        return 2.0 * 10_f32.powi(exp);
    }
    if mantissa > 1.0 {
        return 1.0 * 10_f32.powi(exp);
    }
    5.0 * 10_f32.powi(exp - 1)
}

fn decompose(value: f32) -> (f32, i32) {
    if value <= 0.0 {
        return (1.0, 0);
    }
    let exp = value.log10().floor() as i32;
    let mantissa = value / 10_f32.powi(exp);
    (mantissa, exp)
}
