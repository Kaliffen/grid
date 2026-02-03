#import bevy_sprite::mesh2d_vertex_output::VertexOutput

struct GridStyle {
    half_extent: vec2<f32>,
    axis_color_x: vec4<f32>,
    axis_color_y: vec4<f32>,
    major_color: vec4<f32>,
    minor_color: vec4<f32>,
    thickness_px: vec3<f32>,
    opacities: vec3<f32>,
    flags: u32,
    _padding: vec3<f32>,
}

struct GridGlobals {
    ppu: f32,
    major_step_world: f32,
    minor_step_world: f32,
    alpha_minor_global: f32,
}

@group(2) @binding(0)
var<uniform> style: GridStyle;

@group(2) @binding(1)
var<uniform> globals: GridGlobals;

const SHOW_GRID: u32 = 1u;
const SHOW_AXES: u32 = 2u;
const AXES_ONLY: u32 = 4u;

fn coverage(dist: f32, thickness: f32) -> f32 {
    let aa = fwidth(dist);
    return 1.0 - smoothstep(thickness - aa, thickness + aa, dist);
}

fn layer_color(color: vec4<f32>, opacity: f32, cov: f32) -> vec4<f32> {
    let alpha = color.a * opacity * cov;
    return vec4(color.rgb * alpha, alpha);
}

fn blend_over(base: vec4<f32>, layer: vec4<f32>) -> vec4<f32> {
    return layer + base * (1.0 - layer.a);
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let local = (in.uv * 2.0 - vec2<f32>(1.0, 1.0)) * style.half_extent;

    let axis_th_w = style.thickness_px.x / globals.ppu;
    let major_th_w = style.thickness_px.y / globals.ppu;
    let minor_th_w = style.thickness_px.z / globals.ppu;

    let step = globals.major_step_world;
    let sub = globals.minor_step_world;

    let dist_axis_x = abs(local.y);
    let dist_axis_y = abs(local.x);

    let dx_major = abs(fract(local.x / step) - 0.5) * step;
    let dy_major = abs(fract(local.y / step) - 0.5) * step;
    let dist_major = min(dx_major, dy_major);

    let dx_minor = abs(fract(local.x / sub) - 0.5) * sub;
    let dy_minor = abs(fract(local.y / sub) - 0.5) * sub;
    let dist_minor = min(dx_minor, dy_minor);

    let show_grid = (style.flags & SHOW_GRID) != 0u && (style.flags & AXES_ONLY) == 0u;
    let show_axes = (style.flags & SHOW_AXES) != 0u;

    var color = vec4<f32>(0.0, 0.0, 0.0, 0.0);

    if show_grid {
        let cov_minor = coverage(dist_minor, minor_th_w);
        let cov_major = coverage(dist_major, major_th_w);
        let minor_layer = layer_color(style.minor_color, style.opacities.z * globals.alpha_minor_global, cov_minor);
        let major_layer = layer_color(style.major_color, style.opacities.y, cov_major);
        color = blend_over(color, minor_layer);
        color = blend_over(color, major_layer);
    }

    if show_axes {
        let cov_axis_x = coverage(dist_axis_x, axis_th_w);
        let cov_axis_y = coverage(dist_axis_y, axis_th_w);
        let axis_layer_x = layer_color(style.axis_color_x, style.opacities.x, cov_axis_x);
        let axis_layer_y = layer_color(style.axis_color_y, style.opacities.x, cov_axis_y);
        color = blend_over(color, axis_layer_x);
        color = blend_over(color, axis_layer_y);
    }

    return color;
}
