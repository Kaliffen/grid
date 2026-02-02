use bevy::prelude::*;
use std::time::Instant;

// ------------------------------------
// Gas model
// ------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GasKind {
    Oxygen,
    Nitrogen,
    CarbonDioxide,
    Methane,
    Hydrogen,
    Chlorine,
    Steam,
}

#[derive(Clone, Debug)]
pub struct GasDef {
    pub kind: GasKind,
    pub name: &'static str,

    // "of breathability, chemical and thermal / explosive interest"
    pub breathable: bool,
    pub chemical_interest: f32,
    pub thermal_interest: f32,
    pub explosive_interest: f32,

    // diffusion coefficient (alpha) for this gas (can differ by gas)
    pub diffusion_alpha: f32,
}

impl GasDef {
    pub fn defaults() -> Vec<GasDef> {
        vec![
            GasDef {
                kind: GasKind::Oxygen,
                name: "O2",
                breathable: true,
                chemical_interest: 0.2,
                thermal_interest: 0.2,
                explosive_interest: 0.4,
                diffusion_alpha: 0.25,
            },
            GasDef {
                kind: GasKind::Nitrogen,
                name: "N2",
                breathable: true,
                chemical_interest: 0.05,
                thermal_interest: 0.2,
                explosive_interest: 0.0,
                diffusion_alpha: 0.25,
            },
            GasDef {
                kind: GasKind::CarbonDioxide,
                name: "CO2",
                breathable: false,
                chemical_interest: 0.1,
                thermal_interest: 0.25,
                explosive_interest: 0.0,
                diffusion_alpha: 0.22,
            },
            GasDef {
                kind: GasKind::Methane,
                name: "CH4",
                breathable: false,
                chemical_interest: 0.6,
                thermal_interest: 0.2,
                explosive_interest: 1.0,
                diffusion_alpha: 0.28,
            },
            GasDef {
                kind: GasKind::Hydrogen,
                name: "H2",
                breathable: false,
                chemical_interest: 0.8,
                thermal_interest: 0.15,
                explosive_interest: 1.0,
                diffusion_alpha: 0.35,
            },
            GasDef {
                kind: GasKind::Chlorine,
                name: "Cl2",
                breathable: false,
                chemical_interest: 0.9,
                thermal_interest: 0.1,
                explosive_interest: 0.2,
                diffusion_alpha: 0.20,
            },
            GasDef {
                kind: GasKind::Steam,
                name: "H2O(g)",
                breathable: false,
                chemical_interest: 0.1,
                thermal_interest: 0.7,
                explosive_interest: 0.0,
                diffusion_alpha: 0.18,
            },
        ]
    }
}

// ------------------------------------
// Grid + sim state (double buffered)
// ------------------------------------

#[derive(Resource, Clone)]
pub struct GridSettings {
    pub width: u32,
    pub height: u32,

    /// Optional global scalar multiplier on diffusion coefficient.
    /// Final alpha used: gas.diffusion_alpha * global_alpha
    pub global_alpha: f32,
}

impl Default for GridSettings {
    fn default() -> Self {
        Self {
            width: 2048,
            height: 2048,
            global_alpha: 1.0,
        }
    }
}

#[derive(Resource, Clone)]
pub struct GasCatalog {
    pub gases: Vec<GasDef>,
}

impl Default for GasCatalog {
    fn default() -> Self {
        Self { gases: GasDef::defaults() }
    }
}

/// Per-cell fields are stored in SoA: `moles[gas][cell]` flattened into one Vec.
/// Layout: index = gas_index * cell_count + cell_index
#[derive(Resource, Clone)]
struct PressureSimState {
    tick: u64,

    width: u32,
    height: u32,
    gas_count: usize,
    cell_count: usize,

    // for interpolation
    prev_moles: Vec<f32>,
    curr_moles: Vec<f32>,

    // step buffer
    next_moles: Vec<f32>,
}

impl PressureSimState {
    fn new(width: u32, height: u32, gas_count: usize) -> Self {
        let cell_count = (width * height) as usize;
        let total_len = gas_count * cell_count;

        Self {
            tick: 0,
            width,
            height,
            gas_count,
            cell_count,
            prev_moles: vec![0.0; total_len],
            curr_moles: vec![0.0; total_len],
            next_moles: vec![0.0; total_len],
        }
    }

    #[inline]
    fn cell_index(&self, x: u32, y: u32) -> usize {
        (y * self.width + x) as usize
    }

    #[inline]
    fn idx(&self, gas_i: usize, cell_i: usize) -> usize {
        gas_i * self.cell_count + cell_i
    }

    #[inline]
    fn read_curr(&self, gas_i: usize, cell_i: usize) -> f32 {
        self.curr_moles[self.idx(gas_i, cell_i)]
    }

    #[inline]
    fn write_next(&mut self, gas_i: usize, cell_i: usize, v: f32) {
        let k = self.idx(gas_i, cell_i);
        self.next_moles[k] = v;
    }

    fn total_moles_at_curr(&self, cell_i: usize) -> f32 {
        let mut sum = 0.0;
        for g in 0..self.gas_count {
            sum += self.read_curr(g, cell_i);
        }
        sum
    }
}

/// Presented/interpolated values. For now, keep it lightweight:
/// - we interpolate a "pressure" field (derived from moles) for visualization
/// - and retain sim tick
#[derive(Resource, Clone)]
struct PressurePresented {
    tick: u64,
    width: u32,
    height: u32,
    pressure: Vec<f32>,
    // optional: also expose a single cell composition for UI (center cell)
    center_total_moles: f32,
    center_gas_fractions: Vec<f32>, // length gas_count
}

impl PressurePresented {
    fn new(width: u32, height: u32, gas_count: usize) -> Self {
        let cell_count = (width * height) as usize;
        Self {
            tick: 0,
            width,
            height,
            pressure: vec![0.0; cell_count],
            center_total_moles: 0.0,
            center_gas_fractions: vec![0.0; gas_count],
        }
    }
}

// ------------------------------------
// Diagnostics (SIM + Render)
// ------------------------------------

#[derive(Resource)]
struct SimDiag {
    last_report: Instant,
    last_step_wall: Instant,
    steps_since_report: u64,
    wall_accum_since_report: f64,
}

impl Default for SimDiag {
    fn default() -> Self {
        let now = Instant::now();
        Self {
            last_report: now,
            last_step_wall: now,
            steps_since_report: 0,
            wall_accum_since_report: 0.0,
        }
    }
}

#[derive(Resource, Default)]
struct RenderDiag {
    wall_accum: f64,
    frames: u64,
}

// ------------------------------------
// UI
// ------------------------------------

#[derive(Component)]
struct VizLabel;

// ------------------------------------
// Plugin
// ------------------------------------

pub struct PressureDiffusionPlugin;

impl Plugin for PressureDiffusionPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(GridSettings::default())
            .insert_resource(GasCatalog::default())
            .add_systems(Startup, setup);

        // Simulation in FixedUpdate; interpolation in AfterFixedMainLoop; UI in Update.
        app.add_systems(FixedUpdate, (advance_diffusion_fixed, sim_diagnostics))
            .add_systems(
                RunFixedMainLoop,
                build_presented_state.in_set(RunFixedMainLoopSystems::AfterFixedMainLoop),
            )
            .add_systems(Update, (update_ui, render_diagnostics));
    }
}

fn setup(
    mut commands: Commands,
    grid: Res<GridSettings>,
    gases: Res<GasCatalog>,
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

    // Sim state resources
    let gas_count = gases.gases.len();
    let mut sim = PressureSimState::new(grid.width, grid.height, gas_count);

    // Seed example: put some oxygen + methane in the center so we can see diffusion.
    let cx = grid.width / 2;
    let cy = grid.height / 2;
    let c = sim.cell_index(cx, cy);

    // Find indices for a couple gases
    let mut o2_i = None;
    let mut ch4_i = None;
    for (i, g) in gases.gases.iter().enumerate() {
        if g.kind == GasKind::Oxygen {
            o2_i = Some(i);
        }
        if g.kind == GasKind::Methane {
            ch4_i = Some(i);
        }
    }

    if let Some(i) = o2_i {
        let k = sim.idx(i, c);
        sim.curr_moles[k] = 10.0;
    }
    if let Some(i) = ch4_i {
        let k = sim.idx(i, c);
        sim.curr_moles[k] = 5.0;
    }

    // Keep prev equal initially (avoid a weird first-frame lerp)
    sim.prev_moles.clone_from(&sim.curr_moles);

    let presented = PressurePresented::new(grid.width, grid.height, gas_count);

    commands.insert_resource(sim);
    commands.insert_resource(presented);
    commands.insert_resource(SimDiag::default());
    commands.insert_resource(RenderDiag::default());
}

// ------------------------------------
// Simulation: FixedUpdate diffusion
// ------------------------------------

fn advance_diffusion_fixed(
    fixed_time: Res<Time<Fixed>>,
    grid: Res<GridSettings>,
    gases: Res<GasCatalog>,
    mut sim: ResMut<PressureSimState>,
) {
    // Copy curr -> prev (for interpolation) without conflicting borrows
    let mut prev_buf = std::mem::take(&mut sim.prev_moles);
    prev_buf.copy_from_slice(&sim.curr_moles);
    sim.prev_moles = prev_buf;

    sim.tick += 1;

    let dt = fixed_time.delta_secs();
    let w = sim.width;
    let h = sim.height;

    // Diffuse each gas independently (mixing is emergent from diffusion of moles).
    for gas_i in 0..sim.gas_count {
        let alpha = gases.gases[gas_i].diffusion_alpha * grid.global_alpha;

        for y in 0..h {
            for x in 0..w {
                let c = sim.cell_index(x, y);

                // 4-neighbor Laplacian with "reflecting" boundaries (Neumann-like)
                let c_val = sim.read_curr(gas_i, c);

                let l = sim.read_curr(gas_i, sim.cell_index(x.saturating_sub(1), y));
                let r = sim.read_curr(gas_i, sim.cell_index((x + 1).min(w - 1), y));
                let u = sim.read_curr(gas_i, sim.cell_index(x, y.saturating_sub(1)));
                let d = sim.read_curr(gas_i, sim.cell_index(x, (y + 1).min(h - 1)));

                let lap = (l + r + u + d) - 4.0 * c_val;
                let mut new_val = c_val + alpha * dt * lap;

                // Keep non-negative moles
                if new_val < 0.0 {
                    new_val = 0.0;
                }

                sim.write_next(gas_i, c, new_val);
            }
        }
    }


    // curr <-> next (double buffer swap)
    let curr = std::mem::take(&mut sim.curr_moles);
    sim.curr_moles = std::mem::replace(&mut sim.next_moles, curr);

    // Optional: clear next (not required, because we fully overwrite it each step)
    // sim.next_moles.fill(0.0);
}

// ------------------------------------
// Interpolation: AfterFixedMainLoop
// ------------------------------------

fn build_presented_state(
    fixed_time: Res<Time<Fixed>>,
    gases: Res<GasCatalog>,
    sim: Res<PressureSimState>,
    mut presented: ResMut<PressurePresented>,
) {
    // Interpolation factor between prev and curr fixed states.
    let alpha = fixed_time.overstep_fraction();

    presented.tick = sim.tick;

    // "Pressure" proxy: total moles (you can later swap to ideal gas law).
    // Interpolate per-gas moles then sum (so pressure field is smooth).
    let cell_count = sim.cell_count;
    for cell_i in 0..cell_count {
        let mut total = 0.0;

        for gas_i in 0..sim.gas_count {
            let k = sim.idx(gas_i, cell_i);
            let prev = sim.prev_moles[k];
            let curr = sim.curr_moles[k];
            let interp = prev.lerp(curr, alpha);
            total += interp;
        }

        presented.pressure[cell_i] = total;
    }

    // Center-cell composition summary for UI
    let cx = sim.width / 2;
    let cy = sim.height / 2;
    let c = sim.cell_index(cx, cy);

    let mut center_total = 0.0;
    for gas_i in 0..sim.gas_count {
        let k = sim.idx(gas_i, c);
        let prev = sim.prev_moles[k];
        let curr = sim.curr_moles[k];
        let interp = prev.lerp(curr, alpha);
        presented.center_gas_fractions[gas_i] = interp; // temporarily store moles
        center_total += interp;
    }

    presented.center_total_moles = center_total;

    if center_total > 0.0 {
        for gas_i in 0..sim.gas_count {
            presented.center_gas_fractions[gas_i] /= center_total;
        }
    } else {
        for gas_i in 0..sim.gas_count {
            presented.center_gas_fractions[gas_i] = 0.0;
        }
    }

    // Keep `gases` in signature so we can later use properties for viz without changing shape.
    let _ = &gases;
}

// ------------------------------------
// UI: Update (vsync)
// ------------------------------------

fn update_ui(
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
            presented.tick,
            presented.center_total_moles,
            comp
        ));
    }
}

// ------------------------------------
// Diagnostics (requested format)
// ------------------------------------

fn sim_diagnostics(fixed_time: Res<Time<Fixed>>, mut diag: ResMut<SimDiag>) {
    let now = Instant::now();
    let wall_dt = now.duration_since(diag.last_step_wall).as_secs_f64();
    diag.last_step_wall = now;

    diag.steps_since_report += 1;
    diag.wall_accum_since_report += wall_dt;

    if now.duration_since(diag.last_report).as_secs_f64() >= 1.0 {
        let actual_dt = if diag.steps_since_report > 0 {
            diag.wall_accum_since_report / (diag.steps_since_report as f64)
        } else {
            0.0
        };

        let configured_dt = fixed_time.delta_secs_f64();
        let achieved = if diag.wall_accum_since_report > 0.0 {
            (diag.steps_since_report as f64) / diag.wall_accum_since_report
        } else {
            0.0
        };

        info!(
            "SIM: actual dt={:.6}s, configured dt={:.6}s, achieved {:.0} steps/sec.",
            actual_dt, configured_dt, achieved
        );

        diag.last_report = now;
        diag.steps_since_report = 0;
        diag.wall_accum_since_report = 0.0;
    }
}

fn render_diagnostics(time: Res<Time>, mut diag: ResMut<RenderDiag>) {
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
