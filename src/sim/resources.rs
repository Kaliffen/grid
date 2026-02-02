use bevy::prelude::*;

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

    /// Ideal gas law: P = n R T / V (constant per cell for now).
    pub gas_constant: f32,
    pub temperature: f32,
    pub cell_volume: f32,

    /// Optional global scalar multiplier on diffusion coefficient.
    /// Final alpha used: gas.diffusion_alpha * global_alpha
    pub global_alpha: f32,

    /// Bulk-flow permeability constant for pressure-driven advection.
    pub bulk_flow_k: f32,

    /// Under-relaxation factor for bulk-flow flux updates (0 = keep old, 1 = full new).
    pub bulk_flow_relax: f32,

    /// Exponential damping applied to bulk-flow fluxes each step.
    pub bulk_flow_damping: f32,

    /// Presentation-only smoothing for wind vectors (0 = keep old, 1 = full new).
    pub wind_visual_relax: f32,

    /// Presentation-only smoothing for pressure visualization (0 = keep old, 1 = full new).
    pub pressure_visual_relax: f32,

    /// Minimum wind speed used for visualization scaling (prevents tiny noise from saturating).
    pub wind_visual_min_speed: f32,

    /// Max fraction of a cell's total moles that can move across a face per step.
    pub max_flow_fraction: f32,
}

impl Default for GridSettings {
    fn default() -> Self {
        Self {
            width: 64,
            height: 64    ,
            gas_constant: 8.314,
            temperature: 293.15,
            cell_volume: 1.0,
            global_alpha: 1.0,
            bulk_flow_k: 0.4,
            bulk_flow_relax: 0.35,
            bulk_flow_damping: 0.8,
            wind_visual_relax: 0.25,
            pressure_visual_relax: 0.35,
            wind_visual_min_speed: 0.15,
            max_flow_fraction: 0.25,
        }
    }
}

#[derive(Resource, Clone)]
pub struct GasCatalog {
    pub gases: Vec<GasDef>,
}

impl Default for GasCatalog {
    fn default() -> Self {
        Self {
            gases: GasDef::defaults(),
        }
    }
}

/// Per-cell fields are stored in SoA: `moles[gas][cell]` flattened into one Vec.
/// Layout: index = gas_index * cell_count + cell_index
#[derive(Resource, Clone)]
pub(crate) struct PressureSimState {
    pub(crate) tick: u64,

    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) gas_count: usize,
    pub(crate) cell_count: usize,

    // for interpolation
    pub(crate) prev_moles: Vec<f32>,
    pub(crate) curr_moles: Vec<f32>,

    // step buffer
    pub(crate) next_moles: Vec<f32>,

    // derived pressure and bulk-flow fluxes
    pub(crate) total_moles_curr: Vec<f32>,
    pub(crate) pressure_curr: Vec<f32>,
    pub(crate) prev_flux_x: Vec<f32>,
    pub(crate) prev_flux_y: Vec<f32>,
    pub(crate) flux_x: Vec<f32>,
    pub(crate) flux_y: Vec<f32>,
    pub(crate) edges_x: Vec<(usize, usize)>,
    pub(crate) edges_y: Vec<(usize, usize)>,
}

impl PressureSimState {
    pub(crate) fn new(width: u32, height: u32, gas_count: usize) -> Self {
        let cell_count = (width * height) as usize;
        let total_len = gas_count * cell_count;
        let flux_x_len = ((width - 1) * height) as usize;
        let flux_y_len = (width * (height - 1)) as usize;
        let mut edges_x = Vec::with_capacity(flux_x_len);
        let mut edges_y = Vec::with_capacity(flux_y_len);

        if width > 1 {
            for y in 0..height {
                let row = y * width;
                for x in 0..(width - 1) {
                    let left = (row + x) as usize;
                    let right = left + 1;
                    edges_x.push((left, right));
                }
            }
        }

        if height > 1 {
            for y in 0..(height - 1) {
                let row = y * width;
                for x in 0..width {
                    let top = (row + x) as usize;
                    let bottom = top + width as usize;
                    edges_y.push((top, bottom));
                }
            }
        }

        Self {
            tick: 0,
            width,
            height,
            gas_count,
            cell_count,
            prev_moles: vec![0.0; total_len],
            curr_moles: vec![0.0; total_len],
            next_moles: vec![0.0; total_len],
            total_moles_curr: vec![0.0; cell_count],
            pressure_curr: vec![0.0; cell_count],
            prev_flux_x: vec![0.0; flux_x_len],
            prev_flux_y: vec![0.0; flux_y_len],
            flux_x: vec![0.0; flux_x_len],
            flux_y: vec![0.0; flux_y_len],
            edges_x,
            edges_y,
        }
    }

    #[inline]
    pub(crate) fn cell_index(&self, x: u32, y: u32) -> usize {
        (y * self.width + x) as usize
    }

    #[inline]
    pub(crate) fn idx(&self, gas_i: usize, cell_i: usize) -> usize {
        gas_i * self.cell_count + cell_i
    }

    #[inline]
    pub(crate) fn read_curr(&self, gas_i: usize, cell_i: usize) -> f32 {
        self.curr_moles[self.idx(gas_i, cell_i)]
    }

    #[inline]
    pub(crate) fn write_next(&mut self, gas_i: usize, cell_i: usize, v: f32) {
        let k = self.idx(gas_i, cell_i);
        self.next_moles[k] = v;
    }

    #[inline]
    pub(crate) fn flux_x_index(&self, x: u32, y: u32) -> usize {
        (y * (self.width - 1) + x) as usize
    }

    #[inline]
    pub(crate) fn flux_y_index(&self, x: u32, y: u32) -> usize {
        (y * self.width + x) as usize
    }
}

/// Presented/interpolated values. For now, keep it lightweight:
/// - we interpolate a "pressure" field (derived from moles) for visualization
/// - and retain sim tick
#[derive(Resource, Clone)]
pub struct PressurePresented {
    pub tick: u64,
    pub width: u32,
    pub height: u32,
    pub pressure: Vec<f32>,
    pub wind: Vec<Vec2>,
    pub max_wind_speed: f32,
    // optional: also expose a single cell composition for UI (selected cell)
    pub selected_total_moles: f32,
    pub selected_gas_fractions: Vec<f32>, // length gas_count
}

impl PressurePresented {
    pub fn new(width: u32, height: u32, gas_count: usize) -> Self {
        let cell_count = (width * height) as usize;
        Self {
            tick: 0,
            width,
            height,
            pressure: vec![0.0; cell_count],
            wind: vec![Vec2::ZERO; cell_count],
            max_wind_speed: 0.0,
            selected_total_moles: 0.0,
            selected_gas_fractions: vec![0.0; gas_count],
        }
    }
}

#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq)]
pub struct SelectedCell {
    pub x: u32,
    pub y: u32,
}

impl SelectedCell {
    pub fn center(grid: &GridSettings) -> Self {
        Self {
            x: grid.width / 2,
            y: grid.height / 2,
        }
    }

    #[inline]
    pub fn index(self, grid: &GridSettings) -> usize {
        (self.y * grid.width + self.x) as usize
    }
}

impl Default for SelectedCell {
    fn default() -> Self {
        Self { x: 0, y: 0 }
    }
}
