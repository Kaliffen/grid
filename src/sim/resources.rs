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
}

impl PressureSimState {
    pub(crate) fn new(width: u32, height: u32, gas_count: usize) -> Self {
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
    // optional: also expose a single cell composition for UI (center cell)
    pub center_total_moles: f32,
    pub center_gas_fractions: Vec<f32>, // length gas_count
}

impl PressurePresented {
    pub fn new(width: u32, height: u32, gas_count: usize) -> Self {
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
