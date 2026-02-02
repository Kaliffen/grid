use bevy::prelude::*;

use crate::sim::resources::{GasCatalog, GridSettings};
use crate::sim::systems::{
    advance_diffusion_fixed, build_presented_state, setup_sim, sim_diagnostics,
};

pub struct PressureSimPlugin;

impl Plugin for PressureSimPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(GridSettings::default())
            .insert_resource(GasCatalog::default())
            .add_systems(Startup, setup_sim)
            .add_systems(FixedUpdate, (advance_diffusion_fixed, sim_diagnostics))
            .add_systems(
                RunFixedMainLoop,
                build_presented_state.in_set(RunFixedMainLoopSystems::AfterFixedMainLoop),
            );
    }
}
