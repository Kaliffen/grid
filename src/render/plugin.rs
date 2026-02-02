use bevy::prelude::*;

use crate::render::systems::{
    render_diagnostics, setup_render, update_heatmap_texture, update_ui,
};
use crate::sim::systems::setup_sim;

pub struct PressureRenderPlugin;

impl Plugin for PressureRenderPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_render.after(setup_sim))
            .add_systems(Update, (update_ui, update_heatmap_texture, render_diagnostics));
    }
}
