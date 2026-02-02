use bevy::prelude::*;
use std::time::Duration;

mod render;
mod sim;

use render::PressureRenderPlugin;
use sim::PressureSimPlugin;

fn main() {
    // Pick a reasonable sim rate (e.g. 200 Hz). Change as needed.
    let sim_hz = 120.0;

    App::new()
        .insert_resource(Time::<Fixed>::from_duration(Duration::from_secs_f64(
            1.0 / sim_hz,
        )))
        .add_plugins(DefaultPlugins)
        .add_plugins((PressureSimPlugin, PressureRenderPlugin))
        .run();
}
