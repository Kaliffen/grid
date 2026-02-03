use bevy::prelude::*;

mod coords_overlay;

use coords_overlay::CoordOverlayPlugin;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(CoordOverlayPlugin)
        .run();
}
