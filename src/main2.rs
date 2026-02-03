use bevy::{color::palettes::basic::PURPLE, prelude::*};

use bevy_asset::RenderAssetUsages;
use bevy_mesh::{Indices, Mesh};
use bevy::render::render_resource::PrimitiveTopology;

const W: usize = 64;
const H: usize = 64;
const TILE_SIZE: f32 = 16.0;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .init_resource::<TileModel>()
        .add_systems(Startup, setup)
        .add_systems(Update, (edit_tiles, remesh_if_dirty))
        .run();
}

#[derive(Resource)]
struct TileModel {
    occ: Vec<bool>,
    dirty: bool,
}

impl Default for TileModel {
    fn default() -> Self {
        Self {
            occ: vec![false; W * H],
            dirty: true,
        }
    }
}

impl TileModel {
    #[inline]
    fn idx(x: usize, y: usize) -> usize {
        y * W + x
    }

    #[inline]
    fn get(&self, x: usize, y: usize) -> bool {
        self.occ[Self::idx(x, y)]
    }

    fn set(&mut self, x: usize, y: usize, v: bool) {
        let i = Self::idx(x, y);
        if self.occ[i] != v {
            self.occ[i] = v;
            self.dirty = true;
        }
    }
}

#[derive(Component)]
struct TileMeshEntity;

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut tiles: ResMut<TileModel>,
) {
    commands.spawn(Camera2d);

    // Seed some filled tiles.
    for y in 0..64 {
        for x in 0..64 {
            tiles.set(x, y, true);
        }
    }

    // Bevy 0.18 Mesh::new requires RenderAssetUsages.
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    build_mesh_from_tiles(&tiles, &mut mesh);

    let mesh_handle = meshes.add(mesh);

    commands.spawn((
        Mesh2d(mesh_handle),
        MeshMaterial2d(materials.add(Color::from(PURPLE))),
        Transform::default(),
        TileMeshEntity,
    ));
}

// Demo mutation: toggles a tile occasionally.
fn edit_tiles(mut tiles: ResMut<TileModel>, time: Res<Time>) {
    // Bevy 0.18: elapsed_secs()
    let t = time.elapsed_secs();

    if (t * 4.0).fract() < 0.02 {
        let x = (t as usize * 13) % W;
        let y = (t as usize * 7) % H;

        // Fix borrow error: read first, then write.
        let cur = tiles.get(x, y);
        tiles.set(x, y, !cur);
    }
}

fn remesh_if_dirty(
    mut tiles: ResMut<TileModel>,
    mut meshes: ResMut<Assets<Mesh>>,
    q: Query<&Mesh2d, With<TileMeshEntity>>,
) {
    if !tiles.dirty {
        return;
    }
    tiles.dirty = false;

    // Your build has single(), not get_single().
    let mesh2d = q.single();

    let Some(mesh) = meshes.get_mut(&mesh2d.unwrap().0) else {
        return;
    };

    build_mesh_from_tiles(&tiles, mesh);
}

fn build_mesh_from_tiles(tiles: &TileModel, mesh: &mut Mesh) {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    for y in 0..H {
        for x in 0..W {
            if !tiles.get(x, y) {
                continue;
            }

            let x0 = x as f32 * TILE_SIZE;
            let y0 = y as f32 * TILE_SIZE;
            let x1 = x0 + TILE_SIZE;
            let y1 = y0 + TILE_SIZE;

            let base = positions.len() as u32;

            positions.extend_from_slice(&[
                [x0, y0, 0.0],
                [x1, y0, 0.0],
                [x1, y1, 0.0],
                [x0, y1, 0.0],
            ]);

            // Optional for ColorMaterial, but fine.
            uvs.extend_from_slice(&[
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ]);

            indices.extend_from_slice(&[
                base, base + 1, base + 2,
                base, base + 2, base + 3,
            ]);
        }
    }

    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(Indices::U32(indices));
}
