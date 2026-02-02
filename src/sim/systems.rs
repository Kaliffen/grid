use bevy::prelude::*;
use std::time::Instant;

use crate::sim::resources::{
    GasCatalog, GasKind, GridSettings, PressurePresented, PressureSimState,
};

// ------------------------------------
// Diagnostics (SIM)
// ------------------------------------

#[derive(Resource)]
pub(crate) struct SimDiag {
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

// ------------------------------------
// Startup: Simulation setup
// ------------------------------------

pub(crate) fn setup_sim(mut commands: Commands, grid: Res<GridSettings>, gases: Res<GasCatalog>) {
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
}

// ------------------------------------
// Simulation: FixedUpdate diffusion
// ------------------------------------

pub(crate) fn advance_diffusion_fixed(
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

    // Compute pressure as total moles per cell.
    sim.pressure_curr.fill(0.0);
    for cell_i in 0..sim.cell_count {
        let mut total = 0.0;
        for gas_i in 0..sim.gas_count {
            total += sim.read_curr(gas_i, cell_i);
        }
        sim.pressure_curr[cell_i] = total;
    }

    // Compute bulk-flow fluxes from pressure gradients.
    if w > 1 {
        for y in 0..h {
            for x in 0..(w - 1) {
                let left = sim.cell_index(x, y);
                let right = sim.cell_index(x + 1, y);
                let p_left = sim.pressure_curr[left];
                let p_right = sim.pressure_curr[right];
                let flux = grid.bulk_flow_k * (p_left - p_right);
                let k = sim.flux_x_index(x, y);
                sim.flux_x[k] = flux;
            }
        }
    }

    if h > 1 {
        for y in 0..(h - 1) {
            for x in 0..w {
                let top = sim.cell_index(x, y);
                let bottom = sim.cell_index(x, y + 1);
                let p_top = sim.pressure_curr[top];
                let p_bottom = sim.pressure_curr[bottom];
                let flux = grid.bulk_flow_k * (p_top - p_bottom);
                let k = sim.flux_y_index(x, y);
                sim.flux_y[k] = flux;
            }
        }
    }

    // Advection: move mixture along fluxes using upwind composition (curr state).
    let (curr_moles, next_moles) = {
        let s = &mut *sim;
        (&s.curr_moles, &mut s.next_moles)
    };

    next_moles.clone_from(curr_moles);

    if w > 1 {
        for y in 0..h {
            for x in 0..(w - 1) {
                let flux = sim.flux_x[sim.flux_x_index(x, y)];
                let (src_x, dst_x, amount) = if flux >= 0.0 {
                    (x, x + 1, flux * dt)
                } else {
                    (x + 1, x, -flux * dt)
                };

                let src = sim.cell_index(src_x, y);
                let dst = sim.cell_index(dst_x, y);
                let src_total = sim.pressure_curr[src];
                if src_total <= 0.0 {
                    continue;
                }

                let max_amount = src_total * grid.max_flow_fraction;
                let transfer = amount.min(max_amount);
                if transfer <= 0.0 {
                    continue;
                }

                for gas_i in 0..sim.gas_count {
                    let k_src = sim.idx(gas_i, src);
                    let k_dst = sim.idx(gas_i, dst);
                    let frac = sim.curr_moles[k_src] / src_total;
                    let delta = transfer * frac;
                    sim.next_moles[k_src] -= delta;
                    sim.next_moles[k_dst] += delta;
                }
            }
        }
    }

    if h > 1 {
        for y in 0..(h - 1) {
            for x in 0..w {
                let flux = sim.flux_y[sim.flux_y_index(x, y)];
                let (src_y, dst_y, amount) = if flux >= 0.0 {
                    (y, y + 1, flux * dt)
                } else {
                    (y + 1, y, -flux * dt)
                };

                let src = sim.cell_index(x, src_y);
                let dst = sim.cell_index(x, dst_y);
                let src_total = sim.pressure_curr[src];
                if src_total <= 0.0 {
                    continue;
                }

                let max_amount = src_total * grid.max_flow_fraction;
                let transfer = amount.min(max_amount);
                if transfer <= 0.0 {
                    continue;
                }

                for gas_i in 0..sim.gas_count {
                    let k_src = sim.idx(gas_i, src);
                    let k_dst = sim.idx(gas_i, dst);
                    let frac = sim.curr_moles[k_src] / src_total;
                    let delta = transfer * frac;
                    sim.next_moles[k_src] -= delta;
                    sim.next_moles[k_dst] += delta;
                }
            }
        }
    }

    // curr <-> next (advection result becomes current)
    let curr = std::mem::take(&mut sim.curr_moles);
    sim.curr_moles = std::mem::replace(&mut sim.next_moles, curr);

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

pub(crate) fn build_presented_state(
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

    // Build a per-cell wind vector from the current pressure-driven fluxes.
    let width = sim.width;
    let height = sim.height;
    let mut max_wind_sq = 0.0;
    for y in 0..height {
        for x in 0..width {
            let mut wind_x = 0.0;
            let mut wind_y = 0.0;
            let mut count_x = 0.0;
            let mut count_y = 0.0;

            if width > 1 {
                if x > 0 {
                    wind_x += sim.flux_x[sim.flux_x_index(x - 1, y)];
                    count_x += 1.0;
                }
                if x + 1 < width {
                    wind_x += sim.flux_x[sim.flux_x_index(x, y)];
                    count_x += 1.0;
                }
            }

            if height > 1 {
                if y > 0 {
                    wind_y += sim.flux_y[sim.flux_y_index(x, y - 1)];
                    count_y += 1.0;
                }
                if y + 1 < height {
                    wind_y += sim.flux_y[sim.flux_y_index(x, y)];
                    count_y += 1.0;
                }
            }

            if count_x > 0.0 {
                wind_x /= count_x;
            }
            if count_y > 0.0 {
                wind_y /= count_y;
            }

            let idx = sim.cell_index(x, y);
            presented.wind[idx] = Vec2::new(wind_x, wind_y);
            max_wind_sq = max_wind_sq.max(wind_x * wind_x + wind_y * wind_y);
        }
    }
    presented.max_wind_speed = max_wind_sq.sqrt();

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
// Diagnostics (requested format)
// ------------------------------------

pub(crate) fn sim_diagnostics(fixed_time: Res<Time<Fixed>>, mut diag: ResMut<SimDiag>) {
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
