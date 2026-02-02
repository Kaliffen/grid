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
        sim.curr_moles[k] = 1240.0;
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
    // Copy curr -> prev (for interpolation)
    let curr = sim.curr_moles.clone();
    sim.prev_moles.resize(curr.len(), 0.0);
    sim.prev_moles.copy_from_slice(&curr);

    sim.tick += 1;

    let dt = fixed_time.delta_secs();
    let w = sim.width;
    let h = sim.height;
    let cell_count = sim.cell_count; 
    let gas_count = sim.gas_count;
    let width_usize = w as usize;
    let height_usize = h as usize;

    // Compute total moles and pressure per cell (ideal gas law).
    sim.total_moles_curr.fill(0.0);
    {
        let PressureSimState {
            curr_moles,
            total_moles_curr,
            ..
        } = &mut *sim;
        for gas_i in 0..gas_count {
            let base = gas_i * cell_count;
            let curr = &curr_moles[base..base + cell_count];
            for cell_i in 0..cell_count {
                total_moles_curr[cell_i] += curr[cell_i];
            }
        }
    }

    let inv_volume = if grid.cell_volume > 0.0 {
        1.0 / grid.cell_volume
    } else {
        0.0
    };
    sim.pressure_curr.fill(0.0);
    {
        let PressureSimState {
            total_moles_curr,
            pressure_curr,
            ..
        } = &mut *sim;
        for cell_i in 0..cell_count {
            let total = total_moles_curr[cell_i];
            pressure_curr[cell_i] = total * grid.gas_constant * grid.temperature * inv_volume;
        }
    }

    // Compute bulk-flow fluxes from pressure gradients.
    {
        let PressureSimState {
            pressure_curr,
            flux_x,
            flux_y,
            ..
        } = &mut *sim;
        if w > 1 {
            for y in 0..height_usize {
                let row = y * width_usize;
                let flux_row = y * (width_usize - 1);
                for x in 0..(width_usize - 1) {
                    let left = row + x;
                    let right = left + 1;
                    let p_left = pressure_curr[left];
                    let p_right = pressure_curr[right];
                    let flux = grid.bulk_flow_k * (p_left - p_right);
                    flux_x[flux_row + x] = flux;
                }
            }
        }

        if h > 1 {
            for y in 0..(height_usize - 1) {
                let row = y * width_usize;
                let row_down = row + width_usize;
                for x in 0..width_usize {
                    let top = row + x;
                    let bottom = row_down + x;
                    let p_top = pressure_curr[top];
                    let p_bottom = pressure_curr[bottom];
                    let flux = grid.bulk_flow_k * (p_top - p_bottom);
                    flux_y[row + x] = flux;
                }
            }
        }
    }

    // Advection: move mixture along fluxes using upwind composition (curr state).
    let (curr_moles, next_moles, total_moles_curr, flux_x, flux_y) = {
        let PressureSimState {
            curr_moles,
            next_moles,
            total_moles_curr,
            flux_x,
            flux_y,
            ..
        } = &mut *sim;
        (
            curr_moles.as_slice(),
            next_moles.as_mut_slice(),
            total_moles_curr.as_slice(),
            flux_x.as_slice(),
            flux_y.as_slice(),
        )
    };

    next_moles.clone_from_slice(curr_moles);

    let mut outgoing = vec![0.0_f32; cell_count];
    if w > 1 {
        for y in 0..height_usize {
            let row = y * width_usize;
            let flux_row = y * (width_usize - 1);
            for x in 0..(width_usize - 1) {
                let flux = flux_x[flux_row + x];
                let amount = flux.abs() * dt;
                if amount <= 0.0 {
                    continue;
                }

                let (src_x, _) = if flux >= 0.0 { (x, x + 1) } else { (x + 1, x) };
                let src = row + src_x;
                let src_total = total_moles_curr[src];
                if src_total <= 0.0 {
                    continue;
                }

                let max_amount = src_total * grid.max_flow_fraction;
                let desired = amount.min(max_amount);
                if desired > 0.0 {
                    outgoing[src] += desired;
                }
            }
        }
    }

    if h > 1 {
        for y in 0..(height_usize - 1) {
            let row = y * width_usize;
            for x in 0..width_usize {
                let flux = flux_y[row + x];
                let amount = flux.abs() * dt;
                if amount <= 0.0 {
                    continue;
                }

                let (src_y, _) = if flux >= 0.0 { (y, y + 1) } else { (y + 1, y) };
                let src = src_y * width_usize + x;
                let src_total = total_moles_curr[src];
                if src_total <= 0.0 {
                    continue;
                }

                let max_amount = src_total * grid.max_flow_fraction;
                let desired = amount.min(max_amount);
                if desired > 0.0 {
                    outgoing[src] += desired;
                }
            }
        }
    }

    let mut outgoing_scale = vec![1.0_f32; cell_count];
    for cell_i in 0..cell_count {
        let total = total_moles_curr[cell_i];
        let max_out = total * grid.max_flow_fraction;
        let out = outgoing[cell_i];
        if out > max_out && out > 0.0 {
            outgoing_scale[cell_i] = max_out / out;
        }
    }

    if w > 1 {
        for y in 0..height_usize {
            let row = y * width_usize;
            let flux_row = y * (width_usize - 1);
            for x in 0..(width_usize - 1) {
                let flux = flux_x[flux_row + x];
                let amount = flux.abs() * dt;
                if amount <= 0.0 {
                    continue;
                }

                let (src_x, dst_x) = if flux >= 0.0 { (x, x + 1) } else { (x + 1, x) };
                let src = row + src_x;
                let dst = row + dst_x;
                let src_total = total_moles_curr[src];
                if src_total <= 0.0 {
                    continue;
                }

                let max_amount = src_total * grid.max_flow_fraction;
                let transfer = amount.min(max_amount) * outgoing_scale[src];
                if transfer <= 0.0 {
                    continue;
                }

                for gas_i in 0..gas_count {
                    let k_src = gas_i * cell_count + src;
                    let k_dst = gas_i * cell_count + dst;
                    let frac = curr_moles[k_src] / src_total;
                    let delta = transfer * frac;
                    next_moles[k_src] -= delta;
                    next_moles[k_dst] += delta;
                }
            }
        }
    }

    if h > 1 {
        for y in 0..(height_usize - 1) {
            let row = y * width_usize;
            for x in 0..width_usize {
                let flux = flux_y[row + x];
                let amount = flux.abs() * dt;
                if amount <= 0.0 {
                    continue;
                }

                let (src_y, dst_y) = if flux >= 0.0 { (y, y + 1) } else { (y + 1, y) };
                let src = src_y * width_usize + x;
                let dst = dst_y * width_usize + x;
                let src_total = total_moles_curr[src];
                if src_total <= 0.0 {
                    continue;
                }

                let max_amount = src_total * grid.max_flow_fraction;
                let transfer = amount.min(max_amount) * outgoing_scale[src];
                if transfer <= 0.0 {
                    continue;
                }

                for gas_i in 0..gas_count {
                    let k_src = gas_i * cell_count + src;
                    let k_dst = gas_i * cell_count + dst;
                    let frac = curr_moles[k_src] / src_total;
                    let delta = transfer * frac;
                    next_moles[k_src] -= delta;
                    next_moles[k_dst] += delta;
                }
            }
        }
    }

    // curr <-> next (advection result becomes current)
    {
        let PressureSimState {
            curr_moles,
            next_moles,
            ..
        } = &mut *sim;
        std::mem::swap(curr_moles, next_moles);
    }

    // Diffuse each gas independently (mixing is emergent from diffusion of moles).
    for gas_i in 0..gas_count {
        let alpha = gases.gases[gas_i].diffusion_alpha * grid.global_alpha;
        if alpha == 0.0 || dt == 0.0 {
            continue;
        }

        let (curr, next) = {
            let PressureSimState {
                curr_moles,
                next_moles,
                ..
            } = &mut *sim;
            let base = gas_i * cell_count;
            (
                &curr_moles[base..base + cell_count],
                &mut next_moles[base..base + cell_count],
            )
        };
        next.copy_from_slice(curr);

        let mut desired_out = vec![0.0_f32; cell_count];

        if w > 1 {
            for y in 0..height_usize {
                let row = y * width_usize;
                for x in 0..(width_usize - 1) {
                    let a = row + x;
                    let b = a + 1;
                    let diff = curr[b] - curr[a];
                    let delta = alpha * dt * diff;
                    if delta > 0.0 {
                        desired_out[b] += delta;
                    } else if delta < 0.0 {
                        desired_out[a] += -delta;
                    }
                }
            }
        }

        if h > 1 {
            for y in 0..(height_usize - 1) {
                let row = y * width_usize;
                let row_down = row + width_usize;
                for x in 0..width_usize {
                    let a = row + x;
                    let b = row_down + x;
                    let diff = curr[b] - curr[a];
                    let delta = alpha * dt * diff;
                    if delta > 0.0 {
                        desired_out[b] += delta;
                    } else if delta < 0.0 {
                        desired_out[a] += -delta;
                    }
                }
            }
        }

        let mut scale = vec![1.0_f32; cell_count];
        for cell_i in 0..cell_count {
            let out = desired_out[cell_i];
            if out > 0.0 {
                let available = curr[cell_i];
                if out > available {
                    scale[cell_i] = available / out;
                }
            }
        }

        if w > 1 {
            for y in 0..height_usize {
                let row = y * width_usize;
                for x in 0..(width_usize - 1) {
                    let a = row + x;
                    let b = a + 1;
                    let diff = curr[b] - curr[a];
                    let delta = alpha * dt * diff;
                    if delta > 0.0 {
                        let scaled = delta * scale[b];
                        next[a] += scaled;
                        next[b] -= scaled;
                    } else if delta < 0.0 {
                        let scaled = delta * scale[a];
                        next[a] += scaled;
                        next[b] -= scaled;
                    }
                }
            }
        }

        if h > 1 {
            for y in 0..(height_usize - 1) {
                let row = y * width_usize;
                let row_down = row + width_usize;
                for x in 0..width_usize {
                    let a = row + x;
                    let b = row_down + x;
                    let diff = curr[b] - curr[a];
                    let delta = alpha * dt * diff;
                    if delta > 0.0 {
                        let scaled = delta * scale[b];
                        next[a] += scaled;
                        next[b] -= scaled;
                    } else if delta < 0.0 {
                        let scaled = delta * scale[a];
                        next[a] += scaled;
                        next[b] -= scaled;
                    }
                }
            }
        }
    }

    // curr <-> next (double buffer swap)
    {
        let PressureSimState {
            curr_moles,
            next_moles,
            ..
        } = &mut *sim;
        std::mem::swap(curr_moles, next_moles);
    }

    // Optional: clear next (not required, because we fully overwrite it each step)
    // sim.next_moles.fill(0.0);
}

// ------------------------------------
// Interpolation: AfterFixedMainLoop
// ------------------------------------

pub(crate) fn build_presented_state(
    fixed_time: Res<Time<Fixed>>,
    grid: Res<GridSettings>,
    gases: Res<GasCatalog>,
    sim: Res<PressureSimState>,
    mut presented: ResMut<PressurePresented>,
) {
    // Interpolation factor between prev and curr fixed states.
    let alpha = fixed_time.overstep_fraction();

    presented.tick = sim.tick;

    // Interpolate per-gas moles then sum (so pressure field is smooth).
    let cell_count = sim.cell_count;
    let inv_volume = if grid.cell_volume > 0.0 {
        1.0 / grid.cell_volume
    } else {
        0.0
    };
    for cell_i in 0..cell_count {
        let mut total = 0.0;

        for gas_i in 0..sim.gas_count {
            let k = sim.idx(gas_i, cell_i);
            let prev = sim.prev_moles[k];
            let curr = sim.curr_moles[k];
            let interp = prev.lerp(curr, alpha);
            total += interp;
        }

        presented.pressure[cell_i] = total * grid.gas_constant * grid.temperature * inv_volume;
    }

    // Build a per-cell wind vector from the current pressure-driven fluxes.
    let width = sim.width;
    let height = sim.height;
    let mut max_wind_sq: f32 = 0.0;
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
