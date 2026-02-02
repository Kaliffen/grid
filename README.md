# Grid Gas Simulation Visualization

This project simulates gas diffusion + pressure-driven bulk flow on a 2D grid and visualizes the
result as a heatmap. The visualization can show pressure, wind (bulk flow), or a blended view.

## Controls

- **Space**: Cycle overlay mode (Pressure → Wind → Both).
- **1**: Pressure overlay only.
- **2**: Wind overlay only.
- **3**: Blended overlay (pressure + wind).

## Visualization Guide

### Pressure overlay
Pressure is computed as total moles per cell (a proxy for pressure) and mapped to a linear color
ramp:

- **Low pressure**: deep blue.
- **High pressure**: warm orange/red.

### Wind overlay
Wind is derived from the pressure-driven bulk-flow fluxes and visualized as a color field:

- **Hue (color wheel)** = **direction** of flow.
  - Rightward flow ≈ red/orange.
  - Upward flow ≈ magenta.
  - Leftward flow ≈ cyan/blue.
  - Downward flow ≈ green.
- **Brightness** = **speed (magnitude)**.
  - **Black** means no (or extremely low) wind.
  - **Brighter** colors mean stronger flow.

### Blended overlay
The blended view linearly mixes the pressure and wind colors (50/50) in linear space, then
converts to sRGB so the blend stays visually correct.
