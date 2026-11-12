# Numerical Advection Schemes: TVD and Boundedness Testing

This is MSc coursework on numerical schemes for the 1-D linear advection equation on a periodic domain.

## Scripts

| Script | What it does |
|---|---|
| `TVD_scheme.py` | A flux-limited **TVD scheme**: first-order upwind plus a Lax–Wendroff correction, limited by the **van Leer** limiter ψ(r) = (r + \|r\|) / (1 + \|r\|). Fully vectorised with `np.roll`. |
| `boundness_experiment.py` | A test harness that runs each scheme at Courant numbers c = 0.5, 1.0 and 1.5 on a cosine-bell initial condition. It records the max and min of φ at every time step and plots them against the initial bounds, to show whether each scheme stays bounded. |

## Implementation notes

- **Robust slope ratio.** The ratio r = (φᵢ − φᵢ₋₁)/(φᵢ₊₁ − φᵢ) is undefined where the downwind difference is about zero. Those points get r = 0 if the upwind difference is also about zero, and ±10⁶ otherwise, so the limiter acts correctly at extrema and in flat regions.
- **Generic scheme interface.** Every scheme takes `(phiLevels, c)` and advances the solution one step in place, so new schemes plug into the experiment without changes.

## Dependencies not included

`boundness_experiment.py` also imports course-provided or separate modules that are not in this folder: `semiLagrangian`, `schemes_sws02hs` (ARTD4) and `advectionHelpers` (`savePlot`, `cosBell_0_05`). Add them to this folder to run the experiment.
