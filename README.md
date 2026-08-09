# Relapse timing and the endemic basin

Figure code accompanying *Relapse timing and the endemic basin: how the shape of the
relapse hazard moves the fold* (F. Sanchez).

## Requirements

Python 3.10+ with `numpy`, `scipy`, and `matplotlib`.

```
pip install numpy scipy matplotlib
```

Run the scripts from the directory containing `exact_model.py`, which holds the baseline
parameters every figure depends on. Running it directly prints the reference values quoted
in the paper (classical fold `0.9014`, the `nu = 0` closed form `0.64955`); if those do not
appear, nothing downstream is right either. Create a `figs/` directory before running, or
edit the output path at the bottom of each script.

## Scripts

| script | figure | what it draws |
|---|---|---|
| `fig1_fix.py` | Figure 1 | Endemic branch `R_0(a)` for early, constant and late relapse at matched relapse intensity. Detects whether the minimum is interior and labels the late-relapse branch as having none. |
| `estab500_fig.py` | Figure 2 | Establishment probability versus initial seed at `R_0 = 0.95`, 500 realizations per point, with Wilson intervals; panel B shows the mesh, threshold and system-size sensitivity. Reads the Gillespie output produced by `g500a.py` and `g_bcd.py`. |
| `rebuild_figs.py` | Figures 3 and 6 | Joint fold map over hazard shape and activity gap, and the basin-enlargement surface across the whole `k < 1` range. |
| `s10_only.py` | Figure 6 | The enlargement surface alone, if you do not need Figure 3. Omits points where the branch has no interior minimum. |
| `plateau_fig.py` | Figure 4 | Fold versus hazard shape: exact quadrature, the asymptotic closed form, the fitted shapes; and the parameter-robustness bars. |
| `s6_fig.py` | Figure 5 | Weibull fits to the two published relapse curves and the folds they imply. |
| `s4_fig.py` | Figure 7 | Fold versus regularization of the hazard singularity, and the bounded hazard families that bracket it. |

`estab_fig_v2.py` is the earlier 60-realization version of Figure 2, superseded by
`estab500_fig.py`.

## Supporting files

These are imported by the figure scripts rather than run directly:

- `exact_model.py` — baseline parameters and reference folds.
- `chain_v2.py` — the relapse profiles (constant, early, late) and the exact continuum
  equilibrium.
- `chain_v3.py` — the graded-mesh box scheme.
- `s4_regularize.py` — the regularized hazard family used by Figure 7.
- `gill_core.py`, `g_lib.py` — the Gillespie chain and establishment helpers.
- `g500a.py`, `g_bcd.py` — produce the simulation output that Figure 2 plots. Run
  `python g500a.py 0.03`, `0.07`, `0.11`, `0.15` for the four seeds.

## Numerical caution

Two things genuinely matter when rerunning these:

1. The hazard `phi(tau) ~ tau^(k-1)` is unbounded at `tau = 0` for `k < 1`. Quadratures
   must be graded near the origin; a uniform grid gives wrong answers silently.
2. Fold searches must check that the minimum is interior. Taking `argmin` over a grid
   reports a spurious fold on branches that have none, which is why Figure 1 tests for it
   explicitly.

## License

MIT

## Citation

[to be completed on acceptance]
