# elmer2pyissm

Convert Elmer/Ice + XIOS output (UGRID netCDF: `*_states.nc`, `*_forcing.nc`, `*_scalars.nc`)
into a **pyISSM `Model`**, so that pyISSM's plotting and analysis tools work on Elmer runs.

Tested with pyISSM 0.2.1 (pip) and **no ISSM build**: the converter computes all mesh connectivity
in numpy.

## Layout

```
elmer2pyissm/          the package, one module per step
  reader.py      0  open files, decode the 360_day time axis, read one timestep at a time
  mesh.py        1  UGRID -> md.mesh (1-based, CCW, edges, segments, connectivity)
  geometry.py    2  md.geometry, md.mask
  dynamics.py    3  md.initialization, md.friction (regcoulomb), md.materials.rheology_B
  forcing.py     4  md.smb, md.basalforcings (snapshot or timeseries), provenance notes
  results.py     5  md.results.TransientSolution (ISSM field names + Elmer extras)
  validate.py    6  checks against Elmer's scalars file (+ template truth)
  convert.py        elmer_to_pyissm() and the command line
notebooks/
  elmer_to_pyissm_walkthrough.ipynb   steps 0-6 one at a time, with plots (pre-run on templates)
tests/
  test_templates.py                   pytest: every planted answer in the templates
templates/                            synthetic template files + generator + header checker
```

## Quick start

```bash
pip install pyissm netCDF4 numpy matplotlib pytest     # or use the issm_env conda env

# 1. confirm everything passes on the synthetic templates
pytest -q tests/

# 2. convert the real run (use the ORIGINAL 5-timestep files, not the cdo subsets)
python -m elmer2pyissm.convert \
    --states  ssp370_c003_m005_v2_states.nc \
    --forcing ssp370_c003_m005_v2_forcing.nc \
    --scalars ssp370_c003_m005_v2_scalars.nc \
    --step -1 --validate --out ssp370_c003_m005_v2_pyissm.nc
```

From Python:

```python
import elmer2pyissm, pyissm
md, report = elmer2pyissm.elmer_to_pyissm(states, forcing, scalars, step=-1)
elmer2pyissm.validate(md, report["elmer"], report)
pyissm.plot.plot_model_field(md, md.results.TransientSolution.Vel[-1], show_cbar=True)
```

Options: `friction_method="invert"|"params"`, `timeseries_forcing=True` (SMB/melt as ISSM
timeseries over all forcing steps), `with_results=False`.

## The CHECK list (do this before trusting the real conversion)

Every decision that depends on the real data or the Elmer `.sif` is marked in the code:

```bash
grep -rn "CHECK:" elmer2pyissm/
```

The main ones:

1. **Time:** forcing/scalars time stamps are mid-interval for annual means. Which forcing record
   pairs with which snapshot (`results.forcing_step_map`)?
2. **Mesh:** forcing faces in the same order as states (printed); number of clockwise triangles;
   projected vs true area (`scale_factor`); ISSM `edges` and `elementconnectivity` conventions
   (these only matter before *running* ISSM).
3. **Constants:** `geometry.MIN_THICKNESS`, `convert.RHO_ICE/RHO_WATER`, `dynamics.YTS_ELMER`,
   Glen `n`. All must match the `.sif`.
4. **Masks:** groundedmask is 1/0/−1; the ocean levelset needs reinitialising as a signed distance
   before grounding-line-migration runs.
5. **Friction:** what `tau_RC`, `alphaRC`, `betaRC` and `ceff` mean. `invert` (default) reproduces
   `strbasemag` whatever the law is; `params` needs the law confirmed first.
6. **Rheology:** `mu` units (MPa a^1/3?), the 2^(1/n) factor, and Glen parameter vs effective
   viscosity.
7. **Forcing:** which SMB field; melt sign (validate prints `bmb` vs `−libmassbf`).
8. **Volume definition:** step 6 reports whether Elmer's `volume` is over all faces or over ice only.

## pyISSM 0.2.1 issues found while building this

- `pyissm.model.Model()` fails if `$USER` is unset (some containers or CI). The package sets a default.
- `save_model`/`load_model` turns **scalar** results (floats) into **strings**. Scalars are stored
  as 1-element arrays to work around it (`results._scalar`). There's a test for this.
- The pyISSM default `md.materials.rho_water = 1023` and `md.constants.yts = 365 d`. Both are
  overridden with Elmer's values.
- `plot_model_ts` wants `data_list=[arrays]`, not field names (or call it with no arguments to plot
  every 1-D series).

## What it does NOT do (yet). Good hackathon extensions

- Signed-distance levelsets, and a consistency check for an actual ISSM SSA solve
  (`md.check_consistency`, then a stress-balance run compared with Elmer's velocities).
- An SMB elevation feedback via `smb.gradients` with `dacabfdz_forcing`.
- ISMIP-style local-quadratic melt in ISSM (`basalforcings.ismip6` needs 3-D thermal forcing).
- `vertexconnectivity` and `scale_factor`.
- Streaming results straight from disk for very long runs. Results are currently held in memory:
  about 5 snapshots × 39 fields × 1.8 M doubles is roughly 2-3 GB for the real mesh. Use
  `with_results=False` or drop fields you don't need.
