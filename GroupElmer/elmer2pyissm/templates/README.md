# Elmer/Ice → pyISSM: synthetic template files

These are small, synthetic stand-ins for the three Elmer/Ice + XIOS output files from
`AA_SSA_ISMIP7_SSP370_CESM2_C003_m005_v2`. They let you develop and test an Elmer → pyISSM
converter on a laptop in seconds instead of streaming 1.9 GB files.

**The data are synthetic.** The structure (variable names, dimensions, dtypes, fill values,
attributes) copies the real files. The values are made up but physically consistent, and
every "right answer" is recorded in `template_truth.json`.

## What's here

| File | What it is |
|---|---|
| `templates/template_ssp370_c003_m005_v2_states.nc` | 5 instantaneous snapshots (1 Jan 2016–2020): geometry, masks, velocity, friction fields |
| `templates/template_ssp370_c003_m005_v2_forcing.nc` | 5 annual means: SMB, basal melt, ocean forcing at draft |
| `templates/template_ssp370_c003_m005_v2_scalars.nc` | 5 annual values: integrated volume / SMB / BMB plus per-node ocean diagnostics |
| `templates/template_truth.json` | Known answers a correct converter must reproduce |
| `make_elmer_templates.py` | Regenerates the templates (`--nx`, `--ny`, `--seed`) |
| `compare_headers.py` | Compares the structure of any two files or `ncdump -h` outputs |
| `reference_headers/` | `ncdump -h` of the real files, for comparison |

Mesh: 1,271 nodes, 2,400 triangles, 3,670 edges, in EPSG:3031 metres (Amundsen Sea sector box,
400 × 300 km). Inland is at larger x, and the ocean / ice front is at smaller x. The domain has
grounded ice, a floating shelf and an "ice-free" strip at the front.

Requires `numpy` and `netCDF4` (plus `xarray` if you use it for reading).

## How the templates relate to the real files

- **states**: copied from the header of the **original** 5-step file, including the `mesh2D`
  topology variable, `time_instant` and `axis_nbounds`.
- **forcing** and **scalars**: we only had headers of the **cdo `seltimestep` subsets**, so these
  copy that layout (`bnds`/`time_bnds`, `CDI_grid_type`, `missing_value`). The one deliberate change
  is that **forcing includes the `mesh2D` topology variable**, which cdo dropped.
  - To-do: run `ncdump -h` on the *original* forcing and scalars files and check them with
    `compare_headers.py`, then tell whoever maintains the templates what differs.
- **Time stamps on forcing/scalars**: the template puts annual means at mid-year (bounds = the
  preceding year). Check what XIOS actually wrote with `cdo showtimestamp` or by reading
  `time_bnds` in the real files.

```bash
ncdump -h ssp370_c003_m005_v2_forcing.nc > forcing_header.cdl
python compare_headers.py forcing_header.cdl templates/template_ssp370_c003_m005_v2_forcing.nc
```

## Planted traps (the real data probably has these too)

1. **0-based connectivity.** `start_index = 0`; ISSM `md.mesh.elements` is 1-based.
2. **Clockwise triangles.** 740 of the 2,400 faces are stored clockwise. ISSM expects
   counterclockwise.
3. **Shuffled numbering.** Node and face order are permuted, as from an XIOS parallel gather. Use
   only the connectivity in the file, never assume grid order.
4. **Boundary markers differ between arrays:** `mesh2D_face_links` uses `-1` (out_of_mesh) and
   `mesh2D_edge_face_links` uses `-999` (_FillValue). netCDF libraries may mask fill values to NaN
   or masked arrays, so read with masking off or handle both cases.
5. **Elmer groundedmask** is 1 grounded / 0 grounding-line node / −1 floating. ISSM's
   `ocean_levelset` is > 0 grounded and < 0 floating.
6. **Minimum thickness.** "Ice-free" nodes have `h = 1 m`, not 0. ISSM's `ice_levelset` must come
   from a threshold (or from `sftgif`).
7. **Opposite melt signs:** `bmb` is positive for melt, and `libmassbf` is negative for melt.
8. **360_day calendar.** `decimal_year = 2005 + days / 360`. Don't let cftime/xarray convert this
   to a standard calendar along the way.
9. **The same quantity at nodes and faces.** `dhdt` is on nodes in states but on faces in forcing.
   Fill ISSM from node fields and use `*_elem` fields to cross-check.
10. **Units:** Elmer uses m, a and MPa. ISSM uses SI for friction, effective pressure and
    rheology, so check `md.constants.yts`.

## Checklist: run your converter on the templates and tick off

Compare against `template_truth.json` (indices there are 1-based).

**Mesh**
- [ ] `md.mesh.numberofvertices == 1271`, `numberofelements == 2400`
- [ ] `md.mesh.elements.min() == 1`, and every element has positive (CCW) signed area
- [ ] Your element areas sum to `total_area_m2` (1.2e11 m²) and match `cell_area` per face
- [ ] 140 boundary edges / 140 boundary vertices (`md.mesh.vertexonboundary.sum()`)
- [ ] `md.mesh.lat`/`long` match `mesh2D_node_y/x` (and `x, y → lon, lat` agrees with EPSG:3031)

**Geometry and masks** (per timestep)
- [ ] `surface − base == thickness` to round-off
- [ ] Grounded / floating / GL vertex counts match `per_timestep[*].n_vertices_*`
- [ ] Number of vertices with `ice_levelset < 0` equals `n_vertices_with_ice`

**Velocity and friction**
- [ ] Max speed matches `max_speed_m_per_a`
- [ ] Recomputing basal stress from the friction class you chose (using the template's planted law)
      gives `strbasemag`. The planted law is a **placeholder**: once you know the real Elmer/Ice
      definitions, update `synth_state()` in `make_elmer_templates.py` and regenerate.

**Forcing**
- [ ] Σ (SMB × element area) matches `smb_int_m3_per_a`
- [ ] Σ (melt × element area) over floating ice matches `bmb_int_m3_per_a_melt_positive`
      (melt is positive here)

**Results / time series**
- [ ] `md.results.TransientSolution` has 5 steps with times 2016.0 … 2020.0
- [ ] Ice volume computed from the pyISSM results matches the scalars file `volume` and
      `volume_m3_sum_h_elem_x_cell_area`. The scalars file stores float32, so use a relative
      tolerance of about 1e-6.
- [ ] The pyISSM plotting and analysis tools run on the converted model without errors

**Then switch to the real files** and repeat the mesh and geometry checks. Run the integrated
checks against the real scalars file.

## Regenerating

```bash
python make_elmer_templates.py                 # default (41 x 31 nodes)
python make_elmer_templates.py --nx 201 --ny 151 --outdir templates_big   # stress-test
```

`template_truth.json` is rewritten each time, so keep it with its templates.
