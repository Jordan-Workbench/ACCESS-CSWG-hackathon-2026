"""
elmer_to_pyissm: run steps 0-6 and return a pyISSM Model.

Command line
------------
python -m elmer2pyissm.convert \
    --states  ssp370_c003_m005_v2_states.nc \
    --forcing ssp370_c003_m005_v2_forcing.nc \
    --scalars ssp370_c003_m005_v2_scalars.nc \
    --step -1 --out ssp370_c003_m005_v2_pyissm.nc --validate
"""

import argparse

import numpy as np
import pyissm

from . import dynamics, forcing, geometry, mesh, results
from .reader import ElmerOutput

# CHECK: Elmer densities from the .sif. pyISSM's defaults are rho_ice = 917 and
# rho_water = 1023. Elmer/Ice Antarctic setups commonly use 917 / 1028.
RHO_ICE = 917.0
RHO_WATER = 1028.0


def elmer_to_pyissm(states, forcing_file=None, scalars=None, step=-1,
                    friction_method="invert", timeseries_forcing=False,
                    with_results=True, verbose=True):
    """Convert one Elmer/Ice run to a pyISSM Model.

    Parameters
    ----------
    states, forcing_file, scalars : paths to the three Elmer/Ice netCDF files
    step : int
        Which states timestep initialises the model (-1 = last).
    friction_method : "invert" or "params" (see dynamics.build_friction)
    timeseries_forcing : bool
        If True, SMB and melt are ISSM timeseries over all forcing steps,
        not a single snapshot.
    with_results : bool
        Load every Elmer snapshot into md.results.TransientSolution.

    Returns
    -------
    md : pyissm.model.Model
    report : dict of diagnostics, used by validate()
    """
    elmer = ElmerOutput(states, forcing_file, scalars)
    if verbose:
        print(elmer.summary())
    step = step % elmer.states.ntimes
    report = {"step": step}

    md = pyissm.model.Model()
    md.miscellaneous.name = elmer.states.global_attrs().get("model_case", "elmer_run")

    # constants and materials --------------------------------------------------
    md.constants.yts = dynamics.YTS_ELMER          # CHECK: see dynamics.YTS_ELMER
    md.materials.rho_ice = RHO_ICE
    md.materials.rho_water = RHO_WATER

    # Step 1: mesh ---------------------------------------------------------
    md.mesh, info = mesh.build_mesh(elmer, verbose=verbose)
    report["mesh_info"] = info
    if elmer.forcing is not None:
        same = elmer.check_same_connectivity()
        if verbose:
            print(f"[mesh] forcing file has identical face_nodes to states: {same}")
        # CHECK: if this prints False, the forcing file's faces are in a different
        # order and every forcing face field needs remapping before use.

    # Step 2: geometry and masks ------------------------------------------
    md.geometry = geometry.build_geometry(elmer, step, verbose=verbose)
    md.mask, has_ice = geometry.build_mask(elmer, step, md.geometry, RHO_ICE, RHO_WATER, verbose=verbose)
    floating = md.mask.ocean_levelset < 0

    # Step 3: velocity, friction, rheology --------------------------------
    md.initialization = dynamics.build_initialization(elmer, step, verbose=verbose)
    md.friction, report["friction_diag"] = dynamics.build_friction(
        elmer, step, info, has_ice, floating, method=friction_method, verbose=verbose)
    md.materials.rheology_B = dynamics.build_rheology_B(elmer, step, verbose=verbose)
    md.materials.rheology_n = 3.0 * np.ones(md.mesh.numberofelements)
    # CHECK: Glen exponent n = 3 assumed. Use the n from the .sif.

    # Step 4: forcing ------------------------------------------------------
    if elmer.forcing is not None:
        md.smb = forcing.build_smb(elmer, step, info, timeseries=timeseries_forcing, verbose=verbose)
        md.basalforcings = forcing.build_basalforcings(elmer, step, info, floating,
                                                       timeseries=timeseries_forcing, verbose=verbose)
    md.miscellaneous.notes = forcing.forcing_notes(elmer)

    # Step 5: results ------------------------------------------------------
    if with_results:
        md.results.TransientSolution = results.build_transient_results(
            elmer, info, RHO_ICE, RHO_WATER, verbose=verbose)

    report["elmer"] = elmer
    return md, report


def main():
    ap = argparse.ArgumentParser(description="Convert Elmer/Ice XIOS output to a pyISSM model")
    ap.add_argument("--states", required=True)
    ap.add_argument("--forcing")
    ap.add_argument("--scalars")
    ap.add_argument("--step", type=int, default=-1)
    ap.add_argument("--friction-method", default="invert", choices=["invert", "params"])
    ap.add_argument("--timeseries-forcing", action="store_true")
    ap.add_argument("--no-results", action="store_true")
    ap.add_argument("--out", help="save the model with pyissm.model.io.save_model")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--truth", help="template_truth.json (synthetic templates only)")
    a = ap.parse_args()

    md, report = elmer_to_pyissm(a.states, a.forcing, a.scalars, a.step, a.friction_method,
                                 a.timeseries_forcing, not a.no_results)
    if a.validate:
        from .validate import validate
        validate(md, report["elmer"], report, truth_json=a.truth)
    if a.out:
        pyissm.model.io.save_model(md, a.out)
        print(f"saved {a.out}")


if __name__ == "__main__":
    main()
