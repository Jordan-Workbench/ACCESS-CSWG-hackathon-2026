"""
Step 2 - Geometry and masks.

Search for "CHECK:" to find decisions the group must verify.
"""

import numpy as np
import pyissm

# CHECK: Elmer/Ice keeps a minimum ice thickness everywhere (often 1 m or 10 m)
# so "ice-free" nodes are not h = 0. Find the value in the Elmer .sif
# ("Min H" / "Minimum Thickness") and set it here. Nodes with
# h <= MIN_THICKNESS + ICE_TOL count as ice-free.
MIN_THICKNESS = 1.0
ICE_TOL = 1e-3


def build_geometry(elmer, step, verbose=True):
    """pyISSM geometry class from one states timestep (node fields)."""
    st = elmer.states
    geo = pyissm.model.classes.geometry()
    geo.thickness = st.var("h", step)
    geo.surface = st.var("zs", step)
    geo.base = st.var("zb", step)
    geo.bed = st.var("bedrock", step)
    # CHECK: Elmer's `bedrock` is time-dependent in the file (GIA/solid-earth
    # coupling?). ISSM's geometry.bed is static unless solid-earth is on.
    # If bedrock changes between steps, decide which step to use.

    err = np.abs(geo.surface - geo.base - geo.thickness)
    if verbose:
        print(f"[geometry] step {step}: max |surface - base - thickness| = {err.max():.3e} m")
        if err.max() > 1e-3:
            print("[geometry] WARNING: surface - base != thickness. Check which fields are consistent.")
    return geo


def build_mask(elmer, step, geometry=None, rho_ice=917.0, rho_water=1028.0, verbose=True):
    """pyISSM mask class (ocean_levelset, ice_levelset) from one timestep.

    ISSM conventions
    ----------------
    ocean_levelset > 0 grounded, < 0 floating, 0 on the grounding line
    ice_levelset   < 0 ice present, > 0 no ice, 0 on the ice front
    """
    st = elmer.states
    mask = pyissm.model.classes.mask()

    # --- ocean_levelset from Elmer groundedmask ----------------------------
    gm = st.var("groundedmask", step)
    # CHECK: The Elmer/Ice GroundedMask convention is
    #   1 = grounded, 0 = grounding-line node, -1 = floating.
    # Confirm on the real file: np.unique(groundedmask) should give [-1, 0, 1].
    # Elmer's 1/0/-1 maps directly onto ISSM's sign convention. ISSM would
    # normally store a signed DISTANCE. If a solver needs that (e.g. for
    # grounding-line migration), recompute it with pyissm or ISSM's
    # reinitializelevelset before running.
    vals = np.unique(gm)
    if not set(np.round(vals).tolist()) <= {-1.0, 0.0, 1.0}:
        print(f"[mask] WARNING: unexpected groundedmask values {vals[:10]}")
    mask.ocean_levelset = gm.astype(float)

    # --- ice_levelset from thickness ---------------------------------------
    h = st.var("h", step) if geometry is None else geometry.thickness
    has_ice = h > MIN_THICKNESS + ICE_TOL
    # CHECK: An alternative is the face field `sftgif` (land_ice_area_fraction).
    # Compare the two definitions on the real data. They should agree, except
    # possibly along the ice front.
    mask.ice_levelset = np.where(has_ice, -1.0, 1.0)

    # --- consistency: does flotation agree with the mask? -------------------
    if geometry is not None:
        bed, H = geometry.bed, geometry.thickness
        # CHECK: rho_ice / rho_water must be Elmer's values. pyISSM's default
        # md.materials.rho_water is 1023, while Elmer/Ice setups often use 1028.
        # Use the .sif values and set md.materials to match (done in convert.py).
        floating_by_flotation = (bed < 0) & (H * rho_ice / rho_water < -bed)
        disagree = (floating_by_flotation != (gm < 0)) & (gm != 0) & has_ice
        if verbose:
            print(f"[mask] step {step}: grounded {int((gm > 0).sum())}, GL {int((gm == 0).sum())}, "
                  f"floating {int((gm < 0).sum())}, with ice {int(has_ice.sum())}")
            print(f"[mask] nodes where groundedmask disagrees with flotation: {int(disagree.sum())}")
    return mask, has_ice
