"""
Step 5 - Elmer time series -> md.results.TransientSolution.

pyISSM's plotting and analysis tools read results by ISSM's own field names
(Vx, Thickness, IceVolume, GroundedArea, ...). Build one
pyissm.model.classes.results.solutionstep per Elmer snapshot and wrap them in
a results.solution. md.results.TransientSolution.<Field> then returns a
time-stacked array, just like results loaded from a real ISSM run.

Fields that exist only in Elmer are stored with the prefix "Elmer" (e.g.
ElmerStrbasemag) so nobody mistakes them for ISSM outputs.
"""

import numpy as np
import pyissm

from .geometry import MIN_THICKNESS, ICE_TOL
from .mesh import face_to_node, node_to_face

# CHECK: ISSM reports TotalSmb and TotalFloatingBmb in Gt/yr. The conversion
# needs the ice density. Use Elmer's value (see convert.py).
GT = 1.0e12  # kg


def _scalar(v):
    """Scalars are stored as 1-element arrays.

    pyISSM 0.2.1's save_model/load_model turns plain Python or numpy float
    scalars inside results into STRINGS. 1-element arrays survive, and
    md.results.TransientSolution.<Field> still returns a 1-D time series.
    CHECK: remove this workaround if a later pyISSM fixes the round trip.
    """
    return np.array([float(v)])


def integrated_quantities(st, k, el0, areas, rho_ice, rho_water):
    """ISSM-style scalar outputs computed from Elmer fields at states step k."""
    h_f = st.var("h_elem", k) if st.has("h_elem") else node_to_face(st.var("h", k), el0)
    bed_f = st.var("bedrock_elem", k) if st.has("bedrock_elem") else node_to_face(st.var("bedrock", k), el0)
    sftgif = st.var("sftgif", k)
    sftgrf = st.var("sftgrf", k)
    sftflf = st.var("sftflf", k)

    out = {}
    # CHECK: ISSM's IceVolume sums only elements that contain ice, while the
    # Elmer `volume` scalar may include the minimum-thickness layer
    # everywhere. Both are reported and validate.py says which matches Elmer.
    out["IceVolume"] = float((h_f * areas).sum())
    out["IceVolumeIceOnly"] = float((h_f * areas * (sftgif > 0)).sum())
    haf_f = h_f + np.minimum(bed_f, 0.0) * rho_water / rho_ice
    # CHECK: ISSM computes IceVolumeAboveFloatation per grounded element. Here
    # it is weighted by Elmer's grounded fraction (sftgrf), which is
    # close but not identical in partly grounded elements.
    out["IceVolumeAboveFloatation"] = float((np.clip(haf_f, 0, None) * areas * sftgrf).sum())
    out["GroundedArea"] = float((sftgrf * areas).sum())
    out["FloatingArea"] = float((sftflf * areas).sum())
    return out


def build_transient_results(elmer, info, rho_ice=917.0, rho_water=1028.0,
                            forcing_step_map=None, include_elmer_extras=True, verbose=True):
    """Return a results.solution with one step per states timestep.

    forcing_step_map : dict {states_step: forcing_step}, optional
        CHECK: which forcing record belongs with which snapshot? States are
        instantaneous (1 Jan of each year), forcing is an annual MEAN. The
        default pairs index k with index k, i.e. the snapshot at the END of
        the averaging year in the templates. Confirm with the time bounds of
        the real files. It may need to be k -> k + 1 or k -> k - 1.
    """
    st, fo = elmer.states, elmer.forcing
    el0, areas = info["elements0"], info["areas"]
    nv = len(st.var("x"))
    steps = []
    for k in range(st.ntimes):
        s = pyissm.model.classes.results.solutionstep()
        s.time = _scalar(st.years[k])        # decimal years
        s.step = np.array([k + 1])
        vx, vy = st.var("ssavelocity_x", k), st.var("ssavelocity_y", k)
        s.Vx, s.Vy, s.Vel = vx, vy, np.hypot(vx, vy)          # m/yr, as in ISSM results
        s.Thickness = st.var("h", k)
        s.Surface = st.var("zs", k)
        s.Base = st.var("zb", k)
        s.Bed = st.var("bedrock", k)
        s.MaskOceanLevelset = st.var("groundedmask", k)
        s.MaskIceLevelset = np.where(s.Thickness > MIN_THICKNESS + ICE_TOL, -1.0, 1.0)
        for key, val in integrated_quantities(st, k, el0, areas, rho_ice, rho_water).items():
            setattr(s, key, _scalar(val))

        if fo is not None:
            kf = forcing_step_map.get(k, k) if forcing_step_map else k
            if kf < fo.ntimes:
                acabf_f = fo.var("acabf", kf)
                s.SmbMassBalance = face_to_node(acabf_f, el0, areas, nv)      # m ice/yr
                bmb = fo.var("bmb", kf)
                floating = s.MaskOceanLevelset < 0
                s.BasalforcingsFloatingiceMeltingRate = np.where(floating, bmb, 0.0)
                s.TotalSmb = _scalar((acabf_f * areas).sum() * rho_ice / GT)   # Gt/yr
                # CHECK: TotalFloatingBmb (Gt/yr, melt positive). Preferred source is
                # Elmer's own applied basal mass flux `libmassbf` (negative = melt,
                # per face), since it is exactly what the model applied, grounding-line
                # treatment included. The fallback integrates the nodal `bmb`, which
                # differs from libmassbf by up to tens of % in the templates,
                # depending on how partly floating faces are weighted.
                if fo.has("libmassbf"):
                    melt_face = -fo.var("libmassbf", kf)
                else:
                    melt_face = node_to_face(np.where(floating, bmb, 0.0), el0) * st.var("sftflf", k)
                s.TotalFloatingBmb = _scalar((melt_face * areas).sum() * rho_ice / GT)
                s.ElmerForcingTime = _scalar(fo.years[kf])

        if include_elmer_extras:
            s.ElmerStrbasemag = st.var("strbasemag", k)                 # MPa, ELEMENT field
            s.ElmerEffectivePressure = st.var("effective_pressure", k)  # MPa, vertex field
            s.ElmerDhdt = st.var("dhdt", k)                             # m/yr, vertex field
            s.ElmerHaf = st.var("haf", k)                               # m, vertex field
        steps.append(s)

    sol = pyissm.model.classes.results.solution(steps)
    if verbose:
        print(f"[results] TransientSolution with {len(steps)} steps: "
              f"{[round(float(s.time[0]), 3) for s in steps]}")
    return sol
