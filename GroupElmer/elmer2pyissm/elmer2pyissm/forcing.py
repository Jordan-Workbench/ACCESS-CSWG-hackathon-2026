"""
Step 4 - Surface mass balance and basal melt.

pyISSM expects SMB and melt rates on VERTICES, in m ice / yr (it divides by
yts when writing the binary). Elmer writes most forcing on FACES, so face
fields are moved to vertices by area weighting.

Time-varying forcing in ISSM is a matrix of shape (nv + 1, nt). The last row
holds the times in decimal years.
"""

import numpy as np
import pyissm

from .mesh import face_to_node


def _to_vertices(f, name, step, info, nv):
    data = f.var(name, step)
    if f.location(name) == "face":
        return face_to_node(data, info["elements0"], info["areas"], nv)
    return data


def _timeseries(f, name, info, nv, steps=None):
    steps = range(f.ntimes) if steps is None else steps
    cols = [_to_vertices(f, name, k, info, nv) for k in steps]
    ts = np.vstack([np.column_stack(cols), f.years[list(steps)][None, :]])
    return ts


def build_smb(elmer, step, info, timeseries=False, field="acabf", verbose=True):
    """pyISSM smb.default from Elmer forcing.

    CHECK: Which field is the SMB the ice model actually saw?
      acabf             - "applied" SMB (zero where there is no ice in the template)
      smb_total_flux    - acabf_forcing + smb_feedback_flux
      acabf_forcing     - the SMB from the climate forcing, before the elevation feedback
    To RE-RUN with ISSM's own elevation feedback, use smb.gradients with
    dacabfdz_forcing instead. That isn't coded here.
    CHECK: units are "m ice a-1", which is what ISSM expects (ice equivalent, per year).
    """
    f = elmer.forcing
    nv = len(elmer.states.var("x"))
    smb = pyissm.model.classes.smb.default()
    if timeseries:
        smb.mass_balance = _timeseries(f, field, info, nv)
    else:
        smb.mass_balance = _to_vertices(f, field, step, info, nv)
    if verbose:
        mb = smb.mass_balance if not timeseries else smb.mass_balance[:-1, -1]
        print(f"[smb] {field}: range {mb.min():.3f} .. {mb.max():.3f} m ice/a"
              f"{' (timeseries, ' + str(f.ntimes) + ' steps)' if timeseries else ''}")
    return smb


def build_basalforcings(elmer, step, info, floating, timeseries=False, verbose=True):
    """pyISSM basalforcings.default from Elmer `bmb`.

    CHECK: sign convention. ISSM floatingice_melting_rate is POSITIVE for melt.
    In the templates `bmb` is positive = melt and `libmassbf` is negative = melt.
    Confirm on the real data: integrate both over the floating area. One
    should be approximately minus the other (validate.py does this).
    CHECK: bmb is on nodes and libmassbf on faces. Elmer probably applies melt
    only where the node is floating (perhaps with a sub-grid scheme at the
    grounding line). ISSM decides per element using ocean_levelset and its
    groundingline.melt_interpolation setting. Expect some differences
    in the grounding-line elements.
    CHECK: groundedice_melting_rate is set to 0 and geothermalflux is left
    unset. Only a thermal ISSM run needs geothermalflux.
    """
    f = elmer.forcing
    nv = len(elmer.states.var("x"))
    bf = pyissm.model.classes.basalforcings.default()
    if timeseries:
        ts = _timeseries(f, "bmb", info, nv)
        bf.floatingice_melting_rate = ts
        bf.groundedice_melting_rate = np.vstack([np.zeros((nv, ts.shape[1])), ts[-1:]])
    else:
        bmb = f.var("bmb", step)
        bf.floatingice_melting_rate = np.where(floating, bmb, 0.0)
        bf.groundedice_melting_rate = np.zeros(nv)
    if verbose:
        r = bf.floatingice_melting_rate if not timeseries else bf.floatingice_melting_rate[:-1, -1]
        print(f"[basal] floating melt: range {r.min():.3f} .. {r.max():.3f} m ice/a")
    return bf


def forcing_notes(elmer):
    """Provenance text for md.miscellaneous.notes."""
    lines = ["Converted from Elmer/Ice output by elmer2pyissm"]
    for label, f in (("states", elmer.states), ("forcing", elmer.forcing), ("scalars", elmer.scalars)):
        if f is None:
            continue
        lines.append(f"{label}: {f.path}")
        for k, v in f.global_attrs().items():
            if k in ("history", "CDI", "CDO", "uuid", "timeStamp"):
                continue
            lines.append(f"  {k}: {v}")
    return "\n".join(lines)
