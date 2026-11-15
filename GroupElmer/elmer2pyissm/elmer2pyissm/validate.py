"""
Step 6 - Validate the converted model.

Each check returns a row (name, ours, reference, relative error, status).
Run `validate(md, elmer, report)` after convert, or `python -m elmer2pyissm.convert ... --validate`.

Tolerances are deliberately loose defaults.
CHECK: tighten them once you understand where the real data differ, e.g.
grounding-line elements, or float32 storage in the scalars file (rel ~1e-7).
"""

import json

import numpy as np

from .mesh import signed_area, node_to_face

TOL = {"volume": 1e-4, "smb": 1e-3, "bmb": 2e-2, "area": 1e-4, "taub_median": 5e-2}


def _row(name, ours, ref, tol):
    if ref is None or ours is None:
        return (name, ours, ref, np.nan, "SKIP")
    rel = abs(ours - ref) / max(abs(ref), 1e-30)
    return (name, ours, ref, rel, "PASS" if rel <= tol else "FAIL")


def validate(md, elmer, report, truth_json=None, verbose=True):
    rows = []
    st, sc, fo = elmer.states, elmer.scalars, elmer.forcing
    info = report["mesh_info"]
    el0, areas = info["elements0"], info["areas"]

    # ---- mesh ----------------------------------------------------------------
    el = md.mesh.elements
    rows.append(("elements are 1-based", int(el.min()), 1, 0.0 if el.min() == 1 else 1.0,
                 "PASS" if el.min() == 1 else "FAIL"))
    ccw = signed_area(md.mesh.x, md.mesh.y, el - 1) > 0
    rows.append(("all elements CCW", int(ccw.sum()), len(ccw), 0.0 if ccw.all() else 1.0,
                 "PASS" if ccw.all() else "FAIL"))
    if "elmer_n_boundary_edges" in info:
        rows.append(_row("boundary edges vs Elmer", info["n_boundary_edges"],
                         info["elmer_n_boundary_edges"], 0))
    if "elmer_numberofedges" in info:
        rows.append(_row("edge count vs Elmer", md.mesh.numberofedges, info["elmer_numberofedges"], 0))
    if sc is not None and sc.has("domain_area"):
        rows.append(_row("total area vs scalars domain_area", float(areas.sum()),
                         float(sc.var("domain_area")), TOL["area"]))
    elif st.has("cell_area"):
        rows.append(_row("total area vs sum(cell_area)", float(areas.sum()),
                         float(st.var("cell_area").sum()), TOL["area"]))

    # ---- geometry ------------------------------------------------------------
    g = md.geometry
    err = float(np.abs(g.surface - g.base - g.thickness).max())
    rows.append(("max |s - b - H| [m]", err, 0.0, err, "PASS" if err < 1e-3 else "FAIL"))

    # ---- friction ------------------------------------------------------------
    fd = report.get("friction_diag")
    if fd is not None:
        rows.append(("tau_b ISSM vs strbasemag (median rel)", fd["taub_median_rel_err"], 0.0,
                     fd["taub_median_rel_err"],
                     "PASS" if fd["taub_median_rel_err"] < TOL["taub_median"] else "FAIL"))

    # ---- integrated quantities vs scalars file -------------------------------
    TS = getattr(md.results, "TransientSolution", None)
    if TS is not None and sc is not None:
        nts = min(len(TS.steps), sc.ntimes)
        vol_sc = sc.var("volume") if sc.has("volume") else None
        smb_sc = sc.var("smb_int") if sc.has("smb_int") else None
        bmb_sc = sc.var("bmb_int") if sc.has("bmb_int") else None
        rho_i = md.materials.rho_ice
        for k in range(nts):
            s = TS.steps[k]
            t_k = float(np.ravel(s.time)[0])
            if vol_sc is not None:
                # CHECK: two candidate definitions of Elmer's `volume`. The row
                # label says which one matches better. If "ice only" wins on the
                # real data, ISSM-style IceVolume (ice elements only) is the
                # right comparison, and results.py should report that as IceVolume.
                ref = float(vol_sc[k])
                v_all, v_ice = float(np.ravel(s.IceVolume)[0]), float(np.ravel(s.IceVolumeIceOnly)[0])
                r_all = abs(v_all - ref) / ref
                r_ice = abs(v_ice - ref) / ref
                best, label = (v_all, "all faces") if r_all <= r_ice else (v_ice, "ice only")
                rows.append(_row(f"[{t_k:.2f}] IceVolume ({label}) vs scalars volume",
                                 best, ref, TOL["volume"]))
            if smb_sc is not None and hasattr(s, "TotalSmb"):
                rows.append(_row(f"[{t_k:.2f}] TotalSmb vs smb_int [m3/a]",
                                 float(np.ravel(s.TotalSmb)[0]) * 1e12 / rho_i, float(smb_sc[k]), TOL["smb"]))
            if bmb_sc is not None and hasattr(s, "TotalFloatingBmb"):
                rows.append(_row(f"[{t_k:.2f}] TotalFloatingBmb vs bmb_int [m3/a]",
                                 float(np.ravel(s.TotalFloatingBmb)[0]) * 1e12 / rho_i, float(bmb_sc[k]), TOL["bmb"]))

    # ---- melt sign consistency ----------------------------------------------
    if fo is not None and fo.has("libmassbf") and fo.has("bmb"):
        k = report["step"]
        bmb_int = float((node_to_face(md.basalforcings.floatingice_melting_rate, el0) * areas).sum()) \
            if np.ndim(md.basalforcings.floatingice_melting_rate) == 1 else None
        lib_int = float((fo.var("libmassbf", k) * areas).sum())
        # CHECK: if this FAILs with ratio ~ +1 instead of -1, bmb and libmassbf
        # use the SAME sign and the melt sign in forcing.py needs flipping.
        rows.append(_row("melt: integral(bmb) vs -integral(libmassbf)", bmb_int, -lib_int, TOL["bmb"]))

    # ---- optional: synthetic-template truth file -----------------------------
    if truth_json:
        t = json.load(open(truth_json))
        m = t["mesh"]
        rows.append(_row("[truth] numberofvertices", md.mesh.numberofvertices, m["numberofvertices"], 0))
        rows.append(_row("[truth] numberofelements", md.mesh.numberofelements, m["numberofelements"], 0))
        rows.append(_row("[truth] clockwise faces flipped", info["n_clockwise_flipped"],
                         m["n_faces_stored_clockwise_in_file"], 0))
        rows.append(_row("[truth] boundary vertices", int(md.mesh.vertexonboundary.sum()),
                         m["n_boundary_vertices"], 0))
        rows.append(_row("[truth] total area", float(areas.sum()), m["total_area_m2"], 1e-6))
        p = t["per_timestep"][report["step"]]
        ol = md.mask.ocean_levelset
        rows.append(_row("[truth] grounded vertices (incl GL)", int((ol >= 0).sum()),
                         p["n_vertices_grounded_incl_GL"], 0))
        rows.append(_row("[truth] floating vertices", int((ol < 0).sum()), p["n_vertices_floating"], 0))
        rows.append(_row("[truth] vertices with ice", int((md.mask.ice_levelset < 0).sum()),
                         p["n_vertices_with_ice"], 0))
        rows.append(_row("[truth] max speed", float(md.initialization.vel.max()), p["max_speed_m_per_a"], 1e-9))

    if verbose:
        print(f"\n{'check':58s} {'ours':>14s} {'reference':>14s} {'rel err':>9s}  status")
        print("-" * 106)
        for name, ours, ref, rel, status in rows:
            fo_ = lambda v: f"{v:14.6g}" if isinstance(v, (int, float, np.floating, np.integer)) else f"{str(v):>14s}"
            print(f"{name:58s} {fo_(ours)} {fo_(ref)} {rel:9.2e}  {status}")
        n_fail = sum(r[4] == "FAIL" for r in rows)
        print(f"\n{len(rows)} checks, {n_fail} FAIL")
    return rows
