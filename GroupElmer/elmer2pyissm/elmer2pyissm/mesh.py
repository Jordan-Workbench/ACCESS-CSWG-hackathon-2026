"""
Step 1 - Elmer UGRID mesh -> pyISSM mesh2d.

Everything is vectorised numpy, because the real mesh has 1.8 M triangles.
ISSM's C++ wrappers (ElementConnectivity, NodeConnectivity) are not needed,
so this runs with a pip-installed pyISSM and no ISSM build.

Search for "CHECK:" to find decisions the group must verify.
"""

import numpy as np
import pyissm


# ---------------------------------------------------------------------------
# helpers usable on their own (also used by later steps)
# ---------------------------------------------------------------------------
def signed_area(x, y, elements0):
    """Signed triangle area (positive = counterclockwise). elements0 is 0-based."""
    i, j, k = elements0[:, 0], elements0[:, 1], elements0[:, 2]
    return 0.5 * ((x[j] - x[i]) * (y[k] - y[i]) - (x[k] - x[i]) * (y[j] - y[i]))


def face_to_node(values_face, elements0, areas, nnode):
    """Area-weighted average of a face (element) field onto nodes (vertices)."""
    num = np.zeros(nnode)
    den = np.zeros(nnode)
    w = values_face * areas
    for c in range(3):
        np.add.at(num, elements0[:, c], w)
        np.add.at(den, elements0[:, c], areas)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def node_to_face(values_node, elements0):
    """Plain mean of the three corner values (what ISSM does for P1 -> P0)."""
    return values_node[elements0].mean(axis=1)


def _edges_and_neighbours(elements0):
    """Unique edges, the faces on each side, and face-face connectivity.

    Returns 0-based arrays:
      edges       (ne, 2)  sorted vertex pairs
      edge_faces  (ne, 2)  face on each side, -1 if boundary
      face_nbr    (nf, 3)  neighbour across local edge k = (n_k, n_{k+1}), -1 if none
      face_edge   (nf, 3)  edge id of local edge k
    """
    nf = len(elements0)
    loc = np.stack([elements0[:, [0, 1]], elements0[:, [1, 2]], elements0[:, [2, 0]]], axis=1)
    key = np.sort(loc.reshape(-1, 2), axis=1)
    edges, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    if counts.max() > 2:
        raise ValueError("Non-manifold mesh: an edge is shared by more than two faces")
    face_of_slot = np.repeat(np.arange(nf), 3)

    order = np.argsort(inv, kind="stable")
    sorted_e = inv[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = sorted_e[1:] != sorted_e[:-1]

    edge_faces = np.full((len(edges), 2), -1, dtype=np.int64)
    edge_faces[sorted_e[first], 0] = face_of_slot[order[first]]
    edge_faces[sorted_e[~first], 1] = face_of_slot[order[~first]]

    face_edge = inv.reshape(nf, 3)
    a = edge_faces[face_edge, 0]
    b = edge_faces[face_edge, 1]
    me = np.arange(nf)[:, None]
    face_nbr = np.where(a == me, b, a)
    return edges, edge_faces, face_nbr, face_edge


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------
def build_mesh(elmer, verbose=True):
    """Build a pyISSM mesh2d from the Elmer states file.

    Returns
    -------
    mesh : pyissm.model.classes.mesh.mesh2d
    info : dict
        Diagnostics, and `elements0` / `areas` for the later steps.
    """
    st = elmer.states

    # --- coordinates -------------------------------------------------------
    # CHECK: `x`/`y` are the projected node coordinates in metres. The global
    # attribute says projection = EPSG:3031. Plot x,y and confirm they look
    # like Antarctica in polar stereographic (|x|,|y| < ~3e6 m).
    x = st.var("x")
    y = st.var("y")
    lon = st.var("mesh2D_node_x")
    lat = st.var("mesh2D_node_y")

    # --- elements ----------------------------------------------------------
    elements0 = st.var("mesh2D_face_nodes", dtype=np.int64)
    start = st.nc.variables["mesh2D_face_nodes"].getncattr("start_index") \
        if "start_index" in st.nc.variables["mesh2D_face_nodes"].ncattrs() else 0
    # CHECK: start_index is 0 in the Elmer/XIOS files. ISSM is 1-based. If a
    # file ever says start_index = 1, this subtraction makes it 0-based first.
    elements0 = elements0 - int(start)
    nv, ne_ = len(x), len(elements0)
    if elements0.min() < 0 or elements0.max() >= nv:
        raise ValueError("face_nodes out of range after start_index correction")

    # --- orientation -------------------------------------------------------
    # CHECK: ISSM assumes counterclockwise triangles. Elmer/XIOS does not
    # guarantee an ordering, so any clockwise triangles are flipped here. The
    # number flipped is reported; on the real mesh it might be 0 or ~50 %.
    a2 = signed_area(x, y, elements0)
    cw = a2 < 0
    n_cw = int(cw.sum())
    elements0[cw] = elements0[cw][:, [0, 2, 1]]
    areas = np.abs(a2)
    if (areas == 0).any():
        raise ValueError(f"{int((areas == 0).sum())} degenerate (zero-area) triangles")

    # --- topology ----------------------------------------------------------
    edges0, edge_faces0, face_nbr0, face_edge0 = _edges_and_neighbours(elements0)
    boundary = edge_faces0[:, 1] == -1

    # Boundary segments [v1 v2 element] oriented so the domain is on the left
    # (consistent with the CCW elements).
    bf = edge_faces0[boundary, 0]
    bedge = edges0[boundary]
    seg = np.empty((boundary.sum(), 3), dtype=np.int64)
    loc = np.stack([elements0[:, [0, 1]], elements0[:, [1, 2]], elements0[:, [2, 0]]], axis=1)
    for k in range(3):
        sel = np.all(np.sort(loc[bf, k], axis=1) == bedge, axis=1)
        seg[sel, 0:2] = loc[bf[sel], k]
    seg[:, 2] = bf

    vertexonboundary = np.zeros(nv)
    vertexonboundary[np.unique(bedge)] = 1

    # --- fill the pyISSM class (1-based indices) ---------------------------
    mesh = pyissm.model.classes.mesh.mesh2d()
    mesh.x = x.astype(float)
    mesh.y = y.astype(float)
    mesh.lat = lat.astype(float)
    mesh.long = lon.astype(float)
    mesh.epsg = 3031
    mesh.elements = elements0 + 1
    mesh.numberofvertices = nv
    mesh.numberofelements = ne_
    mesh.numberofedges = len(edges0)
    # CHECK: ISSM's `edges` layout is [v1 v2 elementLeft elementRight] (1-based).
    # Here a missing neighbour is written as -1. Compare with an ISSM-made mesh
    # (e.g. one from md.mesh after triangle()) to confirm ISSM's marker for
    # "no element" (0 or -1) before running any solver that reads edges.
    mesh.edges = np.column_stack([edges0 + 1,
                                  edge_faces0[:, 0] + 1,
                                  np.where(edge_faces0[:, 1] >= 0, edge_faces0[:, 1] + 1, -1)])
    mesh.segments = seg + 1
    mesh.segmentmarkers = np.ones(len(seg), dtype=np.int64)
    mesh.vertexonboundary = vertexonboundary
    # ISSM convention: 0 = no neighbour.
    # CHECK: ISSM's own ElementConnectivity orders neighbours by the edge
    # opposite each vertex. Here column k is the neighbour across edge
    # (n_k, n_{k+1}). That doesn't matter for plotting or analysis, but check
    # it before handing the mesh to a solver.
    mesh.elementconnectivity = np.where(face_nbr0 >= 0, face_nbr0 + 1, 0)
    # CHECK: scale_factor corrects areas and volumes for polar-stereographic
    # distortion. It is left unset (NaN) here, so ISSM areas are PROJECTED
    # areas. Step 6 compares projected-area volume with Elmer's `volume`
    # scalar to find out which convention Elmer used. If they disagree by
    # ~1-4 %, Elmer used true areas and scale_factor should be set.

    # --- cross-checks against the Elmer arrays -----------------------------
    info = {"n_clockwise_flipped": n_cw,
            "n_boundary_edges": int(boundary.sum()),
            "n_boundary_vertices": int(vertexonboundary.sum()),
            "total_area_m2": float(areas.sum()),
            "elements0": elements0, "areas": areas}
    if st.has("mesh2D_edge_nodes"):
        info["elmer_numberofedges"] = int(st.nc.dimensions["nmesh2D_edge"].size)
    if st.has("mesh2D_edge_face_links"):
        efl = st.var("mesh2D_edge_face_links", dtype=np.int64)
        # CHECK: boundary edges are marked with the _FillValue (-999) in
        # edge_face_links. Our count must equal Elmer's.
        info["elmer_n_boundary_edges"] = int((efl[:, 1] == -999).sum() + (efl[:, 0] == -999).sum())
    if st.has("cell_area"):
        ca = st.var("cell_area")
        rel = np.abs(areas - ca) / ca
        info["max_rel_area_diff_vs_cell_area"] = float(rel.max())
        # CHECK: a large difference here means `cell_area` is a TRUE (ellipsoidal)
        # area rather than the projected area, or the face order differs.

    if verbose:
        print(f"[mesh] {nv} vertices, {ne_} elements, {len(edges0)} edges")
        print(f"[mesh] flipped {n_cw} clockwise elements to CCW")
        print(f"[mesh] {info['n_boundary_edges']} boundary edges "
              f"(Elmer says {info.get('elmer_n_boundary_edges', '?')}); "
              f"edges: ours {len(edges0)} vs Elmer {info.get('elmer_numberofedges', '?')}")
        if "max_rel_area_diff_vs_cell_area" in info:
            print(f"[mesh] max relative area difference vs cell_area: "
                  f"{info['max_rel_area_diff_vs_cell_area']:.2e}")
    return mesh, info
