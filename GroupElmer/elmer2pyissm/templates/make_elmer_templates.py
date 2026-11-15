#!/usr/bin/env python3
"""
make_elmer_templates.py
=======================

Build small, synthetic stand-ins for the three Elmer/Ice + XIOS output files

    ssp370_c003_m005_v2_states.nc
    ssp370_c003_m005_v2_forcing.nc
    ssp370_c003_m005_v2_scalars.nc

The templates copy the variable names, dimensions, dtypes, fill values and
attributes from the real files' ncdump headers. The mesh is a few thousand
triangles instead of 1.8 million, so a converter can be developed on a laptop
in seconds.

The data are synthetic but physically consistent (flotation, grounded/floating
masks, friction law, melt law, integrated scalars). Every "right answer" a
converter should reproduce is written to template_truth.json so the group can
check their Elmer -> pyISSM code against it.

Some deliberate traps are planted, matching issues expected in the real data
(see README.md):
  * 0-based connectivity (start_index = 0)
  * a fraction of triangles stored clockwise
  * node and face ordering shuffled (as from an XIOS parallel gather)
  * Elmer-style groundedmask (1 grounded / 0 grounding line / -1 floating)
  * thin "minimum thickness" ice in ice-free areas
  * bmb positive = melt, while libmassbf is negative = melt
  * a 360_day calendar

Usage
-----
    python make_elmer_templates.py                    # default ~41 x 31 nodes
    python make_elmer_templates.py --nx 81 --ny 61    # bigger mesh
    python make_elmer_templates.py --outdir templates --seed 1

Requires: numpy, netCDF4
"""

import argparse
import json
import os

import numpy as np
from netCDF4 import Dataset

# ----------------------------------------------------------------------------
# Constants (Elmer/Ice style units: m, a, MPa)
# ----------------------------------------------------------------------------
RHO_I = 917.0          # kg m-3
RHO_W = 1028.0         # kg m-3
G = 9.81               # m s-2
YTS = 365.25 * 86400.0 # s per year (Elmer/Ice usually uses 31556926)
H_MIN = 1.0            # Elmer minimum thickness in "ice-free" areas [m]
U0 = 300.0             # regularised Coulomb threshold speed [m a-1]
M_EXP = 3.0            # regularised Coulomb exponent
COULOMB_FRAC = 0.01    # tau_RC = COULOMB_FRAC * N  (gives tau_b ~ 0.1-0.2 MPa)
GAMMA0 = 14477.0       # local-quadratic melt coefficient [m a-1]
CP_W = 3974.0          # J kg-1 K-1
L_F = 3.34e5           # J kg-1
OMEGA = 7.2921e-5      # rad s-1
LAMBDA1, LAMBDA2, LAMBDA3 = -0.0575, 0.0832, 7.59e-4  # freezing point (ISMIP6)

FILL_FACE_EDGES = 999999
FILL_FACE_LINKS = 999999
FILL_EDGE_FACE = -999
OUT_OF_MESH = -1

YEARS = [2016, 2017, 2018, 2019, 2020]   # states snapshot dates (1 Jan)
T0_YEAR = 2005                           # "days since 2005-01-01", 360_day


# ----------------------------------------------------------------------------
# EPSG:3031 inverse (polar stereographic south, WGS84, lat_ts = -71)
# ----------------------------------------------------------------------------
def epsg3031_to_lonlat(x, y):
    a = 6378137.0
    f = 1 / 298.257223563
    e = np.sqrt(f * (2 - f))
    phic = np.deg2rad(71.0)
    tc = np.tan(np.pi / 4 - phic / 2) / ((1 - e * np.sin(phic)) / (1 + e * np.sin(phic))) ** (e / 2)
    mc = np.cos(phic) / np.sqrt(1 - e**2 * np.sin(phic) ** 2)
    rho = np.hypot(x, y)
    t = rho * tc / (a * mc)
    phi = np.pi / 2 - 2 * np.arctan(t)
    for _ in range(10):
        es = e * np.sin(phi)
        phi = np.pi / 2 - 2 * np.arctan(t * ((1 - es) / (1 + es)) ** (e / 2))
    lat = -np.rad2deg(phi)
    lon = np.rad2deg(np.arctan2(x, y))
    return lon, lat


# ----------------------------------------------------------------------------
# Mesh
# ----------------------------------------------------------------------------
def build_mesh(nx, ny, rng, frac_cw=0.3):
    """Jittered structured triangulation in EPSG:3031 metres.

    Domain: a 400 km x 300 km box in the Amundsen Sea sector. Inland is at
    larger x (toward the pole side), the ocean at smaller x.
    """
    x0, x1 = -1.75e6, -1.35e6
    y0, y1 = -0.55e6, -0.25e6
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    # jitter interior nodes so the mesh is not perfectly regular
    jit = np.zeros_like(X)
    jit[1:-1, 1:-1] = 1
    X = X + jit * rng.uniform(-0.25, 0.25, X.shape) * dx
    Y = Y + jit * rng.uniform(-0.25, 0.25, Y.shape) * dy

    idx = np.arange(nx * ny).reshape(nx, ny)
    tris = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a, b, c, d = idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]
            if (i + j) % 2 == 0:
                tris += [[a, b, c], [a, c, d]]
            else:
                tris += [[a, b, d], [b, c, d]]
    tris = np.array(tris, dtype=np.int64)
    x = X.ravel()
    y = Y.ravel()

    # make every triangle CCW first
    area2 = ((x[tris[:, 1]] - x[tris[:, 0]]) * (y[tris[:, 2]] - y[tris[:, 0]])
             - (x[tris[:, 2]] - x[tris[:, 0]]) * (y[tris[:, 1]] - y[tris[:, 0]]))
    cw = area2 < 0
    tris[cw] = tris[cw][:, [0, 2, 1]]

    # shuffle node numbering (mimics XIOS gather order)
    perm = rng.permutation(len(x))          # new_index -> old_index
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))         # old_index -> new_index
    x, y = x[perm], y[perm]
    tris = inv[tris]

    # shuffle face order
    tris = tris[rng.permutation(len(tris))]

    # plant clockwise triangles
    flip = rng.random(len(tris)) < frac_cw
    tris[flip] = tris[flip][:, [0, 2, 1]]

    return x, y, tris, int(flip.sum())


def build_topology(tris, nnode):
    nf = len(tris)
    # local edges in face order: (n0,n1), (n1,n2), (n2,n0)
    loc = np.stack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]], axis=1)  # nf,3,2
    key = np.sort(loc, axis=2).reshape(-1, 2)
    uniq, inverse = np.unique(key, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    edges = uniq                                  # ne,2 (sorted node pairs)
    ne = len(edges)
    face_edges = inverse.reshape(nf, 3)

    edge_faces = np.full((ne, 2), FILL_EDGE_FACE, dtype=np.int64)
    count = np.zeros(ne, dtype=np.int64)
    for f in range(nf):
        for k in range(3):
            e = face_edges[f, k]
            edge_faces[e, count[e]] = f
            count[e] += 1
    assert count.max() <= 2

    face_links = np.full((nf, 3), OUT_OF_MESH, dtype=np.int64)
    for f in range(nf):
        for k in range(3):
            e = face_edges[f, k]
            a, b = edge_faces[e]
            other = b if a == f else a
            if other != FILL_EDGE_FACE:
                face_links[f, k] = other

    boundary_edges = np.where(edge_faces[:, 1] == FILL_EDGE_FACE)[0]
    boundary_nodes = np.unique(edges[boundary_edges])
    return edges, face_edges, edge_faces, face_links, boundary_edges, boundary_nodes


def tri_area(x, y, tris):
    return 0.5 * np.abs((x[tris[:, 1]] - x[tris[:, 0]]) * (y[tris[:, 2]] - y[tris[:, 0]])
                        - (x[tris[:, 2]] - x[tris[:, 0]]) * (y[tris[:, 1]] - y[tris[:, 0]]))


# ----------------------------------------------------------------------------
# Synthetic physics
# ----------------------------------------------------------------------------
def node_to_face(v, tris):
    return v[tris].mean(axis=1)


def synth_state(x, y, tris, k, neighbours):
    """State at snapshot k (0..4)."""
    xmin, xmax = x.min(), x.max()
    s = (xmax - x) / (xmax - xmin)            # 0 inland -> 1 ocean edge
    yn = (y - y.min()) / (y.max() - y.min())

    bed = 600.0 - 1700.0 * s + 150.0 * np.sin(2 * np.pi * yn) * (1 - s)
    front = 0.85 - 0.01 * k                    # ice front retreats slightly
    thin = 1.0 - 0.015 * k * np.clip((s - 0.4) / 0.45, 0, 1)
    H = (250.0 + 2600.0 * np.sqrt(np.clip(1 - s / front, 0, 1))) * thin
    H = np.where(s < front, H, H_MIN)

    float_thresh = -bed * RHO_W / RHO_I       # thickness needed to ground
    floating = (bed < 0) & (H < float_thresh)
    zb = np.where(floating, -H * RHO_I / RHO_W, bed)
    zs = zb + H
    haf = np.where(bed < 0, H - float_thresh, H)

    grounded = ~floating
    gmask = np.where(grounded, 1.0, -1.0)
    # grounding-line nodes: grounded nodes with a floating neighbour
    gl = np.array([grounded[n] and floating[list(neighbours[n])].any() for n in range(len(x))])
    gmask[gl] = 0.0

    has_ice = H > H_MIN + 1e-6
    speed = np.where(has_ice, (30.0 + 2500.0 * s**3) * (1 + 0.04 * k), 0.0)
    vx = -speed * 0.98                         # flow toward ocean (decreasing x)
    vy = speed * 0.2 * np.sin(2 * np.pi * yn)

    N = np.clip(RHO_I * G * H - RHO_W * G * np.clip(-bed, 0, None), 0, None) / 1e6  # MPa
    N = np.where(grounded, N, 0.0)
    tau_RC = COULOMB_FRAC * N
    u = np.hypot(vx, vy)
    taub = np.where(grounded & has_ice, tau_RC * (u / (u + U0)) ** (1 / M_EXP), 0.0)
    ceff = np.where(u > 0, taub / np.maximum(u, 1e-12), 0.0)          # MPa m-1 a
    mu = np.where(has_ice, 0.4 + 0.2 * (1 - s), 0.4)                 # placeholder

    return dict(bedrock=bed, h=H, zb=zb, zs=zs, haf=haf, groundedmask=gmask,
                ssavelocity_x=vx, ssavelocity_y=vy, effective_pressure=N,
                tau_RC=tau_RC, alphaRC=np.full_like(H, 1 / M_EXP),
                betaRC=np.full_like(H, U0), ceff=ceff, mu=mu, taub_node=taub,
                floating=floating, has_ice=has_ice)


def face_fractions(st, tris):
    ice = st["has_ice"][tris].mean(axis=1)
    grd = (st["has_ice"] & ~st["floating"])[tris].mean(axis=1)
    flt = (st["has_ice"] & st["floating"])[tris].mean(axis=1)
    return ice, grd, flt


def slope_sin_theta(x, y, tris, zb):
    """sin of basal slope angle per face, then averaged to nodes."""
    x1, y1, z1 = x[tris[:, 0]], y[tris[:, 0]], zb[tris[:, 0]]
    x2, y2, z2 = x[tris[:, 1]], y[tris[:, 1]], zb[tris[:, 1]]
    x3, y3, z3 = x[tris[:, 2]], y[tris[:, 2]], zb[tris[:, 2]]
    det = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    gx = ((z2 - z1) * (y3 - y1) - (z3 - z1) * (y2 - y1)) / det
    gy = ((x2 - x1) * (z3 - z1) - (x3 - x1) * (z2 - z1)) / det
    g = np.hypot(gx, gy)
    sin_face = g / np.sqrt(1 + g**2)
    sin_node = np.zeros(len(x))
    cnt = np.zeros(len(x))
    for c in range(3):
        np.add.at(sin_node, tris[:, c], sin_face)
        np.add.at(cnt, tris[:, c], 1)
    return sin_face, sin_node / cnt


# ----------------------------------------------------------------------------
# netCDF helpers
# ----------------------------------------------------------------------------
def add_var(nc, name, dtype, dims, attrs, data=None, fill=None):
    kw = {}
    if fill is not None:
        kw["fill_value"] = fill
    v = nc.createVariable(name, dtype, dims, **kw)
    for k_, val in attrs.items():
        v.setncattr(k_, val)
    if data is not None:
        v[...] = data
    return v


def mesh_coord_vars(nc, lon_n, lat_n, lon_e, lat_e, lon_f, lat_f, cdo_style):
    """mesh2D_*_x/_y coordinate variables. cdo_style = forcing/scalar header order."""
    defs = {
        "mesh2D_node_x": ("nmesh2D_node", "longitude", "Longitude of mesh nodes.", "degrees_east", lon_n),
        "mesh2D_node_y": ("nmesh2D_node", "latitude", "Latitude of mesh nodes.", "degrees_north", lat_n),
        "mesh2D_edge_x": ("nmesh2D_edge", "longitude", "Characteristic longitude of mesh edges.", "degrees_east", lon_e),
        "mesh2D_edge_y": ("nmesh2D_edge", "latitude", "Characteristic latitude of mesh edges.", "degrees_north", lat_e),
        "mesh2D_face_x": ("nmesh2D_face", "longitude", "Characteristic longitude of mesh faces.", "degrees_east", lon_f),
        "mesh2D_face_y": ("nmesh2D_face", "latitude", "Characteristic latitude of mesh faces.", "degrees_north", lat_f),
    }
    order = (["mesh2D_face_x", "mesh2D_face_y", "mesh2D_node_x", "mesh2D_node_y", "mesh2D_edge_x", "mesh2D_edge_y"]
             if cdo_style else
             ["mesh2D_node_x", "mesh2D_node_y", "mesh2D_edge_x", "mesh2D_edge_y"])
    for name in order:
        d, sn, ln, un, data = defs[name]
        add_var(nc, name, "f4", (d,), {"standard_name": sn, "long_name": ln, "units": un}, data)
    return defs


def connectivity_vars(nc, edges, face_nodes, face_edges, edge_faces, face_links, with_missing):
    add_var(nc, "mesh2D_edge_nodes", "i4", ("nmesh2D_edge", "Two"),
            ({"long_name": "Maps every edge/link to two nodes that it connects.",
              "cf_role": "edge_node_connectivity", "start_index": np.int32(0)} if with_missing else
             {"cf_role": "edge_node_connectivity",
              "long_name": "Maps every edge/link to two nodes that it connects.",
              "start_index": np.int32(0)}), edges)
    add_var(nc, "mesh2D_face_nodes", "i4", ("nmesh2D_face", "nmesh2D_vertex"),
            ({"long_name": "Maps every face to its corner nodes.",
              "cf_role": "face_node_connectivity", "start_index": np.int32(0)} if with_missing else
             {"cf_role": "face_node_connectivity",
              "long_name": "Maps every face to its corner nodes.",
              "start_index": np.int32(0)}), face_nodes)
    if with_missing:
        a_fe = {"long_name": "Maps every face to its edges.", "missing_value": np.int32(FILL_FACE_EDGES),
                "cf_role": "face_edge_connectivity", "start_index": np.int32(0)}
        a_ef = {"long_name": "neighbor faces for edges", "missing_value": np.int32(FILL_EDGE_FACE),
                "cf_role": "edge_face_connectivity", "start_index": np.int32(0),
                "comment": "missing neighbor faces are indicated using _FillValue"}
        a_fl = {"long_name": "Indicates which other faces neighbor each face",
                "missing_value": np.int32(FILL_FACE_LINKS), "cf_role": "face_face_connectivity",
                "start_index": np.int32(0), "flag_values": np.int32(-1), "flag_meanings": "out_of_mesh"}
    else:
        a_fe = {"cf_role": "face_edge_connectivity", "long_name": "Maps every face to its edges.",
                "start_index": np.int32(0)}
        a_ef = {"cf_role": "edge_face_connectivity", "long_name": "neighbor faces for edges",
                "start_index": np.int32(0),
                "comment": "missing neighbor faces are indicated using _FillValue"}
        a_fl = {"cf_role": "face_face_connectivity",
                "long_name": "Indicates which other faces neighbor each face",
                "start_index": np.int32(0), "flag_values": np.int32(-1), "flag_meanings": "out_of_mesh"}
    add_var(nc, "mesh2D_face_edges", "i4", ("nmesh2D_face", "nmesh2D_vertex"), a_fe, face_edges,
            fill=np.int32(FILL_FACE_EDGES))
    add_var(nc, "mesh2D_edge_face_links", "i4", ("nmesh2D_edge", "Two"), a_ef, edge_faces,
            fill=np.int32(FILL_EDGE_FACE))
    add_var(nc, "mesh2D_face_links", "i4", ("nmesh2D_face", "nmesh2D_vertex"), a_fl, face_links,
            fill=np.int32(FILL_FACE_LINKS))


def mesh2d_topology_var(nc):
    v = nc.createVariable("mesh2D", "i4", ())
    for k_, val in [("cf_role", "mesh_topology"),
                    ("long_name", "Topology data of 2D unstructured mesh"),
                    ("topology_dimension", np.int32(2)),
                    ("node_coordinates", "mesh2D_node_x mesh2D_node_y"),
                    ("edge_coordinates", "mesh2D_edge_x mesh2D_edge_y"),
                    ("edge_node_connectivity", "mesh2D_edge_nodes"),
                    ("face_edge_connectivity", "mesh2D_face_edges"),
                    ("edge_face_connectivity", "mesh2D_edge_face_links"),
                    ("face_face_connectivity", "mesh2D_face_links"),
                    ("face_coordinates", "mesh2D_face_x mesh2D_face_y"),
                    ("face_node_connectivity", "mesh2D_face_nodes")]:
        v.setncattr(k_, val)


def time_attrs(bounds_name, std=True):
    a = {}
    if std:
        a["standard_name"] = "time"
    a.update({"long_name": "Time axis", "calendar": "360_day",
              "units": "days since 2005-01-01 00:00:00",
              "time_origin": "2005-01-01 00:00:00", "bounds": bounds_name})
    return a


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nx", type=int, default=41)
    ap.add_argument("--ny", type=int, default=31)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--outdir", default="templates")
    ap.add_argument("--prefix", default="template_ssp370_c003_m005_v2")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    x, y, tris, n_cw = build_mesh(args.nx, args.ny, rng)
    nn, nf = len(x), len(tris)
    edges, face_edges, edge_faces, face_links, b_edges, b_nodes = build_topology(tris, nn)
    ne = len(edges)

    neighbours = [set() for _ in range(nn)]
    for a, b in edges:
        neighbours[a].add(b)
        neighbours[b].add(a)

    lon_n, lat_n = epsg3031_to_lonlat(x, y)
    xe, ye = x[edges].mean(axis=1), y[edges].mean(axis=1)
    lon_e, lat_e = epsg3031_to_lonlat(xe, ye)
    xf, yf = node_to_face(x, tris), node_to_face(y, tris)
    lon_f, lat_f = epsg3031_to_lonlat(xf, yf)
    area = tri_area(x, y, tris)

    nt = len(YEARS)
    t_inst = np.array([(yr - T0_YEAR) * 360.0 for yr in YEARS])
    # states: instantaneous snapshots; bounds = (t, t) as XIOS writes for instant ops
    tb_inst = np.stack([t_inst, t_inst], axis=1)
    # forcing/scalars: annual means over the year ending at each snapshot
    t_mean = t_inst - 180.0
    tb_mean = np.stack([t_inst - 360.0, t_inst], axis=1)

    states = [synth_state(x, y, tris, k, neighbours) for k in range(nt)]

    # dhdt from consecutive snapshots (first step: forward difference)
    H_all = np.array([s["h"] for s in states])
    dhdt_all = np.gradient(H_all, axis=0)  # per year

    # ------------------------------------------------------------------ STATES
    sp = os.path.join(args.outdir, f"{args.prefix}_states.nc")
    with Dataset(sp, "w", format="NETCDF4") as nc:
        nc.createDimension("axis_nbounds", 2)
        nc.createDimension("Two", 2)
        nc.createDimension("nmesh2D_node", nn)
        nc.createDimension("nmesh2D_edge", ne)
        nc.createDimension("nmesh2D_face", nf)
        nc.createDimension("nmesh2D_vertex", 3)
        nc.createDimension("time", None)

        mesh2d_topology_var(nc)
        mesh_coord_vars(nc, lon_n, lat_n, lon_e, lat_e, lon_f, lat_f, cdo_style=False)
        # original-header order: edge_nodes, face_x/y, face_nodes, face_edges, edge_face_links, face_links
        add_var(nc, "mesh2D_edge_nodes", "i4", ("nmesh2D_edge", "Two"),
                {"cf_role": "edge_node_connectivity",
                 "long_name": "Maps every edge/link to two nodes that it connects.",
                 "start_index": np.int32(0)}, edges)
        add_var(nc, "mesh2D_face_x", "f4", ("nmesh2D_face",),
                {"standard_name": "longitude", "long_name": "Characteristic longitude of mesh faces.",
                 "units": "degrees_east"}, lon_f)
        add_var(nc, "mesh2D_face_y", "f4", ("nmesh2D_face",),
                {"standard_name": "latitude", "long_name": "Characteristic latitude of mesh faces.",
                 "units": "degrees_north"}, lat_f)
        add_var(nc, "mesh2D_face_nodes", "i4", ("nmesh2D_face", "nmesh2D_vertex"),
                {"cf_role": "face_node_connectivity", "long_name": "Maps every face to its corner nodes.",
                 "start_index": np.int32(0)}, tris)
        add_var(nc, "mesh2D_face_edges", "i4", ("nmesh2D_face", "nmesh2D_vertex"),
                {"cf_role": "face_edge_connectivity", "long_name": "Maps every face to its edges.",
                 "start_index": np.int32(0)}, face_edges, fill=np.int32(FILL_FACE_EDGES))
        add_var(nc, "mesh2D_edge_face_links", "i4", ("nmesh2D_edge", "Two"),
                {"cf_role": "edge_face_connectivity", "long_name": "neighbor faces for edges",
                 "start_index": np.int32(0),
                 "comment": "missing neighbor faces are indicated using _FillValue"},
                edge_faces, fill=np.int32(FILL_EDGE_FACE))
        add_var(nc, "mesh2D_face_links", "i4", ("nmesh2D_face", "nmesh2D_vertex"),
                {"cf_role": "face_face_connectivity",
                 "long_name": "Indicates which other faces neighbor each face",
                 "start_index": np.int32(0), "flag_values": np.int32(-1),
                 "flag_meanings": "out_of_mesh"}, face_links, fill=np.int32(FILL_FACE_LINKS))

        add_var(nc, "time_instant", "f8", ("time",), time_attrs("time_instant_bounds"), t_inst)
        add_var(nc, "time_instant_bounds", "f8", ("time", "axis_nbounds"), {}, tb_inst)
        ta = time_attrs("time_bounds")
        ta = {"axis": "T", **ta}
        add_var(nc, "time", "f8", ("time",), ta, t_inst)
        add_var(nc, "time_bounds", "f8", ("time", "axis_nbounds"), {}, tb_inst)

        def once(name, loc, units, data, std=None):
            a = {}
            if std:
                a["standard_name"] = std
            a["units"] = units
            a.update({"mesh": "mesh2D", "location": loc, "online_operation": "once",
                      "coordinates": ("mesh2D_face_y mesh2D_face_x" if loc == "face"
                                      else "mesh2D_node_y mesh2D_node_x")})
            d = "nmesh2D_face" if loc == "face" else "nmesh2D_node"
            add_var(nc, name, "f4", (d,), a, data)

        def inst(name, loc, data, units=None, std=None, long=None):
            a = {}
            if std:
                a["standard_name"] = std
            if long:
                a["long_name"] = long
            if units is not None:
                a["units"] = units
            a.update({"mesh": "mesh2D", "location": loc, "online_operation": "instant",
                      "interval_operation": "1 yr", "interval_write": "1 yr",
                      "cell_methods": "time: point",
                      "coordinates": ("time_instant mesh2D_face_y mesh2D_face_x" if loc == "face"
                                      else "time_instant mesh2D_node_y mesh2D_node_x")})
            d = "nmesh2D_face" if loc == "face" else "nmesh2D_node"
            add_var(nc, name, "f8", ("time", d), a, data)

        F = lambda key: np.array([node_to_face(s[key], tris) for s in states])
        Nd = lambda key: np.array([s[key] for s in states])
        fr = [face_fractions(s, tris) for s in states]
        sftgif = np.array([f[0] for f in fr])
        sftgrf = np.array([f[1] for f in fr])
        sftflf = np.array([f[2] for f in fr])

        taub_face = np.array([node_to_face(s["taub_node"], tris) for s in states])
        speed_face = np.hypot(F("ssavelocity_x"), F("ssavelocity_y"))
        h_face = F("h")
        gl_face = ((sftgrf > 0) & (sftflf > 0)).astype(float)
        ligroundf = gl_face * speed_face            # m a-1 (flux per unit width / H)
        # calving front: faces with some ice and some ice-free nodes
        front_face = ((sftgif > 0) & (sftgif < 1)).astype(float)
        cff = front_face * speed_face

        once("cell_area", "face", "m2", area)
        once("x", "node", "m", x, std="projection_x_coordinate")
        inst("h_elem", "face", h_face, "m", "land_ice_thickness")
        once("y", "node", "m", y, std="projection_y_coordinate")
        inst("zs_elem", "face", F("zs"), "m", "surface_altitude")
        inst("h", "node", Nd("h"), "m", "land_ice_thickness")
        inst("zb_elem", "face", F("zb"), "m", "base_altitude")
        inst("zb", "node", Nd("zb"), "m", "base_altitude")
        inst("zs", "node", Nd("zs"), "m", "surface_altitude")
        inst("bedrock_elem", "face", F("bedrock"), "m", "bedrock_altitude")
        inst("haf_elem", "face", F("haf"), "m", long="height above flotation")
        inst("bedrock", "node", Nd("bedrock"), "m", "bedrock_altitude")
        inst("sftgif", "face", sftgif, "1", "land_ice_area_fraction")
        inst("haf", "node", Nd("haf"), "m", long="height above flotation")
        inst("sftgrf", "face", sftgrf, "1", "grounded_ice_sheet_area_fraction")
        inst("sftflf", "face", sftflf, "1", "floating_ice_shelf_area_fraction")
        inst("groundedmask", "node", Nd("groundedmask"), "1")
        inst("ssavelocity_x", "node", Nd("ssavelocity_x"), "m a-1")
        inst("ssavelocity_y", "node", Nd("ssavelocity_y"), "m a-1")
        inst("ssavelocity_x_elem", "face", F("ssavelocity_x"), "m a-1")
        inst("dhdt_elem", "face", np.array([node_to_face(d, tris) for d in dhdt_all]), "m a-1",
             "tendency_of_land_ice_thickness")
        inst("ssavelocity_y_elem", "face", F("ssavelocity_y"), "m a-1")
        inst("strbasemag", "face", taub_face, "MPa")
        inst("ligroundf", "face", ligroundf, "m a-1")
        inst("dhdt", "node", dhdt_all, "m a-1", "tendency_of_land_ice_thickness")
        inst("ceff", "node", Nd("ceff"), "MPa m-1 a")
        inst("alphaRC", "node", Nd("alphaRC"))
        inst("betaRC", "node", Nd("betaRC"))
        inst("mu", "node", Nd("mu"), long="SSA mean viscosity")
        inst("tau_RC", "node", Nd("tau_RC"), "MPa")
        inst("effective_pressure", "node", Nd("effective_pressure"), "MPa")
        inst("calving_front_flux", "face", cff, "m a-1")

        nc.setncatts({"name": f"{args.prefix}_states", "description": "Created by xios",
                      "title": "Created by xios", "Conventions": "UGRID",
                      "timeStamp": "2026-Oct-14 02:29:25 GMT",
                      "uuid": "00000000-0000-0000-0000-000000000001",
                      "model_case": "AA_SSA_ISMIP7_SSP370_CESM2_C003_m005_v2_TEMPLATE",
                      "projection": "EPSG:3031", "time_calendar": "360_day",
                      "time_origin": "days since 2005-01-01",
                      "template_note": "SYNTHETIC TEMPLATE generated by make_elmer_templates.py - not model output"})

    # ----------------------------------------------------------------- FORCING
    zs_face = F("zs")
    zs_ref = zs_face[0]
    acabf_forcing = 0.6 - 0.00015 * np.clip(zs_face, 0, None)          # m ice a-1
    dacabfdz = np.full_like(acabf_forcing, 1.0e-4)                     # m ice a-1 m-1
    smb_feedback = dacabfdz * (zs_face - zs_ref)
    smb_total = acabf_forcing + smb_feedback
    acabf = smb_total * (sftgif > 0)

    salinity_n = np.array([34.4 + 0.1 * np.clip(-s["zb"], 0, 1500) / 1500 for s in states])
    tf_n = np.array([np.clip(0.2 + 1.8 * np.clip(-s["zb"], 0, 1200) / 1200 + 0.05 * k, 0, None)
                     for k, s in enumerate(states)])                   # degC above freezing
    tfreeze_n = np.array([LAMBDA1 * salinity_n[k] + LAMBDA2 + LAMBDA3 * s["zb"]
                          for k, s in enumerate(states)])
    coeff = GAMMA0 * (RHO_W * CP_W / (RHO_I * L_F)) ** 2
    bmb_n = np.array([np.where(s["floating"] & s["has_ice"], coeff * tf_n[k] * np.abs(tf_n[k]), 0.0)
                      for k, s in enumerate(states)])                  # +ve = melt
    libmassbf = -np.array([node_to_face(b, tris) for b in bmb_n]) * (sftflf > 0)   # -ve = melt

    sin_f, sin_n = zip(*[slope_sin_theta(x, y, tris, s["zb"]) for s in states])
    sin_f, sin_n = np.array(sin_f), np.array(sin_n)
    fcor_n = 2 * OMEGA * np.sin(np.deg2rad(lat_n)) * YTS                # a-1
    fcor_n = np.tile(fcor_n, (nt, 1))
    fcor_f = np.array([node_to_face(f, tris) for f in fcor_n])

    # nodal (lumped) areas for h_loads
    node_area = np.zeros(nn)
    for c in range(3):
        np.add.at(node_area, tris[:, c], area / 3)
    h_loads = dhdt_all * node_area[None, :]                            # m3 a-1

    fp = os.path.join(args.outdir, f"{args.prefix}_forcing.nc")
    with Dataset(fp, "w", format="NETCDF4") as nc:
        nc.createDimension("time", None)
        nc.createDimension("bnds", 2)
        nc.createDimension("nmesh2D_edge", ne)
        nc.createDimension("Two", 2)
        nc.createDimension("nmesh2D_vertex", 3)
        nc.createDimension("nmesh2D_face", nf)
        nc.createDimension("nmesh2D_node", nn)

        add_var(nc, "time", "f8", ("time",),
                {"standard_name": "time", "long_name": "Time axis", "bounds": "time_bnds",
                 "units": "days since 2005-01-01 00:00:00", "calendar": "360_day", "axis": "T"}, t_mean)
        add_var(nc, "time_bnds", "f8", ("time", "bnds"), {}, tb_mean)
        mesh2d_topology_var(nc)   # NOTE: absent in the cdo-subset header; present in XIOS originals
        mesh_coord_vars(nc, lon_n, lat_n, lon_e, lat_e, lon_f, lat_f, cdo_style=True)
        connectivity_vars(nc, edges, tris, face_edges, edge_faces, face_links, with_missing=True)

        def c_once(name, loc, units, data, std=None):
            a = {}
            if std:
                a["standard_name"] = std
            a["units"] = units
            a.update({"CDI_grid_type": "unstructured",
                      "coordinates": ("mesh2D_face_y mesh2D_face_x" if loc == "face"
                                      else "mesh2D_node_y mesh2D_node_x"),
                      "mesh": "mesh2D", "location": loc, "online_operation": "once"})
            d = "nmesh2D_face" if loc == "face" else "nmesh2D_node"
            add_var(nc, name, "f4", (d,), a, data)

        def avg(name, loc, data, units, std=None, long=None):
            a = {}
            if std:
                a["standard_name"] = std
            if long:
                a["long_name"] = long
            a["units"] = units
            a.update({"CDI_grid_type": "unstructured",
                      "coordinates": ("mesh2D_face_y mesh2D_face_x" if loc == "face"
                                      else "mesh2D_node_y mesh2D_node_x"),
                      "mesh": "mesh2D", "location": loc, "online_operation": "average",
                      "interval_operation": "15 d", "interval_write": "1 yr",
                      "cell_methods": "time: mean (interval: 15 d)"})
            d = "nmesh2D_face" if loc == "face" else "nmesh2D_node"
            add_var(nc, name, "f8", ("time", d), a, data)

        melt_long = "sub-shelf basal melt from calibrated ISMIP7 local-quadratic law"
        c_once("cell_area", "face", "m2", area)
        c_once("x", "node", "m", x, std="projection_x_coordinate")
        avg("acabf", "face", acabf, "m ice a-1")
        avg("libmassbf", "face", libmassbf, "m ice a-1")
        c_once("y", "node", "m", y, std="projection_y_coordinate")
        avg("bmb", "node", bmb_n, "m ice a-1", long=melt_long)
        avg("thermal_forcing_at_draft_elem", "face", np.array([node_to_face(t, tris) for t in tf_n]), "degC")
        avg("thermal_forcing_at_draft", "node", tf_n, "degC")
        avg("freezing_temperature_reference", "node", tfreeze_n, "degC")
        avg("freezing_temperature_reference_elem", "face",
            np.array([node_to_face(t, tris) for t in tfreeze_n]), "degC")
        avg("salinity_at_draft_elem", "face", np.array([node_to_face(s_, tris) for s_ in salinity_n]), "1e-3")
        avg("sin_theta_elem", "face", sin_f, "1")
        avg("salinity_at_draft", "node", salinity_n, "1e-3")
        avg("coriolis_parameter_elem", "face", fcor_f, "a-1")
        avg("sin_theta", "node", sin_n, "1")
        avg("acabf_forcing", "face", acabf_forcing, "m ice a-1")
        avg("dacabfdz_forcing", "face", dacabfdz, "m ice a-1 m-1")
        avg("smb_feedback_flux", "face", smb_feedback, "m ice a-1")
        avg("smb_total_flux", "face", smb_total, "m ice a-1")
        avg("ligroundf", "face", ligroundf, "m a-1")
        avg("coriolis_parameter", "node", fcor_n, "a-1")
        avg("calving_front_flux", "face", cff, "m a-1")
        avg("h_loads", "node", h_loads, "m3 a-1")
        avg("dhdt", "face", np.array([node_to_face(d, tris) for d in dhdt_all]), "m a-1",
            std="tendency_of_land_ice_thickness")

        nc.setncatts({"Conventions": "UGRID", "name": f"{args.prefix}_forcing",
                      "description": "Created by xios", "title": "Created by xios",
                      "timeStamp": "2026-Oct-14 02:29:32 GMT",
                      "uuid": "00000000-0000-0000-0000-000000000002",
                      "acabf_source": "TEMPLATE (synthetic, elevation-dependent)",
                      "dacabfdz_source": "TEMPLATE (constant 1e-4 m ice a-1 m-1)",
                      "thermal_forcing_source": "TEMPLATE (linear in draft depth)",
                      "salinity_source": "TEMPLATE (linear in draft depth)",
                      "melt_parameterisation": "calibrated ISMIP7 local quadratic",
                      "basin_temperature_correction": "TEMPLATE (none, deltaT = 0)",
                      "basin_temperature_correction_selection": "TEMPLATE",
                      "time_calendar": "360_day", "time_origin": "days since 2005-01-01",
                      "template_note": "SYNTHETIC TEMPLATE generated by make_elmer_templates.py - not model output"})

    # ----------------------------------------------------------------- SCALARS
    volume = (h_face * area[None, :]).sum(axis=1)                       # m3, all faces
    smb_int = (acabf * area[None, :]).sum(axis=1)                       # m3 a-1
    bmb_int = (-libmassbf * area[None, :]).sum(axis=1)                  # m3 a-1, +ve = melt

    scp = os.path.join(args.outdir, f"{args.prefix}_scalars.nc")
    with Dataset(scp, "w", format="NETCDF4") as nc:
        nc.createDimension("time", None)
        nc.createDimension("bnds", 2)
        nc.createDimension("cell", nn)
        add_var(nc, "time", "f8", ("time",),
                {"standard_name": "time", "long_name": "Time axis", "bounds": "time_bnds",
                 "units": "days since 2005-01-01 00:00:00", "calendar": "360_day", "axis": "T"}, t_mean)
        add_var(nc, "time_bnds", "f8", ("time", "bnds"), {}, tb_mean)
        add_var(nc, "lon", "f4", ("cell",), {"standard_name": "longitude", "long_name": "Longitude",
                                            "units": "degrees_east"}, lon_n)
        add_var(nc, "lat", "f4", ("cell",), {"standard_name": "latitude", "long_name": "Latitude",
                                            "units": "degrees_north"}, lat_n)
        add_var(nc, "domain_area", "f4", (), {"units": "m2", "online_operation": "once"}, area.sum())

        def sc(name, data, units, op, cm, std=None, long=None):
            a = {}
            if std:
                a["standard_name"] = std
            if long:
                a["long_name"] = long
            a.update({"units": units, "online_operation": op})
            a.update({"interval_operation": "1 yr" if op == "instant" else "15 d",
                      "interval_write": "1 yr", "cell_methods": cm})
            add_var(nc, name, "f4", ("time",), a, data)

        def cl(name, data, units, op, cm, std=None):
            a = {}
            if std:
                a["standard_name"] = std
            a.update({"units": units, "CDI_grid_type": "unstructured", "coordinates": "lat lon",
                      "online_operation": op, "interval_operation": "15 d", "interval_write": "1 yr",
                      "cell_methods": cm})
            add_var(nc, name, "f4", ("time", "cell"), a, data)

        mean_cm = "time: mean (interval: 15 d)"
        sc("volume", volume, "m3", "instant", "time: point", std="land_ice_thickness")
        sc("smb_int", smb_int, "m3 a-1", "average", mean_cm)
        sc("bmb_int", bmb_int, "m3 a-1", "average", mean_cm,
           long="sub-shelf basal melt from calibrated ISMIP7 local-quadratic law")
        cl("tloc_mean", tf_n, "degC", "average", mean_cm)
        cl("tloc_max", tf_n + 0.3 * (tf_n > 0), "degC", "maximum", "time: maximum (interval: 15 d)")
        cl("sloc_mean", salinity_n, "1e-3", "average", mean_cm)
        cl("sintheta_mean", sin_n, "1", "average", mean_cm)
        cl("fcoriolis_mean", fcor_n, "a-1", "average", mean_cm)
        cl("dhdt_max", dhdt_all + 0.1, "m a-1", "maximum", "time: maximum (interval: 15 d)",
           std="tendency_of_land_ice_thickness")
        cl("dhdt_min", dhdt_all - 0.1, "m a-1", "minimum", "time: minimum (interval: 15 d)",
           std="tendency_of_land_ice_thickness")

        nc.setncatts({"Conventions": "CF-1.6", "name": f"{args.prefix}_scalars",
                      "description": "Created by xios", "title": "Created by xios",
                      "timeStamp": "2026-Oct-14 02:29:32 GMT",
                      "uuid": "00000000-0000-0000-0000-000000000003",
                      "time_calendar": "360_day", "time_origin": "days since 2005-01-01",
                      "template_note": "SYNTHETIC TEMPLATE generated by make_elmer_templates.py - not model output"})

    # ------------------------------------------------------------------ TRUTH
    last = states[-1]
    truth = {
        "_about": "Known answers planted in the synthetic templates. A correct Elmer->pyISSM "
                  "converter should reproduce these. Indices here are 1-based (ISSM convention) "
                  "unless stated.",
        "seed": args.seed, "nx": args.nx, "ny": args.ny,
        "mesh": {
            "numberofvertices": nn, "numberofelements": nf, "numberofedges": ne,
            "n_faces_stored_clockwise_in_file": n_cw,
            "n_boundary_edges": int(len(b_edges)),
            "n_boundary_vertices": int(len(b_nodes)),
            "total_area_m2": float(area.sum()),
            "connectivity_start_index_in_file": 0,
            "boundary_neighbour_marker_in_face_links": OUT_OF_MESH,
            "boundary_marker_in_edge_face_links": FILL_EDGE_FACE,
            "projection": "EPSG:3031",
        },
        "time": {
            "calendar": "360_day",
            "states_decimal_years": [float(T0_YEAR + t / 360.0) for t in t_inst],
            "forcing_scalars_decimal_years_centre": [float(T0_YEAR + t / 360.0) for t in t_mean],
        },
        "conventions": {
            "groundedmask": "1 = grounded, 0 = grounding-line node (grounded with a floating neighbour), -1 = floating",
            "issm_ocean_levelset": "> 0 grounded, < 0 floating",
            "issm_ice_levelset": "< 0 ice present, > 0 no ice",
            "ice_free_definition": f"h <= {H_MIN} m (Elmer minimum thickness)",
            "bmb_sign": "positive = melt (m ice a-1)",
            "libmassbf_sign": "negative = melt (m ice a-1)",
            "friction_law_planted": "tau_b = tau_RC * (|u| / (|u| + u0))**(1/m); tau_RC = 0.01*N; "
                                    "alphaRC = 1/m = 1/3; betaRC = u0 = 300 m a-1; ceff = tau_b/|u|. "
                                    "PLACEHOLDER - replace with the real Elmer/Ice definitions.",
            "melt_law_planted": "bmb = gamma0*(rho_w*cp/(rho_i*L))**2 * TF*|TF|, gamma0 = 14477 m a-1",
            "units": "Elmer: m, a, MPa. ISSM: SI (Pa, s) except velocities/SMB in m/yr - check yts",
            "rho_ice": RHO_I, "rho_water": RHO_W,
        },
        "per_timestep": [],
    }
    for k in range(nt):
        st = states[k]
        truth["per_timestep"].append({
            "decimal_year": float(T0_YEAR + t_inst[k] / 360.0),
            "volume_m3_sum_h_elem_x_cell_area": float(volume[k]),
            "volume_above_floatation_m3": float((node_to_face(np.clip(st["haf"], 0, None), tris) * area).sum()),
            "smb_int_m3_per_a": float(smb_int[k]),
            "bmb_int_m3_per_a_melt_positive": float(bmb_int[k]),
            "n_vertices_grounded_incl_GL": int((st["groundedmask"] >= 0).sum()),
            "n_vertices_floating": int((st["groundedmask"] < 0).sum()),
            "n_vertices_GL": int((st["groundedmask"] == 0).sum()),
            "n_vertices_with_ice": int(st["has_ice"].sum()),
            "max_speed_m_per_a": float(np.hypot(st["ssavelocity_x"], st["ssavelocity_y"]).max()),
            "max_strbasemag_MPa": float(node_to_face(st["taub_node"], tris).max()),
            "flotation_check_max_abs_err_m": float(np.abs(st["zs"] - st["zb"] - st["h"]).max()),
        })
    with open(os.path.join(args.outdir, "template_truth.json"), "w") as fh:
        json.dump(truth, fh, indent=2)

    print(f"Wrote templates to {args.outdir}/  ({nn} nodes, {nf} faces, {ne} edges, {nt} timesteps)")
    for p in (sp, fp, scp):
        print("  ", p, f"{os.path.getsize(p)/1e6:.2f} MB")


if __name__ == "__main__":
    main()
