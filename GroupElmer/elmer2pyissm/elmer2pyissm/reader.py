"""
Step 0 - Read Elmer/Ice + XIOS netCDF output.

`ElmerOutput` wraps the three files (states, forcing, scalars) and hands back
plain numpy arrays one timestep at a time, so the 1-2 GB files never have to
fit in memory at once.

Search for "CHECK:" to find every place the group must verify against the
real files.
"""

import re

import numpy as np
from netCDF4 import Dataset


def _decimal_years(time_var):
    """Convert an XIOS time variable to decimal years.

    CHECK: The Elmer runs use calendar = "360_day" and
    units = "days since 2005-01-01 00:00:00". With a 360_day calendar every
    year is exactly 360 days, so decimal_year = ref_year + days / 360.
    If a file has a different calendar or units, this function raises
    rather than guessing.
    """
    units = time_var.getncattr("units")
    cal = time_var.getncattr("calendar") if "calendar" in time_var.ncattrs() else "standard"
    m = re.match(r"days since (\d{4})-(\d{2})-(\d{2})", units)
    if not m:
        raise ValueError(f"Unexpected time units '{units}'")
    if cal != "360_day":
        raise ValueError(f"Expected a 360_day calendar, found '{cal}'. Update _decimal_years().")
    y0, mo0, d0 = (int(g) for g in m.groups())
    t0 = y0 + (mo0 - 1) / 12.0 + (d0 - 1) / 360.0
    return t0 + np.asarray(time_var[:], dtype=float) / 360.0


class ElmerFile:
    """Thin wrapper around one netCDF file with masking switched off."""

    def __init__(self, path):
        self.path = path
        self.nc = Dataset(path)
        # Masking OFF: connectivity arrays use _FillValue (999999 / -999) as
        # real markers, and auto-masking would turn them into masked values.
        self.nc.set_auto_mask(False)

        # Pick the time coordinate.
        # CHECK: states files carry both `time_instant` (snapshot times) and
        # `time` (here identical). For instant fields `time_instant` is the
        # physically correct one. For averaged fields (forcing, scalars) XIOS
        # writes `time` as the CENTRE of the averaging interval. Confirm with
        # `ncdump -v time,time_bounds` on the real files.
        if "time_instant" in self.nc.variables:
            tname = "time_instant"
        elif "time" in self.nc.variables:
            tname = "time"
        else:
            tname = None
        self.time_name = tname
        self.years = _decimal_years(self.nc.variables[tname]) if tname else np.array([])

    @property
    def ntimes(self):
        return len(self.years)

    def has(self, name):
        return name in self.nc.variables

    def var(self, name, step=None, dtype=float):
        """Return a variable; for time-dependent ones, only `step` (default: all)."""
        v = self.nc.variables[name]
        if step is not None and v.dimensions and v.dimensions[0] == "time":
            data = v[step, ...]
        else:
            data = v[...]
        return np.asarray(data, dtype=dtype)

    def location(self, name):
        """'node', 'face' or None, from the UGRID `location` attribute or the dims."""
        v = self.nc.variables[name]
        if "location" in v.ncattrs():
            return v.getncattr("location")
        if "nmesh2D_node" in v.dimensions or "cell" in v.dimensions:
            return "node"
        if "nmesh2D_face" in v.dimensions:
            return "face"
        return None

    def units(self, name):
        v = self.nc.variables[name]
        return v.getncattr("units") if "units" in v.ncattrs() else None

    def global_attrs(self):
        return {k: self.nc.getncattr(k) for k in self.nc.ncattrs()}

    def close(self):
        self.nc.close()


class ElmerOutput:
    """The three Elmer/Ice output files of one run, opened together.

    Parameters
    ----------
    states, forcing, scalars : str
        Paths. forcing and scalars are optional (None skips them).

    Notes
    -----
    CHECK: Use the ORIGINAL multi-timestep files where possible. `cdo
    seltimestep` drops the `mesh2D` topology variable and `time_instant`.
    The connectivity arrays survive, so this reader still works, but other
    UGRID tools (xugrid, uxarray) will not.
    """

    def __init__(self, states, forcing=None, scalars=None):
        self.states = ElmerFile(states)
        self.forcing = ElmerFile(forcing) if forcing else None
        self.scalars = ElmerFile(scalars) if scalars else None
        self._check_same_mesh()

    def _check_same_mesh(self):
        nn = len(self.states.nc.dimensions["nmesh2D_node"])
        nf = len(self.states.nc.dimensions["nmesh2D_face"])
        if self.forcing is not None:
            assert len(self.forcing.nc.dimensions["nmesh2D_node"]) == nn, "forcing/states node count differ"
            assert len(self.forcing.nc.dimensions["nmesh2D_face"]) == nf, "forcing/states face count differ"
            # CHECK: equal sizes do not prove the meshes are the same mesh in the
            # same order. Compare face_nodes from both files (see check_same_connectivity).
        if self.scalars is not None and "cell" in self.scalars.nc.dimensions:
            assert len(self.scalars.nc.dimensions["cell"]) == nn, \
                "scalars 'cell' dimension is expected to be the node dimension"

    def check_same_connectivity(self):
        """True if forcing and states have identical face_nodes (same order)."""
        if self.forcing is None or not self.forcing.has("mesh2D_face_nodes"):
            return None
        a = self.states.var("mesh2D_face_nodes", dtype=np.int64)
        b = self.forcing.var("mesh2D_face_nodes", dtype=np.int64)
        return bool(np.array_equal(a, b))

    def summary(self):
        lines = []
        for label, f in (("states", self.states), ("forcing", self.forcing), ("scalars", self.scalars)):
            if f is None:
                continue
            lines.append(f"{label:8s} {f.path}")
            lines.append(f"         times ({f.time_name}): {np.round(f.years, 3).tolist()}")
        nn = len(self.states.nc.dimensions["nmesh2D_node"])
        nf = len(self.states.nc.dimensions["nmesh2D_face"])
        lines.append(f"mesh: {nn} nodes, {nf} faces")
        return "\n".join(lines)

    def close(self):
        for f in (self.states, self.forcing, self.scalars):
            if f is not None:
                f.close()
