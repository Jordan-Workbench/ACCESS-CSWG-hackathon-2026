"""
elmer2pyissm - convert Elmer/Ice (XIOS UGRID) output into a pyISSM Model.

Steps (one module each):
  0 reader.py    open the states / forcing / scalars netCDF files
  1 mesh.py      UGRID mesh -> md.mesh (mesh2d)
  2 geometry.py  md.geometry, md.mask
  3 dynamics.py  md.initialization, md.friction, md.materials.rheology_B
  4 forcing.py   md.smb, md.basalforcings, provenance notes
  5 results.py   md.results.TransientSolution
  6 validate.py  checks against the Elmer scalars file (and template truth)

`grep -rn "CHECK:" elmer2pyissm/` lists every decision the group must verify.
"""

import os
import warnings

# pyISSM's default cluster class reads $USER at Model() creation and fails
# without it (e.g. in some containers or CI jobs).
os.environ.setdefault("USER", "elmer2pyissm")
# Without an ISSM build pyISSM warns about missing wrappers. The converter
# does not need them.
warnings.filterwarnings("ignore", message=".*ISSM_DIR.*")
warnings.filterwarnings("ignore", message=".*Python wrappers not installed.*")

from .convert import elmer_to_pyissm  # noqa: E402
from .reader import ElmerOutput        # noqa: E402
from .validate import validate         # noqa: E402

__all__ = ["elmer_to_pyissm", "ElmerOutput", "validate"]
