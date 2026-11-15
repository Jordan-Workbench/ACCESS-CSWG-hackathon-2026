#!/usr/bin/env python3
"""
compare_headers.py
==================

Compare the structure of two netCDF files (or saved `ncdump -h` text) and
report differences in dimensions, variables, dtypes, variable dimensions and
attributes. Dimension sizes are ignored, since the templates use a small mesh
and fewer timesteps.

Typical uses
------------
  # Check a template against the real file's header (no need to copy 1.9 GB):
  ncdump -h ssp370_c003_m005_v2_states.nc > states_header.cdl
  python compare_headers.py states_header.cdl templates/template_ssp370_c003_m005_v2_states.nc

  # Or compare two .nc files directly:
  python compare_headers.py real_states.nc templates/template_..._states.nc

Each argument may be a .nc file (read with netCDF4) or a text file holding
`ncdump -h` output (shell prompt lines before "netcdf ..." are ignored).

Exit code is 0 if the structures match (apart from the ignorable attributes
listed in IGNORE_GLOBAL_ATTRS), 1 otherwise.
"""

import re
import sys

IGNORE_GLOBAL_ATTRS = {"name", "uuid", "timeStamp", "template_note", "history", "CDI", "CDO",
                       "model_case", "acabf_source", "dacabfdz_source", "thermal_forcing_source",
                       "salinity_source", "basin_temperature_correction",
                       "basin_temperature_correction_selection"}

CDL_TYPES = {"float": "f4", "double": "f8", "int": "i4", "short": "i2", "byte": "i1",
             "char": "S1", "int64": "i8", "ubyte": "u1", "ushort": "u2", "uint": "u4",
             "uint64": "u8", "string": "str"}


def read_cdl(path):
    text = open(path).read()
    m = re.search(r"^netcdf\s+\S+\s*\{", text, re.M)
    if not m:
        raise ValueError(f"{path}: no 'netcdf ... {{' line found")
    text = text[m.start():]
    end = text.find("\n}")
    text = text[: end + 2] if end >= 0 else text

    dims, vars_, gattrs = {}, {}, {}
    section = None
    current = None
    for raw in text.splitlines()[1:]:
        line = raw.strip()
        if not line or line == "}":
            continue
        if line.startswith("dimensions:"):
            section = "dims"; continue
        if line.startswith("variables:"):
            section = "vars"; continue
        if line.startswith("// global attributes:"):
            section = "global"; continue
        if section == "dims":
            dm = re.match(r"(\S+)\s*=\s*(\S+)", line)
            if dm:
                dims[dm.group(1)] = dm.group(2).rstrip(";").strip()
        elif section == "vars":
            vm = re.match(r"(\w+)\s+([\w]+)\s*(\(([^)]*)\))?\s*;", line)
            am = re.match(r"(\w+):(\w+)\s*=", line)
            if vm and not am and vm.group(1) in CDL_TYPES:
                current = vm.group(2)
                vd = tuple(d.strip() for d in vm.group(4).split(",")) if vm.group(4) else ()
                vars_[current] = {"dtype": CDL_TYPES[vm.group(1)], "dims": vd, "attrs": set()}
            elif am and am.group(1) in vars_:
                vars_[am.group(1)]["attrs"].add(am.group(2))
        elif section == "global":
            gm = re.match(r":(\w+)\s*=", line)
            if gm:
                gattrs[gm.group(1)] = True
    return dims, vars_, set(gattrs)


def read_nc(path):
    from netCDF4 import Dataset
    with Dataset(path) as nc:
        dims = {d: ("UNLIMITED" if nc.dimensions[d].isunlimited() else len(nc.dimensions[d]))
                for d in nc.dimensions}
        vars_ = {}
        for name, v in nc.variables.items():
            dt = v.dtype.str.lstrip("<>|=") if hasattr(v.dtype, "str") else "str"
            vars_[name] = {"dtype": dt, "dims": tuple(v.dimensions), "attrs": set(v.ncattrs())}
        gattrs = set(nc.ncattrs())
    return dims, vars_, gattrs


def read_any(path):
    if path.endswith((".nc", ".nc4")):
        return read_nc(path)
    return read_cdl(path)


def compare(a_path, b_path):
    da, va, ga = read_any(a_path)
    db, vb, gb = read_any(b_path)
    problems = []

    def diff(label, sa, sb):
        only_a, only_b = sorted(sa - sb), sorted(sb - sa)
        if only_a:
            problems.append(f"{label} only in REFERENCE: {only_a}")
        if only_b:
            problems.append(f"{label} only in CANDIDATE: {only_b}")

    diff("dimensions", set(da), set(db))
    diff("variables", set(va), set(vb))
    for name in sorted(set(va) & set(vb)):
        a, b = va[name], vb[name]
        if a["dtype"] != b["dtype"]:
            problems.append(f"{name}: dtype {a['dtype']} vs {b['dtype']}")
        if a["dims"] != b["dims"]:
            problems.append(f"{name}: dims {a['dims']} vs {b['dims']}")
        missing = a["attrs"] - b["attrs"]
        extra = b["attrs"] - a["attrs"]
        if missing:
            problems.append(f"{name}: attrs missing in CANDIDATE {sorted(missing)}")
        if extra:
            problems.append(f"{name}: extra attrs in CANDIDATE {sorted(extra)}")
    diff("global attrs", ga - IGNORE_GLOBAL_ATTRS, gb - IGNORE_GLOBAL_ATTRS)

    print(f"REFERENCE: {a_path}\nCANDIDATE: {b_path}")
    print(f"  {len(va)} vs {len(vb)} variables, {len(da)} vs {len(db)} dimensions")
    if problems:
        print(f"  {len(problems)} difference(s):")
        for p in problems:
            print("   -", p)
        return 1
    print("  structures match")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(compare(sys.argv[1], sys.argv[2]))
