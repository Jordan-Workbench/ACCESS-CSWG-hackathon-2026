"""
pytest suite: run the converter on the synthetic templates and check every
planted answer in template_truth.json.

    pytest -q tests/

All of these should pass BEFORE anyone changes the converter. If you change
a convention (e.g. after confirming the real Elmer friction law), update the
template generator and regenerate the templates so the tests keep meaning
something.
"""

import json
import os

import numpy as np
import pytest

import elmer2pyissm
from elmer2pyissm.mesh import signed_area

HERE = os.path.dirname(__file__)
TDIR = os.path.join(HERE, "..", "templates", "templates")
PFX = os.path.join(TDIR, "template_ssp370_c003_m005_v2")
TRUTH = json.load(open(os.path.join(TDIR, "template_truth.json")))


@pytest.fixture(scope="module")
def converted():
    md, report = elmer2pyissm.elmer_to_pyissm(PFX + "_states.nc", PFX + "_forcing.nc",
                                              PFX + "_scalars.nc", step=-1, verbose=False)
    return md, report


def test_mesh_counts(converted):
    md, _ = converted
    assert md.mesh.numberofvertices == TRUTH["mesh"]["numberofvertices"]
    assert md.mesh.numberofelements == TRUTH["mesh"]["numberofelements"]
    assert md.mesh.numberofedges == TRUTH["mesh"]["numberofedges"]


def test_one_based_and_ccw(converted):
    md, _ = converted
    assert md.mesh.elements.min() == 1
    assert md.mesh.elements.max() == md.mesh.numberofvertices
    assert (signed_area(md.mesh.x, md.mesh.y, md.mesh.elements - 1) > 0).all()


def test_clockwise_faces_flipped(converted):
    _, rep = converted
    assert rep["mesh_info"]["n_clockwise_flipped"] == TRUTH["mesh"]["n_faces_stored_clockwise_in_file"]


def test_boundary(converted):
    md, _ = converted
    assert int(md.mesh.vertexonboundary.sum()) == TRUTH["mesh"]["n_boundary_vertices"]
    assert len(md.mesh.segments) == TRUTH["mesh"]["n_boundary_edges"]


def test_area(converted):
    _, rep = converted
    assert np.isclose(rep["mesh_info"]["areas"].sum(), TRUTH["mesh"]["total_area_m2"], rtol=1e-9)


def test_masks_last_step(converted):
    md, _ = converted
    p = TRUTH["per_timestep"][-1]
    ol = md.mask.ocean_levelset
    assert int((ol >= 0).sum()) == p["n_vertices_grounded_incl_GL"]
    assert int((ol < 0).sum()) == p["n_vertices_floating"]
    assert int((ol == 0).sum()) == p["n_vertices_GL"]
    assert int((md.mask.ice_levelset < 0).sum()) == p["n_vertices_with_ice"]


def test_geometry_consistent(converted):
    md, _ = converted
    g = md.geometry
    assert np.abs(g.surface - g.base - g.thickness).max() < 1e-6


def test_velocity(converted):
    md, _ = converted
    assert np.isclose(md.initialization.vel.max(), TRUTH["per_timestep"][-1]["max_speed_m_per_a"])


def test_friction_reproduces_taub(converted):
    _, rep = converted
    # 'invert' reproduces strbasemag up to node<->face averaging
    assert rep["friction_diag"]["taub_median_rel_err"] < 0.05


def test_results_timeseries(converted):
    md, _ = converted
    TS = md.results.TransientSolution
    years = [p["decimal_year"] for p in TRUTH["per_timestep"]]
    assert np.allclose(TS.time, years)
    assert TS.Thickness.shape == (len(years), TRUTH["mesh"]["numberofvertices"])


@pytest.mark.parametrize("k", range(5))
def test_integrated_scalars(converted, k):
    md, _ = converted
    TS = md.results.TransientSolution
    p = TRUTH["per_timestep"][k]
    rho_i = md.materials.rho_ice
    assert np.isclose(TS.IceVolume[k], p["volume_m3_sum_h_elem_x_cell_area"], rtol=1e-6)
    assert np.isclose(TS.TotalSmb[k] * 1e12 / rho_i, p["smb_int_m3_per_a"], rtol=1e-6)
    assert np.isclose(TS.TotalFloatingBmb[k] * 1e12 / rho_i, p["bmb_int_m3_per_a_melt_positive"], rtol=1e-6)


def test_validate_has_no_failures(converted):
    md, rep = converted
    rows = elmer2pyissm.validate(md, rep["elmer"], rep,
                                 truth_json=os.path.join(TDIR, "template_truth.json"), verbose=False)
    fails = [r for r in rows if r[4] == "FAIL"]
    assert not fails, fails


def test_save_load_roundtrip(converted, tmp_path):
    import pyissm
    md, _ = converted
    out = tmp_path / "md.nc"
    pyissm.model.io.save_model(md, str(out))
    md2 = pyissm.model.io.load_model(str(out))
    TS2 = md2.results.TransientSolution
    assert TS2.IceVolume.dtype.kind == "f"   # not strings (see results._scalar)
    assert np.allclose(TS2.IceVolume, md.results.TransientSolution.IceVolume)
    assert np.allclose(md2.geometry.thickness, md.geometry.thickness)
