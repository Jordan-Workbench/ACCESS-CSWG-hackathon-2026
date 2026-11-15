# ACCESS-CSWG Hackathon 2026

Materials for the ACCESS-NRI / Cryospheric Sciences Working Group (CSWG) hackathon: ice-sheet modelling with
**pyISSM** and the **Cryosphere Community Datapool (CCD)** on NCI's **Gadi**.

There are three groups. Choose your own adventure:

| Group | Folder | The task |
|---|---|---|
| **GroupElmer** | [`GroupElmer/`](GroupElmer/) | Port Elmer/Ice (XIOS UGRID) output into a pyISSM `Model`, so the pyISSM toolboxes work on Elmer runs. Resolve the `CHECK:` items and validate against Elmer's own diagnostics |
| **GroupVisualise** | [`GroupVisualise/`](GroupVisualise/) | Use `ccdtools` to interrogate the datapool (velocity datasets, BedMachine extremes, ice-shelf speeds) and make a publication-quality figure with IMBIE basins |
| **GroupSensitivity** | [`GroupSensitivity/`](GroupSensitivity/) | Build a Pine Island Glacier initial state, then run perturbed ensembles (grounded melt, bed, friction) to 2300 under low and high emission scenarios |

Everyone starts with the same setup below.

---

## Repository layout

```
ACCESS-CSWG-hackathon-2026/
├── README.md                     <- you are here
├── install_hackathon_env.sh      <- one-off install of pyISSM, ccdtools and extra packages
├── getting_started_gadi.ipynb    <- step-by-step setup + checks + reference material
├── GroupElmer/                   <- elmer2pyissm package, templates, tests, walkthrough notebook, slides
├── GroupVisualise/               <- ccdtools question notebook, slides
└── GroupSensitivity/             <- ensemble projection notebook, slides
```

---

## Quick start on Gadi

### 1. Before the hackathon: NCI account and projects

Sign up at [my.nci.org.au/mancini](https://my.nci.org.au/mancini) and request to join:

| Project | What for |
|---|---|
| `au88` | Hackathon project: your working folder `/g/data/au88/<username>` and compute |
| `av17` | Cryosphere Community Datapool |
| `xp65` | `conda/analysis3` Python environment |
| `vk83` | `access-issm` module (the compiled ISSM model) |

Approval can take a few days, so **do this early**.

### 2. Start an ARE JupyterLab session

Log in at [are.nci.org.au](https://are.nci.org.au) → **JupyterLab**, and set:

| Field | Value |
|---|---|
| Project | `au88` |
| Storage | `gdata/au88+gdata/av17+gdata/xp65+gdata/vk83+scratch/au88` |
| Module directories | `/g/data/vk83/modules /g/data/xp65/public/modules` |
| Modules | `access-issm/2025.11.0 conda/analysis3 gdal proj` |
| Compute size | `small` (2 cpus) is fine for Elmer/Visualise. GroupSensitivity should use 4+ cpus and a longer walltime |

### 3. Clone this repository and install (one-off)

In JupyterLab, open **File → New Launcher → Terminal** and run:

```bash
mkdir -p /g/data/au88/$USER
cd /g/data/au88/$USER

git clone https://github.com/fmccormack/ACCESS-CSWG-hackathon-2026.git
cd ACCESS-CSWG-hackathon-2026

bash install_hackathon_env.sh
```

The script installs everything into `/g/data/au88/$USER/python_packages`:
* numpy < 1.26, PyYAML, cmocean, geopandas, rasterio, rioxarray and access-ccdtools;
* source clones of **pyISSM** and **ccdtools** from GitHub (`ACCESS-NRI/pyISSM`, `ACCESS-NRI/ccdtools`), installed
  with `pip`.

It ends by printing the versions and `Install OK`.

### 4. Check your setup

Open **`getting_started_gadi.ipynb`** and select the **`analysis3`** kernel (top right). Then run the check cells.
They confirm that every package imports, the ISSM model is loaded, the datapool is readable, and your folder is
writable.

Your own notebooks need these lines at the top, because the packages live in your folder rather than the system
environment. The group notebooks already include the equivalent.

```python
import os, sys
PYBASE = f"/g/data/au88/{os.environ['USER']}/python_packages"
sys.path.insert(0, f"{PYBASE}/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages")
sys.path.append(f"{PYBASE}/pyISSM/src")
sys.path.append(f"{PYBASE}/ccdtools/src")
```

### 5. Go to your group

| Group | First steps |
|---|---|
| GroupElmer | Read `GroupElmer/README.md`, run `pytest -q tests/` on the synthetic templates, then open `notebooks/elmer_to_pyissm_walkthrough.ipynb` |
| GroupVisualise | Open `GroupVisualise/GroupVisualise_ccdtools.ipynb` and work through Q1–Q4 using the hints |
| GroupSensitivity | Copy `pig_outline.exp` into `/g/data/au88/$USER/pig_ensemble/data/`, open `GroupSensitivity/pig_perturbed_ensemble_projections.ipynb`, and run with `QUICK_TEST = True` first |

Each folder also has three short slides describing the task.

---

## Working with git during the hackathon

```bash
cd /g/data/au88/$USER/ACCESS-CSWG-hackathon-2026
git checkout -b $USER-work      # your own branch, so updates never clash with your edits
git pull origin main            # get updates the organisers push
git add <files>; git commit -m "what I did"
```

* **Keep large outputs out of the repository.** Models, netCDF files, plots and ensemble results go under
  `/g/data/au88/$USER/...`. The notebooks write there by default.
* To share your work back, push your branch and open a pull request. That needs a GitHub account with access.
  Otherwise, show it at the report-back.

---

## Long runs (PBS)

Long prognostic runs, e.g. GroupSensitivity ensembles to 2300, should be submitted as PBS jobs rather than run in
ARE. A template is in Section 6 of `getting_started_gadi.ipynb`. The essentials:

```bash
#PBS -P au88
#PBS -l storage=gdata/au88+gdata/av17+gdata/xp65+gdata/vk83+scratch/au88
module use /g/data/vk83/modules /g/data/xp65/public/modules
module load access-issm/2025.11.0 conda/analysis3 gdal proj
```

Gadi has **no PBS job arrays**. Submit one job per scenario or group, or use `nci-parallel`.

---

## Troubleshooting

| Problem | Check |
|---|---|
| ARE session won't start | Project `au88` and the Storage string exactly as above; your project memberships |
| `ModuleNotFoundError` | `analysis3` kernel; path lines at the top of the notebook; install script printed `Install OK` |
| geopandas / rasterio fail | `gdal proj` in the ARE Modules field; re-run the install script |
| ISSM not found (`ISSM_DIR` unset) | `access-issm/2025.11.0` in Modules, `/g/data/vk83/modules` in Module directories, `vk83` membership |
| Datapool / dataset not found | `av17` membership and `gdata/av17` in Storage |
| Anything else | Search the [ACCESS-Hive forum](https://forum.access-hive.org.au), then ask an organiser |

---

## Resources

* pyISSM: [docs](https://pyissm.readthedocs.io/latest/) · [GitHub](https://github.com/ACCESS-NRI/pyISSM) · [DeepWiki](https://deepwiki.com/ACCESS-NRI/pyISSM) · tutorials in `python_packages/pyISSM/tutorials/`
* ccdtools: [docs](https://ccdtools.readthedocs.io/latest/) · [GitHub](https://github.com/ACCESS-NRI/ccdtools) · [CCD dataset list](https://access-hive-docs.readthedocs.io/model_evaluation/data/cryosphere_datapool/)
* ISSM: [documentation](https://issm.jpl.nasa.gov/documentation/) · [GitHub](https://github.com/ISSMteam/ISSM)
* Gadi: [user guide](https://opus.nci.org.au/display/Help/Gadi+User+Guide) · [ARE](https://are.nci.org.au)
* Community help: [ACCESS-Hive forum](https://forum.access-hive.org.au)

## Citing

If you use ISSM, cite Larour, E., Seroussi, H., Morlighem, M. and Rignot, E. (2012), *Continental scale, high
order, high spatial resolution, ice sheet modeling using the Ice Sheet System Model (ISSM)*, JGR Earth Surface,
117, F01022, [doi:10.1029/2011JF002140](https://doi.org/10.1029/2011JF002140). Also cite the datasets you load
from the CCD. `catalog.help('<dataset>')` gives their DOIs.
