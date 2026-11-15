#!/bin/bash
# =============================================================================
# install_hackathon_env.sh
#
# Installs pyISSM, ccdtools and the extra Python packages needed for the
# ACCESS-CSWG hackathon into your own folder on Gadi:
#
#     /g/data/au88/$USER/python_packages
#
# Run it ONCE from a terminal on Gadi (e.g. a Terminal inside an ARE JupyterLab session):
#
#     bash install_hackathon_env.sh
#
# Then, in every notebook, make Python look in that folder (see the README or
# getting_started_gadi.ipynb for the two lines to add).
# =============================================================================

module purge

# Load conda analysis3
module use /g/data/xp65/public/modules
module load conda/analysis3

# Load inbuilt libraries for rasterio, pyproj, and geopandas
module load gdal
module load proj

# Everything is installed under your own au88 folder ($USER is your NCI username)
export PYTHONUSERBASE=/g/data/au88/${USER}/python_packages

# site-packages folder that matches the Python version of conda/analysis3
PYVER=$(python -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
SITE_PACKAGES=${PYTHONUSERBASE}/lib/python${PYVER}/site-packages
mkdir -p "${SITE_PACKAGES}"
echo "Installing into ${SITE_PACKAGES}"

# Install
pip install --target="${SITE_PACKAGES}" \
            --upgrade --no-cache-dir --force-reinstall \
            "numpy<1.26.0" \
            PyYAML \
            cmocean \
            geopandas \
            rasterio \
            rioxarray \
            access-ccdtools

# Get the pyISSM and ccdtools source code (skipped if you already have it)
for repo in pyISSM ccdtools; do
    if [ ! -d "${PYTHONUSERBASE}/${repo}" ]; then
        git clone "https://github.com/ACCESS-NRI/${repo}.git" "${PYTHONUSERBASE}/${repo}"
    fi
done

cd "${PYTHONUSERBASE}/pyISSM"
pip install --user .

cd "${PYTHONUSERBASE}/ccdtools"
pip install --user .

# Quick check
export PYTHONPATH="${SITE_PACKAGES}:${PYTHONPATH}"
python - <<'EOF'
import pyissm, ccdtools, geopandas, rasterio, rioxarray, cmocean
print("pyISSM  ", pyissm.__version__, "->", pyissm.__file__)
print("ccdtools", ccdtools.__version__, "->", ccdtools.__file__)
print("Install OK")
EOF
