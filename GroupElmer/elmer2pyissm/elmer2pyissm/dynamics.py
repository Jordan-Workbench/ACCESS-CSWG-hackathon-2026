"""
Step 3 - Velocity, friction and rheology.

This is the hard physics part. Elmer/Ice and ISSM write the same physical laws
in different forms and units, so every conversion here is marked CHECK.

Units
-----
Elmer/Ice : m, a (year), MPa
pyISSM    : velocities in m/yr in md.initialization; friction C, effective
            pressure and rheology_B in SI (Pa, s); friction u0 in m/yr (pyISSM
            divides by md.constants.yts when writing the binary file).
"""

import numpy as np
import pyissm

from .mesh import face_to_node, node_to_face

MPA = 1.0e6

# CHECK: Elmer/Ice .sif files usually define the year in seconds as
# yearinsec = 365.25*24*60*60 = 31557600 s. pyISSM's default
# md.constants.yts is 365*24*3600 = 31536000 s (a 0.07 % difference). Set
# this to the value in your .sif. convert.py copies it into md.constants.yts.
YTS_ELMER = 365.25 * 24 * 3600


def build_initialization(elmer, step, verbose=True):
    st = elmer.states
    init = pyissm.model.classes.initialization()
    # CHECK: ssavelocity_x/_y are in m a-1 (units attribute), which is what
    # md.initialization expects. They are the SSA (depth-averaged) velocity,
    # so they suit an SSA ISSM model directly. For HO/FS you would also need vz
    # and a 3D mesh.
    init.vx = st.var("ssavelocity_x", step)
    init.vy = st.var("ssavelocity_y", step)
    init.vel = np.hypot(init.vx, init.vy)
    if verbose:
        print(f"[init] step {step}: max speed {init.vel.max():.1f} m/a")
    return init


# ---------------------------------------------------------------------------
# friction
# ---------------------------------------------------------------------------
def regcoulomb_taub(C, u_ma, m, u0_ma, yts=YTS_ELMER):
    """Basal shear stress magnitude [Pa] from the ISSM regcoulomb law.

    ISSM (pyissm.model.classes.friction.regcoulomb):

                   C^2 |u|^(1/m)
        tau_b = -------------------      u in m/s (SI), u0 in m/s
                 (|u|/u0 + 1)^(1/m)
    """
    u = np.abs(u_ma) / yts
    u0 = u0_ma / yts
    return C**2 * u ** (1.0 / m) / (u / u0 + 1.0) ** (1.0 / m)


def build_friction(elmer, step, info, has_ice, floating, method="invert",
                   m=3.0, u0_ma=None, yts=YTS_ELMER, verbose=True):
    """pyISSM regcoulomb friction from Elmer output.

    Two methods
    -----------
    method="invert" (default, works whatever Elmer's exact law is)
        Fix m and u0, then solve the ISSM law for C so that ISSM reproduces
        Elmer's basal stress `strbasemag` exactly at this timestep's velocity.
        Elmer's stress is on faces and is moved to nodes by area weighting.

    method="params"
        Map Elmer's regularised-Coulomb parameter fields directly. This is ONLY
        valid once the group knows exactly what tau_RC, alphaRC and betaRC mean
        (see the CHECK below). The mapping coded here is for the law
            tau_b = tau_RC * (|u| / (|u| + u0))^(1/m),
        with alphaRC = 1/m and betaRC = u0 [m/a], which is the PLACEHOLDER law
        used in the synthetic templates.
    """
    st = elmer.states
    el0, areas = info["elements0"], info["areas"]
    nv = len(st.var("x"))
    vx, vy = st.var("ssavelocity_x", step), st.var("ssavelocity_y", step)
    u = np.hypot(vx, vy)
    grounded_ice = has_ice & ~floating

    fr = pyissm.model.classes.friction.regcoulomb()

    if method == "params":
        # CHECK: What do Elmer's fields mean? Look at the friction section of the
        # .sif (e.g. "SSA Friction Law = Regularized coulomb",
        # "SSA Friction Parameter", "SSA Friction Exponent",
        # "SSA Friction Threshold Velocity") and the Elmer/Ice source for that
        # law. Then rewrite the three lines below. If Elmer's u0 or m vary in
        # space, note that ISSM's u0 is a SINGLE scalar. Use method="invert"
        # in that case.
        alpha = st.var("alphaRC", step)
        beta = st.var("betaRC", step)
        tau_rc = st.var("tau_RC", step) * MPA                 # Pa
        m_field = 1.0 / alpha
        m = float(np.median(m_field[grounded_ice]))
        u0_ma = float(np.median(beta[grounded_ice]))
        if np.ptp(m_field[grounded_ice]) > 1e-6 or np.ptp(beta[grounded_ice]) > 1e-6:
            print("[friction] WARNING: Elmer m or u0 vary in space. ISSM u0 is a scalar; use method='invert'.")
        # tau_RC * (u/(u+u0))^(1/m) == [tau_RC / u0^(1/m)] * u^(1/m) / (u/u0+1)^(1/m)
        C2 = tau_rc / (u0_ma / yts) ** (1.0 / m)
        C = np.sqrt(np.clip(C2, 0, None))
    elif method == "invert":
        if u0_ma is None:
            # CHECK: choose u0. Using Elmer's own threshold velocity (betaRC?)
            # keeps the velocity dependence of the two laws as close as possible.
            # 300 m/a is just a placeholder.
            u0_ma = float(np.median(st.var("betaRC", step)[grounded_ice])) if st.has("betaRC") else 300.0
        taub_face = st.var("strbasemag", step) * MPA          # Pa, on faces
        # CHECK: strbasemag is per face (element). Area-weighted averaging to
        # nodes smooths it across the grounding line, so expect the biggest
        # mismatches there. An alternative is to invert per element and
        # average C instead.
        taub = face_to_node(taub_face, el0, areas, nv)
        us = np.maximum(u, 1e-3) / yts                        # avoid u = 0
        C2 = taub * (us / (u0_ma / yts) + 1.0) ** (1.0 / m) / us ** (1.0 / m)
        C = np.sqrt(np.clip(C2, 0, None))
    else:
        raise ValueError("method must be 'invert' or 'params'")

    # CHECK: Floating ice should have C = 0. Ice-free nodes get the median
    # grounded C so a later ISSM run that advances the ice has a sensible
    # value. Decide whether that is what you want.
    Cmed = float(np.median(C[grounded_ice])) if grounded_ice.any() else 0.0
    C = np.where(floating & has_ice, 0.0, C)
    C = np.where(~has_ice, Cmed, C)
    fr.C = C
    fr.m = np.full(len(el0), m)          # ISSM m is per ELEMENT
    fr.u0 = u0_ma                        # m/yr (pyISSM scales by 1/yts when writing)

    # Check: recompute tau_b with ISSM's law and compare with Elmer's.
    taub_issm = regcoulomb_taub(C, u, m, u0_ma, yts)                     # Pa, nodes
    taub_issm_face = node_to_face(taub_issm, el0) / MPA
    taub_elmer_face = st.var("strbasemag", step)
    sel = (taub_elmer_face > 1e-4)
    rel = np.abs(taub_issm_face[sel] - taub_elmer_face[sel]) / taub_elmer_face[sel]
    diag = {"method": method, "m": m, "u0_ma": u0_ma,
            "taub_median_rel_err": float(np.median(rel)) if sel.any() else np.nan,
            "taub_p95_rel_err": float(np.percentile(rel, 95)) if sel.any() else np.nan,
            "taub_issm_face_MPa": taub_issm_face}
    if verbose:
        print(f"[friction] regcoulomb ({method}): m = {m:g}, u0 = {u0_ma:g} m/a, "
              f"C median (grounded) = {Cmed:.3g} SI")
        print(f"[friction] tau_b ISSM vs Elmer strbasemag: median rel err "
              f"{diag['taub_median_rel_err']:.2e}, 95th pct {diag['taub_p95_rel_err']:.2e}")
    return fr, diag


# ---------------------------------------------------------------------------
# rheology
# ---------------------------------------------------------------------------
def build_rheology_B(elmer, step, n=3.0, yts=YTS_ELMER, factor=None, verbose=True):
    """rheology_B [Pa s^(1/n)] from Elmer's `mu` (SSA mean viscosity).

    CHECK (three things):
    1. Units of `mu`. The file gives none. Elmer/Ice SSA setups normally give
       "SSA Mean Viscosity" in MPa a^(1/n). The conversion assumes that:
           Pa s^(1/n) = MPa a^(1/n) * 1e6 * yts^(1/n)
    2. Definition. Elmer/Ice usually defines the viscosity parameter as
       eta = (2A)^(-1/n), while ISSM uses B = A^(-1/n), so
       B = 2^(1/n) * eta  ->  factor = 2**(1/n) (the default here).
       Check your .sif. If it sets the viscosity some other way, change `factor`.
    3. Is `mu` the Glen parameter (independent of strain rate) or the
       effective viscosity (depends on strain rate)? If it changes a lot
       between timesteps in fast-flowing areas, it is probably the effective
       viscosity, and B must be computed from it and the strain rates instead.
    """
    if factor is None:
        factor = 2.0 ** (1.0 / n)
    mu = elmer.states.var("mu", step)
    B = factor * mu * MPA * yts ** (1.0 / n)
    if verbose:
        print(f"[rheology] B from mu: median {np.median(B):.3g} Pa s^(1/3) "
              f"(typical ice: 0.5e8 - 2.5e8)")
    return B
