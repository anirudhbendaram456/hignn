from cmath import sqrt
import os
import hignn
import numpy as np
import sys
from mpi4py import MPI
import time
import h5py
from scipy.spatial import cKDTree
import torch
from predict_force_models import load_model, predict_force_pN, predict_eo_flow_force_N

os.system('clear')

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VDW_MODEL_PATH = os.path.join(BASE_DIR, "checkpoints", "F_vdw_scalar_pN_model.pt")
ELEC_MODEL_PATH = os.path.join(BASE_DIR, "checkpoints", "F_elec_scalar_pN_model.pt")
FLOW_MODEL_PATH = os.path.join(BASE_DIR, "checkpoints", "eo_pair_radial_mlp_raw.pt")

# ============================================================
# MOBILITY-BASED BROWNIAN MOTION
# ============================================================

USE_BROWNIAN = True
USE_HIGNN_MOBILITY_BROWNIAN = True

BROWNIAN_SEED = 12345
BROWNIAN_TEMP_K = 293.15

# length scale: 1 sim length = 30 nm
BROWNIAN_L0 = 30e-9

# Brownian only in xy for your fixed-z setup
BROWNIAN_ACTIVE_AXES = (True, True, False)

# safety limiter
BROWNIAN_MAX_RMS_SIM = 0.05   # 0.05 sim = 1.5 nm

REMOVE_BROWNIAN_COM_KICK = False

# ============================================================
# TIME-SCALE-BASED FORCE CONVERSION
# ============================================================

USE_TIME_SCALE_FORCE_CONVERSION = True

# Calibration from matching closest-approach time:
# HIGNN:    t = 38.068820 simulation time
# PyStokes: t = 30629.377 microseconds
HIGNN_CLOSEST_TIME = 0.00352669
PYSTOKES_CLOSEST_TIME_S = 3535.903e-6

HIGNN_TIME_UNIT_S = PYSTOKES_CLOSEST_TIME_S / HIGNN_CLOSEST_TIME

# force scale implied by the calibrated time scale
HIGNN_ETA_PHYS = 1e-3
HIGNN_RADIUS_PHYS = 30e-9
HIGNN_SELF_MOBILITY_SIM = 1.0

HIGNN_FORCE_SCALE_TIME_N = (
    6.0 * np.pi * HIGNN_ETA_PHYS * HIGNN_RADIUS_PHYS
    * BROWNIAN_L0 * HIGNN_SELF_MOBILITY_SIM
    / HIGNN_TIME_UNIT_S
)

# HIGNN mobility normalization.
# Use 1.0 first. If HIGNN unit force does not produce unit self-speed,
# this can be adjusted later.
HIGNN_SELF_MOBILITY_SIM = 1.405 #4.7427

def get_active_dof_list(sel, n_total, active_axes=(True, True, False)):
    """
    Return list of (particle_index, component_index) for active Brownian DOFs.
    component_index: 0=x, 1=y, 2=z
    """
    particle_ids = np.arange(n_total)[sel]
    comps = [c for c, active in enumerate(active_axes) if active]

    dof_list = []
    for p in particle_ids:
        for c in comps:
            dof_list.append((int(p), int(c)))

    return dof_list


def build_hignn_submobility_matrix(hignn_model, X, sel, active_axes=(True, True, False)):
    """
    Build explicit HIGNN mobility matrix for selected dynamic DOFs.

    M_sub[row, col] maps:
        unit force in DOF col
        -> velocity in DOF row

    This uses repeated hignn_model.dot calls.
    Expensive for large N.
    """
    X = np.asarray(X, dtype=np.float32)
    n_total = X.shape[0]

    dof_list = get_active_dof_list(sel, n_total, active_axes=active_axes)
    ndof = len(dof_list)

    M = np.zeros((ndof, ndof), dtype=np.float64)

    # make sure HIGNN sees current coordinates
    hignn_model.update_coord(X[:, :3])

    for col, (p_force, c_force) in enumerate(dof_list):
        force = np.zeros((n_total, 3), dtype=np.float32)
        vel = np.zeros((n_total, 3), dtype=np.float32)

        force[p_force, c_force] = 1.0

        hignn_model.dot(vel, force)

        for row, (p_vel, c_vel) in enumerate(dof_list):
            M[row, col] = vel[p_vel, c_vel]

    return M, dof_list


def force_scale_from_time():
    """
    N per simulation-force unit from calibrated time scale.
    """
    return HIGNN_FORCE_SCALE_TIME_N


def brownian_beta_sim():
    """
    Coefficient for covariance in sim units:

        cov(dX_B) = beta * M_sim * dt_sim

    where

        beta = 2 kBT / (F_scale * L0)
    """
    kB = 1.380649e-23
    F_scale = force_scale_from_time()
    return 2.0 * kB * BROWNIAN_TEMP_K / (F_scale * BROWNIAN_L0)


def brownian_dt_limit_from_mobility(M_sub, max_rms_sim=0.05):
    """
    Limit dt using diagonal variance:

        var_i = beta * M_ii * dt

    so sqrt(var_i) <= max_rms_sim.
    """
    beta = brownian_beta_sim()

    diag = np.diag(M_sub).copy()
    diag = np.maximum(diag, 0.0)

    max_diag = np.max(diag)
    if max_diag <= 0.0:
        return np.inf

    return max_rms_sim**2 / (beta * max_diag + 1e-30)


def sample_brownian_from_mobility(
    M_sub,
    dof_list,
    X_shape,
    dt,
    rng,
    comm=None,
    rank=0,
    remove_com_kick=False,
):
    """
    Sample Brownian displacement using explicit HIGNN mobility matrix.

        cov(dX_B) = beta * M_sub * dt

    Uses eigenvalue square root of symmetrized M.
    """
    ndof = M_sub.shape[0]

    # Symmetrize because Brownian covariance must be symmetric
    M_sym = 0.5 * (M_sub + M_sub.T)

    # Eigen square root
    evals, evecs = np.linalg.eigh(M_sym)

    # Clip small negative eigenvalues caused by HIGNN/numerical approximation
    evals_clipped = np.clip(evals, 0.0, None)

    beta = brownian_beta_sim()

    if rank == 0:
        xi = rng.normal(size=ndof)
        dvec = np.sqrt(beta * dt) * (evecs @ (np.sqrt(evals_clipped) * xi))
    else:
        dvec = np.zeros(ndof, dtype=np.float64)

    if comm is not None:
        comm.Bcast(dvec, root=0)

    dX_B = np.zeros(X_shape, dtype=np.float64)

    for val, (p, c) in zip(dvec, dof_list):
        dX_B[p, c] = val

    if remove_com_kick:
        # remove random translation of dynamic active DOFs only
        particles = sorted(set(p for p, c in dof_list))
        comps = sorted(set(c for p, c in dof_list))

        for c in comps:
            dX_B[particles, c] -= np.mean(dX_B[particles, c])

    return dX_B.astype(np.float32), evals, evals_clipped


def brownian_rms_xy_nm(dX_B, sel):
    dxy = dX_B[sel, :2].astype(np.float64)
    rms_sim = np.sqrt(np.mean(np.sum(dxy**2, axis=1)))
    return rms_sim * BROWNIAN_L0 * 1e9

# Placeholder for static floor particle indices
floor_indices = None

# def velocity_update(t, position):
#     # Print timestep on rank 0
#     if rank == 0:
#         print(f"t = {t:.4f}")

#     # Update HIGNN coordinates
#     hignn_model.update_coord(position[:, :3])

#     # Initialize velocity and force arrays
#     velocity = np.zeros((position.shape[0], 3), dtype=np.float32)
#     force = np.zeros((position.shape[0], 3), dtype=np.float32)
#     # Apply force in Y
    
#     force = potential_force.get_potential_force(position[:, :3]).astype(np.float32)

#     force[:, 2] = 0.0

#     # force[:, 1] += 1.0

#     # Zero out forces on floor particles so they remain static
#     if floor_indices is not None:
#         force[floor_indices] = 0.0

#     # Compute velocities via HIGNN
#     hignn_model.dot(velocity, force)

#     # Prevent floor particles from moving by zeroing their velocity
#     if floor_indices is not None:
#         velocity[floor_indices] = 0.0

#     return velocity


######## E-Field (UIUC) version with vdW + Yukawa + flow + floor ########
# def velocity_update(t, position):
#     """
#     Adds: vdW + Yukawa + flow pair forces (SI), converts to sim units, then calls HIGNN mobility.
#     Floor particles (floor_indices) are held fixed (force and velocity set to 0).
#     """

#     # if rank == 0:
#     #     print("init exists?", hasattr(velocity_update, "_init_done"),
#     #       "value:", getattr(velocity_update, "_init_done", None), flush=True)

#     # if rank == 0:
#     #     print("DEBUG init:",
#     #         "has:", hasattr(velocity_update, "_init_done"),
#     #         "val:", getattr(velocity_update, "_init_done", None),
#     #         flush=True)


#     # ------------------------ user toggles ------------------------
#     USE_MORSE = False                 # True if you also want PotentialForce.hpp Morse force
#     BODY_FORCE_SIM = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # e.g. [0,0,-1]

#     R_MIN_CC = 2.1          # minimum center-to-center distance in SIM units
#     K_CONTACT = 5e2 #88.836457          # penalty stiffness in SIM force units per SIM length (tune)
#     # --------------------------------------------------------------

#     if rank == 0:
#         print(f"t = {t:.4f}")

#     # Update HIGNN coordinates
#     hignn_model.update_coord(position[:, :3])

#     # Allocate arrays
#     pos = position[:, :3].astype(np.float64)
#     N = pos.shape[0]
#     velocity = np.zeros((N, 3), dtype=np.float32)

#     # Base force in sim units
#     force = np.zeros((N, 3), dtype=np.float32)

#     if USE_MORSE:
#         # PotentialForce.hpp Morse force (already in your sim units)
#         force += potential_force.get_potential_force(position[:, :3]).astype(np.float32)

#     # Add constant body force in sim units
#     force += BODY_FORCE_SIM[None, :]

#     # print("Adding pairwise vdW + Yukawa + flow forces...\n")

#     # ------------------------ one-time init & caching ------------------------
#     if not getattr(velocity_update, "_init_done", False):

#         velocity_update._init_done = True
#         if rank == 0:
#             print("Initialized pair-force params ...", flush=True)

#         # --- geometry scaling: radius_sim=1 corresponds to radius_phys=30 nm
#         a_phys = 30e-9   # meters
#         a_sim  = 1.0
#         L0 = a_phys / a_sim               # meters per sim-length

#         # --- force scaling: choose 1 sim-force unit = 1e-13 N
#         F0 = 1e-13  # N per sim-force-unit

#         # --- vdW parameters (SI)
#         H = 1e-19        # J  (you set this)
#         # choose Delta_b consistent with your particle size:
#         # if Delta_b is "diameter-like", use 2*a_phys; if "radius-like", use a_phys.
#         Delta_b = 3.0 * 0.332 * 1e-9  # <-- common choice if Δ_b means bead diameter
#         C_vdw = (H * (Delta_b**6)) / (np.pi**2)  # Joule * m^6

#         # --- Yukawa parameters (SI)
#         Zb = 1.0
#         e_charge = 1.602176634e-19
#         eps0 = 8.854187817e-12
#         eps_r = 80.0
#         K_el = (Zb**2 * e_charge**2) / (4*np.pi*eps0*eps_r)  # J*m

#         # Debye screening (use I in mol/m^3)
#         # If you intended 1 mM: I = 1e-3 mol/L = 1 mol/m^3
#         # I_molar_L = 1e-24
#         I_molm3 = 7.23

#         kB = 1.380649e-23
#         R = 8.314
#         F = 96485.3329
#         T  = 293.15
#         NA = 6.02214076e23

#         kappa = np.sqrt(2.0 * I_molm3 * F**2 / (eps0 * eps_r * R * T))  # 1/m

#         # --- flow force parameters (SI) and calibration at r_ref = 121 nm
#         A_flow = -1.114644e+01 * 1e-12   # your values (units should be N·m)
#         B_flow =  -3.002574e+03 * 1e-12   # your values (units should be N·m^2)

#         r_ref = 121e-9   # meters
#         F_ref_target = 1e-13  # N (given)
#         F_ref_current = abs(A_flow/r_ref + B_flow/(r_ref**2))
#         scale = F_ref_target / (F_ref_current + 1e-30)
#         A_flow *= scale
#         B_flow *= scale

#         # --- neighbor cutoff (in meters) and convert to sim
#         # pick something that covers your 121 nm reference and several Debye lengths
#         lambda_D = 1.0 / kappa
#         r_cut_phys = max(5.0*lambda_D, 5.0*r_ref)    # meters
#         r_cut_sim  = r_cut_phys / L0                 # sim units

#         # --- safety
#         r_min_sim = 1e-6

#         # store
#         velocity_update._L0 = L0
#         velocity_update._F0 = F0
#         velocity_update._C_vdw = C_vdw
#         velocity_update._K_el  = K_el
#         velocity_update._kappa = kappa
#         velocity_update._A_flow = A_flow
#         velocity_update._B_flow = B_flow
#         velocity_update._r_cut_sim = float(r_cut_sim)
#         velocity_update._r_min_sim = float(r_min_sim)
#         velocity_update._init_done = True

#         print("1\n")  # blank line

#         if rank == 0:
#             print("Initialized pair-force params:")
#             print(f"  L0 = {L0:.3e} m/sim_length")
#             print(f"  F0 = {F0:.3e} N/sim_force")
#             print(f"  kappa^-1 = {lambda_D:.3e} m (Debye length)")
#             print(f"  r_cut_sim = {r_cut_sim:.3f} (sim units)")
#             print(f"  Calibrated |F_flow(121 nm)| = {abs(A_flow/r_ref + B_flow/(r_ref**2)):.3e} N")

#     # unpack cached constants
#     L0       = velocity_update._L0
#     F0       = velocity_update._F0
#     C_vdw    = velocity_update._C_vdw
#     K_el     = velocity_update._K_el
#     kappa    = velocity_update._kappa
#     A_flow   = velocity_update._A_flow
#     B_flow   = velocity_update._B_flow
#     r_cut_sim = velocity_update._r_cut_sim
#     r_min_sim = velocity_update._r_min_sim
#     # ------------------------------------------------------------------------

#     # Build neighbor lists in SIM units
#     tree = cKDTree(pos)
#     neigh = tree.query_ball_tree(tree, r_cut_sim)

#     # Decide which particles are "dynamic" (non-floor) to avoid floor-floor work
#     if floor_indices is not None and len(floor_indices) > 0:
#         dyn_mask = np.ones(N, dtype=bool)
#         dyn_mask[np.asarray(floor_indices, dtype=int)] = False
#         dyn_idx = np.where(dyn_mask)[0]
#     else:
#         dyn_mask = np.ones(N, dtype=bool)
#         dyn_idx = np.arange(N, dtype=int)

#     F_extra = np.zeros((N, 3), dtype=np.float64)

#     for i in dyn_idx:
#         for j in neigh[i]:
#             if j == i:
#                 continue

#             # avoid double-counting only for dynamic-dynamic pairs
#             if dyn_mask[j] and (j < i):
#                 continue

#             # rij_sim = pos[i] - pos[j]
#             # r_sim = np.linalg.norm(rij_sim)
#             # if r_sim < r_min_sim:
#             #     continue

#             # rhat = rij_sim / r_sim
#             # # r_phys = L0 * r_sim  # meters
#             # r_contact_sim = 2.0
#             # r_eff_sim  = max(r_sim, r_contact_sim + 1e-3)
#             # r_phys     = L0 * r_eff_sim

#             rij_sim = pos[i] - pos[j]
#             r_sim = np.linalg.norm(rij_sim)
#             if r_sim < r_min_sim:
#                 continue

#             rhat = rij_sim / (r_sim + 1e-30)

#             # --- HARD-CORE penalty (SIM units): only active if too close ---
#             F_contact_sim = 0.0
#             if r_sim < R_MIN_CC:
#                 F_contact_sim = K_CONTACT * (R_MIN_CC - r_sim) * rhat  # pushes i away from j (sim units)

#             # Use an "effective" r for SI force formulas to avoid singularities
#             r_eff_sim = max(r_sim, R_MIN_CC)
#             r_phys = L0 * r_eff_sim



#             # ----- vdW (SI): F = -(6*C_vdw/r^7) rhat
#             F_vdw_phys = -(6.0 * C_vdw / (r_phys**7)) * rhat

#             # ----- Yukawa (SI): F = K exp(-k r) (k/r + 1/r^2) rhat
#             expkr = np.exp(-kappa * (r_phys-2*30e-9))
#             F_el_phys = (K_el * expkr * (kappa/(r_phys-2*30e-9) + 1/(r_phys-2*30e-9)**2)) * rhat

#             # ----- Flow (SI): F_flow = (A/r + B/r^2) rhat   (given as FORCE law)
#             F_flow_phys = (A_flow/r_phys + B_flow/(r_phys**2)) * rhat

#             # total physical force on i due to j
#             F_ij_phys = F_vdw_phys + F_el_phys + F_flow_phys

#             # print(f"i={i} j={j} r_phys={r_phys:.3e} F_vdw={np.linalg.norm(F_vdw_phys):.3e} "+
#             #       f"F_el={np.linalg.norm(F_el_phys):.3e} F_flow={np.linalg.norm(F_flow_phys):.3e}")

#             # convert to sim units
#             # F_ij_sim = F_ij_phys / F0
#             F_ij_sim = F_ij_phys / F0 + F_contact_sim

#             # apply to i
#             F_extra[i] += F_ij_sim

#             # apply reaction to j only if j is dynamic; floor is fixed
#             if dyn_mask[j]:
#                 F_extra[j] -= F_ij_sim

#     # Add into sim force array
#     force += F_extra.astype(np.float32)

#     force[:,2] = 0.0

#     # Freeze floor (infinite-mass floor)
#     if floor_indices is not None and len(floor_indices) > 0:
#         force[floor_indices] = 0.0

#     # Mobility solve
#     hignn_model.dot(velocity, force)

#     # Freeze floor velocity too
#     if floor_indices is not None and len(floor_indices) > 0:
#         velocity[floor_indices] = 0.0

#     return velocity

def velocity_update(t, position):
    """
    Adds:
      - vdW pair force from equal-sphere Hamaker energy using gap D_vdw = r_cc - 2R
      - electrostatic repulsion from your log-energy using gap D_el = r_cc - 2*a_el
      - flow force (left as your current A/r + B/r^2 form)
    Converts SI forces to sim units, then calls HIGNN mobility.
    Floor particles are held fixed.
    """

    # ------------------------ user toggles ------------------------
    USE_MORSE = False
    BODY_FORCE_SIM = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    R_MIN_CC = 2.1
    K_CONTACT = 5e2
    # --------------------------------------------------------------

    if rank == 0:
        print(f"t = {t:.4f}")

    hignn_model.update_coord(position[:, :3])

    pos = position[:, :3].astype(np.float64)
    N = pos.shape[0]
    velocity = np.zeros((N, 3), dtype=np.float32)
    force = np.zeros((N, 3), dtype=np.float32)

    if USE_MORSE:
        force += potential_force.get_potential_force(position[:, :3]).astype(np.float32)

    force += BODY_FORCE_SIM[None, :]

    # ------------------------ one-time init & caching ------------------------
    if not getattr(velocity_update, "_init_done", False):

                # ---------------- NN surrogate settings ----------------
        NN_DEVICE = "cpu"   # use cpu for robust inference
        VDW_MODEL_PATH = "/local/python/checkpoints/F_vdw_scalar_pN_model.pt"
        ELEC_MODEL_PATH = "/local/python/checkpoints/F_elec_scalar_pN_model.pt"
        FLOW_MODEL_PATH = "/local/python/checkpoints/eo_pair_radial_mlp_raw.pt"

        vdw_model, vdw_ckpt = load_model(VDW_MODEL_PATH, NN_DEVICE)
        elec_model, elec_ckpt = load_model(ELEC_MODEL_PATH, NN_DEVICE)
        flow_model, flow_ckpt = load_model(FLOW_MODEL_PATH, NN_DEVICE)

        velocity_update._nn_device = NN_DEVICE
        velocity_update._vdw_model = vdw_model
        velocity_update._vdw_ckpt = vdw_ckpt
        velocity_update._elec_model = elec_model
        velocity_update._elec_ckpt = elec_ckpt
        velocity_update._flow_model = flow_model
        velocity_update._flow_ckpt = flow_ckpt

        # lower bounds to keep NN inside its valid log-gap domain
        # choose small positive values or read from your dataset if you want
        velocity_update._min_gap_core_nm = 0.2
        velocity_update._min_gap_shell_nm = 0.2

        velocity_update._init_done = True
        if rank == 0:
            print("Initialized pair-force params ...", flush=True)

        # geometry scaling: radius_sim=1 corresponds to radius_phys=30 nm
        R_core = 30e-9          # core radius, meters
        a_sim = 1.0
        L0 = R_core / a_sim     # meters per sim-length

        # Time-scale-based force conversion.
        # This replaces arbitrary F0.
        force_to_sim_from_time = (
            HIGNN_TIME_UNIT_S
            / (6.0 * np.pi * HIGNN_ETA_PHYS * R_core * L0 * HIGNN_SELF_MOBILITY_SIM)
        )

        force_scale_time_N = 1.0 / force_to_sim_from_time

        # vdW parameters
        A_ham = 1e-19           # Hamaker constant [J]

        # electrostatic parameters
        eps0 = 8.854187817e-12
        eps_r = 80.0
        sigma_surf = -0.014     # C/m^2

        ligand_length = 2.4e-9
        a_el = R_core + ligand_length

        I_molm3 = 7.23
        Rgas = 8.314
        FARADAY = 96485.3329
        T_kelvin = 293.15

        kappa = np.sqrt(2.0 * I_molm3 * FARADAY**2 / (eps0 * eps_r * Rgas * T_kelvin))

        # flow parameters (left unchanged from your current model)
        # A_flow = 4.63e-9*1e-18
        B_flow = -1.08e-7*1e-27
        C_flow = -1.8e-13

        A_flow = -7.652013e-8*10**(-25.47)

        # r_ref = 121e-9
        # F_ref_target = 1e-13
        # F_ref_current = abs(A_flow / r_ref + B_flow / (r_ref**2))
        # scale = F_ref_target / (F_ref_current + 1e-30)
        # A_flow *= scale
        # B_flow *= scale

        lambda_D = 1.0 / kappa
        r_cut_phys = 5.0 * lambda_D
        r_cut_sim = r_cut_phys / L0

        r_min_sim = 1e-6
        D_min_phys = 0.2e-9   # minimum gap for regularization

        # store
        velocity_update._L0 = L0
        velocity_update._force_to_sim_from_time = force_to_sim_from_time
        velocity_update._force_scale_time_N = force_scale_time_N
        velocity_update._A_ham = A_ham
        velocity_update._R_core = R_core
        velocity_update._a_el = a_el
        velocity_update._eps0 = eps0
        velocity_update._eps_r = eps_r
        velocity_update._sigma_surf = sigma_surf
        velocity_update._kappa = kappa
        velocity_update._A_flow = A_flow
        velocity_update._B_flow = B_flow
        velocity_update._C_flow = C_flow
        velocity_update._r_cut_sim = float(r_cut_sim)
        velocity_update._r_min_sim = float(r_min_sim)
        velocity_update._D_min_phys = float(D_min_phys)

        if rank == 0:
            print("Initialized pair-force params:")
            print(f"  L0 = {L0:.3e} m/sim_length")
            print(f"  HIGNN_TIME_UNIT_S = {HIGNN_TIME_UNIT_S:.6e} s")
            print(f"  1 HIGNN time unit = {HIGNN_TIME_UNIT_S*1e6:.3f} microseconds")
            print(f"  force_to_sim_from_time = {force_to_sim_from_time:.3e} sim_force/N")
            print(f"  implied force scale = {force_scale_time_N:.3e} N per sim_force")
            print(f"  kappa^-1 = {lambda_D:.3e} m (Debye length)")
            print(f"  r_cut_sim = {r_cut_sim:.3f} (sim units)")
            # print(f"  Calibrated |F_flow(121 nm)| = {abs(A_flow/r_ref + B_flow/(r_ref**2)):.3e} N")

    # unpack
    L0 = velocity_update._L0
    force_to_sim_from_time = velocity_update._force_to_sim_from_time
    force_scale_time_N = velocity_update._force_scale_time_N
    A_ham = velocity_update._A_ham
    R_core = velocity_update._R_core
    a_el = velocity_update._a_el
    eps0 = velocity_update._eps0
    eps_r = velocity_update._eps_r
    sigma_surf = velocity_update._sigma_surf
    kappa = velocity_update._kappa
    A_flow = velocity_update._A_flow
    B_flow = velocity_update._B_flow
    C_flow = velocity_update._C_flow
    r_cut_sim = velocity_update._r_cut_sim
    r_min_sim = velocity_update._r_min_sim
    D_min_phys = velocity_update._D_min_phys
    nn_device = velocity_update._nn_device
    vdw_model = velocity_update._vdw_model
    vdw_ckpt = velocity_update._vdw_ckpt
    elec_model = velocity_update._elec_model
    elec_ckpt = velocity_update._elec_ckpt
    flow_model = velocity_update._flow_model
    flow_ckpt = velocity_update._flow_ckpt
    min_gap_core_nm = velocity_update._min_gap_core_nm
    min_gap_shell_nm = velocity_update._min_gap_shell_nm

    # # ------------------------ helper force formulas ------------------------
    # def F_vdw_equal_spheres(D, R, A_ham):
    #     # F = -dW/dD from your vdW energy
    #     return (A_ham / 6.0) * (
    #         -(4.0 * R**2 * (2.0 * R + D)) / (D**2 * (4.0 * R + D)**2)
    #         - (4.0 * R**2) / ((2.0 * R + D)**3)
    #         + 1.0 / (4.0 * R + D)
    #         + 1.0 / D
    #         - 2.0 / (2.0 * R + D)
    #     )

    # def F_el_const_sigma(D, a_el, sigma_surf, eps_r, eps0, kappa):
    #     # F = -dV/dD from your electrostatic energy
    #     pref = (2.0 * a_el * sigma_surf**2) / (eps_r * eps0 * kappa)
    #     return pref / (np.exp(kappa * D) - 1.0)

    # ------------------------------------------------------------------------
    # tree = cKDTree(pos)
    # neigh = tree.query_ball_tree(tree, r_cut_sim)

    if floor_indices is not None and len(floor_indices) > 0:
        dyn_mask = np.ones(N, dtype=bool)
        dyn_mask[np.asarray(floor_indices, dtype=int)] = False
        dyn_idx = np.where(dyn_mask)[0]
    else:
        dyn_mask = np.ones(N, dtype=bool)
        dyn_idx = np.arange(N, dtype=int)

        F_extra = np.zeros((N, 3), dtype=np.float64)

    # ------------------------------------------------------------------
    # Batch pair collection
    # ------------------------------------------------------------------
    pair_i = []
    pair_j = []
    pair_rhat = []
    pair_r_phys = []
    pair_contact_sim = []

    for i in dyn_idx:
        for j in range(N):
            if j == i:
                continue
            if dyn_mask[j] and (j < i):
                continue

            rij_sim = pos[i] - pos[j]
            r_sim = np.linalg.norm(rij_sim)
            if r_sim < r_min_sim:
                continue

            rhat = rij_sim / (r_sim + 1e-30)

            # hard-core penalty in sim units
            F_contact_sim = np.zeros(3, dtype=np.float64)
            # if r_sim < R_MIN_CC:
            #     F_contact_sim = K_CONTACT * (R_MIN_CC - r_sim) * rhat

            # keep the same physical distance convention as your current code
            r_phys = L0 * r_sim

            pair_i.append(i)
            pair_j.append(j)
            pair_rhat.append(rhat)
            pair_r_phys.append(r_phys)
            pair_contact_sim.append(F_contact_sim)

    if len(pair_i) > 0:
        pair_i = np.asarray(pair_i, dtype=np.int64)
        pair_j = np.asarray(pair_j, dtype=np.int64)
        pair_rhat = np.asarray(pair_rhat, dtype=np.float64)              # (M, 3)
        pair_r_phys = np.asarray(pair_r_phys, dtype=np.float64)          # (M,)
        pair_contact_sim = np.asarray(pair_contact_sim, dtype=np.float64)  # (M, 3)

        # --------------------------------------------------------------
        # Batch NN evaluation: vdW
        # --------------------------------------------------------------
        gap_core_nm_all = (pair_r_phys - 2.0 * R_core) * 1e9
        gap_core_nm_all = np.maximum(gap_core_nm_all, min_gap_core_nm)

        F_vdw_scalar_pN_all = predict_force_pN(
            vdw_model,
            vdw_ckpt,
            gap_core_nm_all.astype(np.float64),
            nn_device,
        )

        # --------------------------------------------------------------
        # Batch NN evaluation: electrostatics
        # --------------------------------------------------------------
        gap_shell_nm_all = (pair_r_phys - 2.0 * a_el) * 1e9
        F_el_scalar_pN_all = np.zeros_like(gap_shell_nm_all, dtype=np.float64)

        mask_shell_valid = gap_shell_nm_all > 0.0
        if np.any(mask_shell_valid):
            gap_shell_nm_eval = np.maximum(
                gap_shell_nm_all[mask_shell_valid],
                min_gap_shell_nm
            )

            F_el_scalar_pN_all[mask_shell_valid] = predict_force_pN(
                elec_model,
                elec_ckpt,
                gap_shell_nm_eval.astype(np.float64),
                nn_device,
            )

        # --------------------------------------------------------------
        # Flow force (unchanged, but vectorized)
        # --------------------------------------------------------------
        # F_flow_scalar_N_all = A_flow / (pair_r_phys ** 2.83)
        pair_flow_nm_all = pair_r_phys * 1e9

        F_flow_scalar_N_all = predict_eo_flow_force_N(
            flow_model,
            flow_ckpt,
            pair_flow_nm_all.astype(np.float64),
            nn_device,
        )

        # --------------------------------------------------------------
        # Combine all scalar forces -> vector forces
        # --------------------------------------------------------------
        F_vdw_scalar_N_all = F_vdw_scalar_pN_all * 1e-12
        F_el_scalar_N_all = F_el_scalar_pN_all * 1e-12

        F_total_scalar_N_all = (
            F_vdw_scalar_N_all +
            F_el_scalar_N_all + 0.0
            - F_flow_scalar_N_all
        )

        F_ij_phys_all = F_total_scalar_N_all[:, None] * pair_rhat
        F_ij_sim_all = F_ij_phys_all * force_to_sim_from_time + pair_contact_sim

        # --------------------------------------------------------------
        # Scatter pair forces back to particles
        # --------------------------------------------------------------
        np.add.at(F_extra, pair_i, F_ij_sim_all)

        mask_j_dynamic = dyn_mask[pair_j]
        if np.any(mask_j_dynamic):
            np.add.at(F_extra, pair_j[mask_j_dynamic], -F_ij_sim_all[mask_j_dynamic])

    force += F_extra.astype(np.float32)
    force[:, 2] = 0.0

    if floor_indices is not None and len(floor_indices) > 0:
        force[floor_indices] = 0.0

    hignn_model.dot(velocity, force)

    if floor_indices is not None and len(floor_indices) > 0:
        velocity[floor_indices] = 0.0

    return velocity

def reset_velocity_update_cache():
    for a in [
        "_init_done",
        "_L0",
        "_force_to_sim_from_time",
        "_force_scale_time_N",
        "_C_vdw",
        "_K_el",
        "_kappa",
        "_A_flow",
        "_B_flow",
        "_r_cut_sim",
        "_r_min_sim",
    ]:
        if hasattr(velocity_update, a):
            delattr(velocity_update, a)

def project_min_distance(X, r_min, floor_indices=None, max_iter=30, tol=1e-6, use_xy_only=False):
    """
    Enforce hard minimum center-center distance >= r_min by position projection.

    - If both particles are dynamic: push both apart half-half.
    - If one is floor (static): push only the dynamic particle.
    - Iterates because resolving one overlap can create another.

    Returns:
        X_corrected, n_violations_last_iter
    """
    X = X.copy()
    N = X.shape[0]

    if floor_indices is None:
        floor_indices = np.array([], dtype=int)
    else:
        floor_indices = np.asarray(floor_indices, dtype=int)

    is_floor = np.zeros(N, dtype=bool)
    if floor_indices.size > 0:
        is_floor[floor_indices] = True

    # Work in 2D if requested (faster) — OK when everything lies in a plane (z constant)
    def get_coords(arr):
        return arr[:, :2] if use_xy_only else arr[:, :3]

    coords = get_coords(X)

    n_viol = 0
    for it in range(max_iter):
        tree = cKDTree(coords)
        pairs = list(tree.query_pairs(r_min - tol))  # pairs that violate

        n_viol = len(pairs)
        if n_viol == 0:
            break

        # Resolve overlaps
        for i, j in pairs:
            rij = coords[i] - coords[j]
            dist = np.linalg.norm(rij)
            if dist < 1e-12:
                # degenerate: random tiny nudge
                rij = np.random.randn(rij.size)
                dist = np.linalg.norm(rij)

            overlap = (r_min - dist)
            if overlap <= 0:
                continue

            rhat = rij / dist

            if (not is_floor[i]) and (not is_floor[j]):
                # split correction
                shift = 0.5 * overlap * rhat
                if use_xy_only:
                    X[i, :2] += shift
                    X[j, :2] -= shift
                else:
                    X[i, :3] += shift
                    X[j, :3] -= shift
            else:
                # push only the dynamic particle
                if is_floor[i] and (not is_floor[j]):
                    # push j away from i
                    shift = overlap * (-rhat)
                    if use_xy_only:
                        X[j, :2] += shift
                    else:
                        X[j, :3] += shift
                elif is_floor[j] and (not is_floor[i]):
                    # push i away from j
                    shift = overlap * (rhat)
                    if use_xy_only:
                        X[i, :2] += shift
                    else:
                        X[i, :3] += shift
                # if both are floor, do nothing

        # refresh coords for next iteration
        coords = get_coords(X)

    return X, n_viol

if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    velocity_update.__dict__.clear()

    reset_velocity_update_cache()   # <-- add this

    hignn.Init()

    USE_FLOOR = False   # <-- flip this True/False

    # --- Generate a tightly-packed floor in the XY plane ---
    min_dist = 2.02   # minimal spacing between floor particles
    floor_nx = 50    # number of particles along X
    floor_ny = 50    # number of particles along Y
    floor_z = 0.0    # Z-coordinate of the floor

    # x_floor = np.arange(0, floor_nx * min_dist, min_dist)
    # y_floor = np.arange(0, floor_ny * min_dist, min_dist)
    # xx, yy = np.meshgrid(x_floor, y_floor)
    # floor_pts = np.vstack((xx.ravel(), yy.ravel(), np.full(xx.size, floor_z))).T.astype(np.float32)

    # --------- floor construction ----------
    if USE_FLOOR:
        # # --- Circular hex floor ---
        # --- Hex-packed floor clipped to a circle of radius R ---
        # want ~200 pts across the diameter → use floor_n×floor_n grid spacing
        floor_n = 10
        dx = min_dist                      # horizontal spacing
        dy = dx * np.sqrt(3)/2            # vertical row spacing for hex packing

        # circle radius and center
        diameter = (floor_n - 1) * dx
        R = diameter / 2.0
        cx = cy = R

        coords = []
        # generate hexagonal lattice over sufficient bounding box
        # go slightly beyond 0→2R so edges get clipped cleanly
        y_vals = np.arange(-dy, 2*R+dy, dy, dtype=np.float32)
        for i, y in enumerate(y_vals):
            # offset every other row
            x_offset = (dx/2) if (i % 2 == 1) else 0.0
            x_vals = np.arange(-dx, 2*R+dx, dx, dtype=np.float32) + x_offset
            for x in x_vals:
                # clip to circle
                if (x - cx)**2 + (y - cy)**2 <= R**2:
                    coords.append((x, y, 0.0))

        floor_pts = np.array(coords, dtype=np.float32)
        floor_indices = np.arange(len(floor_pts), dtype=np.int32)

        # now compute its center:
        xmin, xmax = floor_pts[:,0].min(), floor_pts[:,0].max()
        ymin, ymax = floor_pts[:,1].min(), floor_pts[:,1].max()

        floor_x_center = 0.5*(xmin + xmax)
        floor_y_center = 0.5*(ymin + ymax)

        floor_center_orig = np.array([
        floor_pts[:,0].mean(),
        floor_pts[:,1].mean(),
        0.0
        ], dtype=np.float32)

    else:
        floor_pts = np.empty((0, 3), dtype=np.float32)
        floor_indices = np.array([], dtype=np.int32)
        floor_center_orig = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        floor_x_center = 0.0
        floor_y_center = 0.0

    # # --- Generate 10 dynamic particles in a straight line above the floor ---
    # dynamic_n = 15
    # dynamic_spacing = 2 * min_dist  # spacing between dynamic particles
    # # x_dyn = np.arange(0, dynamic_n * dynamic_spacing, dynamic_spacing)
    # # y_dyn = np.zeros(dynamic_n)      # straight line along X-axis

    # # Center the dynamic particles above the floor
    # floor_x_center = 0.5 * (floor_nx - 1) * min_dist
    # floor_y_center = 0.5 * (floor_ny - 1) * min_dist

    # # Place dynamic particles in a line along x centered on floor
    # x_dyn = np.linspace(-0.5 * (dynamic_n - 1) * dynamic_spacing, 
    #                     0.5 * (dynamic_n - 1) * dynamic_spacing, 
    #                     dynamic_n) + floor_x_center

    # y_dyn = np.full(dynamic_n, floor_y_center)

    # z_dyn = np.full(dynamic_n, min_dist)  # one min_dist above the floor
    # dynamic_pts = np.vstack((x_dyn, y_dyn, z_dyn)).T.astype(np.float32)

    # # # --- Generate dynamic particles randomly above the floor with minimum spacing and centered ---
    # dynamic_n = 100
    # dynamic_spacing = 2.2  # required spacing between dynamic particles
    # z_dyn = 2.267                # height above the floor

    # # Bounds for sampling
    # x_min, x_max = floor_x_center - 53, floor_x_center + 53
    # y_min, y_max = floor_y_center - 5, floor_y_center + 5

    # # Poisson-disk-like sampling via rejection
    # np.random.seed(12345)
    # dynamic_pts_list = []
    # while len(dynamic_pts_list) < dynamic_n:
    #     cand_xy = np.array([np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max)], dtype=np.float32)
    #     # ensure min spacing
    #     if all(np.linalg.norm(cand_xy - existing[:2]) >= dynamic_spacing for existing in dynamic_pts_list):
    #         dynamic_pts_list.append(np.hstack((cand_xy, z_dyn)))
    # dynamic_pts = np.array(dynamic_pts_list, dtype=np.float32)

    # # Shift so centroid aligns with floor center
    # centroid = np.mean(dynamic_pts[:, :2], axis=0)
    # dynamic_pts[:, 0] += floor_center_orig[0] - centroid[0]
    # dynamic_pts[:, 1] += floor_center_orig[1] - centroid[1]

    # # # --- ----------------------------------Generate dynamic particles in a blob above the floor ----------------------------------
    # dynamic_n = 500
    # dynamic_spacing = 3.067   # nearest-neighbor spacing (≈ min distance)
    # z_dyn = 2.267                       # height above the floor

    # x_min, x_max = floor_x_center - 2.5*40, floor_x_center + 2.5*40
    # y_min, y_max = floor_y_center - 2.5*40, floor_y_center + 2.5*7
    # center_xy=[floor_x_center, floor_y_center]

    # rng = np.random.default_rng(12345)

    # dx = dynamic_spacing
    # dy = dynamic_spacing * np.sqrt(3) / 2

    # # Build a hex lattice that fully covers the bounds (+ margin so clipping is clean)
    # margin = 2 * dynamic_spacing
    # X0, X1 = x_min - margin, x_max + margin
    # Y0, Y1 = y_min - margin, y_max + margin

    # # number of rows/cols needed
    # ny = int(np.ceil((Y1 - Y0) / dy)) + 1
    # nx = int(np.ceil((X1 - X0) / dx)) + 1

    # pts = []
    # for j in range(ny):
    #     y = Y0 + j * dy
    #     x_off = 0.5 * dx if (j % 2 == 1) else 0.0
    #     for i in range(nx):
    #         x = X0 + i * dx + x_off
    #         # HARD rectangular clip
    #         if (x_min <= x <= x_max) and (y_min <= y <= y_max):
    #             pts.append((x, y))

    # pts = np.array(pts, dtype=np.float32)

    # if pts.shape[0] == 0:
    #     raise RuntimeError("No lattice points fell inside bounds. Increase bounds or reduce spacing.")

    # if pts.shape[0] < dynamic_n:
    #     raise RuntimeError(
    #         f"Bounds too small for dynamic_n={dynamic_n} with spacing={dynamic_spacing}. "
    #         f"Only {pts.shape[0]} points fit. Enlarge bounds or reduce dynamic_n."
    #     )

    # # Choose the most central points to form a compact blob (no holes)
    # if center_xy is None:
    #     center_xy = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2], dtype=np.float32)
    # else:
    #     center_xy = np.array(center_xy, dtype=np.float32)

    # d2 = (pts[:,0] - center_xy[0])**2 + (pts[:,1] - center_xy[1])**2
    # order = np.argsort(d2)
    # kept = pts[order][:dynamic_n]

    # # (optional) randomize ordering so indices aren’t perfectly radial
    # rng.shuffle(kept)

    # dynamic_pts = np.hstack([kept, np.full((dynamic_n, 1), z_dyn, dtype=np.float32)])

    # -------------------- Manually place 3 dynamic particles at chosen (x, y), all with z = 2.267 ---------------------
    dynamic_n = 8
    z_dyn = 2.267

    # Option 1: absolute coordinates in your simulation frame
    # Edit these three (x, y) pairs however you want
    xy_positions = np.array([
        [100, 0],
        [70.71,  70.71],
        [  0,  100.0],
        [ 70.71,  -70.71],
        [0, -100.0],
        [ -70.71, -70.71],
        [  -100,  0.0],
        [  -70.71,  70.71],
    ], dtype=np.float32)
    # xy_positions = np.array([
    #     [-150.0, 0.0],
    #     [150.0,  0.0],
    # ], dtype=np.float32)

    xy_positions = xy_positions/15.0   # scale from nm to sim units (since R_core=30nm corresponds to 1 sim unit)
    # xy_positions = xy_positions / 30.0

    # Option 2: if you want positions centered relative to the floor center, use this instead:
    # xy_positions = np.array([
    #     [floor_x_center - 2.0, floor_y_center + 0.0],
    #     [floor_x_center + 2.0, floor_y_center + 0.0],
    #     [floor_x_center + 0.0, floor_y_center + 3.0],
    # ], dtype=np.float32)

    # if xy_positions.shape != (3, 2):
    #     raise ValueError("xy_positions must be a (3,2) array for 3 particles.")

    dynamic_pts = np.hstack((
        xy_positions,
        np.full((dynamic_n, 1), z_dyn, dtype=np.float32)
    )).astype(np.float32)

    print("Initial dynamic particle positions:")
    for idx, p in enumerate(dynamic_pts, start=1):
        print(f"  particle {idx}: x={p[0]:.3f}, y={p[1]:.3f}, z={p[2]:.3f}")

    # # -------------------- Generate a FIXED number of particles randomly inside an annulus at z = 2.267 --------------------
    # z_dyn = 2.267

    # # Choose annulus geometry in SIM units
    # inner_radius = 12.0
    # outer_radius = 30.0

    # # Choose annulus center
    # annulus_center = np.array([0.0, 0.0], dtype=np.float32)
    # # Or use floor-centered annulus:
    # # annulus_center = np.array([floor_x_center, floor_y_center], dtype=np.float32)

    # # Choose how many particles you want
    # dynamic_n = 100

    # # Minimum allowed center-to-center spacing
    # min_center_dist = 3 

    # rng = np.random.default_rng(12345)

    # if inner_radius <= 0:
    #     raise ValueError("inner_radius must be > 0")
    # if outer_radius <= inner_radius:
    #     raise ValueError("outer_radius must be greater than inner_radius")
    # if dynamic_n <= 0:
    #     raise ValueError("dynamic_n must be > 0")
    # if min_center_dist <= 0:
    #     raise ValueError("min_center_dist must be > 0")

    # cx, cy = annulus_center

    # def sample_random_points_in_annulus(n_particles, cx, cy, r_in, r_out, z_val,
    #                                     min_dist, rng, max_tries=500000):
    #     """
    #     Randomly sample exactly n_particles inside the annulus:
    #         r_in <= sqrt((x-cx)^2 + (y-cy)^2) <= r_out
    #     with fixed z = z_val and minimum spacing min_dist.
    #     """
    #     pts = []
    #     tries = 0

    #     while len(pts) < n_particles and tries < max_tries:
    #         tries += 1

    #         # Uniform-in-area annulus sampling
    #         theta = rng.uniform(0.0, 2.0 * np.pi)
    #         r = np.sqrt(rng.uniform(r_in**2, r_out**2))

    #         x = cx + r * np.cos(theta)
    #         y = cy + r * np.sin(theta)

    #         cand = np.array([x, y, z_val], dtype=np.float32)

    #         ok = True
    #         for p in pts:
    #             if np.linalg.norm(cand[:2] - p[:2]) < min_dist:
    #                 ok = False
    #                 break

    #         if ok:
    #             pts.append(cand)

    #     if len(pts) < n_particles:
    #         raise RuntimeError(
    #             f"Could only place {len(pts)} / {n_particles} particles inside annulus "
    #             f"(r_in={r_in}, r_out={r_out}) with min_dist={min_dist}. "
    #             f"Increase outer_radius, decrease inner_radius, reduce dynamic_n, or reduce min_center_dist."
    #         )

    #     return np.array(pts, dtype=np.float32)

    # dynamic_pts = sample_random_points_in_annulus(
    #     n_particles=dynamic_n,
    #     cx=cx,
    #     cy=cy,
    #     r_in=inner_radius,
    #     r_out=outer_radius,
    #     z_val=z_dyn,
    #     min_dist=min_center_dist,
    #     rng=rng
    # ).astype(np.float32)

    # print(f"Generated {dynamic_pts.shape[0]} particles randomly inside annulus.")
    # print(f"  center = ({cx:.3f}, {cy:.3f})")
    # print(f"  inner_radius = {inner_radius:.3f}")
    # print(f"  outer_radius = {outer_radius:.3f}")
    # print(f"  min_center_dist = {min_center_dist:.3f}")
    # print(f"  z = {z_dyn:.3f}")

    # # -------------------- Generate TWO blobs + manually placed particles at z = 2.267 --------------------
    # z_dyn = 2.267
    # min_center_dist = 2.8   # minimum allowed center-to-center spacing in SIM units

    # # ---------------- Blob 1 settings ----------------
    # blob1_n = 500
    # blob1_xmin, blob1_xmax = -30.0, 30.0
    # blob1_ymin, blob1_ymax = -30.0, 30.0

    # # ---------------- Blob 2 settings ----------------
    # blob2_n = 1
    # blob2_xmin, blob2_xmax = -30.0, 30.0
    # blob2_ymin, blob2_ymax = -30.0, 30.0

    # rng = np.random.default_rng(12345)

    # # ---------------- Manual points (edit these) ----------------
    # manual_xy = np.array([
    #     [0.0,   0.0],
    #     # [-5.833,  0.0],
    #     # [-2.8, 0.0],
    #     # [0.0,   2.8],
    #     # [2.8,  2.8],
    #     # [-2.8, 2.8],
    #     # [0.0,   -2.8],
    #     # [2.8,  -2.8],
    #     # [-2.8, -2.8],
    # ], dtype=np.float32)

    # manual_pts = np.hstack((
    #     manual_xy,
    #     np.full((manual_xy.shape[0], 1), z_dyn, dtype=np.float32)
    # )).astype(np.float32)

    # def check_spacing_against_existing(cand_xy, existing_pts, min_dist):
    #     for p in existing_pts:
    #         if np.linalg.norm(cand_xy - p[:2]) < min_dist:
    #             return False
    #     return True

    # def check_manual_points(manual_pts, min_dist):
    #     for i in range(len(manual_pts)):
    #         for j in range(i + 1, len(manual_pts)):
    #             if np.linalg.norm(manual_pts[i, :2] - manual_pts[j, :2]) < min_dist:
    #                 raise ValueError(
    #                     f"Manual points {i} and {j} are closer than min_center_dist={min_dist}"
    #                 )

    # def sample_blob_with_existing(n_particles, xmin, xmax, ymin, ymax, z_value,
    #                             min_dist, rng, existing_pts=None, max_tries=200000):
    #     """
    #     Rejection-sample particles in a rectangular xy window with fixed z,
    #     enforcing minimum spacing both within the blob and against existing_pts.
    #     """
    #     if xmin >= xmax or ymin >= ymax:
    #         raise ValueError("Invalid blob bounds: need xmin < xmax and ymin < ymax")
    #     if n_particles <= 0:
    #         raise ValueError("n_particles must be positive")
    #     if min_dist <= 0:
    #         raise ValueError("min_dist must be positive")

    #     if existing_pts is None:
    #         existing_pts = np.empty((0, 3), dtype=np.float32)

    #     pts = []
    #     tries = 0

    #     while len(pts) < n_particles and tries < max_tries:
    #         tries += 1
    #         cand_xy = np.array([
    #             rng.uniform(xmin, xmax),
    #             rng.uniform(ymin, ymax)
    #         ], dtype=np.float32)

    #         ok = True

    #         # check against points already placed in this blob
    #         for p in pts:
    #             if np.linalg.norm(cand_xy - p[:2]) < min_dist:
    #                 ok = False
    #                 break

    #         # check against pre-existing particles (manual points, previous blob, etc.)
    #         if ok and existing_pts.shape[0] > 0:
    #             ok = check_spacing_against_existing(cand_xy, existing_pts, min_dist)

    #         if ok:
    #             pts.append(np.array([cand_xy[0], cand_xy[1], z_value], dtype=np.float32))

    #     if len(pts) < n_particles:
    #         raise RuntimeError(
    #             f"Could only place {len(pts)} / {n_particles} particles in box "
    #             f"x:[{xmin},{xmax}], y:[{ymin},{ymax}] with min_dist={min_dist}. "
    #             f"Increase box size or reduce n_particles/min_dist."
    #         )

    #     return np.array(pts, dtype=np.float32)

    # # validate manual points first
    # check_manual_points(manual_pts, min_center_dist)

    # # create blob 1 avoiding manual points
    # blob1_pts = sample_blob_with_existing(
    #     blob1_n, blob1_xmin, blob1_xmax, blob1_ymin, blob1_ymax,
    #     z_dyn, min_center_dist, rng,
    #     existing_pts=manual_pts
    # )

    # # create blob 2 avoiding manual points + blob 1
    # existing_for_blob2 = np.vstack((manual_pts, blob1_pts)).astype(np.float32)
    # blob2_pts = sample_blob_with_existing(
    #     blob2_n, blob2_xmin, blob2_xmax, blob2_ymin, blob2_ymax,
    #     z_dyn, min_center_dist, rng,
    #     existing_pts=existing_for_blob2
    # )

    # # combine everything
    # dynamic_pts = np.vstack((manual_pts, blob1_pts, blob2_pts)).astype(np.float32)
    # dynamic_n = dynamic_pts.shape[0]

    # print(f"Generated {manual_pts.shape[0]} manual particles")
    # print(f"Generated blob 1 with {blob1_pts.shape[0]} particles")
    # print(f"Generated blob 2 with {blob2_pts.shape[0]} particles")
    # print(f"Total dynamic particles = {dynamic_n}")
    # print(f"All particles placed at z = {z_dyn:.3f}")

    # print("\nManual particle positions:")
    # for i, p in enumerate(manual_pts, start=1):
    #     print(f"  manual {i}: x={p[0]:.3f}, y={p[1]:.3f}, z={p[2]:.3f}")


    # Combine floor and dynamic particles
    # X = np.vstack((floor_pts, dynamic_pts))
    if USE_FLOOR:
        X = np.vstack((floor_pts, dynamic_pts))
    else:
        X = dynamic_pts.copy()

    X_prev = X.copy()
    freeze_count = np.zeros(X.shape[0], dtype=np.int32)   # how many consecutive “bad” steps


    # Record which indices correspond to the static floor
    floor_indices = np.arange(floor_pts.shape[0], dtype=np.int32)

    # Initialize HIGNN
    NN = X.shape[0]
    hignn_model = hignn.HignnModel(X, 3)
    # three-body interactions
    hignn_model.load_two_body_model('nn/two_body_unbounded')
    # hignn_model.load_three_body_model('nn/three_body')

    # set parameters for far dot, the following parameters are default values
    hignn_model.set_epsilon(0.1)
    hignn_model.set_max_iter(15)
    hignn_model.set_mat_pool_size_factor(30)
    hignn_model.set_post_check_flag(False)
    hignn_model.set_use_symmetry_flag(True)
    hignn_model.set_max_far_dot_work_node_size(10000)
    hignn_model.set_max_relative_coord(1000000)

    # hignn_model.set_epsilon(0.01)
    # hignn_model.set_max_iter(50)
    # hignn_model.set_mat_pool_size_factor(200)
    # hignn_model.set_post_check_flag(False)
    # hignn_model.set_use_symmetry_flag(False)
    # hignn_model.set_max_far_dot_work_node_size(10000)
    # hignn_model.set_max_relative_coord(100000)

    # Setup the time integrator
    # time_integrator = hignn.ExplicitEuler()
    # time_integrator.set_time_step(0.001)
    # time_integrator.set_final_time(0.1)
    # time_integrator.set_num_rigid_body(X.shape[0])
    # time_integrator.set_output_step(10)
    # time_integrator.set_velocity_func(velocity_update)
    # time_integrator.initialize(X)
    # time_integrator.run()

    # --- domain bounds for potential-force periodic images ---
    x_min = X[:,0].min() - 5
    x_max = X[:,0].max() + 5
    y_min = X[:,1].min() - 5
    y_max = X[:,1].max() + 5
    z_min = X[:,2].min() - 5
    z_max = X[:,2].max() + 5

    # # --- create PotentialForce (GLOBAL name) ---
    # potential_force = hignn.PotentialForce()
    # potential_force.set_two_body_epsilon(5)

    # domain = np.array([[x_min, y_min, z_min],
    #                 [x_max, y_max, z_max]], dtype=np.float32)
    # potential_force.set_domain(domain)

    rank_range = np.linspace(0, NN, comm.Get_size() + 1, dtype=np.int32)
    
    # for i in range(10):
    #     if rank == 0:
    #         print(i)
    #     v = velocity_update(0, X)
    ts = 0
    dt = 1e-7
    ite = 0

    time_history = []
    dt_history = []
    step_history = []

    brownian_rng = np.random.default_rng(BROWNIAN_SEED)
    brownian_rms_history = []

    if rank == 0:
        print("\nMobility-based Brownian settings:")
        print(f"  HIGNN_TIME_UNIT_S = {HIGNN_TIME_UNIT_S:.6e} s")
        print(f"  HIGNN_FORCE_SCALE_TIME_N = {HIGNN_FORCE_SCALE_TIME_N:.6e} N/sim_force")
        print(f"  beta = {brownian_beta_sim():.6e}")

    particle_start = floor_pts.shape[0]

    # chain_center = np.array([floor_x_center, floor_y_center, min_dist], dtype=np.float32)
    chain_center = np.array([0.0, 0.0, z_dyn], dtype=np.float32)

    # create a list to hold the max distance at each step
    farthest_points = []
    particle_distances = []

    steady_state_time = None
    position_history = []  # stores last 5 positions of the two particles
    
    t1 = time.time()

    for i in range(12501):
        with h5py.File('Result/pos'+str(ite)+'rank'+str(rank)+'.h5', 'w') as f:
            f.create_dataset('pos', data=X[rank_range[rank]:rank_range[rank+1], :])
        
        tt1 = time.time()
        V = velocity_update(ts, X)

        if rank == 0:
            step_history.append(ite)
            time_history.append(ts)
            dt_history.append(dt)

        # -------- print only dynamic particle positions and velocities --------
        if rank == 0:
            print(f"\nStep {ite}, t = {ts:.8f}")

            # for local_idx, p in enumerate(range(particle_start, particle_start + dynamic_n), start=1):
            #     print(
            #         f"Dynamic particle {local_idx:4d} (global {p+1:4d}): "
            #         f"pos = ({X[p,0]: .8f}, {X[p,1]: .8f}, {X[p,2]: .8f})   "
            #         f"vel = ({V[p,0]: .8e}, {V[p,1]: .8e}, {V[p,2]: .8e})"
            #     )

                # ---- detect "chattering" particles: huge V but almost no net motion ----
        NN = X.shape[0]

        # # dynamic mask: exclude floor (never freeze floor logic here)
        # dyn_mask = np.ones(NN, dtype=bool)
        # if floor_indices is not None and len(floor_indices) > 0:
        #     dyn_mask[np.asarray(floor_indices, dtype=int)] = False

        # speed = np.linalg.norm(V, axis=1)

        # # displacement from previous step (what actually happened last step)
        # if ite == 0:
        #     disp_prev = np.full(NN, np.inf, dtype=np.float64)  # skip detection on step 0
        # else:
        #     disp_prev = np.linalg.norm(X - X_prev, axis=1)

        # # robust “high velocity” threshold from dynamic particles only
        # s_dyn = speed[dyn_mask]
        # med = np.median(s_dyn)
        # mad = np.median(np.abs(s_dyn - med)) + 1e-12
        # v_thresh = med + 25.0 * mad            # tune 15–40 if needed

        # # “didn’t really move” threshold in sim units (tune)
        # disp_eps = 1e-4                        # try 1e-3 if you still see chatter

        # bad_now = dyn_mask & (speed > v_thresh) & (disp_prev < disp_eps)

        # # require K consecutive detections before freezing
        # K = 3
        # freeze_count[bad_now] += 1
        # freeze_count[~bad_now] = 0
        # freeze = freeze_count >= K

        # # zero the velocities
        # if np.any(freeze):
        #     V[freeze] = 0.0
        #     if rank == 0:
        #         idx = np.where(freeze)[0]
        #         print(f"[freeze] {len(idx)} particles: {idx[:20]}{'...' if len(idx)>20 else ''}")


        # # adaptive dt so max displacement per step is small
        sel = slice(particle_start, particle_start + dynamic_n)
        vmax = np.max(np.linalg.norm(V[sel], axis=1))

        dx_max = 0.01
        dt = min(0.1, dx_max / (vmax + 1e-12))

        M_brown = None
        dof_list_brown = None

        if USE_BROWNIAN and USE_HIGNN_MOBILITY_BROWNIAN:
            M_brown, dof_list_brown = build_hignn_submobility_matrix(
                hignn_model,
                X,
                sel,
                active_axes=BROWNIAN_ACTIVE_AXES,
            )

            dt_brown = brownian_dt_limit_from_mobility(
                M_brown,
                max_rms_sim=BROWNIAN_MAX_RMS_SIM,
            )

            dt = min(dt, dt_brown)

        # dt = 5e-6

        if rank == 0:
            print("Time for velocity_update: {t:.4f}s".format(t = time.time() - tt1))

        with h5py.File('Result/vel'+str(ite)+'rank'+str(rank)+'.h5', 'w') as f:
            f.create_dataset('vel', data=V[rank_range[rank]:rank_range[rank+1], :])
        
        dX_det = dt * V

        if USE_BROWNIAN and USE_HIGNN_MOBILITY_BROWNIAN:
            dX_B, evals_M, evals_clip = sample_brownian_from_mobility(
                M_brown,
                dof_list_brown,
                X.shape,
                dt,
                brownian_rng,
                comm=comm,
                rank=rank,
                remove_com_kick=REMOVE_BROWNIAN_COM_KICK,
            )
        else:
            dX_B = np.zeros_like(X, dtype=np.float32)

        X = X + dX_det + dX_B
        ts = ts + dt

        # # trial step
        # X_trial = X + dt * V

        # # HARD STOP: enforce min center-center distance
        # R_MIN_CC = 2.03

        # # If your system is planar (z fixed), use_xy_only=True is faster
        # X_trial, n_bad = project_min_distance(
        #     X_trial,
        #     r_min=R_MIN_CC,
        #     floor_indices=floor_indices if USE_FLOOR else None,
        #     max_iter=30,
        #     tol=1e-6,
        #     use_xy_only=False  # set True if truly 2D-in-plane
        # )

        # if rank == 0 and n_bad > 0:
        #     print(f"[contact] still had {n_bad} violating pairs after projection iterations")

        # # update velocity to match corrected motion (important if you save vel files)
        # V = (X_trial - X) / dt

        # # accept step
        # X = X_trial
        # ts += dt

        X_prev = X.copy()

        # ---------- after X = X + dt*V and ts update ----------

        # 1) compute filament centroid (just beads, not floor)
        
        fil_cent = X[sel, :2].mean(axis=0)   # shape (2,) ← (x,y) of filament

        # 2) compute how far floor needs to shift
        #    floor_center_orig is a global you set once when you build floor_pts
        delta = fil_cent - floor_center_orig[:2]   # (dx,dy)

        # 3) shift ONLY the floor points
        if USE_FLOOR and floor_indices.size > 0:
            fidx = floor_indices
            X[fidx, 0] += delta[0]
            X[fidx, 1] += delta[1]
            floor_center_orig[:2] += delta

        # --- record the farthest bead in the chain from the center ---
        # select the chain beads
        sel = slice(particle_start, particle_start + dynamic_n)
        chain_positions = X[sel]             # shape (n_chain,3)

        # --- steady state detection ---
        position_history.append(chain_positions[:, :2].copy())  # store x,y of both particles
        if len(position_history) > 5:
            position_history.pop(0)  # keep only last 5

        if len(position_history) == 5 and steady_state_time is None:
            # max displacement across all particles over the last 5 steps
            pos_array = np.array(position_history)  # shape (5, 2, 2)
            max_displacement = np.max(np.linalg.norm(pos_array - pos_array[-1], axis=-1))
            
            STEADY_TOL = 0.05  # sim units — tune this if needed
            if max_displacement < STEADY_TOL:
                steady_state_time = ts
                if rank == 0:
                    print(f"*** Steady state reached at t = {ts:.6f}, step {ite} ***")

        if dynamic_n == 2:
            dist = np.linalg.norm(chain_positions[0, :2] - chain_positions[1, :2])
            particle_distances.append(dist)

        V[sel, 2] = 0.0
        X[sel, 2] = z_dyn

        # compute Euclidean distance to the chain_center
        y_offsets = np.abs(chain_positions[:,0] - chain_center[0])
        farthest_points.append(y_offsets.max())
        
        ite = ite + 1
        if rank == 0:
            print()            

    if rank == 0:
        
        # for step, r in enumerate(farthest_points):
        #     print(f"Step {step:4d}: farthest chain bead distance = {r:.4f}")

        # for step, d in enumerate(particle_distances):
        #     print(f"Step {step:4d}: particle distance = {d:.6f}")   

        # if steady_state_time is not None:
        #     print(f"Steady state reached at t = {steady_state_time:.6f}")
        # else:
        #     print("Steady state was NOT reached within the simulation time.")

        print("Time for simulation: {t:.4f}s".format(t = time.time() - t1))

        print(f"t = {ts:.4f}")

    if rank == 0:
        time_phys_out = os.path.join("Result", "time_history_physical_s_rank0.npy")
        dt_phys_out = os.path.join("Result", "dt_history_physical_s_rank0.npy")
        step_out = os.path.join("Result", "step_history_rank0.npy")

        np.save(
            time_phys_out,
            np.asarray(time_history, dtype=np.float64) * HIGNN_TIME_UNIT_S
        )
        np.save(
            dt_phys_out,
            np.asarray(dt_history, dtype=np.float64) * HIGNN_TIME_UNIT_S
        )
        np.save(step_out, np.asarray(step_history, dtype=np.int32))

        print(f"Saved physical time history to: {time_phys_out}")
        print(f"Saved physical dt history to: {dt_phys_out}")

        if USE_BROWNIAN:
            brown_out = os.path.join("Result", "brownian_rms_xy_nm_rank0.npy")
            np.save(brown_out, np.asarray(brownian_rms_history, dtype=np.float64))
            print(f"Saved Brownian RMS history to: {brown_out}")

    if rank == 0 and USE_BROWNIAN:
        rms_B_nm = brownian_rms_xy_nm(dX_B, sel)
        brownian_rms_history.append(rms_B_nm)

        if ite % 50 == 0:
            rms_det_nm = (
                np.sqrt(np.mean(np.sum(dX_det[sel, :2]**2, axis=1)))
                * BROWNIAN_L0 * 1e9
            )

            nneg = np.sum(evals_M < -1e-10) if USE_HIGNN_MOBILITY_BROWNIAN else 0

            print(f"Brownian RMS xy step      = {rms_B_nm:.4f} nm")
            print(f"Deterministic RMS xy step = {rms_det_nm:.4f} nm")
            print(f"M_brown min eig = {np.min(evals_M):.3e}, negative eig count = {nneg}")
   
    # edgeInfo = hignn.BodyEdgeInfo()
    # edgeInfo.setThreeBodyEpsilon(5.0)

    # edgeInfo.setTargetSites(X)

    # edgeInfo.buildThreeBodyEdgeInfo()
    
    del hignn_model
    # del potential_force
    # del edgeInfo
    # del time_integrator

    # Finalize
    hignn.Finalize()
