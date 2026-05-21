#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# --- shield SD from our CLI flags during import ---
import sys as _sys
_real_argv = _sys.argv[:]
_sys.argv  = [_sys.argv[0]]

from stokesian_dynamics.functions.timestepping import generate_output_FTSUOE
from stokesian_dynamics.functions.shared import add_sphere_rotations_to_positions

_sys.argv = _real_argv
# -------------------------------------------------

import numpy as np

# ---------- minimal UFTE driver: holding forces ----------
def solve_holding_forces(X_subset, input_number=60, flow_form='ufte'):
    """
    X_subset: (M,3) array of sphere centers present in this solve.
    Returns:  Fa_out (M,3) hydrodynamic holding forces for UFTE.
    """
    X_subset = np.asarray(X_subset, dtype=float)
    M = X_subset.shape[0]
    sphere_sizes = np.ones(M)
    sphere_rot   = add_sphere_rotations_to_positions(
        X_subset, sphere_sizes, np.array([[1,0,0],[0,0,1]])
    )
    posdata = (sphere_sizes, X_subset, sphere_rot,
               np.array([]), np.empty((0,3)), np.empty((0,3)))

    last_Minfty_inv   = np.array([0])
    regenerate_Minfty = True
    frameno, timestep = 0, 0.1

    ret = generate_output_FTSUOE(
        posdata,
        frameno, timestep,
        input_number,                 # your UFTE "hold-fixed-flow" case
        last_Minfty_inv, regenerate_Minfty,
        flow_form,                    # 'ufte'
        2,                            # cutoff_factor
        0,                            # printout
        False,                        # use_drag_Minfinity
        False,                        # use_Minfinity_only
        False,                        # extract_force_on_wall_due_to_dumbbells
        (np.zeros((M,3)), np.array([]), np.array([]), np.zeros((M,3))),
        [0]*(M*11)
    )
    Fa_out = ret[0]
    return Fa_out

def three_body_increment_forces(X):
    X = np.asarray(X, dtype=float).reshape(3,3)
    F_all = solve_holding_forces(X)                 # (3,3)
    F_ij  = solve_holding_forces(X[[0,1], :])       # (2,3)
    F_ik  = solve_holding_forces(X[[0,2], :])       # (2,3)
    F_jk  = solve_holding_forces(X[[1,2], :])       # (2,3)
    F_i   = solve_holding_forces(X[[0],   :])       # (1,3)
    F_j   = solve_holding_forces(X[[1],   :])       # (1,3)
    F_k   = solve_holding_forces(X[[2],   :])       # (1,3)

    dF3 = np.zeros_like(X)
    dF3[0] = F_all[0] - F_ij[0] - F_ik[0] + F_i[0]
    dF3[1] = F_all[1] - F_ij[1] - F_jk[0] + F_j[0]
    dF3[2] = F_all[2] - F_ik[1] - F_jk[1] + F_k[0]

    return dF3, dict(F_all=F_all, F_ij=F_ij, F_ik=F_ik, F_jk=F_jk,
                     F_i=F_i, F_j=F_j, F_k=F_k)

def decompose_forces_3body(X):
    F_three, P = three_body_increment_forces(X)
    F_all, F_ij, F_ik, F_jk, F_i, F_j, F_k = (
        P['F_all'], P['F_ij'], P['F_ik'], P['F_jk'], P['F_i'], P['F_j'], P['F_k'])
    F_single = np.zeros_like(X)
    F_single[0] = F_i[0]; F_single[1] = F_j[0]; F_single[2] = F_k[0]
    Fpair_ij = np.zeros_like(X); Fpair_ij[0] = F_ij[0] - F_i[0]; Fpair_ij[1] = F_ij[1] - F_j[0]
    Fpair_ik = np.zeros_like(X); Fpair_ik[0] = F_ik[0] - F_i[0]; Fpair_ik[2] = F_ik[1] - F_k[0]
    Fpair_jk = np.zeros_like(X); Fpair_jk[1] = F_jk[0] - F_j[0]; Fpair_jk[2] = F_jk[1] - F_k[0]
    Fpair_sum = Fpair_ij + Fpair_ik + Fpair_jk
    return dict(F_total=P['F_all'],
                F_single=F_single,
                F_pair_sum=Fpair_sum,
                F_three=F_three)

# ---------- sanity checks ----------
def _permute_rows(A, p):
    A = np.asarray(A)
    return A[p, :]

def check_single_sphere_drag(tol=5e-3):
    """
    For UFTE with uniform U∞=(1,0,0), μ=a=1:
    holding force on a single sphere should be ~ +6π e_x.
    NOTE: if you changed case 60 to *shear*, change target accordingly.
    """
    X = np.array([[0.,0.,0.]])
    F = solve_holding_forces(X)[0]
    target = np.array([6*np.pi, 0., 0.])
    err = np.linalg.norm(F - target)
    print("[single_sphere_drag] F =", F, " target =", target, "  L2 err =", err)
    print("PASS" if err < tol else "WARN: large deviation (did you switch to shear?)")

def check_decomposition_identity(tol=1e-10):
    X = np.array([[0.,0.,0.],
                  [3.,0.,0.],
                  [0.,3.,0.]])
    D = decompose_forces_3body(X)
    resid = np.linalg.norm(D['F_total'] - (D['F_single'] + D['F_pair_sum'] + D['F_three']))
    print("[decomposition_identity] residual =", resid)
    print("PASS" if resid < tol else "FAIL")

def check_far_third_decay():
    base = np.array([[0.,0.,0.],
                     [3.,0.,0.],
                     [0.,3.,0.]])
    print("[far_third_decay]  R    ||F_three||")
    prev = None; monotone = True
    for s in [1,2,4,8,16,32]:
        X = base.copy(); X[2] = np.array([0., 3.*s, 0.])
        F3, _ = three_body_increment_forces(X)
        val = np.linalg.norm(F3)
        if prev is not None and val > prev * (1 + 1e-3):  # allow tiny num jitter
            monotone = False
        prev = val
        print(f"  {3*s:4.1f}  {val:.6e}")
    print("PASS (monotone non-increasing)" if monotone else "WARN: not strictly decaying")

def check_translation_invariance(tol=1e-10):
    X = np.array([[0.,0.,0.],
                  [3.,0.,0.],
                  [0.,3.,0.]])
    c = np.array([7.0, -4.0, 2.0])
    D1 = decompose_forces_3body(X)
    D2 = decompose_forces_3body(X + c)
    diffs = {k: np.linalg.norm(D1[k] - D2[k]) for k in D1}
    print("[translation_invariance] L2 diffs:", diffs)
    ok = all(v < tol for v in diffs.values())
    print("PASS" if ok else "FAIL")

def check_permutation_consistency(tol=1e-10):
    X = np.array([[0.2, 0.1, 0.0],
                  [2.7,-0.4, 0.0],
                  [-0.3,1.8, 0.0]])
    D  = decompose_forces_3body(X)
    p  = np.array([1,2,0], dtype=int)  # relabel 0->1,1->2,2->0
    Dp = decompose_forces_3body(X[p])
    diffs = {k: np.linalg.norm(_permute_rows(D[k], p) - Dp[k]) for k in D}
    print("[permutation_consistency] L2 diffs:", diffs)
    ok = all(v < tol for v in diffs.values())
    print("PASS" if ok else "FAIL")

def main():
    print("\n=== Three-body tiny self-test (UFTE, case n==60) ===")
    check_single_sphere_drag()
    check_decomposition_identity()
    check_far_third_decay()
    check_translation_invariance()
    check_permutation_consistency()
    print("\nDone.")

if __name__ == "__main__":
    main()
