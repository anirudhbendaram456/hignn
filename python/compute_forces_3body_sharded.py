#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ------------------- guard SD from our CLI flags during import -------------------
import sys as _sys
_real_argv = _sys.argv[:]
_sys.argv = [_sys.argv[0]]

from stokesian_dynamics.functions.shared import add_sphere_rotations_to_positions
from stokesian_dynamics.functions.generate_grand_resistance_matrix import generate_grand_resistance_matrix
from stokesian_dynamics.setups.inputs import input_ftsuoe

_sys.argv = _real_argv
# ---------------------------------------------------------------------------------

import os, argparse, numpy as np
from multiprocessing import Pool

# ================================================================
# Helpers: background preparation (ONE call per triplet)
# ================================================================

def _mk_posdata(centres):
    """Build the posdata tuple SD expects for a set of spheres at `centres`."""
    centres = np.asarray(centres, float)
    m = centres.shape[0]
    sizes = np.ones(m, dtype=float)
    sphere_rot = add_sphere_rotations_to_positions(
        centres, sizes, np.array([[1,0,0],[0,0,1]], dtype=float)
    )
    # No dumbbells here
    dumb_sizes  = np.array([])
    dumb_pos    = np.empty((0,3))
    dumb_deltax = np.empty((0,3))
    return (sizes, centres, sphere_rot, dumb_sizes, dumb_pos, dumb_deltax)

def prepare_background_for_triplet(X_triplet, input_number=60, input_form='ufte'):
    """
    Call inputs.py ONCE for the 3-sphere triplet. Reuse the returned background
    for every subset (pairs/singles) so inclusion–exclusion remains consistent.
    """
    X_triplet = np.asarray(X_triplet, float).reshape(3,3)
    posdata = _mk_posdata(X_triplet)

    (Fa_in, Ta_in, Sa_in, Sa_c_in, Fb_in, DFb_in,
     Ua_in, Oa_in, Ea_in, Ea_c_in,
     Ub_in, HalfDUb_in, desc, Uinf, Oinf, centre,
     Ot, Et, boxL, boxR, mu) = input_ftsuoe(
        input_number, posdata, frameno=0, timestep=0.1,
        last_velocities=(np.zeros((3,3)), np.array([]), np.array([]), np.zeros((3,3))),
        input_form=input_form, skip_computation=False, grand_resistance_matrix_fte=0
    )

    # E3 (full 3x3) for possible kinematic folding; E5 (5 comps) for an E-block if present.
    if isinstance(Ea_in, (list, tuple, np.ndarray)):
        E3 = np.array(Ea_in[0], dtype=float)  # usually identical per sphere
    else:
        E3 = np.zeros((3,3), dtype=float)

    if isinstance(Ea_c_in, np.ndarray):
        if Ea_c_in.ndim == 1 and Ea_c_in.size == 5:
            E5 = np.tile(Ea_c_in[None, :], (3,1))  # (3,5)
        elif Ea_c_in.ndim == 2 and Ea_c_in.shape[1] == 5:
            E5 = Ea_c_in[:3, :]
        else:
            E5 = None
    else:
        E5 = None

    bg = dict(
        Uinf=np.asarray(Uinf, float).reshape(3,),
        Oinf=np.asarray(Oinf, float).reshape(3,),
        centre=np.asarray(centre, float).reshape(3,),
        E3=E3,                 # 3x3 full tensor
        E5=E5,                 # (3,5) compressed, or None
        mu=float(mu)
    )
    return bg

# ================================================================
# Core solver: uses the SAME background for any subset
# ================================================================

def _build_R_for_subset(centres_subset, mu):
    """
    Call stokesian_dynamics.functions.generate_grand_resistance_matrix with
    version-agnostic handling and ALWAYS return (R, heading).
    """
    posdata = _mk_posdata(centres_subset)
    last_Minfty_inv = np.array([0])

    # Try common signatures in order, catching TypeError when args don't match
    out = None
    try:
        # Newer/common: expects last_Minfty_inv as 2nd *positional* arg
        out = generate_grand_resistance_matrix(
            posdata, last_Minfty_inv,
            regenerate_Minfinity=True, cutoff_factor=2, printout=0,
            use_drag_Minfinity=False, use_Minfinity_only=False,
            frameno=0, mu=mu
        )
    except TypeError:
        try:
            # Variant without last_Minfty_inv
            out = generate_grand_resistance_matrix(
                posdata,
                regenerate_Minfinity=True, cutoff_factor=2, printout=0,
                use_drag_Minfinity=False, use_Minfinity_only=False,
                frameno=0, mu=mu
            )
        except TypeError:
            # Very old/minimal signature
            out = generate_grand_resistance_matrix(posdata, last_Minfty_inv)

    # Normalize to (R, heading)
    if isinstance(out, tuple):
        if len(out) == 4:
            R, heading, Minv, times = out
        elif len(out) == 3:
            R, heading, times = out
        elif len(out) == 2:
            R, heading = out
        else:
            # Unexpected arity; best effort
            R, heading = out[0], out[1]
    else:
        # Some exotic builds could return an object with attributes
        try:
            R, heading = out.R, out.heading
        except Exception as e:
            raise RuntimeError(f"Unexpected return from generate_grand_resistance_matrix: {type(out)}") from e

    return R, heading


def _detect_E_block(R, m):
    """
    If the grand resistance uses [U(3m), Ω(3m), E(5m)] -> ..., then ncols = 11m.
    If it uses only [U, Ω] and treats linear flow via kinematics, then ncols = 6m.
    """
    ncols = R.shape[1]
    if ncols == 11*m:
        return True
    if ncols == 6*m:
        return False
    raise ValueError(f"Unexpected R shape: {R.shape}; cannot infer E-block presence.")

def solve_holding_forces_subset(X_subset, bg):
    """
    Holding forces for a subset under the SAME background as the triplet.
    Robust to SD variants:
      - If R has an E-block but bg['E5'] is missing, we emulate the no-E-block path:
        fold E3·x into Urel and pass zeros for the E-block (shape-compatible).
    Returns: F (m,3)
    """
    X_subset = np.asarray(X_subset, float).reshape(-1,3)
    m = X_subset.shape[0]

    R, heading = _build_R_for_subset(X_subset, bg['mu'])
    has_E_block = _detect_E_block(R, m)

    xrel = X_subset - bg['centre']                    # (m,3)
    Urel = -(bg['Uinf'] + np.cross(np.tile(bg['Oinf'], (m,1)), xrel))  # (m,3)
    Orel = -np.tile(bg['Oinf'], (m,1))                                     # (m,3)

    uomega = None
    if has_E_block:
        if bg.get('E5') is not None:
            # Normal path: do NOT fold E·x into Urel; feed E via the E-block
            uomega = np.concatenate([Urel.ravel(), Orel.ravel()], axis=0)
            E5_subset = np.tile(bg['E5'][0], (m,1))   # (m,5)  (adjust if per-sphere varies)
            v = np.concatenate([uomega, (-E5_subset).ravel()], axis=0)
        else:
            # Fallback path: emulate no-E-block
            # 1) Inject the linear flow into Urel
            Urel = Urel - (xrel @ bg['E3'].T)
            uomega = np.concatenate([Urel.ravel(), Orel.ravel()], axis=0)
            # 2) Provide zeros for E-block so dimensions match
            E5_subset = np.zeros((m,5), dtype=float)
            v = np.concatenate([uomega, E5_subset.ravel()], axis=0)
    else:
        # No E-block expected: fold E into Urel and pass only [U,Ω]
        Urel = Urel - (xrel @ bg['E3'].T)
        uomega = np.concatenate([Urel.ravel(), Orel.ravel()], axis=0)
        v = uomega

    y = R @ v
    F = y[:3*m].reshape(m,3)
    return F


# ================================================================
# Three-body decomposition with frozen background
# ================================================================

def three_body_increment_forces(X):
    """Return ΔF^(3) and the pieces for a triplet X (3,3)."""
    X = np.asarray(X, float).reshape(3,3)
    bg = prepare_background_for_triplet(X)

    F_all = solve_holding_forces_subset(X, bg)              # (3,3)
    F_ij  = solve_holding_forces_subset(X[[0,1], :], bg)    # (2,3)
    F_ik  = solve_holding_forces_subset(X[[0,2], :], bg)    # (2,3)
    F_jk  = solve_holding_forces_subset(X[[1,2], :], bg)    # (2,3)
    F_i   = solve_holding_forces_subset(X[[0],   :], bg)    # (1,3)
    F_j   = solve_holding_forces_subset(X[[1],   :], bg)    # (1,3)
    F_k   = solve_holding_forces_subset(X[[2],   :], bg)    # (1,3)

    # Inclusion–exclusion for true 3-body increment per sphere
    dF3 = np.zeros_like(X)
    dF3[0] = F_all[0] - F_ij[0] - F_ik[0] + F_i[0]
    dF3[1] = F_all[1] - F_ij[1] - F_jk[0] + F_j[0]
    dF3[2] = F_all[2] - F_ik[1] - F_jk[1] + F_k[0]

    pieces = dict(F_all=F_all, F_ij=F_ij, F_ik=F_ik, F_jk=F_jk,
                  F_i=F_i, F_j=F_j, F_k=F_k)
    return dF3, pieces

def decompose_forces_3body(X):
    """
    For a triplet X (3,3), return:
      F_total, F_single, F_pair_ij, F_pair_ik, F_pair_jk, F_pair_sum, F_three
    with identity: F_total = F_single + F_pair_sum + F_three.
    """
    F_three, P = three_body_increment_forces(X)
    F_all, F_ij, F_ik, F_jk, F_i, F_j, F_k = (
        P['F_all'], P['F_ij'], P['F_ik'], P['F_jk'], P['F_i'], P['F_j'], P['F_k'])

    F_single = np.zeros_like(X)
    F_single[0] = F_i[0]; F_single[1] = F_j[0]; F_single[2] = F_k[0]

    Fpair_ij = np.zeros_like(X); Fpair_ij[0] = F_ij[0] - F_i[0]; Fpair_ij[1] = F_ij[1] - F_j[0]
    Fpair_ik = np.zeros_like(X); Fpair_ik[0] = F_ik[0] - F_i[0]; Fpair_ik[2] = F_ik[1] - F_k[0]
    Fpair_jk = np.zeros_like(X); Fpair_jk[1] = F_jk[0] - F_j[0]; Fpair_jk[2] = F_jk[1] - F_k[0]
    Fpair_sum = Fpair_ij + Fpair_ik + Fpair_jk

    return dict(F_total=F_all, F_single=F_single,
                F_pair_ij=Fpair_ij, F_pair_ik=Fpair_ik, F_pair_jk=Fpair_jk,
                F_pair_sum=Fpair_sum, F_three=F_three)

# -------------- Your requested spot-check function ----------------

def check_identity_once(X):
    D = decompose_forces_3body(X)
    lhs = D['F_total']
    rhs = D['F_single'] + D['F_pair_sum'] + D['F_three']
    res = np.max(np.abs(lhs - rhs))
    print("max |F_total - (F_single + F_pair_sum + F_three)| =", res)
    return res

# top-level (picklable) worker for Pool.map
def _compute_row(x_flat):
    X = np.asarray(x_flat, dtype=float).reshape(3,3)
    D = decompose_forces_3body(X)
    return (D['F_total'].reshape(9),
            D['F_pair_sum'].reshape(9),
            D['F_three'].reshape(9),
            D['F_single'].reshape(9),
            D['F_pair_ij'].reshape(9),
            D['F_pair_ik'].reshape(9),
            D['F_pair_jk'].reshape(9))

# ================================================================
# IO + sharding
# ================================================================

def load_positions(path):
    import numpy as np

    def _coerce_to_float2d(X):
        # If it's a single sample, make it (1, ...)
        if X.ndim == 1:
            X = X[None, ...]
        # Accept (N,3,3) or (N,9)
        if X.ndim == 3 and X.shape[1:] == (3,3):
            X = X.reshape(X.shape[0], 9)
        elif X.ndim == 2 and X.shape[1] == 9:
            pass
        else:
            raise ValueError(f"Expected shape (N,9) or (N,3,3), got {X.shape}")
        return np.asarray(X, dtype=float)

    if path.endswith('.npy'):
        X = np.load(path, allow_pickle=True)
    elif path.endswith('.npz'):
        npz = np.load(path, allow_pickle=True)
        for key in ('positions', 'pos', 'X', 'arr_0'):
            if key in npz.files:
                X = npz[key]; break
        else:
            X = npz[npz.files[0]]
    else:
        X = np.loadtxt(path)

    if getattr(X, 'dtype', None) == object:
        try:
            X = np.stack([np.ravel(np.array(row, dtype=float)) for row in X], axis=0)
        except Exception as e:
            raise ValueError(f"Could not coerce object array to float: {e}")

    X = _coerce_to_float2d(np.array(X))
    return X

# ================================================================
# CLI
# ================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  required=True, help="path to positions (.npy/.npz/.txt), shape (N,9) or (N,3,3)")
    ap.add_argument("--outdir", default="data_output")
    ap.add_argument("--shard",  type=int, default=None, help="rows per shard; default = all rows in one shard")
    ap.add_argument("--workers", type=int, default=1, help="processes per shard")
    ap.add_argument("--txt", action="store_true", help="also write per-shard human-readable .txt summaries")
    ap.add_argument("--check", action="store_true", help="assert identity on each shard result array")
    ap.add_argument("--spotcheck", type=int, default=0, help="run spot checks on N random triplets using check_identity_once")
    ap.add_argument("--spotseed", type=int, default=12345, help="RNG seed for spot checks")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    data_pos = load_positions(args.input)  # (N,9)
    N = data_pos.shape[0]

    # Optional spot checks before heavy processing
    if args.spotcheck > 0:
        rng = np.random.default_rng(args.spotseed)
        idx = rng.choice(N, size=min(args.spotcheck, N), replace=False)
        print(f"[spotcheck] Running {len(idx)} random triplet checks...")
        vals = []
        for i in idx:
            X = data_pos[i].reshape(3,3)
            vals.append(check_identity_once(X))
        print(f"[spotcheck] max residual over samples: {np.max(vals):.3e}")

    shard_size = args.shard or N
    shard_idx = 0

    for start in range(0, N, shard_size):
        end = min(start + shard_size, N)
        Xchunk = data_pos[start:end]
        nrows = Xchunk.shape[0]
        print(f"[shard {shard_idx}] rows {start}:{end} (n={nrows}), workers={args.workers}")

        # compute in parallel per row
        if args.workers > 1:
            with Pool(processes=args.workers) as pool:
                results = pool.map(_compute_row, Xchunk)
        else:
            results = list(map(_compute_row, Xchunk))

        # unpack
        F_total, F_pair_sum, F_three, F_single, F_ij, F_ik, F_jk = [
            np.stack([r[k] for r in results], axis=0) for k in range(7)
        ]

        if args.check:
            # Assert identity: F_total ≈ F_single + F_pair_sum + F_three
            resid = F_total - (F_single + F_pair_sum + F_three)
            rmax = float(np.max(np.abs(resid)))
            print(f"  [check] max|residual| = {rmax:.3e}")
            if not np.allclose(F_total, F_single + F_pair_sum + F_three, atol=1e-6, rtol=1e-6):
                print("  [warn] identity deviation above tolerance on this shard")

        # save this shard (compact .npy)
        base = os.path.join(args.outdir, f"triplet_forces_shard{shard_idx:05d}")
        np.save(base + "_positions.npy",   Xchunk.astype(np.float32))
        np.save(base + "_F_total.npy",     F_total.astype(np.float32))
        np.save(base + "_F_pair_sum.npy",  F_pair_sum.astype(np.float32))
        np.save(base + "_F_three.npy",     F_three.astype(np.float32))
        np.save(base + "_F_single.npy",    F_single.astype(np.float32))
        np.save(base + "_F_ij.npy",        F_ij.astype(np.float32))
        np.save(base + "_F_ik.npy",        F_ik.astype(np.float32))
        np.save(base + "_F_jk.npy",        F_jk.astype(np.float32))

        if args.txt:
            summary = np.hstack([Xchunk, F_total, F_pair_sum, F_three])
            np.savetxt(base + "_summary.txt", summary,
                       header='[pos(9)] [F_total(9)] [F_pair_sum(9)] [F_three(9)]')

        shard_idx += 1

if __name__ == "__main__":
    main()
