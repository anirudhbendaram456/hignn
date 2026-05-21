#!/usr/bin/env python3
import os
import time
import h5py
import numpy as np
import torch
from scipy.optimize import minimize
import hignn
from mpi4py import MPI

# Import the surrogate model and physics helpers from the offline training script
from MPC_filament_offline import SurrogateModel, velocity_update

# --- Simulation & MPC parameters ---
dt          = 0.001      # simulation timestep
k_t         = 100.0      # tension stiffness (unchanged)
rest_length = 2.4        # rest length between beads
max_kb      = 500.0      # allowable range for bending stiffness

# MPC settings
H            = 5         # prediction horizon (number of steps ahead)
rho          = 1e-13      # penalty on rate-of-change of k_b
k_min, k_max = 0.0, max_kb
EMBED_DIM    = 4         # same embedding depth used offline

# --- Build initial filament geometry (same as offline) ---
n_chain = 31
nx, ny, nz = 1, 1, n_chain
dx, dy, dz = 3.0, 3.0, 2.4
x = np.arange(0, nx*dx, dx)
y = np.arange(0, ny*dy, dy)
z = np.arange(0, nz*dz, dz)
zz, yy, xx = np.meshgrid(x, y, z)
X0 = np.concatenate([xx.reshape(-1,1), yy.reshape(-1,1), zz.reshape(-1,1)], axis=1).astype(np.float32)
#X0 += np.random.rand(*X0.shape)*0.5  # small perturbation

# Precompute feature & output dimensions for the surrogate net
num_dof = X0.size                # 3*N beads flattened
feat_dim = EMBED_DIM * num_dof + 1
out_dim  = num_dof               # velocity field flattened

# --- Helper: predict velocity from the surrogate twin ---
@torch.no_grad()
def predict_velocity(history_states, kb, model):
    inp = np.concatenate([h.reshape(-1) for h in history_states] + [[kb]])
    x = torch.from_numpy(inp).float().unsqueeze(0)
    v_hat = model(x).cpu().numpy().reshape(history_states[0].shape)
    return v_hat

# --- MPC solver: optimize a sequence of bending stiffnesses ---
def solve_mpc(history, k_prev, model, X_ref_seq):
    def cost_fn(k_seq):
        cost = 0.0
        sim_hist = history.copy()
        for i, kb in enumerate(k_seq):
            V_hat = predict_velocity(sim_hist[-EMBED_DIM:], kb, model)
            X_next = sim_hist[-1] + dt * V_hat
            cost += np.sum((X_next - X_ref_seq[i])**2)                    # tracking error
            if i == 0:
                cost += rho * (kb - k_prev)**2                               # smoothness from previous
            else:
                cost += rho * (kb - k_seq[i-1])**2                          # smoothness step-to-step
            sim_hist.append(X_next)
        return cost

    k0 = np.full(H, k_prev, dtype=np.float32)                                # initial guess
    bounds = [(k_min, k_max)] * H
    res = minimize(cost_fn, k0, bounds=bounds, method="L-BFGS-B")
    return res.x

# --- Main closed-loop control loop ---
def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Create directory for results
    if rank == 0:
        os.makedirs('Result', exist_ok=True)
    comm.Barrier()

    # 1) Load surrogate twin
    model = SurrogateModel(feat_dim, out_dim)
    model.load_state_dict(torch.load('surrogate_g_model.pth'))
    model.eval()

    # 2) Initialize high-fidelity HIGNN simulator
    hignn.Init()
    hmodel = hignn.HignnModel(X0, 15)
    hmodel.load_two_body_model('nn/two_body_unbounded')

    # 3) Set up initial state & history buffer
    X_t = X0.copy()
    history = [X_t.copy() for _ in range(EMBED_DIM)]
    k_prev = 0 #(k_min + k_max) / 2.0

    # 4) Define reference shape using provided x, z arrays
    # x_ref = np.array([14.0, 15.4667, 16.9333, 18.4, 19.8667, 21.3333, 22.8, 24.2667,
    #                   25.7333, 27.2, 28.6667, 30.1333, 31.6, 33.0667, 34.5333,
    #                   36.0, 37.4667, 38.9333, 40.4, 41.8667, 43.3333, 44.8,
    #                   46.2667, 47.7333, 49.2, 50.6667, 52.1333, 53.6, 55.0667,
    #                   56.5333, 58.0])
    # z_ref = np.array([-150.8, -153.9191, -156.8231, -159.512 , -161.9858, -164.2444,
    #                   -166.288 , -168.1164, -169.7298, -171.128 , -172.3111, -173.2791,
    #                   -174.032 , -174.5698, -174.8924, -175.   , -174.8924, -174.5698,
    #                   -174.032 , -173.2791, -172.3111, -171.128 , -169.7298, -168.1164,
    #                   -166.288 , -164.2444, -161.9858, -159.512 , -156.8231, -153.9191,
    #                   -150.8])
    # y_ref = np.zeros_like(x_ref)
    # X_ref = np.vstack([x_ref, y_ref, z_ref]).T.astype(np.float32)
    X_ref = X0.copy()
    X_ref_seq = [X_ref for _ in range(H)]

    # 5) Closed-loop: solve MPC, apply, and log
    t_start = time.time()
    for step in range(2000):
        # write positions every 10 steps
        if step % 1 == 0:
            with h5py.File(f'Result/pos{step}rank{rank}.h5', 'w') as f:
                f.create_dataset('pos', data=X_t)

        # compute control via MPC
        solve_start = time.time()
        k_seq = solve_mpc(history, k_prev, model, X_ref_seq)
        k_use = k_seq[0]

        # measure surrogate call time
        tt1 = time.time()
        V = velocity_update(X_t, k_use, k_t, rest_length, hmodel)
        if rank == 0:
            print(f"Time for velocity_update: {time.time()-tt1:.4f}s")

        # write velocities every 10 steps
        if step % 1 == 0:
            with h5py.File(f'Result/vel{step}rank{rank}.h5', 'w') as f:
                f.create_dataset('vel', data=V)

        # apply to high-fidelity model
        X_next = X_t + dt * V

        # update state and history
        history.pop(0)
        history.append(X_next)
        X_t, k_prev = X_next, k_use

        if rank == 0:
            print(f"[Step {step}] Applied k_b = {k_use:.3f}")

    # total simulation time
    if rank == 0:
        print(f"Total simulation time: {time.time()-t_start:.4f}s")

    # 6) Finalize HIGNN
    del hmodel
    hignn.Finalize()

if __name__ == '__main__':
    main()
