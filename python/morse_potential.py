import os
import hignn
import numpy as np
import sys
from mpi4py import MPI
import time
import h5py
from scipy.spatial import cKDTree

# ─── New block ────────────────────────────────────────────────────────────────
# Physical constants for screened Coulomb
epsilon0  = 8.854187817e-12       # vacuum permittivity (F/m)
epsilon_r = 78.5                  # relative permittivity of water
kB        = 1.380649e-23          # Boltzmann constant (J/K)
T         = 298.15                # temperature (K)
e_charge  = 1.602176634e-19       # elementary charge (C)
NA        = 6.02214076e23         # Avogadro’s number (1/mol)

# Your chosen ionic strength (mol/m^3; =Molar if you use mol/L)
I         = 1e-3                  # e.g. 1 mM
# Debye screening parameter κ (1/m)
kappa     = np.sqrt(2*NA*e_charge**2 * I/(epsilon0*epsilon_r*kB*T))
# r_cut     = 5.0 / kappa  # e.g. 5 Debye lengths

# --- convert Debye to simulation‐units ---
L0         = 1e-6     # meters per simulation‐unit (adjust to your length‐scale)
kappa_sim  = kappa * L0
r_cut      = 5.0 / kappa_sim
# ───────────────────────────────────────────────────────────────────────────────

os.system("clear")

def velocity_update(t, position):
    # update coordinates
    hignn_model.update_coord(position[:, 0:3])
    
    # update force/potential
    force = potential_force.get_potential_force(position[:, 0:3])
    
    # create velocity array
    velocity = np.zeros((position.shape[0], 3), dtype=np.float32)
    
    hignn_model.dot(velocity, force)
    
    return velocity

# def velocity_update(t, position):
#     # print time on rank 0
#     if rank == 0:
#         print(f"t = {t:.4f}")

#     # update the HIGNN model’s coordinates
#     hignn_model.update_coord(position[:, :3])

#     # get the inter-particle potential forces
#     force = potential_force.get_potential_force(position[:, :3]).astype(np.float32)

#     # add gravity/body-force in −Z
#     force[:, 2] += -1.0

#     # --- ADD EXTERNAL ELECTRIC FORCE q·E ---
#     # position.shape == (NN,3)
#     # broadcast charges and field to get an (NN,3) electric-force array
#     # F_elec = charges[:, None] * E_field[None, :]
#     # force += F_elec

#         # ─── new: screened‐Coulomb (Yukawa) pairwise repulsion ────────────────────
#     pos = position[:, :3]                            # (N,3)
#     tree = cKDTree(pos)
#     neighbors = tree.query_ball_tree(tree, r_cut)

#     if rank==0:
#         counts = [len(js) for js in neighbors]
#         print(f"[t={t:.4f}]  min/neigh/max neighbors = {min(counts)}/{np.mean(counts):.1f}/{max(counts)}")


#     F_yuk = np.zeros_like(pos, dtype=np.float32)
#     for i, js in enumerate(neighbors):
#         if not js: continue
#         diffs = pos[i] - pos[js]                     # (ni,3)
#         dists = np.linalg.norm(diffs, axis=1)
#         mask  = dists>0
#         diffs = diffs[mask]; dists = dists[mask]

#         pref = (charges[i] * charges[js][mask] * e_charge**2
#                 /(4*np.pi*epsilon0*epsilon_r))
#         yuk  = np.exp(-kappa_sim*dists)/dists
#         coef = pref * yuk * (1.0/dists + kappa_sim)

#         if rank==0 and i==0:
#             print(" example dists[:5], yuk[:5], coef[:5] =",
#                 dists[:5], yuk[:5], coef[:5])

#         # force on i
#         F_i = np.einsum('n,nj->j', coef, diffs/dists[:,None])
#         F_yuk[i] += F_i
#         # Newton’s 3rd: equal-and-opposite on js
#         F_yuk[js][mask] -= (diffs/dists[:,None]*coef[:,None])

#     force += 1*F_yuk
   
#     # ─── log average Yukawa force magnitude ───────────────────────────────
#     # compute per‐particle magnitudes
#     mags = np.linalg.norm(F_yuk, axis=1)
#     # average over all particles
#     avg_mag = mags.mean()
#     if rank == 0:
#         print(f"[t={t:.4f}] avg |F_yuk| = {avg_mag:.6e}")
#     # ─────────────────────────────────────────────────────────────────────────

#     # prepare velocity array
#     velocity = np.zeros_like(position, dtype=np.float32)

#     # perform the mobility solve
#     hignn_model.dot(velocity, force)

#     return velocity


if __name__ == '__main__':
    # initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    hignn.Init()
    
    ##X = np.loadtxt('output0.txt', dtype=np.float32)
    # X = np.loadtxt('python/cloud/particle_cloud1.pos', dtype=np.float32)

    # N = 2
    nx = 2
    ny = 1
    nz = 1
    dx = 1
    x = np.arange(0, nx * dx, dx)
    y = np.arange(0, ny * dx, dx)
    z = np.arange(0, nz * dx, dx)
    xx, yy, zz = np.meshgrid(x, y, z)
    X = np.concatenate(
        (xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)), axis=1)
    X = X.astype(np.float32)
    
    NN = X.shape[0]

    # # --- ELECTROSTATICS SETUP ---
    # # Give every particle the same charge q (you can make this a vector if you want per-particle variation)
    # after X = … and NN = X.shape[0]:
    # q = 1e11
    # charges = np.zeros((NN,), dtype=np.float32)

    # # compute threshold halfway between min(Z) and max(Z)
    # z_vals = X[:,2]
    # threshold_z = 0.5*(np.min(z_vals) + np.max(z_vals))
    # # threshold_z = np.median(z_vals)

    # # give charge only to those above threshold_z (the “top” drop)
    # charges[z_vals < threshold_z] = q
    # charges[z_vals > threshold_z] = q
    
    # # # Define a constant electric field E⃗ (for example, pointing upward along +Z)
    # E_field = np.array([0.0, 0.0,  1.0], dtype=np.float32)  
    # # Adjust the third component to tune field strength

    x_min = np.min(X[:, 0]) - 5
    x_max = np.max(X[:, 0]) + 5
    y_min = np.min(X[:, 1]) - 5
    y_max = np.max(X[:, 1]) + 5
    z_min = np.min(X[:, 2]) - 5
    z_max = np.max(X[:, 2]) + 5

    hignn_model = hignn.HignnModel(X, 1)
    
    hignn_model.load_two_body_model('nn/two_body_unbounded')
    
    # set parameters for far dot, the following parameters are default values
    # warning: hignn_model does not support periodic boundary conditions
    hignn_model.set_epsilon(0.1)
    hignn_model.set_max_iter(15)
    hignn_model.set_mat_pool_size_factor(30)
    hignn_model.set_post_check_flag(False)
    hignn_model.set_use_symmetry_flag(True)
    hignn_model.set_max_far_dot_work_node_size(10000)
    hignn_model.set_max_relative_coord(1000000)
    
    # setup time integrator
    # warning: only ExplicitEuler supports periodic boundary conditions
    # time_integrator = hignn.ExplicitEuler()
    
    # time_integrator.set_time_step(0.01)
    # time_integrator.set_final_time(80)
    # time_integrator.set_num_rigid_body(NN)
    # time_integrator.set_output_step(1)

    # time_integrator.set_x_lim([x_min, x_max])
    # time_integrator.set_y_lim([y_min, y_max])
    # time_integrator.set_z_lim([z_min, z_max])
    
    # time_integrator.set_velocity_func(velocity_update)
    # time_integrator.initialize(X)

    dt     = 0.1
    t_max  = 100
    n_steps = int(t_max / dt)
    
    potential_force = hignn.PotentialForce()
    potential_force.set_two_body_epsilon(7.5)
    
    domain = np.array([[x_min, y_min, z_min],[x_max, y_max, z_max]], dtype=np.float32)
    potential_force.set_domain(domain)
    
    # rank‐based slicing
    rank_range = np.linspace(0, NN, comm.Get_size()+1, dtype=np.int32)

    # time‐loop
    ts  = 0.0
    ite = 0
    t1  = time.time()

    farthest_points_x = []
    farthest_points_y = []

    for step in range(5001):
        # 1) write out positions for this step
        os.makedirs('Result', exist_ok=True)
        with h5py.File(f'Result/pos{ite}rank{rank}.h5','w') as f:
            f.create_dataset('pos',
                data=X[ rank_range[rank]:rank_range[rank+1], : ])

        # 2) compute velocity
        tt1 = time.time()
        V = velocity_update(ts, X)
        if rank == 0:
            print(f"Time for velocity_update: {time.time()-tt1:.4f}s")

        # 3) write out velocities
        with h5py.File(f'Result/vel{ite}rank{rank}.h5','w') as f:
            f.create_dataset('vel',
                data=V[ rank_range[rank]:rank_range[rank+1], : ])

        # 4) Euler update
        X  = X + dt * V
        ts += dt
        ite += 1

        x_offsets = np.abs(X[0,0])
        y_offsets = np.abs(X[0,1])
        farthest_points_x.append(x_offsets.max())
        farthest_points_y.append(y_offsets.max())

        if rank == 0:
            print(f" Completed step {step+1}/{n_steps}\n")

    if rank == 0:
        print("Time for simulation: {t:.4f}s".format(t = time.time() - t1))
        for step, r in enumerate(farthest_points_x):
            print(f"Step {step:4d}: farthest chain bead distance in x = {r:.4f}")
        # for step, r in enumerate(farthest_points_y):
        #     print(f"Step {step:4d}: farthest chain bead distance in y = {r:.4f}")

    # total time
    if rank == 0:
        print(f"Total simulation time: {time.time() - t1:.2f} s")

    # cleanup
    # del potential_force
    
    del hignn_model
    # del time_integrator
    del potential_force
    
    hignn.Finalize()