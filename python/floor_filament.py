import os
import hignn
import numpy as np
import sys
from mpi4py import MPI
import time
import h5py
import torch
import torch.nn as nn
from torch_scatter import scatter
# from HIGNN.model_structure import HIGNN_model
import HIGNN.model_structure as ms

# Bring in the real class…
HIGNN_model = ms.HIGNN_model

# …and alias it to the typo’d name in __main__
HIGNN_mdoel = HIGNN_model

# --- Filament and floor simulation ---

def chain_bending(X, k_b):
    N = X.shape[0]
    F = np.zeros((N, 3), dtype=np.float32)
    arm1 = X[1:-1] - X[:-2]
    arm2 = X[2:] - X[1:-1]
    n1 = arm1 / np.linalg.norm(arm1, axis=1, keepdims=True)
    n2 = arm2 / np.linalg.norm(arm2, axis=1, keepdims=True)
    cosθ = np.einsum('ij,ij->i', n1, n2)
    sinθ = np.sqrt(np.clip(1 - cosθ**2, 0, None))
    axis = np.cross(n1, n2)
    torque = -axis * sinθ[:, None]
    f21 = np.cross(torque, n1)
    f23 = np.cross(torque, n2)
    F[:-2] += f21
    F[1:-1] -= (f21 + f23)
    F[2:] += f23
    return k_b * F


def chain_tension(X, k_t, rest_length):
    N = X.shape[0]
    F = np.zeros((N, 3), dtype=np.float32)
    diffs = X[:-1] - X[1:]
    d = np.linalg.norm(diffs, axis=1, keepdims=True)
    u = diffs / d
    stretch = k_t * (d - 2.6)#rest_length)
    f = u * stretch
    F[:-1] -= f
    F[1:] += f
    return F

# # Create Edge
# # 3body edge: (j, k, i), attr: F_j, from j to k to i
# # 2body_self edge: (j, i), attr: F_i, from i to j to i
def build_neighbor_list(X, three_body_cutoff):
    
    neighbor_list = hignn.NeighborLists()
    neighbor_list.set_three_body_epsilon(three_body_cutoff)
    neighbor_list.update_coord(X)
    neighbor_list.build_three_body_info()
    three_body_edge_info = neighbor_list.get_three_body_edge_info()
    three_body_edge_info = torch.tensor(three_body_edge_info, dtype=torch.long)
    # print(three_body_edge_info)
    two_body_edge_self_info = neighbor_list.get_three_body_edge_self_info()
    two_body_edge_self_info = torch.tensor(two_body_edge_self_info, dtype=torch.long)
    # print(two_body_edge_self_info)
    del neighbor_list
    return three_body_edge_info, two_body_edge_self_info

# calculate velocity without 2body
def velocity_update_without_2body(X, force, device):

    edge_3body, edge_2bodySelf = build_neighbor_list(X, 5.0)
    edge_3body = edge_3body.to(device)
    edge_2bodySelf = edge_2bodySelf.to(device)
    edge_2body = torch.zeros((2, 0), dtype=torch.long).to(device)
    force = torch.tensor(force, dtype = torch.float32).to(device)

    edge_attr_2body = force[edge_2body[0, :], :]
    edge_attr_3body = force[edge_3body[0, :], :]
    edge_attr_2bodySelf = force[edge_2bodySelf[1, :], :]
    Nc = X.shape[0]
    edge_1body = torch.arange(0, Nc, 1).reshape((1, Nc))
    edge_1body = edge_1body.long().to(device)
    edge_attr_1body = force
    X = torch.tensor(X, dtype = torch.float32)
    X = X.to(device)
    with torch.no_grad():
        velocity = original_hignn(X, edge_2body, edge_3body, edge_2bodySelf, edge_1body, edge_attr_2body, edge_attr_3body, edge_attr_2bodySelf, edge_attr_1body).cpu().numpy()
    torch.cuda.empty_cache()
    
    return velocity

def velocity_update(t, position):
    if rank == 0:
        print(f"t = {t:.4f}", flush = True)

    hignn_model.update_coord(position[:, :3])
    vel = np.zeros_like(position, dtype=np.float32)
    force = np.zeros_like(position, dtype=np.float32)

    # force = potential_force.get_potential_force(position[:, 0:3])

    # pot_force = force.copy()

    # gravity on all particles
    # force[:, 0] += 4.8436

    # apply filament forces on beads
    sel = slice(particle_start, particle_start + n_chain)
    pos_chain = position[sel]
    force[sel] += chain_bending(pos_chain, k_b)

    force[particle_start,1] += np.float32(4.8436e-1)
    force[particle_start+1,1] -= np.float32(4.8436e-1)

    tension_force = chain_tension(pos_chain, k_t, rest_length) 
    force[sel] += tension_force

    # avg_pot     = np.linalg.norm(pot_force,     axis=1).mean()
    # avg_tension = np.linalg.norm(tension_force, axis=1).mean()

    # # 5) Print them (only on rank 0)
    # if rank == 0:
    #     print(f"    avg |F_pot|     = {avg_pot:.3e}", flush = True)
    #     print(f"    avg |F_tension| = {avg_tension:.3e}", flush = True)

    # zero out floor forces
    force[floor_indices] = 0.0

    # hydrodynamic coupling
    hignn_model.dot(vel, force)

    vel = vel + velocity_update_without_2body(position[:, 0:3], force, device)

    # prevent floor movement
    vel[floor_indices] = 0.0

    return vel

# def velocity_update(t, position):
#     if rank == 0:
#         print(f"t = {t:.4f}")

#     # 1) Get the potential‐force on every particle
#     pot_force = potential_force.get_potential_force(position[:, :3])    # shape (N,3)

#     # 2) Build your total force array, starting with pot_force
#     force = pot_force.copy()
#     # add gravity if you still want it
#     force[:, 1] += 1.0

#     # 3) Compute chain tension *by itself* on just the filament beads
#     sel = slice(particle_start, particle_start + n_chain)
#     pos_chain = position[sel]
#     tension_force = chain_tension(pos_chain, k_t, rest_length)   # shape (n_chain,3)

#     # 4) Compute the average magnitudes
#     avg_pot     = np.linalg.norm(pot_force,     axis=1).mean()
#     avg_tension = np.linalg.norm(tension_force, axis=1).mean()

#     # 5) Print them (only on rank 0)
#     if rank == 0:
#         print(f"    avg |F_pot|     = {avg_pot:.3e}")
#         print(f"    avg |F_tension| = {avg_tension:.3e}")

#     # 6) Now add your filament forces into the global array
#     force[sel] += chain_bending(pos_chain, k_b)
#     force[sel] += tension_force

#     # 7) Zero out floor
#     force[floor_indices] = 0.0

#     # … rest of your code unchanged:
#     vel = np.zeros_like(position, dtype=np.float32)
#     hignn_model.dot(vel, force)
#     vel = vel + velocity_update_without_2body(position[:, :3], force, device)
#     vel[floor_indices] = 0.0

#     return vel

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # --- Simulation parameters ---
    n_chain = 2
    k_t = 100.0
    k_b = 30.0
    rest_length = 2.6
    min_dist = 2.02

    # load original hignn model for 3-body and 2-body self calculation 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    original_hignn = torch.load('python/Saved_Model/Unbounded_try1/HIGNN.pkl', weights_only = False).to(device)
    print(original_hignn)

    # --- Build  grid ---
    floor_nx = 50
    floor_ny = 50

    # --- Reectangular Grid Floor ---
    # x_vals = np.linspace(0, (floor_nx - 1) * min_dist, floor_nx)
    # y_vals = np.linspace(0, (floor_ny - 1) * min_dist, floor_ny)
    # xx, yy = np.meshgrid(x_vals, y_vals)
    # floor_pts = np.vstack((xx.ravel(), yy.ravel(), np.zeros(xx.size))).T.astype(np.float32)
    # floor_indices = np.arange(floor_pts.shape[0], dtype=np.int32)

    # --- Hex Grid Floor ---
    # dx = min_dist         # nearest‐neighbor spacing
    # dy = dx * np.sqrt(3)/2  # vertical spacing for hex packing

    # coords = []
    # for row in range(floor_ny):
    #     y = row * dy
    #     # offset every other row by half‐dx
    #     x_offset = 0.5 * dx if (row % 2) else 0.0
    #     for col in range(floor_nx):
    #         x = col * dx + x_offset
    #         coords.append((x, y, 0.0))
    # floor_pts = np.array(coords, dtype=np.float32)
    # floor_indices = np.arange(floor_pts.shape[0], dtype=np.int32)

    # --- Circular Grid Floor ---
    # floor_n = 112
    # # grid spacing (so that we end up with 200 points across the diameter):
    # dx = min_dist  # keep your same min_dist
    # diameter = (floor_n - 1) * dx
    # R = diameter / 2.0

    # # center of the disk
    # cx = R
    # cy = R

    # # create regular 1D arrays from 0→diameter
    # x_vals = np.linspace(0, diameter, floor_n, dtype=np.float32)
    # y_vals = np.linspace(0, diameter, floor_n, dtype=np.float32)

    # # build full mesh and then mask
    # xx, yy = np.meshgrid(x_vals, y_vals)
    # mask = (xx - cx)**2 + (yy - cy)**2 <= R**2

    # # pick only the points inside the disk
    # floor_pts = np.vstack((
    #     xx[mask].ravel(),
    #     yy[mask].ravel(),
    #     np.zeros(mask.sum(), dtype=np.float32)
    # )).T.astype(np.float32)

    # # record their indices so we can freeze them later
    # floor_indices = np.arange(floor_pts.shape[0], dtype=np.int32)

    # --- Circular hex floor ---
        # --- Hex-packed floor clipped to a circle of radius R ---
    # want ~200 pts across the diameter → use floor_n×floor_n grid spacing
    floor_n = 112
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

    # --- Filament ---
    # Compute the total length of the filament
    filament_length = (n_chain - 1) * rest_length

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

    # Compute floor center in x
    # floor_x_center = 0.5 * (floor_nx - 1) * min_dist

    # # Position filament centered in x
    # x_chain = np.linspace(-0.5 * filament_length, 0.5 * filament_length, n_chain) + floor_x_center

    # # For variety, you could also center in y if you wish:
    # # floor_y_center = 0.5 * (floor_ny - 1) * min_dist
    # y_chain = np.full(n_chain, floor_y_center)

    # # z remains at min_dist
    # z_chain = np.full(n_chain, min_dist)

    # desired spacing
    separation = 2.5
    offset = separation/2.0    # = 1.25

    # y-positions of the two filaments
    y_centers = [floor_y_center - offset,
                floor_y_center + offset]

    # build both filaments
    all_positions = []
    for y0 in y_centers:
        # x positions still span the filament
        x_chain = np.linspace(-0.5*filament_length,
                            0.5*filament_length,
                            n_chain) + floor_x_center
        # all beads at y = y0, z = min_dist
        y_chain = np.full(n_chain, y0)
        z_chain = np.full(n_chain, min_dist)

        # stack into (n_chain × 3) array
        pos = np.vstack((x_chain, y_chain, z_chain)).T
        all_positions.append(pos)

    # final (2*n_chain × 3) array
    filament_pts = np.vstack(all_positions).astype(np.float32)

    # filament_pts = np.vstack((x_chain, y_chain, z_chain)).T.astype(np.float32)

    particle_start = floor_pts.shape[0]

    # --- Combine particles ---
    X = np.vstack((floor_pts, filament_pts))
    NN = X.shape[0]

    # --- Initialize HIGNN ---
    hignn.Init()

    hignn_model = hignn.HignnModel(X, 15)
    
    hignn_model.load_two_body_model('nn/two_body_unbounded')

    x_min = np.min(X[:, 0]) - 5
    x_max = np.max(X[:, 0]) + 5
    y_min = np.min(X[:, 1]) - 5
    y_max = np.max(X[:, 1]) + 5
    z_min = np.min(X[:, 2]) - 5
    z_max = np.max(X[:, 2]) + 5

    potential_force = hignn.PotentialForce()
    potential_force.set_two_body_epsilon(7.5)
    
    domain = np.array([[x_min, y_min, z_min],[x_max, y_max, z_max]], dtype=np.float32)
    potential_force.set_domain(domain)
    
    # # set parameters for far dot, the following parameters are default values
    # hignn_model.set_epsilon(0.01)
    # hignn_model.set_max_iter(50)
    # hignn_model.set_mat_pool_size_factor(200)
    # hignn_model.set_post_check_flag(False)
    # hignn_model.set_use_symmetry_flag(False)
    # hignn_model.set_max_far_dot_work_node_size(10000)
    # hignn_model.set_max_relative_coord(100000)

    hignn_model.set_epsilon(0.1)
    hignn_model.set_max_iter(15)
    hignn_model.set_mat_pool_size_factor(30)
    hignn_model.set_post_check_flag(False)
    hignn_model.set_use_symmetry_flag(True)
    hignn_model.set_max_far_dot_work_node_size(10000)
    hignn_model.set_max_relative_coord(1000000)

    rank_range = np.linspace(0, NN, comm.Get_size() + 1, dtype=np.int32)
    
    # for i in range(10):
    #     if rank == 0:
    #         print(i)
    #     v = velocity_update(0, X)
    ts = 0
    dt = 0.01
    ite = 0

    chain_center = np.array([floor_x_center, floor_y_center, min_dist], dtype=np.float32)

    # create a list to hold the max distance at each step
    farthest_points = []
    
    t1 = time.time()

    os.system('clear')

    prev_speed3 = None      # will hold the previous step’s speed
    tol = 1e-5              # convergence tolerance on speed change
    history_len = 5
    speed_history = [] 
    terminal_reached = False

    for i in range(5001):
        if i % 1 == 0:
            with h5py.File('Result/pos'+str(int(i/1))+'rank'+str(rank)+'.h5', 'w') as f:
                f.create_dataset('pos', data=X[rank_range[rank]:rank_range[rank+1], :])
        
        tt1 = time.time()
        V = velocity_update(ts, X)
        if rank == 0:
            print("Time for velocity_update: {t:.4f}s".format(t = time.time() - tt1))

        if i % 1 == 0:
            with h5py.File('Result/vel'+str(int(i/1))+'rank'+str(rank)+'.h5', 'w') as f:
                f.create_dataset('vel', data=V[rank_range[rank]:rank_range[rank+1], :])
        
        X = X + dt * V
        ts = ts + dt

        # ---------- after X = X + dt*V and ts update ----------

        # 1) compute filament centroid (just beads, not floor)
        sel = slice(particle_start, particle_start + n_chain)
        fil_cent = X[sel, :2].mean(axis=0)   # shape (2,) ← (x,y) of filament

        # 2) compute how far floor needs to shift
        #    floor_center_orig is a global you set once when you build floor_pts
        delta = fil_cent - floor_center_orig[:2]   # (dx,dy)

        # 3) shift ONLY the floor points
        fidx = floor_indices
        X[fidx, 0] += delta[0]
        X[fidx, 1] += delta[1]

        # (optionally update floor_center_orig so you can keep re‐using it
        #  or just leave it constant if you always recalc against the true orig.)
        floor_center_orig[:2] += delta

        # --- record the farthest bead in the chain from the center ---
        # select the chain beads
        sel = slice(particle_start, particle_start + n_chain)
        chain_positions = X[sel]             # shape (n_chain,3)
        # compute Euclidean distance to the chain_center
        y_offsets = np.abs(chain_positions[:,1] - chain_center[1])
        farthest_points.append(y_offsets.max())

        # check the third particle (index 2)
        if rank == 0:
            vel3 = (V[particle_start,1]+V[particle_start+1,1])/2
            speed3 = np.linalg.norm(vel3)
            print(f"Frame {i}: speed of 3rd particle = {speed3:.6f}")

            # append and trim history
            speed_history.append(speed3)
            if len(speed_history) > history_len + 1:
                speed_history.pop(0)

            # once we have enough history, check all last five changes
            if len(speed_history) == history_len + 1:
                # compute consecutive differences
                diffs = [abs(speed_history[j+1] - speed_history[j])
                        for j in range(history_len)]
                if all(d < tol for d in diffs):
                    print(f"→ Converged over last {history_len} steps (Δspeed < {tol}).")
                    print(f"Terminal velocity of 3rd particle: {speed3:.6f}")
                    drag = 1 / (6 * np.pi * speed3)
                    print(f"Corresponding drag coefficient: {drag:.6f}")
                    terminal_reached = True
        
        ite = ite + 1
        if rank == 0:
            print()

        # if converged, break out of the loop
        if terminal_reached:
            break         

    if rank == 0:
        
        for step, r in enumerate(farthest_points):
            print(f"Step {step:4d}: farthest chain bead distance = {r:.4f}")
        
        print("Time for simulation: {t:.4f}s".format(t = time.time() - t1))
        print("Time step: {t:.4f}s".format(t = ts))
    
    del hignn_model

    # Finalize
    hignn.Finalize()
