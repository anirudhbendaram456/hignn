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
from scipy.spatial import cKDTree
# from HIGNN.model_structure import HIGNN_model
import HIGNN.model_structure as ms

# Bring in the real class…
HIGNN_model = ms.HIGNN_model

# …and alias it to the typo’d name in __main__
HIGNN_mdoel = HIGNN_model

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

    
    # update coordinates
    hignn_model.update_coord(position[:, 0:3])
    
    # update force/potential
    force = np.zeros((position.shape[0], 3), dtype=np.float32)
    force[:, 0] = 1.0
    
    # create velocity array
    velocity = np.zeros((position.shape[0], 3), dtype=np.float32)
    
    hignn_model.dot(velocity, force)
    # print(velocity)
    velocity = velocity #+ velocity_update_without_2body(position[:, 0:3], force, device)
    # print(velocity)
    return velocity


# def compute_visibility_mask_all(positions, epsilon=1e-3):
#     """
#     Visibility over *all* pairs: (i,j) is visible if no k lies within
#     epsilon of the line segment i→j (strictly between i and j).
#     O(N^3), but matches 'no radius' semantics.
#     """
#     P = positions  # (N,3)
#     N = P.shape[0]
#     visible = set()

#     for i in range(N):
#         pi = P[i]
#         for j in range(N):
#             if i == j:
#                 continue
#             pj = P[j]
#             vec = pj - pi
#             dist = np.linalg.norm(vec)
#             if dist == 0.0:
#                 continue
#             u = vec / dist  # unit direction i→j

#             blocked = False
#             for k in range(N):
#                 if k == i or k == j:
#                     continue
#                 pk = P[k]
#                 rel = pk - pi
#                 t = np.dot(rel, u)  # projection coordinate along segment
#                 if t <= 0.0 or t >= dist:
#                     continue
#                 # closest point on segment and perpendicular distance
#                 closest = pi + t * u
#                 if np.linalg.norm(pk - closest) < epsilon:
#                     blocked = True
#                     break

#             if not blocked:
#                 visible.add((i, j))

#     return visible


# def velocity_update(t, position, epsilon=1e-3):
#     """
#     For each target i:
#       - compute visibility against *all* particles (no radius)
#       - build a force vector with nonzeros only at {i} U {visible sources to i}
#       - apply HI map once and keep v[i]
#     """
#     if rank == 0:
#         print(f"t = {t:.4f}")

#     N = position.shape[0]

#     # Update the mobility with current coordinates
#     hignn_model.update_coord(position[:, :3])

#     # Base external forces (e.g., gravity in Y)
#     base_force = np.zeros((N, 3), dtype=np.float32)
#     base_force[:, 0] = 1.0

#     # Global visibility (no radius)
#     visibility = compute_visibility_mask_all(position, epsilon=epsilon)

#     # Output and reusable buffers
#     velocity  = np.zeros((N, 3), dtype=np.float32)
#     tmp_force = np.zeros((N, 3), dtype=np.float32)
#     tmp_vel   = np.zeros((N, 3), dtype=np.float32)

#     # Per-target screened solve
#     for i in range(N):
#         # sources are exactly those j with (i,j) visible, plus i itself
#         sources_i = [i] + [j for j in range(N) if j != i and (i, j) in visibility]

#         # If all those sources have zero force, skip
#         if np.allclose(base_force[sources_i], 0.0):
#             continue

#         tmp_force.fill(0.0)
#         tmp_force[sources_i] = base_force[sources_i]

#         hignn_model.dot(tmp_vel, tmp_force)
#         tmp_vel = tmp_vel + velocity_update_without_2body(position[:, 0:3], tmp_force, device)
#         velocity[i] = tmp_vel[i]

#     return velocity


if __name__ == '__main__':
    # initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    hignn.Init()
    
    # initial position
    nx = 1
    ny = 1
    nz = 1
    dx = 2.05
    x = np.arange(0, nx * dx, dx)
    y = np.arange(0, ny * dx, dx)
    z = np.arange(0, nz * dx, dx)
    xx, yy, zz = np.meshgrid(x, y, z)
    X = np.concatenate(
        (xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)), axis=1)
    X = X.astype(np.float32)
    NN = X.shape[0]

    # load original hignn model for 3-body and 2-body self calculation 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    original_hignn = torch.load('python/Saved_Model/Unbounded_try1/HIGNN.pkl', weights_only = False).to(device)
    print(original_hignn)

    # initialize hierarchical_hignn_model for 2body interaction
    hignn_model = hignn.HignnModel(X, 1)
    hignn_model.load_two_body_model('nn/two_body_unbounded')
    
    # set parameters for far dot, the following parameters are default values
    hignn_model.set_epsilon(0.1)
    hignn_model.set_max_iter(15)
    hignn_model.set_mat_pool_size_factor(30)
    hignn_model.set_post_check_flag(False)
    hignn_model.set_use_symmetry_flag(True)
    hignn_model.set_max_far_dot_work_node_size(10000)
    hignn_model.set_max_relative_coord(1000000)
    
    
    # setup time integrator
    time_integrator = hignn.ExplicitEuler()
    
    time_integrator.set_time_step(0.005)
    time_integrator.set_final_time(1.0)
    time_integrator.set_num_rigid_body(NN)
    time_integrator.set_output_step(10)
    
    time_integrator.set_velocity_func(velocity_update)
    time_integrator.initialize(X)
    
    # t1 = time.time()
    
    # time_integrator.run()

    rank_range = np.linspace(0, NN, comm.Get_size() + 1, dtype=np.int32)
    
    ts = 0
    dt = 0.01
    ite = 0
    
    t1 = time.time()

    prev_speed3 = None      # will hold the previous step’s speed
    tol = 1e-5              # convergence tolerance on speed change
    history_len = 5
    speed_history = [] 
    terminal_reached = False

    for i in range(501):
        with h5py.File('Result/pos'+str(ite)+'rank'+str(rank)+'.h5', 'w') as f:
            f.create_dataset('pos', data=X[rank_range[rank]:rank_range[rank+1], :])
        
        if rank == 0:
            print("Time step: {t:.4f}s".format(t = i*dt))

        tt1 = time.time()
        V = velocity_update(ts, X)
        if rank == 0:
            print("Time for velocity_update: {t:.4f}s".format(t = time.time() - tt1))

        with h5py.File('Result/vel'+str(ite)+'rank'+str(rank)+'.h5', 'w') as f:
            f.create_dataset('vel', data=V[rank_range[rank]:rank_range[rank+1], :])
        
        X = X + dt * V
        ts = ts + dt

        # check the third particle (index 2)
        if rank == 0:
            vel3 = V[2]
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
                    # terminal_reached = True
        
        ite = ite + 1
        if rank == 0:
            print()

        # if converged, break out of the loop
        if terminal_reached:
            break

    if rank == 0:
        print("Time for simulation: {t:.4f}s".format(t = time.time() - t1))

        # 3rd particle is index 2
        vel3 = V[2]
        speed3 = np.linalg.norm(vel3)

        dist = X[0,2]
        print(f"Final distance travelled by the 1st particle: {dist}")
        print(f"Final velocity of the 3rd particle: {speed3}")
        drag = 1/(6*np.pi*speed3)
        print(f"Final Drag coefficient on the 3rd particle: {drag}")
    
    del hignn_model
    del time_integrator

    hignn.Finalize()