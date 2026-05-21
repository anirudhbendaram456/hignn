import os
import hignn
import numpy as np
import sys
from mpi4py import MPI
import time
import h5py
from scipy.spatial import cKDTree

os.system('clear')

# Placeholder for static floor particle indices
floor_indices = None

def compute_visibility_mask_local(positions, neighbors_list, epsilon=1e-3):
    """
    Sparse visibility: returns a set of (i,j) pairs where j is visible to i
    (i.e., j's effect on i is NOT occluded).
    Only considers j in neighbors_list[i] to limit cost.
    """
    N = positions.shape[0]
    visible_pairs = set()
    for i in range(N):
        pi = positions[i]
        for j in neighbors_list[i]:
            if i == j:
                continue
            pj = positions[j]
            vec_ij = pj - pi
            dist_ij = np.linalg.norm(vec_ij)
            if dist_ij == 0.0:
                continue
            u = vec_ij / dist_ij

            # Candidate blockers: intersection of neighbor lists of i and j (approximate)
            candidates = set(neighbors_list[i]) & set(neighbors_list[j])
            blocked = False
            for k in candidates:
                if k == i or k == j:
                    continue
                pk = positions[k]
                rel = pk - pi
                t = np.dot(rel, u)
                if t <= 0 or t >= dist_ij:
                    continue
                closest = pi + t * u
                if np.linalg.norm(pk - closest) < epsilon:
                    blocked = True
                    break
            if not blocked:
                visible_pairs.add((i, j))
    return visible_pairs  # set of tuples

def velocity_update(t, position):
    if rank == 0:
        print(f"t = {t:.4f}")

    N = position.shape[0]

    # Update HIGNN with current coordinates
    hignn_model.update_coord(position[:, :3])

    # Base external force (gravity in Y)
    base_force = np.zeros((N, 3), dtype=np.float32)
    base_force[:, 1] = 1.0
    if floor_indices is not None:
        base_force[floor_indices] = 0.0

    # Build spatial tree once
    tree = cKDTree(position)

    # Visibility neighborhood radius (for occlusion checking)
    R_vis = 5.0  # tune to physical interaction distance
    neighbors_list = [tree.query_ball_point(position[i], R_vis) for i in range(N)]
    visibility = compute_visibility_mask_local(position, neighbors_list)  # sparse set

    # Interaction radius for accumulating contributions
    R_interact = 10.0  # adjust if needed
    interaction_neighbors = [tree.query_ball_point(position[j], R_interact) for j in range(N)]

    # Prepare accumulation
    velocity = np.zeros((N, 3), dtype=np.float32)
    temp_force = np.zeros((N, 3), dtype=np.float32)
    temp_velocity = np.zeros((N, 3), dtype=np.float32)

    for j in range(N):
        # skip if source has no forcing (optional; here base_force[j] might be zero for floor)
        if np.allclose(base_force[j], 0.0):
            continue

        # single-source force vector
        temp_force.fill(0.0)
        temp_force[j] = base_force[j]

        # get the response from HIGNN
        hignn_model.dot(temp_velocity, temp_force)

        # accumulate only to targets i that are both within interaction radius and visible
        for i in interaction_neighbors[j]:
            if (i, j) in visibility:
                velocity[i] += temp_velocity[i]

    # Enforce immobile floor
    if floor_indices is not None:
        velocity[floor_indices] = 0.0

    return velocity

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    hignn.Init()

    # --- Generate a tightly-packed floor in the XY plane ---
    min_dist = 2.02   # minimal spacing between floor particles
    floor_nx = 50    # number of particles along X
    floor_ny = 50    # number of particles along Y
    floor_z = 0.0    # Z-coordinate of the floor

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

    # --- Generate dynamic particles randomly above the floor with minimum spacing and centered ---
    dynamic_n = 20
    dynamic_spacing = min_dist  # required spacing between dynamic particles
    z_dyn = min_dist                 # height above the floor

    # Bounds for sampling
    x_min, x_max = floor_x_center - 5, floor_x_center + 5
    y_min, y_max = floor_y_center - 5, floor_y_center + 5

    # Poisson-disk-like sampling via rejection
    np.random.seed(12345)
    dynamic_pts_list = []
    while len(dynamic_pts_list) < dynamic_n:
        cand_xy = np.array([np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max)], dtype=np.float32)
        # ensure min spacing
        if all(np.linalg.norm(cand_xy - existing[:2]) >= dynamic_spacing for existing in dynamic_pts_list):
            dynamic_pts_list.append(np.hstack((cand_xy, z_dyn)))
    dynamic_pts = np.array(dynamic_pts_list, dtype=np.float32)

    # Shift so centroid aligns with floor center
    centroid = np.mean(dynamic_pts[:, :2], axis=0)
    dynamic_pts[:, 0] += floor_center_orig[0] - centroid[0]
    dynamic_pts[:, 1] += floor_center_orig[1] - centroid[1]


    # Combine floor and dynamic particles
    X = np.vstack((floor_pts, dynamic_pts))

    # Record which indices correspond to the static floor
    floor_indices = np.arange(floor_pts.shape[0], dtype=np.int32)

    # Initialize HIGNN
    NN = X.shape[0]
    hignn_model = hignn.HignnModel(X, 50)
    hignn_model.load_two_body_model('nn/two_body_unbounded')

    hignn_model.set_epsilon(0.01)
    hignn_model.set_max_iter(50)
    hignn_model.set_mat_pool_size_factor(200)
    hignn_model.set_post_check_flag(False)
    hignn_model.set_use_symmetry_flag(False)
    hignn_model.set_max_far_dot_work_node_size(10000)
    hignn_model.set_max_relative_coord(100000)

    rank_range = np.linspace(0, NN, comm.Get_size() + 1, dtype=np.int32)
    
    ts = 0
    dt = 0.01
    ite = 0

    particle_start = floor_pts.shape[0]

    chain_center = np.array([floor_x_center, floor_y_center, min_dist], dtype=np.float32)

    # create a list to hold the max distance at each step
    farthest_points = []
    
    t1 = time.time()

    for i in range(5001):
        with h5py.File('Result/pos'+str(ite)+'rank'+str(rank)+'.h5', 'w') as f:
            f.create_dataset('pos', data=X[rank_range[rank]:rank_range[rank+1], :])
        
        tt1 = time.time()
        V = velocity_update(ts, X)
        if rank == 0:
            print("Time for velocity_update: {t:.4f}s".format(t = time.time() - tt1))

        with h5py.File('Result/vel'+str(ite)+'rank'+str(rank)+'.h5', 'w') as f:
            f.create_dataset('vel', data=V[rank_range[rank]:rank_range[rank+1], :])
        
        X = X + dt * V
        ts = ts + dt

        # ---------- after X = X + dt*V and ts update ----------

        # 1) compute filament centroid (just beads, not floor)
        sel = slice(particle_start, particle_start + dynamic_n)
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
        sel = slice(particle_start, particle_start + dynamic_n)
        chain_positions = X[sel]             # shape (n_chain,3)
        # compute Euclidean distance to the chain_center
        y_offsets = np.abs(chain_positions[:,1] - chain_center[1])
        farthest_points.append(y_offsets.max())
        
        ite = ite + 1
        if rank == 0:
            print()            

    if rank == 0:
        print("Time for simulation: {t:.4f}s".format(t = time.time() - t1))
        for step, r in enumerate(farthest_points):
            print(f"Step {step:4d}: farthest chain bead distance = {r:.4f}")
    
    del hignn_model

    # Finalize
    hignn.Finalize()
