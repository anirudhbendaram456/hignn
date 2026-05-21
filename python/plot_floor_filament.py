# import os
# import numpy as np
# import h5py
# import matplotlib.pyplot as plt
# from matplotlib.collections import PatchCollection

# # Number of particles per filament (chain length)
# n_chain = 3
# n_filaments = 1 

# CONNECT_FILAMENTS = False   # <-- set True to connect as filaments, False = just particles

# # Determine the folder in which this script lives
# script_dir = os.path.dirname(os.path.abspath(__file__))

# # We want to save into:
# #   /home/bendaram/Projects/hignn_internal/Result
# # Given script_dir = .../hignn-internal/python,
# #  └─ parent    = .../hignn-internal
# #     └─ parent  = .../hignn_internal
# # So we go two levels up, then “Result”
# result_dir = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir, 'Result'))

# # Ensure the Result folder exists
# os.makedirs(result_dir, exist_ok=True)

# # ============================================================
# # FIXED AXES SETTINGS
# # ============================================================
# PARTICLE_RADIUS = 1.0
# AX_PAD = 1.5   # extra padding beyond particle radius

# def square_limits(vmin, vmax, wmin, wmax, pad=0.0):
#     """
#     Build square limits for a 2D view so aspect='equal' does not
#     change the visible zoom from frame to frame.
#     """
#     vc = 0.5 * (vmin + vmax)
#     wc = 0.5 * (wmin + wmax)
#     half = 0.5 * max(vmax - vmin, wmax - wmin) + pad
#     return (vc - half, vc + half), (wc - half, wc + half)

# # Pass 1: find all existing frames and compute GLOBAL limits
# frame_ids = []
# xmin = ymin = zmin = np.inf
# xmax = ymax = zmax = -np.inf

# for N in range(1500):
#     pos_file = f"Result/pos{N}rank0.h5"
#     vel_file = f"Result/vel{N}rank0.h5"
#     if not (os.path.exists(pos_file) and os.path.exists(vel_file)):
#         break

#     with h5py.File(pos_file, 'r') as f_pos:
#         pts = f_pos['pos'][:]

#     frame_ids.append(N)

#     xmin = min(xmin, pts[:, 0].min())
#     xmax = max(xmax, pts[:, 0].max())
#     ymin = min(ymin, pts[:, 1].min())
#     ymax = max(ymax, pts[:, 1].max())
#     zmin = min(zmin, pts[:, 2].min())
#     zmax = max(zmax, pts[:, 2].max())

# if len(frame_ids) == 0:
#     raise RuntimeError("No position/velocity frames found in Result/")

# pad = PARTICLE_RADIUS + AX_PAD

# XZ_XLIM, XZ_YLIM = square_limits(xmin, xmax, zmin, zmax, pad=pad)
# XY_XLIM, XY_YLIM = square_limits(xmin, xmax, ymin, ymax, pad=pad)
# YZ_XLIM, YZ_YLIM = square_limits(ymin, ymax, zmin, zmax, pad=pad)

# print("Using fixed axes:")
# print("  XZ:", XZ_XLIM, XZ_YLIM)
# print("  XY:", XY_XLIM, XY_YLIM)
# print("  YZ:", YZ_XLIM, YZ_YLIM)

# for N in frame_ids:
#     os.system('clear')
#     print(f'N = {N}', end='\r')

#     points_list = []
#     move_forward = True

#     # In your original code, you looped over rank from 0 .. (n_ranks−1).
#     # Here, we assume only rank=0. If you have more ranks, increase the range.
#     for rank in range(1):
#         try:
#             # Read the HDF5 file containing positions
#             with h5py.File(f'Result/pos{N}rank{rank}.h5', 'r') as f_pos:
#                 pts = f_pos['pos'][:]    # shape = (n_particles_this_rank, 3)
#             with h5py.File(f'Result/vel{N}rank0.h5','r') as fvel:
#                 vel = fvel['vel'][:]   # same shape
#         except Exception as e:
#             print(f"\nError reading file for N={N}, rank={rank}: {e}")
#             move_forward = False
#             break

#         points_list.append(pts)

#     if not move_forward:
#         break

#     # extract coords
#     x_all = pts[:, 0]
#     y_all = pts[:, 1]
#     z_all = pts[:, 2]
#     speed = np.linalg.norm(vel, axis=1)

#     total_points = pts.shape[0]

#     # assume filament is the last n_chain points
#     chain_start = total_points - n_filaments * n_chain

#     if N % 1 == 0:
#         # --- x–z view with radius=1 ---
#         fig, ax = plt.subplots(figsize=(6,6))

#         # 1) draw all particles (floor + filaments) as speed‐colored circles
#         patches = [
#             plt.Circle((x_all[i], z_all[i]), radius=1.0)
#             for i in range(total_points)
#         ]
#         coll = PatchCollection(
#             patches,
#             array=speed,
#             cmap='viridis',
#             edgecolors='none',
#             alpha=0.8,
#             zorder=1
#         )
#         ax.add_collection(coll)

#         # reuse the same normalization so filament speeds map on the global color scale
#         norm = coll.norm

#         # 2) draw filament beads on top, with shared norm
#         fil_patches, fil_speeds = [], []
#         for j in range(n_filaments):
#             i0 = chain_start + j*n_chain
#             i1 = i0 + n_chain
#             for idx in range(i0, i1):
#                 fil_patches.append(plt.Circle((x_all[idx], z_all[idx]), radius=1.0))
#                 fil_speeds.append(speed[idx])

#         fil_coll = PatchCollection(
#             fil_patches,
#             array=np.array(fil_speeds),
#             cmap='viridis',
#             norm=norm,          # ← use the same Normalize as coll
#             edgecolors='none',
#             alpha=1.0,
#             zorder=5
#         )
#         ax.add_collection(fil_coll)

#         # 3) connect each filament
#         # 3) either connect each filament OR just plot filament beads (no connections)
#         if CONNECT_FILAMENTS:
#             colors = ['C1','C2','C3','C4']
#             for j in range(n_filaments):
#                 i0 = chain_start + j*n_chain
#                 i1 = i0 + n_chain
#                 xj, zj = x_all[i0:i1], z_all[i0:i1]
#                 ax.plot(
#                     xj, zj, '-',
#                     color=colors[j % len(colors)],
#                     lw=1.5,
#                     zorder=6,
#                     label=f'filament {j+1}'
#                 )
#         else:
#             # no connections; beads already drawn via fil_coll, so just add a legend handle
#             ax.scatter([], [], s=30, color='orange', label='filament beads (no links)')


#         # # 4) overlay velocity‐vectors on top
#         # ax.quiver(
#         #     x_all, z_all,
#         #     vel[:,0], vel[:,2],
#         #     angles='xy',
#         #     scale_units='xy',
#         #     scale=0.1,
#         #     width=0.005,
#         #     color='k',
#         #     alpha=0.8,
#         #     zorder=10,
#         #     label='velocity'
#         # )

#         # ax.legend(loc='upper right')

#         # 5) finish formatting
#         ax.set_xlim(*XZ_XLIM)
#         ax.set_ylim(*XZ_YLIM)
#         ax.set_aspect('equal', adjustable='box')
#         # cbar = plt.colorbar(coll, ax=ax, label='speed')
#         ax.set_xlabel('x')
#         ax.set_ylabel('z')
#         ax.set_title(f'Frame {N}: speed & velocity vectors (xz)')

#         plt.tight_layout()
#         out_path = os.path.join(
#             result_dir,
#             f'03_22_{n_chain}_xz_speed_vectors_{N//50:03d}.png'
#         )
#         plt.savefig(out_path, dpi=150)
#         plt.show()
#         plt.close()

#         # prepare for next figure
#         plt.figure(figsize=(6, 6))

#         # --- x–y view with velocity‐colored particles and velocity vectors ---
#         # --- x–y view with shared colormap and velocity vectors ---
#         fig, ax = plt.subplots(figsize=(6,6))

#         # # 1) draw all particles as speed‐colored circles
#         patches = [plt.Circle((x_all[i], y_all[i]), radius=1.0)
#                 for i in range(total_points)]
#         coll = PatchCollection(
#             patches,
#             facecolor='orange',
#             # array=speed,
#             # cmap='viridis',
#             edgecolors='none',
#             alpha=0.8,
#             zorder=1
#         )
#         ax.add_collection(coll)

#         # # reuse the same Normalize for filament overlay
#         norm = coll.norm

#         # # 2) overlay filament beads on top, colored by speed
#         fil_patches, fil_speeds = [], []
#         for j in range(n_filaments):
#             i0 = chain_start + j * n_chain
#             i1 = i0 + n_chain
#             for idx in range(i0, i1):
#                 fil_patches.append(
#                     plt.Circle((x_all[idx], y_all[idx]), radius=1.0)
#                 )
#                 fil_speeds.append(speed[idx])
#         fil_coll = PatchCollection(
#             fil_patches,
#             facecolor='orange',
#             # array=np.array(fil_speeds),
#             # cmap='viridis',
#             # norm=norm,
#             edgecolors='none',
#             alpha=1.0,
#             zorder=5
#         )
#         ax.add_collection(fil_coll)

#         # 3) connect each filament with lines
#         if CONNECT_FILAMENTS:
#             colors = ['C1','C2','C3','C4']
#             for j in range(n_filaments):
#                 i0 = chain_start + j*n_chain
#                 i1 = i0 + n_chain
#                 xj = x_all[i0:i1]
#                 yj = y_all[i0:i1]
#                 ax.plot(
#                     xj, yj, '-',
#                     color=colors[j % len(colors)],
#                     lw=1.5,
#                     zorder=6,
#                     label=f'filament {j+1}'
#                 )
#         else:
#             ax.scatter([], [], s=30, color='orange', label='filament beads (no links)')


#         # # 4) overlay velocity vectors on top
#         # ax.quiver(
#         #     x_all, y_all,
#         #     vel[:,0], vel[:,1],
#         #     angles='xy',
#         #     scale_units='xy',
#         #     scale=0.1,
#         #     width=0.005,
#         #     color='k',
#         #     alpha=0.8,
#         #     zorder=10,
#         #     label='velocity'
#         # )

#         # ax.legend(loc='upper right')

#         # 5) finalize
#         ax.set_xlim(*XY_XLIM)
#         ax.set_ylim(*XY_YLIM)
#         ax.set_aspect('equal', adjustable='box')
#         # cbar = plt.colorbar(coll, ax=ax, label='speed')
#         ax.set_xlabel('x')
#         ax.set_ylabel('y')
#         ax.set_title(f'Frame {N}: speed & velocity vectors (xy)')

#         plt.tight_layout()
#         out_path = os.path.join(
#             result_dir,
#             f'03_30_{n_chain}_xy_speed_vectors_{N:03d}.png'
#         )
#         plt.savefig(out_path, dpi=150)
#         plt.show()
#         plt.close()

#         # prepare for next figure
#         plt.figure(figsize=(6, 6))

#         # --- y–z view with shared colormap and velocity vectors ---
#         fig, ax = plt.subplots(figsize=(6,6))

#         # # 1) draw all particles as speed‐colored circles
#         # patches = [plt.Circle((y_all[i], z_all[i]), radius=1.0)
#         #         for i in range(total_points)]
#         # coll = PatchCollection(
#         #     patches,
#         #     array=speed,
#         #     cmap='viridis',
#         #     edgecolors='none',
#         #     alpha=0.8,
#         #     zorder=1
#         # )
#         # ax.add_collection(coll)

#         # # reuse normalization
#         # norm = coll.norm

#         # # 2) overlay filament beads on top, colored by speed
#         # fil_patches, fil_speeds = [], []
#         # for j in range(n_filaments):
#         #     i0 = chain_start + j * n_chain
#         #     i1 = i0 + n_chain
#         #     for idx in range(i0, i1):
#         #         fil_patches.append(
#         #             plt.Circle((y_all[idx], z_all[idx]), radius=1.0)
#         #         )
#         #         fil_speeds.append(speed[idx])
#         # fil_coll = PatchCollection(
#         #     fil_patches,
#         #     array=np.array(fil_speeds),
#         #     cmap='viridis',
#         #     norm=norm,
#         #     edgecolors='none',
#         #     alpha=1.0,
#         #     zorder=5
#         # )
#         # ax.add_collection(fil_coll)

#         # 3) connect each filament with lines
#         if CONNECT_FILAMENTS:
#             colors = ['C1','C2','C3','C4']
#             for j in range(n_filaments):
#                 i0 = chain_start + j*n_chain
#                 i1 = i0 + n_chain
#                 yj = y_all[i0:i1]
#                 zj = z_all[i0:i1]
#                 ax.plot(
#                     yj, zj, '-',
#                     color=colors[j % len(colors)],
#                     lw=1.5,
#                     zorder=6,
#                     label=f'filament {j+1}'
#                 )
#         else:
#             ax.scatter([], [], s=30, color='orange', label='filament beads (no links)')


#         # # 4) overlay velocity vectors on top
#         # ax.quiver(
#         #     y_all, z_all,
#         #     vel[:,1], vel[:,2],
#         #     angles='xy',
#         #     scale_units='xy',
#         #     scale=0.1,
#         #     width=0.005,
#         #     color='k',
#         #     alpha=0.8,
#         #     zorder=10,
#         #     label='velocity'
#         # )

#         # ax.legend(loc='upper right')

#         # 5) finalize
#         ax.set_xlim(*YZ_XLIM)
#         ax.set_ylim(*YZ_YLIM)
#         ax.set_aspect('equal', adjustable='box')
#         # cbar = plt.colorbar(coll, ax=ax, label='speed')
#         ax.set_xlabel('y')
#         ax.set_ylabel('z')
#         ax.set_title(f'Frame {N}: speed & velocity vectors (yz)')

#         plt.tight_layout()
#         out_path = os.path.join(
#             result_dir,
#             f'03_22_{n_chain}_yz_speed_vectors_{N//50:03d}.png'
#         )
#         plt.savefig(out_path, dpi=150)
#         plt.show()
#         plt.close()

#         # prepare for next figure
#         plt.figure(figsize=(6, 6))

#         #############################

#         # plot floor points lightly
#         plt.scatter(x_all[:chain_start], z_all[:chain_start],
#                     s=9, color='lightgray', alpha=0.4,
#                     label='floor particles')
        

#         # plot filament
#         x_chain = x_all[chain_start:]
#         z_chain = z_all[chain_start:]
#         # plt.plot(x_chain, z_chain, '-o',
#         #          markersize=4, linewidth=1.5,
#         #          color='C1', label='filament')
#         plt.scatter(x_chain, z_chain, s=9, color='orange', alpha=0.4)


#         plt.xlabel('x')
#         plt.ylabel('z')
#         plt.title(f'Frame {N}: particles over floor')
#         plt.axis('equal')
#         plt.legend(loc='best')
#         plt.tight_layout()

#         # save frame
#         out_path = os.path.join(result_dir, f'03_22_{n_chain}_floor_filament_{int(N/50):03d}.png')
#         plt.savefig(out_path, dpi=150)

#         # display
#         plt.show()
#         plt.close()

#         plt.figure(figsize=(6, 6))

#         # plot floor points lightly
#         plt.scatter(x_all[:chain_start], y_all[:chain_start],
#                     s=9, color='lightgray', alpha=0.4,
#                     label='floor particles')
        

#         # plot filament
#         x_chain = x_all[chain_start:]
#         z_chain = y_all[chain_start:]
#         # plt.plot(x_chain, z_chain, '-o',
#         #          markersize=4, linewidth=1.5,
#         #          color='C1', label='filament')
#         plt.scatter(x_chain, z_chain, s=9, color='orange', alpha=0.4)

#         plt.xlabel('x')
#         plt.ylabel('y')
#         plt.title(f'Frame {N}: particles over floor')
#         plt.axis('equal')
#         plt.legend(loc='best')
#         plt.tight_layout()

#         # save frame
#         out_path = os.path.join(result_dir, f'03_22_{n_chain}_floor_filament_xy_{int(N/50):03d}.png')
#         plt.savefig(out_path, dpi=150)

#         # display
#         plt.show()
#         plt.close()

#         plt.figure(figsize=(6, 6))

#         # plot floor points lightly
#         plt.scatter(y_all[:chain_start], z_all[:chain_start],
#                     s=9, color='lightgray', alpha=0.4,
#                     label='floor particles')
        

#         # plot filament
#         x_chain = y_all[chain_start:]
#         z_chain = z_all[chain_start:]
#         # plt.plot(x_chain, z_chain, '-o',
#         #          markersize=4, linewidth=1.5,
#         #          color='C1', label='filament')
#         plt.scatter(x_chain, z_chain, s=9, color='orange', alpha=0.4)

#         plt.xlabel('y')
#         plt.ylabel('z')
#         plt.title(f'Frame {N}: particles over floor')
#         plt.axis('equal')
#         plt.legend(loc='best')
#         plt.tight_layout()

#         # save frame
#         out_path = os.path.join(result_dir, f'03_22_{n_chain}_floor_filament_yz_{int(N/50):03d}.png')
#         plt.savefig(out_path, dpi=150)

#         # display
#         plt.show()
#         plt.close()

# # ------------------ AFTER the for-loop finishes ------------------
# import glob
# import imageio.v2 as imageio

# # pattern for the XY frames you already saved
# pattern = os.path.join(result_dir, f"03_30_{n_chain}_xy_speed_vectors_*.png")
# frame_files = sorted(glob.glob(pattern))

# if len(frame_files) == 0:
#     print(f"\nNo frames found for pattern:\n  {pattern}")
# else:
#     print(f"\nFound {len(frame_files)} frames. Building animation...")

#     # --- Try MP4 first (best) ---
#     mp4_out = os.path.join(result_dir, f"03_30_{n_chain}_floor_filament_xy.mp4")
#     fps = 8 # adjust speed (e.g., 5 slower, 15 faster)

#     try:
#         with imageio.get_writer(mp4_out, fps=fps, codec="libx264", quality=8) as writer:
#             for fn in frame_files:
#                 writer.append_data(imageio.imread(fn))
#         print(f"Saved MP4: {mp4_out}")

#     except Exception as e:
#         print(f"MP4 failed ({e}). Falling back to GIF...")

#         # --- Fallback GIF ---
#         gif_out = os.path.join(result_dir, f"03_30_{n_chain}_floor_filament_xy.gif")
#         frames = [imageio.imread(fn) for fn in frame_files]
#         imageio.mimsave(gif_out, frames, fps=fps)
#         print(f"Saved GIF: {gif_out}")

import os
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib import colors, cm
import imageio.v2 as imageio
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist

# ---------------- user settings ----------------
n_chain = 8
n_filaments = 1
CONNECT_FILAMENTS = False

STEPS = 12501

PARTICLE_RADIUS = 30.0          # nm shown in plot
TRAIL_LENGTH = 1500             # only used for XY animation
FPS = 25
TRACK_ONLY_DYNAMIC = True
SHOW_DISTANCE_TEXT = True

# 3D settings
THREE_D_ELEV = 25               # fixed camera elevation
THREE_D_AZIM = 45               # fixed camera azimuth
SPHERE_RES_U = 10               # sphere resolution for 3D (larger = smoother, slower)
SPHERE_RES_V = 10

# ---------------- cluster metrics settings ----------------
SAVE_CLUSTER_METRICS = True
CLUSTER_METRICS_DIR = None  # if None, saved inside result_dir

# If no floor, all particles are dynamic.
# If you have floor particles first and dynamic particles last, use:
# DYNAMIC_START_INDEX = chain_start after it is computed.
DYNAMIC_START_INDEX = 0
DYNAMIC_END_INDEX = None

# Structural order
ORDER_N = 6                 # 6 = hexatic/triangular-like; 4 = square-like
ORDER_CUTOFF_NM = 85.0      # None = infer from first frame
ORDER_CUTOFF_FACTOR = 1.35
LOCAL_ORDER_POWER = 1      # local color-style value = |psi_n(i)|^2

# Use every frame for metrics. Increase to 5, 10, 50 if it becomes slow.
METRIC_FRAME_STRIDE = 1
SAVE_LOCAL_ORDER_CSV = True
# ----------------------------------------------------------

# ---------------- ordering / COM overlay settings ----------------
COLOR_BY_LOCAL_ORDER = True
SHOW_COLORBAR = True

# If TRACK_ONLY_DYNAMIC=True, metrics use only the dynamic particles
# (same particles as in the filament / chain part).
# ---------------------------------------------------------------
# ------------------------------------------------

script_dir = os.path.dirname(os.path.abspath(__file__))
result_dir = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir, 'Result'))
os.makedirs(result_dir, exist_ok=True)

time_file = os.path.join("Result", "time_history_physical_s_rank0.npy")
step_file = os.path.join("Result", "step_history_rank0.npy")

time_hist = None
saved_steps = None

if os.path.exists(time_file):
    time_hist = np.load(time_file)
    print(f"Loaded time history from: {time_file}")
else:
    print("WARNING: time_history_rank0.npy not found. Falling back to frame index.")

if os.path.exists(step_file):
    saved_steps = np.load(step_file)
    print(f"Loaded step history from: {step_file}")
else:
    saved_steps = None

def compute_com(traj_nm):
    return np.mean(traj_nm, axis=1)

def compute_com_speed_from_time(traj_nm, time_hist):
    """
    traj_nm : shape (n_steps, n_particles, 3), in nm
    time_hist : shape (n_steps,), in simulation time units

    Returns
    -------
    com_nm : shape (n_steps, 3)
    com_speed : shape (n_steps,)
        in nm / simulation-time-unit
        If your simulation time is in seconds, this is nm/s.
    """
    com_nm = compute_com(traj_nm)
    speed = np.zeros(traj_nm.shape[0], dtype=np.float64)

    if len(time_hist) != traj_nm.shape[0]:
        raise ValueError(
            f"time_hist length {len(time_hist)} does not match number of frames {traj_nm.shape[0]}"
        )

    if len(time_hist) > 1:
        dt = np.diff(time_hist)
        dcom = np.diff(com_nm, axis=0)

        vel = np.zeros_like(dcom)
        good = dt > 0.0
        vel[good] = dcom[good] / dt[good, None]

        speed[1:] = np.linalg.norm(vel, axis=1)
        speed[0] = speed[1]

    return com_nm, speed

def save_com_speed_csv(out_path, frame_ids, time_hist_used, com_nm, com_speed):
    import pandas as pd

    df = pd.DataFrame({
        "frame": np.asarray(frame_ids[:len(time_hist_used)], dtype=int),
        "time": time_hist_used,
        "com_x_nm": com_nm[:, 0],
        "com_y_nm": com_nm[:, 1],
        "com_z_nm": com_nm[:, 2],
        "com_speed_nm_per_time": com_speed,
    })
    df.to_csv(out_path, index=False)
    print(f"Saved COM speed CSV to: {out_path}")

def save_com_speed_plot(out_path, time_hist_used, com_speed):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(time_hist_used, com_speed, linewidth=2)
    ax.set_xlabel("time")
    ax.set_ylabel("COM speed [nm / time-unit]")
    ax.set_title("Cluster COM speed")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved COM speed plot to: {out_path}")

def load_time_history_if_available(time_file, n_frames):
    """
    Load actual simulation time if available.
    Returns:
        time_used, time_label
    """
    if os.path.exists(time_file):
        t = np.load(time_file).astype(np.float64)

        if len(t) < n_frames:
            print(
                f"WARNING: time history has {len(t)} entries, "
                f"but trajectory has {n_frames} frames. Truncating trajectory metrics."
            )
            return t, "time_s"

        return t[:n_frames], "time_s"

    print("WARNING: time_history_rank0.npy not found. Falling back to frame index.")
    return np.arange(n_frames, dtype=np.float64), "frame"


def get_cluster_traj_nm(all_positions_npy, start_index=0, end_index=None):
    """
    Convert HIGNN sim coordinates to nm and select the dynamic cluster.

    In your postprocessing, positions are converted to nm using *30.0,
    so this keeps the same convention.
    """
    traj_nm_all = all_positions_npy * 30.0
    if end_index is None:
        return traj_nm_all[:, start_index:, :]
    return traj_nm_all[:, start_index:end_index, :]


def compute_com_nm(traj_nm):
    return np.mean(traj_nm, axis=1)


def compute_com_speed_nm_per_time(data_nm, time_axis):
    """
    Accepts either:
      data_nm shape (n_frames, n_particles, 3): full trajectory
      data_nm shape (n_frames, 3): already-computed COM trajectory

    Returns
    -------
    com_nm : shape (n_frames, 3)
    speed : shape (n_frames,)
    """
    data_nm = np.asarray(data_nm, dtype=np.float64)
    time_axis = np.asarray(time_axis, dtype=np.float64).ravel()

    if data_nm.ndim == 3:
        # full particle trajectory
        com_nm = np.mean(data_nm, axis=1)
    elif data_nm.ndim == 2 and data_nm.shape[1] == 3:
        # already COM trajectory
        com_nm = data_nm
    else:
        raise ValueError(
            f"Expected data_nm shape (n_frames,n_particles,3) or (n_frames,3), got {data_nm.shape}"
        )

    n_frames = com_nm.shape[0]

    if len(time_axis) != n_frames:
        raise ValueError(
            f"time_axis length {len(time_axis)} does not match n_frames {n_frames}"
        )

    speed = np.zeros(n_frames, dtype=np.float64)

    if n_frames > 1:
        dt = np.diff(time_axis)
        dcom = np.diff(com_nm, axis=0)

        good = dt > 0.0
        vel = np.zeros_like(dcom)

        vel[good] = dcom[good] / dt[good, None]

        speed[1:] = np.linalg.norm(vel, axis=1)
        speed[0] = speed[1]

    return com_nm, speed


def compute_mean_pair_distance_nm(traj_nm, xy_only=True):
    """
    Exact mean all-pair distance per frame.
    Uses scipy pdist for speed.
    """
    n_steps = traj_nm.shape[0]
    out = np.zeros(n_steps, dtype=np.float64)

    for k in range(n_steps):
        pts = traj_nm[k, :, :2] if xy_only else traj_nm[k, :, :3]
        if pts.shape[0] < 2:
            out[k] = 0.0
        else:
            out[k] = np.mean(pdist(pts))

    return out


def compute_radius_of_gyration_nm(traj_nm, xy_only=True):
    """
    Radius of gyration:
        Rg = sqrt(mean_i |r_i - r_COM|^2)
    Smaller Rg means more compact assembly.
    """
    pts = traj_nm[:, :, :2] if xy_only else traj_nm[:, :, :3]
    com = np.mean(pts, axis=1, keepdims=True)
    rg2 = np.mean(np.sum((pts - com) ** 2, axis=2), axis=1)
    return np.sqrt(rg2)


def infer_neighbor_cutoff_nm(first_frame_xy_nm, factor=1.35):
    """
    cutoff = factor * median nearest-neighbor distance
    """
    n_particles = first_frame_xy_nm.shape[0]
    if n_particles < 2:
        return np.inf

    tree = cKDTree(first_frame_xy_nm)
    dists, _ = tree.query(first_frame_xy_nm, k=2)

    nearest = dists[:, 1]
    return factor * float(np.median(nearest))


# def compute_local_and_global_bond_order(
#     traj_nm,
#     order_n=6,
#     cutoff_nm=None,
#     cutoff_factor=1.35,
#     local_power=2,
# ):
#     """
#     Local order:
#         local_vals[k,i] = |psi_n(i)|^local_power

#     Global order:
#         global_vals[k] = |Psi_n|
#         where Psi_n = mean_i psi_n(i)

#     psi_n(i) = mean_j exp(i*n*theta_ij)
#     """
#     n_steps, n_particles, _ = traj_nm.shape
#     xy = traj_nm[:, :, :2]

#     if cutoff_nm is None:
#         cutoff_used_nm = infer_neighbor_cutoff_nm(xy[0], factor=cutoff_factor)
#     else:
#         cutoff_used_nm = float(cutoff_nm)

#     local_vals = np.zeros((n_steps, n_particles), dtype=np.float64)
#     global_vals = np.zeros(n_steps, dtype=np.float64)

#     for k in range(n_steps):
#         pts = xy[k]
#         tree = cKDTree(pts)
#         neigh = tree.query_ball_point(pts, cutoff_used_nm)

#         psi_i = np.zeros(n_particles, dtype=np.complex128)

#         for i in range(n_particles):
#             vals = []

#             xi, yi = pts[i]

#             for j in neigh[i]:
#                 if j == i:
#                     continue

#                 dx = pts[j, 0] - xi
#                 dy = pts[j, 1] - yi
#                 theta = np.arctan2(dy, dx)
#                 vals.append(np.exp(1j * order_n * theta))

#             if len(vals) > 0:
#                 psi_i[i] = np.mean(vals)
#                 local_vals[k, i] = np.abs(psi_i[i]) ** local_power
#             else:
#                 psi_i[i] = 0.0 + 0.0j
#                 local_vals[k, i] = 0.0

#         global_vals[k] = np.abs(np.mean(psi_i))

#     return local_vals, global_vals, cutoff_used_nm

def compute_local_and_global_bond_order(
    traj_nm,
    order_n=6,
    cutoff_nm=None,
    cutoff_factor=1.35,
    local_power=None,   # kept only so old function calls do not break
):
    """
    Compute local and global bond-orientational order, matching the reference code.

    For each particle j:

        psi_n(j) = (1 / z_j) sum_{k in neighbors(j)} exp(i * n * beta_jk)

    where beta_jk is the angle from particle j to particle k in the xy-plane.

    Differences from the previous version:
      - Uses the same explicit neighbor-loop logic as the reference code.
      - Uses strict r_jk < r_cut, matching the reference code.
      - If a particle has no neighbors, psi_n(j) is NaN.
      - Returns local magnitude |psi_n(j)|, not |psi_n(j)|^2.
      - Global order is |nanmean_i psi_n(i)|, ignoring isolated NaN particles.

    Returns
    -------
    local_vals : ndarray, shape (n_steps, n_particles)
        local_vals[k, i] = |psi_n(i)|

    global_vals : ndarray, shape (n_steps,)
        global_vals[k] = |mean_i psi_n(i)|, ignoring NaN particles

    cutoff_used_nm : float
        Neighbor cutoff used in nm
    """
    traj_nm = np.asarray(traj_nm, dtype=float)
    n_steps, n_particles, _ = traj_nm.shape

    xy = traj_nm[:, :, :2]

    if cutoff_nm is None:
        cutoff_used_nm = infer_neighbor_cutoff_nm(
            xy[0],
            factor=cutoff_factor
        )
    else:
        cutoff_used_nm = float(cutoff_nm)

    local_vals = np.full((n_steps, n_particles), np.nan, dtype=np.float64)
    global_vals = np.full(n_steps, np.nan, dtype=np.float64)

    for k in range(n_steps):
        positions = xy[k]

        psi_i = np.zeros(n_particles, dtype=np.complex128)
        psi_i[:] = np.nan + 1j * np.nan

        for j in range(n_particles):
            neighbors = []

            for m in range(n_particles):
                if m == j:
                    continue

                dx = positions[m, 0] - positions[j, 0]
                dy = positions[m, 1] - positions[j, 1]
                r_jm = np.sqrt(dx**2 + dy**2)

                # strict cutoff, same as your reference code
                if r_jm < cutoff_used_nm:
                    neighbors.append(m)

            z_j = len(neighbors)

            if z_j == 0:
                psi_i[j] = np.nan + 1j * np.nan
                local_vals[k, j] = np.nan
                continue

            total = 0.0 + 0.0j

            for m in neighbors:
                dx = positions[m, 0] - positions[j, 0]
                dy = positions[m, 1] - positions[j, 1]

                beta_jm = np.arctan2(dy, dx)

                total += np.exp(1j * order_n * beta_jm)

            psi_i[j] = total / z_j

            # magnitude value only: |psi_n(j)|
            local_vals[k, j] = np.abs(psi_i[j])

        # Global order: magnitude of complex average, ignoring isolated particles
        valid = np.isfinite(psi_i.real) & np.isfinite(psi_i.imag)

        if np.any(valid):
            global_vals[k] = np.abs(np.mean(psi_i[valid]))
        else:
            global_vals[k] = np.nan

    return local_vals, global_vals, cutoff_used_nm


def save_cluster_metrics_outputs(
    traj_nm,
    frame_ids,
    time_axis,
    time_label,
    out_dir,
    order_n=6,
    cutoff_nm=None,
    cutoff_factor=1.35,
    local_power=2,
    save_local_csv=True,
):
    os.makedirs(out_dir, exist_ok=True)

    com_nm, com_speed = compute_com_speed_nm_per_time(traj_nm, time_axis)
    mean_pair_nm = compute_mean_pair_distance_nm(traj_nm, xy_only=True)
    rg_nm = compute_radius_of_gyration_nm(traj_nm, xy_only=True)

    local_order, global_order, cutoff_used_nm = compute_local_and_global_bond_order(
        traj_nm,
        order_n=order_n,
        cutoff_nm=cutoff_nm,
        cutoff_factor=cutoff_factor,
        local_power=local_power,
    )

    if time_label == "time_s":
        speed_col = "com_speed_nm_per_s"
        speed_ylabel = "COM speed [nm/s]"
    else:
        speed_col = "com_speed_nm_per_frame"
        speed_ylabel = "COM speed [nm/frame]"

    # ---------------- global metrics CSV ----------------
    global_arr = np.column_stack([
        np.asarray(frame_ids, dtype=np.float64),
        time_axis,
        com_nm[:, 0],
        com_nm[:, 1],
        com_nm[:, 2],
        com_speed,
        mean_pair_nm,
        rg_nm,
        global_order,
        np.mean(local_order, axis=1),
    ])

    global_header = (
        f"frame,{time_label},com_x_nm,com_y_nm,com_z_nm,"
        f"{speed_col},mean_pair_distance_nm,radius_of_gyration_nm,"
        f"global_psi_{order_n}_mag,mean_local_psi_{order_n}_power_{local_power}"
    )

    global_csv = os.path.join(out_dir, "cluster_metrics.csv")
    np.savetxt(
        global_csv,
        global_arr,
        delimiter=",",
        header=global_header,
        comments=""
    )

    # ---------------- local order CSV ----------------
    if save_local_csv:
        n_steps, n_particles = local_order.shape

        frames_rep = np.repeat(np.asarray(frame_ids, dtype=np.float64), n_particles)
        time_rep = np.repeat(time_axis, n_particles)
        particle_rep = np.tile(np.arange(n_particles, dtype=np.float64), n_steps)

        local_arr = np.column_stack([
            frames_rep,
            time_rep,
            particle_rep,
            local_order.reshape(-1),
        ])

        local_header = (
            f"frame,{time_label},particle_index,"
            f"local_psi_{order_n}_power_{local_power}"
        )

        local_csv = os.path.join(out_dir, f"local_psi_{order_n}_power_{local_power}.csv")
        np.savetxt(
            local_csv,
            local_arr,
            delimiter=",",
            header=local_header,
            comments=""
        )
        print(f"Saved local structural order CSV to: {local_csv}")

    # ---------------- plots ----------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(time_axis, com_speed, linewidth=2)
    ax.set_xlabel(time_label.replace("_", " "))
    ax.set_ylabel(speed_ylabel)
    ax.set_title("Cluster COM speed")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "com_speed.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(time_axis, mean_pair_nm, linewidth=2)
    ax.set_xlabel(time_label.replace("_", " "))
    ax.set_ylabel("Mean pair distance [nm]")
    ax.set_title("Mean pair distance")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "mean_pair_distance.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(time_axis, rg_nm, linewidth=2)
    ax.set_xlabel(time_label.replace("_", " "))
    ax.set_ylabel("Radius of gyration [nm]")
    ax.set_title("Cluster compactness")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "radius_of_gyration.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(time_axis, global_order, linewidth=2)
    ax.set_xlabel(time_label.replace("_", " "))
    ax.set_ylabel(f"|Psi_{order_n}|")
    ax.set_title(f"Global {order_n}-fold structural order")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"global_psi_{order_n}.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(time_axis, np.mean(local_order, axis=1), linewidth=2)
    ax.set_xlabel(time_label.replace("_", " "))
    ax.set_ylabel(f"mean |psi_{order_n}(i)|^{local_power}")
    ax.set_title(f"Mean local {order_n}-fold structural order")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"mean_local_psi_{order_n}_power_{local_power}.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    # ---------------- summary ----------------
    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Trajectory shape used: {traj_nm.shape}\n")
        f.write(f"Time label: {time_label}\n")
        f.write(f"Bond order n: {order_n}\n")
        f.write(f"Neighbor cutoff used [nm]: {cutoff_used_nm:.6f}\n")
        f.write(f"Initial mean pair distance [nm]: {mean_pair_nm[0]:.6f}\n")
        f.write(f"Final mean pair distance [nm]: {mean_pair_nm[-1]:.6f}\n")
        f.write(f"Initial radius of gyration [nm]: {rg_nm[0]:.6f}\n")
        f.write(f"Final radius of gyration [nm]: {rg_nm[-1]:.6f}\n")
        f.write(f"Initial global |Psi_{order_n}|: {global_order[0]:.6f}\n")
        f.write(f"Final global |Psi_{order_n}|: {global_order[-1]:.6f}\n")
        f.write(f"Initial mean local |psi_{order_n}|^{local_power}: {np.mean(local_order[0]):.6f}\n")
        f.write(f"Final mean local |psi_{order_n}|^{local_power}: {np.mean(local_order[-1]):.6f}\n")

    print(f"Saved cluster metrics CSV to: {global_csv}")
    print(f"Saved cluster metrics plots to: {out_dir}")
    print(f"Structural-order cutoff used = {cutoff_used_nm:.3f} nm")

    return {
        "com_nm": com_nm,
        "com_speed": com_speed,
        "mean_pair_nm": mean_pair_nm,
        "rg_nm": rg_nm,
        "local_order": local_order,
        "global_order": global_order,
        "cutoff_used_nm": cutoff_used_nm,
        "time_axis": time_axis,
        "time_label": time_label,
    }

def load_time_axis(frame_ids, time_file):
    frame_ids_arr = np.asarray(frame_ids, dtype=int)

    if os.path.exists(time_file):
        t_all = np.load(time_file).astype(np.float64).ravel()

        if frame_ids_arr.max() < len(t_all):
            return t_all[frame_ids_arr], "s", "nm/s"

        if len(t_all) >= len(frame_ids_arr):
            return t_all[:len(frame_ids_arr)], "s", "nm/s"

        print(
            f"WARNING: time file has {len(t_all)} entries, "
            f"but there are {len(frame_ids_arr)} frames. Using frame index."
        )

    return frame_ids_arr.astype(np.float64), "frame", "nm/frame"

# def compute_local_and_global_order_xy(points_xy_nm, order_n=6, cutoff_nm=None):
#     """
#     Compute local bond-orientational order psi_n(i) in 2D:
#         psi_n(i) = (1/Nb) sum_j exp(i * n * theta_ij)

#     Returns
#     -------
#     local_scalar : ndarray, shape (N,)
#         |psi_n(i)| in [0,1]
#     global_scalar : float
#         |mean_i psi_n(i)|
#     psi_complex : ndarray, shape (N,), complex
#     """
#     Np = points_xy_nm.shape[0]
#     if Np == 0:
#         return np.array([]), 0.0, np.array([], dtype=np.complex128)

#     if Np == 1:
#         return np.array([0.0]), 0.0, np.array([0.0 + 0.0j])

#     if cutoff_nm is None or not np.isfinite(cutoff_nm):
#         cutoff_nm = infer_neighbor_cutoff_nm(points_xy_nm, factor=ORDER_CUTOFF_FACTOR)

#     tree = cKDTree(points_xy_nm)
#     neighbor_lists = tree.query_ball_point(points_xy_nm, r=cutoff_nm)

#     psi = np.zeros(Np, dtype=np.complex128)

#     for i in range(Np):
#         nbrs = [j for j in neighbor_lists[i] if j != i]
#         if len(nbrs) == 0:
#             psi[i] = 0.0 + 0.0j
#             continue

#         vecs = points_xy_nm[nbrs] - points_xy_nm[i]
#         theta = np.arctan2(vecs[:, 1], vecs[:, 0])
#         psi[i] = np.mean(np.exp(1j * order_n * theta))

#     local_scalar = np.abs(psi)
#     global_scalar = np.abs(np.mean(psi))
#     return local_scalar, global_scalar, psi

def compute_local_and_global_order_xy(points_xy_nm, order_n=6, cutoff_nm=None):
    """
    Compute local and global bond-orientational order for one XY frame,
    matching the reference compute_psi6 style but returning magnitudes.

    For each particle j:

        psi_n(j) = (1 / z_j) sum_{k in neighbors(j)} exp(i * n * beta_jk)

    where beta_jk is the angle from particle j to particle k in the xy-plane.

    Parameters
    ----------
    points_xy_nm : ndarray, shape (N, 2)
        Particle xy positions in nm.

    order_n : int
        Bond-order symmetry. Use 6 for hexatic/triangular order.

    cutoff_nm : float or None
        Neighbor cutoff in nm. If None, it is inferred using
        infer_neighbor_cutoff_nm(...).

    Returns
    -------
    local_scalar : ndarray, shape (N,)
        local_scalar[j] = |psi_n(j)|.
        If particle j has no neighbors, local_scalar[j] = NaN.

    global_scalar : float
        |mean_j psi_n(j)|, ignoring NaN particles.

    psi_complex : ndarray, shape (N,), complex
        Complex psi_n(j). Isolated particles are NaN + i NaN.
    """
    points_xy_nm = np.asarray(points_xy_nm, dtype=float)
    Np = points_xy_nm.shape[0]

    if Np == 0:
        return np.array([]), np.nan, np.array([], dtype=np.complex128)

    if cutoff_nm is None or not np.isfinite(cutoff_nm):
        cutoff_nm = infer_neighbor_cutoff_nm(
            points_xy_nm,
            factor=ORDER_CUTOFF_FACTOR
        )

    psi_complex = np.empty(Np, dtype=np.complex128)
    psi_complex[:] = np.nan + 1j * np.nan

    local_scalar = np.full(Np, np.nan, dtype=np.float64)

    for j in range(Np):
        neighbors = []

        for k in range(Np):
            if k == j:
                continue

            dx = points_xy_nm[k, 0] - points_xy_nm[j, 0]
            dy = points_xy_nm[k, 1] - points_xy_nm[j, 1]
            r_jk = np.sqrt(dx**2 + dy**2)

            # strict cutoff, same as your reference code
            if r_jk < cutoff_nm:
                neighbors.append(k)

        z_j = len(neighbors)

        if z_j == 0:
            psi_complex[j] = np.nan + 1j * np.nan
            local_scalar[j] = np.nan
            continue

        total = 0.0 + 0.0j

        for k in neighbors:
            dx = points_xy_nm[k, 0] - points_xy_nm[j, 0]
            dy = points_xy_nm[k, 1] - points_xy_nm[j, 1]

            beta_jk = np.arctan2(dy, dx)
            total += np.exp(1j * order_n * beta_jk)

        psi_complex[j] = total / z_j
        local_scalar[j] = np.abs(psi_complex[j])

    valid = np.isfinite(psi_complex.real) & np.isfinite(psi_complex.imag)

    if np.any(valid):
        global_scalar = np.abs(np.mean(psi_complex[valid]))
    else:
        global_scalar = np.nan

    return local_scalar, global_scalar, psi_complex

def precompute_xy_metrics(
    frames_pos,
    frame_ids,
    metric_indices,
    total_points,
    order_n=6,
    cutoff_nm=None,
):
    time_axis, time_label, speed_label = load_time_axis(frame_ids, time_file)

    first_pts_nm = frames_pos[0] * 30.0
    first_metric_xy = first_pts_nm[metric_indices, :2]

    if cutoff_nm is None:
        cutoff_nm = infer_neighbor_cutoff_nm(
            first_metric_xy,
            factor=ORDER_CUTOFF_FACTOR
        )

    print(f"Using structural-order cutoff = {cutoff_nm:.3f} nm")

    local_order_all_frames = []
    global_order_frames = np.zeros(len(frames_pos), dtype=np.float64)
    com_frames = np.zeros((len(frames_pos), 3), dtype=np.float64)

    for k, pts in enumerate(frames_pos):
        pts_nm = pts * 30.0
        pts_metric = pts_nm[metric_indices]

        com_frames[k] = np.mean(pts_metric, axis=0)

        local_metric, global_scalar, _ = compute_local_and_global_order_xy(
            pts_metric[:, :2],
            order_n=order_n,
            cutoff_nm=cutoff_nm,
        )

        global_order_frames[k] = global_scalar

        local_full = np.full(total_points, np.nan, dtype=np.float64)
        local_full[metric_indices] = local_metric
        local_order_all_frames.append(local_full)

    _, com_speed_frames = compute_com_speed_nm_per_time(com_frames, time_axis)

    return {
        "time_axis": time_axis,
        "time_label": time_label,
        "speed_label": speed_label,
        "cutoff_nm": cutoff_nm,
        "local_order_frames": local_order_all_frames,
        "global_order_frames": global_order_frames,
        "com_frames": com_frames,
        "com_speed_frames": com_speed_frames,
    }

# ---------------- load all frames first ----------------
expected_shape = None
frames_pos = []
frames_vel = []
frame_ids = []

for N in range(STEPS):
    pos_file = os.path.join("Result", f"pos{N}rank0.h5")
    vel_file = os.path.join("Result", f"vel{N}rank0.h5")
    if not (os.path.exists(pos_file) and os.path.exists(vel_file)):
        break

    with h5py.File(pos_file, 'r') as f_pos:
        pts = f_pos['pos'][:]
    with h5py.File(vel_file, 'r') as f_vel:
        vel = f_vel['vel'][:]

    # Check shape consistency
    if expected_shape is None:
        expected_shape = pts.shape
    elif pts.shape != expected_shape:
        print(f"Warning: Frame {N} has shape {pts.shape}, expected {expected_shape}. Skipping.")
        continue

    frames_pos.append(pts)
    frames_vel.append(vel)
    frame_ids.append(N)

if len(frames_pos) == 0:
    raise RuntimeError("No frames found in Result/")

# ---------------- save all loaded positions to one .npy file ----------------
all_positions_npy = np.stack(frames_pos, axis=0)   # shape = (n_steps, n_particles, 3)

total_points = frames_pos[0].shape[0]
chain_start = total_points - n_filaments * n_chain

if TRACK_ONLY_DYNAMIC:
    dynamic_start = chain_start
    dynamic_end = total_points
else:
    dynamic_start = 0
    dynamic_end = total_points

metric_indices = np.arange(dynamic_start, dynamic_end, dtype=int)

print(f"Using particles {dynamic_start} to {dynamic_end-1} for COM/order metrics")
print(f"Number of metric particles = {len(metric_indices)}")

traj_nm_all = all_positions_npy * 30.0
traj_nm_dyn = traj_nm_all[:, chain_start:, :]

if time_hist is not None:
    if len(time_hist) >= traj_nm_dyn.shape[0]:
        time_hist_used = time_hist[:traj_nm_dyn.shape[0]]
        com_nm, com_speed = compute_com_speed_from_time(traj_nm_dyn, time_hist_used)
        print("Computed actual COM speed using saved simulation times.")
        print(f"Mean COM speed = {np.mean(com_speed):.6e} nm / time-unit")
    else:
        print("WARNING: time history is shorter than trajectory; cannot compute actual COM speed.")

npy_out = os.path.join(result_dir, f"03_30_{n_chain}_all_positions.npy")
np.save(npy_out, all_positions_npy)

print(f"Saved all particle positions to: {npy_out}")
print(f"Array shape: {all_positions_npy.shape}")

# ---------------- cluster metrics from actual trajectory ----------------
if SAVE_CLUSTER_METRICS:
    if CLUSTER_METRICS_DIR is None:
        CLUSTER_METRICS_DIR = os.path.join(result_dir, f"03_30_{n_chain}_cluster_metrics")

    # Full trajectory in nm, selecting only the dynamic cluster
    traj_nm_cluster_full = get_cluster_traj_nm(
        all_positions_npy,
        start_index=DYNAMIC_START_INDEX,
        end_index=DYNAMIC_END_INDEX,
    )

    # Optional stride if metrics are slow for very large assemblies
    metric_idx = np.arange(0, traj_nm_cluster_full.shape[0], METRIC_FRAME_STRIDE, dtype=int)

    traj_nm_cluster = traj_nm_cluster_full[metric_idx]
    frame_ids_metrics = [frame_ids[i] for i in metric_idx]

    # Load actual simulation time if available
    time_full, time_label = load_time_history_if_available(
        time_file,
        all_positions_npy.shape[0],
    )
    time_axis_metrics = time_full[metric_idx]

    cluster_metrics = save_cluster_metrics_outputs(
        traj_nm=traj_nm_cluster,
        frame_ids=frame_ids_metrics,
        time_axis=time_axis_metrics,
        time_label=time_label,
        out_dir=CLUSTER_METRICS_DIR,
        order_n=ORDER_N,
        cutoff_nm=ORDER_CUTOFF_NM,
        cutoff_factor=ORDER_CUTOFF_FACTOR,
        local_power=LOCAL_ORDER_POWER,
        save_local_csv=SAVE_LOCAL_ORDER_CSV,
    )
else:
    cluster_metrics = None
# ------------------------------------------------------------------------

rng = np.random.default_rng(12345)

if TRACK_ONLY_DYNAMIC:
    candidate_indices = list(range(chain_start, total_points))
else:
    candidate_indices = list(range(total_points))

if len(candidate_indices) <= 6:
    tracked_indices = candidate_indices
else:
    tracked_indices = sorted(rng.choice(candidate_indices, size=6, replace=False).tolist())

print("Tracked particle indices:", tracked_indices)

# ---------------- fixed axes over all frames ----------------
all_x = np.concatenate([P[:, 0] for P in frames_pos]) * 30.0
all_y = np.concatenate([P[:, 1] for P in frames_pos]) * 30.0
all_z = np.concatenate([P[:, 2] for P in frames_pos]) * 30.0

pad = 2.5 * PARTICLE_RADIUS
xlim = (all_x.min() - pad, all_x.max() + pad)
ylim = (all_y.min() - pad, all_y.max() + pad)
zlim = (all_z.min() - pad, all_z.max() + pad)

track_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']

# ---------------- precompute COM speed + structural order ----------------
metrics = precompute_xy_metrics(
    frames_pos=frames_pos,
    frame_ids=frame_ids,
    metric_indices=metric_indices,
    total_points=total_points,
    order_n=ORDER_N,
    cutoff_nm=ORDER_CUTOFF_NM,
)

local_order_frames = metrics["local_order_frames"]
global_order_frames = metrics["global_order_frames"]
com_speed_frames = metrics["com_speed_frames"]

order_cmap = cm.get_cmap("viridis")
order_norm = colors.Normalize(vmin=0.0, vmax=1.0)

# ---------------- helper: animation writer ----------------
def write_animation(frame_files, mp4_out, gif_out, fps=25):
    try:
        with imageio.get_writer(mp4_out, fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
            for fn in frame_files:
                writer.append_data(imageio.imread(fn))
        print(f"Saved MP4: {mp4_out}")
    except Exception as e:
        print(f"MP4 failed ({e}), writing GIF instead.")
        frames = [imageio.imread(fn) for fn in frame_files]
        imageio.mimsave(gif_out, frames, fps=fps)
        print(f"Saved GIF: {gif_out}")

# ---------------- helper: text overlay ----------------
def build_text_lines(step_id, pts_nm, frame_idx, metrics):
    t_now = metrics["time_axis"][frame_idx]
    time_label = metrics["time_label"]

    speed_now = metrics["com_speed_frames"][frame_idx]
    speed_label = metrics["speed_label"]

    global_order = metrics["global_order_frames"][frame_idx]

    text_lines = [
        f"step = {step_id}",
        f"t = {t_now:.6g} {time_label}",
        f"COM speed = {speed_now:.3e} {speed_label}",
        f"global |Psi_{ORDER_N}| = {global_order:.4f}",
    ]

    if SHOW_DISTANCE_TEXT and len(tracked_indices) == 3:
        p0 = pts_nm[tracked_indices[0], :3]
        p1 = pts_nm[tracked_indices[1], :3]
        p2 = pts_nm[tracked_indices[2], :3]

        d01 = np.linalg.norm(p0 - p1)
        d02 = np.linalg.norm(p0 - p2)
        d12 = np.linalg.norm(p1 - p2)

        text_lines.append(f"1-2: {d01:.2f} nm")
        text_lines.append(f"1-3: {d02:.2f} nm")
        text_lines.append(f"2-3: {d12:.2f} nm")

    return text_lines

# ---------------- precompute sphere surface ----------------
u = np.linspace(0, 2 * np.pi, SPHERE_RES_U)
v = np.linspace(0, np.pi, SPHERE_RES_V)
sphere_x = PARTICLE_RADIUS * np.outer(np.cos(u), np.sin(v))
sphere_y = PARTICLE_RADIUS * np.outer(np.sin(u), np.sin(v))
sphere_z = PARTICLE_RADIUS * np.outer(np.ones_like(u), np.cos(v))

# ---------------- build XY frames with trails ----------------
xy_frame_files = []

for k, N in enumerate(frame_ids):
    if N % 1 == 0:
        print(f'Building XY frame {k+1}/{len(frame_ids)}', end='\r')

        pts = frames_pos[k]
        vel = frames_vel[k]

        x_all = pts[:, 0] * 30.0
        y_all = pts[:, 1] * 30.0
        pts_nm = pts * 30.0

        fig, ax = plt.subplots(figsize=(6, 6))

        # --- trails for tracked particles in XY only ---
        hist_start = max(0, k - TRAIL_LENGTH + 1)
        for local_id, pidx in enumerate(tracked_indices):
            traj = np.array([frames_pos[m][pidx, :2] for m in range(hist_start, k + 1)]) * 30.0
            ax.plot(
                traj[:, 0], traj[:, 1],
                color=track_colors[local_id % len(track_colors)],
                lw=1.2,
                alpha=0.9,
                zorder=2
            )

        # --- current particle circles ---
        # --- current particle circles, colored by local structural order ---
        local_vals = metrics["local_order_frames"][k]

        patches = []
        facecolors = []

        for i in range(total_points):
            patches.append(plt.Circle((x_all[i], y_all[i]), radius=PARTICLE_RADIUS))

            if COLOR_BY_LOCAL_ORDER and np.isfinite(local_vals[i]):
                facecolors.append(order_cmap(order_norm(local_vals[i])))
            else:
                facecolors.append((0.83, 0.83, 0.83, 0.95))

        coll = PatchCollection(
            patches,
            facecolor=facecolors,
            edgecolors="black",
            linewidths=0.6,
            alpha=0.95,
            zorder=5,
        )
        ax.add_collection(coll)

        if SHOW_COLORBAR:
            sm = cm.ScalarMappable(norm=order_norm, cmap=order_cmap)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(f"Local order |psi_{ORDER_N}|")

        if CONNECT_FILAMENTS:
            for j in range(n_filaments):
                i0 = chain_start + j * n_chain
                i1 = i0 + n_chain
                ax.plot(
                    x_all[i0:i1], y_all[i0:i1],
                    '-',
                    color='k',
                    lw=1.2,
                    zorder=6
                )

        ax.text(
            0.03, 0.97,
            "\n".join(build_text_lines(N, pts_nm, k, metrics)),
            transform=ax.transAxes,
            ha='left', va='top',
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.70, edgecolor='none')
        )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('x [nm]')
        ax.set_ylabel('y [nm]')
        ax.set_title('XY view')

        plt.tight_layout()

        out_path = os.path.join(result_dir, f"03_30_{n_chain}_tracked_xy_{k:04d}.png")
        plt.savefig(out_path, dpi=150)
        plt.close(fig)

        xy_frame_files.append(out_path)

print()

# ---------------- build XY animation ----------------
xy_mp4_out = os.path.join(result_dir, f"03_30_{n_chain}_tracked_xy.mp4")
xy_gif_out = os.path.join(result_dir, f"03_30_{n_chain}_tracked_xy.gif")
write_animation(xy_frame_files, xy_mp4_out, xy_gif_out, fps=FPS)

# ---------------- build 3D frames with full spheres ----------------
three_d_frame_files = []

for k, N in enumerate(frame_ids):
    if N % 1 == 0:
        print(f'Building 3D frame {k+1}/{len(frame_ids)}', end='\r')

        pts = frames_pos[k]
        pts_nm = pts * 30.0

        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection='3d')

        xmin0, xmax0 = xlim
        ymin0, ymax0 = ylim
        zmin0, zmax0 = zlim

        xmid = 0.5 * (xmin0 + xmax0)
        ymid = 0.5 * (ymin0 + ymax0)
        zmid = 0.5 * (zmin0 + zmax0)

        half_span = 0.5 * max(
            xmax0 - xmin0,
            ymax0 - ymin0,
            zmax0 - zmin0
        )

        common_xlim = (xmid - half_span, xmid + half_span)
        common_ylim = (ymid - half_span, ymid + half_span)
        common_zlim = (zmid - half_span, zmid + half_span)

        # draw all particles as full spheres
        for i in range(total_points):
            cx, cy, cz = pts_nm[i, 0], pts_nm[i, 1], pts_nm[i, 2]
            ax.plot_surface(
                sphere_x + cx,
                sphere_y + cy,
                sphere_z + cz,
                color='lightgray',
                edgecolor='black',
                linewidth=0.2,
                antialiased=True,
                shade=True,
                alpha=0.95
            )

        # optional filament connection in 3D
        if CONNECT_FILAMENTS:
            x_all = pts_nm[:, 0]
            y_all = pts_nm[:, 1]
            z_all = pts_nm[:, 2]
            for j in range(n_filaments):
                i0 = chain_start + j * n_chain
                i1 = i0 + n_chain
                ax.plot(
                    x_all[i0:i1],
                    y_all[i0:i1],
                    z_all[i0:i1],
                    '-',
                    color='k',
                    lw=1.2
                )

        # fixed view
        ax.view_init(elev=THREE_D_ELEV, azim=THREE_D_AZIM)

        # same overlay text in 3D
        ax.text2D(
            0.03, 0.97,
            "\n".join(build_text_lines(N, pts_nm, k, metrics)),
            transform=ax.transAxes,
            ha='left', va='top',
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.70, edgecolor='none')
        )

        ax.set_xlim(*common_xlim)
        ax.set_ylim(*common_ylim)
        ax.set_zlim(*common_zlim)

        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect((1, 1, 1))

        ax.set_xlabel('x [nm]')
        ax.set_ylabel('y [nm]')
        ax.set_zlabel('z [nm]')
        ax.set_title(f'3D view (elev={THREE_D_ELEV}, azim={THREE_D_AZIM})')

        plt.tight_layout()

        out_path = os.path.join(result_dir, f"03_30_{n_chain}_tracked_3d_{k:04d}.png")
        plt.savefig(out_path, dpi=150)
        plt.close(fig)

        three_d_frame_files.append(out_path)

print()

# ---------------- build 3D animation ----------------
three_d_mp4_out = os.path.join(result_dir, f"03_30_{n_chain}_tracked_3d.mp4")
three_d_gif_out = os.path.join(result_dir, f"03_30_{n_chain}_tracked_3d.gif")
write_animation(three_d_frame_files, three_d_mp4_out, three_d_gif_out, fps=FPS)