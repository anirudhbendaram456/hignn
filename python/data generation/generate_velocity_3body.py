import numpy as np
import time
# from functions_shared import add_sphere_rotations_to_positions, same_setup_as
# from functions_timestepping import euler_timestep, did_something_go_wrong_with_dumbells, euler_timestep_rotation, \
#     orthogonal_proj, do_we_have_all_size_ratios, generate_output_FTSUOE, are_some_of_the_particles_too_close
from functions.timestepping import generate_output_FTSUOE
from functions.shared import add_sphere_rotations_to_positions
from setups.inputs import input_ftsuoe        # if you call it directly
from setups.functions_positions import simple_cubic_8

# ---------- helpers ----------
def solve_holding_forces(X_subset, input_number=60, flow_form='ufte'):
    """
    X_subset: (M,3) positions of the spheres present in this run.
    Returns: Fa_out (M,3) = hydrodynamic holding forces on those spheres.
    """
    M = X_subset.shape[0]
    sphere_sizes = np.ones(M)
    dumbbell_sizes = np.array([])
    dumbbell_positions = np.empty([0, 3])
    dumbbell_deltax = np.empty([0, 3])

    sphere_rot = add_sphere_rotations_to_positions(
        X_subset, sphere_sizes, np.array([[1,0,0],[0,0,1]])
    )
    posdata = (sphere_sizes, X_subset, sphere_rot,
               dumbbell_sizes, dumbbell_positions, dumbbell_deltax)

    # Unbounded domain; set periodic box here if needed
    box_bottom_left = np.array([0,0,0])
    box_top_right   = np.array([0,0,0])

    last_Minfty_inv = np.array([0])
    regenerate_Minfty = True
    frameno = 0
    timestep = 0.1

    (Fa_out, Ta_out, Sa_out, Fb_out, DFb_out,
     Ua_out, Oa_out, Ea_out, Ub_out, DUb_out,
     last_Minfty_inv, gen_times, Uinf, Oinf,
     centre, F_wall, last_vel_vec, grand_R) = generate_output_FTSUOE(
        posdata, frameno, timestep, input_number,
        last_Minfty_inv, regenerate_Minfty, flow_form,
        cutoff_factor=2, printout=0, use_XYZd_values=True,
        use_drag_Minfinity=False, use_Minfinity_only=False,
        extract_force_on_wall_due_to_dumbbells=False,
        last_velocities=(np.zeros((M,3)), np.array([]), np.array([]), np.zeros((M,3))),
        last_velocity_vector=[0]*(M*11),
        checkpoint_start_from_frame=0,
        box_bottom_left=box_bottom_left, box_top_right=box_top_right,
        feed_every_n_timesteps=0
    )
    return Fa_out

def three_body_increment_forces(X):
    """ΔF^(3): pure 3-body hydrodynamic increment per sphere."""
    F_all = solve_holding_forces(X)                 # (3,3): i,j,k with all present
    F_ij  = solve_holding_forces(X[[0,1], :])       # (2,3): i,j
    F_ik  = solve_holding_forces(X[[0,2], :])       # (2,3): i,k
    F_jk  = solve_holding_forces(X[[1,2], :])       # (2,3): j,k
    F_i   = solve_holding_forces(X[[0],   :])       # (1,3): i
    F_j   = solve_holding_forces(X[[1],   :])       # (1,3): j
    F_k   = solve_holding_forces(X[[2],   :])       # (1,3): k

    dF3 = np.zeros_like(X)
    dF3[0] = F_all[0] - F_ij[0] - F_ik[0] + F_i[0]
    dF3[1] = F_all[1] - F_ij[1] - F_jk[0] + F_j[0]
    dF3[2] = F_all[2] - F_ik[1] - F_jk[1] + F_k[0]

    # Also return pieces we'll reuse
    pieces = dict(F_all=F_all, F_ij=F_ij, F_ik=F_ik, F_jk=F_jk, F_i=F_i, F_j=F_j, F_k=F_k)
    return dF3, pieces

def decompose_forces_3body(X):
    """
    Returns a dict with:
      F_total      : (3,3) = forces with all spheres present
      F_single     : (3,3) = single-body (each alone) forces
      F_pair_ij    : (3,3) = pairwise hydrodynamic increment from pair (i,j) on all spheres (k row is zero)
      F_pair_ik    : (3,3)
      F_pair_jk    : (3,3)
      F_pair_sum   : (3,3) = F_pair_ij + F_pair_ik + F_pair_jk
      F_three      : (3,3) = pure 3-body hydrodynamic increment ΔF^(3)
    And it satisfies:  F_total = F_single + F_pair_sum + F_three
    """
    F_three, P = three_body_increment_forces(X)
    F_all, F_ij, F_ik, F_jk, F_i, F_j, F_k = P['F_all'], P['F_ij'], P['F_ik'], P['F_jk'], P['F_i'], P['F_j'], P['F_k']

    F_single = np.zeros_like(X)
    F_single[0] = F_i[0]; F_single[1] = F_j[0]; F_single[2] = F_k[0]

    # Pairwise hydrodynamic increments (extend to 3x3 with zeros for the absent sphere)
    Fpair_ij = np.zeros_like(X)
    Fpair_ij[0] = F_ij[0] - F_i[0]
    Fpair_ij[1] = F_ij[1] - F_j[0]
    # Fpair_ij[2] stays zero

    Fpair_ik = np.zeros_like(X)
    Fpair_ik[0] = F_ik[0] - F_i[0]
    Fpair_ik[2] = F_ik[1] - F_k[0]
    # Fpair_ik[1] stays zero

    Fpair_jk = np.zeros_like(X)
    Fpair_jk[1] = F_jk[0] - F_j[0]
    Fpair_jk[2] = F_jk[1] - F_k[0]
    # Fpair_jk[0] stays zero

    Fpair_sum = Fpair_ij + Fpair_ik + Fpair_jk

    # Sanity: F_all == F_single + Fpair_sum + F_three
    # err = np.linalg.norm((F_single + Fpair_sum + F_three) - F_all)
    # print('decomp error', err)

    return dict(F_total=F_all,
                F_single=F_single,
                F_pair_ij=Fpair_ij, F_pair_ik=Fpair_ik, F_pair_jk=Fpair_jk,
                F_pair_sum=Fpair_sum,
                F_three=F_three)

# ############ load position ############
data_pos = np.loadtxt('sample_3cir_box32.txt')
# data in a shape of (number, 9)
# in the form of (x1, y1, z1, x2, y2, z2, x3, y3, z3)
# You need to generate the sample that you want.
Nc = 3
print(data_pos.shape)
print(data_pos[0, :])

# ############ input mode ############
# change this part according to the boundary condition. 
# Please carefully read and understand this part of the original code in https://github.com/Pecnut/stokesian-dynamics.

# unbounded
# box_bottom_left = np.array([0, 0, 0])
# box_top_right = np.array([0, 0, 0])

# ############ Don't change ############
num_spheres = int(data_pos.shape[1] / 3)
num_data = data_pos.shape[0]
sphere_sizes = np.array([1.0 for i in range(num_spheres)])
dumbbell_sizes = np.array([])
dumbbell_positions = np.empty([0, 3])
dumbbell_deltax = np.empty([0, 3])
input_form = 'fte'
use_drag_Minfinity = False
use_Minfinity_only = False
extract_force_on_wall_due_to_dumbbells = False
last_velocities = (np.zeros((num_spheres, 3)), np.array([]), np.array([]), np.zeros((num_spheres, 3)))
last_velocity_vector = [0] * num_spheres * 11
checkpoint_start_from_frame = 0
feed_every_n_timesteps = 0
frameno = 0
last_generated_Minfinity_inverse = np.array([0])
regenerate_Minfinity = True
timestep = 0.1
cutoff_factor = 2
printout = 0
use_XYZd_values = True


# # ############ Output ############
# data_vel = np.zeros((num_data, 15))
# t1 = time.time()
# for i in range(data_pos.shape[0]):
#     sphere_positions = data_pos[i, :].reshape((num_spheres, 3))
    
#     # IMPORTANT: input number is defined in input_setups.py
#     # it is related to the boundary condition. Please make sure you understand it.
#     input_number = i % 3 + 10
    

#     sphere_rotations = add_sphere_rotations_to_positions(sphere_positions, sphere_sizes,
#                                                          np.array([[1, 0, 0], [0, 0, 1]]))
#     posdata = (sphere_sizes, sphere_positions, sphere_rotations, dumbbell_sizes, dumbbell_positions, dumbbell_deltax)
#     Fa_out_k1, Ta_out_k1, Sa_out_k1, Fb_out_k1, DFb_out_k1, Ua_out_k1, Oa_out_k1, Ea_out_k1, Ub_out_k1, DUb_out_k1, \
#     last_generated_Minfinity_inverse, gen_times, U_infinity_k1, O_infinity_k1, centre_of_background_flow, force_on_wall_due_to_dumbbells_k1, \
#     last_velocity_vector, grand_resistance_matrix = generate_output_FTSUOE(
#         posdata, frameno, timestep, input_number, last_generated_Minfinity_inverse, regenerate_Minfinity, input_form,
#         cutoff_factor, printout, use_XYZd_values, use_drag_Minfinity, use_Minfinity_only,
#         extract_force_on_wall_due_to_dumbbells, last_velocities, last_velocity_vector, checkpoint_start_from_frame,
#         box_bottom_left, box_top_right, feed_every_n_timesteps=feed_every_n_timesteps)

    
#     Ua_out_k1[:, 0] = 6 * np.pi * Ua_out_k1[:, 0]
#     Ua_out_k1[:, 1] = 6 * np.pi * Ua_out_k1[:, 1]
#     Ua_out_k1[:, 2] = 6 * np.pi * Ua_out_k1[:, 2]
#     # print Ua_out_k1
    
#     Ua_out_k1 = Ua_out_k1.reshape(-1)
#     # print Ua_out_k1
#     vel = np.concatenate((data_pos[i, 3:9], Ua_out_k1), axis=0)
    
#     # print vel
#     # velocity_temp = np.concatenate((posdata[1], Ua_out_k1), axis=1)
#     # velocity_temp = velocity_temp.reshape(1, 6 * num_spheres)
#     data_vel[i, :] = vel
    
#     if i % 100 == 0:
#         print i

# t2 = time.time()
# print t2 - t1
# print data_vel.shape
# # print data_vel[0:4, :]
# np.savetxt('data_output/data_3cir_box32.txt', data_vel)

pos_forces_summary  = np.zeros((num_data, 9 + 9 + 9 + 9))   # pos + F_total + F_pair_sum + F_three
pos_forces_detailed = np.zeros((num_data, 9 + 9 + 9 + 9 + 9 + 9 + 9 + 9))  # pos + F_single + F_ij + F_ik + F_jk + F_total + F_pair_sum + F_three

t1 = time.time()
for n in range(num_data):
    X = data_pos[n, :].reshape((3,3))

    D = decompose_forces_3body(X)

    # summary: positions + F_total + F_pair_sum + F_three
    row_summary = np.concatenate([
        X.reshape(-1),
        D['F_total'].reshape(-1),
        D['F_pair_sum'].reshape(-1),
        D['F_three'].reshape(-1)
    ])
    pos_forces_summary[n, :] = row_summary

    # detailed: positions + F_single + F_ij + F_ik + F_jk + F_total + F_pair_sum + F_three
    row_detailed = np.concatenate([
        X.reshape(-1),
        D['F_single'].reshape(-1),
        D['F_pair_ij'].reshape(-1),
        D['F_pair_ik'].reshape(-1),
        D['F_pair_jk'].reshape(-1),
        D['F_total'].reshape(-1),
        D['F_pair_sum'].reshape(-1),
        D['F_three'].reshape(-1)
    ])
    pos_forces_detailed[n, :] = row_detailed

    if n % 100 == 0:
        print("processed", n)

t2 = time.time()
print("elapsed [s]:", (t2 - t1))

# Save (headers describe blocks; columns are 3D vectors per sphere in i,j,k order)
np.savetxt('data_output/pos_forces_summary.txt',  pos_forces_summary,
           header='[pos(9)] [F_total(9)] [F_pair_sum(9)] [F_three(9)]')
np.savetxt('data_output/pos_forces_detailed.txt', pos_forces_detailed,
           header='[pos(9)] [F_single(9)] [F_ij(9)] [F_ik(9)] [F_jk(9)] [F_total(9)] [F_pair_sum(9)] [F_three(9)]')
