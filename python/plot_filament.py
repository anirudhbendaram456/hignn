# import numpy as np
# import vtk
# from vtk.util import numpy_support as VN
# import os
# import h5py
# #from filament import MAXIMUM_NUMBER_OF_ITERATIONS, N_CHAIN
  
# # MAXIMUM_NUMBER_OF_ITERATIONS = 100000
# # N_CHAIN = 31

# for N in range(9800):
#     os.system('clear')
#     print(f'N = {N}', end='\r')

#     points = vtk.vtkPoints()
#     velocity = vtk.vtkFloatArray()
#     velocity.SetNumberOfComponents(3)
#     velocity.SetName('u')

#     move_forward = True
#     count = 0

#     for rank in range(1):  # Adjust if you have multiple ranks
#         try:
#             with h5py.File(f'Result/pos{N}rank{rank}.h5', 'r') as f_pos:
#                 points_array = f_pos['pos'][:]
#             with h5py.File(f'Result/vel{N}rank{rank}.h5', 'r') as f_vel:
#                 velocity_array = f_vel['vel'][:]
#         except Exception as e:
#             print(f"Error reading file for N={N}, rank={rank}: {e}")
#             move_forward = False
#             break

#         # Insert points and velocity data
#         for i in range(points_array.shape[0]):
#             points.InsertNextPoint(points_array[i, :])
#             velocity.InsertNextTuple3(*velocity_array[i])

#         count += points_array.shape[0]

#     if not move_forward:
#         break

#     # Create polydata
#     polydata = vtk.vtkPolyData()
#     polydata.SetPoints(points)
#     polydata.GetPointData().SetVectors(velocity)

#     # Create line connectivity
#     # lines = vtk.vtkCellArray()
#     # n_chain = 30  # Particles per filament
#     # num_chains = points_array.shape[0] // n_chain  # Number of filaments

#     # if num_chains > 0:
#     #     for n in range(num_chains):
#     #         line = vtk.vtkPolyLine()
#     #         line.GetPointIds().SetNumberOfIds(n_chain)

#     #         for i in range(n_chain):
#     #             global_index = n * n_chain + i  
#     #             line.GetPointIds().SetId(i, global_index)

#     #         lines.InsertNextCell(line)

#     #     polydata.SetLines(lines)
#     # else:
#     #     print(f"⚠ Warning: No lines created for N={N}")

#     # Debugging output
#     print(f"Frame {N}: Points = {polydata.GetNumberOfPoints()}, Lines = {polydata.GetNumberOfLines()}")

#     # Write to VTP file
#     writer = vtk.vtkXMLPolyDataWriter()
#     writer.SetInputData(polydata)
#     writer.SetFileName(f'Result/w_multi_filament{N}.vtp')
#     writer.Write()

#!/usr/bin/env python3
#!/usr/bin/env python3
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt

# Number of particles per filament (chain length)
n_chain = 31

# Determine the folder in which this script lives
script_dir = os.path.dirname(os.path.abspath(__file__))

# We want to save into:
#   /home/bendaram/Projects/hignn_internal/Result
# Given script_dir = .../hignn-internal/python,
#  └─ parent    = .../hignn-internal
#     └─ parent  = .../hignn_internal
# So we go two levels up, then “Result”
result_dir = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir, 'Result'))

# Ensure the Result folder exists
os.makedirs(result_dir, exist_ok=True)

# Loop over time‐step indices N
for N in range(5000):
    os.system('clear')
    print(f'N = {N}', end='\r')

    points_list = []
    move_forward = True

    # In your original code, you looped over rank from 0 .. (n_ranks−1).
    # Here, we assume only rank=0. If you have more ranks, increase the range.
    for rank in range(1):
        try:
            # Read the HDF5 file containing positions
            with h5py.File(f'Result/pos{N}rank{rank}.h5', 'r') as f_pos:
                pts = f_pos['pos'][:]    # shape = (n_particles_this_rank, 3)
        except Exception as e:
            print(f"\nError reading file for N={N}, rank={rank}: {e}")
            move_forward = False
            break

        points_list.append(pts)

    if not move_forward:
        break

    # Stack all rank‐specific arrays together (if >1 rank)
    points_combined = np.vstack(points_list)  # shape = (total_particles, 3)
    total_points = points_combined.shape[0]

    # Compute how many full filaments (each of length n_chain) exist
    num_chains = total_points // n_chain

    # Extract x and z coordinates
    x_all = points_combined[:, 1]
    z_all = points_combined[:, 2]

    # ------------------------------------------------------------------------
    # Plot setup: draw each filament’s 30 points and connect them in x–z plane
    # ------------------------------------------------------------------------

    if (N%100==0):
        plt.figure(figsize=(6, 6))

        # (Optional) show all points faintly in the background for context
        plt.scatter(x_all, z_all, s=5, color='lightgray', alpha=0.4)

        # for chain_idx in range(num_chains):
        #     start = chain_idx * n_chain
        #     end = start + n_chain

        #     x_chain = x_all[start:end]
        #     z_chain = z_all[start:end]

        #     # Draw straight‐line connections between consecutive points,
        #     # with a small circle at each point
        #     plt.plot(x_chain, z_chain, '-o', markersize=3, linewidth=1)

        plt.xlabel('x')
        plt.ylabel('z')
        plt.title(f'Frame {N}: {num_chains} filaments, {total_points} pts')
        plt.axis('equal')
        plt.tight_layout()

        # ------------------------------------------------------------------------
        # 1) Save each frame as a PNG into /home/bendaram/Projects/hignn_internal/Result/
        # ------------------------------------------------------------------------
        out_path = os.path.join(result_dir, f'15_floor_yz_{int(N/100):03d}.png')
        plt.savefig(out_path, dpi=150)

        # ------------------------------------------------------------------------
        # 2) Display the plot live (pauses 0.1 seconds to render)
        # ------------------------------------------------------------------------
        plt.show()
        #plt.pause(0.1)

        # Close the figure to free memory before next iteration
        plt.close()