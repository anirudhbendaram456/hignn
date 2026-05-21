import os
import hignn
import json
import numpy as np
import sys
from mpi4py import MPI
import time
import h5py

os.system("clear")
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def chain_bending(X, k_b):
    """
    Compute the bending forces for a single filament chain.

    Parameters:
    -----------
    X : np.ndarray
        Coordinates of the particles in the filament, of shape (N, 3).

    k_b : float
          Bending stiffness parameter.

    Returns:
    --------
    F: np.ndarray
       Bending forces for each particle, of shape (N, 3).
    """
    N = X.shape[0]
    F = np.zeros((N, 3))
    arm1 = X[1 : (N - 1), :] - X[0 : (N - 2), :]
    arm2 = X[2:N, :] - X[1 : (N - 1), :]

    arm1_norm = np.linalg.norm(arm1, ord=2, axis=1).reshape((-1, 1))
    arm2_norm = np.linalg.norm(arm2, ord=2, axis=1).reshape((-1, 1))
    arm1 = arm1 / arm1_norm
    arm2 = arm2 / arm2_norm
    theta_cos = np.sum(arm1 * arm2, axis=1)
    tmp = 1 - theta_cos * theta_cos
    tmp[(tmp < 0).nonzero()] = 0.0
    theta_sin = np.sqrt(tmp).reshape((-1, 1))
    uv = np.cross(arm1, arm2)
    torque = -uv * theta_sin
    f21 = np.cross(torque, arm1)
    f23 = np.cross(torque, arm2)
    F[0 : (N - 2), :] += f21
    F[1 : (N - 1), :] -= f21 + f23
    F[2:N, :] += f23
    F = k_b * F

    return F


def chain_tension(X, k_t, rest_length):
    """
    Compute the tension forces for a single filament chain.

    Parameters:
    -----------
    X : np.ndarray
        Coordinates of the particles in the filament, of shape (N, 3).

    k_t : float
          Tension stiffness parameter.

    rest_length : float
                  Rest length between consecutive particles in the chain.

    Returns:
    --------
    F: np.ndarray
       Tension forces for each particle, of shape (N, 3).
    """
    N = X.shape[0]
    F = np.zeros((N, 3))

    r = X[0 : (N - 1), :] - X[1:N, :]
    r_norm = np.linalg.norm(r, ord=2, axis=1)
    r_norm = r_norm.reshape((N - 1, 1))
    r = r / r_norm
    Fm = k_t * (r_norm - rest_length)
    nodal_force = r * Fm
    F[0 : (N - 1), :] -= nodal_force
    F[1:N, :] += nodal_force

    return F


def velocity_update(
    hignn_model,
    t,
    position,
    b,
    n_filament,
    n_chain,
    rest_length,
    k_t,
    k_b,
    divm_enabled,
    divm_scale,
):
    """
    Update the velocity of each particle based on external forces, chain tension, and bending forces.

    Parameters:
    -----------
    hignn_model: HignnModel
                 The object used to assemble mobility tensor and perform dot product
    t : float
        Current simulation time.

    position : np.ndarray
               Current positions of all particles, of shape (N, 3).

    b : float array, optional, default=-1
        Body force applied.

    n_filament : int, optional
                 Number of filaments in the system.

    n_chain : int, optional
              Number of particles in each filament chain.

    rest_length : float, optional
                  Rest length between consecutive particles in the chain.

    k_t : float, optional
          Tension stiffness parameter.

    K_b : float, optional
          Bending stiffness parameter.

    Returns:
    --------
    velocity: np.ndarray
              Updated velocities for each particle, of shape (N, 3).
    """
    if rank == 0:
        print("t = {t:.4f}".format(t=t))
    hignn_model.update_coord(position[:, 0:3])
    velocity = np.zeros((position.shape[0], 3), dtype=np.float32)
    force = np.zeros((position.shape[0], 3), dtype=np.float32)
    divM = np.zeros((position.shape[0], 3), dtype=np.float32)
    force += b

    # Filaments
    if n_filament > 0:
        for i in range(n_filament):
            pos = position[n_chain * i : n_chain * i + n_chain, :]
            pos3 = pos[:, :3]
            force[n_chain * i : n_chain * i + n_chain, :] += chain_bending(pos3, k_b)
            force[n_chain * i : n_chain * i + n_chain, :] += chain_tension(
                pos3, k_t, rest_length
            )

    hignn_model.dot(velocity, force, divM)

    if divm_enabled:
        velocity += divm_scale * divM
    return velocity


class Simulator:
    def __init__(self, config_filename):
        # Load simulation parameters from JSON
        with open(f"{config_filename}", "r") as f:
            config = json.load(f)

        simulation_params = config["simulation"]
        self.time_integrator = simulation_params[
            "time_integrator"
        ]  # Time integration method
        self.b = simulation_params["body_force"]  # Constant body force
        self.dt = simulation_params["dt"]  # Time step size
        self.t_max = simulation_params["t_max"]  # Total simulation time
        self.t_meas = simulation_params["t_meas"]  # Interval for outputting data
        divm_params = simulation_params.get("divm", {})
        self.divm_enabled = divm_params.get("enabled", True)
        divm_scale = divm_params.get("scale")
        if divm_scale is None:
            temperature = divm_params.get("temperature", 300.0)
            boltzmann_constant = divm_params.get(
                "boltzmann_constant", 1.380649e-23
            )
            divm_scale = boltzmann_constant * temperature
        self.divm_scale = divm_scale

        # Load cloud parameters from JSON
        cloud_params = config["cloud"]
        self.cloud_type = cloud_params["type"]
        self.filament_cloud_params = cloud_params.get("filament")

        self.hignn_model_params = config["model"]
        self.two_body_model = self.hignn_model_params["two_body_model"]

        self.output_counter = 0

    def run(self):
        # Initialize hignn
        hignn.Init()

        if self.filament_cloud_params:
            self.n_filament = self.filament_cloud_params[
                "n_filament"
            ]  # Number of filaments
            self.n_chain = self.filament_cloud_params[
                "n_chain"
            ]  # Particles per filament
            self.rest_length = self.filament_cloud_params[
                "rest_length"
            ]  # Rest length between particles
            self.k_t = self.filament_cloud_params["k_t"]  # Tension stiffness parameter
            self.k_b = self.filament_cloud_params["k_b"]  # Bending stiffness parameter
        else:
            self.n_filament = self.n_chain = self.rest_length = self.k_t = self.k_b = 0

        self.working_directory = os.path.dirname(os.path.abspath(__file__))
        X = np.loadtxt(self.working_directory + "/cloud/" + self.cloud_type + ".pos")

        self.rank_range = np.linspace(
            0, X.shape[0], comm.Get_size() + 1, dtype=np.int32
        )

        # HIGNN model
        self.hignn_model = hignn.HignnModel(
            X, 100 if X.shape[0] > 100 else X.shape[0] // 2
        )  # The minimal number of particles in a leaf node
        self.hignn_model.load_two_body_model(self.two_body_model)

        # Set parameters for far dot, the following parameters are default values
        self.hignn_model.set_epsilon(0.01)
        self.hignn_model.set_max_iter(50)
        self.hignn_model.set_mat_pool_size_factor(200)
        self.hignn_model.set_post_check_flag(False)
        self.hignn_model.set_use_symmetry_flag(True)
        self.hignn_model.set_max_far_dot_work_node_size(10000)
        self.hignn_model.set_max_relative_coord(100000)

        # Start timing the entire simulation
        t1 = time.time()

        if self.time_integrator == "Explicit Euler":
            time_integrator = hignn.ExplicitEuler()
        elif self.time_integrator == "Explicit RK4":
            time_integrator = hignn.ExplicitRk4()
        else:
            raise ValueError(f"Unknown time integrator: {self.time_integrator}")

        time_integrator.set_time_step(self.dt)
        time_integrator.set_final_time(self.t_max)
        time_integrator.set_num_rigid_body(X.shape[0])
        time_integrator.set_output_step(self.t_meas)
        time_integrator.set_output_func(self.output_hdf5)

        time_integrator.set_velocity_func(self.velocity_update_wrapper)
        time_integrator.initialize(X)
        time_integrator.run()

        """
                # Initialize simulation time and iteration counter
        ts = 0        # Initial time
        ite = 0       # Iteration counter
        os.makedirs(self.working_directory+'/output/hdf5', exist_ok=True)
        
        # Main simulation loop
        for i in range(int(self.t_max/self.dt)):

            # Every 100 iterations, save particle positions to an HDF5 file
            if i % self.t_meas == 0:
                with h5py.File(f'{self.working_directory}/output/hdf5/pos_rank_{rank}_{self.output_counter}.h5', 'w') as f:
                    f.create_dataset('pos', data=X[self.rank_range[rank]:self.rank_range[rank+1], :])

            # Measure time for velocity update step
            tt1 = time.time()
            V = self.velocity_update_wrapper(ts, X)

            # Print velocity update time only on rank 0 (to avoid redundant logging)
            if rank == 0:
                print("Time for velocity_update: {t:.4f}s".format(t=time.time() - tt1))

            # Every 100 iterations, save particle velocities to an HDF5 file
            if i % self.t_meas == 0:
                with h5py.File(f'{self.working_directory}/output/hdf5/vel_rank_{rank}_{self.output_counter}.h5', 'w') as f:
                    f.create_dataset('vel', data=V[self.rank_range[rank]:self.rank_range[rank+1], :])
                self.output_counter += 1

            # Update particle positions using velocity
            X = X + self.dt * V

            # Advance simulation time
            ts = ts + self.dt
        """

        # Print total simulation time (only for rank 0)
        if rank == 0:
            print("--------------------------------")
            print("Time for simulation: {t:.2f} s".format(t=time.time() - t1))

        del self.hignn_model

        # Finalize hignn
        hignn.Finalize()

    def velocity_update_wrapper(self, ts, X):
        V = velocity_update(
            self.hignn_model,
            ts,
            X,
            self.b,
            self.n_filament,
            self.n_chain,
            self.rest_length,
            self.k_t,
            self.k_b,
            self.divm_enabled,
            self.divm_scale,
        )

        return V

    def output_hdf5(self, position, velocity):
        os.makedirs(self.working_directory + "/output/hdf5", exist_ok=True)
        with h5py.File(
            self.working_directory
            + "/output/hdf5/pos_rank_"
            + str(rank)
            + "_"
            + str(self.output_counter)
            + ".h5",
            "w",
        ) as f:
            f.create_dataset(
                "pos",
                data=position[self.rank_range[rank] : self.rank_range[rank + 1], :],
            )

        with h5py.File(
            self.working_directory
            + "/output/hdf5/vel_rank_"
            + str(rank)
            + "_"
            + str(self.output_counter)
            + ".h5",
            "w",
        ) as f:
            f.create_dataset(
                "vel",
                data=velocity[self.rank_range[rank] : self.rank_range[rank + 1], :],
            )

        self.output_counter += 1
