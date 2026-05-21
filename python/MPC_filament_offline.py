import os
import hignn
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from mpi4py import MPI
import time
import csv  # for saving kb schedules

# --- Existing physics functions (from your simulation script) ---
def chain_bending(X, k_b):
    N = X.shape[0]
    F = np.zeros((N, 3))
    arm1 = X[1:(N - 1), :] - X[0:(N - 2), :]
    arm2 = X[2:N, :] - X[1:(N - 1), :]
    arm1 /= np.linalg.norm(arm1, axis=1).reshape(-1, 1)
    arm2 /= np.linalg.norm(arm2, axis=1).reshape(-1, 1)
    theta_cos = np.sum(arm1 * arm2, axis=1)
    sin2 = 1 - theta_cos**2
    sin2[sin2 < 0] = 0.0
    theta_sin = np.sqrt(sin2).reshape(-1, 1)
    uv = np.cross(arm1, arm2)
    torque = -uv * theta_sin
    f21 = np.cross(torque, arm1)
    f23 = np.cross(torque, arm2)
    F[0:(N - 2)] += f21
    F[1:(N - 1)] -= (f21 + f23)
    F[2:N]        += f23
    return k_b * F


def chain_tension(X, k_t, rest_length):
    N = X.shape[0]
    F = np.zeros((N, 3))
    r = X[:-1] - X[1:]
    r_norm = np.linalg.norm(r, axis=1).reshape(-1, 1)
    r_unit = r / r_norm
    Fm = k_t * (r_norm - rest_length)
    f_node = r_unit * Fm
    F[:-1] -= f_node
    F[1:]  += f_node
    return F


def velocity_update(position, k_b, k_t, rest_length, hignn_model):
    # update HIGNN with new coords
    hignn_model.update_coord(position[:, :3].astype(np.float32))
    # compute forces
    F = np.zeros_like(position)
    F[:, 2] = -1.0
    n_chain = 31
    num_chains = position.shape[0] // n_chain
    for i in range(num_chains):
        seg = slice(i*n_chain, (i+1)*n_chain)
        F[seg] += chain_bending(position[seg], k_b)
        F[seg] += chain_tension(position[seg], k_t, rest_length)
    # HIGNN computes velocity = K^{-1} F
    velocity = np.zeros_like(position, dtype=np.float32)
    hignn_model.dot(velocity, F.astype(np.float32))
    return velocity

# --- Parameters for data collection & embedding ---
NUM_TRAJ    = 20      # number of simulated trajectories
TRAJ_LEN    = 2000     # timesteps per trajectory
EMBED_DIM   = 4       # delay embedding depth
dt          = 0.001    # simulation timestep
k_t         = 100.0   # tension stiffness
tau         = 1       # embedding lag (in steps)
rest_length = 2.4
max_kb      = 500.0    # max bending stiffness for randomization

# Initialize grid of beads (same as in your simulation)
n_chain = 31
nx, ny, nz = 1, 1, n_chain
dx, dy, dz = 3.0, 3.0, 2.4
x = np.arange(0, nx*dx, dx)
y = np.arange(0, ny*dy, dy)
z = np.arange(0, nz*dz, dz)
zz, yy, xx = np.meshgrid(x, y, z)
X0 = np.concatenate([xx.reshape(-1,1), yy.reshape(-1,1), zz.reshape(-1,1)], axis=1).astype(np.float32)
X0 += np.random.rand(*X0.shape)*0.5

# MPI setup for potential parallel data generation
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

class SurrogateDataset(Dataset):
    def __init__(self, inputs, targets):
        self.X = torch.from_numpy(inputs).float()
        self.y = torch.from_numpy(targets).float()
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class SurrogateModel(nn.Module):
    def __init__(self, in_dim, out_dim, hiddens=[512,256]):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hiddens:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)


def build_dataset():
    all_inputs = []
    all_targets = []

    # 1) Init HIGNN once per rank
    hignn.Init()
    hmodel = hignn.HignnModel(X0, 15)
    hmodel.load_two_body_model('nn/two_body_unbounded')

    for traj in range(NUM_TRAJ):
        print(f"[Rank {rank}] Entering trajectory {traj}", flush=True)
        # random stiffness schedule
        kb_schedule = np.random.uniform(0, max_kb, TRAJ_LEN)
        # append this schedule as a new row in CSV
        with open('kb_schedules.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(kb_schedule.tolist())

        # initialize HIGNN
        # hignn.Init()
        # hmodel = hignn.HignnModel(X0, 15)
        # hmodel.load_two_body_model('nn/two_body_unbounded')

        # simulate
        X_seq = []
        V_seq = []
        X = X0.copy()
        for t in range(TRAJ_LEN):
            X_seq.append(X.copy())
            V = velocity_update(X, kb_schedule[t], k_t, rest_length, hmodel)
            V_seq.append(V.copy())
            X = X + dt*V
        # del hmodel; hignn.Finalize()

        # build delay embeddings
        for t in range((EMBED_DIM-1)*tau, TRAJ_LEN-1):
            hist = []
            for i in range(EMBED_DIM):
                hist.append(X_seq[t - i*tau].reshape(-1))
            inp = np.concatenate(hist + [ [kb_schedule[t]] ])
            tgt = V_seq[t].reshape(-1)
            all_inputs.append(inp)
            all_targets.append(tgt)
        print(f"[Rank {rank}] Exiting trajectory {traj}", flush=True)
    
    del hmodel; hignn.Finalize()
    
    # gather across MPI
    inputs = np.vstack(comm.allgather(np.array(all_inputs)))
    targets = np.vstack(comm.allgather(np.array(all_targets)))
    return inputs, targets


def train():
    inputs, targets = build_dataset()
    dataset = SurrogateDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    model = SurrogateModel(inputs.shape[1], targets.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn   = nn.MSELoss()
    for epoch in range(20):
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        print(f"Epoch {epoch+1}/20  Loss: {epoch_loss/len(dataset):.6f}")
    torch.save(model.state_dict(), 'surrogate_g_model.pth')

if __name__ == '__main__':
    train()
