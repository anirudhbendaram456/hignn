import numpy as np
import argparse, os
from numpy.lib.format import open_memmap  # <— add this import

def gen_triplets(N, L, dmin, dmax=None, batch=200000, seed=0):
    rng = np.random.default_rng(seed)
    out = open_memmap('sample_3cir_box32.npy',  # <— writes a valid .npy
                      dtype=np.float32, mode='w+', shape=(N, 9))
    written = 0
    while written < N:
        m = min(batch, N - written)
        pts = rng.uniform(-L/2, L/2, size=(m, 3, 3))
        r12 = np.linalg.norm(pts[:,0]-pts[:,1], axis=1)
        r13 = np.linalg.norm(pts[:,0]-pts[:,2], axis=1)
        r23 = np.linalg.norm(pts[:,1]-pts[:,2], axis=1)
        mask = (r12 >= dmin) & (r13 >= dmin) & (r23 >= dmin)
        if dmax is not None:
            mask &= (r12 <= dmax) & (r13 <= dmax) & (r23 <= dmax)
        acc = pts[mask]
        k = acc.shape[0]
        if k:
            out[written:written+k] = acc.reshape(k, 9).astype(np.float32)
            written += k
    out.flush()
    return os.path.abspath('sample_3cir_box32.npy')

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=1_000_000)
    ap.add_argument("--L", type=float, default=32.0)
    ap.add_argument("--a", type=float, default=1.0)
    ap.add_argument("--dmin", type=float, default=2.05)     # slightly above 2a to prevent near overlaps
    ap.add_argument("--dmax", type=float, default=None)     # optionally cap distances to keep interactions relevant
    ap.add_argument("--batch", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    path = gen_triplets(args.N, args.L, args.dmin, args.dmax, args.batch, args.seed)
    print("Wrote", args.N, "triplets to", path)
