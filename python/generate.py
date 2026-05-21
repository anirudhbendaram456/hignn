import sys
import numpy as np
import scipy.spatial as KDTreed
import json
import time
import os

class ParticleSphericalCloud:
  """
  Class to generate the initial positions of a cloud of particles
  randomly placed within a spherical shell domain (between radii R1 and R2),
  while ensuring that the minimum distance between particles is satisfied.

  Parameters:
  -----------
  n_particle : int
               total number of particles to generate.
  min_dist : float
             Minimum allowed distance between any two particles.
  R1 : float
       Inner radius of the spherical domain.
  R2 : float
       Outer radius of the spherical domain.
  batch_size : int, optional
               Number of candidate particles to generate at each trial.
               Default is 10000 if n_particle > 10000, else 1000.
  """

  def __init__(self, n_particle, min_dist, R1, R2, batch_size=None):
    self.n_particle = n_particle
    self.min_dist = min_dist
    self.R1 = R1
    self.R2 = R2
    self.batch_size = batch_size if batch_size else (10000 if n_particle > 10000 else 1000)
    self.positions = None  # will be filled after generate()
  
  def find_over_proximity(self, X):
    """
    Find particles that are too close to each other.

    Parameters:
    ----------
    X : np.ndarray
        Array of candidate particle positions with shape (N, 3).

    Returns:
    -------
    idx : np.ndarray
          Indices of particles whose nearest neighbor is closer than min_dist.
    """
    tree = KDTreed.KDTree(X)
    d, _ = tree.query(X, k=2)
    idx = np.where(d[:, 1] > self.min_dist)[0]
    return idx
  
  def generate_status(self, n_generated, n_total, object_name="particle"):
    """
    Displays a progress bar showing the status of generation.

    Parameters:
    ----------
    n_generated : int
                  The number of generated objects so far.

    n_total : int
              The total number of objects to be generated.
              
    object_name: string
                 The name of object.

    Returns:
    -------
    None (this function prints the progress bar to the console and does not return any value).
    """
    progress = n_generated / n_total
    bar_length = 50
    filled_length = int(bar_length * progress)
    bar = '█' * filled_length + ' ' * (bar_length - filled_length)
    print(f"\r  Generate status: |{bar}| {n_generated}/{n_total} {object_name}s ({progress*100:.2f}%)", end='', flush=True)

  def generate(self):
    """
    Generates a cloud of particles randomly placed within a spherical domain, 
    ensuring that they are between two radii (R1 and R2) and respect a minimum distance.

    Parameters:
    -----------
    None (uses class instance variables `n_particle`, `batch_size`, `R1`, `R2`).

    Returns:
    --------
    None (the generated particle positions are stored in the `positions` attribute).
    """
    X = np.zeros((self.n_particle, 3), dtype=np.float32)
    
    print(f"  Generating particles...")
    start_time = time.time()
    
    n = 0
    
    while n < self.n_particle:
      # Generate candidate particles
      X_candidate = 2 * np.random.rand(self.batch_size * 3) * self.R2 - self.R2
      X_candidate = X_candidate.reshape((self.batch_size, 3))
      r = np.linalg.norm(X_candidate, axis=1)

      # Keep only points between R1 and R2
      mask = (r < self.R2) & (r > self.R1)
      X_candidate = X_candidate[mask]

      # Remove candidates too close to each other
      idx = self.find_over_proximity(X_candidate)
      X_candidate = X_candidate[idx]

      # How many can we accept
      selected_num = min(self.n_particle - n, X_candidate.shape[0])
      X[n:n + selected_num] = X_candidate[:selected_num]
      n += selected_num

      # Show generate status
      self.generate_status(n, self.n_particle)

    end_time = time.time()
    self.positions = X

    print(f"\n  Elapsed time: {end_time - start_time:.2f} seconds")
    print("  Successfully generated particles!\n")
    

class FilamentSphericalCloud(ParticleSphericalCloud):
  """
  Class to generate a cloud of filaments (chains of particles)
  randomly placed within a spherical shell domain (between radii R1 and R2), 
  ensuring a minimum distance between particles and preventing intersections.

  Parameters:
  -----------
  n_filament : int
               Number of filaments to generate.

  n_chain : int
            Number of particles in each filament (i.e., the number of particles per chain).

  min_dist : float
             Minimum allowed distance between any two particles to avoid overlap.

  R1 : float
       Inner radius of the spherical shell domain.

  R2 : float
       Outer radius of the spherical shell domain.

  batch_size : int, optional
               Number of candidate filaments to generate at each trial.
               Defaults to 1000 if n_filament <= 1000, otherwise 10000.  
  """

  def __init__(self, n_filament, n_chain, rest_length, min_dist, R1, R2, batch_size=None):
      super().__init__(n_filament, min_dist, R1, R2, batch_size)
      self.n_filament = n_filament
      self.n_chain = n_chain
      self.rest_length = rest_length
      self.len_filament = rest_length * (n_chain - 1)
      self.batch_size = batch_size if batch_size else (1000 if self.n_filament > 1000 else 100)
      self.positions = None
  
  def generate_fiber(self, Xc, chain):
    """
    Generate filament fibers based on given center points and a chain shape.

    Parameters
    ----------
    Xc : np.ndarray
         Array of center positions for the filaments, shape (n_filament, 3).
         
    chain : np.ndarray
            Array representing the particle locations along a normalized filament, shape (n_chain, 1).

    Returns
    -------
    X : np.ndarray
        Generated particle positions along all filaments, shape (n_filament * n_chain, 3).
    """
    n_filament = Xc.shape[0]
    n_chain = chain.shape[0]
    direction = np.random.randn(n_filament, 3)
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)

    X = np.zeros((n_chain * n_filament, 3), dtype=np.float32)

    for i in range(n_filament):
      X[i * n_chain:(i + 1) * n_chain, :] = Xc[i] + chain * direction[i]

    return X
  
  def remove_intersections(self, X_chain):
    """
    Remove filaments that intersect with each other from the provided chain of particles.

    Parameters:
    ----------
    X_chain : np.ndarray
              Array of particle positions forming filaments, with shape (N, 3), where N is the total number of particles in the filaments.

    Returns:
    -------
    X_chain[non_intersected_particle_idx]: np.ndarray
                                           The filtered array of particle positions with the intersecting filaments removed. 
                                           The shape remains (M, 3), where M is the number of particles after removing intersections.
    """
    tree = KDTreed.KDTree(X_chain)
    d, _ = tree.query(X_chain, k=2)
    intersected_chain_idx = np.unique(np.where(d[:, 1] < self.min_dist)[0] // self.n_chain)
    non_intersected_chain_idx = np.setdiff1d(
        np.arange(X_chain.shape[0] // self.n_chain),
        intersected_chain_idx
    )
    non_intersected_particle_idx = np.arange(X_chain.shape[0], dtype=np.int32).reshape(-1, self.n_chain)[non_intersected_chain_idx].flatten()
    return X_chain[non_intersected_particle_idx]
  
  def find_intersection(self, X0, X):
    """
    Find the indices of filaments in `X` that intersect with existing filaments in `X0`.

    Parameters:
    ----------
    X0 : np.ndarray
         Array of positions of particles in existing filaments, with shape (M, 3), 
         where M is the number of particles in the existing filaments.

    X : np.ndarray
        Array of candidate particle positions with shape (N, 3), 
        where N is the number of particles in the new filaments.

    Returns:
    -------
    np.unique(idx // self.n_chain): np.ndarray
                                    Indices of the filaments in `X` that intersect with any filament in `X0`. 
                                    The returned indices correspond to the filaments whose particles are closer than 
                                    `min_dist` to any of the particles in `X0`.
    """
    
    if X0.shape[0] == 0:
        return np.array([], dtype=np.int32)

    tree = KDTreed.KDTree(X0)
    d, _ = tree.query(X)
    idx = np.where(d < self.min_dist)[0]

    return np.unique(idx // self.n_chain)
  
  def generate(self):
    """
    Generate a cloud of filaments randomly placed within a spherical domain, 
    ensuring that they are between two radii (R1 and R2) and don't have intersections.

    Parameters:
    ----------
    None

    Returns:
    -------
    None (the generated filaments positions are stored in the `positions` attribute).
    """
    print(f"  Generating filaments...")

    start_time = time.time()
    X = np.zeros((self.n_filament * self.n_chain, 3), dtype=np.float32)

    chain = np.linspace(-0.5, 0.5, self.n_chain).reshape(-1, 1) * self.len_filament

    n = 0
    remaining_fibers = self.n_filament

    while n < self.n_filament:
      # Generate candidate centers
      X_candidate = 2 * np.random.rand(self.batch_size * 3) * self.R2 - self.R2
      X_candidate = X_candidate.reshape((self.batch_size, 3))
      r = np.linalg.norm(X_candidate, axis=1)
      mask = (r < self.R2 - self.len_filament/2) & (r > self.R1 + (self.len_filament/2 if self.R1 else 0))
      X_candidate = X_candidate[mask]

      # Remove too close centers
      idx = self.find_over_proximity(X_candidate)
      X_candidate = X_candidate[idx]

      # Generate fibers
      X_chain = self.generate_fiber(X_candidate, chain)

      # Remove self-intersections
      X_chain = self.remove_intersections(X_chain)

      # Remove intersections with existing fibers
      idx_chain = np.setdiff1d(
          np.arange(X_chain.shape[0] // self.n_chain, dtype=np.int32),
          self.find_intersection(X[:n*self.n_chain], X_chain).astype(np.int32)
      )
      idx_chain = idx_chain[:min(idx_chain.shape[0], remaining_fibers)]
      idx_particle = np.arange(X_chain.shape[0], dtype=np.int32).reshape(-1, self.n_chain)[idx_chain].flatten()
      X_chain = X_chain[idx_particle]

      # Save valid generated filaments
      m = idx_chain.shape[0]
      X[n * self.n_chain:(n + m) * self.n_chain] = X_chain
      remaining_fibers -= m
      n += m

      self.generate_status(n, self.n_filament, "filament")

    self.positions = X

    elapsed_time = time.time() - start_time
    print(f"\n  Elapsed time: {elapsed_time:.2f} seconds")
    print(f"  Successfully generated filaments!\n")
    
class FilamentParticleSphericalCloud:
  """
  Class to handle the generation of different cloud types:
  'filament_cloud', 'particle_cloud', or a combination of both ('filament_particle_cloud').

  Parameters:
  -----------
  filament_cloud : FilamentCloud, optional
                   The object that handles the filament cloud generation. 
                   Must be provided if 'cloud_type' is 'filament_cloud' or 'filament_particle_cloud'.
        
  particle_cloud : ParticleCloud, optional
                   The object that handles the particle cloud generation. 
                   Must be provided if 'cloud_type' is 'particle_cloud' or 'filament_particle_cloud'.
        
  cloud_type : str, optional, default="filament_particle_cloud"
               Type of cloud to generate. Can be one of:
                - 'filament_cloud': Only generates filament cloud.
                - 'particle_cloud': Only generates particle cloud.
                - 'filament_particle_cloud': Generates both filament and particle clouds.
  """
  
  def __init__(self, filament_cloud=None, particle_cloud=None, cloud_type="filament_particle_cloud"):
    self.filament_cloud = filament_cloud
    self.particle_cloud = particle_cloud
    self.cloud_type = cloud_type
    self.positions = None
    
    # Assert that at least one cloud type is provided
    assert filament_cloud is not None or particle_cloud is not None
    
  def generate(self):
    """
    Generates the cloud based on the specified `cloud_type`.
        
    Parameters:
    ----------
    None

    Returns:
    -------
    None (the generated positions for each cloud are stored in the `positions` attribute).
    """
    print(f"Cloud type: {self.cloud_type}")
    print("Generating cloud...\n")
    
    # Generate filament cloud
    if self.filament_cloud:
      self.filament_cloud.generate()  
    
    # Generate particle cloud 
    if self.particle_cloud:
      self.particle_cloud.generate()  

    if self.cloud_type=="filament_particle_cloud":
      self.positions = np.vstack((self.filament_cloud.positions, self.particle_cloud.positions))  
    else:
      self.positions = self.filament_cloud.positions if self.filament_cloud else self.particle_cloud.positions
      
    print("Successfully generated cloud!")
  
class CloudGenerator:
  """
  Class to parse config input file.

  Parameters:
  -----------
  config_filename : Filename of the input config file.
  """
  def __init__(self, config_filename):    
    # Load the parameters from JSON
    with open(f"{config_filename}", "r") as f:
      config = json.load(f)

    # cloud to generate
    cloud_params = config.get('cloud')
    if cloud_params:
      cloud_type = cloud_params.get('type')
    else:
      return
      
    # Extract cloud particle parameters
    particle_cloud_params = cloud_params.get('particle')
    cloud_seed = cloud_params.get('seed', 0)
    
    np.random.seed(cloud_seed)
      
    if particle_cloud_params:
      n_particle = particle_cloud_params['n_particle']       # Number of particles
      min_dist = particle_cloud_params['min_dist']           # Min distance between particles
      R1 = particle_cloud_params['R1']                       # Inner radius
      R2 = particle_cloud_params['R2']                       # Outer radius
      particle_cloud = ParticleSphericalCloud(n_particle, min_dist, R1, R2)
    else:
      particle_cloud = None
        
    # Extract filament cloud parameters
    filament_cloud_params = cloud_params.get('filament')
      
    if filament_cloud_params:
      n_filament = filament_cloud_params['n_filament']       # Number of filaments
      n_chain = filament_cloud_params['n_chain']             # Particles per filament
      rest_length = filament_cloud_params['rest_length']     # Rest length between particles
      min_dist = filament_cloud_params['min_dist']           # Min distance between particles
      R1 = filament_cloud_params['R1']                       # Inner radius
      R2 = filament_cloud_params['R2']                       # Outer radius
      filament_cloud = FilamentSphericalCloud(n_filament, n_chain, rest_length, min_dist, R1, R2)
    else:
      filament_cloud = None
      
    # Generate cloud
    cloud = FilamentParticleSphericalCloud(
      filament_cloud,
      particle_cloud,
      cloud_type
    )
        
    cloud.generate()

    dir_name = './python/cloud'
    file_name = os.path.join(dir_name, cloud_type + '.pos')

    # … after generating the first drop …
    positions1 = cloud.positions.copy()
    np.savetxt(os.path.join(dir_name, cloud_type + '_drop1.pos'), positions1)

    # regenerate for second drop
    np.random.seed(cloud_seed + 1)
    cloud.generate()
    positions2 = cloud.positions.copy()
    # this line shifts the second drop by the diameter (2×175)
    positions2 += np.array([0, 0, 180], dtype=np.float32)

    np.savetxt(os.path.join(dir_name, cloud_type + '_drop2.pos'), positions2)

    # now merge for the simulator
    merged = np.vstack((positions1, positions2))
    np.savetxt(file_name, merged)

    # os.makedirs(dir_name, exist_ok=True)
    # np.savetxt(file_name, cloud.positions)