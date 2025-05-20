import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_hexagonal_prism_polytope(side_length, z_min, z_max):
    s = side_length
    sqrt3 = np.sqrt(3)

    # Define hexagon inequalities in XY plane
    A_hex = np.array([
        [0, 1],
        [0, -1],
        [sqrt3, 1],
        [sqrt3, -1],
        [-sqrt3, 1],
        [-sqrt3, -1]
    ], dtype=float)

    b_hex = np.array([
        s * sqrt3 / 2,
        s * sqrt3 / 2,
        s * sqrt3,
        s * sqrt3,
        s * sqrt3,
        s * sqrt3
    ], dtype=float)

    # Add Z constraints to form a prism
    A_prism = np.zeros((8, 3), dtype=float)
    A_prism[:6, :2] = A_hex
    A_prism[6, 2] = 1
    A_prism[7, 2] = -1

    b_prism = np.zeros(8, dtype=float)
    b_prism[:6] = b_hex
    b_prism[6] = z_max
    b_prism[7] = -z_min
    
    return A_prism, b_prism

def generate_hexagonal_prism_points(num_points, side_length, z_min, z_max, seed=None):
    if seed is not None:
        np.random.seed(seed)

    s = side_length
    h_hex = s * np.sqrt(3) / 2

    A_prism, b_prism = get_hexagonal_prism_polytope(side_length, z_min, z_max)
    A_hex = A_prism[:6, :2]
    b_hex = b_prism[:6]

    points = []
    max_attempts = int(num_points * 5 / 0.75)
    attempts = 0

    while len(points) < num_points and attempts < max_attempts:
        attempts += 1
        x = np.random.uniform(-s, s)
        y = np.random.uniform(-h_hex, h_hex)
        
        in_hexagon = True
        for i in range(A_hex.shape[0]):
            if np.dot(A_hex[i], [x, y]) > b_hex[i] + 1e-9:
                in_hexagon = False
                break
        
        if in_hexagon:
            z = np.random.uniform(z_min, z_max)
            points.append([x, y, z])

    if len(points) < num_points:
        print(f"Warning: Generated only {len(points)}/{num_points} points after {max_attempts} attempts")
        
    return np.array(points)