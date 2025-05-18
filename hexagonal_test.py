import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Required for 3D plotting

def get_hexagonal_prism_polytope(side_length, z_min, z_max):
    """
    Generates the A, b matrices for a hexagonal prism Ax <= b.
    The hexagon lies in the XY plane, centered at the origin, with two sides parallel to the X-axis.
    The prism is extruded along the Z-axis.

    Args:
        side_length (float): The length of a side of the hexagon.
        z_min (float): The minimum z-coordinate of the prism.
        z_max (float): The maximum z-coordinate of the prism.

    Returns:
        tuple: (A_prism, b_prism)
            A_prism (np.ndarray): Matrix A of shape (8, 3).
            b_prism (np.ndarray): Vector b of shape (8,).
    """
    s = side_length
    sqrt3 = np.sqrt(3)

    # Inequalities for the hexagon in the XY plane (Ax_hex * [x, y]^T <= b_hex)
    # 1. y <= s*sqrt3/2
    # 2. y >= -s*sqrt3/2  => -y <= s*sqrt3/2
    # 3. sqrt3*x + y <= s*sqrt3
    # 4. sqrt3*x - y <= s*sqrt3
    # 5. -sqrt3*x + y <= s*sqrt3
    # 6. -sqrt3*x - y <= s*sqrt3
    A_hex_coeffs = np.array([
        [0, 1],
        [0, -1],
        [sqrt3, 1],
        [sqrt3, -1],
        [-sqrt3, 1],
        [-sqrt3, -1]
    ], dtype=float)

    b_hex_coeffs = np.array([
        s * sqrt3 / 2,
        s * sqrt3 / 2,
        s * sqrt3,
        s * sqrt3,
        s * sqrt3,
        s * sqrt3
    ], dtype=float)

    # Combine with Z-axis constraints for the prism
    A_prism = np.zeros((8, 3), dtype=float)
    A_prism[:6, :2] = A_hex_coeffs  # Hexagon constraints
    A_prism[6, 2] = 1               # z <= z_max
    A_prism[7, 2] = -1              # -z <= -z_min  (i.e., z >= z_min)

    b_prism = np.zeros(8, dtype=float)
    b_prism[:6] = b_hex_coeffs
    b_prism[6] = z_max
    b_prism[7] = -z_min
    
    return A_prism, b_prism

def generate_hexagonal_prism_points(num_points, side_length, z_min, z_max, seed=None):
    """
    Generates uniformly distributed random points within a hexagonal prism.

    Args:
        num_points (int): The number of points to generate.
        side_length (float): The length of a side of the hexagon.
        z_min (float): The minimum z-coordinate of the prism.
        z_max (float): The maximum z-coordinate of the prism.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        np.ndarray: Array of points of shape (num_points, 3).
    """
    if seed is not None:
        np.random.seed(seed)

    s = side_length
    h_hex = s * np.sqrt(3) / 2  # Max y-extent of the hexagon (half-height)

    # Get hexagon definition for point checking (Ax_hex * [x,y]^T <= b_hex)
    A_prism_def, b_prism_def = get_hexagonal_prism_polytope(side_length, z_min, z_max)
    A_hex_coeffs = A_prism_def[:6, :2]
    b_hex_coeffs = b_prism_def[:6]

    X_hist_prism = []
    
    # Rejection sampling parameters
    # Bounding box for sampling: x in [-s, s], y in [-h_hex, h_hex]
    # Theoretical acceptance rate: Area_hexagon / Area_bounding_rectangle_xy
    # Area_hexagon = (3 * sqrt(3) / 2) * s^2
    # Area_bounding_rectangle_xy = (2*s) * (2*h_hex) = (2*s) * (s*sqrt(3)) = 2*sqrt(3)*s^2
    # Acceptance rate = (1.5 * sqrt(3)) / (2 * sqrt(3)) = 1.5 / 2 = 0.75
    # So, expect to need num_points / 0.75 attempts.
    max_attempts_factor = 5 # Generous factor, e.g. 5 times the expected number of attempts
    max_total_attempts = int(num_points / 0.75 * max_attempts_factor) 
    current_attempts = 0

    while len(X_hist_prism) < num_points and current_attempts < max_total_attempts:
        current_attempts +=1
        # Sample (x,y) from the bounding box of the hexagon
        px = np.random.uniform(-s, s)
        py = np.random.uniform(-h_hex, h_hex)
        
        point_xy = np.array([px, py])
        is_in_hexagon = True
        # Check if the point (px, py) is inside the hexagon using Ax <= b
        for i in range(A_hex_coeffs.shape[0]):
            # Add a small epsilon for floating point comparisons
            if np.dot(A_hex_coeffs[i], point_xy) > b_hex_coeffs[i] + 1e-9:
                is_in_hexagon = False
                break
        
        if is_in_hexagon:
            # If in hexagon, sample z uniformly within the prism's height
            pz = np.random.uniform(z_min, z_max)
            X_hist_prism.append([px, py, pz])

    if len(X_hist_prism) < num_points:
        print(f"Warning: Generated {len(X_hist_prism)} points out of {num_points} requested after {max_total_attempts} attempts.")
        
    return np.array(X_hist_prism)

if __name__ == "__main__":
    # Parameters for the hexagonal prism
    NUM_POINTS = 5000
    SIDE_LENGTH = 1.0
    Z_MIN = 0.0
    Z_MAX = 2.0
    SEED = 42 # For reproducible results

    print(f"Generating {NUM_POINTS} points for a hexagonal prism with:")
    print(f"  Side length (s): {SIDE_LENGTH}")
    print(f"  Z range: [{Z_MIN}, {Z_MAX}]")
    print(f"  Seed: {SEED}")

    # Generate the points
    X_prism_points = generate_hexagonal_prism_points(
        NUM_POINTS, SIDE_LENGTH, Z_MIN, Z_MAX, seed=SEED
    )

    # Get the polytope definition (A, b) for this prism
    A_prism, b_prism = get_hexagonal_prism_polytope(
        SIDE_LENGTH, Z_MIN, Z_MAX
    )

    print(f"\nGenerated {X_prism_points.shape[0]} points.")
    print(f"Shape of X_prism_points (should be ({NUM_POINTS}, 3) or less if max attempts reached): {X_prism_points.shape}")
    if X_prism_points.shape[0] > 0:
        print("Sample point (first point):", np.round(X_prism_points[0], 3))

    print(f"\nPolytope definition A_prism (shape {A_prism.shape}), b_prism (shape {b_prism.shape}).")
    # print("A_prism:\n", np.round(A_prism,3))
    # print("b_prism:\n", np.round(b_prism,3))
    
    # Expected true bounds for this prism:
    # x: [-SIDE_LENGTH, SIDE_LENGTH] = [-1.0, 1.0]
    # y: [-SIDE_LENGTH*sqrt(3)/2, SIDE_LENGTH*sqrt(3)/2] approx [-0.866, 0.866]
    # z: [Z_MIN, Z_MAX] = [0.0, 2.0]
    print("\nExpected true axis-aligned bounds for this prism:")
    print(f"  x: [{-SIDE_LENGTH:.3f}, {SIDE_LENGTH:.3f}]")
    print(f"  y: [{-SIDE_LENGTH * np.sqrt(3) / 2:.3f}, {SIDE_LENGTH * np.sqrt(3) / 2:.3f}]")
    print(f"  z: [{Z_MIN:.3f}, {Z_MAX:.3f}]")


    print("\n--- How to use this data with your main.py script ---")
    print("1. Ensure this script (`hexagonal_test.py`) is in the same directory as `main.py` or accessible in PYTHONPATH.")
    print("2. In your `main.py` script, add the following import at the top:")
    print("   `from hexagonal_test import generate_hexagonal_prism_points, get_hexagonal_prism_polytope`")
    print("3. In the `main()` function of `main.py`, you can replace or bypass your existing data loading/simulation steps:")
    print("   ```python")
    print("   # --- Start: Added for hexagonal prism test ---")
    print("   # Parameters for the test prism")
    print(f"   SIDE_LENGTH_PRISM = {SIDE_LENGTH}")
    print(f"   Z_MIN_PRISM = {Z_MIN}")
    print(f"   Z_MAX_PRISM = {Z_MAX}")
    print(f"   NUM_POINTS_PRISM = {NUM_POINTS} # This will be 'm'")
    print(f"   SEED_PRISM = {SEED}")
    print("")
    print("   print('\\n--- Using Hexagonal Prism Test Data ---')")
    print("   # 1) & 4) Generate/load points (X_hist) directly")
    print("   # Instead of simulate_full_grid and filter_feasible_points")
    print("   X_hist = generate_hexagonal_prism_points(NUM_POINTS_PRISM, SIDE_LENGTH_PRISM, Z_MIN_PRISM, Z_MAX_PRISM, seed=SEED_PRISM)")
    print("   m = X_hist.shape[0]")
    print("   k = m  # All generated points are feasible by construction")
    print("   print(f'Total generated points for prism: {m}, Feasible: {k}')")
    print("")
    print("   # 2) Define branch thermal ratings (can be mocked or ignored if not used by subsequent steps under test)")
    print("   # rates = [...] # Not directly applicable here unless specific plots are desired")
    print("")
    print("   # 3) Build/load the polytope A x <= b")
    print("   A, b = get_hexagonal_prism_polytope(SIDE_LENGTH_PRISM, Z_MIN_PRISM, Z_MAX_PRISM)")
    print("   print(f'Using prism polytope A: {A.shape}, b: {b.shape}')")
    print("   # --- End: Added for hexagonal prism test ---")
    print("")
    print("   # Now, continue with step 5) from your main.py:")
    print("   # print('\\n5) Compute true axis-aligned bounds')")
    print("   # true_min, true_max = compute_true_bounds(A, b)")
    print("   # ... and so on for steps 6, 7, 8.")
    print("   ```")
    print("4. Run `main.py`. The `compute_true_bounds` function should yield the expected bounds printed above.")
    print("   The `find_best_rectangle` function should find a hyperrectangle whose bounds are very close to these true bounds,")
    print("   and it should cover a large percentage of the `NUM_POINTS_PRISM` points.")

    # Optional: Plot the generated points for visual verification
    if X_prism_points.shape[0] > 0 and X_prism_points.shape[1] == 3:
        print("\nPlotting generated points (close plot window to exit)...")
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_prism_points[:,0], X_prism_points[:,1], X_prism_points[:,2], s=5, alpha=0.3, label=f"{X_prism_points.shape[0]} points")
        
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")
        ax.set_title(f"Generated Points in a Hexagonal Prism")
        
        # Set axis limits to the true bounds for better visualization
        ax.set_xlim([-SIDE_LENGTH * 1.1, SIDE_LENGTH * 1.1])
        ax.set_ylim([-SIDE_LENGTH * np.sqrt(3) / 2 * 1.1, SIDE_LENGTH * np.sqrt(3) / 2 * 1.1])
        ax.set_zlim([Z_MIN - 0.1*(Z_MAX-Z_MIN) if Z_MAX > Z_MIN else Z_MIN - 0.1, 
                     Z_MAX + 0.1*(Z_MAX-Z_MIN) if Z_MAX > Z_MIN else Z_MAX + 0.1])
        ax.legend()
        ax.view_init(elev=25., azim=45) # Adjust viewing angle
        plt.tight_layout()
        plt.show()
