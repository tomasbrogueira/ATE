import numpy as np
import itertools
import matplotlib.pyplot as plt

from region_finder import (
    simulate_full_grid,simulate_full_grid_random_currents, build_A_b_from_Y, filter_feasible_points, calculate_axis_aligned_bounds,
    generate_rectangles_from_polytope, generate_random_rectangles, find_best_rectangle,
    filter_contained_rectangles, calculate_focused_bounds, generate_improved_rectangles
)
from hexagonal_test import generate_hexagonal_prism_points, get_hexagonal_prism_polytope
from plots import plot_time_series, plot_region
from constrain_builder import build_A_b_linf, build_A_b_ols

def kyte_grid(rates=[0.5, 1.0, 0.25, 0.75, 0.6], v_min=[-5.0, -5.0, -5.0, -5.0], v_max=[5.0, 5.0, 5.0, 5.0], n_points=5000, n_rectangles=500, 
              use_polytope_sampling=True, dims=[0, 1, 2], variance=0.5):
    """Generate and visualize a hyperrectangle that fits within feasible power flow region."""
    # Generate simulated grid data
    injections, branch_currents, Y, branch_list = simulate_full_grid_random_currents(m=n_points, seed=98, variance=variance)
    X_all = injections.T
    
    # Check for dimensions with low variation
    if dims[2] is not None and injections.shape[0] > dims[2]:
        flat_dim = np.abs(injections[dims[2], :].max() - injections[dims[2], :].min()) < 1e-6
        if flat_dim:
            print(f"\nWARNING: Dimension {dims[2]} is flat (no variation)")
            variations = [np.var(injections[i, :]) for i in range(injections.shape[0])]
            alt_dims = np.argsort(variations)[::-1][:3]
            print(f"Suggested dimensions with most variation: {alt_dims}")
    
    # Plot time series data
    plot_time_series(branch_currents, rates, "Branch Current", block=False)
    
    branch_currents_array = np.column_stack([branch_currents[key] for key in branch_currents.keys()])
    
    num_non_slack_buses = Y.shape[0]
    
    # Validate voltage limit vectors
    if len(v_min) != num_non_slack_buses or len(v_max) != num_non_slack_buses:
        raise ValueError(f"Voltage limit arrays must have length {num_non_slack_buses} "
                         f"(provided: v_min={len(v_min)}, v_max={len(v_max)})")
    
    # Build constraints with both current and voltage limits
    A, b = build_A_b_ols(
        X_all, branch_currents_array, rates, 
        Y_full=Y, v_min=v_min, v_max=v_max,
        already_reduced=True
    )
    X_hist, k = filter_feasible_points(injections, A, b)
    print(f"Simulated {X_all.shape[0]} points; {k} are feasible.")
    
    if k == 0:
        print("No feasible points found with the given rates.")
        return
    
    true_lo, true_hi = calculate_axis_aligned_bounds(A, b)
    focused_min, focused_max = calculate_focused_bounds(A, b, dims)

    X_sub = X_hist[:, dims]
    A_sub = A[:, dims]

    # Find candidate rectangles using different methods
    candidates = []
    if use_polytope_sampling:
        try:
            print("Generating rectangles from polytope...")
            I = np.eye(len(dims))
            A_box = np.vstack([A_sub, I, -I])
            
            if X_sub.shape[0] > 0:
                data_lo = X_sub.min(axis=0)
                data_hi = X_sub.max(axis=0)
            else:
                data_lo = focused_min
                data_hi = focused_max
            
            b_box = np.concatenate([b, data_hi, -data_lo])
            polytope_rects = generate_rectangles_from_polytope(
                A_box, b_box, n_rectangles=n_rectangles, dim=len(dims), random_state=98
            )
            candidates = filter_contained_rectangles(polytope_rects, A_sub, b)
            print(f"Found {len(candidates)} rectangles from polytope sampling")
        except ValueError as e:
            print(f"Polytope sampling error: {e}")
    
    if len(candidates) < 5:
        print("Generating improved rectangles...")
        improved_rects = generate_improved_rectangles(A_sub, b, X_sub, n_rectangles)
        improved_candidates = filter_contained_rectangles(improved_rects, A_sub, b)
        print(f"Found {len(improved_candidates)} valid improved rectangles")
        candidates.extend(improved_candidates)
    
    if len(candidates) < 5:
        print("Using historical sampling as fallback")
        random_rects = generate_random_rectangles(A_sub, b, n_rectangles=n_rectangles, X_hist=X_sub)
        random_candidates = filter_contained_rectangles(random_rects, A_sub, b)
        print(f"Found {len(random_candidates)} rectangles from historical sampling")
        candidates.extend(random_candidates)
    
    if not candidates:
        print("No rectangle candidates found.")
        return

    best_idx, (best_lo, best_hi), best_count = find_best_rectangle(X_sub, candidates)
    print(f"Best hyperrectangle covers {best_count}/{k} points.")

    lo_vis = np.where(np.isfinite(true_lo), true_lo, X_hist.min(axis=0))
    hi_vis = np.where(np.isfinite(true_hi), true_hi, X_hist.max(axis=0))
    lo_vis[dims] = best_lo
    hi_vis[dims] = best_hi

    # Plot 3D region with the best rectangle
    if X_hist.shape[1] >= 3 and len(dims) >= 3:
        # Create 3D subset from the original data
        X_3d = X_hist[:, dims[:3]]
        
        # Ensure the rectangle is truly 3D
        best_lo_3d = np.zeros(3) 
        best_hi_3d = np.zeros(3)
        
        for i in range(3):
            orig_dim_idx = dims.index(dims[:3][i]) if dims[:3][i] in dims else i
            if orig_dim_idx < len(best_lo):
                best_lo_3d[i] = best_lo[orig_dim_idx]
                best_hi_3d[i] = best_hi[orig_dim_idx]
        
        print("\nPreparing 3D plot...")
        plot_region(
            dims=(0, 1, 2),
            pts=X_3d,
            poly_eq=(A_sub, b),
            rects_from_pts=[(best_lo_3d, best_hi_3d)],
            labels={'poly_eq': 'Feasible region', 
                   'poly_hull': 'Points convex hull',
                   'rect_pts': 'Best rectangle'},
            view_elevation=30,
            view_azimuth=45,
            show=True,
            interactive=True,
            block=False,
            dim_labels=dims[:3]
        )
    else:
        # Fall back to 2D plot
        if X_hist.shape[1] >= 2 and len(dims) >= 2:
            plot_dims = tuple(dims[:2])
            plot_lo = best_lo[:2]
            plot_hi = best_hi[:2]
            plot_region(
                dims=plot_dims,
                pts=X_hist,
                poly_eq=(A, b),
                rects_from_pts=[(plot_lo, plot_hi)],
                labels={'poly_eq': 'Feasible region', 'poly_hull': 'Points convex hull', 
                        'rect_pts': 'Best rectangle'},
                show=True,
                block=False
            )

    # Plot 2D projections
    valid_dims = min(X_hist.shape[1], len(dims))
    valid_dims_indices = dims[:valid_dims]
    
    projection_pairs = list(itertools.combinations(valid_dims_indices, 2))
    for idx, pair in enumerate(projection_pairs):
        i, j = pair
        i_idx = valid_dims_indices.index(i)
        j_idx = valid_dims_indices.index(j)
        
        X2 = X_hist[:, [i, j]]
        lo2 = best_lo[[i_idx, j_idx]]
        hi2 = best_hi[[i_idx, j_idx]]
        
        # Project constraints to these two dimensions
        A_proj = A[:, [i, j]]

        is_last = (idx == len(projection_pairs) - 1)
        
        projection_labels = {
            'poly_eq': 'Feasible Region',
            'poly_hull': 'Feasible Points', 
            'rect_pts': 'Rectangle Projection'
        }
        
        plot_region(
            dims=(0, 1),
            pts=X2,
            poly_eq=(A_proj, b),
            rects_from_pts=[(lo2, hi2)],
            labels=projection_labels,
            show=True,
            block=is_last,
            dim_labels=[i, j]
        )
    
    # If no 2D projections were made, block to keep all windows open
    if not projection_pairs:
        plt.show(block=True)


def hexagonal_prism_grid(use_polytope_sampling=True, num_points=500, num_rectangles=500, seed=42):
    """Generate and visualize a hyperrectangle within a hexagonal prism."""
    side, z_min, z_max = 1.0, 0.0, 2.0
    data = generate_hexagonal_prism_points(num_points, side, z_min, z_max, seed=seed)

    A_p, b_p = get_hexagonal_prism_polytope(side, z_min, z_max)
    prism_bounds_min, prism_bounds_max = calculate_axis_aligned_bounds(A_p, b_p)

    candidates = []
    if use_polytope_sampling:
        print("Generating rectangles from polytope...")
        polytope_rects = generate_rectangles_from_polytope(
            A_p, b_p, n_rectangles=num_rectangles, dim=A_p.shape[1], random_state=seed
        )
        candidates = filter_contained_rectangles(polytope_rects, A_p, b_p)
        print(f"Found {len(candidates)} rectangles from polytope sampling")
    
    if len(candidates) < 5:
        print("Generating improved rectangles...")
        improved_rects = generate_improved_rectangles(A_p, b_p, data, num_rectangles)
        improved_candidates = filter_contained_rectangles(improved_rects, A_p, b_p)
        print(f"Found {len(improved_candidates)} valid improved rectangles")
        candidates.extend(improved_candidates)
    
    if len(candidates) < 5:
        print("Using historical sampling as fallback")
        random_rects = generate_random_rectangles(A_p, b_p, n_rectangles=num_rectangles, X_hist=data)
        random_candidates = filter_contained_rectangles(random_rects, A_p, b_p)
        print(f"Found {len(random_candidates)} rectangles from historical sampling")
        candidates.extend(random_candidates)

    if not candidates:
        print("No contained rectangles generated.")
        return

    best_rect_idx, (best_lower, best_upper), points_covered = find_best_rectangle(data, candidates)
    print(f"Best rectangle covers {points_covered}/{len(data)} points.")

    # Plot 3D visualization if applicable
    if data.shape[1] >= 3:
        plot_region(
            dims=tuple(range(min(3, data.shape[1]))), 
            pts=data,
            poly_eq=(A_p, b_p),
            rects_from_pts=[(best_lower, best_upper)],
            labels={'poly_eq': 'Hexagonal prism', 'poly_hull': 'Points convex hull', 
                    'rect_pts': 'Best rectangle'},
            show=True
        )
    
    # Plot 2D projections
    for projection_dims in itertools.combinations(range(data.shape[1]), 2):
        i, j = projection_dims
        projected_data = data[:, list(projection_dims)]
        proj_lower = best_lower[list(projection_dims)]
        proj_upper = best_upper[list(projection_dims)]
        
        # Project the polytope to this 2D subspace
        A_proj = A_p[:, [i, j]]
        
        # Simplified projection labels
        projection_labels = {
            'poly_eq': 'Hexagonal Prism Projection',
            'poly_hull': 'Feasible Points',
            'rect_pts': 'Rectangle Projection'
        }
        
        plot_region(
            dims=(0, 1),
            pts=projected_data,
            poly_eq=(A_proj, b_p),
            rects_from_pts=[(proj_lower, proj_upper)],
            labels=projection_labels,
            show=True,
            dim_labels=[i, j]
        )

    return best_lower, best_upper, points_covered


if __name__ == "__main__":
    # Run kyte_grid with different variance values
    for variance in [0.1, 0.5, 1.0, 10.0, 100.0, 1000.0]:
        print(f"Running kyte_grid with variance={variance}")
        kyte_grid(rates=[0.5, 1.0, 0.75, 40, 40], n_points=1000, n_rectangles=1000, variance=variance)