import json
import torch

class EtaScheduler:
    def __init__(self, method, task, T, K, loss_type,
                       noise_function, lambda_val=None):
        self.task = task
        self.T = T
        self.K = K
        self.loss_type = loss_type
        self.lambda_val = lambda_val
        self.method = method

        self.precomputed_etas = self._load_precomputed_etas()

        # Couldn't find expected gradnorm
        if not self.precomputed_etas and method == "expected_gradnorm":
            self.method = "gradnorm"
            print("Etas for this configuration not found. Switching to gradnorm.")


        # Precomputed gradients are only for gaussian noise
        if noise_function.name != "gaussian" and method == "expected_gradnorm":
            self.method = "gradnorm"
            print("Precomputed gradients are only for gaussian noise. Switching to gradnorm.")


        # Get the best lambda_val if it's not passed
        if self.lambda_val is None:
            if self.method == "expected_gradnorm":
                self.lambda_val = self.precomputed_etas["lambda"]
            else:
                self.lambda_val = self.best_guess_lambda()
            print(f"Using lambda {self.lambda_val}")

    def _load_precomputed_etas(self):
        steps_key = f"T{self.T}_K{self.K}"
        with open("cdim/etas.json") as f:
            all_etas = json.load(f)

        return all_etas.get(self.task, {}).get(self.loss_type, {}).get(steps_key, {})

    def get_step_size(self, t, grad_norm):
        """Use either precomputed expected gradnorm or gradnorm."""
        if self.method == "expected_gradnorm":
            step_size = self.lambda_val * 1 / self.precomputed_etas["etas"][t]
        else:
            step_size = self.lambda_val * 1 / grad_norm
        return step_size

    def best_guess_lambda(self):
        """Guess a lambda value if not provided. Based on trial and error"""
        total_steps = self.T * self.K

        # L2 tends to over optimize too aggressively, so the default lr is lower
        if self.loss_type == "kl":
            return 350 / total_steps
        elif self.loss_type == "l2":
            return 220 / total_steps
        else:
            raise ValueError(f"Please provide learning rate for loss type {self.loss_type}")

def initial_guess_step_size(T, grad_norm):
    best_guess_lambda = 220 / T
    return best_guess_lambda / grad_norm


# def calculate_best_step_size(image, y, operator, gradient, target_distance, initial_guess,
#                              max_iters=20, tol=1e-4, bracket_factor=1.4):
#     def compute_distance(eta):
#         x_new = image - eta * gradient
#         diff = operator(x_new) - y
#         return torch.linalg.norm(diff)**2

#     def objective(eta):
#         return torch.abs(compute_distance(eta) - target_distance)

#     # Try to bracket the root
#     eta_low = initial_guess / bracket_factor
#     eta_high = initial_guess * bracket_factor

#     for _ in range(10):
#         import pdb
#         pdb.set_trace()
#         dist_low = compute_distance(eta_low)
#         dist_high = compute_distance(eta_high)
#         if (dist_low - target_distance) * (dist_high - target_distance) < 0:
#             break
#         eta_low /= bracket_factor
#         eta_high *= bracket_factor
#     else:
#         # Fallback: brute-force line search over eta to minimize distance
#         best_eta = None
#         best_val = float('inf')
#         for eta in torch.linspace(0, initial_guess * 5, steps=100, device=image.device):
#             val = objective(eta)
#             # print(f"ETA {eta} distance {compute_distance(eta)}")
#             if val < best_val:
#                 best_val = val
#                 best_eta = eta
#         return best_eta.item()

#     # Binary search
#     for _ in range(max_iters):
#         eta_mid = (eta_low + eta_high) / 2
#         dist_mid = compute_distance(eta_mid)
#         error = dist_mid - target_distance

#         if abs(error) < tol:
#             return eta_mid

#         if (compute_distance(eta_low) - target_distance) * error < 0:
#             eta_high = eta_mid
#         else:
#             eta_low = eta_mid

#     return eta_mid


import torch
from cdim.image_utils import compute_operator_distance

def calculate_best_step_size(
    image: torch.Tensor,
    y: torch.Tensor,
    operator,
    gradient: torch.Tensor,
    target_distance: float,
    threshold: float,
    initial_guess: float,
    *,
    tol: float = 1e-4,
    max_iters: int = 50,
    debug: bool = False,
    distance_fn=None,
):
    """
    Find the smallest η ≥ 0 that makes  distance(x − η g)  ≈ target_distance + threshold.

    Uses a robust grid search followed by golden section search for fine-grained optimization.
    
    Note: For inpainting operators with 'select' method, distances are computed
    over only the observed pixels.
    
    Args:
        debug: If True, prints detailed search information
        distance_fn: Optional custom distance function. Should take (operator, x, y) and return
                     a scalar distance. If None, uses L2 distance (compute_operator_distance).
                     For Poisson noise, pass compute_pearson_energy to use Pearson residuals.
    """
    target_boundary = target_distance + threshold
    
    def distance(η: torch.Tensor) -> torch.Tensor:
        x_new = image - η * gradient
        if distance_fn is not None:
            return distance_fn(operator, x_new, y)
        return compute_operator_distance(operator, x_new, y, squared=True)

    def error(η):
        return distance(η) - target_boundary
    
    # Phase 1: Coarse grid search to find promising regions
    # Search from very small to larger step sizes
    # Allow step sizes larger than 1 if needed
    max_eta = initial_guess * 200.0 if initial_guess > 0 else 10.0
    
    # Create a logarithmically-spaced grid for better coverage of small values
    # This ensures we search finely near 0 and coarser at larger values
    # Start from 1e-12 to handle cases with very large gradients
    n_coarse = 100  # Increased for better resolution
    eta_grid = torch.cat([
        torch.tensor([0.0]),
        torch.logspace(-12, torch.log10(torch.tensor(max_eta)), n_coarse - 1)
    ]).to(image.device)
    
    if debug:
        print(f"[Step Size] Searching from {eta_grid[1]:.2e} to {eta_grid[-1]:.2e} ({len(eta_grid)} points)")
        print(f"[Step Size] Sample grid points: {[f'{x:.2e}' for x in eta_grid[1:11].tolist()]}")
    
    # Evaluate distances at all grid points
    distances = torch.tensor([distance(eta).item() for eta in eta_grid])
    errors = distances - target_boundary
    
    dist_at_zero = distances[0].item()
    
    # Strategy: Find the SMALLEST eta that gets us AT OR BELOW target_boundary
    # Only consider non-zero etas
    below_target_mask = distances[1:] <= target_boundary
    
    if below_target_mask.any():
        # Found etas that reach target - pick the SMALLEST one (most conservative)
        below_indices = torch.where(below_target_mask)[0] + 1  # +1 because we excluded index 0
        best_idx = below_indices[0].item()  # Smallest eta that reaches target
        best_eta = eta_grid[best_idx].item()
        best_distance = distances[best_idx].item()
        
        if debug:
            print(f"[Step Size] Coarse grid: found eta={best_eta:.2e} that reaches target")
            print(f"[Step Size] Distance: {best_distance:.2f} (target: {target_boundary:.2f}, under by {target_boundary - best_distance:.2f})")
    else:
        # No eta reaches target - find the one that gets closest (minimize distance to target)
        non_zero_distances = distances[1:]
        closest_idx = torch.argmin(torch.abs(non_zero_distances - target_boundary)) + 1
        best_idx = closest_idx
        best_eta = eta_grid[best_idx].item()
        best_distance = distances[best_idx].item()
        
        if debug:
            print(f"[Step Size] Coarse grid: cannot reach target, best eta={best_eta:.2e}")
            print(f"[Step Size] Distance: {best_distance:.2f} (target: {target_boundary:.2f}, over by {best_distance - target_boundary:.2f})")
    
    # Check if eta=0 is better (already at target)
    if dist_at_zero <= target_boundary:
        if debug:
            print(f"[Step Size] Distance at eta=0: {dist_at_zero:.2f} - already at/below target")
        return 0.0
    
    if debug:
        print(f"[Step Size] Distance at eta=0: {dist_at_zero:.2f} (need to step)")
    
    # Phase 1.5: Fine search around the best point found
    # If best_eta is not at the boundaries, do a fine search around it
    if best_idx > 0 and best_idx < len(eta_grid) - 1:
        eta_low_bound = eta_grid[best_idx - 1].item()
        eta_high_bound = eta_grid[best_idx + 1].item()
        
        # Create a very fine linear grid between the neighboring points
        fine_grid = torch.linspace(eta_low_bound, eta_high_bound, 50).to(image.device)
        fine_distances = torch.tensor([distance(eta).item() for eta in fine_grid])
        
        # Find the SMALLEST eta in fine grid that gets us AT OR BELOW target
        fine_below_mask = fine_distances <= target_boundary
        
        if fine_below_mask.any():
            # Found fine etas that reach target - pick the SMALLEST
            fine_below_indices = torch.where(fine_below_mask)[0]
            fine_best_idx = fine_below_indices[0].item()
            fine_best_eta = fine_grid[fine_best_idx].item()
            fine_best_distance = fine_distances[fine_best_idx].item()
            
            # Only update if this is better (smaller eta that still reaches target, or gets closer)
            if fine_best_distance <= target_boundary and (best_distance > target_boundary or fine_best_eta < best_eta):
                best_eta = fine_best_eta
                best_distance = fine_best_distance
                best_idx = len(eta_grid) + fine_best_idx
                
                if debug:
                    print(f"[Step Size] Fine grid: improved to eta={best_eta:.2e}, distance={best_distance:.2f} (under by {target_boundary - best_distance:.2f})")
        else:
            # No fine eta reaches target - find closest
            fine_best_idx = torch.argmin(torch.abs(fine_distances - target_boundary))
            fine_best_eta = fine_grid[fine_best_idx].item()
            fine_best_distance = fine_distances[fine_best_idx].item()
            
            # Only update if closer to target than current best
            if abs(fine_best_distance - target_boundary) < abs(best_distance - target_boundary):
                best_eta = fine_best_eta
                best_distance = fine_best_distance
                best_idx = len(eta_grid) + fine_best_idx
                
                if debug:
                    print(f"[Step Size] Fine grid: improved to eta={best_eta:.2e}, distance={best_distance:.2f} (over by {best_distance - target_boundary:.2f})")
        
        # Always update the grid for potential bracketing
        distances = torch.cat([distances, fine_distances])
        errors = torch.cat([errors, fine_distances - target_boundary])
        eta_grid = torch.cat([eta_grid, fine_grid])
    
    # If best_eta is 0 and we're already at or below target, return 0
    if best_eta == 0.0 and dist_at_zero <= target_boundary:
        if debug:
            print(f"[Step Size] Already at target, no step needed")
        return 0.0
    
    # If we've reached target, we can return (no need for golden section)
    if best_distance <= target_boundary:
        if debug:
            print(f"[Step Size] Reached target with eta={best_eta:.2e}, returning")
        return best_eta
    
    # Phase 2: Check for bracketing around the best point
    # Look for a sign change (crossing the target boundary)
    bracket_found = False
    eta_lo, eta_hi = None, None
    
    # Check neighbors of best point
    for i in range(len(eta_grid) - 1):
        if errors[i] * errors[i + 1] < 0:  # Sign change
            eta_lo, eta_hi = eta_grid[i].item(), eta_grid[i + 1].item()
            bracket_found = True
            break
    
    # Phase 3: Refine using golden section search
    # Only refine if we haven't reached target yet and have a valid bracket
    if best_eta > 0 and best_distance > target_boundary and best_idx > 0 and best_idx < len(eta_grid) - 1:
        # Golden section search to find the smallest eta that reaches target_boundary
        phi = (1 + 5**0.5) / 2  # Golden ratio
        resphi = 2 - phi
        
        # Search in a small window around best_eta
        a = eta_grid[max(0, best_idx - 1)].item()
        b = eta_grid[min(len(eta_grid) - 1, best_idx + 1)].item()
        
        # Make sure we have a valid interval
        if b - a < 1e-20:
            if debug:
                print(f"[Step Size] Interval too small for refinement, returning eta={best_eta:.2e}")
            return best_eta
        
        dist_a = distance(torch.tensor(a)).item()
        dist_b = distance(torch.tensor(b)).item()
        
        for _ in range(max_iters):
            if abs(b - a) < 1e-20:  # Extremely tight tolerance
                break
                
            # Golden section points
            x1 = a + resphi * (b - a)
            x2 = b - resphi * (b - a)
            
            dist_x1 = distance(torch.tensor(x1)).item()
            dist_x2 = distance(torch.tensor(x2)).item()
            
            # Priority: prefer points that reach target (dist <= target_boundary)
            # Among those, prefer smaller eta
            # If neither reaches, prefer closer to target
            
            x1_reaches = dist_x1 <= target_boundary
            x2_reaches = dist_x2 <= target_boundary
            
            if x1_reaches and not x2_reaches:
                # x1 reaches target, x2 doesn't -> prefer x1's half
                b = x2
                dist_b = dist_x2
                if x1 < best_eta or not (best_distance <= target_boundary):
                    best_eta = x1
                    best_distance = dist_x1
            elif x2_reaches and not x1_reaches:
                # x2 reaches target, x1 doesn't -> prefer x2's half
                a = x1
                dist_a = dist_x1
                if x2 < best_eta or not (best_distance <= target_boundary):
                    best_eta = x2
                    best_distance = dist_x2
            elif x1_reaches and x2_reaches:
                # Both reach target -> prefer smaller eta (which is x1)
                b = x2
                dist_b = dist_x2
                best_eta = x1
                best_distance = dist_x1
            else:
                # Neither reaches target -> prefer closer to target
                if abs(dist_x1 - target_boundary) < abs(dist_x2 - target_boundary):
                    b = x2
                    dist_b = dist_x2
                    if abs(dist_x1 - target_boundary) < abs(best_distance - target_boundary):
                        best_eta = x1
                        best_distance = dist_x1
                else:
                    a = x1
                    dist_a = dist_x1
                    if abs(dist_x2 - target_boundary) < abs(best_distance - target_boundary):
                        best_eta = x2
                        best_distance = dist_x2
        
        if debug:
            if best_distance <= target_boundary:
                print(f"[Step Size] Final: eta={best_eta:.2e}, distance={best_distance:.2f} (under by {target_boundary - best_distance:.2f})")
            else:
                print(f"[Step Size] Final: eta={best_eta:.2e}, distance={best_distance:.2f} (over by {best_distance - target_boundary:.2f})")
    else:
        if debug:
            print(f"[Step Size] No refinement needed, returning best: eta={best_eta:.2e}")
    
    return best_eta

