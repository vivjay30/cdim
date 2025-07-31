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

def calculate_best_step_size(
    image: torch.Tensor,
    y: torch.Tensor,
    operator,
    gradient: torch.Tensor,
    target_distance: float,
    initial_guess: float,
    *,
    tol: float = 1e-4,
    max_iters: int = 30,
):
    """
    Find η ≥ 0 that makes  ||A(x − η g) − y||²  ≈ target_distance.

    1.  If A is linear we solve the quadratic analytically.
    2.  If that has no positive real root we return the *minimiser*
        of the quadratic (closest possible distance, still ≥ target).
    3.  If A is non-linear we fall back to a safe 1-D Brent-style search.
    """
    # ---- analytic branch (A assumed linear) ----------------------------
    r = operator(image)   - y           # residual  r = A x − y
    s = operator(gradient)              # search-dir in meas. space  s = A g
    a = torch.sum(r * r)                # ||r||²
    b = torch.sum(r * s)                # rᵀ s
    c = torch.sum(s * s) + 1e-12        # ||s||²   (ε avoids div-by-zero)

    # Solve  c η² - 2 b η + (a - target) = 0
    disc = b * b - c * (a - target_distance)
    if disc >= 0:
        sqrt_disc = torch.sqrt(disc)
        roots = [(b - sqrt_disc) / c, (b + sqrt_disc) / c]
        roots = [η for η in roots if η >= 0]
        if roots:                        # pick the root closest to the guess
            return float(min(roots, key=lambda η: abs(η - initial_guess)))

    # No positive real root → take the minimiser η* = b / c (projects to ≥0)
    eta_star = max(b / c, torch.zeros(1, device=image.device))
    # If the function really is quadratic, eta_star already minimises it:
    if disc >= 0:                        # analytic branch but no +ve root
        return float(eta_star)

    # ---- generic fall-back (A non-linear) ------------------------------
    def distance(η: torch.Tensor) -> torch.Tensor:
        diff = operator(image - η * gradient) - y
        diff = diff.flatten()            # 1-D vector
        return torch.dot(diff, diff)

    def error(η):                        # signed error wrt target
        return distance(η) - target_distance

    # Initial bracket around the guess
    eta_lo, eta_hi = initial_guess * 0.5, initial_guess * 1.5
    best_eta, best_err = initial_guess, abs(error(torch.tensor(initial_guess)))

    # Expand until we either bracket a sign-change or hit max expansions
    for _ in range(20):
        err_lo, err_hi = error(torch.tensor(eta_lo)), error(torch.tensor(eta_hi))
        # keep track of the closest value seen so far
        for η, err in ((eta_lo, err_lo), (eta_hi, err_hi)):
            if abs(err) < best_err:
                best_err, best_eta = abs(err), η
        if err_lo * err_hi < 0:          # sign change ⇒ root is bracketed
            break
        eta_lo, eta_hi = eta_lo / 2, eta_hi * 2
    else:
        return float(best_eta)           # never bracketed ⇒ give best seen

    # Brent-style secant / bisection hybrid
    φ_lo, φ_hi = error(torch.tensor(eta_lo)), error(torch.tensor(eta_hi))
    for _ in range(max_iters):
        eta = eta_hi - φ_hi * (eta_hi - eta_lo) / (φ_hi - φ_lo + 1e-12)  # secant
        φ = error(torch.tensor(eta))

        if abs(φ) < tol:
            return float(eta)
        if φ * φ_lo < 0:                 # maintain bracket
            eta_hi, φ_hi = eta, φ
        else:
            eta_lo, φ_lo = eta, φ

        # fall back to bisection if secant stagnates
        if abs(eta_hi - eta_lo) < 1e-12:
            eta = 0.5 * (eta_lo + eta_hi)
            if abs(error(torch.tensor(eta))) < best_err:
                return float(eta)
            break

    # Ran out of iterations – return the best in-bracket estimate
    return float(0.5 * (eta_lo + eta_hi))

