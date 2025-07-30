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


def calculate_best_step_size(image, y, operator, gradient, target_distance, initial_guess,
                             max_iters=20, tol=1e-4, bracket_factor=1.4):
    def compute_distance(eta):
        x_new = image - eta * gradient
        diff = operator(x_new) - y
        return torch.linalg.norm(diff)**2

    def objective(eta):
        return torch.abs(compute_distance(eta) - target_distance)

    # Try to bracket the root
    eta_low = initial_guess / bracket_factor
    eta_high = initial_guess * bracket_factor

    for _ in range(10):
        dist_low = compute_distance(eta_low)
        dist_high = compute_distance(eta_high)
        if (dist_low - target_distance) * (dist_high - target_distance) < 0:
            break
        eta_low /= bracket_factor
        eta_high *= bracket_factor
    else:
        # Fallback: brute-force line search over eta to minimize distance
        best_eta = None
        best_val = float('inf')
        for eta in torch.linspace(0, initial_guess * 5, steps=100, device=image.device):
            val = objective(eta)
            # print(f"ETA {eta} distance {compute_distance(eta)}")
            if val < best_val:
                best_val = val
                best_eta = eta
        return best_eta.item()

    # Binary search
    for _ in range(max_iters):
        eta_mid = (eta_low + eta_high) / 2
        dist_mid = compute_distance(eta_mid)
        error = dist_mid - target_distance

        if abs(error) < tol:
            return eta_mid

        if (compute_distance(eta_low) - target_distance) * error < 0:
            eta_high = eta_mid
        else:
            eta_low = eta_mid

    return eta_mid
