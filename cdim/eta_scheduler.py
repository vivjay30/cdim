import json

class EtaScheduler:
    def __init__(self, method, task, T, K, loss_type, lambda_val=None):
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

