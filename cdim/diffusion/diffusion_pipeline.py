import torch
from tqdm import tqdm

from cdim.image_utils import randn_tensor, trace_AAt, estimate_variance, save_to_image, compute_operator_distance
from cdim.discrete_kl_loss import discrete_kl_loss
from cdim.eta_utils import calculate_best_step_size, initial_guess_step_size


def compute_kl_gaussian(residuals, sigma):
    # Only 0 centered for now
    if sigma == 0:
        raise ValueError("Can't do KL Divergence when sigma is 0")
    sample_mean = (residuals).mean()
    sample_var = (((residuals - sample_mean) **2).mean())
    kl_div = torch.log(sample_var**0.5 / sigma) + (sigma**2 + sample_mean**2) / (2*sample_var) - 0.5
    print(f"KL Divergence {kl_div}")
    return kl_div


@torch.no_grad()
def run_diffusion(
        model,
        scheduler,
        noisy_observation,
        operator,
        noise_function,
        device,
        stopping_sigma,
        num_inference_steps: int = 1000,
        K=5,
        image_dim=256,
        image_channels=3,
        model_type="diffusers",
        original_image=None
    ):
    batch_size = noisy_observation.shape[0]
    image_shape = (batch_size, image_channels, image_dim, image_dim)
    image = randn_tensor(image_shape, device=device)

    scheduler.set_timesteps(num_inference_steps, device=device)
    t_skip = scheduler.timesteps[0] - scheduler.timesteps[1]

    data = []
    TOTAL_UPDATE_STEPS = 0
    trace = trace_AAt(operator)
    for i, t in tqdm(enumerate(scheduler.timesteps), total=len(scheduler.timesteps), desc="Processing timesteps"):
        # Using GT image noised for now       
        # image = original_image * scheduler.alphas_cumprod[t] ** 0.5 + torch.randn_like(original_image) * (1 - scheduler.alphas_cumprod[t]) ** 0.5

         # 1. predict noise model_output
        model_output = model(image, t.unsqueeze(0).to(device))
        model_output = model_output.sample if model_type == "diffusers" else model_output[:, :3]

        # Single time
        # save_to_image(image, f"intermediates/{t}_xt.png")

        # 2. compute previous image: x_t -> x_t-1
        image = scheduler.step(model_output, t, image).prev_sample
        image.requires_grad_()
        alpha_prod_t_prev = scheduler.alphas_cumprod[t-t_skip] if t-t_skip >= 0 else 1
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        k = 0
        while k < K:
            if t <= 0: break
            a = scheduler.alphas_cumprod[t-t_skip]**0.5 - 1
            # For inpainting, use the number of observed pixels
            num_elements = operator.get_num_observed() if hasattr(operator, 'get_num_observed') else noisy_observation.numel()
            target_distance = (a**2 * torch.linalg.norm(noisy_observation)**2 + (1 - scheduler.alphas_cumprod[t-t_skip]) * trace).item()
            target_distance += num_elements * noise_function.sigma**2*(1-a**2)
            actual_distance = compute_operator_distance(operator, image, noisy_observation, squared=True).item()            
            variance = estimate_variance(
                operator,
                noisy_observation,
                scheduler.alphas_cumprod[t-t_skip],
                image.shape,
                trace=trace,
                sigma_y=noise_function.sigma,
                n_trace_samples=64,
                n_y_samples=64,
                device=image.device)

            # Can set a hard boundary here
            # if t-t_skip == 0: target_distance = 0
            # correction = torch.sqrt(1 - scheduler.alphas_cumprod[t-t_skip]) / scheduler.alphas_cumprod[t-t_skip]

            # target_distance -= 2.0 * correction * num_elements / image.numel()

            threshold = stopping_sigma * variance**0.5
            print(f"Target Distance mean {target_distance} max {target_distance + threshold} actual distance {actual_distance}")
            if actual_distance <= target_distance + threshold:
                break


            with torch.enable_grad():
                # Calculate x^hat_0
                model_output = model(image, (t - t_skip).unsqueeze(0).to(device))
                model_output = model_output.sample if model_type == "diffusers" else model_output[:, :3]
                x_0 = (image - beta_prod_t_prev ** (0.5) * model_output) / alpha_prod_t_prev ** (0.5)

                # save_to_image(x_0, f"intermediates/{t}_x0.png")
                loss = compute_operator_distance(operator, x_0, noisy_observation, squared=True).mean()

                print(f"L2 loss {compute_operator_distance(operator, x_0, noisy_observation, squared=False)}")
                data.append((t.item(), compute_operator_distance(operator, image, noisy_observation, squared=False).item()))
                loss.backward()

            initial_step_size = initial_guess_step_size(t.item(), torch.linalg.norm(image.grad)) # eta_scheduler.get_step_size(str(t.item()), torch.linalg.norm(image.grad))
            with torch.no_grad():
                # Set debug=True to see detailed step size search information
                step_size = calculate_best_step_size(image, noisy_observation, operator, image.grad, target_distance, threshold, initial_step_size, debug=False)

            print(f"Step Size {step_size:.6e} initial guess {initial_step_size:.6e}")

            image -= step_size * image.grad
            new_distance = compute_operator_distance(operator, image, noisy_observation, squared=True).item()
            print(f"New distance {new_distance}")
            image = image.detach().requires_grad_()
            TOTAL_UPDATE_STEPS += 1

            if step_size <= 1e-12: break

            # with torch.no_grad():
            #     model_output = model(image, (t - t_skip).unsqueeze(0).to(device))
            #     model_output = model_output.sample if model_type == "diffusers" else model_output[:, :3]
            #     x_0 = (image - beta_prod_t_prev ** (0.5) * model_output) / alpha_prod_t_prev ** (0.5)
            #     print(f"L2 loss After {compute_operator_distance(operator, x_0, noisy_observation, squared=False)}")

            k += 1

            # Check here because threshold is stochastic and can change from iteration to iteration
            if new_distance <= target_distance + threshold:
                break

        print("Step", t.item())
        # Use num_elements for proper normalization with inpainting
        num_elements = operator.get_num_observed() if hasattr(operator, 'get_num_observed') else noisy_observation.numel()
        print("Distance", 1 / num_elements * compute_operator_distance(operator, image, noisy_observation, squared=True).item())
        if hasattr(operator, 'select'):
            # Compute MAE over observed pixels only
            Ax = operator.select(image).flatten()
            y_selected = operator.select(noisy_observation).flatten()
            print("MAE", (torch.abs(Ax - y_selected).mean().item()))
        else:
            print("MAE", (torch.abs(operator(image) - noisy_observation).mean().item()))

    print(f"TOTAL_UPDATE_STEPS {TOTAL_UPDATE_STEPS}")
    return image
