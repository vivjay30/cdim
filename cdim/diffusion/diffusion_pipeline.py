import torch
from tqdm import tqdm

from cdim.image_utils import randn_tensor
from cdim.discrete_kl_loss import discrete_kl_loss

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
        eta_scheduler,
        num_inference_steps: int = 1000,
        K=5,
        image_dim=256,
        image_channels=3,
        model_type="diffusers",
        loss_type="l2"
    ):
    batch_size = noisy_observation.shape[0]
    image_shape = (batch_size, image_channels, image_dim, image_dim)
    image = randn_tensor(image_shape, device=device)

    scheduler.set_timesteps(num_inference_steps, device=device)
    t_skip = scheduler.timesteps[0] - scheduler.timesteps[1]

    for i, t in tqdm(enumerate(scheduler.timesteps), total=len(scheduler.timesteps), desc="Processing timesteps"):
         # 1. predict noise model_output
        model_output = model(image, t.unsqueeze(0).to(device))
        model_output = model_output.sample if model_type == "diffusers" else model_output[:, :3]

        # 2. compute previous image: x_t -> x_t-1
        image = scheduler.step(model_output, t, image).prev_sample
        image.requires_grad_()
        alpha_prod_t_prev = scheduler.alphas_cumprod[t-t_skip] if t-t_skip >= 0 else 1
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        for j in range(K):
            if t <= 0: break

            with torch.enable_grad():
                # Calculate x^hat_0
                model_output = model(image, (t - t_skip).unsqueeze(0).to(device))
                model_output = model_output.sample if model_type == "diffusers" else model_output[:, :3]
                x_0 = (image - beta_prod_t_prev ** (0.5) * model_output) / alpha_prod_t_prev ** (0.5)

                if loss_type == "l2" and noise_function.name == "gaussian":
                    distance = operator(x_0) - noisy_observation
                    if (distance ** 2).mean() < noise_function.sigma ** 2:
                        break
                    loss = ((distance) ** 2).mean()
                    print(f"L2 loss {loss}")
                    loss.backward()

                elif loss_type == "kl" and noise_function.name == "gaussian":
                    diff = (operator(x_0) - noisy_observation)  # Residuals
                    kl_div = compute_kl_gaussian(diff, noise_function.sigma)
                    kl_div.backward()

                elif loss_type == "kl" and noise_function.name == "poisson":
                    residuals = (operator(x_0) * noise_function.rate - noisy_observation * noise_function.rate) * 127.5  # Residuals
                    x_0_pixel = operator((x_0 + 1) * 127.5)
                    mask = x_0_pixel > 2 # Avoid numeric issues with pixel values near 0
                    pearson = residuals[mask] / torch.sqrt(x_0_pixel[mask] * noise_function.rate)
                    pearson_flat = pearson.view(-1)
                    kl_div = compute_kl_gaussian(pearson_flat, 1.0)
                    kl_div.backward()

                elif loss_type == "categorical_kl" and noise_function.name == "bimodal":
                    diff = (operator(x_0) - noisy_observation)
                    indices = operator(torch.ones(image.shape).to(device))
                    diff = diff[indices > 0]  # Don't consider masked out pixels in the distribution
                    empirical_distribution = noise_function.sample_noise_distribution(image).to(device).view(-1)
                    loss = discrete_kl_loss(diff, empirical_distribution, num_bins=15)
                    print(f"Categorical KL {loss}")
                    loss.backward()

                else:
                    raise ValueError(f"Unsupported combination: loss {loss_type} noise {noise_function.name}")

            step_size = eta_scheduler.get_step_size(str(t.item()), torch.linalg.norm(image.grad))
            image -= step_size * image.grad
            image = image.detach().requires_grad_()

    return image
