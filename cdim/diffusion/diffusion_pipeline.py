import torch
from tqdm import tqdm

from cdim.image_utils import randn_tensor


@torch.no_grad()
def run_diffusion(
        model,
        scheduler,
        noisy_observation,
        operator,
        noise_function,
        device,
        num_inference_steps: int = 1000,
        K=5,
        image_dim=256,
        image_channels=3,
        model_type="diffusers"
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

                distance = operator(x_0) - noisy_observation
                if (distance ** 2).mean() < noise_function.sigma ** 2:
                    break
                loss = ((distance) ** 2).mean()
                print(loss.mean())
                loss.mean().backward()

            image -= 15 / torch.linalg.norm(image.grad) * image.grad

    return image