import torch
from tqdm import tqdm

from cdim.image_utils import randn_tensor, trace_AAt, estimate_variance, save_to_image
from cdim.discrete_kl_loss import discrete_kl_loss
from cdim.eta_scheduler import calculate_best_step_size


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
        loss_type="l2",
        original_image=None
    ):
    batch_size = noisy_observation.shape[0]
    image_shape = (batch_size, image_channels, image_dim, image_dim)
    image = randn_tensor(image_shape, device=device)

    scheduler.set_timesteps(num_inference_steps, device=device)
    t_skip = scheduler.timesteps[0] - scheduler.timesteps[1]

    data = []
    data2 = []
    data3 = []
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
            target_distance = ((scheduler.alphas_cumprod[t-t_skip]**0.5-1)**2 * torch.linalg.norm(noisy_observation)**2 + (1 - scheduler.alphas_cumprod[t-t_skip]) * trace).item()
            target_distance += noisy_observation.numel() * noise_function.sigma**2*(1-a**2)
            actual_distance = (torch.linalg.norm(operator(image) - noisy_observation) ** 2).item()            
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

            threshold = 0.5 * variance**0.5
            # print(variance_Axt_minus_y_sq(operator, noisy_observation, scheduler.alphas_cumprod[t-t_skip]))
            print(f"Target Distance mean {target_distance} max {target_distance + threshold} actual distance {actual_distance}")
            # print(((1 - scheduler.alphas_cumprod[t-t_skip])**0.5)/scheduler.alphas_cumprod[t-t_skip])
            if actual_distance <= target_distance + threshold:
                break


            with torch.enable_grad():
                # Calculate x^hat_0
                model_output = model(image, (t - t_skip).unsqueeze(0).to(device))
                model_output = model_output.sample if model_type == "diffusers" else model_output[:, :3]
                x_0 = (image - beta_prod_t_prev ** (0.5) * model_output) / alpha_prod_t_prev ** (0.5)

                # save_to_image(x_0, f"intermediates/{t}_x0.png")
                if loss_type == "l2" and noise_function.name == "gaussian":
                    distance = operator(x_0) - noisy_observation
                    # if (distance ** 2).mean() < noise_function.sigma ** 2:
                    #     break
                    loss = ((distance) ** 2).mean()
                    # print(f"L2 loss {torch.linalg.norm(x_0 - image).item()}")

                    print(f"L2 loss {torch.linalg.norm(operator(x_0) - noisy_observation)}")
                    # import pdb
                    # pdb.set_trace()
                    data.append((t.item(), torch.linalg.norm(operator(image) - noisy_observation).item()))
                    # a = scheduler.alphas_cumprod[t-t_skip]**0.5 - 1
                    # data2.append((t.item(), ((target_distance + noisy_observation.numel() * noise_function.sigma**2*(1-a**2))**0.5).item()))
                    # break
                    # import pdb
                    # pdb.set_trace()
                    # data.append((t.item(),  (torch.linalg.norm(operator(image) - noisy_observation)**2).item()))
                    # data2.append((t.item(), (2 * ((operator(image) - noisy_observation) * operator(x_0 - image)).sum()).item()))
                    # data3.append((t.item(), (torch.linalg.norm(operator(x_0 - image))**2).item()))
                    # data.append((t.item(), ((1 - scheduler.alphas_cumprod[t]) ** 0.5 / scheduler.alphas_cumprod[t]).item()))
                    # data2.append((t.item(), (scheduler.betas[t]).item()))
                    # data2.append((t.item(), (1 - scheduler.alphas_cumprod[t]).item()))
                    # data2.append((t.item(), torch.linalg.norm(model_output).item()))
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

            initial_guess_step_size = eta_scheduler.get_step_size(str(t.item()), torch.linalg.norm(image.grad))
            with torch.no_grad():
                step_size = calculate_best_step_size(
                    image, noisy_observation, operator,
                    image.grad, target_distance, threshold, initial_guess_step_size)
                
                # if step_size < initial_guess_step_size:
                #     print("HEEEEEEREEEEE")
                # step_size = initial_guess_step_size
                # step_size = min(step_size, initial_guess_step_size)
            print(f"Step Size {step_size} initial guess {initial_guess_step_size}")
            if step_size <= 0.0001: break
            image -= step_size * image.grad
            new_distance = torch.linalg.norm(operator(image) - noisy_observation).item() ** 2
            print(f"New distance {new_distance}")
            image = image.detach().requires_grad_()
            TOTAL_UPDATE_STEPS += 1

            # with torch.no_grad():
            #     model_output = model(image, (t - t_skip).unsqueeze(0).to(device))
            #     model_output = model_output.sample if model_type == "diffusers" else model_output[:, :3]
            #     x_0 = (image - beta_prod_t_prev ** (0.5) * model_output) / alpha_prod_t_prev ** (0.5)
            #     print(f"L2 loss After {torch.linalg.norm(operator(x_0) - noisy_observation)}")

            k += 1

            # Check here because threshold is stochastic and can change from iteration to iteration
            if new_distance <= target_distance + threshold:
                break

        print("Step", t.item())
        print("Distance", 1 / noisy_observation.numel() * (torch.linalg.norm(operator(image) - noisy_observation).item() **2))
        print("MAE", (torch.abs(operator(image) - noisy_observation).mean().item()))

    print(f"TOTAL_UPDATE_STEPS {TOTAL_UPDATE_STEPS}")
    return image
