import gradio as gr
import torch
import yaml
import numpy as np
from PIL import Image
from cdim.noise import get_noise
from cdim.operators import get_operator
from cdim.diffusion.scheduling_ddim import DDIMScheduler
from cdim.diffusion.diffusion_pipeline import run_diffusion
from diffusers import DiffusionPipeline


# Global variables for model and scheduler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
ddim_scheduler = None
model_type = None


def load_image(image_path):
    """Process input image to tensor format."""
    image = Image.open(image_path)
    original_image = np.array(image.resize((256, 256), Image.BICUBIC))
    original_image = torch.from_numpy(original_image).unsqueeze(0).permute(0, 3, 1, 2)
    return (original_image / 127.5 - 1.0).to(torch.float)[:, :3]


def load_yaml(file_path: str) -> dict:
    """Load configurations from a YAML file."""
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def convert_to_np(torch_image):
    return ((torch_image.detach().clamp(-1, 1).cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)


def generate_noisy_image(image_choice, noise_sigma, operator_key):
    """Generate the noisy image and store necessary data for restoration."""
    # Map image choice to path
    image_paths = {
        "CelebA HQ 1": "sample_images/celebhq_29999.jpg",
        "CelebA HQ 2": "sample_images/celebhq_00001.jpg",
        "CelebA HQ 3": "sample_images/celebhq_00000.jpg"
    }

    config_paths = {
        "Box Inpainting": "operator_configs/box_inpainting_config.yaml",
        "Random Inpainting": "operator_configs/random_inpainting_config.yaml", 
        "Super Resolution": "operator_configs/super_resolution_config.yaml",
        "Gaussian Deblur": "operator_configs/gaussian_blur_config.yaml"
    }

    image_path = image_paths[image_choice]
        
    # Load image and get noisy version
    original_image = load_image(image_path).to(device)
    noise_config = load_yaml("noise_configs/gaussian_noise_config.yaml")
    noise_config["sigma"] = noise_sigma
    noise_function = get_noise(**noise_config)
    operator_config = load_yaml(config_paths[operator_key])
    operator_config["device"] = device
    operator = get_operator(**operator_config)
        
    noisy_measurement = noise_function(operator(original_image))
    noisy_image = Image.fromarray(convert_to_np(noisy_measurement[0]))

    # Store necessary data for restoration
    data = {
        'noisy_measurement': noisy_measurement,
        'operator': operator,
        'noise_function': noise_function
    }

    return noisy_image, data  # Return the noisy image and data for restoration


def run_restoration(data, T, stopping_sigma):
    """Run the restoration process and return the restored image."""
    global model, ddim_scheduler, model_type

    # Extract stored data
    noisy_measurement = data['noisy_measurement']
    operator = data['operator']
    noise_function = data['noise_function']

    # Initialize model if not already done
    if model is None:
        model_type = "diffusers"
        model = DiffusionPipeline.from_pretrained("google/ddpm-celebahq-256").to(device).unet
            
        ddim_scheduler = DDIMScheduler(
            num_train_timesteps=1000, 
            beta_start=0.0001, 
            beta_end=0.02, 
            beta_schedule="linear"
        )

    # Run restoration
    output_image = run_diffusion(
        model, ddim_scheduler, noisy_measurement, operator, noise_function, device,
        stopping_sigma, num_inference_steps=T, model_type=model_type
    )
        
    # Convert output image for display
    output_image = Image.fromarray(convert_to_np(output_image[0]))
    return output_image


with gr.Blocks() as demo:
    gr.Markdown("# Noisy Image Restoration with Diffusion Models")
    
    with gr.Row():
        T = gr.Slider(10, 200, value=50, step=1, label="Number of Inference Steps (T)")
        stopping_sigma = gr.Slider(0.1, 5.0, value=0.1, step=0.1, label="Stopping Sigma (c)")
        noise_sigma = gr.Slider(0, 0.6, value=0.05, step=0.01, label="Noise Sigma")
    
    image_select = gr.Dropdown(
        choices=["CelebA HQ 1", "CelebA HQ 2", "CelebA HQ 3"],
        value="CelebA HQ 1",
        label="Select Input Image"
    )
    
    operator_select = gr.Dropdown(
        choices=["Box Inpainting", "Random Inpainting", "Super Resolution", "Gaussian Deblur"],
        value="Box Inpainting",
        label="Select Task"
    )
    
    run_button = gr.Button("Run Inference")
    noisy_image = gr.Image(label="Noisy Image")
    restored_image = gr.Image(label="Restored Image")
    state = gr.State()  # To store intermediate data

    # First function generates the noisy image and stores data
    run_button.click(
        fn=generate_noisy_image,
        inputs=[image_select, noise_sigma, operator_select],
        outputs=[noisy_image, state],
    ).then(
        fn=run_restoration,
        inputs=[state, T, stopping_sigma],
        outputs=restored_image
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
