import argparse
import os
import yaml
import time
from pathlib import Path

from PIL import Image
import numpy as np
import torch

from diffusers import DiffusionPipeline

from cdim.noise import get_noise
from cdim.operators import get_operator
from cdim.image_utils import save_to_image
from cdim.dps_model.dps_unet import create_model
from cdim.diffusion.scheduling_ddim import DDIMScheduler
from cdim.diffusion.diffusion_pipeline import run_diffusion
from cdim.eta_scheduler import EtaScheduler

torch.manual_seed(3)
np.random.seed(3)

def load_image(path):
    """
    Load the image and normalize to [-1, 1]
    """
    original_image = Image.open(path)

    # Resize if needed
    original_image = np.array(original_image.resize((256, 256), Image.BICUBIC))
    original_image = torch.from_numpy(original_image).unsqueeze(0).permute(0, 3, 1, 2)
    return (original_image / 127.5 - 1.0).to(torch.float)[:, :3]
    

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def process_image(image_path, output_dir, model, ddim_scheduler, operator, noise_function, 
                 device, eta_scheduler, args, model_type):
    """
    Process a single image with the given model and parameters
    """
    original_image = load_image(image_path).to(device)
    
    # Get the base filename without extension
    base_name = Path(image_path).stem
    
    noisy_measurement = noise_function(operator(original_image))
    save_to_image(noisy_measurement, os.path.join(output_dir, f"{base_name}_noisy_measurement.png"))

    t0 = time.time()
    output_image = run_diffusion(
        model, ddim_scheduler,
        noisy_measurement, operator, noise_function, device,
        eta_scheduler,
        num_inference_steps=args.T,
        K=args.K,
        model_type=model_type,
        loss_type=args.loss,
        original_image=original_image)
    print(f"Processing time for {base_name}: {time.time() - t0:.2f}s")

    save_to_image(output_image, os.path.join(output_dir, f"{base_name}_output.png"))


def main(args):
    device_str = f"cuda" if args.cuda and torch.cuda.is_available() else 'cpu'
    print(f"Using device {device_str}")
    device = torch.device(device_str) 

    os.makedirs(args.output_dir, exist_ok=True)

    # Load the noise function
    noise_config = load_yaml(args.noise_config)
    noise_function = get_noise(**noise_config)

    # Load the measurement function A
    operator_config = load_yaml(args.operator_config)
    operator_config["device"] = device
    operator = get_operator(**operator_config)

    if args.model_config.endswith(".yaml"):
        # Local model from DPS
        model_type = "dps"
        model_config = load_yaml(args.model_config)
        model = create_model(**model_config)
        model = model.to(device)
        model.eval()

    else:
        # Huggingface diffusers model
        model_type = "diffusers"
        model = DiffusionPipeline.from_pretrained(args.model_config).to(device).unet

    # All the models have the same scheduler.
    # you can change this for different models
    ddim_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        prediction_type="epsilon",
        timestep_spacing="leading",
        steps_offset=0,
    )

    eta_scheduler = EtaScheduler(args.eta_type, operator.name, args.T,
        args.K, args.loss, noise_function, args.lambda_val)

    # Process all images in the input directory
    input_dir = Path(args.input_folder)
    image_files = [f for f in input_dir.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
    image_files = sorted(image_files)
    
    print(f"Found {len(image_files)} images to process")
    
    for image_file in image_files:
        print(f"Processing {image_file.name}...")
        process_image(
            str(image_file), args.output_dir, model, ddim_scheduler,
            operator, noise_function, device, eta_scheduler, args, model_type
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder", type=str, help="Folder containing input images")
    parser.add_argument("T", type=int)
    parser.add_argument("K", type=int)
    parser.add_argument("operator_config", type=str)
    parser.add_argument("noise_config", type=str)
    parser.add_argument("model_config", type=str)
    parser.add_argument("--eta-type", type=str,
        choices=['gradnorm', 'expected_gradnorm'],
        default='expected_gradnorm')
    parser.add_argument("--lambda-val", type=float,
        default=None, help="Constant to scale learning rate. Leave empty to use a heuristic best guess.")
    parser.add_argument("--output-dir", default=".", type=str)
    parser.add_argument("--loss", type=str,
        choices=['l2', 'kl', 'categorical_kl'], default='l2',
        help="Algorithm to use. Options: 'l2', 'kl', 'categorical_kl'. Default is 'l2'."
    )
    parser.add_argument("--cuda", default=True, action=argparse.BooleanOptionalAction)

    main(parser.parse_args())