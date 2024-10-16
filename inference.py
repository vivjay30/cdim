import argparse
import os
import yaml

from PIL import Image
import numpy as np
import torch

from cdim.noise import get_noise
from cdim.operators import get_operator
from cdim.image_utils import save_to_image


def load_image(path):
    """
    Load the image and normalize to [-1, 1]
    """
    original_image = Image.open(path)

    # Resize if needed
    original_image = np.array(original_image.resize((256, 256), Image.BICUBIC))
    original_image = torch.from_numpy(original_image).unsqueeze(0).permute(0, 3, 1, 2)
    return (original_image / 127.5 - 1.0).to(torch.float)
    

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main(args):
    device_str = f"cuda" if args.cuda and torch.cuda.is_available() else 'cpu'
    print(f"Using device {device_str}")
    device = torch.device(device_str) 

    os.makedirs(args.output_dir, exist_ok=True)
    original_image = load_image(args.input_image).to(device)

    # Load the noise function
    noise_config = load_yaml(args.noise_config)
    noise_function = get_noise(**noise_config)
    print(noise_function)    

    # Load the measurement function A
    operator_config = load_yaml(args.operator_config)
    operator_config["device"] = device
    operator = get_operator(**operator_config)
    print(operator)

    noisy_measurement = noise_function(operator(original_image))
    save_to_image(noisy_measurement, os.path.join(args.output_dir, "noisy_measurement.png"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_image", type=str)
    parser.add_argument("T", type=int)
    parser.add_argument("K", type=int)
    parser.add_argument("model", type=str)
    parser.add_argument("operator_config", type=str)
    parser.add_argument("noise_config", type=str)
    parser.add_argument("--output-dir", default=".", type=str)
    parser.add_argument("--cuda", default=True, action=argparse.BooleanOptionalAction)

    main(parser.parse_args())