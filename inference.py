import argparse
import os


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_image", type=str)
    parser.add_argument("T", type=int)
    parser.add_argument("K", type=int)
    parser.add_argument("model", type=str)
    parser.add_argument("operator_config", type=str)
    parser.add_argument("noise_config", type=str)
    parser.add_argument("--output-dir", default=".", type=str)
    main(parser.parse_args())