import argparse
import os
from datasets import load_dataset

DEFAULT_DOWNLOAD_PATH = "/data/hf_datasets"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download_path", type=str, default=DEFAULT_DOWNLOAD_PATH)
    parser.add_argument("--datasets", type=str, nargs="+", default=["flan2021", "t0", "niv2", "cot", "dialog"])
    args = parser.parse_args()
    return args


def main(args):
    os.makedirs(args.download_path, exist_ok=True)
    if "flan2021" in args.datasets:
        load_dataset("conceptofmind/flan2021_submix_original", cache_dir=args.download_path)
    if "t0" in args.datasets:
        load_dataset("conceptofmind/t0_submix_original", cache_dir=args.download_path)
    if "niv2" in args.datasets:
        load_dataset("conceptofmind/niv2_submix_original", cache_dir=args.download_path)
    if "cot" in args.datasets:
        load_dataset("conceptofmind/cot_submix_original", cache_dir=args.download_path)
    if "dialog" in args.datasets:
        load_dataset("conceptofmind/dialog_submix_original", cache_dir=args.download_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
