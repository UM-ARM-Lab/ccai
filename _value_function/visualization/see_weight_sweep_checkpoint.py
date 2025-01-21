import pickle as pkl
from pathlib import Path
import pathlib
CCAI_PATH = pathlib.Path(__file__).resolve().parents[2]
fpath = pathlib.Path(f'{CCAI_PATH}/data')
import torch

checkpoint_path = fpath /'test'/'weight_sweep'/'checkpoint.pkl'

def print_checkpoint():
    """Loads and prints the contents of the checkpoint file."""
    if checkpoint_path.exists():
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pkl.load(f)
        print("Checkpoint Contents:")
        for key, value in checkpoint.items():
            print(f"{key}: {value}")
    else:
        print(f"Checkpoint file not found at {checkpoint_path}")

if __name__ == "__main__":
    print_checkpoint()