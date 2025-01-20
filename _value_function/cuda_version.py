import torch
import sys

print(torch.version.cuda)   # e.g. '11.7'

# Print Python version
print(f"Python version: {sys.version}")

# Print PyTorch version
print(f"PyTorch version: {torch.__version__}")