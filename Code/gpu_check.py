"""Check that PyTorch can see a GPU before any training run."""

from __future__ import annotations

import sys


def main() -> int:
    try:
        import torch
    except ImportError:
        print("PyTorch is not installed in this environment.")
        return 1
    print(f"torch {torch.__version__}")
    print(f"cuda compiled {torch.version.cuda}")
    available = torch.cuda.is_available()
    print(f"cuda available {available}")
    if not available:
        print("No GPU is visible. Training will fall back to CPU.")
        return 0
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        print(f"gpu {index}: {properties.name} memory_gb {properties.total_memory / (1024 ** 3):.1f}")
    tensor = torch.zeros(1, device="cuda")
    print(f"cuda tensor ok device {tensor.device}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
