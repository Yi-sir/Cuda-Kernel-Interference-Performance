import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import logging

import torch

from green_context.green_context import get_all_streams
from triton_example.triton_kernel import vector_add

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    device = torch.device("cuda:7")
    torch.set_default_device(device)

    a = torch.rand((512,))
    b = torch.rand((512,))

    streams = get_all_streams(device)

    for stream, num_sm in streams:
        with torch.cuda.stream(stream):
            c = vector_add(a, b)
            assert bool(torch.all(torch.isclose(a + b, c, rtol=1e-5, atol=1e-8)).item())
