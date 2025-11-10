import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import logging

import torch

from mha.kernel_flashinfer_mha import FlashinferMHADecode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    device = torch.device("cuda:0")

    flashinfer_mha = FlashinferMHADecode(device)

    flashinfer_mha.prepare_input()

    t = flashinfer_mha.profile_kernel_us()

    logger.info(f"Flashinfer MHA Decode costs {t:.3f}us")