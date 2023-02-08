import torch
import os

def main(params):
    RUNS = 1
    MX_ITER = 1000000000
    SAMPLE_PERCENTAGE = params.SAMPLE_PERCENTAGE
    DATASET = params.DATASET
    DHCN_LAYERS = params.DHCN_LAYERS
    CONV_SIZE = params.CONV_SIZE
    H_MIRROR = params.H_MIRROR
    USE_HARD_LABELS = params.USE_HARD_LABELS
    LR = 1e-3
    os.environ["CUDA_VISIBLE_DEVICES"] = params.GPU

    device = torch.device("cuda:0")
    print(torch.cuda.device_count())
