import os
import sys
import ast
import torch
import datetime
import argparse
import setproctitle, socket, uuid

from .technical import my_gpu_info 
from ..ABL_arguments import args_short


def read_args():
    parser = argparse.ArgumentParser(description="Execute script with dynamic parameters.")
    parser.add_argument("--GPU", type=str, default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pretrained", action="store_true") # whether to use pretraining
    parser.add_argument("--c", action="store_true") # whether to use training supervisions
    script_args = parser.parse_args()

    args = args_short    
    args.GPU_ID = script_args.GPU
    args.seed = script_args.seed
    args.pretrained = script_args.pretrained
    args.c = script_args.c
    
    return args


def setup_environment(args):

    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU_ID
    my_gpu_info()
    
    # set job name
    setproctitle.setproctitle(
        "{}_{}_{}".format(
            args.model,
            args.buffer_size if "buffer_size" in args else 0,
            args.dataset,
        )
    )
    

# --- Signal Handler ---
def sigint_handler(signum, frame):
    print("\nSIGINT received. Releasing CUDA memory and exiting...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    sys.exit(0)