import os
import sys
import torch
import datetime
import argparse
import setproctitle, socket, uuid

from .check_gpu import my_gpu_info 
from ..arguments_disj import args_dpl_disj
from ..arguments_joint import args_dpl_joint


def read_args():
    parser = argparse.ArgumentParser(description="Execute notebooks with dynamic parameters.")
    parser.add_argument("--model_parameter_name", type=str, default="dpl")
    parser.add_argument("--uns_parameter_percentage", type=float, default=1.0,
            help="Percentage for the unsupervised parameter (float value)")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--Apretrained", action="store_true") 
    parser.add_argument("--c", action="store_true")
    parser.add_argument("--which_model", type=str, default="joint", choices=["joint", "disj"],
            help="Choose between 'joint' (boiadpl) or 'disj' (bddoiadpldisj) models")
    parser.add_argument("--GPU", type=str, default="0")
    parser.add_argument("--seed", type=int, default=0)
    script_args = parser.parse_args()

    if script_args.which_model == "joint":
        args = args_dpl_joint
    elif script_args.which_model == "disj":
        args = args_dpl_disj
    else:
        raise ValueError("Invalid model choice. Choose 'joint' or 'disj'.")

    args.model_parameter_name = script_args.model_parameter_name
    args.uns_parameter_percentage = script_args.uns_parameter_percentage
    args.pretrained = script_args.pretrained
    args.Apretrained = script_args.Apretrained
    args.c = script_args.c
    args.GPU_ID = script_args.GPU
    args.seed = script_args.seed
    
    return args



def setup_environment(args):

    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU_ID
    my_gpu_info()
    
    # logging
    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()

    # set job name
    setproctitle.setproctitle(
        "{}_{}_{}".format(
            args.model,
            args.buffer_size if "buffer_size" in args else 0,
            args.dataset,
        )
    )

    # saving
    save_folder = "bddoia" 
    save_model_name = 'dpl'
    category = "joint" if args.model == "boiadpl" else "disj"
    save_paths = []
    store_folder = "baseline-"
    if args.pretrained:     store_folder += "pretrained-"
    elif args.Apretrained:  store_folder += "Apretrained-"
    elif args.c:            store_folder += "c-"
    else:                   store_folder += "plain-"

    save_path = os.path.join("..",
        "outputs++", 
        save_folder, 
        "baseline", 
        category,
        save_model_name,
        f"{store_folder}{args.uns_parameter_percentage}+c"
    )

    save_paths.append(save_path)

    print("Seed: " + str(args.seed))
    print(f"Save paths: {str(save_paths)}")
    args.save_path = save_path


# --- Signal Handler ---
def sigint_handler(signum, frame):
    print("\nSIGINT received. Releasing CUDA memory and exiting...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    sys.exit(0)