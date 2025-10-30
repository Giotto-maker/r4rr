import os
import sys
import torch
import datetime
import argparse
import setproctitle, socket, uuid

from ..arguments import args_dpl
from baseline_modules.utility_modules.check_gpu import my_gpu_info 


def read_args():
    parser = argparse.ArgumentParser(description="Execute notebooks with dynamic parameters.")
    parser.add_argument("--uns_parameter_percentage", type=float, default=1.0,
                        help="Percentage for the unsupervised parameter (float value)")
    parser.add_argument("--model_parameter_name", type=str, default="dpl")
    parser.add_argument("--GPU", type=str, default="0")
    parser.add_argument("--seed", type=int, default=0)
    script_args = parser.parse_args()

    args = args_dpl

    args.uns_parameter_percentage = script_args.uns_parameter_percentage
    args.model_parameter_name = script_args.model_parameter_name
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
    save_paths = []
    save_path = os.path.join("..",
        "NEW-outputs", 
        save_folder, 
        "my_models", 
        save_model_name,
        f"[R]-episodic-proto-net-pipeline-{args.uns_parameter_percentage}-PROVA",
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