import os
import sys
import ast
import torch
import datetime
import argparse
import setproctitle, socket, uuid

from .check_gpu import my_gpu_info 
from ..arguments import args_sl, args_ltn, args_dpl, args_ccn



def read_args(baseline=False):
    parser = argparse.ArgumentParser(description="Execute notebooks with dynamic parameters.")
    parser.add_argument("--uns_parameter_percentage", type=float, default=1.0,
                        help="Percentage for the unsupervised parameter (float value)")
    parser.add_argument("--model_parameter_name", type=str, default="sl")
    parser.add_argument("--no_augmentations", action="store_true", 
                        help="Do not use augmentations for support sets")
    parser.add_argument("--pretraining_type", type=str, default=None, choices=["pre", "apre"])
    parser.add_argument("--GPU", type=str, default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hide", type=str, default="[]", help="Hidden classes")
    script_args = parser.parse_args()

    if script_args.model_parameter_name == 'sl':       
        args = args_sl
    elif script_args.model_parameter_name == 'ltn':    
        args = args_ltn
    elif script_args.model_parameter_name == 'dpl':    
        args = args_dpl
    else: 
        args = args_ccn


    args.uns_parameter_percentage = script_args.uns_parameter_percentage
    args.model_parameter_name = script_args.model_parameter_name
    args.no_augmentations = script_args.no_augmentations
    args.GPU_ID = script_args.GPU
    args.seed = script_args.seed
    args.pretraining_type = script_args.pretraining_type
    args.hide = ast.literal_eval(script_args.hide)
    args.no_interaction = True
    if (len(args.hide) > args.classes_per_it):
        args.classes_per_it = 10 - len(args.hide)

    assert isinstance(args.hide, list), "hide must be a list"
    assert all(isinstance(x, int) for x in args.hide), "All elements in hide must be integers"

    if baseline:
        # set the concept weight loss to 10.0 for sl to balance the loss that weights 10 in this case
        args.concept_loss_weight = 1.0 if args.model_parameter_name != 'sl' else 10.0
    
    return args



def setup_environment(args, baseline=False):

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
    save_folder = "mnadd-even-odd" 
    save_model_name = args.model_parameter_name
        
    if not baseline:
        save_path = os.path.join("..",
            "NEW-outputs", 
            save_folder, 
            "my_models", 
            save_model_name,
            f"episodic-proto-net-pipeline-{args.uns_parameter_percentage}-HIDE-{args.hide}"
        )
        print(f"Save paths: {str(save_path)}")
        
        if args.model in ['mnistsl', 'mnistltn', 'mnistdpl'] or not args.prototypes:
            raise ValueError("This experiment is NOT meant for baseline models.")
    else:
        prefix = ""
        suffix = ""
        if args.pretraining_type == "pre":
            prefix = "pretrained"
            suffix = "+c"
        else:
            prefix = "supervisions-via-augmentations"
        save_path = os.path.join("..",
            "outputs++", 
            save_folder, 
            "baseline", 
            save_model_name,
            f"{prefix}-{args.uns_parameter_percentage}{suffix}"
        )
        print(f"Save paths: {str(save_path)}")
        if args.model in ['promnistsl', 'promnistltn', 'promnistdpl', 'proshieldedmnist'] or args.prototypes:
            raise ValueError("This experiment is meant for baseline models.")

    args.save_path = save_path


# --- Signal Handler ---
def sigint_handler(signum, frame):
    print("\nSIGINT received. Releasing CUDA memory and exiting...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    sys.exit(0)