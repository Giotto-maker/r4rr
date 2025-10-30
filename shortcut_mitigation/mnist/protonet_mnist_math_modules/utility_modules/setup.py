import os
import sys
import ast
import torch
import datetime
import argparse
import setproctitle, socket, uuid

from .check_gpu import my_gpu_info 
from ..arguments_baseline import args_cbm_base, args_dpl_base
from ..arguments_proto import args_cbm_proto, args_dpl_proto



def read_args(baseline=False):
    parser = argparse.ArgumentParser(description="Execute notebooks with dynamic parameters.")
    parser.add_argument("--uns_parameter_percentage", type=float, default=1.0,
                        help="Percentage for the unsupervised parameter (float value)")
    parser.add_argument("--model_parameter_name", type=str, default="dpl")
    parser.add_argument("--GPU", type=str, default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hide", type=str, default="[]", help="Hidden classes")
    parser.add_argument("--aug" , type=bool, default=False, help="Use data augmentation")
    script_args = parser.parse_args()

    if script_args.model_parameter_name == 'dpl' and baseline:
        print("Using DPL baseline arguments")   
        args = args_dpl_base
    elif script_args.model_parameter_name == 'dpl' and not baseline:
        print("Using DPL Prototypical arguments")
        args = args_dpl_proto
    elif script_args.model_parameter_name == 'cbm' and baseline:
        print("Using CBM baseline arguments")    
        args = args_cbm_base
    elif script_args.model_parameter_name == 'cbm' and not baseline:
        print("Using CBM Prototypical arguments")
        args = args_cbm_proto
    else:                                              
        raise ValueError("Invalid model parameter name. Choose from 'dpl', or 'cbm'.")

    args.uns_parameter_percentage = script_args.uns_parameter_percentage
    args.model_parameter_name = script_args.model_parameter_name
    args.GPU_ID = script_args.GPU
    args.seed = script_args.seed
    args.aug = script_args.aug
    args.hide = ast.literal_eval(script_args.hide)
    args.no_interaction = True
    if (len(args.hide) > args.classes_per_it):
        args.classes_per_it = 10 - len(args.hide)

    assert isinstance(args.hide, list), "hide must be a list"
    assert all(isinstance(x, int) for x in args.hide), "All elements in hide must be integers"

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
    save_folder = "mnmath" 
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
        
        if args.model in ['mnmathdpl', 'mnmathcbm'] or not args.prototypes:
            raise ValueError("This experiment is NOT meant for baseline models.")
    else:
        save_path = os.path.join("..",
            "NEW-outputs", 
            save_folder, 
            "baseline", 
            save_model_name,
            f"my_baseline"
        )
        if args.aug:
            save_path += f"-aug-{args.uns_parameter_percentage}"
        print(f"Save paths: {str(save_path)}")
        if args.model in ['promnmathdpl', 'promnmathcbm'] or args.prototypes:
            raise ValueError("This experiment is meant for baseline models.")

    args.save_path = save_path


# --- Signal Handler ---
def sigint_handler(signum, frame):
    print("\nSIGINT received. Releasing CUDA memory and exiting...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    sys.exit(0)