import os
import sys
import ast
import torch
import datetime
import argparse
import setproctitle, socket, uuid

from .check_gpu import my_gpu_info 
from ..arguments import args_sl, args_ltn, args_dpl


def read_args(baseline=False):
    parser = argparse.ArgumentParser(description="Execute notebooks with dynamic parameters.")
    parser.add_argument("--uns_parameter_percentage", type=float, default=1.0,
                        help="Percentage for the unsupervised parameter (float value)")
    parser.add_argument("--model_parameter_name", type=str, default="sl")
    parser.add_argument("--no_augmentations", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--Apretrained", action="store_true") 
    parser.add_argument("--c", action="store_true")
    parser.add_argument("--GPU", type=str, default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hide_shapes", type=str, default="[]", help="Hidden shape classes")
    parser.add_argument("--hide_colors", type=str, default="[]", help="Hidden colors classes")
    script_args = parser.parse_args()

    if script_args.model_parameter_name == 'sl':       
        args = args_sl
    elif script_args.model_parameter_name == 'ltn':    
        args = args_ltn
    else:                                              
        args = args_dpl

    args.no_interaction = True
    args.uns_parameter_percentage = script_args.uns_parameter_percentage
    args.model_parameter_name = script_args.model_parameter_name
    args.no_augmentations = script_args.no_augmentations
    args.GPU_ID = script_args.GPU
    args.seed = script_args.seed
    args.pretrained = script_args.pretrained
    args.Apretrained = script_args.Apretrained
    args.c = script_args.c
    args.hide_shapes = ast.literal_eval(script_args.hide_shapes)
    args.hide_colors = ast.literal_eval(script_args.hide_colors)

    if baseline and (args.model not in [
            'kandsl', 'kandltn', 'kanddpl',
            'kanddplsinglejoint','kandslsinglejoint','kandltnsinglejoint',
            'kanddplsingledisj','kandslsingledisj','kandltnsingledisj',
            ] or args.prototypes):
        raise ValueError("This experiment is meant for baseline models.")
    
    if not baseline and (args.model not in [
            'prokandsl', 'prokandltn', 'prokanddpl',
            ] or not args.prototypes):
        raise ValueError("This experiment is meant for PNet models.")

    assert not (args.pretrained and args.Apretrained),\
        "Both --pretrained and --Apretrained should not be True."
    assert not (args.Apretrained and args.c),\
        "Apretrained should not be used with concept supervision."

    if baseline and args.c:
        # set the concept weight loss to 10.0 for sl to balance the loss that weights 10 in this case
        args.concept_loss_weight = 1.0 if args.model_parameter_name != 'sl' else 10.0

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
    save_folder = "kandinsky" 
    save_model_name = args.model_parameter_name
    final_folder = f"episodic-proto-net-pipeline-{args.uns_parameter_percentage}-HIDE-[]" 
    if args.pretrained:
        final_folder = "pretrained"
    elif args.Apretrained:
        final_folder = "Apretrained"
    if args.no_augmentations:
        final_folder += "-noaug"

    if args.c:
        final_folder += "-c"

    save_path = os.path.join("..",
        "NEW-outputs", 
        save_folder, 
        "my_models", 
        save_model_name,
        final_folder
    )
    print(f"Save paths: {str(save_path)}")
    args.save_path = save_path



# --- Signal Handler ---
def sigint_handler(signum, frame):
    print("\nSIGINT received. Releasing CUDA memory and exiting...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    sys.exit(0)