
# ! Main of pretrained baseline NeSy models for MNEvenOdd
import os 
import sys
import torch
import signal
import random
import numpy as np
import torch.nn.functional as F

from argparse import Namespace
from torch.utils.data import DataLoader
from warmup_scheduler import GradualWarmupScheduler

from protonet_mnist_add_modules.utility_modules import setup
from protonet_mnist_add_modules.utility_modules.pretraining import pre_train
from protonet_mnist_add_modules.utility_modules.proto_utils import init_dataloader

from protonet_mnist_add_modules.data_modules import my_datasets
from protonet_mnist_add_modules.data_modules.proto_data_creation import (
    choose_initial_prototypes, 
    get_augmented_support_query_set, 
    get_augmented_support_query_loader
)

sys.path.append(os.path.abspath(".."))      
sys.path.append(os.path.abspath("../..")) 

from datasets import get_dataset
from models import get_model
from models.mnistdpl import MnistDPL

from utils import fprint
from utils.status import progress_bar
from utils.metrics import evaluate_metrics
from utils.dpl_loss import ADDMNIST_DPL
from utils.checkpoint import save_model



# * Returns the support and query loaders for the annotated augmented dataset
def create_support_query_sets_and_loaders(args):
    args_protonet = Namespace(
        dataset=args.prototypical_dataset,     
        batch_size=args.prototypical_batch_size,
        preprocess=0,
        c_sup=1,    # ^ supervision loaded to simulate direct annotation for prototypes
        which_c=[-1],
        model=args.model,        
        task=args.task,    
    )
    addmnist_dataset = get_dataset(args_protonet)
    addmnist_train_loader, _ , _ = addmnist_dataset.get_data_loaders()

    if ( (not os.path.exists('data/prototypes/proto_loader_dataset.pth')) or args.debug ):
        print("Creating initial prototypes...")
        choose_initial_prototypes(addmnist_train_loader, debug=args.debug)

    tr_dataloader = init_dataloader()
    support_images_aug, support_labels_aug, query_images_aug, query_labels_aug, no_aug = get_augmented_support_query_set(
        tr_dataloader, debug=args.debug)
    support_loader, query_loader = get_augmented_support_query_loader(
        support_images_aug, 
        support_labels_aug, 
        query_images_aug, 
        query_labels_aug,
        query_batch_size=32,
        debug=args.debug
    )
    return support_loader, support_images_aug, support_labels_aug, query_loader, no_aug


# * Training Loop
def train(model:MnistDPL,
        sup_train_loader:DataLoader,
        unsup_train_loader:DataLoader,
        unsup_val_loader:DataLoader,
        _loss: ADDMNIST_DPL, 
        args,
        save_folder,
    ):
    
    # for full reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.enabled = False
    
    best_f1 = 0.0
    epochs_no_improve = 0   # for early stopping

    # model configuration for shortmnist
    if args.dataset == "shortmnist":    model = model.float()
    model.to(model.device)

    # get the data loaders
    scheduler = torch.optim.lr_scheduler.ExponentialLR(model.opt, args.exp_decay)
    w_scheduler = None
    if args.warmup_steps > 0:   w_scheduler = GradualWarmupScheduler(model.opt, 1.0, args.warmup_steps)

    fprint("\n--- Start of Training ---\n")

    # default for warm-up
    model.opt.zero_grad()
    model.opt.step()
    enc_opt = torch.optim.Adam(model.encoder.parameters(), args.lr, weight_decay=args.weight_decay)

    # & FOR EACH EPOCH
    for epoch in range(args.proto_epochs):  # ^ ensure consistency with the number of epochs used for prototypical networks
        
        # * ALTERNATE PRETRAINING
        if args.pretraining_type == "apre":
            model.encoder.train()
            # & FOR EACH (SUPERVISED) BATCH
            fprint("\n--- Start of Supervised Training ---\n")
            for i, (images, labels) in enumerate(sup_train_loader):
                sup_images = images.to(model.device)               # shape: (batch_size, C, 28, 28)
                sup_labels = labels.to(model.device)               # shape: (batch_size, 1)
                enc_opt.zero_grad()
                preds = model.encoder(sup_images)[0].squeeze(1)
                loss = F.cross_entropy(preds, sup_labels)
                loss.backward()
                enc_opt.step()
                progress_bar(i, len(sup_train_loader), epoch, loss.item())
            
        # * UNSUPERVISED TRAINING
        model.train()
        # ys are the predictions of the model, y_true are the true labels, cs are the predictions of the concepts, cs_true are the true concepts
        ys, y_true, cs, cs_true = None, None, None, None
        # & FOR EACH (UNSUPERVISED) BATCH
        print("Start of unsupervised training.")
        for i, data in enumerate(unsup_train_loader):
            if random.random() > args.uns_parameter_percentage:
                continue  # Skip this batch with probability (1 - percentage)

            images, labels, concepts = data
            images, labels, concepts = (
                images.to(model.device),    # input IMAGES
                labels.to(model.device),    # ground truth LABELS
                concepts.to(model.device),  # ground truth CONCEPTS
            )

            # ^ baseline model
            out_dict = model(images)

            ''' Enrich the out_dict with the ground truth labels and concepts '''
            out_dict.update({"LABELS": labels, "CONCEPTS": concepts})

            ''' Extract the predicted concepts for the first image in the batch '''
            model.opt.zero_grad()
            loss, losses = _loss(out_dict, args)
            loss.backward()
            model.opt.step()
            
            if ys is None:  # first iteration
                ys = out_dict["YS"]
                y_true = out_dict["LABELS"]
                cs = out_dict["pCS"]
                cs_true = out_dict["CONCEPTS"]
            else:           # all other iterations
                ys = torch.concatenate((ys, out_dict["YS"]), dim=0)
                y_true = torch.concatenate((y_true, out_dict["LABELS"]), dim=0)
                cs = torch.concatenate((cs, out_dict["pCS"]), dim=0)
                cs_true = torch.concatenate((cs_true, out_dict["CONCEPTS"]), dim=0)

            if i % 10 == 0:
                progress_bar(i, len(unsup_train_loader) - 9, epoch, loss.item())


        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print("End of epoch ", epoch)
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print()
        
        # enter the evaluation phase
        model.eval()
        # ^ baseline model
        tloss, cacc, yacc, f1 = evaluate_metrics(model, unsup_val_loader, args)

        # update the (warmup) scheduler at end of the epoch
        if epoch < args.warmup_steps:
            w_scheduler.step()
        else:
            scheduler.step()
            if hasattr(_loss, "grade"):
                _loss.update_grade(epoch)

        ### LOGGING ###
        fprint("  ACC C", cacc, "  ACC Y", yacc, "F1 Y", f1)
        print()

        if not args.tuning and f1 > best_f1:
            print("Saving...")
            # Update best F1 score
            best_f1 = f1
            epochs_no_improve = 0

            # Save the best model
            torch.save(model.state_dict(), save_folder)
            print(f"Saved best model with F1 score: {best_f1}")
            print()
        
        elif f1 <= best_f1:
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print("End of training")
    return best_f1


if __name__ == "__main__":
    # & Read arguments and setup environment
    args = setup.read_args(baseline=True)
    setup.setup_environment(args, baseline=True)
    signal.signal(signal.SIGINT, setup.sigint_handler)

    # & Get supervised dataset via augmentations
    _, support_images_aug, support_labels_aug, _, _ = \
        create_support_query_sets_and_loaders(args)
    mnist_dataset = my_datasets.MNISTAugDataset(
        support_images_aug, support_labels_aug, hide_labels=args.hide
    )
    sup_train_loader = DataLoader(
        mnist_dataset, batch_size=args.batch_size, shuffle=True
    )

    # & Get the unsupervised dataset and model
    dataset = get_dataset(args)
    n_images, c_split = dataset.get_split()
    encoder, decoder = dataset.get_backbone()
    model = get_model(args, encoder, decoder, n_images, c_split)
    loss = model.get_loss(args)
    model.start_optim(args)
    print("Using Dataset: ", dataset)
    print("Using backbone: ", encoder)
    print("Using Model: ", model)
    print("Using Loss: ", loss)
    unsup_train_loader, unsup_val_loader, _ = dataset.get_data_loaders()

    # & PreTraining
    if args.pretraining_type == "pre":
        print("*** Pretraining model ", args.seed)
        pre_train(model, sup_train_loader, args)            

    # & Training
    print(f"*** Training model with seed {args.seed}")
    print("Chosen device:", model.device)
    print("Save path for this model: ", args.save_path)
    if not os.path.exists(args.save_path): os.makedirs(args.save_path, exist_ok=True)
    save_folder = os.path.join(
        args.save_path, f"{args.model_parameter_name}_{args.seed}.pth"
    )
    print("Saving in folder: ", save_folder)
    log_file_path = os.path.join(args.save_path, f"{args.model_parameter_name}_{args.seed}.log")
    best_f1 = train(model=model,
        sup_train_loader=sup_train_loader,
        unsup_train_loader=unsup_train_loader,
        unsup_val_loader=unsup_val_loader,
        _loss=loss, 
        args=args,
        save_folder=save_folder
    )
    print("Best F1 score:", best_f1)
    save_model(model, args, args.seed)  # save the model parameters

    
print("End of experiment")
sys.exit(0)

    