
# ! Main of baseline NeSy models for MNEvenOdd with augmentations
import os
import sys
import torch
import signal
import random
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm
from argparse import Namespace
from torch.utils.data import DataLoader
from warmup_scheduler import GradualWarmupScheduler
from contextlib import redirect_stdout, redirect_stderr

from protonet_mnist_add_modules.utility_modules import setup
from protonet_mnist_add_modules.data_modules import my_datasets
from protonet_mnist_add_modules.utility_modules.pretraining import pre_train
from protonet_mnist_add_modules.utility_modules.proto_utils import init_dataloader
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


# * Main Training Loop
def train(model:MnistDPL,
        sup_train_loader:DataLoader,
        unsup_train_loader:DataLoader,
        unsup_val_loader:DataLoader,
        _loss: ADDMNIST_DPL, 
        args,
        seed,
        save_folder,
        log_file_path,
        sup_loss_weight=1.0,
        debug=False):
    
    with open(log_file_path, "w") as log_file, \
         redirect_stdout(log_file), \
         redirect_stderr(log_file):
    
        # for full reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
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

        # & FOR EACH EPOCH
        for epoch in range(args.proto_epochs):  # ^ ensure consistency with the number of epochs used for prototypical networks
            model.train()

            ###############################
            # 1. Episodic phase: Teach the model to recognize digits
            ###############################
            print("Start of supervised episodic training.")
            for i, (images, labels) in enumerate(sup_train_loader):
                sup_images = images.to(model.device)  # shape: (batch_size, C, 28, 28)
                sup_labels = labels.to(model.device)  # shape: (batch_size,)
                batch_size = sup_images.size(0)

                assert sup_images.shape == torch.Size([batch_size, 1, 28, 28]), \
                f"Expected shape [{batch_size}, 1, 28, 28], but got {sup_images.shape}"
                assert sup_labels.shape == torch.Size([batch_size]), \
                f"Expected shape [{batch_size}], but got {sup_labels.shape}"

                # Ensure batch size is even to form pairs (if odd, drop the last sample)
                if batch_size % 2 != 0:
                    sup_images = sup_images[:-1]
                    sup_labels = sup_labels[:-1]
                    batch_size -= 1
                    
                # Merge pairs: merge 0 with 1, 2 with 3, and so on. This yields merged_images of shape (batch_size//2, C, 28, 56)
                merged_images = torch.cat([sup_images[0::2], sup_images[1::2]], dim=3)
                
                assert merged_images.shape == torch.Size([batch_size//2, 1, 28, 56]), \
                f"Expected shape [{batch_size//2}, 1, 28, 56], but got {merged_images.shape}"
                
                # Extract corresponding labels for each digit in the pair
                labels_first = sup_labels[0::2]   # labels for the first digit in each pair
                labels_second = sup_labels[1::2]  # labels for the second digit in each pair

                # Plot the first merged image
                if debug:
                    plt.imshow(merged_images[0].cpu().numpy().squeeze(), cmap='gray')
                    plt.title(f"Labels: {labels_first[0].item()}, {labels_second[0].item()}")
                    plt.show()
                    
                # Forward pass: the model expects an image with two digits
                out_dict = model(merged_images)
                nconcept_preds = out_dict["pCS"]
                
                assert nconcept_preds.shape == torch.Size([batch_size//2, 2, 10]), \
                    f"Expected shape [{batch_size//2}, 2, 10], but got {nconcept_preds.shape}"
                
                concept_loss_first = F.cross_entropy(nconcept_preds[:, 0], labels_first)
                concept_loss_second = F.cross_entropy(nconcept_preds[:, 1], labels_second)
                concept_loss = sup_loss_weight * (concept_loss_first + concept_loss_second)

                # Backward pass and optimization step
                model.opt.zero_grad()
                concept_loss.backward()
                model.opt.step()

                progress_bar(i, len(unsup_train_loader) - 9, epoch, concept_loss.item())

            ###############################
            # 2. Original unsupervised training phase (sum prediction)
            ###############################
            # ys are the predictions of the model, y_true are the true labels, cs are the predictions of the concepts, cs_true are the true concepts
            ys, y_true, cs, cs_true = None, None, None, None
            
            # & FOR EACH BATCH
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
            
            if args.uns_parameter_percentage == 0.0:
                print("Saving...")
                torch.save(model.state_dict(), save_folder)
                print(f"Saved best model with F1 score: {best_f1}")
                print()
                continue

            # this are the actual model predictions
            y_pred = torch.argmax(ys, dim=-1)

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
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        return best_f1



if __name__ == "__main__":
    # read arguments and setup environment
    args = setup.read_args(baseline=True)
    setup.setup_environment(args, baseline=True)
    signal.signal(signal.SIGINT, setup.sigint_handler)

    # get supervised dataset via augmentations
    support_loader, support_images_aug, support_labels_aug, query_loader, _ = create_support_query_sets_and_loaders(args)
    mnist_dataset = my_datasets.MNISTAugDataset(support_images_aug, support_labels_aug, hide_labels=args.hide)
    sup_train_loader = DataLoader(mnist_dataset, batch_size=args.batch_size, shuffle=True)

    # get the unsupervised dataset
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

    print(f"*** Training model with seed {args.seed}")
    print("Chosen device:", model.device)
    print("Save path for this model: ", args.save_path)
    if not os.path.exists(args.save_path): os.makedirs(args.save_path, exist_ok=True)
    save_folder = os.path.join(args.save_path, f"{args.model_parameter_name}_{args.seed}.pth")
    print("Saving in folder: ", save_folder)
    log_file_path = os.path.join(args.save_path, f"{args.model_parameter_name}_{args.seed}.log")

    best_f1 = train(model=model,
        sup_train_loader=sup_train_loader,
        unsup_train_loader=unsup_train_loader,
        unsup_val_loader=unsup_val_loader,
        _loss=loss, 
        args=args,
        seed=args.seed,
        sup_loss_weight=args.concept_loss_weight,
        log_file_path=log_file_path,
        save_folder=save_folder
    )
    save_model(model, args, args.seed)  # save the model parameters

    
print("End of experiment")
sys.exit(0)