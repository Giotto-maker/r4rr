
# ! Main of baseline NeSy models for MNMath 
import re
import sys
import os
import torch
import signal
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from argparse import Namespace
from torch.utils.data import Dataset, DataLoader
from warmup_scheduler import GradualWarmupScheduler

from protonet_mnist_add_modules.utility_modules import sanity_checker
from protonet_mnist_add_modules.data_modules import my_datasets
from protonet_mnist_add_modules.data_modules.proto_data_creation import (
    choose_initial_prototypes, 
    get_augmented_support_query_set, 
    get_augmented_support_query_loader
)
from protonet_mnist_add_modules.utility_modules.proto_utils import (
    init_dataloader, 
    get_random_classes
)
from protonet_mnist_add_modules.data_modules.prototypical_batch_sampler import PrototypicalBatchSampler

from protonet_mnist_math_modules.utility_modules import setup
from protonet_mnist_math_modules.utility_modules.evaluate import evaluate_my_model

sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("../.."))

from models import get_model
from models.mnistdpl import MnistDPL
from datasets import get_dataset
from utils import fprint
from utils.checkpoint import save_model
from utils.status import progress_bar
from utils.dpl_loss import ADDMNIST_DPL
from utils.metrics import evaluate_metrics


# * Returns the support and query loaders for the prototypical network training
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


# * Main Loop Training Function
def train(model:MnistDPL,
        train_loader:DataLoader,
        val_loader:DataLoader,
        supervised_dataloader:DataLoader,
        _loss: ADDMNIST_DPL, 
        args,
        save_folder,
        debug=False):
    
    # for full reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.enabled = False
    
    best_f1 = 0.0
    epochs_no_improve = 0   # for early stopping

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
    for epoch in range(args.n_epochs):
        model.train()

        ###############################
        # 1. Train the model to recognize the concepts in the support set
        ###############################
        if args.aug:
            print("Start of supervised episodic training.")
            for i, (images, labels) in enumerate(supervised_dataloader):
                sup_images = images.to(model.device)  # shape: (batch_size, 1, 28, 28)
                sup_labels = labels.to(model.device)  # shape: (batch_size,)
                batch_size = sup_images.size(0)

                assert sup_images.shape == torch.Size([batch_size, 1, 28, 28]), \
                    f"Expected shape [{batch_size}, 1, 28, 28], but got {sup_images.shape}"
                assert sup_labels.shape == torch.Size([batch_size]), \
                    f"Expected shape [{batch_size}], but got {sup_labels.shape}"

                # Ensure batch size is divisible by 8 to form groups of 8 images
                remainder = batch_size % 8
                if remainder != 0:
                    sup_images = sup_images[:-remainder]
                    sup_labels = sup_labels[:-remainder]
                    batch_size -= remainder

                # Merge every 8 images along the width: result shape (batch_size//8, 1, 28, 224)
                merged_images = torch.cat(
                    [sup_images[j::8] for j in range(8)], dim=3
                )

                assert merged_images.shape == torch.Size([batch_size // 8, 1, 28, 224]), \
                    f"Expected shape [{batch_size // 8}, 1, 28, 224], but got {merged_images.shape}"

                # Extract corresponding labels for each digit in the group of 8
                labels_group = [sup_labels[j::8] for j in range(8)]  # list of 8 tensors, each (batch_size // 8,)

                if debug:
                    plt.imshow(merged_images[0].cpu().numpy().squeeze(), cmap='gray')
                    label_text = ", ".join(str(label[0].item()) for label in labels_group)
                    plt.title(f"Labels: {label_text}")
                    plt.show()

                # Forward pass: the model should output (batch_size//8, 8, 10)
                out_dict = model(merged_images)
                nconcept_preds = out_dict["pCS"]

                assert nconcept_preds.shape == torch.Size([batch_size // 8, 8, 10]), \
                    f"Expected shape [{batch_size // 8}, 8, 10], but got {nconcept_preds.shape}"

                # Compute individual cross-entropy losses for each digit position
                losses = [F.cross_entropy(nconcept_preds[:, k], labels_group[k]) for k in range(8)]
                concept_loss = sum(losses)

                # Backward pass and optimization step
                model.opt.zero_grad()
                concept_loss.backward()
                model.opt.step()

                if i % 10 == 0:
                    print(f"Episodic phase, Epoch {epoch}, Batch {i}: Concept Loss = {concept_loss.item():.4f}")

        ###############################
        # 2. Original unsupervised training phase (mnmath prediction)
        ###############################
        # ys are the predictions of the model, y_true are the true labels, cs are the predictions of the concepts, cs_true are the true concepts
        ys, y_true, cs, cs_true = None, None, None, None
        # & FOR EACH BATCH
        for i, data in enumerate(train_loader):

            images, labels, concepts = data
            images, labels, concepts = (
                images.to(model.device),    # input IMAGES
                labels.to(model.device),    # ground truth LABELS
                concepts.to(model.device),  # ground truth CONCEPTS
            )

            # ^ forward pass 
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
                progress_bar(i, len(train_loader) - 9, epoch, loss.item())


        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print("End of epoch ", epoch)
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print()
        
        # ^ enter the evaluation phase
        model.eval()
        tloss, cacc, yacc, f1_y, f1_c = evaluate_metrics(model, val_loader, args)

        # update the (warmup) scheduler at end of the epoch
        if epoch < args.warmup_steps:
            w_scheduler.step()
        else:
            scheduler.step()
            if hasattr(_loss, "grade"):
                _loss.update_grade(epoch)

        ### LOGGING ###
        fprint("  ACC C", cacc, "  ACC Y", yacc, "F1 Y", f1_y, "F1 C", f1_c)
        print()

        if not args.tuning and f1_y > best_f1:
            print("Saving...")
            # Update best F1 score
            best_f1 = f1_y
            epochs_no_improve = 0

            # Save the best model
            torch.save(model.state_dict(), save_folder)
            print(f"Saved best model with F1 score: {best_f1}")
            print()

        elif f1_y <= best_f1:
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print("End of training")
    return best_f1


if __name__ == "__main__":
    # & setup
    args = setup.read_args(baseline=True)
    setup.setup_environment(args, baseline=True)
    signal.signal(signal.SIGINT, setup.sigint_handler)

    # & dataset 
    mnmath_dataset = get_dataset(args)
    mnmath_train_loader, mnmath_val_loader, mnmath_test_loader, mnmath_ood_loader = mnmath_dataset.get_data_loaders()
    mnmath_dataset.print_stats()

    # & OPTIONAL: load augmentations if required
    supervised_dataloader = None
    if args.aug:
        support_loader, support_images_aug, support_labels_aug, query_loader, _ = create_support_query_sets_and_loaders(args)
        mnist_dataset = my_datasets.MNISTAugDataset(support_images_aug, support_labels_aug, hide_labels=args.hide)
        sanity_checker.assert_my_labels(args, support_labels_aug, mnist_dataset)
        proto_labels = mnist_dataset.labels.squeeze().numpy()
        episodic_dataloader = DataLoader(my_datasets.EmptyDataset(), batch_size=1)
        if args.classes_per_it > 0:
            sampler = PrototypicalBatchSampler(proto_labels, args.classes_per_it, args.num_samples, args.iterations)
            supervised_dataloader = DataLoader(mnist_dataset, batch_sampler=sampler)
        assert torch.equal(torch.sort(torch.unique(torch.tensor(proto_labels)))[0], torch.arange(0, 10)),\
            "proto_labels must contain all values from 0 to 9"
        print("Using Augmentations to supervise the baseline model training.")
        
    # & model 
    n_images, c_split = mnmath_dataset.get_split()
    encoder, decoder = mnmath_dataset.get_backbone()
    model = get_model(args, encoder, decoder, n_images, c_split)
    loss = model.get_loss(args)
    model.start_optim(args)
    print("Using Dataset: ", mnmath_dataset)
    print("Using backbone: ", encoder)
    print("Using Model: ", model)
    print("Using Loss: ", loss)

    # & training 
    f1_scores = dict()
    print(f"*** Training model with seed {args.seed}")
    print("Chosen device:", model.device)
    print("Save path for this model: ", args.save_path)
    if not os.path.exists(args.save_path): os.makedirs(args.save_path, exist_ok=True)
    save_folder = os.path.join(args.save_path, f"{args.model_parameter_name}_{args.seed}.pth")
    print("Saving in folder: ", save_folder)
    best_f1 = train(model=model,
        train_loader=mnmath_train_loader,
        val_loader=mnmath_val_loader,
        supervised_dataloader=supervised_dataloader,
        _loss=loss, 
        args=args,
        save_folder=save_folder
    )
    f1_scores[args.seed] = best_f1
    save_model(model, args, args.seed)  # save the model parameters
    best_weight_seed = max(f1_scores, key=f1_scores.get)
    print(f"Best weight and seed combination: {best_weight_seed} with F1 score: {f1_scores[best_weight_seed]}")

    # & Evaluation
    model = get_model(args, encoder, decoder, n_images, c_split)
    model_state_dict = torch.load(save_folder)
    model.load_state_dict(model_state_dict)
    model.to(args.device)

    metrics_log_path = save_folder.replace(".pth", "_test_metrics.log")
    metrics_log_path = re.sub(r'_\d+', '', metrics_log_path)

    # Evaluate the model over the test set
    print(f"Evaluating model on test set with seed {args.seed}...")
    c_true_test, c_pred_test = evaluate_my_model(
        model=model, save_path=metrics_log_path, my_loader=mnmath_test_loader, seed=args.seed, args=args
    )
    print()

    # Evaluate the model over the OOD set
    print(f"Evaluating model on OOD set with seed {args.seed}...")
    metrics_log_path = save_folder.replace(".pth", "_ood_metrics.log")
    metrics_log_path = re.sub(r'_\d+', '', metrics_log_path)
    c_true_ood, c_pred_ood = evaluate_my_model(
        model=model, save_path=metrics_log_path, my_loader=mnmath_ood_loader, seed=args.seed, args=args
    )

    print("End of experiment")
    sys.exit(0)