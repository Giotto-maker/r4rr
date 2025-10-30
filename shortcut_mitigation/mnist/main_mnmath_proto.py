
# ! Main of PNets NeSy model for MNMath
import re
import sys
import os
import torch
import random
import signal
import datetime
import numpy as np
import matplotlib.pyplot as plt
import setproctitle, socket, uuid

from tqdm import tqdm
from argparse import Namespace
from sklearn.metrics import confusion_matrix
from warmup_scheduler import GradualWarmupScheduler
from torch.utils.data import Dataset, DataLoader
from torch.nn.modules import Module

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
from utils.train import convert_to_categories, compute_coverage

from backbones.addmnist_protonet import PrototypicalLoss

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


# * Main Loop Training Function for NeSy models augmented with Prototypical Networks
def train(model:MnistDPL,
        encoder:Module,
        episodic_dataloader:DataLoader,
        unsup_train_loader:DataLoader,
        unsup_val_loader:DataLoader,
        _loss: ADDMNIST_DPL, 
        num_distinct_labels: int,
        args,
        save_folder
    ):
    
    # for full reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.enabled = False
    
    # for early stopping
    best_f1 = 0.0
    epochs_no_improve = 0

    # device configuration
    model.to(args.device)

    # Initialize the optimizer and the scheduler.
    optimizer = torch.optim.Adam(encoder.parameters())
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # scheduler from PNets paper


    print("\n--- Start of Training ---\n")
    proto_train_loss_history = []
    proto_train_acc_history = []
    for epoch in range(args.proto_epochs):
        print(f"Epoch {epoch+1}/{args.proto_epochs}")
        print("--- Training Protonet")

        encoder.train()

        epoch_loss = []
        epoch_acc = []
        
        # ^ PHASE 1: Training the Protonet with the episodic dataloader
        for batch in tqdm(episodic_dataloader, total=args.iterations):
            optimizer.zero_grad()
            
            # Get batch images and labels
            images, labels = batch
            images, labels = images.to(args.device), labels.to(args.device)
            
            # Forward pass: compute embeddings for all images in the episode.
            embeddings = encoder(images)
            
            # Compute prototypical loss.
            pNet_loss = PrototypicalLoss(n_support=args.num_support)
            loss, acc = pNet_loss(input=embeddings, target=labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss.append(loss.item())
            epoch_acc.append(acc.item())
        
        avg_loss = np.mean(epoch_loss)
        avg_acc = np.mean(epoch_acc)
        proto_train_loss_history.append(avg_loss)
        proto_train_acc_history.append(avg_acc)
        print(f"  Avg Loss: {avg_loss:.4f} | Avg Acc: {avg_acc:.4f}")


        # ^ PHASE 2: Training the Protonet with the unsupervised dataloader
        print("--- Training with Unsupervised Data")
        # ys are the predictions of the model, y_true are the true labels, cs are the predictions of the concepts, cs_true are the true concepts
        ys, y_true, cs, cs_true = None, None, None, None
        for i, data in enumerate(unsup_train_loader):
            if random.random() > args.uns_parameter_percentage:
                continue  # Skip this batch with probability (1 - percentage)

            optimizer.zero_grad()

            images, labels, concepts = data
            images, labels, concepts = (
                images.to(model.device),    # input IMAGES
                labels.to(model.device),    # ground truth LABELS
                concepts.to(model.device),  # ground truth CONCEPTS
            )

            # Get a random support set.
            this_support_images, this_support_labels = get_random_classes(
                mnist_dataset.images, mnist_dataset.labels, args.n_support, num_distinct_labels)
            assert this_support_images.shape == (args.n_support * num_distinct_labels, 1, 28, 28), \
                f"Support images shape is not ({args.n_support * num_distinct_labels}, 1, 28, 28), but {this_support_images.shape}"
            assert this_support_labels.shape == (args.n_support * num_distinct_labels, 1), \
                f"Support labels shape is not ({args.n_support * num_distinct_labels}, 1), but {this_support_labels.shape}"
            
            # ^ forward pass 
            out_dict = model(images, this_support_images, this_support_labels)

            ''' Enrich the out_dict with the ground truth labels and concepts '''
            out_dict.update({"LABELS": labels, "CONCEPTS": concepts})

            ''' Extract the predicted concepts for the first image in the batch '''
            loss, losses = _loss(out_dict, args)
            loss.backward()
            optimizer.step()

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


        # Step the scheduler at the end of the epoch
        lr_scheduler.step()

        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print("End of epoch ", epoch)
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print()

        # ^ PHASE 3: Evaluation
        if ys is None:  
            torch.save(model.state_dict(), save_folder)
            print(f"Saved model after prototypical netwrork training.")
            print()
            continue  # Skip evaluation if no unsupervised data was used
        
        # support images and labels for evaluation purposes
        this_support_images, this_support_labels = get_random_classes(
            mnist_dataset.images, mnist_dataset.labels, args.n_support, num_distinct_labels)
        assert this_support_images.shape == (args.n_support * num_distinct_labels, 1, 28, 28), \
            f"Support images shape is not ({args.n_support * num_distinct_labels}, 1, 28, 28), but {this_support_images.shape}"
        assert this_support_labels.shape == (args.n_support * num_distinct_labels, 1), \
            f"Support labels shape is not ({args.n_support * num_distinct_labels}, 1), but {this_support_labels.shape}"
        
        model.eval()
        tloss, cacc, yacc, f1_y, f1_c = evaluate_metrics(model, unsup_val_loader, args, support_images=this_support_images, support_labels=this_support_labels)

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
    # & read arguments and setup environment
    args = setup.read_args()
    setup.setup_environment(args)
    signal.signal(signal.SIGINT, setup.sigint_handler)

    # & load annotated prototypes
    support_loader, support_images_aug, support_labels_aug, query_loader, _ = create_support_query_sets_and_loaders(args)
    mnist_dataset = my_datasets.MNISTAugDataset(support_images_aug, support_labels_aug, hide_labels=args.hide)
    sanity_checker.assert_my_labels(args, support_labels_aug, mnist_dataset)
    proto_labels = mnist_dataset.labels.squeeze().numpy()
    episodic_dataloader = DataLoader(my_datasets.EmptyDataset(), batch_size=1)
    if args.classes_per_it > 0:
        sampler = PrototypicalBatchSampler(proto_labels, args.classes_per_it, args.num_samples, args.iterations)
        episodic_dataloader = DataLoader(mnist_dataset, batch_sampler=sampler)
    assert torch.equal(torch.sort(torch.unique(torch.tensor(proto_labels)))[0], torch.arange(0, 10)),\
        "proto_labels must contain all values from 0 to 9"
    
    # & dataset 
    mnmath_dataset = get_dataset(args)
    mnmath_train_loader, mnmath_val_loader, mnmath_test_loader, mnmath_ood_loader = mnmath_dataset.get_data_loaders()
    mnmath_dataset.print_stats()

    # & model 
    n_images, c_split = mnmath_dataset.get_split()
    encoder, decoder = mnmath_dataset.get_backbone()
    model = get_model(args, encoder, decoder, n_images, c_split)
    loss = model.get_loss(args)
    print("Using Dataset: ", mnmath_dataset)
    print("Using backbone: ", encoder)
    print("Using Model: ", model)
    print("Using Loss: ", loss)

    # & training
    num_distinct_labels = np.unique(proto_labels).size
    print(f"*** Training model with seed {args.seed}")
    print("Chosen device:", model.device)
    print("Save path for this model: ", args.save_path)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)
    save_folder = os.path.join(args.save_path, f"{args.model_parameter_name}_{args.seed}.pth")
    print("Saving in folder: ", save_folder)
    best_f1 = train(model=model,
        encoder=encoder,
        episodic_dataloader=episodic_dataloader,
        unsup_train_loader=mnmath_train_loader,
        unsup_val_loader=mnmath_val_loader,
        _loss=loss, 
        num_distinct_labels=num_distinct_labels,
        args=args,
        save_folder=save_folder,
    )
    save_model(model, args, args.seed)

    # & evaluation
    model = get_model(args, encoder, decoder, n_images, c_split)
    model_state_dict = torch.load(save_folder)
    model.load_state_dict(model_state_dict)
    model.to(args.device)
    metrics_log_path = save_folder.replace(".pth", "_test_metrics.log")
    metrics_log_path = re.sub(r'_\d+', '', metrics_log_path)
    this_support_images, this_support_labels = get_random_classes(
        mnist_dataset.images, mnist_dataset.labels, args.n_support, num_distinct_labels)
    assert this_support_images.shape == (args.n_support * num_distinct_labels, 1, 28, 28), \
        f"Support images shape is not ({args.n_support * num_distinct_labels}, 1, 28, 28), but {this_support_images.shape}"
    assert this_support_labels.shape == (args.n_support * num_distinct_labels, 1), \
        f"Support labels shape is not ({args.n_support * num_distinct_labels}, 1), but {this_support_labels.shape}"

    print(f"Evaluating model on test set with seed {args.seed}...")
    c_true_test, c_pred_test = evaluate_my_model(
        model=model, 
        save_path=metrics_log_path, 
        my_loader=mnmath_test_loader, 
        seed=args.seed,
        baseline=False,
        support_images=this_support_images,
        support_labels=this_support_labels,
        args=args,
    )
    print()
    print(f"Evaluating model on OOD set with seed {args.seed}...")
    metrics_log_path = save_folder.replace(".pth", "_ood_metrics.log")
    metrics_log_path = re.sub(r'_\d+', '', metrics_log_path)
    c_true_ood, c_pred_ood = evaluate_my_model(
        model=model, 
        save_path=metrics_log_path, 
        my_loader=mnmath_ood_loader, 
        seed=args.seed,
        baseline=False,
        support_images=this_support_images,
        support_labels=this_support_labels,
        args=args,
    )