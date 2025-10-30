import sys
import os
import torch
import signal
import random
import numpy as np
import torch.nn as nn
import torchvision.transforms as T

from tqdm import tqdm
from ultralytics import YOLO
from torch.utils.data import DataLoader
from contextlib import redirect_stdout, redirect_stderr
from protonet_kand_modules.utility_modules import setup

from protonet_kand_modules.data_modules import batch_sampler, my_datasets
from protonet_kand_modules.data_modules.proto_data_creation import get_my_initial_prototypes, get_random_classes

sys.path.append(os.path.abspath(".."))  
sys.path.append(os.path.abspath("../.."))  

from datasets import get_dataset
from models import get_model
from utils import fprint
from utils.status import progress_bar
from utils.checkpoint import save_model
from backbones.kand_protonet import PrototypicalLoss
from utils.metrics import evaluate_metrics
from utils.dpl_loss import ADDMNIST_DPL
from utils.checkpoint import save_model



# * Utility method to filter out hidden classes
def filter_hidden_classes(images, labels, hide):
    """
    Filters out images and labels corresponding to the classes in the hide list.

    Args:
        images (torch.Tensor): A tensor of shape (batch_size, 3, 64, 64).
        labels (torch.Tensor): A tensor of shape (batch_size) with labels 0, 1, or 2.
        hide (list): A list of integers representing the classes to hide.

    Returns:
        torch.Tensor, torch.Tensor: Filtered images and labels.
    """
    if hide:
        # Create a mask for labels not in the hide list
        mask = ~torch.isin(labels, torch.tensor(hide, device=labels.device))
        
        # Apply the mask to filter images and labels
        filtered_images = images[mask]
        filtered_labels = labels[mask]
        
        return filtered_images, filtered_labels
    else:
        return images, labels



# * Main Training Loop
def train(model, concept_extractor, concept_extractor_training_path, concept_extractor_project_path, transform,                  
        episodic_shape_dataloader, episodic_color_dataloader, unsup_dataset, 
        _loss, args, seed, save_folder, log_file_path, patience=3):
    
    with open(log_file_path, "w") as log_file, \
         redirect_stdout(log_file), \
         redirect_stderr(log_file):
    
        # for full reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        torch.backends.cudnn.enabled = False
        torch.cuda.synchronize()
        
        best_cacc = 0.0
        epochs_no_improve = 0
        yolo_save_dir = None
        
        model.to(model.device)
        
        # Initialize optimizers and schedulers
        shape_optimizer = torch.optim.Adam(model.encoder[0].parameters())
        color_optimizer = torch.optim.Adam(model.encoder[1].parameters())
        shape_lr_scheduler = torch.optim.lr_scheduler.StepLR(shape_optimizer, step_size=10, gamma=0.5)
        color_lr_scheduler = torch.optim.lr_scheduler.StepLR(color_optimizer, step_size=10, gamma=0.5)
        
        unsup_train_loader, unsup_val_loader, unsup_test_loader = unsup_dataset.get_data_loaders()

        fprint("\n--- Start of Training ---\n")
        for epoch in range(args.proto_epochs + 1):  # first epoch is for determining the baseline accuracy
            print(f"Epoch {epoch+1}/{args.proto_epochs + 1}")

            # ^ PHASE 1: Training the Concept Extractor
            if args.retrain_extractor:
                print('----------------------------------')
                print('--- Concept Extractor Training ---')
                if epoch == 0:
                    results = concept_extractor.train(data=concept_extractor_training_path, 
                                epochs=args.extractor_training_epochs, 
                                imgsz=64, 
                                project=concept_extractor_project_path,
                                device='cuda:'+args.GPU_ID,
                                name=concept_extractor_project_path,
                                seed=seed,
                                cache=False,
                                exist_ok=True)
                    yolo_save_dir = os.path.join(results.save_dir, "weights", "last.pt")
                else:
                    assert yolo_save_dir is not None
                    concept_extractor = YOLO(yolo_save_dir)
                    results = concept_extractor.train(data=concept_extractor_training_path, 
                                epochs=args.extractor_training_epochs, 
                                imgsz=64, 
                                project=concept_extractor_project_path,
                                device='cuda:'+args.GPU_ID,
                                name=concept_extractor_project_path,
                                seed=seed,
                                cache=False,
                                exist_ok=True)
                    yolo_save_dir = os.path.join(results.save_dir, "weights", "last.pt")

            # ^ PHASE 2: Training the Prototypical Networks
            print('----------------------------------')
            print('--- Prototypical Networks Training ---')
            
            model.train()

            epoch_train_loss_shapes, epoch_train_acc_shapes, epoch_train_loss_colors, epoch_train_acc_colors = [], [], [], []
            
            # * Under the assumption that both dataloaders yield the same number of episodes per epoch.
            pNet_loss = PrototypicalLoss(n_support=args.num_support)
            for (shape_batch, color_batch) in tqdm(zip(episodic_shape_dataloader, episodic_color_dataloader), total=args.iterations):
                
                # ------------------
                # & Process shape episode
                # ------------------
                shape_optimizer.zero_grad()
                shape_images, shape_labels, _ = shape_batch
                shape_images, shape_labels = filter_hidden_classes(shape_images, shape_labels, args.hide_shapes)
                if args.hide_shapes:   assert not any(label in args.hide_shapes for label in shape_labels), "shape_labels contains hidden classes"
                shape_images, shape_labels = shape_images.to(args.device), shape_labels.to(args.device)

                # Forward pass: compute embeddings for all images in the episode.
                shape_embeddings = model.encoder[0](shape_images)

                # Compute prototypical loss.
                shape_loss, shape_acc = pNet_loss(input=shape_embeddings, target=shape_labels)
                shape_loss.backward()
                shape_optimizer.step()

                epoch_train_loss_shapes.append(shape_loss.item())
                epoch_train_acc_shapes.append(shape_acc.item())

                # ------------------
                # & Process color episode
                # ------------------
                color_optimizer.zero_grad()
                color_images, _, color_labels = color_batch
                color_images, color_labels = filter_hidden_classes(color_images, color_labels, args.hide_colors)
                if args.hide_colors:   assert not any(label in args.hide_colors for label in color_labels), "color_labels contains hidden classes"
                color_images, color_labels = color_images.to(args.device), color_labels.to(args.device)

                # Forward pass: compute embeddings for all images in the episode.
                color_embeddings = model.encoder[1](color_images)

                # Compute prototypical loss.
                color_loss, color_acc = pNet_loss(input=color_embeddings, target=color_labels)
                color_loss.backward()
                color_optimizer.step()

                epoch_train_loss_colors.append(color_loss.item())
                epoch_train_acc_colors.append(color_acc.item())

            avg_loss_shapes = np.mean(epoch_train_loss_shapes)
            avg_acc_shapes = np.mean(epoch_train_acc_shapes)
            avg_loss_colors = np.mean(epoch_train_loss_colors)
            avg_acc_colors = np.mean(epoch_train_acc_colors)
            
            print(f"Shapes  - Avg Loss: {avg_loss_shapes:.4f} | Avg Acc: {avg_acc_shapes:.4f}")
            print(f"Colors  - Avg Loss: {avg_loss_colors:.4f} | Avg Acc: {avg_acc_colors:.4f}")

        
            # ^ PHASE 3: Training the model with Unsupervised Data
            print('----------------------------------')
            print("--- Training with Unsupervised Data ---")

            # ys are the predictions of the model, y_true are the true labels, cs are the predictions of the concepts, cs_true are the true concepts
            ys, y_true, cs, cs_true = None, None, None, None

            unknown_init = True if (len(args.hide_shapes) > 0 or len(args.hide_colors)) else False
            for i,data in enumerate(unsup_train_loader):
                if random.random() > args.uns_parameter_percentage:
                    continue  # Skip this batch with probability (1 - percentage)

                if epoch == 0:
                    model.eval()
                    if args.debug:  print("Find baseline accuracy, no training.")
                    assert not model.training, "Model should **NOT** be in training mode!"
                    assert not model.encoder[0].training, "Shape encoder should **NOT** be in training mode!"
                    assert not model.encoder[1].training, "Color encoder should **NOT** be in training mode!"
                else:    
                    shape_optimizer.zero_grad()
                    color_optimizer.zero_grad()
                    if args.debug:  print("Reset the optimizers.")
                    assert model.training, "Model should be in training mode!"
                    assert model.encoder[0].training, "Shape encoder should be in training mode!"
                    assert model.encoder[1].training, "Color encoder should be in training mode!"

                # load batch
                images, labels, concepts = data
                images, labels, concepts = (
                    images.to(model.device),
                    labels.to(model.device),
                    concepts.to(model.device),
                )
                batch_size = images.shape[0]
                assert images.shape == (batch_size, 3, 64, 192), f"Expected shape (B, 3, 64, 192), but got {images.shape}"
                assert labels.shape == (batch_size, 4), f"Expected shape (B, 4), but got {labels.shape}"
                assert concepts.shape == (batch_size, 3, 6), f"Expected shape (B, 3, 6), but got {concepts.shape}"

                # Get a random support set.
                if args.no_augmentations:
                    this_support_images = kand_proto_dataset.images
                    this_support_labels = kand_proto_dataset.labels
                else:
                    this_support_images, this_support_labels = get_random_classes(
                        kand_proto_dataset.images, kand_proto_dataset.labels, args.n_support, args.num_distinct_labels)
                    assert this_support_images.shape == (args.n_support * args.num_distinct_labels, 3, 64, 64), \
                        f"Support images shape is not ({args.n_support * args.num_distinct_labels}, 3, 64, 64), but {this_support_images.shape}"
                    assert this_support_labels.shape == (args.n_support * args.num_distinct_labels, 2), \
                        f"Support labels shape is not ({args.n_support * args.num_distinct_labels}, 2), but {this_support_labels.shape}"
                
                out_dict = model(images, concept_extractor, transform, this_support_images, this_support_labels, args, unknown_init=unknown_init)
                out_dict.update({"LABELS": labels, "CONCEPTS": concepts})
                unknown_init = False
                
                loss, losses = _loss(out_dict, args)
                loss.backward()
                
                if epoch != 0:
                    if args.debug:  print("Update the schedulers.")
                    shape_optimizer.step()
                    color_optimizer.step()  

                if ys is None:
                    ys = out_dict["YS"]
                    y_true = out_dict["LABELS"]
                    cs = out_dict["pCS"]
                    cs_true = out_dict["CONCEPTS"]
                else:
                    ys = torch.concatenate((ys, out_dict["YS"]), dim=0)
                    y_true = torch.concatenate((y_true, out_dict["LABELS"]), dim=0)
                    cs = torch.concatenate((cs, out_dict["pCS"]), dim=0)
                    cs_true = torch.concatenate((cs_true, out_dict["CONCEPTS"]), dim=0)

                if i % 10 == 0:
                    progress_bar(i, len(unsup_train_loader) - 9, epoch, loss.item())

            # Step the scheduler (if using)
            shape_lr_scheduler.step()
            color_lr_scheduler.step()

            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            print("End of epoch ", epoch)
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            print()

            # ^ PHASE 4: Evaluation
            print('----------------------------------')
            print('--- Evaluation ---')

            if ys is None:
                # Skip evaluation if no unsupervised data was used for the model  
                torch.save(model.state_dict(), save_folder)
                concept_save_path = os.path.join(os.path.dirname(save_folder), f"best_{seed}.pt")
                concept_extractor.save(concept_save_path)
                print(f"Saved model after prototypical network training.")
                print()
                continue

            if "patterns" in args.task:
                y_true = y_true[:, -1]  # it is the last one

            # Get a random support set.
            if args.no_augmentations:
                this_support_images = kand_proto_dataset.images
                this_support_labels = kand_proto_dataset.labels
            else:
                this_support_images, this_support_labels = get_random_classes(
                    kand_proto_dataset.images, kand_proto_dataset.labels, args.n_support, args.num_distinct_labels)
                assert this_support_images.shape == (args.n_support * args.num_distinct_labels, 3, 64, 64), \
                    f"Support images shape is not ({args.n_support * args.num_distinct_labels}, 3, 64, 64), but {this_support_images.shape}"
                assert this_support_labels.shape == (args.n_support * args.num_distinct_labels, 2), \
                f"Support labels shape is not ({args.n_support * args.num_distinct_labels}, 2), but {this_support_labels.shape}"
            
            model.eval()
            tloss, cacc, yacc, f1 = evaluate_metrics(model, unsup_val_loader, args, 
                                    support_images=this_support_images, support_labels=this_support_labels,
                                    concept_extractor=concept_extractor, transform=transform)
            ### LOGGING ###
            fprint("  ACC C", cacc, "  ACC Y", yacc, "F1 Y", f1)
            print()
            
            if not args.tuning and cacc > best_cacc:
                print("Saving...")
                # Update best F1 score
                if best_cacc == 0.0 and args.debug:     print("Baseline accuracy has been determined.")
                best_cacc = cacc
                epochs_no_improve = 0
                    
                # Save the best model and the concept extractor
                torch.save(model.state_dict(), save_folder)
                concept_save_path = os.path.join(os.path.dirname(save_folder), f"best_{seed}.pt")
                concept_extractor.save(concept_save_path)
                print(f"Saved best model with CACC score: {best_cacc}")
                print()
            
            elif cacc <= best_cacc:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
        
        fprint("\n--- End of Training ---\n")
        return best_cacc


if __name__ == "__main__":
    # read arguments and setup environment
    args = setup.read_args()
    setup.setup_environment(args)
    signal.signal(signal.SIGINT, setup.sigint_handler)
    
    yaml_path = os.path.join(os.getcwd(), f"data/kand_config_yolo_{args.GPU_ID}/kand_config_yolo.yaml")
    my_yolo_project_path = f"ultralytics-{args.GPU_ID}/"
    my_yolo_premodel_path = f"ultralytics-{args.GPU_ID}/pretrained/yolo11n.pt"
    args.yolo_folder = my_yolo_project_path

    # create prototypical dataset
    proto_images, proto_labels, _ = get_my_initial_prototypes(args)
    kand_proto_dataset = my_datasets.PrimitivesDataset(proto_images, proto_labels, transform=None)
    shape_labels = kand_proto_dataset.labels[:, 0].numpy()
    color_labels = kand_proto_dataset.labels[:, 1].numpy()
    shape_sampler = batch_sampler.PrototypicalBatchSampler(shape_labels, 
                        args.classes_per_it, args.num_samples, args.iterations)
    color_sampler = batch_sampler.PrototypicalBatchSampler(color_labels, 
                        args.classes_per_it, args.num_samples, args.iterations)
    episodic_shape_dataloader = DataLoader(kand_proto_dataset, batch_sampler=shape_sampler)
    episodic_color_dataloader = DataLoader(kand_proto_dataset, batch_sampler=color_sampler)

    # unsupervised data loading and model creation
    unsup_dataset = get_dataset(args)
    n_images, c_split = unsup_dataset.get_split()
    encoder, decoder = unsup_dataset.get_backbone()
    model = get_model(args, encoder, decoder, n_images, c_split)    
    loss = model.get_loss(args)
    if args.retrain_extractor:
        yolo = YOLO(my_yolo_premodel_path)
    else:
        print("Using pretrained model: ", args.concept_extractor_path)
        yolo = YOLO(args.concept_extractor_path)
    assert len(model.state_dict()) > 0, "Model state dict is empty. Please check the model initialization."

    model.encoder[0].missing_classes = args.hide_shapes
    model.encoder[0].num_hidden = len(args.hide_shapes)
    model.encoder[1].missing_classes = args.hide_colors
    model.encoder[1].num_hidden = len(args.hide_colors)
    assert model.encoder[0].missing_classes == args.hide_shapes, "Shape encoder should have hidden classes"
    assert model.encoder[1].missing_classes == args.hide_colors, "Color encoder should have hidden classes"
    if args.hide_shapes:
        assert model.encoder[0].num_hidden > 0, "Shape encoder should have hidden classes"
        model.encoder[0].unknown_prototypes = nn.Parameter(
            torch.randn(model.encoder[0].num_hidden, 1024, requires_grad=True)
        )
    if args.hide_colors:
        assert model.encoder[1].num_hidden > 0, "Color encoder should have hidden classes"
        model.encoder[1].unknown_prototypes = nn.Parameter(
            torch.randn(model.encoder[1].num_hidden, 1024, requires_grad=True)
        )

    print("Using Dataset: ", unsup_dataset)
    print("Number of images: ", n_images)
    print("Using backbone: ", encoder)
    print("Using Model: ", model)
    print("Using Loss: ", loss)
    print("Working with taks: ", args.task)
    print("Shapes encoder: ", model.encoder[0].to(args.device))
    print("Colors encoder: ", model.encoder[1].to(args.device))

    print(f"*** Training model with weight with seed {args.seed}")
    print("Chosen device:", model.device)
    print("Save path for this model: ", args.save_path)
    if not os.path.exists(args.save_path): os.makedirs(args.save_path, exist_ok=True)
    save_folder = os.path.join(args.save_path, f"{args.model_parameter_name}_{args.seed}.pth")
    print("Saving in folder: ", save_folder)
    log_file_path = os.path.join(args.save_path, f"{args.model_parameter_name}_{args.seed}.log")
    train(model=model,                         
        concept_extractor=yolo,                              
        concept_extractor_training_path=yaml_path,           
        concept_extractor_project_path=args.yolo_folder,
        transform=T.Resize((64, 64)),                    
        episodic_shape_dataloader=episodic_shape_dataloader, 
        episodic_color_dataloader=episodic_color_dataloader, 
        unsup_dataset=unsup_dataset,
        _loss=loss, 
        args=args,
        seed=args.seed,
        save_folder=save_folder,
        log_file_path=log_file_path
    )
    save_model(model, args, args.seed)


print("End of experiment")
sys.exit(0)