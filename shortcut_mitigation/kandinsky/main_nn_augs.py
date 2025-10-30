import os 
import sys
import torch
import signal
import random
import numpy as np
import torch.nn.functional as F

from torch.utils.data import DataLoader
from protonet_kand_modules.utility_modules import setup
from protonet_kand_modules.data_modules import my_datasets

sys.path.append(os.path.abspath(".."))  
sys.path.append(os.path.abspath("../.."))  

from datasets import get_dataset
from models import get_model
from utils import fprint
from utils.status import progress_bar
from utils.checkpoint import save_model
from utils.metrics import evaluate_metrics
from utils.dpl_loss import ADDMNIST_DPL
from utils.checkpoint import save_model
from warmup_scheduler import GradualWarmupScheduler



def train(model,
        sup_train_loader:DataLoader,
        unsup_train_loader:DataLoader,
        unsup_val_loader:DataLoader,
        _loss: ADDMNIST_DPL, 
        args,
        seed,
        save_folder,
        sup_loss_weight=1.0,
        patience=5,
        debug=False):
    
     # for full reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = False
    
    best_cacc = 0.0
    epochs_no_improve = 0   # for early stopping

    model.to(model.device)

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
        # 1. Supervised phase: Teach the model to recognize primitives (merged in triplets)
        ###############################
        print("Start of supervised episodic training.")
        for i, (images, labels) in enumerate(sup_train_loader):
            sup_images = images.to(model.device)  # shape: (batch_size, 3, 64, 64)
            sup_labels = labels.to(model.device)  # shape: (batch_size, 6)
            batch_size = sup_images.size(0)

            assert sup_images.dim() == 4 and sup_images.size(1) == 3 \
                and sup_images.size(2) == 64 and sup_images.size(3) == 64, \
                f"Expected sup_images [B,3,64,64], got {sup_images.shape}"
            assert sup_labels.shape == torch.Size([batch_size, 6]), \
                f"Expected sup_labels [{batch_size},6], got {sup_labels.shape}"

            # make batch_size divisible by 3 by dropping the extra 1 or 2 samples
            if batch_size % 3 != 0:
                drop = batch_size % 3
                sup_images = sup_images[:-drop]
                sup_labels = sup_labels[:-drop]
                batch_size -= drop

            # now form triplets: 0 with 1 with 2, 3 with 4 with 5, ...
            merged_images = torch.cat([
                sup_images[0::3],   # first in each triplet
                sup_images[1::3],   # second
                sup_images[2::3]    # third
            ], dim=3)  # concat along width → new width = 64*3 = 192

            expected_bs = batch_size // 3
            assert merged_images.shape == torch.Size([expected_bs, 3, 64, 192]), \
                f"Expected merged_images [{expected_bs},3,64,192], got {merged_images.shape}"

            # extract labels for each of the three primitives in the triplet
            labels_first  = sup_labels[0::3]  # [bs//3, 6]
            labels_second = sup_labels[1::3]  # [bs//3, 6]
            labels_third  = sup_labels[2::3]  # [bs//3, 6]

            # Forward pass: now feeding the concatenated triplets
            out_dict = model(merged_images)
            logits = out_dict["CS"]
            
            B = logits.size(0)
            num_objects = 6
            num_classes = 3

            def triplet_loss(logits_slice, labels_slice):
                l = logits_slice.reshape(B, num_objects, num_classes)
                l_flat = l.reshape(B * num_objects, num_classes)
                t_flat = labels_slice.reshape(B * num_objects)
                return F.cross_entropy(l_flat, t_flat)

            logits1 = logits[:, 0, :]  # [B, 18]
            logits2 = logits[:, 1, :]
            logits3 = logits[:, 2, :]

            # compute individual losses
            loss1 = triplet_loss(logits1, labels_first)
            loss2 = triplet_loss(logits2, labels_second)
            loss3 = triplet_loss(logits3, labels_third)

            # total concept loss (you can average instead of sum if you prefer)
            concept_loss = sup_loss_weight * (loss1 + loss2 + loss3)

            # backprop
            model.opt.zero_grad()
            concept_loss.backward()
            model.opt.step()

            if i % 10 == 0:
                print(f"Supervised phase, Epoch {epoch}, Batch {i}: "
                    f"Concept Loss = {concept_loss.item():.4f}")
        
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
                images.to(model.device),
                labels.to(model.device),
                concepts.to(model.device),
            )

            # ^ baseline model
            out_dict = model(images)
            out_dict.update({"LABELS": labels, "CONCEPTS": concepts})
            
            model.opt.zero_grad()
            loss, losses = _loss(out_dict, args)

            loss.backward()
            model.opt.step()

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

        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print("End of epoch ", epoch)
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print()

        if args.uns_parameter_percentage == 0.0:
            print("Saving...")
            torch.save(model.state_dict(), save_folder)
            continue

        y_pred = torch.argmax(ys, dim=-1)
        #print("Argmax predictions have shape: ", y_pred.shape)

        if "patterns" in args.task:
            y_true = y_true[:, -1]  # it is the last one

        model.eval()
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

        if not args.tuning and cacc > best_cacc:
            print("Saving...")
            # Update best F1 score
            best_cacc = cacc
            epochs_no_improve = 0
                
            # Save the best model and the concept extractor
            torch.save(model.state_dict(), save_folder)
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
    args = setup.read_args(baseline=True)
    setup.setup_environment(args)
    signal.signal(signal.SIGINT, setup.sigint_handler)

    # supervised learning for primitives
    proto_images = torch.load('data/kand_annotations/yolo_annotations/images.pt')
    proto_labels = torch.load('data/kand_annotations/yolo_annotations/labels.pt')
    kand_sup_dataset = my_datasets.SupervisedDataset(proto_images, proto_labels, transform=None)
    sup_train_loader = DataLoader(kand_sup_dataset, batch_size=args.batch_size, shuffle=True)

    # unsupervised learning for primitives 
    dataset = get_dataset(args)
    unsup_train_loader, unsup_val_loader, unsup_test_loader = dataset.get_data_loaders()
    dataset.print_stats()    
    n_images, c_split = dataset.get_split()
    encoder, decoder = dataset.get_backbone()
    model = get_model(args, encoder, decoder, n_images, c_split)
    loss = model.get_loss(args)
    model.start_optim(args)
    print("Using Dataset: ", dataset)
    print("Number of images: ", n_images)
    print("Using backbone: ", encoder)
    print("Using Model: ", model)
    print("Using Loss: ", loss)
    print("Working with taks: ", args.task)

    # training
    f1_scores = dict()
    print(f"*** Training model with seed {args.seed}")
    print("Chosen device:", model.device)
    if not os.path.exists(args.save_path): os.makedirs(args.save_path, exist_ok=True)
    save_folder = os.path.join(args.save_path, f"{args.model_parameter_name}_{args.seed}.pth")
    print("Saving model in folder: ", save_folder)

    best_f1 = train(model=model,
            sup_train_loader=sup_train_loader,
            unsup_train_loader=unsup_train_loader,
            unsup_val_loader=unsup_val_loader,
            _loss=loss, 
            args=args,
            seed=args.seed,
            save_folder=save_folder,
            sup_loss_weight=args.concept_loss_weight,
            debug=False
        )
    f1_scores[(args.seed)] = best_f1
    save_model(model, args, args.seed)  # save the model parameters

    print(f"*** Finished training model with seed {args.seed}")

    print("Training finished.")
    best_weight_seed = max(f1_scores, key=f1_scores.get)
    print(f"Best weight and seed combination: {best_weight_seed} with F1 score: {f1_scores[best_weight_seed]}")