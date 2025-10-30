import os
import sys
import signal
import torch
import random
import numpy as np
import torch.nn.functional as F

from argparse import Namespace
from torch.utils.data import Dataset, DataLoader
from warmup_scheduler import GradualWarmupScheduler

from baseline_modules.utility_modules import setup
from baseline_modules.utility_modules.run_evaluation import evaluate_my_model
from baseline_modules.supervision_modules import build_sup_set_joint
from baseline_modules.supervision_modules.joint_pretraining import pre_train

from protonet_STOP_bddoia_modules.proto_modules.proto_helpers import assert_inputs
from baseline_modules.supervision_modules.build_sup_set_joint import get_augmented_train_loader

sys.path.append(os.path.abspath(".."))  
sys.path.append(os.path.abspath("../.."))  

from utils import fprint
from utils.status import progress_bar
from utils.metrics import evaluate_metrics
from utils.dpl_loss import ADDMNIST_DPL
from utils.checkpoint import save_model

from models import get_model
from models.mnistdpl import MnistDPL
from datasets import get_dataset


# * Main Training Loop
def train(
        model: MnistDPL, 
        _loss: ADDMNIST_DPL,
        save_path: str, 
        train_loader: DataLoader,
        val_loader: DataLoader,
        args: Namespace,
        backbone_supervision_loader: DataLoader = None,
        eval_concepts: list = None,
        seed: int = 0,
    ) -> float:

    # for full reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = False
    
    # early stopping
    best_cacc = 0.0
    epochs_no_improve = 0
    
    # scheduler & warmup (not used) for main model
    scheduler = torch.optim.lr_scheduler.ExponentialLR(model.opt, args.exp_decay)
    w_scheduler = None
    if args.warmup_steps > 0:
        w_scheduler = GradualWarmupScheduler(model.opt, 1.0, args.warmup_steps)

    fprint("\n--- Start of Training ---\n")
    model.to(model.device)
    model.opt.zero_grad()
    model.opt.step()

    # start optimizer
    enc_opt = torch.optim.Adam(model.encoder.parameters(), args.lr, weight_decay=args.weight_decay)

    if args.c and backbone_supervision_loader is None:
        raise RuntimeError("Backbone supervision loader must be provided when using augmentation.")
        sys.exit(1)

    # ^ Training start
    for epoch in range(args.n_epochs):
        print(f"Epoch {epoch+1}/{args.n_epochs}")
        
        model.train()

        # * Pretraining
        if args.Apretrained:
            fprint("\n--- Start of pretraining ---\n")
            for i, batch in enumerate(backbone_supervision_loader):
                batch_embeds, batch_labels = batch
                batch_embeds = batch_embeds.to(model.device)
                batch_labels = batch_labels.to(model.device)

                enc_opt.zero_grad()
                preds = model.encoder(batch_embeds)
                assert preds.shape == (batch_embeds.shape[0], 21), f"Expected shape ({batch_embeds.shape[0]}, 21), got {preds.shape}"
                loss = F.binary_cross_entropy(preds, batch_labels.float())

                loss.backward()
                enc_opt.step()

                progress_bar(i, len(train_loader), epoch, loss.item())
            

        # * Backbone supervision phase
        if args.c:
            print("Backbone supervision phase")
            for i, batch in enumerate(backbone_supervision_loader):
                batch_embeds, batch_labels = batch
                batch_embeds = batch_embeds.to(model.device)
                batch_labels = batch_labels.to(model.device)

                model.opt.zero_grad()
                out_dict = model(batch_embeds)
                concept_predictions = out_dict["CS"]
                loss = F.binary_cross_entropy(concept_predictions, batch_labels.float())

                loss.backward()
                model.opt.step()

                progress_bar(i, len(backbone_supervision_loader), epoch, loss.item())

        # * Unsupervised Training
        if args.c:            
            print("\nUnsupervised training phase\n")

        ys, y_true, cs, cs_true, batch = None, None, None, None, 0
        for i, batch in enumerate(train_loader):

            if random.random() > args.uns_parameter_percentage:
                continue  # Skip this batch with probability (1 - percentage)
            
            # ------------------ original embneddings
            images_embeddings = torch.stack(batch['embeddings']).to(model.device)
            attr_labels = torch.stack(batch['attr_labels']).to(model.device)
            class_labels = torch.stack(batch['class_labels'])[:,:-1].to(model.device)
            # ------------------ my extracted features
            images_embeddings_raw = torch.stack(batch['embeddings_raw']).to(model.device)
            detected_rois = batch['rois']
            detected_rois_feats = batch['roi_feats']
            detection_labels = batch['detection_labels']
            detection_scores = batch['detection_scores']
            assert_inputs(images_embeddings, attr_labels, class_labels,
                   detected_rois_feats, detected_rois, detection_labels,
                   detection_scores, images_embeddings_raw)

            out_dict = model(images_embeddings_raw)
            out_dict.update({"LABELS": class_labels, "CONCEPTS": attr_labels})
            
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
                progress_bar(i, len(train_loader) - 9, epoch, loss.item())
            
        # ^ Evaluation phase
        model.eval()
        
        my_metrics = evaluate_metrics(
                model=model, 
                loader=val_loader, 
                args=args,
                eval_concepts=eval_concepts)
        loss = my_metrics[0]
        cacc = my_metrics[1]
        yacc = my_metrics[2]
        f1_y = my_metrics[3]
       
        # update at end of the epoch
        if epoch < args.warmup_steps:   w_scheduler.step()
        else:
            scheduler.step()
            if hasattr(_loss, "grade"):
                _loss.update_grade(epoch)

        ### LOGGING ###
        fprint("  ACC C", cacc, "  ACC Y", yacc, "F1 Y", f1_y)
        
        if not args.tuning and cacc > best_cacc:
            print("Saving...")
            # Update best F1 score
            best_cacc = cacc
            epochs_no_improve = 0

            # Save the best model
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model with CACC: {best_cacc}")

        elif cacc <= best_cacc:
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break


    fprint("\n--- End of Training ---\n")
    return best_cacc



if __name__ == "__main__":
    # & read arguments and setup environment
    args = setup.read_args()
    setup.setup_environment(args)
    signal.signal(signal.SIGINT, setup.sigint_handler)

    # & data loading and model setup
    dataset = get_dataset(args)
    n_images, c_split = dataset.get_split()
    encoder, decoder = dataset.get_backbone()
    model = get_model(args, encoder, decoder, n_images, c_split)
    model.start_optim(args)
    loss = model.get_loss(args)
    print(dataset)
    print("Using Dataset: ", dataset)
    print("Using backbone: ", encoder)
    print("Using Model: ", model)
    print("Using Loss: ", loss)
    unsup_train_loader, unsup_val_loader, unsup_test_loader = dataset.get_data_loaders(args=args)

    # & create supervisions for aggregated backbones training
    aggregated_dataloader = build_sup_set_joint.get_augmented_train_loader(
            unsup_train_loader=unsup_train_loader, device=model.device, args=args
    )
    assert hasattr(aggregated_dataloader, "__iter__"), "aggregated_dataloader must be an iterable (Dataloader object)"

    # & Training
    print(f"*** Training model with seed {args.seed}")
    print("Chosen device:", model.device)
    if not os.path.exists(args.save_path): os.makedirs(args.save_path, exist_ok=True)
    save_folder = os.path.join(args.save_path, f"{args.model_parameter_name}_{args.seed}.pth")
    print("Saving model in folder: ", save_folder)
    eval_concepts = [
        'green_lights', 
        'follow_traffic', 
        'road_clear',
        'traffic_lights', 
        'traffic_signs', 
        'cars', 
        'pedestrians', 
        'riders', 
        'others',
        'no_lane_left', 
        'obstacle_left_lane', 
        'solid_left_line',
        'on_right_turn_lane', 
        'traffic_light_right', 
        'front_car_right', 
        'no_lane_right', 
        'obstacle_right_lane', 
        'solid_right_line',
        'on_left_turn_lane', 
        'traffic_light_left', 
        'front_car_left'
    ]
    # * Pretraining (if specified)
    if args.pretrained:
        pre_train(model, aggregated_dataloader, args)
    # ! Standard Training
    best_cacc = train(
            model=model,
            train_loader=unsup_train_loader,
            val_loader=unsup_val_loader,
            backbone_supervision_loader=aggregated_dataloader,
            save_path=save_folder,
            _loss=loss,
            args=args,
            eval_concepts=eval_concepts,
            seed=args.seed,
    )
    save_model(model, args, args.seed)  # save the model parameters
    print(f"*** Finished training model with seed {args.seed} and best CACC score {best_cacc}")
    print("Training finished.")

    # & Evaluation
    model = get_model(args, encoder, decoder, n_images, c_split)
    model_state_dict = torch.load(save_folder)
    model.load_state_dict(model_state_dict)
    evaluate_my_model(model, save_folder, unsup_test_loader, eval_concepts=eval_concepts, args=args)
    print("End of experiment")
    sys.exit(0)