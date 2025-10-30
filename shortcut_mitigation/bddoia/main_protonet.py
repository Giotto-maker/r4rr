import os
import sys
import signal
import torch
import random
import numpy as np

from typing import List
from argparse import Namespace
from torch.utils.data import Dataset, DataLoader
from warmup_scheduler import GradualWarmupScheduler

from protonet_STOP_bddoia_modules.proto_modules.proto_helpers import assert_inputs
from protonet_STOP_bddoia_modules.proto_modules.proto_functions import train_my_prototypical_network
from baseline_modules.utility_modules.run_evaluation import evaluate_my_model
from protonet_bddoia_modules.utility_modules import setup
from protonet_bddoia_modules.utility_modules.other_utils import (
    check_optimizer_params, 
    get_per_class_support_set,
    save_prototypes
)
from protonet_bddoia_modules.data_modules.proto_data import build_prototypical_dataloaders

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
from backbones.bddoia_protonet import PrototypicalLoss


# * Main Training Loop
def train(
        model: MnistDPL, 
        _loss: ADDMNIST_DPL,
        save_path: str, 
        proto_datasets: dict,
        proto_dataloaders: dict,
        train_loader: DataLoader,
        val_loader: DataLoader,
        pos_examples: dict,
        args: Namespace,
        seed: int = 0,
        eval_concepts: List[str] = ['traffic_lights', 'traffic_signs', 'cars', 'pedestrians', 'riders', 'others'],
        debug=False,
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

    # scheduler & warmup (if used) for main model
    scheduler = torch.optim.lr_scheduler.ExponentialLR(model.opt, args.exp_decay)
    w_scheduler = None
    if args.warmup_steps > 0:
        w_scheduler = GradualWarmupScheduler(model.opt, 1.0, args.warmup_steps)
    
    # --------------------------------------
    # ^ 0. PROTOTYPICAL NETWORKS SCHEDULERS & OPTIMIZERS
    # --------------------------------------
    green_traffic_lights_opt = torch.optim.Adam(model.encoder[0].parameters())
    green_traffic_lights_scheduler = torch.optim.lr_scheduler.StepLR(green_traffic_lights_opt, step_size=10, gamma=0.5)

    follow_traffic_opt = torch.optim.Adam(model.encoder[1].parameters())
    follow_traffic_scheduler = torch.optim.lr_scheduler.StepLR(follow_traffic_opt, step_size=10, gamma=0.5)

    road_is_clear_opt = torch.optim.Adam(model.encoder[2].parameters())
    road_is_clear_scheduler = torch.optim.lr_scheduler.StepLR(road_is_clear_opt, step_size=10, gamma=0.5)

    red_traffic_lights_opt = torch.optim.Adam(model.encoder[3].parameters())
    red_traffic_lights_scheduler = torch.optim.lr_scheduler.StepLR(red_traffic_lights_opt, step_size=10, gamma=0.5)

    traffic_sign_opt = torch.optim.Adam(model.encoder[4].parameters())
    traffic_sign_scheduler = torch.optim.lr_scheduler.StepLR(traffic_sign_opt, step_size=10, gamma=0.5)

    obstacle_car_opt = torch.optim.Adam(model.encoder[5].parameters())
    obstacle_car_scheduler = torch.optim.lr_scheduler.StepLR(obstacle_car_opt, step_size=10, gamma=0.5)

    obstacle_person_opt = torch.optim.Adam(model.encoder[6].parameters())
    obstacle_person_scheduler = torch.optim.lr_scheduler.StepLR(obstacle_person_opt, step_size=10, gamma=0.5)

    obstacle_rider_opt = torch.optim.Adam(model.encoder[7].parameters())
    obstacle_rider_scheduler = torch.optim.lr_scheduler.StepLR(obstacle_rider_opt, step_size=10, gamma=0.5)

    obstacle_other_opt = torch.optim.Adam(model.encoder[8].parameters())
    obstacle_other_scheduler = torch.optim.lr_scheduler.StepLR(obstacle_other_opt, step_size=10, gamma=0.5)

    no_lane_left_opt = torch.optim.Adam(model.encoder[9].parameters())
    no_lane_left_scheduler = torch.optim.lr_scheduler.StepLR(no_lane_left_opt, step_size=10, gamma=0.5)

    obstacles_left_lane_opt = torch.optim.Adam(model.encoder[10].parameters())
    obstacles_left_lane_scheduler = torch.optim.lr_scheduler.StepLR(obstacles_left_lane_opt, step_size=10, gamma=0.5)

    solid_line_left_opt = torch.optim.Adam(model.encoder[11].parameters())
    solid_line_left_scheduler = torch.optim.lr_scheduler.StepLR(solid_line_left_opt, step_size=10, gamma=0.5)

    right_turn_lane_opt = torch.optim.Adam(model.encoder[12].parameters())
    right_turn_lane_scheduler = torch.optim.lr_scheduler.StepLR(right_turn_lane_opt, step_size=10, gamma=0.5)

    traffic_light_allows_right_opt = torch.optim.Adam(model.encoder[13].parameters())
    traffic_light_allows_right_scheduler = torch.optim.lr_scheduler.StepLR(traffic_light_allows_right_opt, step_size=10, gamma=0.5)

    front_car_turning_right_opt = torch.optim.Adam(model.encoder[14].parameters())
    front_car_turning_right_scheduler = torch.optim.lr_scheduler.StepLR(front_car_turning_right_opt, step_size=10, gamma=0.5)

    no_lane_right_opt = torch.optim.Adam(model.encoder[15].parameters())
    no_lane_right_scheduler = torch.optim.lr_scheduler.StepLR(no_lane_right_opt, step_size=10, gamma=0.5)

    obstacles_right_lane_opt = torch.optim.Adam(model.encoder[16].parameters())
    obstacles_right_lane_scheduler = torch.optim.lr_scheduler.StepLR(obstacles_right_lane_opt, step_size=10, gamma=0.5)

    solid_line_right_opt = torch.optim.Adam(model.encoder[17].parameters())
    solid_line_right_scheduler = torch.optim.lr_scheduler.StepLR(solid_line_right_opt, step_size=10, gamma=0.5)

    left_turn_lane_opt = torch.optim.Adam(model.encoder[18].parameters())
    left_turn_lane_scheduler = torch.optim.lr_scheduler.StepLR(left_turn_lane_opt, step_size=10, gamma=0.5)

    traffic_light_allows_left_opt = torch.optim.Adam(model.encoder[19].parameters())
    traffic_light_allows_left_scheduler = torch.optim.lr_scheduler.StepLR(traffic_light_allows_left_opt, step_size=10, gamma=0.5)

    front_car_turning_left_opt = torch.optim.Adam(model.encoder[20].parameters())
    front_car_turning_left_scheduler = torch.optim.lr_scheduler.StepLR(front_car_turning_left_opt, step_size=10, gamma=0.5)

    fprint("\n--- Start of Training ---\n")
    for i in range(len(model.encoder)):
        model.encoder[i].train()
        model.encoder[i].to(model.device)
        
    # --------------------------------------
    # ^ 1. PROTOTYPICAL NETWORKS TRAINING
    # --------------------------------------
    print('----------------------------------')
    print('--- Prototypical Networks Training ---')        
        
    pNet_loss = PrototypicalLoss(n_support=args.num_support)
    for epoch in range(args.n_epochs):

        for e in range(args.proto_epochs):        
            print(f"Prototypical Networks Training Epoch {e + 1}/{args.proto_epochs}")
            epoch_loss_green_traffic_lights, epoch_acc_green_traffic_lights = train_my_prototypical_network(
                    proto_dataloaders[0], args.iterations, model.encoder[0], green_traffic_lights_opt, pNet_loss
                )
            epoch_loss_follow_traffic, epoch_acc_follow_traffic = train_my_prototypical_network(
                    proto_dataloaders[1], args.iterations, model.encoder[1], follow_traffic_opt, pNet_loss
                )
            epoch_loss_road_is_clear, epoch_acc_road_is_clear = train_my_prototypical_network(
                    proto_dataloaders[2], args.iterations, model.encoder[2], road_is_clear_opt, pNet_loss
                )
            epoch_loss_red_traffic_lights, epoch_acc_red_traffic_lights = train_my_prototypical_network(
                    proto_dataloaders[3], args.iterations, model.encoder[3], red_traffic_lights_opt, pNet_loss
                )
            epoch_loss_traffic_sign, epoch_acc_traffic_sign = train_my_prototypical_network(
                    proto_dataloaders[4], args.iterations, model.encoder[4], traffic_sign_opt, pNet_loss
                )
            epoch_loss_obstacle_car, epoch_acc_obstacle_car = train_my_prototypical_network(
                    proto_dataloaders[5], args.iterations, model.encoder[5], obstacle_car_opt, pNet_loss
                )
            epoch_loss_obstacle_person, epoch_acc_obstacle_person = train_my_prototypical_network(
                    proto_dataloaders[6], args.iterations, model.encoder[6], obstacle_person_opt, pNet_loss
                )
            epoch_loss_obstacle_rider, epoch_acc_obstacle_rider = train_my_prototypical_network(
                    proto_dataloaders[7], args.iterations, model.encoder[7], obstacle_rider_opt, pNet_loss
                )
            epoch_loss_obstacle_other, epoch_acc_obstacle_other = train_my_prototypical_network(
                    proto_dataloaders[8], args.iterations, model.encoder[8], obstacle_other_opt, pNet_loss
                )
            epoch_loss_no_lane_left, epoch_acc_no_lane_left = train_my_prototypical_network(
                    proto_dataloaders[9], args.iterations, model.encoder[9], no_lane_left_opt, pNet_loss
                )
            epoch_loss_obstacles_left_lane, epoch_acc_obstacles_left_lane = train_my_prototypical_network(
                    proto_dataloaders[10], args.iterations, model.encoder[10], obstacles_left_lane_opt, pNet_loss
                )
            epoch_loss_solid_line_left, epoch_acc_solid_line_left = train_my_prototypical_network(
                    proto_dataloaders[11], args.iterations, model.encoder[11], solid_line_left_opt, pNet_loss
                )
            epoch_loss_right_turn_lane, epoch_acc_right_turn_lane = train_my_prototypical_network(
                    proto_dataloaders[12], args.iterations, model.encoder[12], right_turn_lane_opt, pNet_loss
                )
            epoch_loss_traffic_light_allows_right, epoch_acc_traffic_light_allows_right = train_my_prototypical_network(
                    proto_dataloaders[13], args.iterations, model.encoder[13], traffic_light_allows_right_opt, pNet_loss
                )
            epoch_loss_front_car_turning_right, epoch_acc_front_car_turning_right = train_my_prototypical_network(
                    proto_dataloaders[14], args.iterations, model.encoder[14], front_car_turning_right_opt, pNet_loss
                )
            epoch_loss_no_lane_right, epoch_acc_no_lane_right = train_my_prototypical_network(
                    proto_dataloaders[15], args.iterations, model.encoder[15], no_lane_right_opt, pNet_loss
                )
            epoch_loss_obstacles_right_lane, epoch_acc_obstacles_right_lane = train_my_prototypical_network(
                    proto_dataloaders[16], args.iterations, model.encoder[16], obstacles_right_lane_opt, pNet_loss  
                )
            epoch_loss_solid_line_right, epoch_acc_solid_line_right = train_my_prototypical_network(
                    proto_dataloaders[17], args.iterations, model.encoder[17], solid_line_right_opt, pNet_loss  
                )
            epoch_loss_left_turn_lane, epoch_acc_left_turn_lane = train_my_prototypical_network(
                    proto_dataloaders[18], args.iterations, model.encoder[18], left_turn_lane_opt, pNet_loss
                )
            epoch_loss_traffic_light_allows_left, epoch_acc_traffic_light_allows_left = train_my_prototypical_network(
                    proto_dataloaders[19], args.iterations, model.encoder[19], traffic_light_allows_left_opt, pNet_loss
                )
            epoch_loss_front_car_turning_left, epoch_acc_front_car_turning_left = train_my_prototypical_network(
                    proto_dataloaders[20], args.iterations, model.encoder[20], front_car_turning_left_opt, pNet_loss
                )
        
        avg_loss_gl = sum(epoch_loss_green_traffic_lights) / len(epoch_loss_green_traffic_lights)
        avg_acc_gl  = sum(epoch_acc_green_traffic_lights)  / len(epoch_acc_green_traffic_lights)
        print(f"Traffic Lights Features  - Avg Loss: {avg_loss_gl:.4f} | Avg Acc: {avg_acc_gl:.4f}")
        
        avg_loss_ft = sum(epoch_loss_follow_traffic) / len(epoch_loss_follow_traffic)
        avg_acc_ft  = sum(epoch_acc_follow_traffic)  / len(epoch_acc_follow_traffic)
        print(f"Follow Traffic Features   - Avg Loss: {avg_loss_ft:.4f} | Avg Acc: {avg_acc_ft:.4f}")
        
        avg_loss_ric = sum(epoch_loss_road_is_clear) / len(epoch_loss_road_is_clear)
        avg_acc_ric  = sum(epoch_acc_road_is_clear)  / len(epoch_acc_road_is_clear)
        print(f"Road Is Clear Features    - Avg Loss: {avg_loss_ric:.4f} | Avg Acc: {avg_acc_ric:.4f}")
        
        avg_loss_rtl = sum(epoch_loss_red_traffic_lights) / len(epoch_loss_red_traffic_lights)
        avg_acc_rtl  = sum(epoch_acc_red_traffic_lights)  / len(epoch_acc_red_traffic_lights)
        print(f"Red Traffic Lights Features - Avg Loss: {avg_loss_rtl:.4f} | Avg Acc: {avg_acc_rtl:.4f}")
        
        avg_loss_ts = sum(epoch_loss_traffic_sign) / len(epoch_loss_traffic_sign)
        avg_acc_ts  = sum(epoch_acc_traffic_sign)  / len(epoch_acc_traffic_sign)
        print(f"Traffic Sign Features     - Avg Loss: {avg_loss_ts:.4f} | Avg Acc: {avg_acc_ts:.4f}")
        
        avg_loss_oc = sum(epoch_loss_obstacle_car) / len(epoch_loss_obstacle_car)
        avg_acc_oc  = sum(epoch_acc_obstacle_car)  / len(epoch_acc_obstacle_car)
        print(f"Obstacle Car Features     - Avg Loss: {avg_loss_oc:.4f} | Avg Acc: {avg_acc_oc:.4f}")
        
        avg_loss_op = sum(epoch_loss_obstacle_person) / len(epoch_loss_obstacle_person)
        avg_acc_op  = sum(epoch_acc_obstacle_person)  / len(epoch_acc_obstacle_person)
        print(f"Obstacle Person Features  - Avg Loss: {avg_loss_op:.4f} | Avg Acc: {avg_acc_op:.4f}")
        
        avg_loss_or = sum(epoch_loss_obstacle_rider) / len(epoch_loss_obstacle_rider)
        avg_acc_or  = sum(epoch_acc_obstacle_rider)  / len(epoch_acc_obstacle_rider)
        print(f"Obstacle Rider Features   - Avg Loss: {avg_loss_or:.4f} | Avg Acc: {avg_acc_or:.4f}")
        
        avg_loss_oo = sum(epoch_loss_obstacle_other) / len(epoch_loss_obstacle_other)
        avg_acc_oo  = sum(epoch_acc_obstacle_other)  / len(epoch_acc_obstacle_other)
        print(f"Obstacle Other Features   - Avg Loss: {avg_loss_oo:.4f} | Avg Acc: {avg_acc_oo:.4f}")
        
        avg_loss_nll = sum(epoch_loss_no_lane_left) / len(epoch_loss_no_lane_left)
        avg_acc_nll  = sum(epoch_acc_no_lane_left)  / len(epoch_acc_no_lane_left)
        print(f"No Lane Left Features     - Avg Loss: {avg_loss_nll:.4f} | Avg Acc: {avg_acc_nll:.4f}")
        
        avg_loss_oll = sum(epoch_loss_obstacles_left_lane) / len(epoch_loss_obstacles_left_lane)
        avg_acc_oll  = sum(epoch_acc_obstacles_left_lane)  / len(epoch_acc_obstacles_left_lane)
        print(f"Obstacles Left Lane Features - Avg Loss: {avg_loss_oll:.4f} | Avg Acc: {avg_acc_oll:.4f}")
        
        avg_loss_sll = sum(epoch_loss_solid_line_left) / len(epoch_loss_solid_line_left)
        avg_acc_sll  = sum(epoch_acc_solid_line_left)  / len(epoch_acc_solid_line_left)
        print(f"Solid Line Left Features  - Avg Loss: {avg_loss_sll:.4f} | Avg Acc: {avg_acc_sll:.4f}")
        
        avg_loss_rtl = sum(epoch_loss_right_turn_lane) / len(epoch_loss_right_turn_lane)
        avg_acc_rtl  = sum(epoch_acc_right_turn_lane)  / len(epoch_acc_right_turn_lane)
        print(f"Right Turn Lane Features  - Avg Loss: {avg_loss_rtl:.4f} | Avg Acc: {avg_acc_rtl:.4f}")
        
        avg_loss_tlar = sum(epoch_loss_traffic_light_allows_right) / len(epoch_loss_traffic_light_allows_right)
        avg_acc_tlar  = sum(epoch_acc_traffic_light_allows_right)  / len(epoch_acc_traffic_light_allows_right)
        print(f"Traffic Light Allows Right Features - Avg Loss: {avg_loss_tlar:.4f} | Avg Acc: {avg_acc_tlar:.4f}")
        
        avg_loss_fctr = sum(epoch_loss_front_car_turning_right) / len(epoch_loss_front_car_turning_right)
        avg_acc_fctr  = sum(epoch_acc_front_car_turning_right)  / len(epoch_acc_front_car_turning_right)
        print(f"Front Car Turning Right Features - Avg Loss: {avg_loss_fctr:.4f} | Avg Acc: {avg_acc_fctr:.4f}")
        
        avg_loss_nlr = sum(epoch_loss_no_lane_right) / len(epoch_loss_no_lane_right)
        avg_acc_nlr  = sum(epoch_acc_no_lane_right)  / len(epoch_acc_no_lane_right)
        print(f"No Lane Right Features    - Avg Loss: {avg_loss_nlr:.4f} | Avg Acc: {avg_acc_nlr:.4f}")
        
        avg_loss_orl = sum(epoch_loss_obstacles_right_lane) / len(epoch_loss_obstacles_right_lane)
        avg_acc_orl  = sum(epoch_acc_obstacles_right_lane)  / len(epoch_acc_obstacles_right_lane)
        print(f"Obstacles Right Lane Features - Avg Loss: {avg_loss_orl:.4f} | Avg Acc: {avg_acc_orl:.4f}")
        
        avg_loss_slr = sum(epoch_loss_solid_line_right) / len(epoch_loss_solid_line_right)
        avg_acc_slr  = sum(epoch_acc_solid_line_right)  / len(epoch_loss_solid_line_right)
        print(f"Solid Line Right Features - Avg Loss: {avg_loss_slr:.4f} | Avg Acc: {avg_acc_slr:.4f}")
        
        avg_loss_ltl = sum(epoch_loss_left_turn_lane) / len(epoch_loss_left_turn_lane)
        avg_acc_ltl  = sum(epoch_acc_left_turn_lane)  / len(epoch_acc_left_turn_lane)
        print(f"Left Turn Lane Features  - Avg Loss: {avg_loss_ltl:.4f} | Avg Acc: {avg_acc_ltl:.4f}")
        
        avg_loss_tlal = sum(epoch_loss_traffic_light_allows_left) / len(epoch_loss_traffic_light_allows_left)
        avg_acc_tlal  = sum(epoch_acc_traffic_light_allows_left)  / len(epoch_acc_traffic_light_allows_left)
        print(f"Traffic Light Allows Left Features - Avg Loss: {avg_loss_tlal:.4f} | Avg Acc: {avg_acc_tlal:.4f}")
        
        avg_loss_fctl = sum(epoch_loss_front_car_turning_left) / len(epoch_loss_front_car_turning_left)
        avg_acc_fctl  = sum(epoch_acc_front_car_turning_left)  / len(epoch_acc_front_car_turning_left)
        print(f"Front Car Turning Left Features - Avg Loss: {avg_loss_fctl:.4f} | Avg Acc: {avg_acc_fctl:.4f}\n")

        green_traffic_lights_scheduler.step()
        follow_traffic_scheduler.step()
        road_is_clear_scheduler.step()
        red_traffic_lights_scheduler.step()
        traffic_sign_scheduler.step()
        obstacle_car_scheduler.step()
        obstacle_person_scheduler.step()
        obstacle_rider_scheduler.step()
        obstacle_other_scheduler.step()
        no_lane_left_scheduler.step()
        obstacles_left_lane_scheduler.step()
        solid_line_left_scheduler.step()
        right_turn_lane_scheduler.step()
        traffic_light_allows_right_scheduler.step()
        front_car_turning_right_scheduler.step()
        no_lane_right_scheduler.step()
        obstacles_right_lane_scheduler.step()
        solid_line_right_scheduler.step()
        left_turn_lane_scheduler.step()
        traffic_light_allows_left_scheduler.step()
        front_car_turning_left_scheduler.step()

        # --------------------------------------
        # ^ 3. MAIN MODEL TRAINING
        # --------------------------------------
        model.to(model.device)
        model.train()
        model.opt.zero_grad()
        model.opt.step()
        
        print(f"Main Model Training Epoch {epoch + 1}/{args.n_epochs}")  
        ys, y_true, cs, cs_true, batch = None, None, None, None, 0
        for i, batch in enumerate(train_loader):
            # ------------------ original embeddings
            images_embeddings = torch.stack(batch['embeddings']).to(model.device)
            attr_labels = torch.stack(batch['attr_labels']).to(model.device)
            class_labels = torch.stack(batch['class_labels'])[:,:-1].to(model.device) # exclude the last column
            # ------------------ my extracted features
            images_embeddings_raw = torch.stack(batch['embeddings_raw']).to(model.device)
            detected_rois = batch['rois']
            detected_rois_feats = batch['roi_feats']
            detection_labels = batch['detection_labels']
            detection_scores = batch['detection_scores']
            assert_inputs(images_embeddings, attr_labels, class_labels,
                    detected_rois_feats, detected_rois, detection_labels,
                    detection_scores, images_embeddings_raw)
            
            support_emb_gl, support_labels_gl = get_per_class_support_set(
                proto_datasets=proto_datasets, pos_examples=pos_examples, class_idx=0, device=model.device
            )
            support_emb_ft, support_labels_ft = get_per_class_support_set(
                proto_datasets=proto_datasets, pos_examples=pos_examples, class_idx=1, device=model.device
            )
            support_emb_ric, support_labels_ric = get_per_class_support_set(
                proto_datasets=proto_datasets, pos_examples=pos_examples, class_idx=2, device=model.device
            )
            support_emb_tl, support_labels_tl = get_per_class_support_set(
                proto_datasets=proto_datasets, pos_examples=pos_examples, class_idx=3, device=model.device
            )
            support_emb_ts, support_labels_ts = get_per_class_support_set(
                proto_datasets=proto_datasets, pos_examples=pos_examples, class_idx=4, device=model.device
            )
            support_emb_oc, support_labels_oc = get_per_class_support_set(
                proto_datasets=proto_datasets, pos_examples=pos_examples, class_idx=5, device=model.device
            )
            support_emb_op, support_labels_op = get_per_class_support_set(
                proto_datasets=proto_datasets, pos_examples=pos_examples, class_idx=6, device=model.device
            )
            support_emb_or, support_labels_or = get_per_class_support_set(
                proto_datasets=proto_datasets, pos_examples=pos_examples, class_idx=7, device=model.device
            )
            support_emb_oo, support_labels_oo = get_per_class_support_set(
                proto_datasets=proto_datasets, pos_examples=pos_examples, class_idx=8, device=model.device
            )
            support_emb_nll, support_labels_nll = get_per_class_support_set(
                proto_datasets=proto_datasets, pos_examples=pos_examples, class_idx=9, device=model.device
            )
            support_emb_oll, support_labels_oll = get_per_class_support_set(
                proto_datasets=proto_datasets, pos_examples=pos_examples, class_idx=10, device=model.device
            )
            support_emb_sll, support_labels_sll = get_per_class_support_set(
                proto_datasets=proto_datasets, pos_examples=pos_examples, class_idx=11, device=model.device
            )
            support_emb_rtl, support_labels_rtl = get_per_class_support_set(
                proto_datasets=proto_datasets, pos_examples=pos_examples, class_idx=12, device=model.device
            )
            support_emb_tlar, support_labels_tlar = get_per_class_support_set(
                proto_datasets=proto_datasets, pos_examples=pos_examples, class_idx=13, device=model.device
            )
            support_emb_fctr, support_labels_fctr = get_per_class_support_set(
                proto_datasets=proto_datasets, pos_examples=pos_examples, class_idx=14, device=model.device
            )
            support_emb_nlr, support_labels_nlr = get_per_class_support_set(
                proto_datasets=proto_datasets, pos_examples=pos_examples, class_idx=15, device=model.device
            )
            support_emb_orl, support_labels_orl = get_per_class_support_set(
                proto_datasets=proto_datasets, pos_examples=pos_examples, class_idx=16, device=model.device
            )
            support_emb_slr, support_labels_slr = get_per_class_support_set(
                proto_datasets=proto_datasets, pos_examples=pos_examples, class_idx=17, device=model.device
            )
            support_emb_ltl, support_labels_ltl = get_per_class_support_set(
                proto_datasets=proto_datasets, pos_examples=pos_examples, class_idx=18, device=model.device
            )
            support_emb_tlal, support_labels_tlal = get_per_class_support_set(
                proto_datasets=proto_datasets, pos_examples=pos_examples, class_idx=19, device=model.device
            )
            support_emb_fctl, support_labels_fctl = get_per_class_support_set(
                proto_datasets=proto_datasets, pos_examples=pos_examples, class_idx=20, device=model.device
            )

            support_emb_dict = {
                0: (support_emb_gl, support_labels_gl),
                1: (support_emb_ft, support_labels_ft),
                2: (support_emb_ric, support_labels_ric),
                3: (support_emb_tl, support_labels_tl),
                4: (support_emb_ts, support_labels_ts),
                5: (support_emb_oc, support_labels_oc),
                6: (support_emb_op, support_labels_op),
                7: (support_emb_or, support_labels_or),
                8: (support_emb_oo, support_labels_oo),
                9: (support_emb_nll, support_labels_nll),
                10: (support_emb_oll, support_labels_oll),
                11: (support_emb_sll, support_labels_sll),
                12: (support_emb_rtl, support_labels_rtl),
                13: (support_emb_tlar, support_labels_tlar),
                14: (support_emb_fctr, support_labels_fctr),
                15: (support_emb_nlr, support_labels_nlr),
                16: (support_emb_orl, support_labels_orl),
                17: (support_emb_slr, support_labels_slr),
                18: (support_emb_ltl, support_labels_ltl),
                19: (support_emb_tlal, support_labels_tlal),
                20: (support_emb_fctl, support_labels_fctl),
            }
            if random.random() > args.uns_parameter_percentage:
                    continue  # Skip this batch with probability (1 - percentage)
            
            out_dict = model(images_embeddings_raw, support_emb_dict)
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

            progress_bar(i, len(train_loader) - 9, epoch, loss.item())
            
        # --------------------------------------
        # ^ 4. Evaluation phase
        # --------------------------------------
        model.eval()
        for i in range(len(model.encoder)): model.encoder[i].eval()
        
        if debug:
            y_pred = torch.argmax(ys, dim=-1)
            print("Argmax predictions have shape: ", y_pred.shape)

        my_metrics = evaluate_metrics(model, val_loader, args, 
                    support_emb_dict=support_emb_dict,
                    eval_concepts=eval_concepts,)

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
        fprint("  ACC C", cacc, "  ACC Y", yacc, "  F1 Y", cacc, "  F1 Y", f1_y)
        
        if not args.tuning and cacc > best_cacc:
            print("Saving...")
            # Update best F1 score
            best_cacc = cacc
            epochs_no_improve = 0

            # Save the best model
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model with CACC score: {best_cacc}")
            
        elif cacc <= best_cacc:
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    fprint("\n--- End of Training ---\n")

    return best_cacc, support_emb_dict
       



if __name__ == "__main__":
    # & Read arguments and setup environment
    args = setup.read_args()
    setup.setup_environment(args)
    signal.signal(signal.SIGINT, setup.sigint_handler)

    # & Data and Main Model Setup
    dataset = get_dataset(args)
    n_images, c_split = dataset.get_split()
    encoder, decoder = dataset.get_backbone()
    assert isinstance(encoder, tuple) and len(encoder) == 21, "encoder must be a tuple of 21 elements"
    
    model = get_model(args, encoder, decoder, n_images, c_split)
    model.start_optim(args)
    check_optimizer_params(model)
    loss = model.get_loss(args)
    
    print(dataset)
    print("Using Dataset: ", dataset)
    print("Using Model: ", model)
    print("Using Loss: ", loss)
    unsup_train_loader, unsup_val_loader, unsup_test_loader = dataset.get_data_loaders(args=args)

    # & Build Prototypical Support Sets
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':    raise RuntimeError("This code requires a GPU to run.")  
    pos_examples, proto_datasets, proto_dataloaders = build_prototypical_dataloaders(
        unsup_train_loader, device=device, args=args
    )
    
    # & Run Training
    print(f"*** Training model with seed {args.seed}")
    print("Chosen device:", model.device)
    if not os.path.exists(args.save_path): os.makedirs(args.save_path, exist_ok=True)
    save_folder = os.path.join(args.save_path, f"{args.model_parameter_name}_{args.seed}.pth")
    print("Saving model in folder: ", save_folder)

    eval_concepts = ['green_lights', 'follow_traffic', 'road_clear',                        # ^ move forward
            'traffic_lights', 'traffic_signs', 'cars', 'pedestrians', 'riders', 'others',   # ^ stop
            'no_lane_left', 'obstacle_left_lane', 'solid_left_line',                        # ^ left turn
                    'on_right_turn_lane', 'traffic_light_right', 'front_car_right',         # ^ right turn 
            'no_lane_right', 'obstacle_right_lane', 'solid_right_line',                     # ^ right turn
                    'on_left_turn_lane', 'traffic_light_left', 'front_car_left']            # ^ left turn

    best_f1, support_emb_dict = train(
            model=model,
            proto_datasets=proto_datasets,
            proto_dataloaders=proto_dataloaders,
            train_loader=unsup_train_loader,
            val_loader=unsup_val_loader,
            pos_examples=pos_examples,
            save_path=save_folder,
            _loss=loss,
            args=args,
            eval_concepts=eval_concepts,
            seed=args.seed,
            )
    print(f"*** Finished training model with seed {args.seed} and best CACC score {best_f1}")
    print("Training finished.")

    # & Run Evaluation
    best_model = get_model(args, encoder, decoder, n_images, c_split)
    best_model_state_dict = torch.load(save_folder)
    best_model.load_state_dict(best_model_state_dict)
    evaluate_my_model(best_model, 
        save_folder, 
        unsup_test_loader, 
        eval_concepts=eval_concepts, 
        support_embeddings=support_emb_dict, 
        args=args
    )

    print("End of experiment")
    sys.exit(0)