import os
import sys
import torch
import signal
import argparse

from protonet_STOP_bddoia_modules.arguments import args_dpl

sys.path.append(os.path.abspath(".."))      
sys.path.append(os.path.abspath("../.."))   
sys.path.append(os.path.abspath("../../.."))

from bdd_models.template_model import RCNN_global
import original_modules.gsenn
import original_modules.aggregators
import original_modules.parametrizers
import original_modules.conceptizers_BDD
from datasets import get_dataset

def sigint_handler(signum, frame):
    print("\nSIGINT received. Releasing CUDA memory and exiting...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute script with dynamic parameters.")
    parser.add_argument("--loader", type=str, default=None, help="Dataloader to be processed (str value)")
    parser.add_argument("--GPU_ID", type=str, default=None),
    script_args = parser.parse_args()
    signal.signal(signal.SIGINT, sigint_handler)
    os.environ['CUDA_VISIBLE_DEVICES'] = script_args.GPU_ID
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using split: ", script_args.loader)
    print("Using GPU ID: ", script_args.GPU_ID)
    assert script_args.GPU_ID is not None, "Please provide a GPU ID."
    assert script_args.loader is not None, "Please provide a dataloader name."
    assert script_args.loader in ['train', 'val', 'test'], "Dataloader name must be one of ['train', 'val', 'test']."

    sys.modules['models'] = original_modules.gsenn
    sys.modules['conceptizers_BDD'] = original_modules.conceptizers_BDD
    sys.modules['aggregators'] = original_modules.aggregators
    sys.modules['parametrizers'] = original_modules.parametrizers

    args = args_dpl
    pretrained_model = RCNN_global()
    checkpoint = torch.load('bdd_models/model_weight.pt')
    pretrained_model.load_state_dict(checkpoint['pretrained'])
    pretrained_model = pretrained_model.to(device)
    pretrained_model.eval()
    print("Pretrained model loaded successfully.")

    dataset = get_dataset(args)
    n_images, c_split = dataset.get_split()
    print(dataset)
    print("Using Dataset: ", dataset)
    unsup_train_loader, unsup_val_loader, unsup_test_loader = dataset.get_data_loaders(batch_size=args.batch_size)
    
    my_loader = None 
    if script_args.loader == 'train':
        my_loader = unsup_train_loader
    elif script_args.loader == 'val':
        my_loader = unsup_val_loader
    elif script_args.loader == 'test':
        my_loader = unsup_test_loader
    print("Using DataLoader: ", my_loader)

    output_dir = os.path.join("../FASTER-BDDOIA", script_args.loader)
    for images, class_labels, attr_labels, img_paths in my_loader:
        images        = images.to(device)
        class_labels  = class_labels.to(device)   # [B, 5]
        attr_labels   = attr_labels.to(device)    # [B, 21]

        scene_feats, scene_feats_raw, all_rois, all_scores, all_labels, all_bbox_feats, _ = (
            pretrained_model(images)
        )
        #   scene_feats:     [B, 2048]
        #   scene_feats_raw: [B, 2048]
        #   all_rois:        [N, 5]  (batch_idx, x1,y1,x2,y2)
        #   all_scores:      [N]
        #   all_labels:      [N]
        #   all_bbox_feats:  [N, 1024]

        all_rois, all_scores, all_labels, all_bbox_feats = (
            all_rois, all_scores, all_labels, all_bbox_feats
        )
        batch_idx = all_rois[:, 0].long()  # [N]

        per_image = []
        B = images.size(0)
        for i in range(B):
            mask = batch_idx == i
            per_image.append({
                "scene_feat_raw": scene_feats_raw[i],          # [2048]
                "scene_feat":     scene_feats[i],              # [2048]

                # KEEP the batch‐index column here:
                "rois":           all_rois[mask],              # [Ni, 5] ← batch_idx included
                "scores":         all_scores[mask],            # [Ni]
                "labels":         all_labels[mask],            # [Ni]
                "roi_feats":      all_bbox_feats[mask],        # [Ni, 1024]
            })

        raise RuntimeError("Stopping execution for debugging purposes. Remove this line to continue with the script.")

        # Save the embeddings
        for i, img_path in enumerate(img_paths):
            key = os.path.splitext(os.path.basename(img_path))[0]
            folder = os.path.join(output_dir, key)
            os.makedirs(folder, exist_ok=True)

            # global embeddings
            torch.save(per_image[i]["scene_feat_raw"].cpu(),
                    os.path.join(folder, "embedded_image_raw.pt"))
            torch.save(per_image[i]["scene_feat"].cpu(),
                    os.path.join(folder, "embedded_image.pt"))

            # all detections
            torch.save(per_image[i]["rois"].cpu(),
                    os.path.join(folder, "detected_rois.pt"))
            torch.save(per_image[i]["scores"].cpu(),
                    os.path.join(folder, "detection_scores.pt"))
            torch.save(per_image[i]["labels"].cpu(),
                    os.path.join(folder, "detection_labels.pt"))
            torch.save(per_image[i]["roi_feats"].cpu(),
                    os.path.join(folder, "detected_rois_feats.pt"))

            # GT labels
            torch.save(class_labels[i].cpu(),
                    os.path.join(folder, "class_labels.pt"))
            torch.save(attr_labels[i].cpu(),
                    os.path.join(folder, "attr_labels.pt"))

            assert per_image[i]["scene_feat_raw"].size() == (2048,), f"Expected scene_feat_raw to have shape [2048], but got {per_image[i]['scene_feat_raw'].size()}."
            assert per_image[i]["scene_feat"].size() == (2048,), f"Expected scene_feat to have shape [2048], but got {per_image[i]['scene_feat'].size()}."
            assert per_image[i]["rois"].size(1) == 5, f"Expected rois to have shape [Ni, 5], but got {per_image[i]['rois'].size()}."
            assert per_image[i]["scores"].size(0) == per_image[i]["rois"].size(0), f"Expected scores to have the same number of elements as rois, but got {per_image[i]['scores'].size(0)} and {per_image[i]['rois'].size(0)}."
            assert per_image[i]["labels"].size(0) == per_image[i]["rois"].size(0), f"Expected labels to have the same number of elements as rois, but got {per_image[i]['labels'].size(0)} and {per_image[i]['rois'].size(0)}."
            assert per_image[i]["roi_feats"].size(1) == 1024, f"Expected roi_feats to have shape [Ni, 1024], but got {per_image[i]['roi_feats'].size()}."
            assert class_labels[i].size(0) == 5, f"Expected class_labels to have shape [5], but got {class_labels[i].size()}."
            assert attr_labels[i].size(0) == 21, f"Expected attr_labels to have shape [21], but got {attr_labels[i].size()}."
            assert class_labels.size(1) == 5, f"Expected class_labels to have shape [batch_size, 5], but got {class_labels.size()}."
