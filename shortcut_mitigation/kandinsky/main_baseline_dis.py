
# ! Baseline with pretraining option for Kandinsky using disentangled shape/colour encoder
import os
import sys
import torch
import signal
import random
import torch.nn.functional as F
import torchvision.transforms as T

from ultralytics import YOLO

from torch.utils.data import DataLoader

from protonet_kand_modules.utility_modules import setup
from protonet_kand_modules.utility_modules.pretraining_disj import pre_train
from protonet_kand_modules.data_modules import batch_sampler, my_datasets
from protonet_kand_modules.data_modules.proto_data_creation import get_my_initial_prototypes

sys.path.append(os.path.abspath(".."))  
sys.path.append(os.path.abspath("../.."))

from utils import fprint
from utils.status import progress_bar
from utils.metrics import evaluate_metrics
from utils.dpl_loss import ADDMNIST_DPL
from utils.checkpoint import save_model

from datasets import get_dataset
from datasets.utils.base_dataset import BaseDataset

from models import get_model
from models.mnistdpl import MnistDPL


# * Training Loop
def train(model: MnistDPL,
    sup_loader: DataLoader,
    c_loader: DataLoader,
    dataset: BaseDataset, 
    concept_extractor,
    concept_extractor_training_path,
    concept_extractor_project_path,
    transform,
    _loss: ADDMNIST_DPL,
    args,
    save_folder: str,
    patience: int = 3
    ):
    
    best_cacc = 0.0
    epochs_no_improve = 0   # for early stopping
    yolo_save_dir = None

    model.to(model.device)

    train_loader, val_loader, test_loader = dataset.get_data_loaders()
    dataset.print_stats()
    
    # Initialize optimizers and schedulers
    shape_optimizer = torch.optim.Adam(model.encoder[0].parameters())
    color_optimizer = torch.optim.Adam(model.encoder[1].parameters())
    shape_lr_scheduler = torch.optim.lr_scheduler.StepLR(shape_optimizer, step_size=10, gamma=0.5)
    color_lr_scheduler = torch.optim.lr_scheduler.StepLR(color_optimizer, step_size=10, gamma=0.5)

    fprint("\n--- Start of Training ---\n")

    # * Start of training
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
                            project=concept_extractor_project_path)
                yolo_save_dir = os.path.join(results.save_dir, "weights", "last.pt")
            else:
                assert yolo_save_dir is not None
                concept_extractor = YOLO(yolo_save_dir)
                results = concept_extractor.train(data=concept_extractor_training_path, 
                            epochs=args.extractor_training_epochs, 
                            imgsz=64, 
                            project=concept_extractor_project_path)
                yolo_save_dir = os.path.join(results.save_dir, "weights", "last.pt")

        # ^ PHASE 2: Backbone Pretraining
        if args.Apretrained:
            pre_train(model, sup_loader, args)

        # ^ PHASE 3: Main Model Concept Supervised Training
        fprint("\n--- Start of Concept Supervised Training ---\n")
        for j, (images, labels) in enumerate(c_loader):
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
            out_dict = model(merged_images, concept_extractor, transform, args)
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
            concept_loss = args.concept_loss_weight * (loss1 + loss2 + loss3)

            # backprop
            shape_optimizer.zero_grad()
            color_optimizer.zero_grad()    
            concept_loss.backward()
            shape_optimizer.step()
            color_optimizer.step()

            # Progress update
            progress_bar(j, len(c_loader), epoch, concept_loss.item())

        # ^ PHASE 4: Main Model Training
        fprint("\n--- Start of Main Model Unsupervised Training ---\n")
        ys, y_true, cs, cs_true = None, None, None, None
        for i, data in enumerate(train_loader):
            if random.random() > args.uns_parameter_percentage:
                continue  # Skip this batch with probability (1 - percentage)

            if epoch == 0:
                model.eval()
                assert not model.training, "Model should **NOT** be in training mode!"
                assert not model.encoder[0].training, "Encoder should **NOT** be in training mode!"
                assert not model.encoder[1].training, "Encoder should **NOT** be in training mode!"
            else:    
                model.train()
                shape_optimizer.zero_grad()
                color_optimizer.zero_grad()
                assert model.training, "Model should be in training mode!"
                assert model.encoder[0].training, "Shape encoder should be in training mode!"
                assert model.encoder[1].training, "Color encoder should be in training mode!"
                
            images, labels, concepts = data
            images, labels, concepts = (
                images.to(model.device),
                labels.to(model.device),
                concepts.to(model.device),
            )

            out_dict = model(images, concept_extractor, transform, args)
            out_dict.update({"LABELS": labels, "CONCEPTS": concepts})
            
            loss, losses = _loss(out_dict, args)
            loss.backward()
            
            if epoch != 0:
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
                progress_bar(i, len(train_loader) - 9, epoch, loss.item())

        if ys is None:
            # Skip evaluation if no unsupervised data was used for the model  
            torch.save(model.state_dict(), save_folder)
            concept_save_path = os.path.join(os.path.dirname(save_folder), f"best_{args.seed}.pt")
            concept_extractor.save(concept_save_path)
            print(f"Saved model without finetuning over unsupervised data.")
            print()
            continue

        # Step the scheduler (if using)
        if epoch != 0:
            shape_lr_scheduler.step()
            color_lr_scheduler.step()

        if "patterns" in args.task:
            y_true = y_true[:, -1]  # it is the last one

        model.eval()
        tloss, cacc, yacc, f1 = evaluate_metrics(model, val_loader, args, concept_extractor=concept_extractor, transform=transform)

        ### LOGGING ###
        fprint("  ACC C", cacc, "  ACC Y", yacc, "F1 Y", f1)
        print()

        if not args.tuning and cacc > best_cacc:
            print("Saving...")
            # Update best F1 score
            if best_cacc == 0.0:     print("Baseline accuracy has been determined.")
            best_cacc = cacc
            epochs_no_improve = 0
                
            # Save the best model and the concept extractor
            torch.save(model.state_dict(), save_folder)
            concept_save_path = os.path.join(os.path.dirname(save_folder), f"best_{args.seed}.pt")
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
    # & read arguments and setup environment
    args = setup.read_args(baseline=True)
    setup.setup_environment(args)
    signal.signal(signal.SIGINT, setup.sigint_handler)

    # & yolo setup
    yaml_path = os.path.join(os.getcwd(), f"data/kand_config_yolo_{args.GPU_ID}/kand_config_yolo.yaml")
    my_yolo_project_path = f"ultralytics-{args.GPU_ID}/"
    my_yolo_premodel_path = f"ultralytics-{args.GPU_ID}/pretrained/yolo11n.pt"
    args.yolo_folder = my_yolo_project_path
    if args.retrain_extractor:
        yolo = YOLO(my_yolo_premodel_path)
    else:
        print("Using pretrained model: ", args.concept_extractor_path)
        yolo = YOLO(args.concept_extractor_path)
    
    # & load supervisions and create training data for shapes and colours
    proto_images, proto_labels, _ = get_my_initial_prototypes()
    kand_proto_dataset = my_datasets.PrimitivesDataset(proto_images, proto_labels, transform=None)
    shape_labels = kand_proto_dataset.labels[:, 0].numpy()
    color_labels = kand_proto_dataset.labels[:, 1].numpy()
    shape_sampler = batch_sampler.PrototypicalBatchSampler(shape_labels, 
                        args.classes_per_it, args.num_samples, args.iterations)
    color_sampler = batch_sampler.PrototypicalBatchSampler(color_labels, 
                        args.classes_per_it, args.num_samples, args.iterations)
    shape_dataloader = DataLoader(kand_proto_dataset, batch_sampler=shape_sampler)
    color_dataloader = DataLoader(kand_proto_dataset, batch_sampler=color_sampler)

    # & load the supervisions for concept loss 
    if args.c:
        proto_images = torch.load('data/kand_annotations/yolo_annotations/images.pt')
        proto_labels = torch.load('data/kand_annotations/yolo_annotations/labels.pt')
        kand_c_dataset = my_datasets.SupervisedDataset(proto_images, proto_labels, transform=None)
        c_loader = DataLoader(kand_c_dataset, batch_size=args.batch_size, shuffle=True)

    # & load unsupervised data and create the model 
    dataset = get_dataset(args)
    n_images, c_split = dataset.get_split()
    encoder, decoder = dataset.get_backbone()
    model = get_model(args, encoder, decoder, n_images, c_split)
    loss = model.get_loss(args)
    assert len(model.state_dict()) > 0, \
        "Model state dict is empty. Please check the model initialization."
    print("Using Dataset: ", dataset)
    print("Number of images: ", n_images)
    print("Using backbone: ", encoder)
    print("Using Model: ", model)
    print("Using Loss: ", loss)
    print("Working with taks: ", args.task)

    # & Pre-Training
    if args.pretrained:
        pre_train(model, shape_dataloader, args)
    
    # & Training    
    print(f"*** Training model with seed {args.seed}")
    print("Chosen device:", model.device)
    if not os.path.exists(args.save_path): os.makedirs(args.save_path, exist_ok=True)
    save_folder = os.path.join(args.save_path, f"{args.model_parameter_name}_{args.seed}.pth")
    print("Saving model in folder: ", save_folder)
    best_cacc = train(model=model,
        dataset=dataset,
        sup_loader=shape_dataloader,
        c_loader=c_loader,
        concept_extractor=yolo,                              # yolo model
        concept_extractor_training_path=yaml_path,           # yolo training data path
        concept_extractor_project_path=my_yolo_project_path, # yolo project path
        transform=T.Resize((64, 64)),                        # resizer     
        _loss=loss,
        args=args,
        save_folder=save_folder,
    )
    save_model(model, args, args.seed)  # save the model parameters
    print(f"*** Finished training model with seed {args.seed}")
    print("Training finished.")
    print(f"Best CACC score: {best_cacc}")