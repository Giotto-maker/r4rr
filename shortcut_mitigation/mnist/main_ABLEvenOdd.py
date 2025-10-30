import os 
import sys
import torch
import os.path as osp
import torch.nn as nn

from torch.utils.data import DataLoader

from ablkit.bridge import SimpleBridge
from ablkit.learning import ABLModel, BasicNN
from ablkit.reasoning import Reasoner
from ablkit.utils import ABLLogger, print_log
from ablkit.data.evaluation import ReasoningMetric, SymbolAccuracy

from ABL_modules.utils import setup
from ABL_modules.utils.knowledge import AddKB, test
from ABL_modules.utils.pretraining import pre_train
from ABL_modules.ABL_arguments import args_short, args_add
from ABL_modules.backbone.mnist_classifier import MNISTSingleEncoder
from ABL_modules.utils.data import (
    MNISTAugDatasetPretraining,
    MNISTAugDatasetTraining,
    dataloader_to_list, 
    print_data_format,
    merge_mnist_pairs
)
from ABL_modules.evaluation.collapse_symb import SymbolCollapse
from ABL_modules.evaluation.f1_macro_reason import ReasoningMacroF1
from ABL_modules.evaluation.f1_macro_symb import SymbolMacroF1
from ABL_modules.evaluation.simple_test import run_test_evaluation

from protonet_mnist_add_modules.utility_modules.proto_utils import init_dataloader
from protonet_mnist_add_modules.data_modules.proto_data_creation import (
    choose_initial_prototypes,
    get_augmented_support_query_set,
)

sys.path.append(os.path.abspath(".."))       
sys.path.append(os.path.abspath("../.."))

from datasets import get_dataset


if __name__ == "__main__":
    # & read arguments and setup environment
    args = setup.read_args()
    setup.setup_environment(args)

    # & load the unsupervised data
    dataset = get_dataset(args_short)
    print_log(f"Using Dataset: {dataset}", logger="current")
    unsup_train_loader, unsup_val_loader, unsup_test_loader = dataset.get_data_loaders()

    # & load the supervisions for PRETRAINING
    addmnist_dataset = get_dataset(args_add)
    addmnist_train_loader, _ , _ = addmnist_dataset.get_data_loaders()
    print_log(addmnist_dataset, logger="current")
    if ( (not os.path.exists('data/prototypes/proto_loader_dataset.pth')) ):
        print_log("Creating proto_loader_dataset.pth", logger="current")
        choose_initial_prototypes(addmnist_train_loader, debug=False)
    tr_dataloader = init_dataloader()
    support_images_aug, support_labels_aug, _, _, _ = get_augmented_support_query_set(
        tr_dataloader, debug=False
    )
    assert support_images_aug.numel() > 0, "support_images_aug is an empty tensor"
    assert support_labels_aug.numel() > 0, "support_labels_aug is an empty tensor"
    assert not torch.all(support_images_aug == 0), "All elements in support_images_aug are zero"
    assert not torch.all(support_labels_aug == 0), "All elements in support_labels_aug are zero"    
    mnist_dataset = MNISTAugDatasetPretraining(support_images_aug, support_labels_aug)
    sup_train_loader_pretraining = DataLoader(mnist_dataset, batch_size=32, shuffle=True)

    # & load the supervisions for TRAINING (loss)
    merged_images, concepts, labels = merge_mnist_pairs(support_images_aug, support_labels_aug)
    mnist_dataset = MNISTAugDatasetTraining(merged_images, labels, concepts)
    sup_train_loader_training = DataLoader(mnist_dataset, batch_size=32, shuffle=True)

    # & convert the data to ABL-ready format
    abl_unsup_train_data = dataloader_to_list(unsup_train_loader)
    abl_sup_train_data  = dataloader_to_list(sup_train_loader_training)
    abl_unsup_val_data   = dataloader_to_list(unsup_val_loader)
    abl_unsup_test_data  = dataloader_to_list(unsup_test_loader)
    print_data_format(abl_unsup_train_data, abl_unsup_val_data)
    
    # & Learning Part
    print_log("Building the Learning Part...", logger="current")
    cls = MNISTSingleEncoder()
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.2)
    optimizer = torch.optim.Adam(
        cls.parameters(), lr=args_short.lr, weight_decay=args_short.weight_decay
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    base_model = BasicNN(
        cls,
        loss_fn,
        optimizer,
        scheduler=scheduler,
        device=device,
        batch_size=32,
        num_epochs=10,
    )
    model = ABLModel(base_model)
    print_log("Learning Part Built!", logger="current")

    # & Reasoning Part
    print_log("Building the Reasoning Part...", logger="current")
    kb = AddKB()
    reasoner = Reasoner(kb)
    pseudo_labels = [1, 2]
    test(kb, pseudo_labels)
    print_log("Reasoning Part Built!", logger="current")

    # & Evaluation Part
    print_log("Building the Evaluation Part...", logger="current")
    metric_list = [
        SymbolAccuracy(prefix="mnist_add"),
        SymbolMacroF1(prefix="mnist_add"),
        SymbolCollapse(prefix="mnist_add"),
        ReasoningMetric(kb=kb, prefix="mnist_add"),
        ReasoningMacroF1(kb=kb, prefix="mnist_add"),
    ]
    run_test_evaluation(kb)
    print_log("Evaluation Part Built!", logger="current")

    # & Bridge
    print_log("Building the Bridge...", logger="current")
    bridge = SimpleBridge(model, reasoner, metric_list)
    print_log("Bridge Built!", logger="current")

    # & PreTraining
    if args.pretrained:
        print_log("Abductive Learning backbone PRETRAINING.", logger="current")
        pre_train(cls, sup_train_loader_pretraining, seed=args.seed)

    # & Training
    print_log("Abductive Learning on the MNIST Addition example.", logger="current")
    log_dir = ABLLogger.get_current_instance().log_dir
    weights_dir = osp.join(log_dir, "weights")
    if args.c:
        print_log("Training SUPERVISIONS enabled.", logger="current")
        bridge.train(
            train_data=abl_unsup_train_data,
            label_data=abl_sup_train_data,
            val_data=abl_unsup_val_data,
            segment_size=0.01,
            loops=1,
            save_interval=1,
            save_dir=weights_dir,
        )
    else:
        bridge.train(
            train_data=abl_unsup_train_data,
            val_data=abl_unsup_val_data,
            segment_size=0.01,
            loops=1,
            save_interval=1,
            save_dir=weights_dir,
        )

    # & Testing
    bridge.test(abl_unsup_test_data)