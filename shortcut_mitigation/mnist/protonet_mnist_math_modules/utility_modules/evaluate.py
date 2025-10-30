import os
import sys

from argparse import Namespace
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

sys.path.append(os.path.abspath(".."))  
sys.path.append(os.path.abspath("../.."))  

from utils.metrics import evaluate_metrics
from models.mnistdpl import MnistDPL
from utils.train import convert_to_categories, compute_coverage


# * Compute Concept Collapse
def compute_concept_collapse(true_concepts, predicted_concepts, multilabel=False):
    return 1 - compute_coverage(confusion_matrix(true_concepts, predicted_concepts))


# * Evaluate the model on the test/OOD set and compute metrics
def evaluate_my_model(model: MnistDPL, 
        save_path: str, 
        my_loader: DataLoader,
        seed: int,
        args: Namespace,
        baseline=True,
        support_images=None,
        support_labels=None,
    ):

    if baseline:
        assert support_images is None and support_labels is None, \
            "Support images and labels should not be provided for baseline evaluation."
    else:
        assert support_images is not None and support_labels is not None, \
            "Support images and labels must be provided for non-baseline evaluation."

    # * Compute test set accuracies and F1 scores
    if baseline:
        _, cacc, yacc, f1_y, f1_c = evaluate_metrics(model, my_loader, args)
    else:
        _, cacc, yacc, f1_y, f1_c = evaluate_metrics(
            model, my_loader, args, support_images=support_images, support_labels=support_labels
    )
    with open(save_path, "a") as f:
        print(f"Evaluation results for seed {seed}:")
        print(f"    ACC(C): {round(cacc/100,2)}, "
              f"ACC(Y): {round(yacc/100,2)}, "
              f"F1(Y): {round(f1_y/100,2)}, "
              f"F1(C): {round(f1_c/100,2)}")
        f.write(
            f"Evaluation results for seed {seed}:\n"
            f"    ACC(C): {round(cacc/100,2)}, "
            f"ACC(Y): {round(yacc/100,2)}, "
            f"F1(Y): {round(f1_y/100,2)}, "
            f"F1(C): {round(f1_c/100,2)}\n"
        )

    # * Compute Concept Collapse
    if baseline:
        y_true, c_true, y_pred, c_pred, _, _, _, _ = evaluate_metrics(model, my_loader, args, last=True)
    else:
        y_true, c_true, y_pred, c_pred, _, _, _, _ = evaluate_metrics(
            model, my_loader, args, support_images=support_images, support_labels=support_labels, last=True
    )

    cls = compute_concept_collapse(c_true, c_pred)
    with open(save_path, "a") as f:
        print(f"Cls(C): {round(cls,2):.4f}")
        f.write(f"    Cls(C): {round(cls,2):.4f}\n")
    return c_true, c_pred