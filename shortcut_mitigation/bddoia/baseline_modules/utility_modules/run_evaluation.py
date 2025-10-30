import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from typing import List
from argparse import Namespace
from torch.utils.data import DataLoader
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix

sys.path.append(os.path.abspath(".."))  
sys.path.append(os.path.abspath("../.."))  
from utils import fprint
from utils.metrics import evaluate_metrics
from models.mnistdpl import MnistDPL


# * helper function for 'plot_multilabel_confusion_matrix'
def convert_to_categories(elements):
    # Convert vector of 0s and 1s to a single binary representation along the first dimension
    binary_rep = np.apply_along_axis(
        lambda x: "".join(map(str, x)), axis=1, arr=elements
    )
    return np.array([int(x, 2) for x in binary_rep])


# * BBDOIA custom confusion matrix for concepts
def plot_multilabel_confusion_matrix(
    y_true, y_pred, class_names, title, save_path=None
):
    y_true_categories = convert_to_categories(y_true.astype(int))
    y_pred_categories = convert_to_categories(y_pred.astype(int))

    to_rtn_cm = confusion_matrix(y_true_categories, y_pred_categories)

    cm = multilabel_confusion_matrix(y_true, y_pred)
    num_classes = len(class_names)
    num_rows = (num_classes + 4) // 5  # Calculate the number of rows needed

    plt.figure(figsize=(20, 4 * num_rows))  # Adjust the figure size

    for i in range(num_classes):
        plt.subplot(num_rows, 5, i + 1)  # Set the subplot position
        plt.imshow(cm[i], interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(f"Class: {class_names[i]}")
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ["0", "1"])
        plt.yticks(tick_marks, ["0", "1"])

        fmt = ".0f"
        thresh = cm[i].max() / 2.0
        for j in range(cm[i].shape[0]):
            for k in range(cm[i].shape[1]):
                plt.text(
                    k,
                    j,
                    format(cm[i][j, k], fmt),
                    ha="center",
                    va="center",
                    color="white" if cm[i][j, k] > thresh else "black",
                )

        plt.ylabel("True label")
        plt.xlabel("Predicted label")

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.suptitle(title)

    if save_path:
        plt.savefig(f"{save_path}_total.png")
    else:
        plt.show()

    plt.close()

    return to_rtn_cm


# * Concept collapse (Soft)
def compute_coverage(confusion_matrix):
    """Compute the coverage of a confusion matrix.

    Essentially this metric is
    """

    max_values = np.max(confusion_matrix, axis=0)
    clipped_values = np.clip(max_values, 0, 1)

    # Redefinition of soft coverage
    coverage = np.sum(clipped_values) / len(clipped_values)

    return coverage


# * BDDOIA custom confusion matrix for actions
def plot_actions_confusion_matrix(c_true, c_pred, title, save_path=None):

    my_scenarios = {
        "forward": [slice(0, 3), slice(0, 3)],  
        "stop": [slice(3, 9), slice(3, 9)],
        "left": [slice(9, 11), slice(18,20)],
        "right": [slice(12, 17), slice(12,17)],
    }

    to_rtn = {}

    # Plot confusion matrix for each scenario
    for scenario, indices in my_scenarios.items():

        g_true = convert_to_categories(c_true[:, indices[0]].astype(int))
        c_pred_scenario = convert_to_categories(c_pred[:, indices[1]].astype(int))

        # Compute confusion matrix
        cm = confusion_matrix(g_true, c_pred_scenario)

        # Plot confusion matrix
        plt.figure()
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(f"{title} - {scenario}")
        plt.colorbar()

        n_classes = c_true[:, indices[0]].shape[1]

        tick_marks = np.arange(2**n_classes)
        plt.xticks(tick_marks, ["" for _ in range(len(tick_marks))])
        plt.yticks(tick_marks, ["" for _ in range(len(tick_marks))])

        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()

        # Save or show plot
        if save_path:
            plt.savefig(f"{save_path}_{scenario}.png")
        else:
            plt.show()

        to_rtn.update({scenario: cm})

        plt.close()

    return to_rtn


# * Evaluate the model and save metrics
def evaluate_my_model(model: MnistDPL, 
        save_path: str, 
        test_loader: DataLoader,
        eval_concepts: List[str],
        args: Namespace,
        support_embeddings=None,
    ):
    
    if args.model == 'probddoiadpl':
        assert support_embeddings is not None, "Support embeddings must be provided for probddoiadpl model evaluation."
        my_metrics = evaluate_metrics(model, test_loader, args,
                        support_emb_dict=support_embeddings, 
                        eval_concepts=eval_concepts
        )
    else:
        my_metrics = evaluate_metrics(model, test_loader, args, 
                        eval_concepts=eval_concepts
        )

    loss = my_metrics[0]
    cacc = my_metrics[1]
    yacc = my_metrics[2]
    f1_y = my_metrics[3]
    f1_micro = my_metrics[4]
    f1_weight = my_metrics[5]
    f1_bin = my_metrics[6]

    metrics_log_path = save_path.replace(".pth", "_metrics.log")
    
    all_concepts = [ 'Green Traffic Light', 'Follow Traffic', 'Road Is Clear',
        'Red Traffic Light', 'Traffic Sign', 'Obstacle Car', 'Obstacle Pedestrian', 'Obstacle Rider', 'Obstacle Others',
        'No Lane On The Left',  'Obstacle On The Left Lane',  'Solid Left Line',
                'On The Right Turn Lane', 'Traffic Light Allows Right', 'Front Car Turning Right', 
        'No Lane On The Right', 'Obstacle On The Right Lane', 'Solid Right Line',
                'On The Left Turn Lane',  'Traffic Light Allows Left',  'Front Car Turning Left' 
    ]
    aggregated_metrics = [
            'F1 - Binary', 'F1 - Macro', 'F1 - Micro', 'F1 - Weighted',
            'Precision - Binary', 'Precision - Macro', 'Precision - Micro', 'Precision - Weighted',
            'Recall - Binary', 'Recall - Macro', 'Recall - Micro', 'Recall - Weighted',
            'Balanced Accuracy'
    ]

    sums = [0.0] * len(aggregated_metrics)
    num_concepts = len(all_concepts)
    with open(metrics_log_path, "a") as log_file:
        log_file.write(f"ACC C: {cacc}, ACC Y: {yacc}\n\n")
        log_file.write(f"F1 Y - Macro: {f1_y}, F1 Y - Micro: {f1_micro}, F1 Y - Weighted: {f1_weight}, F1 Y - Binary: {f1_bin} \n\n")

        def write_metrics(class_name, offset):
            print(f"Reporting Metrics for {class_name} in {metrics_log_path}")
            log_file.write(f"{class_name.upper()}\n")
            for idx, metric_name in enumerate(aggregated_metrics):
                value = my_metrics[offset + idx]
                sums[idx] += value
                log_file.write(f"  {metric_name:<18} {value:.4f}\n")
            log_file.write("\n")

        i = 7
        for concept in all_concepts:
            write_metrics(concept, i)
            i += len(aggregated_metrics)

        log_file.write("**MEAN ACROSS ALL CONCEPTS**\n")
        for idx, metric_name in enumerate(aggregated_metrics):
            mean_value = sums[idx] / num_concepts
            log_file.write(f"  {metric_name:<18} {mean_value:.4f}\n")
        log_file.write("\n")


    assert len(my_metrics) == 7 + len(all_concepts) * len(aggregated_metrics), \
        f"Expected {7 + len(all_concepts) * len(aggregated_metrics)} metrics, but got {len(my_metrics)}"
    
    if args.model == 'probddoiadpl':
        y_true, c_true, y_pred, c_pred, p_cs, p_ys, p_cs_all, p_ys_all = (
            evaluate_metrics(model, test_loader, args,
                        support_emb_dict=support_embeddings, 
                        eval_concepts=eval_concepts,
                        last=True
                )
        )
    else:
        y_true, c_true, y_pred, c_pred, p_cs, p_ys, p_cs_all, p_ys_all = (
            evaluate_metrics(model, test_loader, args,
                        eval_concepts=eval_concepts,
                        last=True
                )
        )
    
    y_labels = ["stop", "forward", "left", "right"]
    concept_labels = [
        "green_light",      
        "follow",           
        "road_clear",       
        "red_light",        
        "traffic_sign",     
        "car",              
        "person",           
        "rider",            
        "other_obstacle",   
        "left_lane",
        "left_green_light",
        "left_follow",
        "no_left_lane",
        "left_obstacle",
        "letf_solid_line",
        "right_lane",
        "right_green_light",
        "right_follow",
        "no_right_lane",
        "right_obstacle",
        "right_solid_line",
    ]

    plot_multilabel_confusion_matrix(y_true, y_pred, y_labels, "Labels", save_path=save_path)
    cfs = plot_actions_confusion_matrix(c_true, c_pred, "Concepts", save_path=save_path)
    cf = plot_multilabel_confusion_matrix(c_true, c_pred, concept_labels, "Concepts", save_path=save_path)
    print("Concept collapse", 1 - compute_coverage(cf))

    with open(metrics_log_path, "a") as log_file:
        for key, value in cfs.items():
            log_file.write(f"Concept collapse: {key}, {1 - compute_coverage(value):.4f}\n")
            log_file.write("\n")

    fprint("\n--- End of Evaluation ---\n")