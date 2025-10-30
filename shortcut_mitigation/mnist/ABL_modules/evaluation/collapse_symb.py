import os 
import sys
import torch

from ablkit.data.evaluation import BaseMetric
from sklearn.metrics import f1_score, confusion_matrix

sys.path.append(os.path.abspath(".."))       
sys.path.append(os.path.abspath("../.."))    

from utils.train import compute_coverage


# * Concept Collapse
class SymbolCollapse(BaseMetric):
    def __init__(self, prefix: str = None) -> None:
        super().__init__(prefix)

    def process(self, data_examples) -> None:
        pred_list = getattr(data_examples, "pred_pseudo_label", None)
        gt_list = getattr(data_examples, "gt_pseudo_label", None)
        if pred_list is None or gt_list is None:
            raise RuntimeError("Empty predictions or ground truths")

        for pred_z, gt_z in zip(pred_list, gt_list):
            for p, g in zip(pred_z, gt_z):
                self.results.append((int(p), int(g)))

    def compute_metrics(self) -> dict:  # Use the internal self.results collected by process()
        metrics = {}
        results = self.results
        predicted_concepts = torch.tensor([p for p, g in results])
        true_concepts = torch.tensor([g for p, g in results])
        metrics["character_collapse"] = 1 - compute_coverage(confusion_matrix(true_concepts, predicted_concepts))
        return metrics