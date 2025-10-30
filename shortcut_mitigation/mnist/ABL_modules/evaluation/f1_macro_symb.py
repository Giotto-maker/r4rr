from ablkit.data.evaluation import BaseMetric
from sklearn.metrics import f1_score


# * F1 MACRO score for concepts
class SymbolMacroF1(BaseMetric):
    """
    Macro F1 over *symbols* (pseudo-labels).
    Expects data_examples to expose:
      - pred_pseudo_label: List[List[int]]  (e.g. [[1,2],[3,4],...])
      - gt_pseudo_label:   List[List[int]]
    Stores flattened (pred, gt) pairs in self.results.
    """
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
        y_pred = [p for p, g in results]
        y_true = [g for p, g in results]
        metrics["character_macro_f1"] = f1_score(y_true, y_pred, average='macro')
        return metrics