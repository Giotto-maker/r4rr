from typing import Iterable
from ablkit.data.evaluation import BaseMetric
from sklearn.metrics import f1_score


# * F1 MACRO score for labels
class ReasoningMacroF1(BaseMetric):
    """
    Macro F1 over *final labels* (reasoning outputs).
    """
    def __init__(self, kb, prefix: str = None) -> None:
        super().__init__(prefix)
        self.kb = kb

    def _get_attr(self, obj, names: Iterable[str]):
        for n in names:
            val = getattr(obj, n, None)
            if val is not None:
                return val
        return None

    def process(self, data_examples) -> None:
        # data_examples: ['_metainfo_fields', '_data_fields', 'X', 'gt_pseudo_label', 'Y', 'pred_idx', 'pred_prob', 'pred_pseudo_label']
        preds = getattr(data_examples, "pred_pseudo_label", None)
        reasoning_result = [self.kb.logic_forward(pair) for pair in preds]
        gts = getattr(data_examples, "Y", None)

        if reasoning_result is None or gts is None:
            raise RuntimeError("Empty predictions or ground truths")

        for p, g in zip(reasoning_result, gts):
            self.results.append((int(p), int(g)))

    def compute_metrics(self) -> dict:
        metrics = {}
        results = self.results
        y_pred = [p for p, g in results]
        y_true = [g for p, g in results]
        metrics["reasoning_macro_f1"] = f1_score(y_true, y_pred, average='macro')
        metrics["reasoning_micro_f1"] = f1_score(y_true, y_pred, average='micro')
        return metrics