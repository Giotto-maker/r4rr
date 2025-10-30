from types import SimpleNamespace
from ABL_modules.evaluation.collapse_symb import SymbolCollapse
from ABL_modules.evaluation.f1_macro_reason import ReasoningMacroF1
from ABL_modules.evaluation.f1_macro_symb import SymbolMacroF1

def run_test_evaluation(kb):
    demo = SimpleNamespace(pred_pseudo_label=[[1,3],[7,3]], gt_pseudo_label=[[1,3],[3,7]])
    m = SymbolMacroF1(prefix="dbg")
    m.process(demo)
    print(m.compute_metrics())

    # Symbol Collapse
    c = SymbolCollapse(prefix="dbg")
    c.process(demo)
    print(c.compute_metrics())

    # Reasoning F1 demo
    demo2 = SimpleNamespace(pred_pseudo_label=[[1,2],[3,4]], Y=[4,8])
    r = ReasoningMacroF1(kb=kb, prefix="dbg")
    r.process(demo2)
    print(r.compute_metrics())