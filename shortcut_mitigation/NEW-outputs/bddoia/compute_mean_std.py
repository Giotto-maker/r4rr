#!/usr/bin/env python3
import os
import re
import statistics
from pathlib import Path

# ^ MACROS
FOLDER = "baseline/dpl/PRE-baseline"
SEEDS  = [0, 128, 256, 512, 1024, 2048]

# ^ PATTERNS
# ACC C, ACC Y, F1 Y − Macro/Micro/Weighted/Binary
glob_re = re.compile(
    r"ACC C:\s*(?P<ACC_C>[\d\.]+),\s*ACC Y:\s*(?P<ACC_Y>[\d\.]+).*?"
    r"F1 Y - Macro:\s*(?P<F1Y_M>[\d\.]+),\s*F1 Y - Micro:\s*(?P<F1Y_mi>[\d\.]+),\s*"
    r"F1 Y - Weighted:\s*(?P<F1Y_w>[\d\.]+),\s*F1 Y - Binary:\s*(?P<F1Y_b>[\d\.]+)",
    re.DOTALL
)

# Concept collapse
concept_block_re = re.compile(
    r"^([A-Z \-]+)\n((?:\s{2}.+\n)+)",
    re.MULTILINE
)
metric_line_re = re.compile(r"^\s{2}(?P<metric>[\w\s\-]+)\s+([\d\.]+)$", re.MULTILINE)
collapse_re = re.compile(r"Concept collapse: (\w+), ([\d\.]+)")

def parse_file(path):
    text = path.read_text()
    out = {
        "global": {},
        "concepts": {},
        "collapse": {}
    }

    m = glob_re.search(text)
    if m:
        d = m.groupdict()
        out["global"] = {
            "ACC C":        float(d["ACC_C"]),
            "ACC Y":        float(d["ACC_Y"]),
            "F1 Y - Macro": float(d["F1Y_M"]),
            "F1 Y - Micro": float(d["F1Y_mi"]),
            "F1 Y - Weighted": float(d["F1Y_w"]),
            "F1 Y - Binary":   float(d["F1Y_b"]),
        }

    for cm in concept_block_re.finditer(text):
        concept = cm.group(1).strip()
        block = cm.group(2)
        metrics = {}
        for ml in metric_line_re.finditer(block):
            mname = ml.group("metric").strip()
            val   = float(ml.group(2))
            metrics[mname] = val
        out["concepts"][concept] = metrics

    for match in collapse_re.finditer(text):
        cname, val = match.groups()
        out["collapse"][cname] = float(val)

    return out

def main():
    root = Path(__file__).resolve().parent
    folder = root / FOLDER

    # collect per-seed data
    all_data = {}
    for seed in SEEDS:
        p1 = folder / f"dpl_{seed}_metrics.log"
        p2 = folder / f"dpl_{seed}_metrics"
        if p1.exists():
            data = parse_file(p1)
        elif p2.exists():
            data = parse_file(p2)
        else:
            raise FileNotFoundError(f"Cannot find metrics file for seed {seed}")
        all_data[seed] = data

    globals_across = {}
    keys_glob = list(all_data[SEEDS[0]]['global'].keys())
    for k in keys_glob:
        vals = [all_data[s]['global'][k] for s in SEEDS]
        globals_across[k] = {
            'mean': statistics.mean(vals),
            'std': statistics.stdev(vals) if len(vals) > 1 else 0.0
        }

    sample_concepts = all_data[SEEDS[0]]['concepts']
    metric_names = list(next(iter(sample_concepts.values())).keys())

    concept_flat = {m: [] for m in metric_names}
    for s in SEEDS:
        for concept, metrics in all_data[s]['concepts'].items():
            for m in metric_names:
                concept_flat[m].append(metrics[m])

    concept_stats = {
        m: {
            'mean': statistics.mean(vals),
            'std': statistics.stdev(vals) if len(vals) > 1 else 0.0
        }
        for m, vals in concept_flat.items()
    }

    collapse_all = {}
    sample_collapse = all_data[SEEDS[0]]['collapse']
    for cname in sample_collapse.keys():
        vals = [all_data[s]['collapse'].get(cname, 0.0) for s in SEEDS]
        collapse_all[cname] = {
            'mean': statistics.mean(vals),
            'std': statistics.stdev(vals) if len(vals) > 1 else 0.0
        }

    collapse_means = [stats['mean'] for stats in collapse_all.values()]
    collapse_stds  = [stats['std']  for stats in collapse_all.values()]
    collapse_summary = {
        'mean_of_means': statistics.mean(collapse_means),
        'mean_of_stds': statistics.mean(collapse_stds)
    }

    out_file = root / "aggregated_metrics.txt"
    with out_file.open("w") as f:
        f.write("=== GLOBAL METRICS ===\n")
        for k, v in globals_across.items():
            f.write(f"{k:20s}  mean = {v['mean']:.4f},  std = {v['std']:.4f}\n")

        f.write("\n=== CONCEPT METRICS (All Concepts Combined) ===\n")
        f.write(f"{'Metric':30s}  {'Mean':>8s}  {'StdDev':>8s}\n")
        f.write("-" * 50 + "\n")
        for m, stats in concept_stats.items():
            f.write(f"{m:30s}  {stats['mean']:8.4f}  {stats['std']:8.4f}\n")

        f.write("\n=== CONCEPT COLLAPSE METRICS ===\n")
        f.write(f"{'Action':30s}  {'Mean':>8s}  {'StdDev':>8s}\n")
        f.write("-" * 50 + "\n")
        for cname, stats in collapse_all.items():
            f.write(f"{cname:30s}  {stats['mean']:8.4f}  {stats['std']:8.4f}\n")

        f.write("\nMean of collapses: {:.4f}, with std: {:.4f}\n".format(
            collapse_summary['mean_of_means'], collapse_summary['mean_of_stds']))

    print(f"Aggregated metrics written to {out_file}")

    concept_across = {}
    all_concepts = all_data[SEEDS[0]]['concepts'].keys()
    for concept in all_concepts:
        concept_across[concept] = {}
        for m in metric_names:
            vals = [all_data[s]['concepts'][concept][m] for s in SEEDS]
            concept_across[concept][m] = {
                'mean': statistics.mean(vals),
                'std': statistics.stdev(vals) if len(vals) > 1 else 0.0
            }

    with open(root / "per_concept_metrics.txt", "w") as f:
        for concept, metrics in concept_across.items():
            f.write(f"=== {concept} ===\n")
            for m, stats in metrics.items():
                f.write(f"{m:25s} mean = {stats['mean']:.4f}, std = {stats['std']:.4f}\n")
            f.write("\n")

    print("Per-concept metrics written to per_concept_metrics.txt")

if __name__ == "__main__":
    main()