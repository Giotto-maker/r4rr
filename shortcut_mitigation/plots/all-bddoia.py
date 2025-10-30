import os
import re
import numpy as np
import matplotlib.pyplot as plt

# Root directories for baseline and episodic-proto
BASELINE_DIR = '../outputs++/bddoia/baseline/disj/dpl'
PROTO_DIR = '../NEW-outputs/bddoia/my_models/dpl'

# Versions to iterate over
versions = [f"{v:.1f}" for v in np.arange(0.0, 1.1, 0.1)]

# Metrics to extract
metrics = [
    'ACC C', 'ACC Y',
    'F1 Y - Macro', 'F1 Y - Micro', 'F1 Y - Weighted', 'F1 Y - Binary',
    'F1 - Binary', 'F1 - Macro', 'F1 - Micro', 'F1 - Weighted',
    'Precision - Binary', 'Precision - Macro', 'Precision - Micro', 'Precision - Weighted',
    'Recall - Binary', 'Recall - Macro', 'Recall - Micro', 'Recall - Weighted',
    'Balanced Accuracy', 'Mean of collapses'
]

y_names = {
    'ACC C': 'Acc(C)',
    'ACC Y': 'Acc(Y)',
    'F1 Y - Macro': 'F1(Y) M',
    'F1 Y - Micro': 'F1(Y) μ',
    'F1 Y - Weighted': 'F1(Y) W',
    'F1 Y - Binary': 'F1(Y) B',
    'F1 - Binary': 'F1(C)',
    'F1 - Macro': 'F1(C) M',
    'F1 - Micro': 'F1(C) μ',
    'F1 - Weighted': 'F1(C) W',
    'Precision - Binary': 'Pr B',
    'Precision - Macro': 'Pr M',
    'Precision - Micro': 'Pr μ',
    'Precision - Weighted': 'Pr W',
    'Recall - Binary': 'Rec B',
    'Recall - Macro': 'Rec M',
    'Recall - Micro': 'Rec μ',
    'Recall - Weighted': 'Rec W',
    'Balanced Accuracy': 'Acc Blc',
    'Mean of collapses': 'Cls(C)'
}

baseline_color = 'royalblue'
pnets_color = 'crimson'

# Data structures: {metric: {version: (mean, std)}}
data_baseline = {m: {} for m in metrics}
data_proto = {m: {} for m in metrics}

# Regex patterns
global_pattern = re.compile(r'^(?P<metric>[^\s].*?)\s+mean = (?P<mean>[0-9.]+),\s*std = (?P<std>[0-9.]+)')
table_pattern = re.compile(r'^(?P<metric>[^\s].*?)\s+(?P<mean>[0-9.]+)\s+(?P<std>[0-9.]+)')
collapse_pattern = re.compile(r'^Mean of collapses:\s*(?P<mean>[0-9.]+), with std:\s*(?P<std>[0-9.]+)')

# Helper to parse a single file
def parse_metrics_file(filepath):
    results = {}
    section = None
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Detect sections
            if line.startswith('=== GLOBAL METRICS ==='):
                section = 'global'
                continue
            if line.startswith('=== CONCEPT METRICS'):
                section = 'table'
                continue
            if line.startswith('=== CONCEPT COLLAPSE METRICS'):
                section = 'collapse_table'
                continue
            # Parse lines based on section
            if section == 'global':
                m = global_pattern.match(line)
                if m:
                    name = m.group('metric').strip()
                    if name in metrics:
                        results[name] = (float(m.group('mean')), float(m.group('std')))
            elif section == 'table':
                # skip header separator lines
                if line.startswith('Metric') or line.startswith('---'):
                    continue
                m = table_pattern.match(line)
                if m:
                    name = m.group('metric').strip()
                    if name in metrics:
                        results[name] = (float(m.group('mean')), float(m.group('std')))
            elif section == 'collapse_table':
                m = collapse_pattern.match(line)
                if m and 'Mean of collapses' in metrics:
                    results['Mean of collapses'] = (float(m.group('mean')), float(m.group('std')))
    return results

# Parse the constant baseline file (baseline-plain-1.0)
baseline_constant_file = os.path.join(BASELINE_DIR, 'baseline-plain-1.0', 'aggregated_metrics.txt')
if not os.path.isfile(baseline_constant_file):
    raise FileNotFoundError(f"Missing constant baseline file: {baseline_constant_file}")
baseline_constant = parse_metrics_file(baseline_constant_file)


# Load data for both baseline and proto
for v in versions:
    # Baseline folder name
    base_folder = os.path.join(BASELINE_DIR, f"baseline-c-{v}")
    base_file = os.path.join(base_folder, 'aggregated_metrics.txt')
    if os.path.isfile(base_file):
        data_baseline_v = parse_metrics_file(base_file)
        for m in metrics:
            if m in data_baseline_v:
                data_baseline[m][float(v)] = data_baseline_v[m]
    else:
        print(f"Missing baseline file: {base_file}")

    # Proto folder name
    proto_folder = os.path.join(PROTO_DIR, f"[R]-episodic-proto-net-pipeline-{v}")
    proto_file = os.path.join(proto_folder, 'aggregated_metrics.txt')
    if os.path.isfile(proto_file):
        data_proto_v = parse_metrics_file(proto_file)
        for m in metrics:
            if m in data_proto_v:
                data_proto[m][float(v)] = data_proto_v[m]
    else:
        print(f"Missing proto file: {proto_file}")

# Plotting
output_dir = './.pdfs/bddoia-plots++'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for metric in metrics:
    # Prepare x and y
    x = sorted(data_baseline[metric].keys())
    if not x:
        continue
    if metric != 'Mean of collapses':
        y_base = [data_baseline[metric][v][0] / 100 for v in x]
        y_base_std = [data_baseline[metric][v][1] / 100 for v in x]
        y_proto = [data_proto[metric].get(v, (np.nan, np.nan))[0] / 100 for v in x]
        y_proto_std = [data_proto[metric].get(v, (np.nan, np.nan))[1] / 100 for v in x]
    else:
        y_base = [data_baseline[metric][v][0] for v in x]
        y_base_std = [data_baseline[metric][v][1] for v in x]
        y_proto = [data_proto[metric].get(v, (np.nan, np.nan))[0] for v in x]
        y_proto_std = [data_proto[metric].get(v, (np.nan, np.nan))[1] for v in x]

    plt.figure(figsize=(8, 5))
    # Baseline line
    plt.plot(x, y_base, label='DPL+C(S)', linestyle='dashed')
    plt.fill_between(x,
                     np.array(y_base) - np.array(y_base_std),
                     np.array(y_base) + np.array(y_base_std), alpha=0.3, color=baseline_color)
    # Proto line
    plt.plot(x, y_proto, label='DPL+PNets', linestyle='solid')
    plt.fill_between(x,
                     np.array(y_proto) - np.array(y_proto_std),
                     np.array(y_proto) + np.array(y_proto_std), alpha=0.15, color=pnets_color)
    
    # Constant baseline
    if metric in baseline_constant:
        if metric != 'Mean of collapses':
            const_val = baseline_constant[metric][0] / 100
        else:
            const_val = baseline_constant[metric][0]
        plt.axhline(y=const_val, color='darkgreen', linestyle='dotted', linewidth=2, label='_nolegend_')

    plt.xlabel('Unsupervised Data Percentage', fontsize=30)
    plt.ylabel(y_names[metric], fontsize=35)
    plt.xlim(left=0, right=1.0)
    min_baseline = min(np.array(y_base) - np.array(y_base_std))
    min_proto = min(np.array(y_proto) - np.array(y_proto_std))
    ymin = max(min(min_baseline, min_proto), 0.0)
    plt.ylim(bottom=ymin)

    plt.legend(title="", loc='best', framealpha=0.85, fontsize=28)
    plt.grid(linestyle=':')
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)

    # Save plot
    file_safe = metric.replace(' ', '_').replace('-', '').replace('/', '')
    save_path = os.path.join(output_dir, f'{file_safe}.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved plot for {metric} to {save_path}")