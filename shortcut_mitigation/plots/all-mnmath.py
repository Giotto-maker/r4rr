import os
import re
import numpy as np
import matplotlib.pyplot as plt

MODEL_NAME = 'cbm'
BASELINE_DIR = f'../NEW-outputs/mnmath/baseline/{MODEL_NAME}-[9]-5'
PROTO_DIR = f'../NEW-outputs/mnmath/my_models/{MODEL_NAME}-[9]-5'

# Unsupervised data percentage versions
versions = [f"{v:.1f}" for v in np.arange(0.0, 1.1, 0.1)]

metrics = ['ACC(Y)', 'F1(Y)', 'ACC(C)', 'F1(C)', 'Cls(C)']
y_names = {m: m for m in metrics}
baseline_color = 'royalblue'
pnets_color = 'darkred'

def parse_results_file(filepath):
    data = {'in': {}, 'out': {}}
    section = None
    header_cols = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith('TEST (in-distribution) RESULTS:'):
                section = 'in'
                continue
            if line.startswith('TEST (out-of-distribution) RESULTS:'):
                section = 'out'
                continue
            if section in ('in', 'out') and '|' in line and 'ACC' in line:
                # header row
                header_cols = [c.strip() for c in line.split('|')]
                continue
            if section in ('in', 'out') and header_cols and '|' in line:
                # data row: assume first row = mean, second = std
                values = [float(v.strip()) for v in line.split('|')]
                if 'mean' not in data[section]:
                    data[section]['mean'] = dict(zip(header_cols, values))
                else:
                    data[section]['std'] = dict(zip(header_cols, values))
    return data

# Load baseline constant
baseline_constant_file = os.path.join(BASELINE_DIR, 'my_baseline', 'results.txt')
if not os.path.isfile(baseline_constant_file):
    raise FileNotFoundError(f"Missing baseline constant file: {baseline_constant_file}")
baseline_constant = parse_results_file(baseline_constant_file)

data_base_in = {m: {} for m in metrics}
data_base_out = {m: {} for m in metrics}
data_proto_in = {m: {} for m in metrics}
data_proto_out = {m: {} for m in metrics}

for v in versions:
    # Baseline
    base_folder = os.path.join(BASELINE_DIR, f"my_baseline-aug-{v}")
    base_file = os.path.join(base_folder, 'results.txt')
    if os.path.isfile(base_file):
        parsed = parse_results_file(base_file)
        for m in metrics:
            data_base_in[m][float(v)] = (parsed['in']['mean'][m], parsed['in']['std'][m])
            data_base_out[m][float(v)] = (parsed['out']['mean'][m], parsed['out']['std'][m])
    else:
        print(f"Missing baseline file: {base_file}")
    # Proto
    proto_folder = os.path.join(PROTO_DIR, f"episodic-proto-net-pipeline-{v}-HIDE-[]")
    proto_file = os.path.join(proto_folder, 'results.txt')
    if os.path.isfile(proto_file):
        parsed = parse_results_file(proto_file)
        for m in metrics:
            data_proto_in[m][float(v)] = (parsed['in']['mean'][m], parsed['in']['std'][m])
            data_proto_out[m][float(v)] = (parsed['out']['mean'][m], parsed['out']['std'][m])
    else:
        print(f"Missing proto file: {proto_file}")


# Plotting stuff
def plot_metric(metric, data_base, data_proto, const_mean, const_std, dist_type):
    x = sorted(data_base[metric].keys())
    if not x: return
    y_base = [data_base[metric][v][0] for v in x]
    y_base_std = [data_base[metric][v][1] for v in x]
    y_proto = [data_proto[metric].get(v, (np.nan, np.nan))[0] for v in x]
    y_proto_std = [data_proto[metric].get(v, (np.nan, np.nan))[1] for v in x]

    plt.figure(figsize=(8, 5))
    plt.plot(x, y_base, label=f'{MODEL_NAME.upper()} (Aug)', linestyle='-')
    plt.fill_between(x, np.array(y_base)-np.array(y_base_std), np.array(y_base)+np.array(y_base_std), alpha=0.2, color=baseline_color)
    plt.plot(x, y_proto, label=f'{MODEL_NAME.upper()} + PNets', linestyle='--')
    plt.fill_between(x, np.array(y_proto)-np.array(y_proto_std), np.array(y_proto)+np.array(y_proto_std), alpha=0.2, color=pnets_color)
    # constant baseline
    const_val = const_mean[dist_type]['mean'][metric]
    plt.axhline(y=const_val, color='darkgreen', linestyle='dotted', linewidth=2)
    plt.xlabel('Unsupervised Data Percentage', fontsize=30)
    y_prefix = 'OOD ' if 'out' in dist_type else 'ID '
    plt.ylabel(f'{y_prefix} {y_names[metric]}', fontsize=35)
    plt.xlim(0,1)
    
    # ensure baseline line is always within y‐limits
    lowers  = (np.array(y_base) - np.array(y_base_std)).tolist() + \
              (np.array(y_proto) - np.array(y_proto_std)).tolist() + [const_val]
    uppers  = (np.array(y_base) + np.array(y_base_std)).tolist() + \
              (np.array(y_proto) + np.array(y_proto_std)).tolist() + [const_val]
    margin = 0.05 * (max(uppers) - min(lowers))  # 5% margin
    ymin = min(lowers) - margin
    ymax = max(uppers) + margin
    plt.ylim(bottom=ymin, top=ymax)

    plt.legend(loc='best', framealpha=0.85, fontsize=28)
    plt.grid(linestyle=':')
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    out_dir = f'./.pdfs/mnmath-plots/{MODEL_NAME}-[9]-5'
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{metric.replace('(', '').replace(')','').replace(' ','_')}_{dist_type}.pdf"
    plt.savefig(os.path.join(out_dir, fname), bbox_inches='tight')
    plt.close()
    print(f"Saved {dist_type}-distribution plot for {metric} to {out_dir}/{fname}")

# run all things :)
for m in metrics:
    plot_metric(m, data_base_in, data_proto_in, baseline_constant, None, 'in')
    plot_metric(m, data_base_out, data_proto_out, baseline_constant, None, 'out')