import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from brokenaxes import brokenaxes

# === Configuration ===

BASE_DIR_AUG = os.path.join(os.getcwd(), '../NEW-outputs/kandinsky', 'baseline')
BASE_DIR_PROTO = os.path.join(os.getcwd(), '../NEW-outputs/kandinsky', 'my_models')

model_names = ['sl', 'ltn', 'dpl']

collapse_type = "Collapse Hard (In)" # Collapse Hard (In)
collapse_flag = "hard" # hard

# Use distinct colors for Aug and Proto (PNet)
color_map_aug = {
    'sl': 'royalblue',
    'ltn': 'forestgreen',
    'dpl': 'darkorange'
}
color_map_proto = {
    'sl': 'royalblue',
    'ltn': 'forestgreen',
    'dpl': 'darkorange'
}
linestyles_aug = {'sl': 'dotted', 'dpl':'dashed', 'ltn': 'dashdot'}
linestyles_proto = {'sl': 'dotted', 'dpl':'dashed', 'ltn': 'dashdot'}
model_alphas = {m: 0.2 for m in model_names}
legend_labels = {
    'sl_aug': "SL (Aug)",
    'ltn_aug': "LTN (Aug)",
    'dpl_aug': "DPL (Aug)",
    'sl_proto': "SL+PNet",
    'ltn_proto': "LTN+PNet",
    'dpl_proto': "DPL+PNet"
}

# Regexes
version_regex_aug = re.compile(r'supervisions-via-augmentations-(\d+\.\d+)$')
version_regex_proto = re.compile(r'episodic-proto-net-pipeline-(\d+\.\d+)-HIDE-\[\]$')
acc_regex = re.compile(r'\$([\d.]+)\s*\\pm\s*([\d.]+)\$')

# === Data Load Function ===

def load_data(base_dir, version_regex, is_proto=False):
    data = {}
    baseline_scores = {}

    for model in model_names:
        model_path = os.path.join(base_dir, model)
        if not os.path.isdir(model_path):
            print(f"Skipping missing model path: {model_path}")
            continue

        versions = []
        concept_scores = []
        label_scores = []
        concept_acc_scores = []
        label_acc_scores = []
        collapse_scores = []

        for subfolder in sorted(os.listdir(model_path)):
            if subfolder == 'plain_baseline':
                results_file = os.path.join(model_path, subfolder, f"{model}_results.txt")
                if os.path.isfile(results_file):
                    concept_val = label_val = None
                    with open(results_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line.startswith("Concept F1 Macro"):
                                match = acc_regex.search(line)
                                if match:
                                    concept_val = float(match.group(1))
                            elif line.startswith("Label F1 Macro"):
                                match = acc_regex.search(line)
                                if match:
                                    label_val = float(match.group(1))
                            elif line.startswith("Concept Accuracy"):
                                match = acc_regex.search(line)
                                if match:
                                    concept_acc_val = float(match.group(1))
                            elif line.startswith("Label Accuracy"):
                                match = acc_regex.search(line)
                                if match:
                                    label_acc_val = float(match.group(1))
                            elif line.startswith(collapse_type):
                                match = acc_regex.search(line)
                                if match:
                                    collapse_val = float(match.group(1))
                    if concept_val is not None and label_val is not None:
                        baseline_scores[model] = {
                            'Concept': concept_val,
                            'Label': label_val, 
                            'Concept_acc': concept_acc_val,
                            'Label_acc': label_acc_val,
                            'Collapse': collapse_val
                        }
                continue

            m = version_regex.search(subfolder)
            if not m:
                continue

            version = float(m.group(1))
            results_file = os.path.join(model_path, subfolder, f"{model}_results.txt")
            if not os.path.isfile(results_file):
                continue

            concept_val = concept_unc = label_val = label_unc = None
            concept_acc_val = concept_acc_unc = label_acc_val = label_acc_unc = None
            collapse_val = collapse_unc = None
            with open(results_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("Concept F1 Macro"):
                        match = acc_regex.search(line)
                        if match:
                            concept_val = float(match.group(1))
                            concept_unc = float(match.group(2))
                    elif line.startswith("Label F1 Macro"):
                        match = acc_regex.search(line)
                        if match:
                            label_val = float(match.group(1))
                            label_unc = float(match.group(2))
                    elif line.startswith("Concept Accuracy"):
                        match = acc_regex.search(line)
                        if match:
                            concept_acc_val = float(match.group(1))
                            concept_acc_unc = float(match.group(2))
                    elif line.startswith("Label Accuracy"):
                        match = acc_regex.search(line)
                        if match:
                            label_acc_val = float(match.group(1))
                            label_acc_unc = float(match.group(2))
                    elif line.startswith(collapse_type):
                        match = acc_regex.search(line)
                        if match:
                            collapse_val = float(match.group(1))
                            collapse_unc = float(match.group(2))
            
            if concept_val is not None and label_val is not None:
                versions.append(version)
                concept_scores.append((concept_val, concept_unc))
                label_scores.append((label_val, label_unc))
                concept_acc_scores.append((concept_acc_val, concept_acc_unc))
                label_acc_scores.append((label_acc_val, label_acc_unc))
                collapse_scores.append((collapse_val, collapse_unc))

        if versions:
            sorted_idx = np.argsort(versions)
            versions = np.array(versions)[sorted_idx]
            concept_scores = np.array(concept_scores, dtype=object)[sorted_idx]
            label_scores = np.array(label_scores, dtype=object)[sorted_idx]
            concept_acc_scores = np.array(concept_acc_scores, dtype=object)[sorted_idx]
            label_acc_scores = np.array(label_acc_scores, dtype=object)[sorted_idx]
            collapse_scores = np.array(collapse_scores, dtype=object)[sorted_idx]
        else:
            versions = np.array([])

        key_prefix = f"{model}_{'proto' if is_proto else 'aug'}"
        data[key_prefix] = {
            'versions': versions,
            'Concept': concept_scores,
            'Label': label_scores,
            'ConceptAcc': concept_acc_scores,
            'LabelAcc': label_acc_scores,
            'Collapse': collapse_scores
        }

    return (data, baseline_scores) if not is_proto else data

# === Load Data ===

data_aug, baseline_scores = load_data(BASE_DIR_AUG, version_regex_aug)
data_proto = load_data(BASE_DIR_PROTO, version_regex_proto, is_proto=True)
data_all = {**data_aug, **data_proto}

# === Plot Function ===

def plot_metric(metric, filename, break_at=None, baseline_scores=None, model_type=None):
    
    assert model_type == 'proto' or model_type == 'aug' 
    
    y_lower = min([v[metric][0][0] for v in data_all.values() if len(v[metric]) > 0])
    lower_ylim = (y_lower, y_lower + 0.01)

    if break_at is not None:
        upper_ylim = (break_at, 1.0)
        ax = brokenaxes(ylims=(lower_ylim, upper_ylim), hspace=0.1, despine=True,
                        fig=plt.figure(figsize=(10, 6)))
    else:
        fig, ax = plt.subplots(figsize=(10, 6))

    for model_key in data_all:
        
        if model_type in model_key:
            versions = data_all[model_key]['versions']
            scores = data_all[model_key][metric]
            if len(versions) == 0:
                continue
            means = np.array([s[0] for s in scores])
            uncs = np.array([s[1] for s in scores])
            lower = means - uncs
            upper = means + uncs

            model = model_key.split('_')[0]
            is_proto = 'proto' in model_key
            linestyle = linestyles_proto[model] if is_proto else linestyles_aug[model]
            color = color_map_proto[model] if is_proto else color_map_aug[model]

            ax.plot(versions, means, label=legend_labels[model_key], color=color,
                    linestyle=linestyle, linewidth=3.0)
            ax.fill_between(versions, lower, upper, color=color, alpha=model_alphas[model])

    all_versions = []
    for v in data_all.values():
        if len(v['versions']) > 0:
            all_versions.extend(v['versions'])

    x_min = 0.1
    x_max = max(all_versions) if all_versions else 1
    ax.set_xlim(x_min, x_max)
    if metric == 'Concept' and model_type == 'aug':
        ax.set_ylim(top=1.0)
    ax.set_xlabel("Unsupervised Data Percentage", fontsize=35)
    if metric == 'Concept':
        ax.set_ylabel("F1(C)", fontsize=35)
    elif metric == 'Label':   
        ax.set_ylabel("F1(Y)", fontsize=35)
    elif metric == 'ConceptAcc':
        ax.set_ylabel("Acc(C)", fontsize=35)
    elif metric == 'LabelAcc':   
        ax.set_ylabel("Acc(Y)", fontsize=35)
    elif metric == 'Collapse' and not collapse_type == "Collapse Hard (In)":   
        ax.set_ylabel("Cls(C)", fontsize=35)
    elif metric == 'Collapse' and collapse_type == "Collapse Hard (In)":
        ax.set_ylabel("Err(C)", fontsize=35)

    # Plot only the maximum baseline across all models
    if baseline_scores and model_type=='aug':
        # max_val = max(baseline_scores[m][metric] for m in baseline_scores)
        # max_model = max(baseline_scores, key=lambda m: baseline_scores[m][metric])
        # print("Max_val:", max_val, type(max_val))
        # print("Means", means[0], type(means[0]))
        # ax.axhline(y=max_val, linestyle='--', linewidth=2.5, color='red', alpha=0.9)
        # ax.text(x_min + 0.01, max_val + 0.002, f'{max_model.upper()} Baseline', color='red',
        #         fontsize=22, verticalalignment='bottom')
        ax.axhline(y=means[0], linestyle='--', linewidth=2.5, color='maroon', alpha=0.9)
        ax.text(x_min + 0.01, means[0] + 0.002, f'Standard NN', color='maroon',
                fontsize=28, verticalalignment='bottom')

    if (metric == 'Collapse' and model_type == 'proto'):# or ((metric == 'ConceptAcc' or metric == 'Concept') and model_type == 'aug'):
        ax.legend(title="", loc='upper right', framealpha=0.85, fontsize=28)
    else:
        ax.legend(title="", loc='lower right', framealpha=0.85, fontsize=28)
    plt.grid(linestyle=':')
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.savefig(filename,bbox_inches="tight")
    print(f"Saved to {filename}")
    plt.close()

# === Generate Plots ===

#plot_metric('Concept', '.pdfs/kand-plots/concept_f1_combined_proto.pdf', break_at=None, model_type='proto', baseline_scores=baseline_scores)
#plot_metric('Label', '.pdfs/kand-plots/label_f1_combined_proto.pdf', break_at=None, model_type='proto', baseline_scores=baseline_scores)
#plot_metric('Concept', '.pdfs/kand-plots/concept_f1_combined_aug.pdf', break_at=None, model_type='aug', baseline_scores=baseline_scores)
#plot_metric('Label', '.pdfs/kand-plots/label_f1_combined_aug.pdf', break_at=None, model_type='aug', baseline_scores=baseline_scores)
#plot_metric('ConceptAcc', '.pdfs/kand-plots/concept_acc_combined_proto.pdf', break_at=None, model_type='proto', baseline_scores=baseline_scores)
#plot_metric('LabelAcc', '.pdfs/kand-plots/label_acc_combined_proto.pdf', break_at=None, model_type='proto', baseline_scores=baseline_scores)
#plot_metric('ConceptAcc', '.pdfs/kand-plots/concept_acc_combined_aug.pdf', break_at=None, model_type='aug', baseline_scores=baseline_scores)
#plot_metric('LabelAcc', '.pdfs/kand-plots/label_acc_combined_aug.pdf', break_at=None, model_type='aug', baseline_scores=baseline_scores)
plot_metric('Collapse', f'.pdfs/kand-plots/collapse_combined_proto_{collapse_flag}.pdf', break_at=None, model_type='proto', baseline_scores=baseline_scores)
plot_metric('Collapse', f'.pdfs/kand-plots/collapse_combined_aug_{collapse_flag}.pdf', break_at=None, model_type='aug', baseline_scores=baseline_scores)