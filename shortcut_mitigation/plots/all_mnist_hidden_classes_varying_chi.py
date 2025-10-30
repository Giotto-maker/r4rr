import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

model_names = ['sl', 'dpl', 'ltn']
model_name = 'sl'

BASE_DIR = os.path.join(os.getcwd(), '../NEW-outputs/mnadd-even-odd')
BASE_DIR_AUG = os.path.join(os.getcwd(), '../NEW-outputs/mnadd-even-odd', 'baseline')
MODELS_DIR = os.path.join(BASE_DIR, 'my_models', model_name)

version_regex_aug = re.compile(r'supervisions-via-augmentations-(\d+\.\d+)$')
acc_regex = re.compile(r'\$([\d.]+)\s*\\pm\s*([\d.]+)\$')

collapse_type = "Collapse (In)" # Collapse Hard (In)
collapse_flag = "soft" # hard

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

        for subfolder in sorted(os.listdir(model_path)):
            if subfolder == 'plain_baseline':
                results_file = os.path.join(model_path, subfolder, f"{model}_results.txt")
                if os.path.isfile(results_file):
                    concept_val = label_val = None
                    acc_concept_val, acc_label_val = None, None
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
                                    acc_concept_val = float(match.group(1))
                            elif line.startswith("Label Accuracy"):
                                match = acc_regex.search(line)
                                if match:
                                    acc_label_val = float(match.group(1))
                            elif line.startswith(collapse_type):
                                match = acc_regex.search(line)
                                if match:
                                    collapse_val = float(match.group(1))
                            
                    if concept_val is not None and label_val is not None:
                        baseline_scores[model] = {
                            'Concept': concept_val,
                            'Label': label_val,
                            'Concept Acc': acc_concept_val,
                            'Label Acc': acc_label_val,
                            'Collapse': collapse_val
                        }
                continue

        #     m = version_regex.search(subfolder)
        #     if not m:
        #         continue

        #     version = float(m.group(1))
        #     results_file = os.path.join(model_path, subfolder, f"{model}_results.txt")
        #     if not os.path.isfile(results_file):
        #         continue

        #     concept_val = concept_unc = label_val = label_unc = None
        #     with open(results_file, 'r') as f:
        #         for line in f:
        #             line = line.strip()
        #             if line.startswith("Concept F1 Macro"):
        #                 match = acc_regex.search(line)
        #                 if match:
        #                     concept_val = float(match.group(1))
        #                     concept_unc = float(match.group(2))
        #             elif line.startswith("Label F1 Macro"):
        #                 match = acc_regex.search(line)
        #                 if match:
        #                     label_val = float(match.group(1))
        #                     label_unc = float(match.group(2))

        #     if concept_val is not None and label_val is not None:
        #         versions.append(version)
        #         concept_scores.append((concept_val, concept_unc))
        #         label_scores.append((label_val, label_unc))

        # if versions:
        #     sorted_idx = np.argsort(versions)
        #     versions = np.array(versions)[sorted_idx]
        #     concept_scores = np.array(concept_scores, dtype=object)[sorted_idx]
        #     label_scores = np.array(label_scores, dtype=object)[sorted_idx]
        # else:
        #     versions = np.array([])

        # key_prefix = f"{model}_{'proto' if is_proto else 'aug'}"
        # data[key_prefix] = {
        #     'versions': versions,
        #     'Concept': concept_scores,
        #     'Label': label_scores
        # }

    return (data, baseline_scores) if not is_proto else data

data_aug, baseline_scores = load_data(BASE_DIR_AUG, version_regex_aug)


folder_pattern = "episodic-proto-net-pipeline-1.0-HIDE-mean({})"

concept_accuracies = {}
label_accuracies = {}
concept_f1s, label_f1s = {},{}
collapses = {}

y_concept, y_concept_unc = {},{}
y_label, y_label_unc = {},{}
y_f1_concept, y_f1_concept_unc = {},{}
y_f1_label, y_f1_label_unc = {},{}
y_collapse, y_collapse_unc = {},{}

for chi in [0.2, 0.5, 0.99]:

    if chi == 0.99:
        results_filename = f"{model_name}_results.txt"
    else:
        results_filename = f"{model_name}_results_{chi}.txt"


    concept_accuracies[chi] = {}
    label_accuracies[chi] = {}
    concept_f1s[chi], label_f1s[chi] = {},{}
    collapses[chi] = {}

    for i in range(1, 10):
        folder_name = folder_pattern.format(i)
        folder_path = os.path.join(MODELS_DIR, folder_name)
        file_path = os.path.join(folder_path, results_filename)
        
        if not os.path.isfile(file_path):
            print(f"File not found: {file_path}")
            continue

        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip() != '']
        
        concept_val = None
        concept_unc = None
        label_val = None
        label_unc = None
        concept_f1_val, concept_f1_unc = None, None
        label_f1_val, label_f1_unc = None, None
        
        for line in lines:
            if line.startswith("Concept Accuracy"):
                match = acc_regex.search(line)
                if match:
                    concept_val = float(match.group(1))
                    concept_unc = float(match.group(2))
            elif line.startswith("Label Accuracy"):
                match = acc_regex.search(line)
                if match:
                    label_val = float(match.group(1))
                    label_unc = float(match.group(2))
            elif line.startswith("Concept F1 Macro"):
                match = acc_regex.search(line)
                if match:
                    concept_f1_val = float(match.group(1))
                    concept_f1_unc = float(match.group(2))
            elif line.startswith("Label F1 Macro (In)"):
                match = acc_regex.search(line)
                if match:
                    label_f1_val = float(match.group(1))
                    label_f1_unc = float(match.group(2))
            elif line.startswith(collapse_type):
                match = acc_regex.search(line)
                if match:
                    collapse_val = float(match.group(1))
                    collapse_unc = float(match.group(2))
    
        if concept_val is not None and label_val is not None and concept_f1_val is not None and label_f1_val is not None and collapse_val is not None:
            concept_accuracies[chi][i] = (concept_val, concept_unc)
            label_accuracies[chi][i] = (label_val, label_unc)
            concept_f1s[chi][i] = (concept_f1_val, concept_f1_unc)
            label_f1s[chi][i] = (label_f1_val, label_f1_unc)
            collapses[chi][i] = (collapse_val, collapse_unc)
        else:
            print(f"Could not parse values in file: {file_path}")

    indices = sorted(concept_accuracies[chi].keys(), reverse=True)
    x = indices
    
    y_concept[chi] = [concept_accuracies[chi][i][0] for i in indices]
    y_concept_unc[chi] = [concept_accuracies[chi][i][1] for i in indices]
    y_label[chi] = [label_accuracies[chi][i][0] for i in indices]
    y_label_unc[chi] = [label_accuracies[chi][i][1] for i in indices]
    y_f1_concept[chi] = [concept_f1s[chi][i][0] for i in indices]
    y_f1_concept_unc[chi] = [concept_f1s[chi][i][1] for i in indices]
    y_f1_label[chi] = [label_f1s[chi][i][0] for i in indices]
    y_f1_label_unc[chi] = [label_f1s[chi][i][1] for i in indices]
    y_collapse[chi] = [collapses[chi][i][0] for i in indices]
    y_collapse_unc[chi] = [collapses[chi][i][1] for i in indices]


colors = {0.2: 'royalblue', 0.5: 'forestgreen', 0.99: 'darkorange'}
plt.figure(figsize=(10, 6))

# Plot Concept Accuracy as a solid line.
plt.plot(x[1:], y_concept[0.2][1:], linestyle='dotted', marker='o', color=colors[0.2],
         label='p=0.2', linewidth=3.0)
plt.plot(x[1:], y_concept[0.5][1:], linestyle='dashed', marker='o', color=colors[0.5],
         label='p=0.5', linewidth=3.0)
plt.plot(x[1:], y_concept[0.99][1:], linestyle='solid', marker='o', color=colors[0.99],
         label='p=0.99', linewidth=3.0)

# Fill the ± uncertainty area for Concept Accuracy.
plt.fill_between(x[1:],
                 [m - u for m, u in zip(y_concept[0.2][1:], y_concept_unc[0.2][1:])],
                 [m + u for m, u in zip(y_concept[0.2][1:], y_concept_unc[0.2][1:])],
                 color=colors[0.2], alpha=0.3)
plt.fill_between(x[1:],
                 [m - u for m, u in zip(y_concept[0.5][1:], y_concept_unc[0.5][1:])],
                 [m + u for m, u in zip(y_concept[0.5][1:], y_concept_unc[0.5][1:])],
                 color=colors[0.5], alpha=0.3)
plt.fill_between(x[1:],
                 [m - u for m, u in zip(y_concept[0.99][1:], y_concept_unc[0.99][1:])],
                 [m + u for m, u in zip(y_concept[0.99][1:], y_concept_unc[0.99][1:])],
                 color=colors[0.99], alpha=0.3)


# Set the x-axis ticks to show the indices and invert the axis.
plt.xticks(x,fontsize=28)
plt.yticks(fontsize=28)
plt.gca().invert_xaxis()
plt.margins(x=0)
plt.xlabel("Number of hidden classes",fontsize=35)
plt.ylabel("Acc(c)",fontsize=35)
#plt.title("PNet + SL with hidden classes (Concept vs Label Accuracy)")
plt.legend(loc='upper left', fontsize=32)
plt.grid(linestyle=':')

# Save the plot instead of showing it.
save_path = f".pdfs/mnist-plots/hidden_digits/varying_chi_{model_name}_acc_hidden_classes_mean_results_trend.pdf"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Plot saved to {save_path}")



plt.figure(figsize=(10, 6))

# Plot Concept F1-score as a solid line.
plt.plot(x[1:], y_f1_concept[0.2][1:], linestyle='dotted', marker='o', color=colors[0.2],
         label='p=0.2', linewidth=3.0)
plt.plot(x[1:], y_f1_concept[0.5][1:], linestyle='dashed', marker='o', color=colors[0.5],
         label='p=0.5', linewidth=3.0)
plt.plot(x[1:], y_f1_concept[0.99][1:], linestyle='solid', marker='o', color=colors[0.99],
         label='p=0.99', linewidth=3.0)

# Fill the ± uncertainty area for Concept Accuracy.
plt.fill_between(x[1:],
                 [m - u for m, u in zip(y_f1_concept[0.2][1:], y_f1_concept_unc[0.2][1:])],
                 [m + u for m, u in zip(y_f1_concept[0.2][1:], y_f1_concept_unc[0.2][1:])],
                 color=colors[0.2], alpha=0.3)
plt.fill_between(x[1:],
                 [m - u for m, u in zip(y_f1_concept[0.5][1:], y_f1_concept_unc[0.5][1:])],
                 [m + u for m, u in zip(y_f1_concept[0.5][1:], y_f1_concept_unc[0.5][1:])],
                 color=colors[0.5], alpha=0.3)
plt.fill_between(x[1:],
                 [m - u for m, u in zip(y_f1_concept[0.99][1:], y_f1_concept_unc[0.99][1:])],
                 [m + u for m, u in zip(y_f1_concept[0.99][1:], y_f1_concept_unc[0.99][1:])],
                 color=colors[0.99], alpha=0.3)


# Set the x-axis ticks to show the indices and invert the axis.
plt.xticks(x,fontsize=28)
plt.yticks(fontsize=28)
plt.gca().invert_xaxis()
plt.margins(x=0)
plt.xlabel("Number of hidden classes",fontsize=35)
plt.ylabel("F1(C)",fontsize=35)
#plt.title("PNet + SL with hidden classes (Concept vs Label Accuracy)")
plt.legend(loc='upper left', fontsize=32)
plt.grid(linestyle=':')

# Save the plot instead of showing it.
save_path = f".pdfs/mnist-plots/hidden_digits/varying_chi_{model_name}_f1_hidden_classes_mean_results_trend.pdf"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Plot saved to {save_path}")




plt.figure(figsize=(10, 6))

# Plot Concept F1-score as a solid line.
plt.plot(x[1:], y_collapse[0.2][1:], linestyle='dotted', marker='o', color=colors[0.2],
         label='p=0.2', linewidth=3.0)
plt.plot(x[1:], y_collapse[0.5][1:], linestyle='dashed', marker='o', color=colors[0.5],
         label='p=0.5', linewidth=3.0)
plt.plot(x[1:], y_collapse[0.99][1:], linestyle='solid', marker='o', color=colors[0.99],
         label='p=0.99', linewidth=3.0)

# Fill the ± uncertainty area for Concept Accuracy.
plt.fill_between(x[1:],
                 [m - u for m, u in zip(y_collapse[0.2][1:], y_collapse_unc[0.2][1:])],
                 [m + u for m, u in zip(y_collapse[0.2][1:], y_collapse_unc[0.2][1:])],
                 color=colors[0.2], alpha=0.3)
plt.fill_between(x[1:],
                 [m - u for m, u in zip(y_collapse[0.5][1:], y_collapse_unc[0.5][1:])],
                 [m + u for m, u in zip(y_collapse[0.5][1:], y_collapse_unc[0.5][1:])],
                 color=colors[0.5], alpha=0.3)
plt.fill_between(x[1:],
                 [m - u for m, u in zip(y_collapse[0.99][1:], y_collapse_unc[0.99][1:])],
                 [m + u for m, u in zip(y_collapse[0.99][1:], y_collapse_unc[0.99][1:])],
                 color=colors[0.99], alpha=0.3)


# Set the x-axis ticks to show the indices and invert the axis.
plt.xticks(x,fontsize=28)
plt.yticks(fontsize=28)
plt.gca().invert_xaxis()
plt.margins(x=0)
plt.xlabel("Number of hidden classes",fontsize=35)
cls_label = 'Cls(C)' if collapse_flag == 'soft' else 'Err(C)'
plt.ylabel(cls_label,fontsize=35)
#plt.title("PNet + SL with hidden classes (Concept vs Label Accuracy)")
plt.legend(loc='lower left', fontsize=32)
plt.grid(linestyle=':')

# Save the plot instead of showing it.
save_path = f".pdfs/mnist-plots/hidden_digits/varying_chi_{model_name}_{collapse_flag}_collapse_hidden_classes_mean_results_trend.pdf"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Plot saved to {save_path}")