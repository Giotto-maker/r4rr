import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

model_names = ['sl', 'dpl', 'ltn']
model_name = 'sl'

BASE_DIR = os.path.join(os.getcwd(), '../NEW-outputs/kandinsky')
BASE_DIR_AUG = os.path.join(os.getcwd(), '../NEW-outputs/kandinsky', 'baseline')
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


folder_pattern = "episodic-proto-net-pipeline-0.6-HIDE-mean({})"
results_filename = f"{model_name}_results.txt"


concept_accuracies = {}
label_accuracies = {}
concept_f1s, label_f1s = {},{}
collapses = {}

for i in range(1, 5):
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
        concept_accuracies[i] = (concept_val, concept_unc)
        label_accuracies[i] = (label_val, label_unc)
        concept_f1s[i] = (concept_f1_val, concept_f1_unc)
        label_f1s[i] = (label_f1_val, label_f1_unc)
        collapses[i] = (collapse_val, collapse_unc)
    else:
        print(f"Could not parse values in file: {file_path}")

indices = sorted(concept_accuracies.keys(), reverse=True)
x = indices

y_concept = [concept_accuracies[i][0] for i in indices]
y_concept_unc = [concept_accuracies[i][1] for i in indices]
y_label = [label_accuracies[i][0] for i in indices]
y_label_unc = [label_accuracies[i][1] for i in indices]
y_f1_concept = [concept_f1s[i][0] for i in indices]
y_f1_concept_unc = [concept_f1s[i][1] for i in indices]
y_f1_label = [label_f1s[i][0] for i in indices]
y_f1_label_unc = [label_f1s[i][1] for i in indices]
y_collapse = [collapses[i][0] for i in indices]
y_collapse_unc = [collapses[i][1] for i in indices]

# Choose colors from the viridis colormap.
concept_color = 'darkorange'#viridis_colors[0]  # Yellow-ish
label_color = 'royalblue' #viridis_colors[1]  # Green-ish

plt.figure(figsize=(10, 6))

# Plot Concept Accuracy as a solid line.
plt.plot(x[1:], y_concept[1:], linestyle='solid', marker='o', color=concept_color,
         label='Acc(C)', linewidth=3.0)
# Fill the ± uncertainty area for Concept Accuracy.
plt.fill_between(x[1:],
                 [m - u for m, u in zip(y_concept[1:], y_concept_unc[1:])],
                 [m + u for m, u in zip(y_concept[1:], y_concept_unc[1:])],
                 color=concept_color, alpha=0.3)

# Plot Label Accuracy as a dotted line.
plt.plot(x[1:], y_label[1:], linestyle='dotted', marker='o', color=label_color,
         label='Acc(Y)',linewidth=3.0)
# Fill the ± uncertainty area for Label Accuracy.
plt.fill_between(x[1:],
                 [m - u for m, u in zip(y_label[1:], y_label_unc[1:])],
                 [m + u for m, u in zip(y_label[1:], y_label_unc[1:])],
                 color=label_color, alpha=0.3)

plt.axhline(y=baseline_scores[model_name]['Label Acc'], linestyle='--', linewidth=2.5, color='maroon', alpha=0.9)
plt.text(x[1] + 0.01, baseline_scores[model_name]['Label Acc'] + 0.01, f'{model_name.upper()} Acc(Y)', color='maroon',
        fontsize=32, verticalalignment='bottom')

plt.axhline(y=baseline_scores[model_name]['Concept Acc'], linestyle='--', linewidth=2.5, color='maroon', alpha=0.9)
plt.text(x[1] + 0.01, baseline_scores[model_name]['Concept Acc'] + 0.002, f'{model_name.upper()} Acc(C)', color='maroon',
        fontsize=32, verticalalignment='bottom')


# Set the x-axis ticks to show the indices and invert the axis.
plt.xticks(x,fontsize=28)
plt.yticks(fontsize=28)
plt.gca().invert_xaxis()
plt.margins(x=0)
plt.xlabel("Number of hidden classes",fontsize=35)
plt.ylabel("Accuracy",fontsize=35)
#plt.title("PNet + SL with hidden classes (Concept vs Label Accuracy)")
plt.legend(loc='lower right', fontsize=32)
plt.grid(linestyle=':')

# Save the plot instead of showing it.
save_path = f".pdfs/kand-plots/hidden_shapes/{model_name}_acc_hidden_classes_mean_results_trend_kand.pdf"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Plot saved to {save_path}")


plt.figure(figsize=(10, 6))

# Plot Concept F1-score as a solid line.
plt.plot(x[1:], y_f1_concept[1:], linestyle='solid', marker='o', color=concept_color,
         label='F1(C)',linewidth=3.0)
# Fill the ± uncertainty area for Concept Accuracy.
plt.fill_between(x[1:],
                 [m - u for m, u in zip(y_f1_concept[1:], y_f1_concept_unc[1:])],
                 [m + u for m, u in zip(y_f1_concept[1:], y_f1_concept_unc[1:])],
                 color=concept_color, alpha=0.3)

# Plot Label F1-score as a dotted line.
plt.plot(x[1:], y_f1_label[1:], linestyle='dotted', marker='o', color=label_color,
         label='F1(Y)',linewidth=3.0)
# Fill the ± uncertainty area for Label F1.
plt.fill_between(x[1:],
                 [m - u for m, u in zip(y_f1_label[1:], y_f1_label_unc[1:])],
                 [m + u for m, u in zip(y_f1_label[1:], y_f1_label_unc[1:])],
                 color=label_color, alpha=0.3)

if model_name == 'ltn':
    plt.axhline(y=baseline_scores[model_name]['Label'], linestyle='--', linewidth=2.5, color='maroon', alpha=0.9)
    plt.text(x[1] + 0.01, baseline_scores[model_name]['Label'] + 0.002, f'{model_name.upper()} F1(Y)', color='maroon',
            fontsize=32, verticalalignment='bottom')
elif model_name == 'sl':
    plt.axhline(y=baseline_scores[model_name]['Label'], linestyle='--', linewidth=2.5, color='maroon', alpha=0.9)
    plt.text(x[1] - 0.50, baseline_scores[model_name]['Label'] + 0.002, f'{model_name.upper()} F1(Y)', color='maroon',
            fontsize=32, verticalalignment='bottom')
else:
    plt.axhline(y=baseline_scores[model_name]['Label'], linestyle='--', linewidth=2.5, color='maroon', alpha=0.9)
    plt.text(x[1] + 0.01, baseline_scores[model_name]['Label'] - 0.125, f'{model_name.upper()} F1(Y)', color='maroon',
            fontsize=32, verticalalignment='bottom')

plt.axhline(y=baseline_scores[model_name]['Concept'], linestyle='--', linewidth=2.5, color='maroon', alpha=0.9)
plt.text(x[1] + 0.01, baseline_scores[model_name]['Concept'] + 0.002, f'{model_name.upper()} F1(C)', color='maroon',
        fontsize=32, verticalalignment='bottom')

# Set the x-axis ticks to show the indices and invert the axis.
plt.xticks(x,fontsize=28)
plt.yticks(fontsize=28)
plt.gca().invert_xaxis()
plt.margins(x=0)
plt.xlabel("Number of hidden classes",fontsize=35)
plt.ylabel("F1-score",fontsize=35)
##plt.title("PNet + SL with hidden classes (Concept vs Label Accuracy)")
plt.legend(loc='lower right', fontsize=32)
plt.grid( linestyle=':')

# Save the plot instead of showing it.
save_path = f".pdfs/kand-plots/hidden_shapes/{model_name}_f1_score_hidden_classes_mean_results_trend_kand.pdf"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Plot saved to {save_path}")



plt.figure(figsize=(10, 6))

# Plot Collapse as a solid line.
cls_label = 'Cls(C)' if collapse_flag == 'soft' else 'Err(C)'
plt.plot(x[1:], y_collapse[1:], linestyle='solid', marker='o', color=concept_color,
         label=cls_label,linewidth=3.0)
# Fill the ± uncertainty area for Concept Accuracy.
plt.fill_between(x[1:],
                 [m - u for m, u in zip(y_collapse[1:], y_collapse_unc[1:])],
                 [m + u for m, u in zip(y_collapse[1:], y_collapse_unc[1:])],
                 color=concept_color, alpha=0.3)

# # Plot Label F1-score as a dotted line.
# plt.plot(x[1:], y_f1_label[1:], linestyle='dotted', marker='o', color=label_color,
#          label='F1(Y)',linewidth=3.0)
# # Fill the ± uncertainty area for Label F1.
# plt.fill_between(x[1:],
#                  [m - u for m, u in zip(y_f1_label[1:], y_f1_label_unc[1:])],
#                  [m + u for m, u in zip(y_f1_label[1:], y_f1_label_unc[1:])],
#                  color=label_color, alpha=0.3)


# plt.axhline(y=baseline_scores[model_name]['Label'], linestyle='--', linewidth=2.5, color='maroon', alpha=0.9)
# plt.text(x[1] + 0.01, baseline_scores[model_name]['Label'] - 0.125, f'{model_name.upper()} F1(Y)', color='maroon',
#         fontsize=32, verticalalignment='bottom')

plt.axhline(y=baseline_scores[model_name]['Collapse'], linestyle='--', linewidth=2.5, color='maroon', alpha=0.9)
if model_name == 'dpl' or model_name == 'ltn':
    plt.text(x[-1] + 0.85, baseline_scores[model_name]['Collapse'] - 0.025, f'{model_name.upper()}', color='maroon',
            fontsize=32, verticalalignment='top')
else:
    plt.text(x[-1] + 0.55, baseline_scores[model_name]['Collapse'] - 0.025, f'{model_name.upper()}', color='maroon',
            fontsize=32, verticalalignment='top')

# Set the x-axis ticks to show the indices and invert the axis.
plt.xticks(x,fontsize=28)
plt.yticks(fontsize=28)
plt.gca().invert_xaxis()
plt.margins(x=0)
plt.xlabel("Number of hidden classes",fontsize=35)
if collapse_flag == 'soft':
    plt.ylabel("Concept Collapse",fontsize=35)
else:
    plt.ylabel("Concept Error",fontsize=35)
##plt.title("PNet + SL with hidden classes (Concept vs Label Accuracy)")
plt.legend(loc='lower left', fontsize=32)
plt.grid( linestyle=':')

# Save the plot instead of showing it.
save_path = f".pdfs/kand-plots/hidden_shapes/{model_name}_collapse_{collapse_flag}_score_hidden_classes_mean_results_trend_kand.pdf"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Plot saved to {save_path}")