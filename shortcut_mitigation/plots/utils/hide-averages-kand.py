import os
import re
import numpy as np
from collections import defaultdict

# Set base directory
BASE_DIR = os.path.join(os.getcwd(), '../../NEW-outputs', 'kandinsky', 'my_models', 'sl')

print(BASE_DIR)

# Match folders like episodic-proto-net-pipeline-0.6-HIDE[s,c]-[x, y]-[z, w]
folder_regex = re.compile(
    r'^episodic-proto-net-pipeline-0\.6-HIDE\[s,c\]-\[(.*?)\]-\[(.*?)\]-Chi0.2$'
)

# Match lines like: Concept Accuracy (In): $0.97 \pm 0.03$
acc_regex = re.compile(r'\$([\d.]+)\s*\\pm\s*([\d.]+)\$')

# Group folders by total digit count across both [x,y] and [z,w]
grouped_folders = defaultdict(list)

def parse_results_file(file_path):
    """Extract metric name and (value, uncertainty) from sl_results.txt"""
    metrics = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if ':' not in line:
                continue
            metric_name, value_part = line.split(':', 1)
            match = acc_regex.search(value_part)
            if match:
                val = float(match.group(1))
                unc = float(match.group(2))
                metrics[metric_name.strip()] = (val, unc)
    return metrics

# Scan directory for matching folders
for folder in os.listdir(BASE_DIR):
    print(folder)
    full_path = os.path.join(BASE_DIR, folder)
    if not os.path.isdir(full_path):
        continue

    match = folder_regex.match(folder)
    print(match)
    if match:
        list1_str, list2_str = match.groups()
        list1 = [d.strip() for d in list1_str.split(',') if d.strip()]
        list2 = [d.strip() for d in list2_str.split(',') if d.strip()]
        total_digits = len(list1) + len(list2)
        grouped_folders[total_digits].append(full_path)

# Aggregate metrics per digit count group
for digit_count, folders in sorted(grouped_folders.items()):
    if not folders:
        print(f"No folders with {digit_count} digits.")
        continue

    print(f"\nProcessing {len(folders)} folders with {digit_count} digit(s):")

    aggregated = defaultdict(lambda: {'vals': [], 'uncs': []})
    processed_files = 0

    for folder_path in folders:
        results_file = os.path.join(folder_path, 'sl_results.txt')
        if not os.path.isfile(results_file):
            print(f"  [Warning] Missing sl_results.txt in {folder_path}")
            continue

        metrics = parse_results_file(results_file)
        if not metrics:
            print(f"  [Warning] No metrics found in {results_file}")
            continue

        processed_files += 1
        for metric, (val, unc) in metrics.items():
            aggregated[metric]['vals'].append(val)
            aggregated[metric]['uncs'].append(unc)

    if processed_files == 0:
        print(f"  [Error] No valid result files for {digit_count} digit(s).")
        continue

    # Compute mean values and uncertainties
    print(f"\nAveraged Metrics for {digit_count} digit(s):")
    for metric, data in aggregated.items():
        mean_val = np.mean(data['vals'])
        mean_unc = np.mean(data['uncs'])
        print(f"  {metric}: ${mean_val:.2f} \\pm {mean_unc:.2f}$")

    # Optional: Save to new folder
    out_folder = os.path.join(BASE_DIR, f'episodic-proto-net-pipeline-0.6-HIDE-mean({digit_count})-Chi0.2')
    os.makedirs(out_folder, exist_ok=True)
    out_file = os.path.join(out_folder, 'sl_results.txt')
    with open(out_file, 'w') as f:
        f.write("\n")
        for metric, data in aggregated.items():
            mean_val = np.mean(data['vals'])
            mean_unc = np.mean(data['uncs'])
            f.write(f"{metric}: ${mean_val:.2f} \\pm {mean_unc:.2f}$\n")

    print(f"\nSaved averaged results to {out_file}")