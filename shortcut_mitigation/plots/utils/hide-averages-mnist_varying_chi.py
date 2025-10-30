import os
import re
import numpy as np

model_name = 'sl'
chi_value = 0.99 #####REMEMBER TO CHANGE LINR 14 AS WEL!!!!!!!!!!!!

# Define the base folder where the 'outputs' folder is located.
BASE_DIR = os.path.join(os.getcwd(), '../../NEW-outputs')
MODELS_DIR = os.path.join(BASE_DIR, 'mnadd-even-odd', 'my_models', model_name)


# Regular expression to match the folder names.
folder_regex = re.compile(r'episodic-proto-net-pipeline-1\.0-HIDE-\[(.*)\]')
folder_regex_chi = re.compile(r'episodic-proto-net-pipeline-1\.0-HIDE-\[(.*)\]-Chi0.5')
if chi_value == 0.99:
    folder_regex_chi = re.compile(r'episodic-proto-net-pipeline-1\.0-HIDE-\[(.*)\](?!-)')

# Regular expression to extract a metric value and its uncertainty from a line.
acc_regex = re.compile(r'\$([\d.]+)\s*\\pm\s*([\d.]+)\$')

def parse_results_file(file_path):
    metrics = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip blank lines.
            if not line:
                continue
            if ':' not in line:
                continue
            metric_name, value_part = line.split(':', 1)
            match = acc_regex.search(value_part)
            if match:
                val = float(match.group(1))
                unc = float(match.group(2))
                metrics[metric_name.strip()] = (val, unc)
    return metrics

grouped_folders = {i: [] for i in range(1, 11)}

# Loop over the subfolders in MODELS_DIR.
for folder in os.listdir(MODELS_DIR):
    folder_path = os.path.join(MODELS_DIR, folder)
    if not os.path.isdir(folder_path):
        continue
    m = folder_regex_chi.match(folder)
    print(folder_path)
    print(m)
    if m:
        digits_str = m.group(1)
        # Split by comma and filter out empty entries.
        digits_list = [d.strip() for d in digits_str.split(',') if d.strip() != '']
        num_digits = len(digits_list)
        if num_digits in grouped_folders:
            grouped_folders[num_digits].append(folder_path)
        else:
            print(f"Warning: Found folder {folder} with {num_digits} digits which is outside the expected range 1-5.")

# For each group, aggregate the metrics and write the new file.
for i in range(1, 11):
    folders = grouped_folders[i]
    if not folders:
        print(f"No folders found for {i} digit(s). Skipping.")
        continue

    # Dictionary to accumulate lists for each metric.
    aggregated = {}
    # Count how many files we processed for debugging.
    num_files = 0

    # Process each folder in the group.
    for folder_path in folders:
        results_file = os.path.join(folder_path, f'{model_name}_results.txt')
        if not os.path.isfile(results_file):
            print(f"Results file not found in {folder_path}. Skipping.")
            continue
        metrics = parse_results_file(results_file)
        if not metrics:
            print(f"No metrics parsed from {results_file}.")
            continue

        num_files += 1
        for metric, (val, unc) in metrics.items():
            if metric not in aggregated:
                aggregated[metric] = {'vals': [], 'uncs': []}
            aggregated[metric]['vals'].append(val)
            aggregated[metric]['uncs'].append(unc)

    if num_files == 0:
        print(f"No valid results files for {i} digit(s).")
        continue

    # Now compute the mean across all metrics for this group.
    mean_metrics = {}
    for metric, data in aggregated.items():
        mean_val = np.mean(data['vals'])
        mean_unc = np.mean(data['uncs'])
        mean_metrics[metric] = (mean_val, mean_unc)

    # Create a new folder for the aggregated result.
    new_folder_name = f'episodic-proto-net-pipeline-1.0-HIDE-mean({i})'
    new_folder_path = os.path.join(MODELS_DIR, new_folder_name)
    os.makedirs(new_folder_path, exist_ok=True)

    # Write the aggregated results to {model_name}_results.txt.
    output_file = os.path.join(new_folder_path, f'{model_name}_results_{chi_value}.txt')
    with open(output_file, 'w') as f:
        # Optionally write a header new line.
        f.write("\n")
        for metric, (mean_val, mean_unc) in mean_metrics.items():
            # Format the line in the same style.
            # e.g., "Concept Accuracy (In): $0.97 \pm 0.03$"
            line = f"{metric}: ${mean_val:.2f} \\pm {mean_unc:.2f}$\n"
            f.write(line)

    print(f"Saved aggregated results for {i} digit(s) to {output_file}")

print("Aggregation complete.")
