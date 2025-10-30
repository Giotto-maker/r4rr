import os
import re
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Aggregate results from log files.")
parser.add_argument('--unsupervised_perc', type=str, required=True, help='Unsupervised percentage value')
args = parser.parse_args()

UNSUPERVISED_PERC = args.unsupervised_perc
LOG_FILES = {
    'TEST (in-distribution) RESULTS:': f'my_models/cbm-[9]-5/episodic-proto-net-pipeline-{UNSUPERVISED_PERC}-HIDE-[]/cbm_test_metrics.log',
    'TEST (out-of-distribution) RESULTS:': f'my_models/cbm-[9]-5/episodic-proto-net-pipeline-{UNSUPERVISED_PERC}-HIDE-[]/cbm_ood_metrics.log'
}

# ^ change to your desired output file path
SAVE_PATH = f'my_models/cbm-[9]-5/episodic-proto-net-pipeline-{UNSUPERVISED_PERC}-HIDE-[]/results.txt'
METRICS = ['ACC(Y)', 'F1(Y)', 'ACC(C)', 'F1(C)', 'Cls(C)']
ORDER = ['ACC(Y)', 'F1(Y)', 'ACC(C)', 'F1(C)', 'Cls(C)']

def parse_log(file_path):
    metric_values = {k: [] for k in METRICS}
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("Evaluation results for seed"):
            acc_f1_line = lines[i + 1].strip()
            cls_line = lines[i + 2].strip()
            
            acc_f1_matches = re.findall(r'(\w+\(\w\)):\s([0-9.]+)', acc_f1_line)
            cls_matches = re.findall(r'(\w+\(\w\)):\s([0-9.]+)', cls_line)
            
            for k, v in acc_f1_matches + cls_matches:
                if k in metric_values:
                    metric_values[k].append(float(v))
            i += 3
        else:
            i += 1

    return metric_values

def compute_stats(metric_values):
    means = []
    stds = []
    for metric in ORDER:
        values = metric_values[metric]
        means.append(np.mean(values))
        stds.append(np.std(values))
    return means, stds

def format_results(means, stds):
    header = ' | '.join([f'{m:>8}' for m in ORDER])
    mean_line = ' | '.join([f'{m:8.4f}' for m in means])
    std_line =  ' | '.join([f'{s:8.4f}' for s in stds])
    return f"{header}\n{mean_line}\n{std_line}"

def main():
    with open(SAVE_PATH, 'w') as out_file:
        for section_title, log_file in LOG_FILES.items():
            metric_values = parse_log(log_file)
            means, stds = compute_stats(metric_values)
            results = format_results(means, stds)
            out_file.write(f"{section_title}\n{results}\n\n")


if __name__ == '__main__':
    main()