import os
import re
import argparse
from collections import defaultdict

def parse_per_concept(file_path):
    data = {}  # concept -> metric -> (mean, std)
    current = None
    pattern = re.compile(r"^(?P<metric>.+?)\s+mean\s*=\s*(?P<mean>[0-9.]+),\s*std\s*=\s*(?P<std>[0-9.]+)")
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # section header
            if line.startswith('===') and line.endswith('==='):
                current = line.strip('= ').lower().replace(' ', '_')
                data[current] = {}
            else:
                m = pattern.match(line)
                if m and current:
                    metric = m.group('metric')
                    mean = float(m.group('mean'))
                    std = float(m.group('std'))
                    data[current][metric] = (mean, std)
    return data

def aggregate(data, groups):
    # groups: action -> list of concepts
    result = {}
    for action, concepts in groups.items():
        # metric -> list of means, list of stds
        agg = defaultdict(lambda: {'means': [], 'stds': []})
        for concept in concepts:
            if concept not in data:
                continue
            for metric, (mean, std) in data[concept].items():
                agg[metric]['means'].append(mean)
                agg[metric]['stds'].append(std)
        # compute averages
        result[action] = {}
        for metric, vals in agg.items():
            if vals['means']:
                mean_avg = sum(vals['means']) / len(vals['means'])
                std_avg = sum(vals['stds']) / len(vals['stds'])
                result[action][metric] = (mean_avg, std_avg)
        
    return result

def print_aggregated(agg):
    for action in ['forward', 'stop', 'left', 'right']:
        if action not in agg:
            continue
        print(f"=== {action.upper()} ===")
        for metric, (mean, std) in sorted(agg[action].items()):
            print(f"{metric:<25} mean = {mean:.4f}, std = {std:.4f}")
        print()

if __name__ == '__main__':
    FOLDER = "baseline/dpl/PRE-baseline"
    parser = argparse.ArgumentParser(description='Aggregate per-concept metrics by action')
    parser.add_argument('--folder', type=str, default=FOLDER, help='Folder containing per_concept_metrics.txt')
    args = parser.parse_args()
    file_path = os.path.join(args.folder, 'per_concept_metrics.txt')

    # define mapping of concepts to actions
    groups = {
        'forward': ['green_traffic_light', 'follow_traffic', 'road_is_clear'],
        'stop': ['red_traffic_light', 'traffic_sign', 'obstacle_car', 'obstacle_pedestrian', 'obstacle_rider', 'obstacle_others'],
        'left': ['no_lane_on_the_left', 'obstacle_on_the_left_lane', 'solid_left_line', 'on_left_turn_lane', 'traffic_light_allows_left', 'front_car_turning_left'],
        'right': ['on_the_right_turn_lane', 'traffic_light_allows_right', 'front_car_turning_right', 'no_lane_on_the_right', 'obstacle_on_the_right_lane', 'solid_right_line'],
    }

    data = parse_per_concept(file_path)
    agg = aggregate(data, groups)
    print_aggregated(agg)
    output_file = os.path.join(args.folder, 'per_concept_action_metrics.txt')
    with open(output_file, 'w') as f:
        for action in ['forward', 'stop', 'left', 'right']:
            if action not in agg:
                continue
            f.write(f"=== {action.upper()} ===\n")
            for metric, (mean, std) in sorted(agg[action].items()):
                f.write(f"{metric:<25} mean = {mean:.4f}, std = {std:.4f}\n")
            f.write('\n')
