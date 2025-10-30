#!/usr/bin/env python3
"""
summarize_results_compat.py

Compatibility-fixed version of the previous script (works on Python 3.6+).

Usage:
  python summarize_results_compat.py --results-dir ./RESULTS_DIR
"""
from pathlib import Path
import re
import argparse
import statistics
import math
import sys
from typing import List, Dict, Optional

METRICS = [
    'character_accuracy',
    'character_macro_f1',
    'character_collapse',
    'reasoning_accuracy',
    'reasoning_macro_f1',
    'reasoning_micro_f1',
]

# Robust float regex (supports scientific notation)
_FLOAT_RE = r'([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)'


def find_last_evaluation_line_in_dir(subdir: Path, match_phrase: str = 'Evaluation ended') -> Optional[str]:
    """
    Search files in the given subdirectory (non-recursive) and return the last
    line that contains `match_phrase`. If nothing is found, return None.
    """
    if not subdir.is_dir():
        return None

    candidate = None
    # iterate files in deterministic order
    for entry in sorted(subdir.iterdir()):
        if not entry.is_file():
            continue
        try:
            with entry.open('r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
        except Exception:
            # skip unreadable files
            continue

        # iterate lines in reverse to find the last matching one in this file
        for line in reversed(lines):
            if match_phrase in line:
                candidate = line.strip()
                # don't return immediately — we want the last match across all files,
                # so keep searching other files (later files in sorted order)
                break
    return candidate


def parse_metrics_from_line(line: str) -> Dict[str, float]:
    """
    Extract metrics values from the evaluation line. Returns a dict metric->float.
    Only metrics present in METRICS will be returned.
    """
    results = {}
    for m in METRICS:
        # look for patterns like "mnist_add/character_accuracy: 0.013" or "character_accuracy: 0.013"
        pat = re.compile(r'(?:[A-Za-z0-9_]+/)?' + re.escape(m) + r'\s*:\s*' + _FLOAT_RE)
        mo = pat.search(line)
        if mo:
            try:
                results[m] = float(mo.group(1))
            except Exception:
                pass
    return results


def mean_std(values: List[float]) -> (float, float):
    # keep only non-NaN numeric values
    clean = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if len(clean) == 0:
        return (math.nan, math.nan)
    if len(clean) == 1:
        return (float(clean[0]), 0.0)
    return (statistics.mean(clean), statistics.stdev(clean))


def write_summary(results_dir: Path, collected: Dict[str, List[float]], per_run_lines: Dict[str, str], out_name: str = 'summary_metrics.txt') -> Path:
    out_path = results_dir / out_name
    lines = []
    lines.append('# Summary of metrics across runs')
    lines.append('# Metric\tMean\tStdDev\tCount')

    for m in METRICS:
        vals = collected.get(m, [])
        mean, std = mean_std(vals)
        count = len([v for v in vals if not (isinstance(v, float) and math.isnan(v))])
        if math.isnan(mean):
            mean_str = 'nan'
            std_str = 'nan'
        else:
            mean_str = f'{mean:.6f}'
            std_str = f'{std:.6f}'
        lines.append(f'{m}\t{mean_str}\t{std_str}\t{count}')

    lines.append('')
    lines.append('# Per-run extracted lines (one per subfolder)')
    for subfolder, line in per_run_lines.items():
        lines.append(f'# {subfolder}: {line}')

    out_path.write_text('\n'.join(lines), encoding='utf-8')
    return out_path


def main():
    p = argparse.ArgumentParser(description='Summarize evaluation metrics across runs (subfolders of RESULTS_DIR).')
    p.add_argument('--results-dir', '-r', type=Path, default=Path('./pretrained+c'),
                   help='Path to the folder that contains run subfolders (default ./RESULTS_DIR)')
    p.add_argument('--match-phrase', type=str, default='Evaluation ended',
                   help='Phrase to look for that precedes the metrics line (default "Evaluation ended")')
    p.add_argument('--output', '-o', type=str, default='summary_metrics.txt',
                   help='Output filename to write in RESULTS_DIR (default summary_metrics.txt)')
    args = p.parse_args()

    results_dir: Path = args.results_dir
    if not results_dir.exists():
        print('ERROR: results-dir does not exist: {}'.format(results_dir), file=sys.stderr)
        sys.exit(2)

    # find immediate subdirectories
    subdirs = [p for p in sorted(results_dir.iterdir()) if p.is_dir()]
    if len(subdirs) == 0:
        print('No subdirectories found in {}'.format(results_dir), file=sys.stderr)
        sys.exit(2)

    collected: Dict[str, List[float]] = {m: [] for m in METRICS}
    per_run_lines: Dict[str, str] = {}

    for sub in subdirs:
        line = find_last_evaluation_line_in_dir(sub, match_phrase=args.match_phrase)
        per_run_lines[sub.name] = line if line is not None else '<no-eval-line-found>'
        if line is None:
            print('WARNING: no evaluation line found in subfolder {}'.format(sub.name), file=sys.stderr)
            # append NaN for each metric so lengths reflect number of runs scanned
            for m in METRICS:
                collected[m].append(math.nan)
            continue

        vals = parse_metrics_from_line(line)
        for m in METRICS:
            if m in vals:
                collected[m].append(vals[m])
            else:
                collected[m].append(math.nan)
                print('WARNING: metric {} missing in subfolder {}'.format(m, sub.name), file=sys.stderr)

    out_path = write_summary(results_dir, collected, per_run_lines, out_name=args.output)
    print('Wrote summary to: {}'.format(out_path))
    print('Done.')


if __name__ == '__main__':
    main()