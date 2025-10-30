import os
import re
from collections import defaultdict

BASE_DIR = os.path.join(os.getcwd(), '../../NEW-outputs', 'kandinsky', 'my_models', 'dpl')

folder_regex = re.compile(
    r'^episodic-proto-net-pipeline-0\.6-HIDE\[s,c\]-\[(.*?)\]-\[(.*?)\]$'
)

# Dictionary to count folders by number of digits (total from both lists).
folder_counts = defaultdict(int)
folder_matches = defaultdict(list)

# Loop through folders in the base directory.
for folder in os.listdir(BASE_DIR):
    folder_path = os.path.join(BASE_DIR, folder)
    if not os.path.isdir(folder_path):
        continue

    match = folder_regex.match(folder)
    if match:
        list1_str, list2_str = match.groups()

        # Parse digits from both bracketed lists
        list1 = [d.strip() for d in list1_str.split(',') if d.strip() != '']
        list2 = [d.strip() for d in list2_str.split(',') if d.strip() != '']
        
        total_digits = len(list1) + len(list2)

        folder_counts[total_digits] += 1
        folder_matches[total_digits].append(folder)

# Output results
for total in sorted(folder_counts.keys()):
    print(f"\nNumber of folders with {total} digit(s): {folder_counts[total]}")
    print("Matching folders:")
    for name in folder_matches[total]:
        print(f"  {name}")
