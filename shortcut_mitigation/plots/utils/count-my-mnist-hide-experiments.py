import os
import re

# Define the base folder where the 'outputs' folder is located.
BASE_DIR = os.path.join(os.getcwd(), '../../NEW-outputs')
MODELS_DIR = os.path.join(BASE_DIR, 'mnadd-even-odd', 'my_models', 'ltn')

# Regular expression to match the folder names and capture the content inside the square brackets.
folder_regex = re.compile(r'episodic-proto-net-pipeline-1\.0-HIDE-\[(.*)\]')

# Initialize a dictionary to count folders with a list length from 1 to 10.
folder_counts = {i: 0 for i in range(1, 11)}
digit_groups = {i: [] for i in range(1, 11)}

# Loop over the subfolders in MODELS_DIR.
for folder in os.listdir(MODELS_DIR):
    folder_path = os.path.join(MODELS_DIR, folder)
    if not os.path.isdir(folder_path):
        continue
    m = folder_regex.match(folder)
    if m:
        # Get the string inside the square brackets.
        digits_str = m.group(1)
        # Split by comma and remove any empty entries.
        digits_list = [d.strip() for d in digits_str.split(',') if d.strip() != '']
        num_digits = len(digits_list)
        if num_digits in folder_counts:
            folder_counts[num_digits] += 1
            digit_groups[num_digits].append(digits_list)
        else:
            print(f"Warning: Folder {folder} has {num_digits} digits, which is outside the expected range 1-10.")

# Print the results.
for length in range(1, 11):
    print(f"\nNumber of folders with {length} digit(s): {folder_counts[length]}")
    if digit_groups[length]:
        print("Digits found:")
        for digits in digit_groups[length]:
            print(f"  {digits}")
