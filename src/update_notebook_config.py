import json
from pathlib import Path

notebook_path = Path('z:/Active_Users_Data/Matthew/Analysis_Pipeline_Tool/Analysis_Workflow.ipynb')

if not notebook_path.exists():
    print(f"Error: {notebook_path} does not exist.")
    exit(1)

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the setup cell
modified = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        # Check if it looks like the setup cell
        if any('import sys' in line for line in source) or any('load_ext autoreload' in line for line in source):
            # Check if already present
            if any('matplotlib widget' in line for line in source):
                print("Notebook already has %matplotlib widget.")
                break
            
            # Insert it
            new_source = []
            inserted = False
            for line in source:
                new_source.append(line)
                if '%autoreload 2' in line and not inserted:
                    new_source.append('%matplotlib widget\n')
                    inserted = True
            
            if not inserted:
                # If autoreload not found, just prepend
                new_source.insert(0, '%matplotlib widget\n')
                
            cell['source'] = new_source
            modified = True
            print("Added %matplotlib widget to setup cell.")
            break

if modified:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=4)
    print("Notebook updated successfully.")
else:
    print("No changes made.")
