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
        # Check if it looks like the setup cell (references sys.path or Config)
        if any('sys.path.append' in line for line in source):
            # Check if import sys is missing
            if not any(line.strip() == 'import sys' for line in source) and not any('import sys' in line for line in source):
                
                # Insert it after magics
                new_source = []
                inserted = False
                for line in source:
                    new_source.append(line)
                    if line.startswith('%') and not inserted:
                        continue # Keep magics at top
                    if not inserted:
                        new_source.insert(len(new_source)-1, 'import sys\n')
                        inserted = True
                
                # If completely empty or just magics, append
                if not inserted:
                     new_source.append('import sys\n')

                cell['source'] = new_source
                modified = True
                print("Added 'import sys' to setup cell.")
                break
            else:
                print("'import sys' already found.")

if modified:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=4)
    print("Notebook updated successfully.")
else:
    print("No changes needed.")
