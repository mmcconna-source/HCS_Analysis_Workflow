import json
import os

notebook_path = r'z:\Active_Users_Data\Matthew\Analysis_Pipeline_Tool\Analysis_Workflow.ipynb'

def add_cell():
    if not os.path.exists(notebook_path):
        print(f"Error: Notebook not found at {notebook_path}")
        return

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Check if cell already exists to avoid duplicates
    for cell in nb['cells']:
        if "WassersteinDistanceWidget" in "".join(cell['source']):
            print("Wasserstein widget cell already exists.")
            return

    new_markdown_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Wasserstein Distance Analysis\n",
            "\n",
            "Calculate and visualize the Wasserstein distance between a Reference condition and Test conditions."
        ]
    }

    new_code_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from src.wasserstein_widget import WassersteinDistanceWidget\n",
            "\n",
            "# Initialize and display the widget\n",
            "# We use df_cell as it contains the feature data\n",
            "# Ensure df_cell is loaded/available from previous cells\n",
            "if 'df_cell' in locals() and df_cell is not None:\n",
            "    wasserstein_widget = WassersteinDistanceWidget(df_cell)\n",
            "    wasserstein_widget.display()\n",
            "else:\n",
            "    print(\"Error: df_cell is not defined. Please load data first.\")"
        ]
    }

    nb['cells'].append(new_markdown_cell)
    nb['cells'].append(new_code_cell)

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print(f"Successfully added Wasserstein widget cells to {notebook_path}")

if __name__ == "__main__":
    add_cell()
