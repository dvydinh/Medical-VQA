import json
import sys

def convert_py_to_ipynb(py_path, ipynb_path):
    with open(py_path, 'r', encoding='utf-8') as f:
        content = f.read()

    cells = []
    
    # Split content by the cell marker
    blocks = content.split("# ==== CELL")
    
    # The first block is just the header comment, maybe we put it in cell 1 or ignore
    for i, block in enumerate(blocks):
        if i == 0:
            if block.strip():
                cells.append({
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [line + '\n' for line in block.strip().split('\n')]
                })
            continue
            
        # Re-add the marker for readability
        full_block = "# ==== CELL" + block
        lines = [line + '\n' for line in full_block.strip().split('\n')]
        
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": lines
        })

    notebook = {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open(ipynb_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)

if __name__ == "__main__":
    convert_py_to_ipynb("notebooks/kaggle_resume_eval.py", "notebooks/kaggle_resume_eval.ipynb")
    print("Conversion complete.")
