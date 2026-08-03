import json

code_cell_1 = """import json
from datasets import load_dataset
from IPython.display import display

# Load basic and grit results
basic_path = '/home/tts26/huyng/MORAI/SSS/results/treebench_qwen2.5-vl-7b-instruct_basic.json'
grit_path = '/home/tts26/huyng/MORAI/SSS/results/treebench_qwen2.5-vl-7b-instruct_grit.json'

with open(basic_path, 'r') as f:
    basic = json.load(f)

with open(grit_path, 'r') as f:
    grit = json.load(f)

basic_details = basic.get('details', [])
grit_details = grit.get('details', [])

changed_to_wrong = []
for b_item, g_item in zip(basic_details, grit_details):
    b_correct = b_item.get('is_correct', b_item.get('pred_ans') == b_item.get('ground_truth'))
    g_correct = g_item.get('is_correct', g_item.get('pred_ans') == g_item.get('ground_truth'))
    
    if b_correct and not g_correct:
        changed_to_wrong.append((b_item, g_item))

print(f"Total questions where GRIT changed from Correct to Wrong: {len(changed_to_wrong)}")"""

code_cell_2 = """# Load dataset for images
print("Loading TreeBench dataset...")
dataset = load_dataset("HaochenWang/TreeBench", split="train")
print("Dataset loaded!")"""

code_cell_3 = """# Display examples from specific categories
categories_to_check = ['Perception/Physical State', 'Perception/OCR', 'Reasoning/Perspective Transform']

for cat in categories_to_check:
    print(f"\\n{'='*40}\\n==== Category: {cat} ====\\n{'='*40}")
    cat_items = [x for x in changed_to_wrong if x[0].get('category') == cat]
    
    for i, (b_item, g_item) in enumerate(cat_items[:2]):  # Show up to 2 examples per category
        idx = b_item.get('index')
        print(f"\\n--- Example {i+1} (Index: {idx}) ---")
        print(f"Question: {b_item.get('question')}")
        print(f"Ground Truth: {b_item.get('ground_truth')}")
        print(f"Basic Text:\\n{b_item.get('prediction_text', b_item.get('text'))[:400]}...")
        print(f"GRIT Text:\\n{g_item.get('prediction_text', g_item.get('text'))[:400]}...")
        
        if idx is not None:
            image = dataset[idx]['image']
            display(image)
        else:
            print("No index found to display image.")
        print("-" * 50)"""

notebook_dict = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# Analyze GRIT Errors"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\\n" for line in code_cell_1.split("\\n")]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\\n" for line in code_cell_2.split("\\n")]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\\n" for line in code_cell_3.split("\\n")]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open('debug.ipynb', 'w') as f:
    json.dump(notebook_dict, f, indent=1)

print("Notebook debug.ipynb created successfully.")
