import json

notebook_path = '/home/tts26/huyng/MORAI/SSS/debug/debug.ipynb'

with open(notebook_path, 'r') as f:
    nb = json.load(f)

new_cell_source = """from datasets import load_dataset
from IPython.display import display

# Load dataset for images
print("Loading TreeBench dataset...")
dataset = load_dataset("HaochenWang/TreeBench", split="train")
print("Dataset loaded!")

print("\\nDisplaying images for the first 2 physical state errors:")
physical_state_items = [x for x in changed_to_wrong if x[0].get('category') == 'Perception/Physical State']

for i, (b_item, g_item) in enumerate(physical_state_items[:2]):
    idx = b_item.get('index')
    print(f"\\n--- Error {i+1} ---")
    print(f"Question: {b_item.get('question')}")
    print(f"Ground Truth: {b_item.get('ground_truth')}")
    
    if idx is not None:
        image = dataset[idx]['image']
        display(image)
    else:
        print("No index found to display image.")
"""

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in new_cell_source.split('\n')][:-1] # avoid extra newline at end
}

nb['cells'].append(new_cell)

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Appended cell to notebook.")
