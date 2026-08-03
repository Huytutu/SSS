import json

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

categories_to_check = ['Perception/Physical State', 'Perception/OCR']

for cat in categories_to_check:
    print(f"\n==== Category: {cat} ====")
    cat_items = [x for x in changed_to_wrong if x[0].get('category') == cat]
    for i, (b_item, g_item) in enumerate(cat_items[:2]):
        print(f"\n--- Example {i+1} ---")
        print(f"Question: {b_item.get('question')}")
        print(f"Ground Truth: {b_item.get('ground_truth')}")
        print(f"Basic Text:\n{b_item.get('prediction_text', b_item.get('text'))[:300]}...")
        print(f"GRIT Text:\n{g_item.get('prediction_text', g_item.get('text'))[:300]}...")
