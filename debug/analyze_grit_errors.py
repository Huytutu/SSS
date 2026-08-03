import json

basic_path = '/home/tts26/huyng/MORAI/SSS/results/treebench_qwen2.5-vl-7b-instruct_basic.json'
grit_path = '/home/tts26/huyng/MORAI/SSS/results/treebench_qwen2.5-vl-7b-instruct_grit.json'

with open(basic_path, 'r') as f:
    basic = json.load(f)

with open(grit_path, 'r') as f:
    grit = json.load(f)

basic_details = basic.get('details', [])
grit_details = grit.get('details', [])

# Create a mapping of question_id or image to the details if they are lists
def get_mapping(details_list):
    # Depending on how details are structured, try to map by a unique identifier
    mapping = {}
    for item in details_list:
        # We assume there might be a 'question_id' or 'id' or we can just use the index if they are in the exact same order.
        # Let's check keys of the first item
        if 'question_id' in item:
            mapping[item['question_id']] = item
        elif 'id' in item:
            mapping[item['id']] = item
        else:
            # use image path + question as key maybe? or just assume same order?
            # if no ID, we will just zip them if lengths match
            pass
    return mapping

if isinstance(basic_details, list) and isinstance(grit_details, list):
    if len(basic_details) > 0 and 'question_id' in basic_details[0]:
        b_map = {item['question_id']: item for item in basic_details}
        g_map = {item['question_id']: item for item in grit_details}
        
        changed_to_wrong = []
        for qid, b_item in b_map.items():
            if qid in g_map:
                g_item = g_map[qid]
                # Assuming 'is_correct' or similar field exists
                # Or 'pred_ans' == 'ground_truth'
                b_correct = b_item.get('is_correct', b_item.get('pred_ans') == b_item.get('ground_truth'))
                g_correct = g_item.get('is_correct', g_item.get('pred_ans') == g_item.get('ground_truth'))
                
                if b_correct and not g_correct:
                    changed_to_wrong.append((qid, b_item, g_item))
                    
        print(f"Total questions where GRIT changed from Correct to Wrong: {len(changed_to_wrong)}")
        for i, (qid, b_item, g_item) in enumerate(changed_to_wrong[:3]): # print 3 examples
            print(f"\n--- Example {i+1} ---")
            print(f"Question ID: {qid}")
            print(f"Category: {b_item.get('category')}")
            print(f"Ground Truth: {b_item.get('ground_truth')}")
            print(f"Basic Prediction: {b_item.get('pred_ans')} (Correct: {b_correct})")
            print(f"GRIT Prediction: {g_item.get('pred_ans')} (Correct: {g_correct})")
            print(f"Basic Text Output:\n{b_item.get('prediction_text', b_item.get('text'))[:500]}...")
            print(f"GRIT Text Output:\n{g_item.get('prediction_text', g_item.get('text'))[:500]}...")
            if 'grit_invocations' in g_item:
                print(f"GRIT Invocations: {g_item['grit_invocations']}")
    else:
        # zip if no ID
        changed_to_wrong = []
        for b_item, g_item in zip(basic_details, grit_details):
            b_correct = b_item.get('is_correct', b_item.get('pred_ans') == b_item.get('ground_truth'))
            g_correct = g_item.get('is_correct', g_item.get('pred_ans') == g_item.get('ground_truth'))
            
            if b_correct and not g_correct:
                changed_to_wrong.append((b_item, g_item))
                
        print(f"Total questions where GRIT changed from Correct to Wrong: {len(changed_to_wrong)}")
        for i, (b_item, g_item) in enumerate(changed_to_wrong[:3]):
            print(f"\n--- Example {i+1} ---")
            print(f"Category: {b_item.get('category')}")
            print(f"Question: {b_item.get('question')}")
            print(f"Ground Truth: {b_item.get('ground_truth')}")
            print(f"Basic Prediction: {b_item.get('pred_ans')}")
            print(f"GRIT Prediction: {g_item.get('pred_ans')}")
            print(f"Basic Text: {b_item.get('prediction_text', b_item.get('output'))[:500]}")
            print(f"GRIT Text: {g_item.get('prediction_text', g_item.get('output'))[:500]}")
