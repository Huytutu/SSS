# mô tả hình ảnh ban đầu (bản VDGD)
GLOBAL_DESCRIPTION_PROMPT = (
    "I have been given this image to complete the task described as: {instruction}.\n\n"
    "To help me complete the task, describe the given image in detail. "
    "For real-world scenes, include foreground/background objects, properties such as color and shape, "
    "spatial relations, counts, and visible text. For charts, graphs, tables, or diagrams, describe axes, "
    "numbers, written text, and all other task-relevant details."
    "Do not infer facts that are not visually observable, and do not answer the question itself -- only describe the image."
)

# # mô tả hình ảnh ban đầu nhưng liên quan đến câu hỏi hơn
# GLOBAL_DESCRIPTION_PROMPT = (
#     "I have been given this image to help answer the following question: {instruction}\n\n"
#     "Describe only the objects, attributes, relations, actions, text, and spatial details"
#     "visible in the image that may help answer the question. Do not infer facts that are not "
#     "visually observable, and do not answer the question itself -- only describe the image."
# )

# # mô tả hình ảnh kèm đưa base attention cho sau này
# GLOBAL_DESCRIPTION_PROMPT = (
#     "Write a general description of the image. "
#     "Describe the visually observable objects, their properties, "
#     "spatial relationships, counts, and any visible text. "
#     "For charts, graphs, tables, or diagrams, describe the visible "
#     "axes, numbers, labels, and other observable elements. "
#     "Do not infer facts that are not visually observable."
# )

# Shown before the real question (opt-in, see build_vdgd_processor's `one_shot`
# arg) to steer the model away from restating/commenting on each multiple-
# choice option one by one -- e.g. "Given the options: - A. ... - B. ..." --
# which segment_steps then splits into several spuriously "confident" steps
# (near-verbatim text is trivially predictable) and derails LeCo rollback.
# Validated on TreeBench index=344: removed the option-echo pattern entirely
# and produced a self-consistent answer, though not yet confirmed to help (or
# at least not hurt) more broadly.
ONE_SHOT_REASONING_EXAMPLE = """Example:
Question: In the image, which object is closest to the wooden chair? Options:
A. The bookshelf
B. The small red lamp
C. The potted plant

Step 1: Locate the wooden chair in the image; it is positioned near the center of the room.
Step 2: Compare the distances from the chair to each candidate object -- the bookshelf is against the far wall, the potted plant is in the corner, and the red lamp sits on a side table directly next to the chair.
Step 3: The red lamp is the nearest object to the chair.
<answer>B. The small red lamp</answer>

Now answer the following question the same way: reason step by step directly about what you see in the image, then give exactly one <answer>...</answer> at the end. Do not restate or comment on each answer option one by one.

"""