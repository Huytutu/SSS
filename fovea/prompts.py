# # mô tả hình ảnh ban đầu (bản VDGD)
# GLOBAL_DESCRIPTION_PROMPT = (
#     "I have been given this image to complete the task described as: {instruction}.\n\n"
#     "To help me complete the task, describe the given image in detail. "
#     "For real-world scenes, include foreground/background objects, properties such as color and shape, "
#     "spatial relations, counts, and visible text. For charts, graphs, tables, or diagrams, describe axes, "
#     "numbers, written text, and all other task-relevant details."
#     "Do not answer the task yet."
# )

# mô tả hình ảnh ban đầu nhưng liên quan đến câu hỏi hơn
GLOBAL_DESCRIPTION_PROMPT = (
    "I have been given this image to help answer the following question: {instruction}\n\n"
    "Describe only the objects, attributes, relations, actions, text, and spatial details "
    "visible in the image that may help answer the question. Do not infer facts that are not "
    "visually observable, and do not answer the question itself -- only describe the image."
)