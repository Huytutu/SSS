__all__ = ["BASELINE_DESCRIPTION_PROMPT", "CORRECT_PROMPT_TEMPLATE"]

# Question-agnostic prompt used only to measure baseline attention, never to
# produce text we keep. Verbatim from MLLMs Know Where to Look
# (arXiv:2502.17422, their run.py `general_question`): the relative-attention
# map is this prompt's image attention divided into the real question's, which
# cancels the question-independent bias -- border patches, high-frequency
# texture -- that makes raw attention a poor localizer.
BASELINE_DESCRIPTION_PROMPT = "Write a general description of the image."

# Shown to the base model together with the full image AND a crop of the region
# it was attending to when the token was flagged. Asks for two things: which
# candidate to emit, and one visual fact justifying it (which becomes evidence
# for the rest of the generation).
CORRECT_PROMPT_TEMPLATE = (
    "You are shown an image, then a zoomed-in crop of the region under discussion.\n\n"
    "A model is writing an answer and is unsure about the next word. Here is the "
    "text it has written so far:\n\"{prefix}\"\n\n"
    "Candidate next words:\n{candidates}\n\n"
    "Look carefully at the crop and answer in exactly this format:\n"
    "<evidence>One short sentence (at most 30 words) stating a fact you can SEE in the "
    "image that makes your choice correct. If nothing visual is relevant here, write: None"
    "</evidence>\n"
    "<answer>the index of your chosen candidate, a single integer</answer>\n\n"
    "Rules:\n"
    "- Choose only from the candidates listed above.\n"
    "- The evidence must describe the image, not the text written so far.\n"
    "- Do not mention candidate words or indices inside <evidence>.\n"
)
