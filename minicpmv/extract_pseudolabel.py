# -*- coding: utf-8 -*-
import os
import json
import re
import torch
from torch.nn.parallel import DataParallel
from chat import MiniCPMVChat

torch.manual_seed(0)

# =========================
# 1. Model path
# =========================
MODEL_PATH = '../Yourmodel_dir/MiniCPM-V-main/MiniCPM-Llama3-V-2_5/'

# =========================
# 2. Input and output directories
# =========================
INPUT_ROOT = '../input_texts/'
OUTPUT_ROOT = '../output_texts/'

# =========================
# 3. Number of candidate pseudo labels generated for each original label
# =========================
NUM_CANDIDATES = 5

# Load MiniCPM-V
chat_model = MiniCPMVChat(MODEL_PATH)


def sanitize_label_for_filename(label):
    """
    Convert an action label into a stable file name.

    Examples:
        'opening door' -> 'openingdoor'
        'pouring water' -> 'pouringwater'
    """
    label = label.strip().lower()
    label = label.replace(" ", "")
    label = re.sub(r'[\\/:*?"<>|]', '', label)
    return label


def build_prompt(original_label, num_candidates=5):
    """
    Build the pseudo-label generation prompt.
    """
    prompt = f"""
You are given an original action label from a human action recognition dataset.
Generate {num_candidates} candidate pseudo labels that describe the same human action as the original label.

Requirements:
1. Each generated label must preserve the core semantics of the original action and refer to the same action category.
2. Each label should be a concise action phrase rather than a full sentence, explanation, or caption.
3. The generated labels should introduce lexical diversity through paraphrasing, synonym substitution, or natural rewording.
4. Do not simply copy the original label or make only trivial word-order changes.
5. Do not introduce new actions, irrelevant objects, scenes, intentions, or details that are not implied by the original label.
6. Ensure that all generated labels remain semantically close to the original label while differing in wording.

Original label: {original_label}

Output format:
1. <candidate_label_1>
2. <candidate_label_2>
...
{num_candidates}. <candidate_label_{num_candidates}>
"""
    return prompt.strip()


def run_chat(prompt):
    """
    Send a text-only prompt to MiniCPM-V.
    """
    msgs = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    inputs = {
        "question": json.dumps(msgs, ensure_ascii=False)
    }

    with torch.no_grad():
        if isinstance(chat_model, DataParallel):
            answer = chat_model.module.chat(inputs)
        else:
            answer = chat_model.chat(inputs)

    if not isinstance(answer, str):
        answer = str(answer)

    return answer.strip()


def process_text_file(text_path, output_dir, num_candidates=5):
    """
    Process one input label file:
    1. Read the original action label.
    2. Build the generation prompt.
    3. Query MiniCPM-V to generate candidate pseudo labels.
    4. Save the raw candidate list.
    """
    with open(text_path, 'r', encoding='utf-8') as f:
        original_label = f.read().strip()

    if len(original_label) == 0:
        print(f"[Warning] Empty file: {text_path}")
        return

    save_name = sanitize_label_for_filename(original_label)
    output_file = os.path.join(output_dir, f"{save_name}.txt")

    if os.path.exists(output_file):
        print(f"[Skip] Output already exists: {output_file}")
        return

    print(f"[Processing] label = {original_label}")
    print(f"[Output file] {output_file}")

    prompt = build_prompt(original_label, num_candidates)

    try:
        answer = run_chat(prompt)
    except Exception as e:
        print(f"[Error] Failed on label '{original_label}': {e}")
        return

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(answer + '\n')

    print(f"[Saved] {output_file}")


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    txt_files = sorted([
        f for f in os.listdir(INPUT_ROOT)
        if os.path.isfile(os.path.join(INPUT_ROOT, f)) and f.endswith('.txt')
    ])

    print(f"Found {len(txt_files)} txt files in {INPUT_ROOT}")

    count = 0
    for file_name in txt_files:
        text_path = os.path.join(INPUT_ROOT, file_name)
        process_text_file(text_path, OUTPUT_ROOT, NUM_CANDIDATES)
        count += 1
        print(f"[Progress] {count}/{len(txt_files)} completed\n")


if __name__ == "__main__":
    main()
