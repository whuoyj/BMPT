# -*- coding: utf-8 -*-
import os
import re
import torch
import clip

# =========================
# 1. Path settings
# =========================
INPUT_ROOT = '../Yourdir/input_texts/'
OUTPUT_ROOT = '../Yourdir/output_texts/'

# =========================
# 2. CLIP model settings
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL_NAME = "ViT-B/32"   # Change this to match the CLIP backbone used in your project.
clip_model, _ = clip.load(CLIP_MODEL_NAME, device=DEVICE)
clip_model.eval()

# =========================
# 3. Scoring hyperparameters
# =========================
LAMBDA = 1.0          # Weight for semantic similarity
ALPHA = 0.3           # Weight for redundancy penalty

DELTA_LOW = 0.60      # Lower similarity threshold
DELTA_HIGH = 0.95     # Upper similarity threshold

# If no candidate passes threshold filtering, fall back to the highest-scoring candidate.
FALLBACK_TO_BEST = True


def sanitize_label_for_filename(label):
    """
    Keep this rule consistent with pseudo-label candidate generation.

    Example:
        'opening door' -> 'openingdoor'
    """
    label = label.strip().lower()
    label = label.replace(" ", "")
    label = re.sub(r'[\\/:*?"<>|]', '', label)
    return label


def normalize_text(text):
    """
    Normalize text to lowercase and remove extra whitespace.
    """
    text = text.strip().lower()
    text = re.sub(r'\s+', ' ', text)
    return text


def tokenize_for_overlap(text):
    """
    Tokenize text for the redundancy calculation.
    """
    text = normalize_text(text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = text.split()
    return tokens


def compute_redundancy(label_a, label_b):
    """
    Compute lexical redundancy with Jaccard overlap.

    Redundancy = |A intersection B| / |A union B|.
    A larger value indicates stronger lexical overlap.
    """
    tokens_a = set(tokenize_for_overlap(label_a))
    tokens_b = set(tokenize_for_overlap(label_b))

    if len(tokens_a) == 0 and len(tokens_b) == 0:
        return 0.0
    if len(tokens_a | tokens_b) == 0:
        return 0.0

    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


@torch.no_grad()
def compute_clip_similarity(text_a, text_b):
    """
    Compute semantic similarity with the CLIP text encoder.
    """
    texts = [text_a, text_b]
    tokens = clip.tokenize(texts).to(DEVICE)

    text_features = clip_model.encode_text(tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    sim = (text_features[0] @ text_features[1].T).item()
    return float(sim)


def parse_candidates_from_file(candidate_file):
    """
    Parse candidate pseudo labels from a MiniCPM output file.

    Supported format:
        1. opening a door
        2. pushing open a door
        ...

    Lines without numbering are also retained when possible.
    """
    with open(candidate_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    candidates = []
    for line in lines:
        m = re.match(r'^\s*\d+\s*[\.\)\u3001]\s*(.+)$', line)
        if m:
            cand = m.group(1).strip()
        else:
            cand = line.strip()

        cand = normalize_text(cand)
        if len(cand) > 0:
            candidates.append(cand)

    unique_candidates = []
    seen = set()
    for c in candidates:
        if c not in seen:
            unique_candidates.append(c)
            seen.add(c)

    return unique_candidates


def score_candidates(original_label, candidates):
    """
    Compute similarity, redundancy, and final score for all candidates.

    Returns:
        A list of dictionaries.
    """
    results = []

    for cand in candidates:
        sim = compute_clip_similarity(original_label, cand)
        red = compute_redundancy(original_label, cand)
        score = LAMBDA * sim - ALPHA * red

        results.append({
            "candidate": cand,
            "sim": sim,
            "redundancy": red,
            "score": score
        })

    return results


def select_best_candidate(results):
    """
    Filter candidates with DELTA_LOW and DELTA_HIGH, then select the highest score.
    """
    valid_results = [
        r for r in results
        if DELTA_LOW <= r["sim"] <= DELTA_HIGH
    ]

    if len(valid_results) > 0:
        best = max(valid_results, key=lambda x: x["score"])
        return best, valid_results, True

    if FALLBACK_TO_BEST and len(results) > 0:
        best = max(results, key=lambda x: x["score"])
        return best, [], False

    return None, [], False


def save_score_report(score_file, original_label, results, best_result, used_threshold):
    """
    Save the scoring report for debugging and reproducibility.
    """
    with open(score_file, 'w', encoding='utf-8') as f:
        f.write(f"Original label: {original_label}\n")
        f.write(f"LAMBDA = {LAMBDA}, ALPHA = {ALPHA}\n")
        f.write(f"DELTA_LOW = {DELTA_LOW}, DELTA_HIGH = {DELTA_HIGH}\n")
        f.write(f"Threshold filtering used: {used_threshold}\n")
        f.write("=" * 80 + "\n\n")

        sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)

        for idx, r in enumerate(sorted_results, 1):
            f.write(f"[{idx}] {r['candidate']}\n")
            f.write(f"    Sim        : {r['sim']:.6f}\n")
            f.write(f"    Redundancy : {r['redundancy']:.6f}\n")
            f.write(f"    Score      : {r['score']:.6f}\n")
            if best_result is not None and r["candidate"] == best_result["candidate"]:
                f.write(f"    <-- SELECTED\n")
            f.write("\n")


def process_one_label(input_txt_path):
    """
    Process one class label:
    - Read the original label from input_texts.
    - Read candidate pseudo labels from output_texts/<label>.txt.
    - Select the final pseudo label.
    - Save it to output_texts/<label>_final.txt.
    """
    with open(input_txt_path, 'r', encoding='utf-8') as f:
        original_label = f.read().strip()

    original_label = normalize_text(original_label)

    if len(original_label) == 0:
        print(f"[Warning] Empty original label file: {input_txt_path}")
        return

    save_name = sanitize_label_for_filename(original_label)
    candidate_file = os.path.join(OUTPUT_ROOT, f"{save_name}.txt")

    if not os.path.exists(candidate_file):
        print(f"[Warning] Candidate file not found: {candidate_file}")
        return

    final_file = os.path.join(OUTPUT_ROOT, f"{save_name}_final.txt")
    score_file = os.path.join(OUTPUT_ROOT, f"{save_name}_scores.txt")

    if os.path.exists(final_file):
        print(f"[Skip] Final pseudo label already exists: {final_file}")
        return

    candidates = parse_candidates_from_file(candidate_file)
    if len(candidates) == 0:
        print(f"[Warning] No valid candidates parsed from: {candidate_file}")
        return

    print(f"\n[Processing] {original_label}")
    print(f"  Candidate file: {candidate_file}")
    print(f"  Num candidates: {len(candidates)}")

    results = score_candidates(original_label, candidates)
    best_result, valid_results, used_threshold = select_best_candidate(results)

    if best_result is None:
        print(f"[Warning] No candidate selected for label: {original_label}")
        return

    with open(final_file, 'w', encoding='utf-8') as f:
        f.write(best_result["candidate"] + '\n')

    save_score_report(score_file, original_label, results, best_result, used_threshold)

    print(f"  [Selected] {best_result['candidate']}")
    print(f"  [Sim] {best_result['sim']:.4f}, [Red] {best_result['redundancy']:.4f}, [Score] {best_result['score']:.4f}")
    print(f"  [Saved final] {final_file}")
    print(f"  [Saved score] {score_file}")


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    txt_files = sorted([
        f for f in os.listdir(INPUT_ROOT)
        if os.path.isfile(os.path.join(INPUT_ROOT, f)) and f.endswith('.txt')
    ])

    print(f"Found {len(txt_files)} original label files in {INPUT_ROOT}")

    for idx, file_name in enumerate(txt_files, 1):
        input_txt_path = os.path.join(INPUT_ROOT, file_name)
        process_one_label(input_txt_path)
        print(f"[Progress] {idx}/{len(txt_files)} done\n")


if __name__ == "__main__":
    main()
