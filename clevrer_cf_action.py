"""CLEVRER counterfactual -> action-conditioned causal QA (action = "remove object X").

CLEVRER counterfactual questions are of the form
  "Which of the following will not happen if the yellow cube is removed?"
  "Without the sphere, which of the following will happen?"
The intervention (removed object) is in the question text / program. We extract it
and emit a unified QA jsonl (keys aligned with CausalSpatial's score.py).

Output fields per line:
  question_id, video, question, action_type="remove", removed_object,
  negate(bool), choices[list], gt_letters[list], gt_answer(first correct letter),
  model_answer="", not_sure="", type="counterfactual", difficulty=""

Usage:
  python clevrer_cf_action.py --root ~/datasets/clevrer --split validation \
      --out out/clevrer_cf_qa.jsonl
"""

import argparse
import collections
import json
import os
import re

LETTERS = "ABCDEFGH"


def extract_removed(q):
    pats = [
        r"if the (.+?) is removed",
        r"without the (.+?)[,?]",
        r"without (?:the )?(.+?)[,?]",
        r"if (.+?) is removed",
        r"without (.+?)[,?]",
        r"the (.+?) is removed",
    ]
    for p in pats:
        m = re.search(p, q, flags=re.I)
        if m:
            return m.group(1).strip()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="~/datasets/clevrer")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--out", default="out/clevrer_cf_qa.jsonl")
    args = ap.parse_args()

    root = os.path.expanduser(args.root)
    qf = os.path.join(root, "questions", f"{args.split}.json")
    scenes = json.load(open(qf, encoding="utf-8"))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n = 0
    n_extracted = 0
    neg = 0
    ncorrect_dist = collections.Counter()
    removed_tokens = collections.Counter()
    with open(args.out, "w", encoding="utf-8") as fo:
        for sc in scenes:
            vfile = sc["video_filename"]
            for q in sc["questions"]:
                if q["question_type"] != "counterfactual":
                    continue
                choices = q["choices"]
                gt_letters = [LETTERS[k] for k, c in enumerate(choices)
                              if c["answer"] == "correct"]
                if not gt_letters:
                    continue
                question = q["question"]
                removed = extract_removed(question)
                if removed:
                    n_extracted += 1
                    removed_tokens[removed.split()[0] if removed else "?"] += 1
                is_neg = bool(re.search(r"\bnot\b|won't|will not", question, re.I))
                neg += int(is_neg)
                ncorrect_dist[len(gt_letters)] += 1
                opt_txt = "\n".join(f"({LETTERS[k]}) {c['choice']}"
                                    for k, c in enumerate(choices))
                ordered_q = f"Question: {question}\n{opt_txt}\nAnswer with the option letter(s)."
                fo.write(json.dumps({
                    "question_id": f"clevrer_cf_{sc['scene_index']}_{q['question_id']}",
                    "video": vfile,
                    "question": ordered_q,
                    "action_type": "remove",
                    "removed_object": removed,
                    "negate": is_neg,
                    "choices": [c["choice"] for c in choices],
                    "gt_letters": gt_letters,
                    "gt_answer": gt_letters[0],          # 简化评分用（完整为 gt_letters）
                    "model_answer": "",
                    "not_sure": "",
                    "type": "counterfactual",
                    "difficulty": "",
                }, ensure_ascii=False) + "\n")
                n += 1
    print(f"counterfactual QA written: {n} -> {args.out}")
    print(f"  removed object extracted: {n_extracted}/{n}")
    print(f"  negation ('not happen') questions: {neg}")
    print(f"  #correct distribution: {dict(ncorrect_dist)}")
    print(f"  top removed-object head words: {removed_tokens.most_common(8)}")


if __name__ == "__main__":
    main()
