"""Build CLEVRER VQA inputs for the benchmark (start with predictive).

For each selected question we sample N frames from the video, lay them out as a
montage image (left-to-right, top-to-bottom = chronological), and write a jsonl
with the question, lettered choices, and ground-truth letter. Output keys mirror
the CausalSpatial jsonl so the same scorer/pipeline can be reused.

Questions are grouped by video so each video is decoded at most once.
"""

import argparse
import json
import os
import glob

import cv2
import numpy as np


def video_index(root):
    idx = {}
    for p in glob.glob(os.path.join(root, "**", "video_*.mp4"), recursive=True):
        b = os.path.basename(p)
        try:
            idx[int(b.replace("video_", "").replace(".mp4", ""))] = p
        except ValueError:
            pass
    return idx


def sample_frames(path, idxs):
    idxs = sorted(set(idxs))
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {}
    want = set(idxs)
    out, i = {}, idxs[0]
    cap.set(cv2.CAP_PROP_POS_FRAMES, idxs[0])
    while i <= idxs[-1]:
        ok, f = cap.read()
        if not ok:
            break
        if i in want:
            out[i] = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()
    return out


def montage(frames, grid, cell=(240, 160)):
    rows, cols = grid
    W, H = cell[0] * cols, cell[1] * rows
    canvas = np.zeros((H, W, 3), np.uint8)
    for k, f in enumerate(frames):
        r, c = divmod(k, cols)
        if r >= rows:
            break
        canvas[r*cell[1]:(r+1)*cell[1], c*cell[0]:(c+1)*cell[0]] = cv2.resize(f, cell)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="~/datasets/clevrer")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--qtypes", default="predictive")
    ap.add_argument("--n_frames", type=int, default=16)
    ap.add_argument("--grid", default="4x4")
    ap.add_argument("--out", default="out/clevrer_vqa_predictive.jsonl")
    ap.add_argument("--frames_dir", default="out/vqa_frames")
    ap.add_argument("--max_videos", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    root = os.path.expanduser(args.root)
    vids = video_index(root)
    qf = os.path.join(root, "questions", f"{args.split}.json")
    scenes = json.load(open(qf))
    qtypes = set(args.qtypes.split(","))
    grid = tuple(int(x) for x in args.grid.split("x"))
    nf = args.n_frames
    total = 128
    sample_idx = np.linspace(0, total - 1, nf).round().astype(int).tolist()

    os.makedirs(args.frames_dir, exist_ok=True)
    letters = "ABCDEFGH"

    # group questions by video index
    by_video = {}
    for sc in scenes:
        vfile = sc["video_filename"]
        try:
            vi = int(vfile.replace("video_", "").replace(".mp4", ""))
        except ValueError:
            continue
        qs = [q for q in sc["questions"] if q["question_type"] in qtypes]
        if qs:
            by_video.setdefault(vi, []).append((vfile, qs))

    vlist = sorted(by_video.items())
    if args.max_videos:
        vlist = vlist[:args.max_videos]

    n_written = 0
    with open(args.out, "w", encoding="utf-8") as fo:
        for vi, items in vlist:
            vpath = vids.get(vi)
            if vpath is None:
                continue
            frames = sample_frames(vpath, sample_idx)
            if not frames:
                continue
            ordered = [frames[i] for i in sample_idx if i in frames]
            img = montage(ordered, grid)
            for vfile, qs in items:
                for q in qs:
                    choices = q["choices"]
                    gt_idx = next((k for k, c in enumerate(choices)
                                   if c["answer"] == "correct"), 0)
                    opt_txt = "\n".join(f"({letters[k]}) {c['choice']}"
                                        for k, c in enumerate(choices))
                    question = (f"The image shows {nf} frames sampled in chronological "
                                f"order from a video, arranged left-to-right then "
                                f"top-to-bottom.\nQuestion: {q['question']}\n{opt_txt}\n"
                                f"Answer with the option letter only.")
                    img_name = f"{vi:05d}_q{q['question_id']}.jpg"
                    cv2.imwrite(os.path.join(args.frames_dir, img_name),
                                cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    fo.write(json.dumps({
                        "question_id": f"clevrer_{vi}_{q['question_id']}",
                        "question": question,
                        "choices": [c["choice"] for c in choices],
                        "gt_answer": letters[gt_idx],
                        "model_answer": "",
                        "not_sure": "",
                        "type": q["question_type"],
                        "difficulty": q.get("question_subtype") or "",
                        "video": vfile,
                        "frame_image": os.path.join(args.frames_dir, img_name),
                    }, ensure_ascii=False) + "\n")
                    n_written += 1
            if n_written % 500 == 0:
                print(f"  {n_written} questions written ...", flush=True)
    print(f"wrote {n_written} questions -> {args.out}")


if __name__ == "__main__":
    main()
