"""Route B: extract object crops from CLEVRER videos using de-rendered masks.

For a window of frames we, for each object, crop a patch around its mask bbox
center in every observation frame. Object identity is recovered by matching the
(de-rendered) (color, material, shape) tuple to the annotation's object_property
(unique per scene in CLEVRER). Future positions / collision labels come from the
annotation. Output mirrors `data.make_sample` (crops + traj + labels).
"""

import cv2
import json
import os

import numpy as np
from pycocotools import mask as cocomask


def load_ann(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_proposal(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def match_objects(ann):
    """返回 [(object_id, (color,material,shape)), ...]，按 object_id 排序。"""
    props = ann["object_property"]
    out = [(p["object_id"], (p["color"], p["material"], p["shape"])) for p in props]
    return sorted(out)


def _decode_frames(video_path, idxs):
    """读取指定帧号 -> {idx: HxWx3 RGB uint8}。"""
    idxs = sorted(idxs)
    want = set(idxs)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"cannot open {video_path}")
    out = {}
    # seek to first needed frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, idxs[0])
    i = idxs[0]
    while i <= idxs[-1]:
        ok, frame = cap.read()
        if not ok:
            break
        if i in want:
            out[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()
    return out


def _crop(img, cx, cy, size, out_size):
    H, W = img.shape[:2]
    size = int(max(8, min(size, min(H, W))))
    half = size // 2
    x0, y0 = int(round(cx)) - half, int(round(cy)) - half
    x1, y1 = x0 + size, y0 + size
    px0, py0 = max(0, -x0), max(0, -y0)
    px1, py1 = max(0, x1 - W), max(0, y1 - H)
    patch = np.zeros((size, size, 3), dtype=np.uint8)
    sx0, sy0 = x0 + px0, y0 + py0
    sx1, sy1 = x1 - px1, y1 - py1
    patch[py0:size - py1, px0:size - px1] = img[sy0:sy1, sx0:sx1]
    return cv2.resize(patch, (out_size, out_size), interpolation=cv2.INTER_LINEAR)


def _boxes_by_key(proposal, t):
    """{ (color,material,shape): bbox } for frame t (bbox = x,y,w,h)."""
    d = {}
    for o in proposal["frames"][t]["objects"]:
        k = (o["color"], o["material"], o["shape"])
        d[k] = cocomask.toBbox(o["mask"])
    return d


def build_sample(video_path, ann, proposal, t0, obs, pred, M, out_size=64,
                 expand=1.6, pos_stats=None, frames_cache=None):
    objs = match_objects(ann)[:M]
    T = obs + pred
    Sm = len(objs)

    # observation frame crops
    frames = list(range(t0, t0 + obs))
    if frames_cache is None:
        imgs = _decode_frames(video_path, frames)
    else:
        imgs = {t: frames_cache[t] for t in frames if t in frames_cache}
    crops = np.zeros((M, obs, 3, out_size, out_size), dtype=np.float32)

    # per-frame boxes; fallback to last seen box
    box_cache = {}   # key -> bbox
    for ti, t in enumerate(frames):
        boxes = _boxes_by_key(proposal, t)
        for j, (oid, key) in enumerate(objs):
            b = boxes.get(key, box_cache.get(key))
            if b is None:
                continue
            box_cache[key] = b
            cx, cy = b[0] + b[2] / 2.0, b[1] + b[3] / 2.0
            size = max(b[2], b[3]) * expand
            img = imgs.get(t)
            if img is None:
                continue
            patch = _crop(img, cx, cy, size, out_size).astype(np.float32) / 255.0
            crops[j, ti] = patch.transpose(2, 0, 1)   # HWC -> CHW

    # positions (obs+pred) from annotation
    traj = np.zeros((T, M, 2), dtype=np.float32)
    speed = np.zeros(M, dtype=np.float32)
    for j, (oid, key) in enumerate(objs):
        for t in range(T):
            fr = ann["motion_trajectory"][t0 + t]["objects"]
            o = next((x for x in fr if x["object_id"] == oid), None)
            if o is not None:
                traj[t, j] = o["location"][:2]
            if t == obs - 1 and o is not None:
                speed[j] = float(np.linalg.norm(np.asarray(o["velocity"][:2])))

    if pos_stats is not None:
        mean, std = pos_stats
        traj = (traj - mean) / std
    box = [1.0, 1.0]

    color_idx = np.array([i % 8 for i in range(M)], dtype=np.int64)
    props = {p["object_id"]: p for p in ann["object_property"]}
    for j, (oid, key) in enumerate(objs):
        p = props.get(oid)
        color_idx[j] = ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"].index(p["color"]) \
            if p and p["color"] in ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"] else 0

    target = int(np.argmax(speed)) if M > 0 else 0
    partner = (target + 1) % M

    def collides(lo, hi):
        for ev in ann.get("collision", []):
            if lo <= ev["frame_id"] < hi:
                return 1
        return 0
    label_yes = collides(t0 + obs, t0 + T)
    label_no = collides(t0, t0 + obs)

    return {
        "m": M,
        "crops": crops,             # (M, obs, 3, S, S)
        "objs": objs,
        "color_idx": color_idx,
        "direction_idx": 0,
        "target_idx": target,
        "partner_idx": partner,
        "label_yes": int(label_yes),
        "label_no": int(label_no),
        "label": int(label_yes),
        "traj_yes": traj,           # (T, M, 2) (normalized if pos_stats)
        "traj_no": traj,
        "box": box,
        "action_text": "",
        "question_text": "",
    }
