"""CLEVRER trajectory-level dataset for the delta-dynamics gate (route A).

Route A uses only the CLEVRER *annotations* (no videos): each annotation has, per
frame, every object's location / velocity, plus collision events. We turn a window
of frames into the same tensor interface used by the synthetic `data.py`, so the
existing `models.py` / `run.py` machinery can be reused with a "state" encoder.

State feature vector per (frame, object), F = STATE_DIM = 18:
    [x, y, vx, vy, color_onehot(8), material_onehot(2), shape_onehot(3), in_view(1)]
positions/velocities are standardized with dataset-wide mean/std.

Sample dict mirrors `data.make_sample`:
    m, enc_feats (obs, M, F), color_idx (M,), direction_idx, target_idx,
    partner_idx, label_yes, label_no, traj_yes (T, M, 2), box, ...
so that `run.collate` works unchanged.
"""

import glob
import json
import os
from collections import OrderedDict

import numpy as np


STATE_DIM = 18
COLORS = ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"]
MATERIALS = ["rubber", "metal"]
SHAPES = ["cube", "sphere", "cylinder"]
N_FRAMES = 128


def _onehot(idx, n):
    v = np.zeros(n, dtype=np.float32)
    if 0 <= idx < n:
        v[idx] = 1.0
    return v


class ClevrerDataset:
    def __init__(self, root, obs=4, pred=8, M=5, stride=8, max_files=None,
                 cache_files=200, stats_files=80, seed=0, files=None,
                 max_windows=None):
        self.root = root
        self.obs, self.pred, self.M = obs, pred, M
        self.T = obs + pred
        if files is not None:
            self.files = sorted(files)
        else:
            self.files = sorted(glob.glob(os.path.join(root, "**", "annotation_*.json"),
                                          recursive=True))
        if max_files:
            self.files = self.files[:max_files]
        if not self.files:
            raise FileNotFoundError(f"no annotation_*.json under {root}")
        # window index (assume every video has N_FRAMES frames)
        self.windows = []
        for fi in range(len(self.files)):
            for t0 in range(0, N_FRAMES - self.T + 1, stride):
                self.windows.append((fi, t0))
        if max_windows and len(self.windows) > max_windows:
            rng = np.random.default_rng(seed)
            sel = rng.choice(len(self.windows), size=max_windows, replace=False)
            self.windows = [self.windows[i] for i in sorted(sel)]
        self._cache = OrderedDict()
        self.cache_files = cache_files
        self.pos_mean, self.pos_std, self.vel_mean, self.vel_std = \
            self._compute_stats(stats_files, seed)

    # ---------- io ----------
    def _load(self, fi):
        fp = self.files[fi]
        if fp in self._cache:
            self._cache.move_to_end(fp)
            return self._cache[fp]
        with open(fp, encoding="utf-8") as f:
            ann = json.load(f)
        self._cache[fp] = ann
        if len(self._cache) > self.cache_files:
            self._cache.popitem(last=False)
        return ann

    def _compute_stats(self, n_files, seed):
        rng = np.random.default_rng(seed)
        idxs = rng.choice(len(self.files), size=min(n_files, len(self.files)),
                          replace=False)
        pos, vel = [], []
        for fi in idxs:
            ann = self._load(int(fi))
            for fr in ann["motion_trajectory"]:
                for o in fr["objects"]:
                    pos.append(o["location"][:2])
                    vel.append(o["velocity"][:2])
        pos = np.asarray(pos, dtype=np.float64)
        vel = np.asarray(vel, dtype=np.float64)
        return (pos.mean(0).astype(np.float32), (pos.std(0) + 1e-6).astype(np.float32),
                vel.mean(0).astype(np.float32), (vel.std(0) + 1e-6).astype(np.float32))

    # ---------- dataset api ----------
    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        if getattr(self, "_mat", False):
            return self._pack(idx)
        fi, t0 = self.windows[idx]
        ann = self._load(fi)
        prop = {p["object_id"]: p for p in ann["object_property"]}
        frames = ann["motion_trajectory"][t0:t0 + self.T]

        # union of object ids across the window
        ids = []
        for fr in frames:
            for o in fr["objects"]:
                oid = o["object_id"]
                if oid not in ids:
                    ids.append(oid)
        ids = ids[:self.M]
        M = self.M

        enc = np.zeros((self.obs, M, STATE_DIM), dtype=np.float32)
        traj = np.zeros((self.T, M, 2), dtype=np.float32)
        # last-seen velocity magnitude for target selection
        speed = np.zeros(M, dtype=np.float32)

        for j, oid in enumerate(ids):
            p = prop.get(oid, {"color": "gray", "material": "rubber", "shape": "cube"})
            c = _onehot(COLORS.index(p["color"]) if p["color"] in COLORS else 0, 8)
            mat = _onehot(MATERIALS.index(p["material"]) if p["material"] in MATERIALS else 0, 2)
            shp = _onehot(SHAPES.index(p["shape"]) if p["shape"] in SHAPES else 0, 3)
            for t in range(self.T):
                o = next((x for x in frames[t]["objects"] if x["object_id"] == oid), None)
                if o is None:
                    continue
                loc = np.asarray(o["location"][:2], dtype=np.float32)
                vel = np.asarray(o["velocity"][:2], dtype=np.float32)
                pn = (loc - self.pos_mean) / self.pos_std
                vn = (vel - self.vel_mean) / self.vel_std
                view = 1.0 if o.get("inside_camera_view", False) else 0.0
                traj[t, j] = loc
                if t < self.obs:
                    enc[t, j, :2] = pn
                    enc[t, j, 2:4] = vn
                    enc[t, j, 4:12] = c
                    enc[t, j, 12:14] = mat
                    enc[t, j, 14:17] = shp
                    enc[t, j, 17] = view
                if t == self.obs - 1:
                    speed[j] = float(np.linalg.norm(vel))

        # normalized position labels (same normalization as state pos)
        traj_n = (traj - self.pos_mean) / self.pos_std
        color_idx = np.array([i % 8 for i in range(M)], dtype=np.int64)
        for j, oid in enumerate(ids):
            p = prop.get(oid)
            color_idx[j] = COLORS.index(p["color"]) if p and p["color"] in COLORS else 0

        target = int(np.argmax(speed)) if M > 0 else 0
        partner = (target + 1) % M

        # collision label: any collision event inside future window / obs window
        def collides(t_lo, t_hi):
            for ev in ann.get("collision", []):
                if t_lo <= ev["frame_id"] < t_hi:
                    return 1
            return 0
        label_yes = collides(t0 + self.obs, t0 + self.T)
        label_no = collides(t0, t0 + self.obs)

        return {
            "m": M,
            "enc_feats": enc,                 # (obs, M, F)
            "color_idx": color_idx,
            "direction_idx": 0,
            "target_idx": target,
            "partner_idx": partner,
            "label_yes": int(label_yes),
            "label_no": int(label_no),
            "label": int(label_yes),
            "traj_yes": traj_n,               # (T, M, 2) normalized positions
            "traj_no": traj_n,
            "box": [1.0, 1.0],
            "action_text": "",
            "question_text": "",
        }

    # ---------- materialize all windows into RAM (avoid re-parsing json) ----------
    def materialize(self):
        if getattr(self, "_mat", False):
            return self
        N = len(self.windows)
        enc = np.zeros((N, self.obs, self.M, STATE_DIM), dtype=np.float32)
        traj = np.zeros((N, self.T, self.M, 2), dtype=np.float32)
        color = np.zeros((N, self.M), dtype=np.int64)
        target = np.zeros(N, dtype=np.int64)
        lyes = np.zeros(N, dtype=np.int64)
        lno = np.zeros(N, dtype=np.int64)
        for i in range(N):
            s = self[i]
            enc[i] = s["enc_feats"]
            traj[i] = s["traj_yes"]
            color[i] = s["color_idx"]
            target[i] = s["target_idx"]
            lyes[i] = s["label_yes"]
            lno[i] = s["label_no"]
        self._enc, self._traj, self._color = enc, traj, color
        self._target, self._lyes, self._lno = target, lyes, lno
        self._mat = True
        self._cache.clear()
        return self

    def _pack(self, i):
        M = self.M
        partner = (int(self._target[i]) + 1) % M
        return {
            "m": M,
            "enc_feats": self._enc[i],
            "color_idx": self._color[i],
            "direction_idx": 0,
            "target_idx": int(self._target[i]),
            "partner_idx": partner,
            "label_yes": int(self._lyes[i]),
            "label_no": int(self._lno[i]),
            "label": int(self._lyes[i]),
            "traj_yes": self._traj[i],
            "traj_no": self._traj[i],
            "box": [1.0, 1.0],
            "action_text": "",
            "question_text": "",
        }
