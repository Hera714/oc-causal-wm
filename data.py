"""Synthetic physical video generator for object-centric causal reasoning.

Design: deterministic causal scenes. M objects are placed on a coarse grid with
small random jitter and near-zero initial velocity. A *hypothetical action*
("push the <color> object to the <direction>") gives the target object an
impulse. Over the rollout window we ask a counterfactual question:

    "will the <color_target> object collide with the <color_partner> object?"

The answer (yes/no) depends on the action direction relative to the target's
geometry, so it genuinely requires modelling the *intervention*. Ground-truth
object boxes, colours, action condition and collision labels are all provided.
"""

from collections import OrderedDict

import numpy as np


N_COLOR = 8
N_DIR = 4
COLOR_NAMES = ["red", "blue", "green", "yellow", "purple", "cyan", "orange", "gray"]
DIR_NAMES = ["left", "right", "up", "down"]

_PALETTE = np.array([
    [230, 60, 60], [60, 120, 230], [60, 200, 90], [230, 190, 40],
    [160, 80, 220], [40, 200, 210], [240, 130, 60], [120, 130, 140],
], dtype=np.float32)


def _step(pos, vel, dt, box, radius):
    """物理仿真的一步：推进位置 + 处理边界反弹（碰墙反弹，速度反向）。

    pos : (m, 2) 每个物体当前中心坐标 (x,y)
    vel : (m, 2) 每个物体当前速度向量 (vx, vy)
    dt  : 时间步长（这里固定为1）
    box : [width, height] 仿真空间大小
    radius : (m,) 每个物体的碰撞半径
    """
    # 1) 用速度推进位置：new_pos = old_pos + vel * dt
    pos = pos + vel * dt
    # 2) 计算边界下限/上限。物体不能越过墙壁，至少要离墙一个半径远。
    #    low 是墙内侧最小值，hi_r 是墙内侧最大值。
    low = np.tile(radius[:, None], (1, pos.shape[1]))
    hi_r = np.stack([box[0] - radius, box[1] - radius], axis=-1)
    # 3) 判断哪些坐标越界（超出下界 ml / 上界 mh）
    ml = pos < low
    mh = pos > hi_r
    # 4) 越界就钳制回墙边，并让该方向速度反向（模拟弹性反弹）
    pos[ml] = low[ml]
    pos[mh] = hi_r[mh]
    vel[ml] = -vel[ml]
    vel[mh] = -vel[mh]
    return pos, vel


def _collide(pos, vel, radius):
    """处理物体两两碰撞（弹性碰撞，简化为等质量交换法向速度分量）。

    当两个物体圆心的距离小于两者半径之和时判定为碰撞；把沿连心线的
    速度分量按动量守恒交换（这里用0.5系数近似等质量），切向速度不变。
    """
    n = len(pos)
    for i in range(n):
        for j in range(i + 1, n):
            # 连心线向量 d = pos_j - pos_i
            d = pos[j] - pos[i]
            dist = np.linalg.norm(d)
            rs = radius[i] + radius[j]
            # 相交且距离非零才处理（防止除以0）
            if dist < rs and dist > 1e-6:
                dn = d / dist                       # 单位法向（连心线方向）
                vi = np.dot(vel[i], dn)             # i 沿法向的速度分量
                vj = np.dot(vel[j], dn)             # j 沿法向的速度分量
                rel = vi - vj                       # 相对法向速度
                # 仅当两者互相靠近（rel>0）才发生碰撞
                if rel > 0:
                    # 等质量弹性碰撞：交换法向分量的一半（动量守恒近似）
                    vel[i] = vel[i] - 0.5 * rel * dn
                    vel[j] = vel[j] + 0.5 * rel * dn
    return vel


def _has_collision(pair, traj, box, radius, t0, t1):
    """判断给定的一对物体在时间窗口 [t0, t1) 内是否发生过碰撞。

    pair : (a, b) 两个物体的索引
    traj : (T, m, 2) 各时间步各物体位置
    radius : (m,) 碰撞半径（用于判定两圆心距离是否小于半径和）
    t0, t1 : 需要检查的起止时间步
    返回 1（发生过碰撞）或 0（没有）。
    """
    a, b = pair
    for t in range(t0, t1):
        # 若某一帧两物体圆心距离 < 半径之和（加一点容差），认定碰撞
        if np.linalg.norm(traj[t, a] - traj[t, b]) < radius[a] + radius[b] + 0.02:
            return 1
    return 0


def _render_crop(pos, color, box, size=32, img_w=64, img_h=64):
    """把单个物体在某一时刻渲染成一个小图像块（crop）。

    把物体画成一个实心圆：圆心由物体在大场景中的位置映射到 crop 内的坐标，
    实心圆用给定颜色填充，其余像素为黑。这样图像输入就携带了物体颜色和位置。
    pos : (2,) 物体在场景中的中心坐标
    color : (3,) RGB 颜色
    size : crop 的边长（像素）
    img_w, img_h : 原场景的像素尺寸（用于坐标映射）
    """
    img = np.zeros((size, size, 3), dtype=np.float32)
    # 把场景坐标映射到 crop 左上角像素坐标（保证物体始终在 crop 内、不裁剪掉）
    cx = pos[0] / box[0] * (img_w - size)
    cy = pos[1] / box[1] * (img_h - size)
    r = size / 2 - 1
    # 生成 crop 内的像素网格，构造一个圆形掩码（圆心在 crop 中心）
    yy, xx = np.mgrid[0:size, 0:size]
    mask = (xx - size / 2) ** 2 + (yy - size / 2) ** 2 <= r * r
    # 圆形区域填色，其余保持黑色背景
    img[mask] = color
    return img


def make_sample(m=3, T=10, img_w=64, img_h=64, size=32, seed=0,
                dt=1.0, box=None, radius_scale=0.12, obs_frames=2,
                push_vel=3.0, label_target=None, pred_steps=6):
    """生成一条"因果反事实"数据样本（一条 3 物体视频 + 动作问题 + 答案）。

    核心逻辑：
      - 随机摆放 m 个物体（目标物体 target=0、伙伴 partner=1，其余远离）。
      - 一个"假设动作"："把<颜色>物体推向<方向>"，给 target 一个定向速度增量。
      - 分别仿真【不施动作】和【施动作】两条轨迹。
      - 判断 target 与 partner 是否在窗口内碰撞：不施动作 vs 施动作，
        若结果不同，则这是一个真正由"干预"改变结果的 counterfactual。
      - 输出观测帧的物体 crop、动作文本、问题文本、碰撞标签、未来位置等。

    label_target (0/1/None): 控制样本答案倾向。
      0 -> 希望答案是"不撞"；1 -> 希望答案是"撞"；None -> 自然随机。
      通过选择"朝向/背离伙伴"的推动方向来控制，从而平衡正负样本。
    """
    if box is None:
        box = [12.0, 12.0]          # 场景默认尺寸（宽、高）

    def bbox_r(m):
        # 每个物体的碰撞半径 = 场景宽度乘以一个比例，保证大小合适
        return np.array([box[0] * radius_scale] * m, dtype=np.float32)

    radius = bbox_r(m)
    rng = np.random.default_rng(seed + 1000)   # 用独立种子，避免和仿真种子冲突
    target = 0        # 被施加动作（干预）的物体
    partner = 1       # 问题里问的"是否会与之相撞"的物体

    # --- 随机布局：target/partner 放场景中部随机位置，其它物体放远处角落 ---
    def layout(m, rng):
        pos = np.zeros((m, 2))
        pos[0] = [box[0] * rng.uniform(0.3, 0.5),
                  box[1] * rng.uniform(0.3, 0.5)]
        pos[1] = [box[0] * rng.uniform(0.5, 0.75),
                  box[1] * rng.uniform(0.5, 0.75)]
        for i in range(2, m):
            pos[i] = [box[0] * (0.15 + 0.5 * (i - 2) / max(1, m - 2)),
                      box[1] * 0.15]
        return pos

    win_start = obs_frames            # 动作在 obs_frames 施加，其后即开始判定碰撞
    win_end = min(T, obs_frames + pred_steps)   # 与预测步长对齐，避免"标签超出预测视野"

    # 动作方向与几何**解耦**：随机打乱 4 个方向的顺序再挑选，而不是固定用
    # "朝/背伙伴"。这样模型无法只靠观测几何推断动作方向，必须真正读取动作输入。
    candidate_idx = list(rng.permutation(N_DIR))

    def simulate(direction):
        """给定动作方向，仿真 no-action / action 两条轨迹并返回碰撞标签。"""
        def sim(do_action):
            rng_l = np.random.default_rng(seed)
            pos = layout(m, rng_l)
            vel = rng_l.uniform([-0.1, -0.1], [0.1, 0.1], size=(m, 2))  # 近静止
            traj = np.zeros((T, m, 2))
            for t in range(T):
                # 在 obs_frames 那个时间步施加动作：给 target 一个定向速度增量
                if do_action and t == obs_frames:
                    dvec = {"left": [-1, 0], "right": [1, 0], "up": [0, 1],
                            "down": [0, -1]}[direction]
                    vel[target] = vel[target] + np.array(dvec, dtype=np.float32) * push_vel
                pos, vel = _step(pos, vel, dt, box, radius)
                vel = _collide(pos, vel, radius)
                traj[t] = pos.copy()
            return traj
        traj_no = sim(False)
        traj_yes = sim(True)
        c_no = _has_collision((target, partner), traj_no, box, radius, win_start, win_end)
        c_yes = _has_collision((target, partner), traj_yes, box, radius, win_start, win_end)
        return traj_no, traj_yes, c_no, c_yes

    # --- 在随机候选方向里，优先选"真反事实(动作改变结果)且符合目标标签"的 ---
    chosen = None          # 真反事实样本
    fallback = None        # 仅标签匹配（非反事实）的兜底
    for di in candidate_idx:
        direction = DIR_NAMES[int(di)]
        traj_no, traj_yes, c_no, c_yes = simulate(direction)
        if c_yes != c_no and (label_target is None or c_yes == label_target):
            chosen = (direction, traj_no, traj_yes, c_no, c_yes)
            break
        if fallback is None and (label_target is None or c_yes == label_target):
            fallback = (direction, traj_no, traj_yes, c_no, c_yes)
    if chosen is None:
        if fallback is not None:
            chosen = fallback
        else:
            direction = DIR_NAMES[int(candidate_idx[-1])]
            traj_no, traj_yes, c_no, c_yes = simulate(direction)
            chosen = (direction, traj_no, traj_yes, c_no, c_yes)
    direction, traj_no, traj_yes, collide_no, collide_yes = chosen

    # --- 生成观测帧的 crop：取"不施动作"轨迹的前 obs_frames 帧（共享可观测上下文） ---
    colors = np.array([_PALETTE[i % N_COLOR] for i in range(m)])
    crops = []
    for i in range(m):
        c = np.stack([_render_crop(traj_no[t, i], colors[i], box, size,
                                   img_w, img_h) for t in range(obs_frames)])
        crops.append(c)
    crops = np.array(crops)                     # (m, obs, size, size, 3)

    color_idx = np.array([i % N_COLOR for i in range(m)])   # 每个物体的颜色编号
    action_text = f"push the {COLOR_NAMES[target]} object to the {direction}"
    question_text = (f"will the {COLOR_NAMES[target]} object collide with "
                     f"the {COLOR_NAMES[partner]} object?")

    return {
        "m": m,
        "color_idx": color_idx,
        "crops": crops,
        "action_text": action_text,
        "question_text": question_text,
        "target_idx": target,
        "partner_idx": partner,
        "direction_idx": DIR_NAMES.index(direction),
        "label_no": int(collide_no),     # 不施动作时 target 与 partner 是否碰撞
        "label_yes": int(collide_yes),   # 施动作后是否碰撞（这是我们要预测的答案）
        "label": int(collide_yes),
        "colors": colors,
        "traj_no": traj_no,    # 不施动作的完整轨迹（用于提取观测帧/几何）
        "traj_yes": traj_yes,  # 施动作的完整轨迹（用于提取未来位置的标签 GT）
        "box": box,
    }


class Dataset:
    """确定性数据生成器，带有界 LRU 内存缓存。

    getitem 时按索引号交替请求 label_target（偶数→想要0，奇数→想要1），
    从而让数据集正负样本各占约一半（避免模型靠"总猜no"这种捷径作弊）。

    max_cache 限定了最多缓存多少条样本。之前用无限字典缓存在升级到
    m=4/size=48/n=4000 后会把 11GB 内存吃满，这里改为有界 LRU：超过容量就
    淘汰最久未用的样本。设为 0 表示完全不用缓存（每次重新生成）。
    """

    def __init__(self, n=1000, m=3, T=10, seed_offset=0, size=32, max_cache=2000,
                 pred_steps=6, radius_scale=0.10, push_vel=4.0):
        self.n, self.m, self.T = n, m, T
        self.seed_offset = seed_offset   # 用于区分 train/dev/test 的种子偏移
        self.size = size
        self.max_cache = max_cache
        self.pred_steps = pred_steps
        self.radius_scale = radius_scale
        self.push_vel = push_vel
        self._cache = OrderedDict()      # 有界缓存：key=idx, value=sample

    def __len__(self):
        # 数据集大小（迭代器按此长度生成样本）
        return self.n

    def __getitem__(self, idx):
        item = self._cache.get(idx)
        if item is not None:
            self._cache.move_to_end(idx)     # 命中则标记为最近使用
            return item
        lt = 0 if idx % 2 == 0 else 1        # 交替期望标签，实现正负均衡
        # 用固定种子 = seed_offset + idx*7+1，保证确定性可复现
        item = make_sample(m=self.m, T=self.T, size=self.size,
                           seed=self.seed_offset + idx * 7 + 1,
                           label_target=lt, pred_steps=self.pred_steps,
                           radius_scale=self.radius_scale,
                           push_vel=self.push_vel)
        if self.max_cache > 0:
            self._cache[idx] = item
            if len(self._cache) > self.max_cache:
                self._cache.popitem(last=False)   # 淘汰最久未用的
        return item


def make_dataset(split, n, m=3, size=32, max_cache=2000, pred_steps=6,
                 radius_scale=0.10, push_vel=4.0):
    offset = {"train": 0, "dev": 100000, "test": 200000}[split]
    return Dataset(n=n, m=m, size=size, seed_offset=offset, max_cache=max_cache,
                   pred_steps=pred_steps, radius_scale=radius_scale,
                   push_vel=push_vel)


if __name__ == "__main__":
    import numpy as np
    ds = make_dataset("train", 300)
    yes, no, flips = [], [], 0
    for i in range(len(ds)):
        s = ds[i]
        yes.append(s["label_yes"])
        no.append(s["label_no"])
        if s["label_yes"] != s["label_no"]:
            flips += 1
    yes = np.array(yes)
    print("label_yes=1:", round(yes.mean(), 3),
          "label_no=1:", round(np.mean(no), 3),
          "flip:", round(flips / len(ds), 3))
    s = ds[0]
    print("action:", s["action_text"])
    print("question:", s["question_text"])
    print("crops:", s["crops"].shape)
