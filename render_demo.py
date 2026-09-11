"""生成一条样本并渲染成 avi 动图，保存到 exp/data 文件夹。

场景：3 个彩色圆球。红色物体（target=0）受到"假设动作"推进，
每帧画出物体的位置与运动轨迹；target 与 partner 会用红/黑框标出。

用法：
    python render_demo.py [--seed N]
输出：
    exp/data/sample_seed<N>.avi
    并打印动作文本、问题、以及"施动作后是否碰撞"的答案（在视频画面上也显示）。
"""

import argparse
import os
import cv2
import numpy as np

from data import make_sample


def render_sample(sample, out_path, img_w=640, img_h=640):
    """把样本轨迹渲染为一共有 T 帧的动图（mp4）。"""
    box = sample["box"]
    T = sample["traj_yes"].shape[0]
    m = sample["m"]
    colors = sample["colors"]
    r = int(img_w * 0.05)            # 物体像素半径

    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"MJPG"), 12, (img_w, img_h))

    def to_px(pos):
        # 把场景坐标 [0,box] 映射到像素坐标 [0,img]
        return int(pos[0] / box[0] * img_w), int(pos[1] / box[1] * img_h)

    for t in range(T):
        frame = np.full((img_h, img_w, 3), 255, dtype=np.uint8)

        # 画"施动作后"各物体最近几帧的轨迹线
        for i in range(m):
            pts = [to_px(sample["traj_yes"][tt, i]) for tt in range(max(0, t - 6), t)]
            if len(pts) >= 2:
                cv2.polylines(frame, [np.asarray(pts, dtype=np.int32)], False,
                              tuple(int(v) for v in colors[i]), 2)

        # 画当前帧的物体（实心圆）
        for i in range(m):
            cx, cy = to_px(sample["traj_yes"][t, i])
            cv2.circle(frame, (int(cx), int(cy)), r,
                       tuple(int(v) for v in colors[i]), -1)
            cv2.putText(frame, str(i), (int(cx) - 6, int(cy) + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # 顶部信息：target/partner 编号
        cv2.putText(frame, f"TARGET={sample['target_idx']} (red)",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f"PARTNER={sample['partner_idx']}",
                    (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # 底部信息：动作文本 + 真值答案
        cv2.putText(frame, f"q: {sample['question_text']} -> ",
                    (10, img_h - 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (100, 100, 100), 1)
        ans = "YES" if sample["label_yes"] else "NO"
        cv2.putText(frame, f"answer(label_yes) = {ans}  |  without action = "
                           f"{'YES' if sample['label_no'] else 'NO'}",
                    (10, img_h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        writer.write(frame)

    writer.release()
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    sample = make_sample(m=3, T=10, seed=args.seed)
    os.makedirs("data", exist_ok=True)
    out = os.path.join("data", f"sample_seed{args.seed}.avi")
    render_sample(sample, out)
    print("saved:", out)
    print("action  :", sample["action_text"])
    print("question:", sample["question_text"])
    print("label_yes:", sample["label_yes"], "| label_no:", sample["label_no"])


if __name__ == "__main__":
    main()
