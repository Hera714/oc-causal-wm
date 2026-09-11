"""Train + evaluate the A/B action-conditioning fork.

Compares:
  joint  (proposed): action injected into the block-causal predictor.
  separate (baseline): action only appended at the answer head.

Metrics on held-out counterfactual collision questions:
  - classification accuracy (yes/no)
  - counterfactual position MSE
  - counterfactual accuracy (only samples where action flips the label): the
    task that truly requires *causal* understanding.

Run small (default, CPU / 2GB) or larger on a remote GPU via flags.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn

from data import make_dataset
from models import Model


# 动作方向翻转映射：left<->right, up<->down（用于干预一致性实验）
FLIP_DIR = {0: 1, 1: 0, 2: 3, 3: 2}
# 动作方向的单位向量（与 data.py 的 DIR_NAMES/仿真约定一致）
DIRVEC = {0: (-1.0, 0.0), 1: (1.0, 0.0), 2: (0.0, 1.0), 3: (0.0, -1.0)}


def collate(batch, pred=4, obs=2):
    """把一批 dict 样本整理成模型直接能用的张量（batched tensor）。

    具体地：从每个样本里读取 crop、颜色编号、方向编号、目标物体编号、
    观测位置、未来位置、答案标签、是否"动作真的改变结果"(flip) 等，
    并用 torch.stack 堆成 (Batch, ...) 形式。
    """
    m = batch[0]["m"]
    if "enc_feats" in batch[0]:
        # 预计算的 DINOv2 特征：(B, obs, m, feat)，模型按 4D 输入处理
        obs = batch[0]["enc_feats"].shape[0]
        crops = torch.stack([torch.from_numpy(b["enc_feats"]) for b in batch]).float()
    else:
        obs = batch[0]["crops"].shape[1]
        # 观测裁剪 patch -> 张量，并把通道维挪到正确位置 (B, m, obs, C, H, W)
        crops = torch.stack([torch.from_numpy(b["crops"]) for b in batch]).float()
        crops = crops.permute(0, 1, 2, 5, 3, 4)  # (B, m, obs, C, H, W)
    # 每个物体的颜色编号 (B, m)
    color_idx = torch.tensor(np.stack([b["color_idx"] for b in batch]),
                             dtype=torch.long)
    # 每个样本的方向编号 (B,)
    dir_idx = torch.tensor([b["direction_idx"] for b in batch], dtype=torch.long)
    # 每个样本被干预的目标物体编号 (B,)
    target_idx = torch.tensor([b["target_idx"] for b in batch], dtype=torch.long)
    # 观测位置：逐帧取 t=0..obs-1（观测阶段动作尚未施加，traj_no 与 traj_yes 相同），
    # 归一化到 [0,1]，形状 (B, obs, m, 2)，供逐帧位置嵌入使用。
    obs_posn = torch.from_numpy(np.array([
        np.stack([b["traj_yes"][t, i] / b["box"][0]
                  for t in range(obs) for i in range(m)]).reshape(obs, m, 2)
        for b in batch])).float()
    # 未来位置标签：取动作后轨迹的 t=obs..obs+pred-1 共 pred 步
    future_posn = torch.from_numpy(np.array(
        [np.stack([b["traj_yes"][t, i] / b["box"][0]
                   for t in range(obs, obs + pred) for i in range(m)]
                  ).reshape(pred, m, 2) for b in batch]
    )).float()
    # 答案：施动作后 target 与 partner 是否相撞（我们要预测的标签）
    label = torch.tensor([b["label_yes"] for b in batch], dtype=torch.long)
    # flip：该样本动作是否真的改变了"是否碰撞"的结果（真反事实样本标记）
    flip = torch.tensor([int(b["label_yes"] != b["label_no"]) for b in batch])
    return crops, color_idx, dir_idx, target_idx, obs_posn, future_posn, label, flip


def iter_batches(ds, batch_size, shuffle, seed=0, pred=4, obs=2):
    """按 batch 迭代数据集。每次把当前 batch 的索引列表交给 collate 转成张量。

    shuffle 时用固定 seed 打乱，保证可复现；seed 变化可让每个 epoch 采到不同顺序。
    """
    idxs = np.arange(len(ds))
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idxs)
    for i in range(0, len(idxs), batch_size):
        yield collate([ds[int(j)] for j in idxs[i:i + batch_size]], pred=pred, obs=obs)


def evaluate(model, ds, device, batch_size, pred=4, obs=2, consistency=True):
    """在数据集 ds 上评估模型，返回多种指标。

    指标：
      - acc        : 二分类 yes/no 准确率
      - flip_acc   : 仅在"动作真实改变结果"的真反事实样本上的准确率（更考验因果）
      - future_mse : 全物体未来位置预测误差（越低越好）
      - target_mse : 被干预物体(target)的未来位置预测误差（越低越好）
      - delta_mag  : 模型预测的变化量 mean|Delta|（仅 delta_mode 非零），监控退化
      - cons       : 干预一致性：把动作方向翻转(left<->right, up<->down)后，
                     被干预物体沿动作轴的预测位移符号是否随之翻转（越高越好）
      - cons_strict: 上面仅在 |位移| 足够大的样本上的比例（过滤噪声）
    """
    model.eval()
    P = pred   # 保存预测步长（循环内 pred 会被分类结果覆盖）
    correct = 0       # 分类正确数
    total = 0         # 分类总数
    flip_correct = 0  # flip 子集上的正确数
    flip_total = 0
    all_mse = 0.0     # 全物体未来位置 MSE 累计（加权 batch 大小）
    targ_mse = 0.0    # target 物体未来位置 MSE 累计
    delta_mag_sum = 0.0
    n_batches = 0
    cons_correct = 0
    cons_total = 0
    cons_correct_s = 0
    cons_total_s = 0
    align_sum = 0.0
    align_flip_sum = 0.0
    follow_sum = 0.0
    follow_n = 0
    all_mse_step = torch.zeros(P)
    targ_mse_step = torch.zeros(P)
    nontarg_mse_step = torch.zeros(P)
    delta_mag_step = torch.zeros(P)
    with torch.no_grad():     # 评估阶段不需要梯度，省显存、更快
        for crops, color_idx, dir_idx, target_idx, obs_posn, future_posn, label, flip in iter_batches(ds, batch_size, shuffle=False, pred=pred, obs=obs):
            # 把所有输入搬到模型所在设备（cuda/cpu）
            crops = crops.to(device)
            color_idx = color_idx.to(device)
            dir_idx = dir_idx.to(device)
            target_idx = target_idx.to(device)
            obs_posn = obs_posn.to(device)
            future_posn = future_posn.to(device)
            label = label.to(device)
            flip = flip.to(device)
            # 模型前向：得到答案 logits 和位置损失
            logits, ploss = model(crops, color_idx, dir_idx, target_idx, obs_posn,
                                  future_posn)
            pred = logits.argmax(-1)               # 预测类别（取分数高的）
            correct += (pred == label).sum().item()
            total += label.numel()
            # 只看那些"动作真的改变了结果"的样本（flip），它们才是真因果问题
            fl = flip.view(-1).bool()
            flip_correct += (pred[fl] == label[fl]).sum().item()
            flip_total += fl.sum().item()
            # 未来位置 MSE：模型存下的预测轨迹 pred_pos (B, pred, m, 2)
            pred_pos = model._pred_pos
            B = pred_pos.shape[0]
            all_mse += (pred_pos - future_posn).pow(2).mean().item() * B
            # 只取"被干预物体 target"那一列的预测，对比其真实未来位置
            sel_p = torch.stack([pred_pos[b, :, target_idx[b]] for b in range(B)])
            sel_t = torch.stack([future_posn[b, :, target_idx[b]] for b in range(B)])
            targ_mse += (sel_p - sel_t).pow(2).mean().item() * B
            delta_mag_sum += float(model._delta_mag) * B

            # 逐预测步误差（用于 delta_recursive 的累积误差分析）
            per_obj = (pred_pos - future_posn).pow(2).mean(-1)      # (B, pred, m)
            all_mse_step += per_obj.mean(dim=(0, 2)).cpu() * B
            idx = target_idx.view(B, 1, 1).expand(-1, P, 1)
            targ_mse_step += per_obj.gather(2, idx).squeeze(-1).mean(0).cpu() * B
            msk = torch.ones_like(per_obj)
            msk.scatter_(2, idx, 0.0)
            nontarg_mse_step += ((per_obj * msk).sum(2) /
                                 msk.sum(2).clamp(min=1)).mean(0).cpu() * B
            delta_mag_step += model._delta_mag_step * B

            if consistency:
                # 干预响应：对 4 个动作方向各前向一次，取 target 的首步预测位移。
                # follow = 各输入方向下"位移方向"与"输入方向"的余弦（对所有方向平均）。
                #   真正使用动作的模型 ≈ 1；只靠几何忽略动作的模型 ≈ 0（4 方向对消）。
                ar = torch.arange(B, device=device)
                ref = obs_posn[:, -1]                   # (B, m, 2) 最后观测帧
                disp = []
                for d_int in range(4):
                    dt = torch.full_like(dir_idx, d_int)
                    model(crops, color_idx, dt, target_idx, obs_posn, future_posn)
                    pf = model._pred_pos                # (B, pred, m, 2)
                    disp.append(pf[ar, 0, target_idx] - ref[ar, target_idx])
                D = torch.stack(disp, dim=1)            # (B, 4, 2)
                for d_int in range(4):
                    dv = torch.tensor(DIRVEC[d_int], dtype=torch.float32, device=device)
                    c = (D[:, d_int, 0] * dv[0] + D[:, d_int, 1] * dv[1]) / \
                        (D[:, d_int].norm(dim=-1) + 1e-6)
                    follow_sum += c.sum().item()
                follow_n += 4 * B

                # align：原动作方向下的对齐
                do = D[ar, dir_idx]                     # (B,2)
                dv_o = torch.tensor([DIRVEC[int(d)] for d in dir_idx],
                                    dtype=torch.float32, device=device)
                align_sum += ((do * dv_o).sum(-1) /
                              (do.norm(dim=-1) * dv_o.norm(dim=-1) + 1e-6)).sum().item()
                # align_flip：翻转动作方向后的对齐
                fl = torch.tensor([FLIP_DIR[int(d)] for d in dir_idx],
                                  dtype=torch.long, device=device)
                df = D[ar, fl]
                dv_f = torch.tensor([DIRVEC[int(d)] for d in fl],
                                    dtype=torch.float32, device=device)
                align_flip_sum += ((df * dv_f).sum(-1) /
                                   (df.norm(dim=-1) * dv_f.norm(dim=-1) + 1e-6)).sum().item()
                # cons：沿动作轴符号翻转率
                axis = torch.where(dir_idx < 2,
                                   torch.zeros_like(dir_idx),
                                   torch.ones_like(dir_idx))
                a0 = do[ar, axis]
                a1 = df[ar, axis]
                sign_flip = (a0 * a1) < 0
                strong = (a0.abs() > 1e-3) & (a1.abs() > 1e-3)
                cons_correct += sign_flip.sum().item()
                cons_total += B
                cons_correct_s += (sign_flip & strong).sum().item()
                cons_total_s += strong.sum().item()
            n_batches += 1
    # 用总数归一化得到最终指标
    acc = correct / max(total, 1)
    flip_acc = flip_correct / max(flip_total, 1)
    return {
        "acc": acc, "flip_acc": flip_acc,
        "future_mse": all_mse / max(total, 1),
        "target_mse": targ_mse / max(total, 1),
        "delta_mag": delta_mag_sum / max(total, 1),
        "cons": cons_correct / max(cons_total, 1),
        "cons_strict": cons_correct_s / max(cons_total_s, 1),
        "align": align_sum / max(cons_total, 1),
        "align_flip": align_flip_sum / max(cons_total, 1),
        "follow": follow_sum / max(follow_n, 1),
        "future_mse_step": (all_mse_step / max(total, 1)).tolist(),
        "target_mse_step": (targ_mse_step / max(total, 1)).tolist(),
        "nontarget_mse_step": (nontarg_mse_step / max(total, 1)).tolist(),
        "delta_mag_step": (delta_mag_step / max(total, 1)).tolist(),
    }


def train_one(model, train_ds, dev_ds, device, epochs, lr, batch_size, pos_w=5.0,
              pred=4, obs=2, consistency=True):
    """训练单个模型（只能是 JOINT 或 SEPARATE 其中之一）。

    损失 = 交叉熵(答案分类) + pos_w × MSE(未来位置回归)。
    pos_w : 位置回归损失的权重，调大能让 predictor 更看重"预测被干预后的
            未来轨迹"这一关键信号（这正是检验"动作是否进入predictor"的地方）。
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr)          # 优化器
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)  # 余弦退火调lr
    for ep in range(epochs):
        model.train()        # 切到训练模式（启用 dropout）
        tot = 0.0            # 累计本轮 loss
        n = 0                # 累计 batch 数
        for crops, color_idx, dir_idx, target_idx, obs_posn, future_posn, label, flip in iter_batches(train_ds, batch_size, shuffle=True, seed=ep, pred=pred, obs=obs):
            # 每个 epoch 用 seed=ep 打乱，保证每轮采到不同顺序
            crops = crops.to(device)
            color_idx = color_idx.to(device)
            dir_idx = dir_idx.to(device)
            target_idx = target_idx.to(device)
            obs_posn = obs_posn.to(device)
            future_posn = future_posn.to(device)
            label = label.to(device)
            logits, ploss = model(crops, color_idx, dir_idx, target_idx, obs_posn,
                                  future_posn)
            # 分类损失（答案）+ 位置回归损失（未来轨迹）
            cls = nn.CrossEntropyLoss()(logits, label)
            loss = cls + pos_w * ploss
            # 反向传播 + 梯度裁剪(防爆炸) + 更新参数
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
            n += 1
        sched.step()   # 每步降学习率
        # 每隔几轮打印一次 dev 指标（便于观察是否过拟合、是否学到东西）
        if (ep + 1) % max(1, epochs // 5) == 0 or ep == 0:
            ev = evaluate(model, dev_ds, device, batch_size, pred=pred, obs=obs,
                          consistency=consistency)
            print(f"  ep {ep+1:02d} loss {tot/max(n,1):.4f} dev_acc {ev['acc']:.3f} "
                  f"flip_acc {ev['flip_acc']:.3f} future_mse {ev['future_mse']:.4f} "
                  f"dmag {ev['delta_mag']:.4f} cons {ev['cons']:.3f}")
    return model


def main():
    """主入口：解析命令行参数，训练 JOINT 与 SEPARATE 两种模型并对比。

    流程：定义超参 -> 建 train/dev 数据集 -> 分别训 JOINT/SEPARATE（可多 seed）
    -> 在 dev 上评估 -> 汇总均值，打印两者的"未来位置 MSE"差距。
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800, help="train samples")
    ap.add_argument("--ne", type=int, default=100, help="eval samples")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--m", type=int, default=3)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--pred", type=int, default=4, help="future steps to predict")
    ap.add_argument("--obs", type=int, default=2, help="observed frames")
    ap.add_argument("--size", type=int, default=32, help="crop size")
    ap.add_argument("--nlayers", type=int, default=2)
    ap.add_argument("--encoder", default="cnn", choices=["cnn", "dinov2"])
    ap.add_argument("--delta_mode", type=int, default=0,
                    help="1 => TDV-style predict Delta instead of absolute future")
    ap.add_argument("--delta_recursive", type=int, default=0,
                    help="1 => base_k = base_{k-1} + Delta_k (else fixed base)")
    ap.add_argument("--cache", type=int, default=2000, help="dataset LRU cache size")
    ap.add_argument("--pos_w", type=float, default=5.0, help="MSE loss weight")
    ap.add_argument("--seeds", type=int, default=2, help="A/B runs per variant")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    # 设备：auto 表示有 GPU 用 GPU，否则 CPU
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"device={device} m={args.m} d={args.d} train={args.n} eval={args.ne} "
          f"enc={args.encoder} delta={args.delta_mode} rec={args.delta_recursive}")

    # 构造训练/验证数据集（通过不同 seed_offset 分开，避免泄漏）
    train_ds = make_dataset("train", args.n, m=args.m, size=args.size,
                            max_cache=args.cache, pred_steps=args.pred)
    dev_ds = make_dataset("dev", args.ne, m=args.m, size=args.size,
                          max_cache=args.cache, pred_steps=args.pred)

    results = {}
    # 对两种模式各跑一遍（可多个随机种子取平均，更稳健）
    for joint in [True, False]:
        name = "JOINT" if joint else "SEPARATE"
        run_accs = []    # 每次种子的准确率
        run_flip = []    # flip 子集准确率
        run_fut = []     # 全物体未来 MSE
        run_tgt = []     # target 物体未来 MSE
        for s in range(args.seeds):
            torch.manual_seed(42 + s)   # 固定随机种子，保证每次结果可比
            np.random.seed(42 + s)
            model = Model(d_model=args.d, m=args.m, obs=args.obs, pred=args.pred,
                          nlayers=args.nlayers, joint=joint, size=args.size,
                          encoder=args.encoder, delta_mode=bool(args.delta_mode),
                          delta_recursive=bool(args.delta_recursive)).to(device)
            model = train_one(model, train_ds, dev_ds, device,
                              args.epochs, args.lr, args.batch, pos_w=args.pos_w,
                              pred=args.pred, obs=args.obs)
            ev = evaluate(model, dev_ds, device, args.batch,
                          pred=args.pred, obs=args.obs)
            run_accs.append(ev["acc"])
            run_flip.append(ev["flip_acc"])
            run_fut.append(ev["future_mse"])
            run_tgt.append(ev["target_mse"])
            print(f"[{name}] seed {s}: acc {ev['acc']:.3f} "
                  f"flip_acc {ev['flip_acc']:.3f} "
                  f"future_mse {ev['future_mse']:.4f} target_mse {ev['target_mse']:.4f}")
        # 多种子平均，作为该模式的代表指标
        results[name] = {
            "acc": float(np.mean(run_accs)),
            "flip_acc": float(np.mean(run_flip)),
            "future_mse": float(np.mean(run_fut)),
            "target_mse": float(np.mean(run_tgt)),
        }
        print(f"== {name} mean: acc {results[name]['acc']:.3f} "
              f"flip_acc {results[name]['flip_acc']:.3f} "
              f"future_mse {results[name]['future_mse']:.4f} "
              f"target_mse {results[name]['target_mse']:.4f}")

    # ----- 结论：对比 JOINT 与 SEPARATE 的"未来位置 MSE"差距（越低越好） -----
    gap = results["JOINT"]["future_mse"] - results["SEPARATE"]["future_mse"]
    rel = gap / max(results["SEPARATE"]["future_mse"], 1e-6)   # 相对差距（百分比）
    tgap = results["JOINT"]["target_mse"] - results["SEPARATE"]["target_mse"]
    trel = tgap / max(results["SEPARATE"]["target_mse"], 1e-6)
    print("\n=== SUMMARY ===")
    print(f"Joint vs Separate future-position MSE gap: {gap:+.4f} "
          f"({rel:+.1%} relative, lower is better)")
    print(f"   separate = {results['SEPARATE']['future_mse']:.4f}")
    print(f"   joint    = {results['JOINT']['future_mse']:.4f}")
    print(f"Target-object future MSE gap: {tgap:+.4f} ({trel:+.1%} relative)")
    print(f"   separate = {results['SEPARATE']['target_mse']:.4f}")
    print(f"   joint    = {results['JOINT']['target_mse']:.4f}")

    # 把结果追加保存到 out/result.txt 便于回溯
    with open("out/result.txt", "a") as f:
        f.write(f"JOINT {results['JOINT']} | SEP {results['SEPARATE']} | "
                f"gap={gap:+.4f} rel={rel:+.1%}\n")
    print("saved -> out/result.txt")


if __name__ == "__main__":
    main()
