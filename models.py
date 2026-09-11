"""Object-centric causal-reasoning model (v2).

Updates over v1:
  * Per-frame object tokens: each observation frame is encoded separately, so the
    block-causal predictor over time blocks is non-trivial and the model can see
    inter-frame motion / velocity (needed for delta dynamics).
  * Pluggable object encoder: "cnn" (tiny conv net) or "dinov2" (timm
    vit_small_patch14_dinov2, pretrained).
  * TDV-style delta dynamics: `future_latent = base + Delta`, with
    `delta_mode` and `delta_recursive` switches (idea 1). The predictor predicts
    the *change* rather than the absolute future latent.

Action-conditioning fork is kept:
  joint=True  : language action injected into the predictor, bound to the target
                object's token at the last observed frame.
  joint=False : predictor is action-agnostic; action used only at the answer head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


N_COLOR = 8
N_DIR = 4


class ObjectEncoder(nn.Module):
    """物体编码器：把每个物体在**每一观测帧**的 crop 编码成一个 token。

    crops: (B, m, obs, C, H, W) -> (B, obs, m, d_model)
    位置 (x,y) 仍由 pos_mlp 编码，作为位置嵌入加到物体 token 上。

    kind="cnn"    : 一个小卷积网络（快，玩具特征）。
    kind="dinov2" : timm 的 vit_small_patch14_dinov2 预训练 backbone（强特征）。
    kind="dinov2_feat": 只保留 proj/pos_mlp，输入改为预计算好的 DINOv2 特征
                        (B, obs, m, 384)，用于"冻结 backbone + 预计算特征"的快速训练。
    """

    def __init__(self, obs=2, size=48, d_model=64, in_ch=3, kind="cnn",
                 state_dim=18):
        super().__init__()
        self.obs = obs
        self.kind = kind
        self.size = size

        if kind == "cnn":
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, 24, 3, 2, 1), nn.ReLU(),
                nn.Conv2d(24, 48, 3, 2, 1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            feat = 48
        elif kind == "dinov2":
            import os
            import timm
            # 用 dynamic_img_size 让位置嵌入在前向时插值到 112x112 (8x8 patch)，
            # 这样 checkpoint 里的 37x37 pos_embed 能直接严格加载，无需预缩放。
            ckpt = os.environ.get("DINOV2_CKPT")
            common = dict(num_classes=0, dynamic_img_size=True)
            if ckpt and os.path.exists(ckpt):
                self.backbone = timm.create_model(
                    "vit_small_patch14_dinov2.lvd142m", pretrained=False,
                    checkpoint_path=ckpt, **common)
            else:
                self.backbone = timm.create_model(
                    "vit_small_patch14_dinov2.lvd142m", pretrained=True, **common)
            feat = self.backbone.num_features
            self.register_buffer("mean",
                                 torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std",
                                 torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
            self.dino_size = 112
        elif kind == "dinov2_feat":
            feat = 384                       # vit_small DINOv2 特征维度
        elif kind == "state":
            feat = state_dim                 # 路线 A：直接输入物体状态向量
        else:
            raise ValueError(f"unknown encoder kind: {kind}")

        self.proj = nn.Linear(feat, d_model)   # 特征 -> d_model，对齐主空间维度
        self.pos_mlp = nn.Linear(2, d_model)   # 位置(x,y) -> d_model 位置嵌入

    def backbone_features(self, crops):
        """只跑到 backbone 池化特征（未经过 proj），返回 (B, obs, m, feat)。"""
        # crops: (B, m, obs, C, H, W)
        B, m, obs = crops.shape[0], crops.shape[1], crops.shape[2]
        # 逐帧逐物体编码：(B,m,obs,...) -> (B*m*obs, C, H, W)
        x = crops.reshape(B * m * obs, *crops.shape[-3:])
        if self.kind == "cnn":
            z = self.net(x).flatten(1)                       # (B*m*obs, 48)
        else:
            x = F.interpolate(x, size=self.dino_size, mode="bilinear",
                              align_corners=False)
            x = (x - self.mean) / self.std
            z = self.backbone(x)                             # (B*m*obs, feat)
        return z.reshape(B, m, obs, -1).permute(0, 2, 1, 3).contiguous()

    def forward_from_features(self, feats):
        """输入预计算特征 (B, obs, m, feat)，输出 (B, obs, m, d)。"""
        return self.proj(feats)

    def forward(self, crops):
        return self.proj(self.backbone_features(crops))     # (B, obs, m, d)


def causal_mask(n_blocks, m, device, dtype):
    """构造 block-causal 注意力掩码（物体间同帧双向、跨帧只允许过去→现在）。

    序列被组织成 n_blocks 个"时间块"，每块含 m 个物体 token。第 i 块只能看到
    j <= i 的块；同一块内 m 个物体完全可见。返回加性掩码（0 允许，-1e9 屏蔽）。
    """
    L = n_blocks * m
    block = torch.arange(L, device=device) // m
    keep = (block[:, None] >= block[None, :])
    return keep.logical_not().to(dtype) * -1e9


class Model(nn.Module):
    """完整模型：逐帧物体编码 → block-causal predictor → 位置头 + 答案头。

    delta_mode=False : predictor 直接输出绝对未来 latent（v1 行为）。
    delta_mode=True  : future_latent = base + Delta，其中 base 是最后一个观测块
                       的输出，Delta 由 delta_head 从 predictor 输出得到。
      delta_recursive=False : 每步相对同一 base 独立预测（无累积误差）。
      delta_recursive=True  : base_k = base_{k-1} + Delta_k（更物理，误差累积）。
    """

    def __init__(self, d_model=64, m=3, T=6, obs=2, pred=4,
                 nhead=4, nlayers=2, joint=True, size=48,
                 encoder="cnn", delta_mode=False, delta_recursive=False,
                 state_dim=18):
        super().__init__()
        self.m, self.T, self.obs, self.pred = m, T, obs, pred
        self.d_model = d_model
        self.joint = joint
        self.encoder = encoder
        self.delta_mode = delta_mode
        self.delta_recursive = delta_recursive

        self.enc = ObjectEncoder(obs=obs, size=size, d_model=d_model, kind=encoder,
                                 state_dim=state_dim)

        # 动作向量投影：把"颜色 one-hot + 方向 one-hot"投影成 d_model 维动作嵌入
        self.act_proj = nn.Sequential(
            nn.Linear(N_COLOR + N_DIR, d_model), nn.GELU(),
            nn.Linear(d_model, d_model))

        # 绑定头：计算每个物体 token 与动作向量的相似度，用于把动作"绑"到目标物体
        self.bind_q = nn.Linear(d_model, d_model)

        # block-causal Transformer 作为 predictor（JEPA predictor）
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=2 * d_model,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
        self.predictor = nn.TransformerEncoder(layer, num_layers=nlayers)

        # delta 头：只学"变了多少"。末层 zero-init => 初始 Delta=0（恒等起步），
        # 避免训练早期发散，也便于观察是否真的学到非零变化量。
        if delta_mode:
            self.delta_head = nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(),
                nn.Linear(d_model, d_model))
            nn.init.zeros_(self.delta_head[-1].weight)
            nn.init.zeros_(self.delta_head[-1].bias)

        # 位置头：从（预测出的）未来 latent 读出每个物体的未来位置 (pred*m*2)
        self.pos_head = nn.Linear(d_model, 2)
        # 答案头：读"预测未来轨迹 + 观测几何"，不给动作直接输入，
        # 强迫模型依靠 predictor 的 rollout 来推断答案。
        in_d = self.pred * self.m * 2 + self.m * 2
        self.ans = nn.Sequential(
            nn.Linear(in_d, 128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, 2))

        self._delta_mag = 0.0   # 最近一次前向的 mean|Delta|，供日志/评估读取

    def act_vector(self, color_idx, direction_idx, target_idx=None):
        """生成动作向量：'把<颜色>物体推向<方向>'。返回 (B, d_model)。"""
        if target_idx is not None:
            tc = color_idx.gather(1, target_idx.unsqueeze(-1)).squeeze(-1)  # (B,)
        else:
            tc = color_idx.mean(dim=1).long()
        c = F.one_hot(tc, N_COLOR).float()
        d = F.one_hot(direction_idx, N_DIR).float()
        return self.act_proj(torch.cat([c, d], dim=-1))

    def forward(self, crops, color_idx, direction_idx, target_idx,
                obs_posn, future_posn, joint=None, return_logits=True):
        # obs_posn: (B, obs, m, 2)  每观测帧的物体位置
        # future_posn: (B, pred, m, 2) 未来位置真值（动作后）
        if joint is None:
            joint = self.joint
        B = crops.shape[0]
        m, obs = self.m, self.obs
        dev = crops.device

        # 1) 逐帧物体编码 -> (B, obs, m, d)
        #    crops 为 6D (B,m,obs,C,H,W)；若为 4D (B,obs,m,feat) 则是预计算特征。
        if crops.dim() == 4:
            obj_tok = self.enc.forward_from_features(crops)
        else:
            obj_tok = self.enc(crops)
        # 2) 动作向量
        act_vec = self.act_vector(color_idx, direction_idx, target_idx)

        # ----- 核心 A/B 分叉 -----
        if joint:
            last = obj_tok[:, -1]                       # 当前(最后观测帧)状态 (B,m,d)
            scores = (self.bind_q(last) * act_vec[:, None, :]).sum(-1)   # (B,m)
            bind = torch.softmax(scores, dim=-1)
            cond = act_vec[:, None, :] * bind[:, :, None]
            # 把动作条件注入到**所有观测帧**（v1 行为）：相当于告诉世界模型
            # "接下来的干预是 X"，让整段观测-未来都带上动作条件。
            obj_tok = obj_tok + cond.unsqueeze(1)

        # 3) 加位置嵌入并拼成 predictor 输入序列（obs 个真实时间块）
        pos = self.enc.pos_mlp(obs_posn)                # (B, obs, m, d)
        obs_seq = (obj_tok + pos).reshape(B, obs * m, self.d_model)

        # 未来块用 0 起始 token，交给因果掩码补全；joint 时同样注入动作条件，
        # 使 predictor 在生成未来时直接以该干预为条件（绑定目标物体）。
        fut_query = torch.zeros(B, self.pred * m, self.d_model, device=dev)
        if joint:
            fut_cond = cond.unsqueeze(1).expand(B, self.pred, m, self.d_model)
            fut_query = fut_query + fut_cond.reshape(B, self.pred * m, self.d_model)
        seq = torch.cat([obs_seq, fut_query], dim=1)

        n_blocks = obs + self.pred
        mask = causal_mask(n_blocks, m, dev, seq.dtype)
        seq_out = self.predictor(seq, mask=mask)

        obs_out = seq_out[:, :obs * m].reshape(B, obs, m, -1)
        base = obs_out[:, -1]                            # (B, m, d) 最后观测块
        fut_part = seq_out[:, obs * m:].reshape(B, self.pred, m, -1)

        # 4) 未来 latent：直接预测 vs TDV 式 base + Delta
        if self.delta_mode:
            delta = self.delta_head(fut_part)            # (B, pred, m, d)
            self._delta_mag = float(delta.detach().abs().mean().item())
            # 逐预测步的 Δ 幅值 (pred,)，供逐步误差分析
            self._delta_mag_step = delta.detach().abs().mean(dim=(0, 2, 3)).cpu()
            if self.delta_recursive:
                fut_latent = base.unsqueeze(1) + torch.cumsum(delta, dim=1)
            else:
                fut_latent = base.unsqueeze(1) + delta
        else:
            self._delta_mag = 0.0
            self._delta_mag_step = torch.zeros(self.pred)
            fut_latent = fut_part

        # 5) 位置头 + 轨迹预测损失
        pred_pos = self.pos_head(fut_latent)             # (B, pred, m, 2)
        pos_loss = F.mse_loss(pred_pos, future_posn)
        self._pred_pos = pred_pos

        # 6) 答案头：预测未来轨迹 + 初始几何
        traj_flat = pred_pos.reshape(B, -1)
        geo_flat = obs_posn[:, -1].reshape(B, -1)        # 最后一帧几何
        logits = self.ans(torch.cat([traj_flat, geo_flat], dim=-1))
        return logits, pos_loss


if __name__ == "__main__":
    B, m, obs, size = 2, 3, 2, 32
    crops = torch.randn(B, m, obs, 3, size, size)
    color_idx = torch.randint(0, 8, (B, m))
    direction_idx = torch.randint(0, 4, (B,))
    target_idx = torch.randint(0, m, (B,))
    obs_posn = torch.rand(B, obs, m, 2)
    future_posn = torch.rand(B, 4, m, 2)
    for dm, dr in [(False, False), (True, False), (True, True)]:
        model = Model(joint=True, delta_mode=dm, delta_recursive=dr)
        logits, ploss = model(crops, color_idx, direction_idx,
                              target_idx, obs_posn, future_posn)
        print(f"delta={dm} rec={dr} logits{tuple(logits.shape)} "
              f"pos={ploss.item():.4f} dmag={model._delta_mag:.4f}")
