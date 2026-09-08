# -*- coding: utf-8 -*-
"""
Prototype-anchor + contextual-bandit defense for federated backdoor attacks.

핵심 아이디어
-------------
1) 앵커(anchor): client update를 한 번도 안 먹는 "독립 reference 모델".
   서버 clean 데이터로 한 번 학습 후 freeze. 이걸로 "정상은 이렇게 생겼다"는
   클래스 관계행렬 S_A (KxK cosine, 회전불변)를 한 번 계산해 고정.
   -> 글로벌을 임베더로 안 쓰므로 "서버가 아직 안 당했다"를 가정하지 않음.

2) state: 매 라운드 각 client 모델을 서버 validation에 통과시켜 얻은 임베딩의
   (a) client들끼리의 분산 통계 (상대),
   (b) 앵커 대비 관계행렬 이탈 (절대, 다수/공모 방어),
   (c) flip 불변성 일관성 (가벼운 기본 augmentation).

3) action: client id가 아니라 "selection rule"(임베딩 거리 점수 + band-pass 도넛).

4) reward (this-round FedAvg(all) 대비 advantage):
   ① acc advantage      - 메인 태스크 제약
   ② anchor 기하 일치     - (B) 다수/공모 방어 (trigger-agnostic 코어)
   ③ flip 일관성          - 가벼운 control + 거친 공격
   ④ class-drift 페널티   - 특정 클래스로의 비정상 흡인(target attractor) 흔적

주의: 완전 stealth 백도어는 ②④로도 부분적으로만 잡힘 -> 한계로 명시.
서버 clean 데이터는 dataset_test에서 잘라 씀(연구용 단순화).
"""

import os
import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as TF

ROT_ANGLE = 20.0  # 회전 perturbation 각도(도)
DUMP_ROTS = [90, 180]  # 덤프용 회전 각도(연구에 쓰는 rot90·rot180만; 회전은 서로 중복 큼)
from torch.utils.data import DataLoader, Subset

from defense.Fed import FedAvg

EPS = 1e-8
# 관측(STATE)은 풍부하게: relmat 포함 모든 detector 분포를 봄 (관측이라 순환 무관).
# 선택(ACTION)은 reward(relmat-aggregate)와 순환 없게 relmat·anchor_dev 제외.
STATE_KEYS = ["relmat", "l2_to_center", "anchor_dev", "flip"]
ACTION_KEYS = ["flip", "l2_to_center"]   # reward에 없는 detector들(순환 X). relmat/anchor_dev는 reward와 순환이라 제외
STATE_DIM = 2 + 5 * len(STATE_KEYS)  # global(n, iter_frac) + 5 feat × 4 score = 22


def add_weight_noise(w, w_glob, w_locals, noise, mode="norm", clip_value=None):
    """weight-space 노이즈(배수구). 노이즈 벡터 norm = noise · median_update_norm.
    per-element std 를 √P 로 정규화해 노이즈 벡터 크기를 update norm 대비로 제어."""
    if not noise or noise <= 0:
        return w
    # 백도어가 사는 conv/linear 가중치에만. BN running_mean/var(정규화 통계)은 제외(스케일 달라 박살남)
    fkeys = [k for k in w_glob if w_glob[k].dtype.is_floating_point
             and not k.endswith(("running_mean", "running_var"))]
    P = sum(int(w_glob[k].numel()) for k in fkeys)
    norms = []
    for wl in w_locals:
        sq = 0.0
        for k in fkeys:
            sq += float(torch.sum((wl[k] - w_glob[k]) ** 2))
        norms.append(sq ** 0.5)
    if str(mode) == "flame":
        # FLAME(USENIX'22) 원식: 원소별 std = λ · S,  S = 선별 클라 업데이트 노름의 중앙값.
        # √P 로 나누지 않으므로 노이즈 벡터 노름이 √P 배 커진다(λ=0.001 기준 업데이트의 ~3배).
        _S = float(clip_value) if clip_value is not None else float(np.median(norms))
        std = float(noise) * _S
    else:
        std = float(noise) * float(np.median(norms)) / (float(P) ** 0.5)
    if int(os.environ.get("PB_NOISE_DEBUG", "0")):
        print("[noise] mode=%s  clip_value=%s  median_norm=%.3f  P=%d  std=%.6f  vec_norm=%.2f"
              % (mode, clip_value, float(np.median(norms)), P, std, std * (P ** 0.5)))
    for k in fkeys:
        w[k] = w[k] + (torch.randn_like(w[k].float()) * std).to(w[k].dtype)
    return w


def fedavg_noise(w_locals, w_glob, noise):
    """밴딧 없이 FedAvg(all) + 노이즈 (ablation: FL+noise)."""
    return add_weight_noise(FedAvg(w_locals), w_glob, w_locals, noise)


# --------------------------------------------------------------------------- #
# 임베딩 / 프로토타입 유틸
# --------------------------------------------------------------------------- #
def _features(net, loader, device, flip=False, rotate=0.0, aug=None):
    """val set 을 net.get_feature 로 통과시켜 (N, D) 임베딩과 라벨 반환.
    aug: ('bright',factor)/('contrast',f)/('blur',sigma)/('gray',None)/('invert',None) 등 추가 perturbation."""
    net.eval()
    feats, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            if flip:
                x = torch.flip(x, dims=[-1])  # horizontal flip
            if rotate:
                x = TF.rotate(x, float(rotate))  # rotation perturbation
            if aug is not None:
                kind, p = aug
                if kind == "bright":   x = TF.adjust_brightness(x, p)
                elif kind == "contrast": x = TF.adjust_contrast(x, p)
                elif kind == "blur":   x = TF.gaussian_blur(x, kernel_size=5, sigma=float(p))
                elif kind == "gray":   x = TF.rgb_to_grayscale(x, num_output_channels=3)
                elif kind == "invert": x = TF.invert(x)
            f = net.get_feature(x)
            if f.ndim == 4:               # (N, C, H, W) -> (N, C)
                f = f.mean(dim=(2, 3))
            f = f.reshape(f.size(0), -1)
            feats.append(f.cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(feats, 0), np.concatenate(labels, 0)


def _prototypes(feats, labels, classes):
    """클래스별 평균 임베딩 (K, D). 비어있는 클래스는 0 벡터."""
    protos = []
    for c in classes:
        m = feats[labels == c]
        protos.append(m.mean(0) if len(m) > 0 else np.zeros(feats.shape[1]))
    return np.stack(protos, 0)


def _relmat(protos):
    """클래스 간 cosine 관계행렬 (K, K). 모델이 달라도 비교 가능(회전불변)."""
    n = protos / (np.linalg.norm(protos, axis=1, keepdims=True) + EPS)
    return n @ n.T


def _rot_angle_of(name):
    """detector 이름 → 회전각(도). 'rotation'=기본각, 'rotXX'=XX도, 회전 아니면 None."""
    if name == "rotation":
        return ROT_ANGLE
    if name.startswith("rot"):
        try:
            return float(name[3:])
        except ValueError:
            return None
    return None


# detector 이름 → augmentation (kind, param). 회전/flip/relmat 제외한 일반 aug detector.
_AUG_MAP = {
    "invert": ("invert", None), "gray": ("gray", None), "blur": ("blur", 1.5),
    "bright05": ("bright", 0.5), "bright15": ("bright", 1.5),
    "contrast05": ("contrast", 0.5), "contrast15": ("contrast", 1.5),
}


def _aug_of(name):
    """detector 이름 → augmentation (kind,param) 또는 None."""
    return _AUG_MAP.get(name)


def _accuracy(net, loader, device):
    """val set clean accuracy (%)."""
    net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = net(x).max(1)[1]
            correct += pred.eq(y).sum().item()
            total += y.size(0)
    return 100.0 * correct / max(total, 1)


# --------------------------------------------------------------------------- #
# action (selection rule)
# --------------------------------------------------------------------------- #
class RuleAction:
    """band-pass(도넛) 선택 + 선택된 업데이트 norm 클리핑."""
    def __init__(self, score_type, drop_inner, keep_outer, clip_q=None):
        self.score_type = score_type
        self.drop_inner = drop_inner
        self.keep_outer = keep_outer
        self.clip_q = clip_q               # None=클립없음, 아니면 clip_q×median_norm 으로 잘라

    def __repr__(self):
        c = "none" if self.clip_q is None else f"{self.clip_q}"
        return f"Rule({self.score_type}, drop={self.drop_inner}, keep={self.keep_outer}, clip={c})"


def build_rule_actions(keys=None):
    keys = keys if keys is not None else ACTION_KEYS
    ratio_pairs = [(0.0, 0.5), (0.0, 0.8), (0.1, 0.8), (0.2, 0.9), (0.0, 1.0)]  # 5 band
    clip_levels = [None, 1.0, 0.5]                                              # 3 clip
    return [RuleAction(s, di, ko, cq)
            for s in keys for (di, ko) in ratio_pairs for cq in clip_levels]


# --------------------------------------------------------------------------- #
# Neural Thompson Sampling contextual bandit (self-contained)
#   reward_net: state -> per-action reward. 탐험은 NTK 그래디언트 기반 대각
#   precision으로 각 arm의 불확실도 sigma 를 구해 N(mean, (nu*sigma)^2) 샘플링.
# --------------------------------------------------------------------------- #
class _QNet(nn.Module):
    def __init__(self, state_dim, num_actions, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )

    def forward(self, x):
        return self.net(x)


class NeuralTS:
    def __init__(self, n_actions, state_dim, hidden=64, lr=1e-2, lam=1.0, nu=1.0,
                 batch_size=16, train_steps=2, device=None):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.n = n_actions
        self.nu = float(nu)
        self.batch_size = batch_size
        self.train_steps = train_steps
        self.net = _QNet(state_dim, n_actions, hidden).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        p = sum(q.numel() for q in self.net.parameters())
        self.precision = torch.full((p,), float(lam), device=self.device)   # 대각 근사
        self.buffer = []   # (state, action, reward)

    def _grad(self, state, a):
        st = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        val = self.net(st)[0, a]
        grads = torch.autograd.grad(val, tuple(self.net.parameters()))
        return torch.cat([g.detach().reshape(-1) for g in grads])

    def select(self, state, explore=True):
        st = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            means = self.net(st).squeeze(0).cpu().numpy()
        if not explore:
            return int(np.argmax(means))
        scores = []
        for a in range(self.n):
            g = self._grad(state, a)
            var = torch.sum((g * g) / torch.clamp(self.precision, min=1e-8))
            sigma = float(torch.sqrt(torch.clamp(var, min=0.0)).item())
            scores.append(float(means[a]) + float(np.random.normal(0.0, self.nu * sigma)))
        return int(np.argmax(scores))

    def update(self, state, a, r):
        self.buffer.append((np.asarray(state, dtype=np.float32), int(a), float(r)))
        # reward_net 학습 (smooth_l1)
        if len(self.buffer) >= 8:
            for _ in range(self.train_steps):
                k = min(self.batch_size, len(self.buffer))
                idx = np.random.randint(0, len(self.buffer), size=k)
                bs = torch.tensor(np.stack([self.buffer[i][0] for i in idx]), device=self.device)
                ba = torch.tensor([self.buffer[i][1] for i in idx], device=self.device).long().unsqueeze(1)
                br = torch.tensor([self.buffer[i][2] for i in idx], device=self.device,
                                  dtype=torch.float32).unsqueeze(1)
                pred = self.net(bs).gather(1, ba)
                loss = F.smooth_l1_loss(pred, br)
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                self.opt.step()
        # 관측한 (state, action)의 그래디언트로 precision 갱신
        g = self._grad(state, a)
        self.precision += g * g


# --------------------------------------------------------------------------- #
# 메인 방어 객체
# --------------------------------------------------------------------------- #
class ProtoBanditDefense:
    def __init__(self, net_glob, server_dataset, args, probe_loader=None):
        self.args = args
        self.device = args.device
        self.num_classes = 100 if getattr(args, "dataset", "cifar") == "cifar100" else 10
        self.classes = list(range(self.num_classes))

        # 하이퍼파라미터 (options.py 안 건드려도 getattr 기본값으로 동작)
        self.w_acc = getattr(args, "pb_w_acc", 1.0)
        self.w_anchor = getattr(args, "pb_w_anchor", 2.0)
        self.w_flip = getattr(args, "pb_w_flip", 0.5)
        self.w_drift = getattr(args, "pb_w_drift", 1.0)
        self.acc_decay = getattr(args, "pb_acc_decay", 0.8)  # 후반 acc 가중 감쇠(0=없음,0.8=후반 0.2x)
        self.noise = getattr(args, "noise", 0.0)             # weight-space 노이즈(배수구). 0=없음
        self.evals_per_round = getattr(args, "pb_evals", 10)  # 라운드당 평가(샘플링)할 액션 수
        self.last_meta = None                                # 매 라운드 방어 액션 기록(wandb용)
        bs = getattr(args, "bs", 64)

        self.template = copy.deepcopy(net_glob)            # 평가/임베딩용 재사용 템플릿

        # --- 룰베이스 설정 (anchor 분기보다 먼저 결정) ---
        self.rule_mode = bool(getattr(args, "pb_rule", 0))       # 1=룰베이스(밴딧 없음)
        self.no_anchor = bool(getattr(args, "pb_no_anchor", 0))  # 1=앵커 없이 또래비교만
        self.perturb_peer = bool(getattr(args, "pb_perturb_peer", 0))  # 1=perturbed 행렬 또래비교(자기차이 대신)
        # 룰베이스 결합 detector 조합 (ablation). 예: relmat / flip / rotation / relmat,flip,rotation
        self.rule_dets = [s.strip() for s in
                          getattr(args, "pb_rule_dets", "relmat,flip").split(",") if s.strip()]
        # 회전 detector(rot90, rot180, rotation=기본각 …)와 각도 매핑
        self.rot_dets = [d for d in self.rule_dets if _rot_angle_of(d) is not None]
        self.rot_angles = {d: _rot_angle_of(d) for d in self.rot_dets}
        # 일반 aug detector(invert/gray/blur/bright/contrast …)
        self.aug_dets = [d for d in self.rule_dets if _aug_of(d) is not None]
        self.aug_specs = {d: _aug_of(d) for d in self.aug_dets}
        self.action_keys = getattr(args, "pb_actions", "flip,l2_to_center").split(",")
        self.reward_mode = getattr(args, "pb_reward", "full")    # full | acc

        if self.no_anchor:
            # anchor-free: 앵커/S_A 없이 train-holdout probe로 또래비교만.
            # flip/rot의 앵커 baseline(상수)은 z-score median-centering에서 상쇄되므로 0으로 둬도 동일.
            assert probe_loader is not None, "pb_no_anchor=1 이면 probe_loader 필요"
            self.val_loader = probe_loader
            self.val_idx = None
            self.eval_idx = None                  # main_fed가 full test set으로 평가
            self.anchor = None
            self.S_A = None
            self.anchor_flipdeg = 0.0
            self.anchor_rotdeg = {d: 0.0 for d in self.rot_dets}
            print("[protorule] ANCHOR-FREE | probe=%d장(train-holdout) | dets=%s | flip·rot=peer-median"
                  % (len(probe_loader.dataset), self.rule_dets))
        else:
            # --- 앵커: 독립 reference 모델 (client update 안 먹음) ---
            ckpt_path = getattr(args, "pb_anchor_ckpt", "./save/anchor_cifar_resnet18.pt")
            if ckpt_path and os.path.isfile(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                self.val_idx = list(ckpt["val_idx"])
                self.eval_idx = list(ckpt["eval_idx"])           # 평가 전용(방어풀과 disjoint)
                self.val_loader = DataLoader(Subset(server_dataset, self.val_idx),
                                             batch_size=bs, shuffle=False)
                anchor = copy.deepcopy(net_glob).to(self.device)
                anchor.load_state_dict(ckpt["state_dict"])
                anchor.eval()
                for p in anchor.parameters():
                    p.requires_grad_(False)
                self.anchor = anchor
                print(f"[protobandit] loaded frozen anchor: {ckpt_path} "
                      f"(val={len(self.val_idx)}, eval={len(self.eval_idx)})")
            else:
                per_cls = max(2, 4000 // self.num_classes)       # anchor-train per class
                per_val = max(2, 1000 // self.num_classes)       # held-out val per class
                a_idx, v_idx, e_idx = self._balanced_split(server_dataset, per_cls, per_val)
                self.val_idx, self.eval_idx = v_idx, e_idx
                self.val_loader = DataLoader(Subset(server_dataset, v_idx), batch_size=bs, shuffle=False)
                anchor_loader = DataLoader(Subset(server_dataset, a_idx), batch_size=bs, shuffle=True)
                self.anchor = self._train_anchor(net_glob, anchor_loader)
                print("[protobandit] anchor trained from scratch "
                      f"(val={len(v_idx)}, eval={len(e_idx)})")
            _, self.S_A, self.anchor_flipdeg = self._model_stats(self.anchor)
            self.anchor_rotdeg = ({d: self._rotdeg(self.anchor, self.rot_angles[d]) for d in self.rot_dets}
                                  if self.rule_mode else {d: 0.0 for d in self.rot_dets})
            print("[protobandit] S_A shape:", self.S_A.shape,
                  "anchor flipdeg: %.4f rotdeg: %s"
                  % (self.anchor_flipdeg, {d: round(v, 4) for d, v in self.anchor_rotdeg.items()}))

        # --- 밴딧 (Neural Thompson Sampling) ---
        self.actions = build_rule_actions(self.action_keys)
        self.bandit = NeuralTS(len(self.actions), STATE_DIM,
                               hidden=getattr(args, "pb_hidden", 64),
                               nu=getattr(args, "pb_nu", 1.0),
                               device=self.device)
        print("[protobandit] actions=%d keys=%s reward=%s rule=%s noise=%s"
              % (len(self.actions), self.action_keys, self.reward_mode, self.rule_mode, self.noise))

    # ------------------------------------------------------------------ #
    def _balanced_split(self, dataset, per_cls, per_val):
        rng = np.random.RandomState(getattr(self.args, "seed", 42))
        by = {c: [] for c in range(self.num_classes)}
        for i in range(len(dataset)):
            y = int(dataset[i][1])
            if y in by:
                by[y].append(i)
        a_idx, v_idx, e_idx = [], [], []
        for c in range(self.num_classes):
            arr = by[c]
            rng.shuffle(arr)
            a_idx += arr[:per_cls]
            v_idx += arr[per_cls:per_cls + per_val]
            e_idx += arr[per_cls + per_val:]
        return a_idx, v_idx, e_idx

    def _train_anchor(self, net_glob, loader):
        anchor = copy.deepcopy(net_glob).to(self.device)
        epochs = getattr(self.args, "pb_anchor_epochs", 10)
        opt = torch.optim.SGD(anchor.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        anchor.train()
        for _ in range(epochs):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad()
                F.cross_entropy(anchor(x), y).backward()
                opt.step()
        anchor.eval()
        for p in anchor.parameters():
            p.requires_grad_(False)
        acc = _accuracy(anchor, self.val_loader, self.device)
        print("[protobandit] anchor val acc after %d ep: %.2f%% "
              "(낮으면 데이터/epoch 늘리거나 pretrained 권장)" % (epochs, acc))
        return anchor

    def _model_stats(self, net):
        """모델 하나의 (proto_vec, 관계행렬 S, flip-degradation) 반환.
        flip detector·앵커를 안 쓰면 flip forward를 생략(클라당 forward 2→1회)."""
        fc, labels = _features(net, self.val_loader, self.device, flip=False)
        pc = _prototypes(fc, labels, self.classes)
        rc = _relmat(pc)
        need_flip = ("flip" in getattr(self, "rule_dets", []) or self.S_A is not None
                     or not self.rule_mode)
        if not need_flip:
            return pc.reshape(-1), rc, 0.0
        ff, _ = _features(net, self.val_loader, self.device, flip=True)
        rf = _relmat(_prototypes(ff, labels, self.classes))
        return pc.reshape(-1), rc, float(np.linalg.norm(rc - rf))

    def _rotdeg(self, net, angle=ROT_ANGLE):
        """회전(angle°) perturbation 시 관계행렬 변동량."""
        fc, labels = _features(net, self.val_loader, self.device)
        fr, _ = _features(net, self.val_loader, self.device, rotate=float(angle))
        rc = _relmat(_prototypes(fc, labels, self.classes))
        rr = _relmat(_prototypes(fr, labels, self.classes))
        return float(np.linalg.norm(rc - rr))

    def _flip_relmat_flat(self, net):
        """flip perturbation 후 관계행렬(K*K flatten). perturb_peer(또래비교)용."""
        fc, labels = _features(net, self.val_loader, self.device, flip=True)
        return _relmat(_prototypes(fc, labels, self.classes)).reshape(-1)

    def _rot_relmat_flat(self, net, angle):
        """rotation(angle°) perturbation 후 관계행렬(K*K flatten). perturb_peer용."""
        fr, labels = _features(net, self.val_loader, self.device, rotate=float(angle))
        return _relmat(_prototypes(fr, labels, self.classes)).reshape(-1)

    def _aug_relmat_flat(self, net, aug):
        """일반 aug(bright/contrast/blur/gray/invert) 후 관계행렬(K*K flatten). perturb_peer용."""
        fa, labels = _features(net, self.val_loader, self.device, aug=aug)
        return _relmat(_prototypes(fa, labels, self.classes)).reshape(-1)

    def _peer_score(self, mats):
        """관계행렬 집합(n, K*K)의 상삼각 표준화 후 median-거리(또래 이상치 점수, 높을수록 의심).
        relmat detector와 동일 로직을 임의 perturbation 행렬에 적용."""
        R = np.asarray(mats, dtype=np.float64)
        Kc = int(round(np.sqrt(R.shape[1])))
        iu0, iu1 = np.triu_indices(Kc, k=1)
        cols = iu0 * Kc + iu1
        Ru = R[:, cols]
        Rz = (Ru - Ru.mean(0)) / (Ru.std(0) + EPS)
        return np.linalg.norm(Rz - np.median(Rz, axis=0), axis=1)

    def _augdeg(self, net, aug):
        """임의 augmentation(aug=(kind,param)) 시 관계행렬 변동량. invert/gray/blur/bright/contrast."""
        fc, labels = _features(net, self.val_loader, self.device)
        fa, _ = _features(net, self.val_loader, self.device, aug=aug)
        rc = _relmat(_prototypes(fc, labels, self.classes))
        ra = _relmat(_prototypes(fa, labels, self.classes))
        return float(np.linalg.norm(rc - ra))

    def _perturb_relmats(self, net):
        """clean + flip + 회전(DUMP_ROTS) 각각의 관계행렬(K*K flatten) dict. 시각화 덤프용."""
        fc, labels = _features(net, self.val_loader, self.device)
        out = {"clean": _relmat(_prototypes(fc, labels, self.classes)).reshape(-1)}
        ff, _ = _features(net, self.val_loader, self.device, flip=True)
        out["flip"] = _relmat(_prototypes(ff, labels, self.classes)).reshape(-1)
        for ang in DUMP_ROTS:
            fr, _ = _features(net, self.val_loader, self.device, rotate=float(ang))
            out["rot%d" % ang] = _relmat(_prototypes(fr, labels, self.classes)).reshape(-1)
        # 추가 perturbation 진단(밝기/대비/블러/흑백/반전) — VGG-LGA 신호 살아나나
        for name, aug in [("bright05", ("bright", 0.5)), ("bright15", ("bright", 1.5)),
                          ("contrast05", ("contrast", 0.5)), ("contrast15", ("contrast", 1.5)),
                          ("blur", ("blur", 1.5)), ("gray", ("gray", None)),
                          ("invert", ("invert", None))]:
            fa, _ = _features(net, self.val_loader, self.device, aug=aug)
            out[name] = _relmat(_prototypes(fa, labels, self.classes)).reshape(-1)
        return out

    def _stats_from_weights(self, w):
        self.template.load_state_dict(w)
        self.template.to(self.device)
        return self._model_stats(self.template)

    # ------------------------------------------------------------------ #
    def _client_scores(self, proto_vecs, anchordev, flipexcess, relmats, rotexcess=None):
        """per-client detector 점수 dict (전부 '값 높을수록 의심')."""
        x = np.asarray(proto_vecs, dtype=np.float64)
        center = np.median(x, axis=0)
        l2c = np.linalg.norm(x - center, axis=1)
        R = np.asarray(relmats, dtype=np.float64)                 # (n, K*K)
        # 상삼각 off-diagonal만 사용: 대각(=상수 1, 수치노이즈)·하삼각(대칭 중복) 제거
        # → 고유 클래스쌍 K(K-1)/2개만. 표준화는 열별 독립이라 상삼각 부분집합에도 동일하게 성립.
        Kc = int(round(np.sqrt(R.shape[1])))
        iu0, iu1 = np.triu_indices(Kc, k=1)
        cols = iu0 * Kc + iu1
        Ru = R[:, cols]                                           # (n, K(K-1)/2)
        Rz = (Ru - Ru.mean(0)) / (Ru.std(0) + EPS)                # within-round 표준화
        relmat_score = np.linalg.norm(Rz - np.median(Rz, axis=0), axis=1)  # 상삼각 편차의 2-norm
        # reldist: 같은 프로토타입을 '유클리드 거리'로 관계행렬화 (cos 대신 거리로 유사도 정의).
        #  ‖p_a−p_b‖² = ‖p_a‖²+‖p_b‖²−2p_a·p_b 로 K×K를 클라마다 O(K²D) 없이 계산.
        Kp = len(self.classes)
        Pv = x.reshape(x.shape[0], Kp, -1)                        # (n, K, D) 원본 프로토타입
        G = Pv @ np.transpose(Pv, (0, 2, 1))                      # (n, K, K) 그람행렬
        sq = np.einsum('nkk->nk', G)                              # (n, K) ‖p_k‖²
        Dm = np.sqrt(np.maximum(sq[:, :, None] + sq[:, None, :] - 2.0 * G, 0.0))
        Du = Dm[:, iu0, iu1]                                      # 상삼각만
        Dz = (Du - Du.mean(0)) / (Du.std(0) + EPS)
        reldist_score = np.linalg.norm(Dz - np.median(Dz, axis=0), axis=1)
        out = {
            "relmat": relmat_score,
            "reldist": reldist_score,
            "l2_to_center": l2c,
            "anchor_dev": np.asarray(anchordev, dtype=np.float64),
            "flip": -np.asarray(flipexcess, dtype=np.float64),     # 악성=낮은 flipexcess → 부호반전
        }
        if rotexcess:
            # detector별(rot90, rot180 …) 회전 변동. flip과 동일 가정(악성=낮은 변동 → 부호반전)
            for d, vals in rotexcess.items():
                out[d] = -np.asarray(vals, dtype=np.float64)
        return out

    def _build_state(self, scores_by_type, n):
        """각 detector(relmat 포함) 점수의 분리도 + 라운드 진행도 → 밴딧이 regime 인식."""
        iter_frac = min(float(getattr(self.args, "iter", 0))
                        / max(float(getattr(self.args, "epochs", 1)), 1.0), 1.0)
        feats = [min(n / 20.0, 1.0), iter_frac]                   # global: 클라 수, 학습 진행도
        for key in STATE_KEYS:
            s = np.asarray(scores_by_type[key], dtype=np.float64)
            med = np.median(s)
            mad = float(np.median(np.abs(s - med))) + EPS
            z = np.sort((s - med) / mad)
            top1 = float(z[-1])                                   # 가장 의심스런 놈
            gap12 = float(z[-1] - z[-2]) if len(z) >= 2 else 0.0  # 1등-2등 격차(고립 신호)
            gap23 = float(z[-2] - z[-3]) if len(z) >= 3 else 0.0  # 2등-3등(악성 2+ 대비)
            frac_above2 = float(np.mean(z > 2.0))                 # outlier 몇 명
            spread = float(z[-1] - z[0])                          # 전체 퍼짐(robust)
            feats += [top1, gap12, gap23, frac_above2, spread]
        state = np.array(feats, dtype=np.float64)
        state[~np.isfinite(state)] = 0.0
        return np.clip(state, -10.0, 10.0)

    def _select_clients(self, action, scores_by_type):
        scores = np.asarray(scores_by_type[action.score_type], dtype=np.float64)
        n = len(scores)
        order = np.argsort(scores)                          # 신뢰 높은 순(점수 낮은 순)
        lo = int(action.drop_inner * n)
        hi = max(lo + 1, int(action.keep_outer * n))
        sel = order[lo:hi]
        return sel if len(sel) > 0 else order[:1]

    def _clipped_fedavg(self, sel, w_locals, w_glob, clip_q, return_clip=False):
        """선택된 업데이트(=w_i-w_glob)를 clip_q×median_norm 으로 잘라 평균. clip_q=None이면 일반 FedAvg."""
        if clip_q is None or len(sel) == 0:
            _r = FedAvg([w_locals[i] for i in sel])
            return (_r, None) if return_clip else _r
        fkeys = [k for k in w_glob if w_glob[k].dtype.is_floating_point
                 and not k.endswith(("running_mean", "running_var"))]
        norms = []
        for i in sel:
            sq = 0.0
            for k in fkeys:
                sq += float(torch.sum((w_locals[i][k] - w_glob[k]) ** 2))
            norms.append(sq ** 0.5)
        clip_value = clip_q * float(np.median(norms))
        new_w = copy.deepcopy(w_glob)
        m = len(sel)
        for k in w_glob:
            if not w_glob[k].dtype.is_floating_point:
                new_w[k] = w_locals[sel[0]][k]
                continue
            agg = torch.zeros_like(w_glob[k], dtype=torch.float32)
            for j, i in enumerate(sel):
                scale = min(1.0, clip_value / (norms[j] + 1e-8))
                agg += (w_locals[i][k] - w_glob[k]).float() * scale
            new_w[k] = w_glob[k] + (agg / m).to(w_glob[k].dtype)
        return (new_w, clip_value) if return_clip else new_w

    def _reward(self, sel, action, w_locals, w_glob, baseline):
        """선택집합 -> (클리핑) 집계 -> baseline(FedAvg-all) 대비 advantage."""
        acc_ref, rel_ref, fd_ref = baseline
        cand = self._clipped_fedavg(sel, w_locals, w_glob, action.clip_q)
        self.template.load_state_dict(cand)
        self.template.to(self.device)
        acc = _accuracy(self.template, self.val_loader, self.device)
        _, rel, fd = self._model_stats(self.template)

        # relmat 계열만 reward에 (action 과 순환 없음). flip_term 은 flip 이 action 이라 제거.
        acc_adv = (acc - acc_ref) / 100.0
        anchor_adv = -(np.linalg.norm(rel - self.S_A) - np.linalg.norm(rel_ref - self.S_A))
        # ④ class-drift: 서로 다른 클래스가 앵커보다 더 가까워졌나(target attractor)
        off = (rel - self.S_A).copy()
        np.fill_diagonal(off, -np.inf)
        drift = max(0.0, float(np.max(off)))

        # 후반엔 acc 페널티 완화 → over-filter 억제를 줄이고 임베딩항이 주도
        iter_frac = min(float(getattr(self.args, "iter", 0))
                        / max(float(getattr(self.args, "epochs", 1)), 1.0), 1.0)
        w_acc_eff = self.w_acc * (1.0 - self.acc_decay * iter_frac)
        if self.reward_mode == "acc":
            reward = w_acc_eff * acc_adv                       # accuracy 만
        else:
            reward = (w_acc_eff * acc_adv + self.w_anchor * anchor_adv - self.w_drift * drift)
        return float(reward), cand

    # ------------------------------------------------------------------ #
    def _dump_relmats(self, relmats, n_malicious, pert=None, clean_accs=None, round_idx=None,
                      protos=None):
        """매 라운드 전체 클라 (clean+flip+회전) 관계행렬 + 악성마스크 + 라운드idx 누적 → npz(덮어쓰기).
        clean은 'relmats' 키(하위호환), 나머지는 'pert_<flip|rot45|...>' 키로 저장.
        viz 가 라운드별 표준화/median편차 재계산 후 정상/악성 히트맵 생성."""
        R = np.asarray(relmats, dtype=np.float32)                 # (n, K*K)
        n = R.shape[0]
        nm = int(n_malicious)
        is_mal = np.array([1] * nm + [0] * (n - nm), dtype=np.int64)  # w_locals[:nm]=악성
        if not hasattr(self, "_dump_buf"):
            self._dump_buf = {"relmats": [], "is_mal": [], "round": [], "clean_acc": [],
                              "protos": [], "cid": [], "dw": [], "dR": []}
            self._dump_pert = {}
            self._dump_ctr = 0
        self._dump_buf["relmats"].append(R)
        for _k in ("dw", "dR"):                                   # 파라미터/표현 이동량 (연결 진단)
            _v = getattr(self, "_dump_%s" % _k, None)
            self._dump_buf[_k].append(np.asarray(_v, dtype=np.float32) if _v is not None
                                      else np.full(n, np.nan, dtype=np.float32))
        _ids = getattr(self.args, "_round_client_ids", None)      # 이번 라운드 클라 id (정체 혼입 검증용)
        self._dump_buf["cid"].append(np.asarray(_ids, dtype=np.int64) if _ids is not None
                                     else np.full(n, -1, dtype=np.int64))
        if protos is not None:                                    # 원본 프로토타입(K*D)
            self._dump_buf["protos"].append(np.asarray(protos, dtype=np.float32))
        self._dump_buf["is_mal"].append(is_mal)
        actual_round = int(round_idx if round_idx is not None else self._dump_ctr + 1)
        self._dump_buf["round"].append(np.full(n, actual_round, dtype=np.int64))
        if clean_accs is not None:
            self._dump_buf["clean_acc"].append(np.asarray(clean_accs, dtype=np.float32))
        if pert:
            for k in pert[0].keys():
                self._dump_pert.setdefault(k, []).append(
                    np.stack([p[k] for p in pert], 0).astype(np.float32))
        self._dump_ctr += 1
        dump_every = max(1, int(getattr(self.args, "pb_dump_every", 1)))
        final_round = actual_round >= int(getattr(self.args, "epochs", actual_round))
        if actual_round % dump_every != 0 and not final_round:
            return
        save = dict(relmats=np.concatenate(self._dump_buf["relmats"], 0),
                    is_mal=np.concatenate(self._dump_buf["is_mal"], 0),
                    round=np.concatenate(self._dump_buf["round"], 0),
                    K=int(round(np.sqrt(R.shape[1]))), round_base=1)
        for k, v in self._dump_pert.items():
            save["pert_" + k] = np.concatenate(v, 0)
        if self._dump_buf["clean_acc"]:
            save["clean_acc"] = np.concatenate(self._dump_buf["clean_acc"], 0)
        if self._dump_buf["protos"]:
            save["protos"] = np.concatenate(self._dump_buf["protos"], 0)
        if self._dump_buf["cid"]:
            save["cid"] = np.concatenate(self._dump_buf["cid"], 0)
        for _k in ("dw", "dR"):
            if self._dump_buf[_k]:
                save[_k] = np.concatenate(self._dump_buf[_k], 0)
        np.savez(self.args.pb_dump_relmat, **save)

    def observe(self, w_locals, n_malicious=0, round_idx=None):
        """FedAvg용 passive observer. 로컬 통계만 dump하고 선택·noise·aggregation은 하지 않는다."""
        if not getattr(self.args, "pb_dump_relmat", ""):
            raise ValueError("pb_observe=1 requires --pb_dump_relmat")
        relmats, pert_buf, clean_accs = [], [], []
        for w in w_locals:
            self.template.load_state_dict(w)
            self.template.to(self.device)
            pert = self._perturb_relmats(self.template)
            relmats.append(pert["clean"])
            pert_buf.append(pert)
            clean_accs.append(float(_accuracy(self.template, self.val_loader, self.device)))
        self._dump_relmats(relmats, n_malicious, pert_buf, clean_accs, round_idx=round_idx)

    # ------------------------------------------------------------------ #
    def _rule_aggregate(self, scores_by_type, w_locals, w_glob, n_malicious):
        """룰베이스: self.rule_dets 결합점수로 신뢰 band keep. 밴딧/학습 없음."""
        def _z(s):
            s = np.asarray(s, dtype=np.float64)
            return (s - np.median(s)) / (np.median(np.abs(s - np.median(s))) + EPS)
        dets = [d for d in self.rule_dets
                if d in scores_by_type and scores_by_type[d] is not None]
        if not dets:
            dets = ["relmat", "flip"]
        # aug(flip/rot) 부호는 backbone·공격에 따라 정/역이 뒤집힘. 처리 방식 3가지:
        #  pb_aug_tails=1: 각 aug를 두 꼬리로 분리 → pos=max(z,0), neg=max(-z,0) 둘 다 combined에.
        #                  도넛/컷이 각 꼬리를 독립적으로 봄(한쪽만 의미있을 때 반대꼬리 노이즈↓).
        #  pb_aug_abs=1  : |z| (양쪽을 한 점수로 합침).
        #  둘 다 0       : -z 고정(기존, 악성=작은변화 가정).
        aug_tails = bool(getattr(self.args, "pb_aug_tails", 0))
        aug_abs = bool(getattr(self.args, "pb_aug_abs", 0))
        # 자동가중(pb_aug_weight): detector마다 그 라운드 '분리력'으로 가중 → 무신호 detector
        #  가 좋은 신호를 희석/상쇄하는 문제 해소. 라벨 없이 계산.
        #  'gap' = (top1 - median)|z|,  'max' = max|z|,  'off'=균등(기존)
        wmode = getattr(self.args, "pb_aug_weight", "off")
        def _sep_weight(term):
            t = np.asarray(term, dtype=np.float64)
            if wmode == "gap":
                return max(np.max(t) - np.median(t), 0.0)
            if wmode == "max":
                return max(np.max(t), 0.0)
            return 1.0
        # detector별 항(term) + 이름 기록 (aug_abs일 때 relmat도 |z|)
        term_map = {}
        for d in dets:
            z = _z(scores_by_type[d])
            is_aug = d not in ("relmat", "reldist", "ratio", "axsum", "maha",
                               "maha_a", "maha_b", "anchor_dev")
            if is_aug and aug_tails:
                term_map[d + "+"] = np.maximum(z, 0.0)
                term_map[d + "-"] = np.maximum(-z, 0.0)
            elif is_aug and aug_abs:
                term_map[d] = np.abs(z)
            else:
                term_map[d] = z if not aug_abs else np.abs(z)
        keys = list(term_map)
        terms = [term_map[k] for k in keys]
        # Top-k 자동선택(pb_topk>0): detector별 gap을 EMA 누적 → 상위 k개만 사용(무신호 배제).
        topk = int(getattr(self.args, "pb_topk", 0))
        if topk > 0 and len(keys) > topk:
            if not hasattr(self, "_gap_ema"):
                self._gap_ema = {}
            for k, t in zip(keys, terms):
                g = max(float(np.max(t) - np.median(t)), 0.0)
                self._gap_ema[k] = 0.7 * self._gap_ema.get(k, g) + 0.3 * g  # EMA(안정화)
            top_keys = sorted(keys, key=lambda k: -self._gap_ema[k])[:topk]
            combined = sum(term_map[k] for k in top_keys)
        elif wmode != "off":
            ws = np.array([_sep_weight(t) for t in terms], dtype=np.float64)
            ws = ws / (ws.sum() + EPS) * len(terms)
            combined = sum(w * t for w, t in zip(ws, terms))
        else:
            combined = sum(terms)  # 균등 z합 (기존)
        n = len(combined)
        # 결합 방식: sum=z 합산 후 밴드(기존) / inter=detector별 밴드의 교집합.
        #  합산은 한 detector가 강해도 다른 detector가 반대로 흔들리면 상쇄된다.
        #  교집합은 "모든 detector가 신뢰한 클라"만 남겨 상쇄를 피한다.
        if str(getattr(self.args, "pb_combine", "sum")) == "inter" and len(terms) > 1:
            _dr = float(getattr(self.args, "pb_rule_drop", 0.0))
            _kp = float(getattr(self.args, "pb_rule_keep", 0.5))
            _sets = []
            for t in terms:
                _o = np.argsort(t); _lo = int(_dr * n); _hi = max(_lo + 1, int(_kp * n))
                _sets.append(set(_o[_lo:_hi].tolist()))
            _inter = set.intersection(*_sets)
            if not _inter:                                  # 공집합이면 합산 순위 최상위 1명
                _inter = {int(np.argsort(combined)[0])}
            sel = np.array(sorted(_inter), dtype=int)
            order = np.argsort(combined)                    # 로그용 순위는 합산 기준 유지
            lo, hi = 0, len(sel)
        else:
            order = np.argsort(combined)                    # 낮은(신뢰) 순
            lo = int(getattr(self.args, "pb_rule_drop", 0.0) * n)   # 도넛: 최상위 신뢰 버림
            # warmup: 초반은 더 깊게 잘라 탐지 lock-on 유도(양성 피드백의 좋은 basin 진입)
            _keep = float(getattr(self.args, "pb_rule_keep", 0.5))
            _wr = int(getattr(self.args, "pb_warmup_rounds", 0))
            if _wr > 0 and int(getattr(self.args, "iter", 0)) < _wr:
                _keep = float(getattr(self.args, "pb_warmup_keep", 0.3))
            hi = max(lo + 1, int(_keep * n))
            sel = order[lo:hi]                             # band-pass
        # 노이즈-클리핑 결합(FLAME/CRFL): 업데이트를 clip_q×median_norm 으로 자른 뒤 같은 스케일의
        # 노이즈를 부어야 "노이즈가 잔여 악성 기여를 지배한다"는 논증이 성립. 0/미지정이면 클리핑 없음.
        _cq = float(getattr(self.args, "pb_clip_q", 0.0) or 0.0)
        cand, _cv = self._clipped_fedavg(sel, w_locals, w_glob,
                                         (_cq if _cq > 0 else None), return_clip=True)
        cand = add_weight_noise(cand, w_glob, w_locals, self.noise,
                                mode=getattr(self.args, "pb_noise_mode", "norm"), clip_value=_cv)
        sel_set = set(int(i) for i in sel)
        mal_kept = sum(1 for i in range(n_malicious) if i in sel_set)
        # 악성의 combined 순위(0=가장 신뢰, n-1=가장 의심). 탐지력 추적용.
        _pos = {int(v): r for r, v in enumerate(order)}
        _mal_rank = [_pos.get(i, -1) for i in range(int(n_malicious))]
        print("[protorule] iter=%d dets=%s |sel|=%d/%d 악성통과=%d/%d 악성순위=%s/%d"
              % (getattr(self.args, "iter", -1), ",".join(dets),
                 len(sel), len(w_locals), mal_kept, n_malicious,
                 ",".join(str(r) for r in _mal_rank), len(w_locals) - 1))
        self.last_meta = {"defense/n_selected": float(len(sel))}
        if n_malicious > 0:
            self.last_meta["defense/mal_pass"] = float(mal_kept) / max(n_malicious, 1)
        return cand

    def aggregate(self, w_locals, w_glob, net_glob, n_malicious=0):
        need_rot = self.rule_mode and len(self.rot_dets) > 0
        need_aug = self.rule_mode and len(self.aug_dets) > 0
        dump_on = bool(getattr(self.args, "pb_dump_relmat", ""))
        # 경량 덤프: 프로토타입만 저장(이미 계산됨) → perturbation 11회 + accuracy 1회 forward 생략
        dump_light = dump_on and bool(getattr(self.args, "pb_dump_light", 0))
        # 1) 각 client 모델 통계
        proto_vecs, anchordev, flipexcess, relmats = [], [], [], []
        rotexcess = {d: [] for d in self.rot_dets}    # detector별 회전 변동 초과량
        flip_mats = []                                # perturb_peer: 클라별 flip 관계행렬
        rot_mats = {d: [] for d in self.rot_dets}     # perturb_peer: 클라별 회전 관계행렬
        aug_mats = {d: [] for d in self.aug_dets}     # perturb_peer: 클라별 aug 관계행렬
        augdeg = {d: [] for d in self.aug_dets}       # detector별 일반aug 변동량(invert 등)
        pert_buf = [] if (dump_on and not dump_light) else None       # 덤프 시 clean+flip+회전 relmat 수집
        clean_accs = [] if (dump_on and not dump_light) else None     # 덤프 시 클라별 clean accuracy(공격자 가정 검증)
        for w in w_locals:
            pv, rel, fd = self._stats_from_weights(w)   # w를 self.template에 로드
            proto_vecs.append(pv)
            anchordev.append(0.0 if self.S_A is None else float(np.linalg.norm(rel - self.S_A)))
            flipexcess.append(fd - self.anchor_flipdeg)
            relmats.append(rel.reshape(-1))   # 진단용: 관계행렬 전체(K*K)
            if need_rot:                                 # 로드된 self.template 재사용
                for d in self.rot_dets:
                    rotexcess[d].append(self._rotdeg(self.template, self.rot_angles[d])
                                        - self.anchor_rotdeg[d])
            if self.perturb_peer:                        # perturbed 행렬 또래비교용 수집
                flip_mats.append(self._flip_relmat_flat(self.template))
                for d in self.rot_dets:
                    rot_mats[d].append(self._rot_relmat_flat(self.template, self.rot_angles[d]))
            if need_aug:                                 # invert/gray/blur … detector
                for d in self.aug_dets:
                    if self.perturb_peer:                # 또래비교: 관계행렬 자체를 모음
                        aug_mats[d].append(self._aug_relmat_flat(self.template, self.aug_specs[d]))
                    else:                                # 자기차이: 변동량
                        augdeg[d].append(self._augdeg(self.template, self.aug_specs[d]))
            if dump_on and not dump_light:               # 로드된 self.template 재사용
                pert_buf.append(self._perturb_relmats(self.template))
                # 공격자 가정 검증: 악성도 clean 정확도는 정상과 같은가?
                clean_accs.append(float(_accuracy(self.template, self.val_loader, self.device)))

        # 2) per-client detector 점수 + state
        scores_by_type = self._client_scores(proto_vecs, anchordev, flipexcess, relmats,
                                             rotexcess=(rotexcess if need_rot else None))
        # 글로벌 관계행렬 기준 detector 2종 (기준점을 '학습 전 글로벌'로 고정 → 또래 흩어짐에 덜 취약)
        #  drift: 이동 '크기'  ‖R_i − R_glob‖      — LGA는 업데이트를 깎아 덜 움직임 → −drift
        #  ddir : 이동 '방향'  cos(d_i, median d)  — 정상은 같은 과제라 방향이 정렬, 백도어는 다른 방향 → −cos
        if ("drift" in self.rule_dets) or ("ddir" in self.rule_dets):
            self.template.load_state_dict(w_glob)
            self.template.to(self.device)
            _fg, _lg = _features(self.template, self.val_loader, self.device)
            _Rg = _relmat(_prototypes(_fg, _lg, self.classes)).reshape(-1)
            _Kc = int(round(np.sqrt(len(_Rg))))
            _iu0, _iu1 = np.triu_indices(_Kc, k=1)
            _cols = _iu0 * _Kc + _iu1
            _Dvec = np.asarray([np.asarray(r, dtype=np.float64)[_cols] - _Rg[_cols] for r in relmats])
            if "drift" in self.rule_dets:
                scores_by_type["drift"] = -np.linalg.norm(_Dvec, axis=1)
            if "ddir" in self.rule_dets:
                _U = _Dvec / (np.linalg.norm(_Dvec, axis=1, keepdims=True) + EPS)  # 크기 제거
                _ref = np.median(_U, axis=0)
                _ref = _ref / (np.linalg.norm(_ref) + EPS)
                scores_by_type["ddir"] = -(_U @ _ref)      # 또래 평균방향과 어긋날수록 의심
        # ratio: 표현 이탈 ‖F_i‖ 을 파라미터 이동 ‖Δw_i‖ 로 나눈 '기하 왜곡 효율'.
        #  데이터 이질성은 분자·분모를 같이 키워 약분되고, 공격 성격만 남는다.
        #   LGA  = 적게 움직이고 많이 틀어짐            → 비율 높음  ↑
        #   LPA  = 보통 움직이고 틀어짐                 → 비율 높음  ↑
        #   BadNet = 많이 움직이고 clean 기하는 그대로   → 비율 낮음  ↓
        #  양쪽 꼬리를 모두 잡아야 하므로 중앙값으로부터의 |log 편차|.
        if "ratio" in self.rule_dets:
            _rk = [k for k in w_glob if w_glob[k].dtype.is_floating_point
                   and not k.endswith(("running_mean", "running_var"))]
            _dw = np.asarray([float(sum(float(torch.sum((_w[k].float() - w_glob[k].float()) ** 2))
                                        for k in _rk) ** 0.5) for _w in w_locals], dtype=np.float64)
            _fn = np.asarray(scores_by_type["relmat"], dtype=np.float64)   # ‖F_i‖ = 또래 이탈
            _lr = np.log((_fn + EPS) / (_dw + EPS))
            scores_by_type["ratio"] = np.abs(_lr - np.median(_lr))
        # axsum: 표현 축과 파라미터 축을 각각 '양방향'으로 만든 뒤 합산.
        #   a = log‖F_i‖ (또래 기하 이탈)   b = log‖Δw_i‖ (파라미터 이동)
        #   score = |z(a)| + |z(b)| + |z(a−b)|
        #  뺄셈 하나(비율)로 접으면 '둘 다 큼'이 상쇄돼 보이지 않는다. 세 방향을 |z|로 두면
        #  부호가 통일돼 증거가 상쇄 대신 누적된다.
        #   LGA=b 매우 작음 / BadNet=b 큼 / LPA=b 흔적 없고 a 만 큼 → 세 공격이 서로 다른 축에 걸림
        if "axsum" in self.rule_dets:
            _ak = [k for k in w_glob if w_glob[k].dtype.is_floating_point
                   and not k.endswith(("running_mean", "running_var"))]
            _b = np.log(np.asarray([float(sum(float(torch.sum((_w[k].float() - w_glob[k].float()) ** 2))
                                              for k in _ak) ** 0.5) for _w in w_locals], dtype=np.float64) + EPS)
            _a = np.log(np.asarray(scores_by_type["relmat"], dtype=np.float64) + EPS)
            def _rzs(v):
                m = np.median(v)
                return (v - m) / (1.4826 * np.median(np.abs(v - m)) + EPS)
            scores_by_type["axsum"] = (np.abs(_rzs(_a)) + np.abs(_rzs(_b)) + np.abs(_rzs(_a - _b)))
        # maha: (표현 이탈, 파라미터 이동) 평면에서 또래 분포에 대한 마할라노비스 거리.
        #   x_i = (log‖F_i‖, log‖Δw_i‖).  세 공격이 평면의 서로 다른 방향에 놓이므로
        #   한 방향만 보는 스칼라(비율=뺄셈 등)는 반드시 하나를 놓친다. 마할라노비스는 전 방향을 덮는다.
        #   공분산은 가장 먼 2개를 빼고 추정 → 악성이 자기 방향 분산을 부풀리는 마스킹 차단.
        #   maha_a / maha_b 는 축 단독 ablation. 2D와 완전히 같은 중심화(median)·스케일(std)을
        #   쓰고 공분산 결합만 뺀 것이라, 차이가 곧 '두 축을 함께 보는 것'의 순수 기여분이다.
        if any(d in self.rule_dets for d in ("maha", "maha_a", "maha_b")):
            _hk = [k for k in w_glob if w_glob[k].dtype.is_floating_point
                   and not k.endswith(("running_mean", "running_var"))]
            _hb = np.log(np.asarray([float(sum(float(torch.sum((_w[k].float() - w_glob[k].float()) ** 2))
                                               for k in _hk) ** 0.5) for _w in w_locals], dtype=np.float64) + EPS)
            _ha = np.log(np.asarray(scores_by_type["relmat"], dtype=np.float64) + EPS)
            _X = np.stack([_ha, _hb], 1)
            _U = _X - np.median(_X, axis=0)
            if "maha_a" in self.rule_dets:                          # 표현 축 단독
                scores_by_type["maha_a"] = np.abs(_U[:, 0]) / (_U[:, 0].std() + EPS)
            if "maha_b" in self.rule_dets:                          # 파라미터 축 단독
                scores_by_type["maha_b"] = np.abs(_U[:, 1]) / (_U[:, 1].std() + EPS)
            if "maha" in self.rule_dets:
                _d0 = np.linalg.norm(_U / (_U.std(0) + EPS), axis=1)
                _kp = np.argsort(_d0)[:max(3, len(_ha) - 2)]        # trim 2
                _S = np.cov(_U[_kp].T)
                _S = _S + np.eye(2) * 1e-6 * max(float(np.trace(_S)), EPS)
                scores_by_type["maha"] = np.sqrt(np.maximum(
                    np.einsum("ij,jk,ik->i", _U, np.linalg.inv(_S), _U), 0.0))
        # unorm: 업데이트 크기(‖w_i − w_glob‖). LGA는 은닉하려 업데이트를 깎아 비정상적으로 작아짐.
        # 부호는 "작을수록 의심"이라 −norm 을 점수로(다른 detector와 방향 통일).
        if "unorm" in self.rule_dets:
            _fk = [k for k in w_glob if w_glob[k].dtype.is_floating_point
                   and not k.endswith(("running_mean", "running_var"))]
            _un = []
            for _w in w_locals:
                _sq = 0.0
                for k in _fk:
                    _sq += float(torch.sum((_w[k].float() - w_glob[k].float()) ** 2))
                _un.append(_sq ** 0.5)
            scores_by_type["unorm"] = -np.asarray(_un, dtype=np.float64)
        if self.perturb_peer:
            # 자기차이(clean-perturbed) 대신 perturbed 행렬 자체를 또래비교 (리뷰어 반박 ablation).
            # relmat(clean 또래)은 _client_scores가 이미 동일 방식으로 계산 → 그대로 둠.
            if "flip" in self.rule_dets:
                scores_by_type["flip"] = self._peer_score(flip_mats)
            for d in self.rot_dets:
                scores_by_type[d] = self._peer_score(rot_mats[d])
        if need_aug:  # aug detector 점수 (peer=또래 이상치, 아니면 자기차이 변동량)
            for d in self.aug_dets:
                scores_by_type[d] = (self._peer_score(aug_mats[d]) if self.perturb_peer
                                     else np.asarray(augdeg[d], dtype=np.float64))
        if dump_on:
            # 연결 진단: 파라미터 이동량 ‖Δw‖ 와 표현 이동량 ‖R_i − R_glob‖.
            #  비율 ‖ΔR‖/‖Δw‖ 는 데이터 이질성(둘 다 키움)이 약분되고 공격 성격만 남는다.
            #  LGA=적게 움직이고 많이 틀어짐(높음) / BadNet=많이 움직이고 안 틀어짐(낮음)
            _fk = [k for k in w_glob if w_glob[k].dtype.is_floating_point
                   and not k.endswith(("running_mean", "running_var"))]
            self._dump_dw = [float(sum(float(torch.sum((wl[k].float() - w_glob[k].float()) ** 2))
                                       for k in _fk) ** 0.5) for wl in w_locals]
            self.template.load_state_dict(w_glob); self.template.to(self.device)
            _fg, _lg = _features(self.template, self.val_loader, self.device)
            _Rg = _relmat(_prototypes(_fg, _lg, self.classes)).reshape(-1)
            _Kc = int(round(np.sqrt(len(_Rg)))); _i0, _i1 = np.triu_indices(_Kc, k=1)
            _cl = _i0 * _Kc + _i1
            self._dump_dR = [float(np.linalg.norm(np.asarray(r, dtype=np.float64)[_cl] - _Rg[_cl]))
                             for r in relmats]
        # (진단) relmat 덤프: 매 라운드 전체 클라 (clean+flip+회전) 관계행렬 + 악성마스크 누적 저장
        if dump_on:
            self._dump_relmats(relmats, n_malicious, pert_buf, clean_accs,
                               round_idx=int(getattr(self.args, "iter", 0)) + 1,
                               protos=(proto_vecs if getattr(self.args, "pb_dump_protos", 0) else None))
        if self.rule_mode:
            return self._rule_aggregate(scores_by_type, w_locals, w_glob, n_malicious)
        state = self._build_state(scores_by_type, len(w_locals))

        # 3) baseline(FedAvg-all) 통계 한 번
        base_w = FedAvg(w_locals)
        self.template.load_state_dict(base_w)
        self.template.to(self.device)
        acc_ref = _accuracy(self.template, self.val_loader, self.device)
        _, rel_ref, fd_ref = self._model_stats(self.template)
        baseline = (acc_ref, rel_ref, fd_ref)

        # 4) 라운드당 evals_per_round 개 액션 샘플링 평가 → 밴딧 학습
        #    (Thompson 선택 우선 + 부족분 랜덤으로 채워 distinct 확보)
        n_eval = min(self.evals_per_round, len(self.actions))
        seen, cand_actions = set(), []
        for _ in range(n_eval * 4):
            if len(cand_actions) >= n_eval:
                break
            a = self.bandit.select(state, explore=True)
            if a not in seen:
                seen.add(a)
                cand_actions.append(a)
        rest = [a for a in range(len(self.actions)) if a not in seen]
        np.random.shuffle(rest)
        cand_actions += rest[: n_eval - len(cand_actions)]
        evaluated = {}
        for a in cand_actions:
            sel = self._select_clients(self.actions[a], scores_by_type)
            r, cand = self._reward(sel, self.actions[a], w_locals, w_glob, baseline)
            self.bandit.update(state, a, r)
            evaluated[a] = (r, cand, sel)

        # 5) 커밋: greedy best 룰로 최종 집계
        a_star = self.bandit.select(state, explore=False)
        if a_star in evaluated:
            r, cand, sel = evaluated[a_star]
        else:
            sel = self._select_clients(self.actions[a_star], scores_by_type)
            r, cand = self._reward(sel, self.actions[a_star], w_locals, w_glob, baseline)

        act = self.actions[a_star]
        print("[protobandit] iter=%d action=%s |sel|=%d/%d acc_ref=%.2f reward=%.4f"
              % (getattr(self.args, "iter", -1), act, len(sel), len(w_locals), acc_ref, r))

        # 매 라운드 방어 액션 기록 (main_fed 가 wandb 로 찍음)
        self.last_meta = {
            "defense/score_is_flip": 1.0 if act.score_type == "flip" else 0.0,
            "defense/clip": 0.0 if act.clip_q is None else float(act.clip_q),
            "defense/keep_outer": float(act.keep_outer),
            "defense/drop_inner": float(act.drop_inner),
            "defense/n_selected": float(len(sel)),
            "defense/reward": float(r),
            "defense/action_id": float(a_star),
            "defense/noise_on": 1.0 if (self.noise and self.noise > 0) else 0.0,
        }

        # ----- 진단: 악성 vs 정상 신호 분리도 (시뮬레이터가 악성 인덱스를 앎) -----
        #   w_locals[:n_malicious] = 악성, 나머지 = 정상 (main_fed 의 주입 순서)
        if n_malicious > 0:
            n = len(w_locals)
            mal = list(range(n_malicious))
            ben = list(range(n_malicious, n))
            x = np.asarray(proto_vecs, dtype=np.float64)
            center = np.median(x, axis=0)
            l2c = np.linalg.norm(x - center, axis=1)
            ad = np.asarray(anchordev)
            fe = np.asarray(flipexcess)
            sel_set = set(int(i) for i in sel)

            def _summ(name, vals):
                vals = np.asarray(vals)
                # 악성의 순위(오름차순; 낮을수록 '신뢰'로 보여 통과됨). rank 0 = 가장 정상스러움
                order = np.argsort(vals)
                rank = {int(idx): r for r, idx in enumerate(order)}
                mal_ranks = [rank[i] for i in mal]
                return ("%s: mal=%s(mean %.4f) | ben mean %.4f max %.4f | mal_rank %s/%d"
                        % (name,
                           np.round([vals[i] for i in mal], 4).tolist(), float(np.mean(vals[mal])),
                           float(np.mean(vals[ben])) if ben else 0.0,
                           float(np.max(vals[ben])) if ben else 0.0,
                           mal_ranks, n))

            mal_kept = [i for i in mal if i in sel_set]
            print("[diag] iter=%d n_mal=%d/%d | action=%s | 악성 선택통과=%d/%d"
                  % (getattr(self.args, "iter", -1), n_malicious, n,
                     self.actions[a_star].score_type, len(mal_kept), n_malicious))
            print("       " + _summ("relmat      ", scores_by_type["relmat"]))  # ← 강한 신호
            print("       " + _summ("l2_to_center", l2c))
            print("       " + _summ("anchor_dev  ", ad))
            print("       " + _summ("flip(-excess)", scores_by_type["flip"]))


            # 임베딩 공간 분리도 측정용: client별 (관계행렬 100-d + 스칼라들) + 라벨 덤프
            if not hasattr(self, "_diag_rows"):
                self._diag_rows = []
            it = int(getattr(self.args, "iter", -1))
            rm = np.asarray(relmats)            # (n, K*K)
            for i in range(n):
                label = 1 if i < n_malicious else 0
                self._diag_rows.append(np.concatenate(
                    [[it, label, ad[i], l2c[i], fe[i]], rm[i]]).astype(np.float32))
            np.save("/tmp/pb_diag_feats_%s.npy" % getattr(self.args, "attack", "x"),
                    np.stack(self._diag_rows))
            self.last_meta["defense/mal_pass"] = float(len(mal_kept)) / max(n_malicious, 1)
        cand = add_weight_noise(cand, w_glob, w_locals, self.noise)  # 배수구(노이즈)
        return cand
