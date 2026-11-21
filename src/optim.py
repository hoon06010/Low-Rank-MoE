"""Spectron Algorithm 1 adapted to W=B@A routed FFNs, with auxiliary AdamW.

Source: https://arxiv.org/html/2602.12429 (Algorithms 1-3).
This is a paper-algorithm implementation, not an exact reproduction of the
upstream training recipe. Finite NS/power iterations give approximate spectral
control, not a certified bound. See docs/optimizer_stability.md.
"""
from __future__ import annotations

import math
import torch
from torch import nn

from src.model import FactorizedLinear


# 역할: 반복 계산으로 업데이트 행렬의 특이값 크기를 고르게 만든다.
def orthogonalize(matrix, steps=5):
    """Quintic Newton-Schulz in FP32 (FP64 retained for reference tests)."""
    x = matrix.to(dtype=torch.float64 if matrix.dtype == torch.float64 else torch.float32)
    # 짧은 축이 행이 되게 전치하면 아래 x @ x.T의 계산량이 줄어든다.
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    # 반복 계산의 시작 크기를 맞춘다. 작은 상수는 영행렬의 0 나눗셈을 막는다.
    x = x / (torch.linalg.vector_norm(x) + 1e-7)
    # 논문의 다항식을 반복해 업데이트 방향의 특이값 크기를 고르게 만든다.
    # 유한 횟수의 근사이므로 정확한 직교행렬이나 엄밀한 상한을 보장하지 않는다.
    for _ in range(steps):
        gram = x @ x.T
        x = 3.4445 * x + (-4.775 * gram + 2.0315 * (gram @ gram)) @ x
    return x.T if transposed else x


# 역할: 이전 방향을 재사용해 행렬의 최대 확대 비율과 그 방향을 추정한다.
def power_norm(matrix, vector, steps=1):
    """Warm-started left singular vector; handle zero/null-space starts safely."""
    # 지난 step의 방향에서 시작해 행렬이 벡터를 가장 크게 늘리는 방향을 추정한다.
    u = vector / vector.norm().clamp_min(1e-12)
    for _ in range(steps):
        v = matrix.T @ u
        if v.norm() <= 1e-12:
            # 이전 방향이 현재 행렬에서 0이 되면 크기가 가장 큰 열에서 다시 시작한다.
            # 난수를 쓰지 않아 학습의 난수 순서도 바뀌지 않는다.
            u = matrix[:, matrix.square().sum(dim=0).argmax()]
            u = u / u.norm().clamp_min(1e-12)
            v = matrix.T @ u
        v = v / v.norm().clamp_min(1e-12)
        u = matrix @ v
        u = u / u.norm().clamp_min(1e-12)
    return torch.dot(u, matrix @ v).abs(), u


# 역할: 저랭크 A/B 쌍을 직교화한 방향과 공통 보폭으로 갱신한다.
class Spectron(torch.optim.Optimizer):
    """One parameter group per (A, B) pair, sharing the pre-update denominator.

    Optional decoupled factor decay is outside the paper's update bound; the
    Algorithm 1 default is zero decay. None/None skips a pair; a partial pair
    is an error. The trainer supplies zero gradients to idle experts explicitly.
    """
    # 역할: A/B 쌍과 설정을 검사하고 쌍마다 갱신 그룹을 만든다.
    def __init__(self, pairs, lr=.0003, momentum=.95, ns_steps=5, power_steps=1, weight_decay=0.):
        if not math.isfinite(lr) or lr <= 0 or not 0 <= momentum < 1:
            raise ValueError("invalid Spectron learning rate or momentum")
        if any(type(n) is not int or n <= 0 for n in (ns_steps, power_steps)):
            raise ValueError("Spectron iteration counts must be positive integers")
        if not math.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("invalid Spectron weight decay")
        groups = []
        for a, b in pairs:
            if a.ndim != 2 or b.ndim != 2 or b.shape[1] != a.shape[0]:
                raise ValueError("Spectron requires W=B@A factor pairs")
            if a.dtype != b.dtype or a.device != b.device or a.dtype not in (torch.float32, torch.float64):
                raise ValueError("Spectron requires matching FP32 or FP64 master factor pairs")
            groups.append({"params": [a, b]})
        if not groups:
            raise ValueError("Spectron requires at least one trainable factor pair")
        super().__init__(groups, dict(lr=lr, momentum=momentum, ns_steps=ns_steps,
                                     power_steps=power_steps, weight_decay=weight_decay))

    # 역할: momentum과 factor 크기로 방향·보폭을 구해 한 step을 수행한다.
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        # A와 B는 한 쌍이다. 한쪽 gradient만 있으면 어떤 파라미터도 바꾸기 전에 중단한다.
        for group in self.param_groups:
            a, b = group["params"]
            if (a.grad is None) != (b.grad is None):
                raise ValueError("Spectron factor pair has only one gradient")
        for group in self.param_groups:
            a, b = group["params"]
            if a.grad is None:
                continue
            directions, norms = [], []
            for p in (a, b):
                state = self.state[p]
                if not state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                    # 별도 난수 생성기를 써서 모델·데이터에 쓰는 난수 상태를 보존한다.
                    generator = torch.Generator(device="cpu").manual_seed(0)
                    state["power_vector"] = torch.randn(p.shape[0], generator=generator, dtype=p.dtype).to(p.device)
                    state["step"] = 0
                state["step"] += 1
                # 이전 gradient의 이동평균을 구한 뒤, 그 방향을 직교화한다.
                state["momentum_buffer"].lerp_(p.grad, 1 - group["momentum"])
                directions.append(orthogonalize(state["momentum_buffer"], group["ns_steps"]))
                sigma, u = power_norm(p, state["power_vector"], group["power_steps"])
                state["power_vector"].copy_(u)
                state["factor_spectral_estimate"] = sigma
                norms.append(sigma)
            # W = B @ A이므로 두 factor의 크기를 함께 보고 공통 보폭 rho를 정한다.
            # A와 B 모두 갱신 전 크기를 사용해야 갱신 순서에 따라 보폭이 달라지지 않는다.
            rho = group["lr"] / (norms[0] + norms[1] + 1)
            for p, direction in zip((a, b), directions):
                # 선택적 weight decay는 별도 축소다. 논문 Algorithm 1에 맞춘 기본값은 0이다.
                p.mul_(1 - group["lr"] * group["weight_decay"])
                update = rho * direction
                p.sub_(update)
                self.state[p]["rho"] = rho.clone()
                self.state[p]["update_frobenius_norm"] = update.norm()
        return loss


# 저랭크 A/B는 Spectron, 나머지는 AdamW로 갱신한다.
# 학습 루프에서는 하나의 optimizer처럼 학습률 조정·저장·복원을 호출한다.
# 역할: factor용 Spectron과 나머지 파라미터용 AdamW를 하나의 인터페이스로 묶는다.
class SpectronAdamW:
    """Disjoint optimizers with one schedule interface and complete saved state."""
    # 역할: 함께 사용할 두 optimizer를 보관한다.
    def __init__(self, spectron, adamw):
        self.spectron, self.adamw = spectron, adamw

    # 역할: 학습률을 함께 설정할 수 있도록 두 optimizer의 파라미터 그룹을 반환한다.
    @property
    def param_groups(self):
        return self.spectron.param_groups + self.adamw.param_groups

    # 역할: 두 optimizer가 관리하는 gradient를 초기화한다.
    def zero_grad(self, set_to_none=True):
        self.spectron.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    # 역할: 두 optimizer가 각자 담당하는 파라미터를 한 번씩 갱신한다.
    def step(self):
        self.spectron.step()
        self.adamw.step()

    # 역할: 두 optimizer의 내부 상태를 저장용 딕셔너리로 반환한다.
    def state_dict(self):
        return {"optimizer_type": "spectron_adamw_v1", "spectron": self.spectron.state_dict(),
                "adamw": self.adamw.state_dict()}

    # 역할: 저장 형식을 확인하고 두 optimizer의 내부 상태를 복원한다.
    def load_state_dict(self, state):
        if state.get("optimizer_type") != "spectron_adamw_v1":
            raise ValueError("Spectron optimizer state type mismatch")
        self.spectron.load_state_dict(state["spectron"])
        self.adamw.load_state_dict(state["adamw"])


# 역할: 설정에 따라 AdamW 또는 파라미터를 나눠 맡는 Spectron·AdamW 조합을 만든다.
def build_optimizer(model, config):
    pairs = []
    if config["optimizer"] == "spectron":
        for module in model.modules():
            if isinstance(module, FactorizedLinear):
                if module.A.requires_grad != module.B.requires_grad:
                    raise ValueError("Spectron requires both factors of a pair to be trainable")
                if module.A.requires_grad:
                    pairs.append((module.A, module.B))
        if not pairs:
            raise ValueError("Spectron optimizer requires trainable factorized layers")
    # 같은 파라미터가 두 optimizer에서 중복 갱신되지 않게 A/B를 제외한다.
    # AdamW 설정에서는 pairs가 비어 있으므로 모든 학습 파라미터가 아래로 간다.
    factor_ids = {id(p) for pair in pairs for p in pair}
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in factor_ids:
            continue
        # bias, 정규화 계수, embedding에는 weight decay를 적용하지 않는다.
        excluded = p.ndim < 2 or "norm" in name.lower() or "embedding" in name.lower() or name.endswith("bias")
        (no_decay if excluded else decay).append(p)
    adamw = torch.optim.AdamW(
        [{"params": decay, "weight_decay": config["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.}],
        lr=config["learning_rate"], betas=tuple(config["betas"]), eps=config["eps"],
    )
    if not pairs:
        return adamw
    spectron = Spectron(pairs, lr=config["learning_rate"], momentum=config["spectron_momentum"],
                        ns_steps=config["spectron_ns_steps"], power_steps=config["spectron_power_steps"],
                        weight_decay=config["spectron_weight_decay"])
    return SpectronAdamW(spectron, adamw)
