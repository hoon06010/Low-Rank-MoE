"""Small, explicit PyTorch implementation of the M3 MoE transformer."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.budget import validate_config


# 역할: 기본 seed와 모듈 이름으로 재현 가능한 모듈별 seed를 만든다.
def _seed_for(seed: int, name: str) -> int:
    payload = f"{seed}:{name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


# 역할: 모듈별 seed를 사용하는 독립 CPU 난수 생성기를 만든다.
def _generator(seed: int, name: str) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_seed_for(seed, name))
    return generator


# 역할: 모듈 전용 난수로 텐서를 지정 표준편차의 정규분포 값으로 초기화한다.
def _normal_(tensor: Tensor, std: float, seed: int, name: str) -> None:
    with torch.no_grad():
        tensor.normal_(mean=0.0, std=std, generator=_generator(seed, name))


# 역할: 토큰 벡터를 제곱평균 크기로 정규화하고 학습 가능한 배율을 곱한다.
class RMSNorm(nn.Module):
    # 역할: 차원별 배율과 0 나눗셈 방지용 상수를 준비한다.
    def __init__(self, width: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    # 역할: 입력의 RMS를 구해 크기를 정규화하고 차원별 배율을 적용한다.
    def forward(self, x: Tensor) -> Tensor:
        inv_rms = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * inv_rms.to(dtype=x.dtype)) * self.weight


# 역할: 큰 선형 변환을 두 작은 행렬 A/B의 곱으로 표현한다.
class FactorizedLinear(nn.Module):
    """Bias-free linear map with W=B@A and no intermediate nonlinearity."""

    # 역할: 입출력 크기와 rank를 검사하고 학습 행렬 A/B를 만든다.
    def __init__(self, in_features: int, out_features: int, rank: int) -> None:
        super().__init__()
        if min(in_features, out_features, rank) <= 0 or rank > min(in_features, out_features):
            raise ValueError("factor rank and dimensions are invalid")
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.A = nn.Parameter(torch.empty(rank, in_features))
        self.B = nn.Parameter(torch.empty(out_features, rank))

    # 역할: 입력 크기와 rank에 맞춘 분산으로 A/B를 초기화한다.
    def reset_parameters(self, seed: int, name: str, depth_scale: float = 1.0) -> None:
        _normal_(self.A, 1.0 / math.sqrt(self.in_features), seed, f"{name}.A")
        _normal_(self.B, depth_scale / math.sqrt(self.rank), seed, f"{name}.B")

    # 역할: 입력을 A로 rank 차원에 보낸 뒤 B로 출력 차원에 보낸다.
    def forward(self, x: Tensor) -> Tensor:
        # 입력 → rank 차원(A) → 출력 차원(B). 중간 활성화 없이 W = B @ A를 구현한다.
        return F.linear(F.linear(x, self.A), self.B)


# 역할: gate와 up의 곱을 down으로 변환하는 FFN으로 저랭크·일반 행렬을 지원한다.
class SwiGLU(nn.Module):
    # 역할: 실험 family에 따라 gate·up·down을 저랭크 또는 일반 선형층으로 만든다.
    def __init__(self, d_model: int, width: int, family: str, rank: int = 0) -> None:
        super().__init__()
        if family == "LR":
            self.gate = FactorizedLinear(d_model, width, rank)
            self.up = FactorizedLinear(d_model, width, rank)
            self.down = FactorizedLinear(width, d_model, rank)
        elif family == "DW" or family == "SHARED_ONLY":
            self.gate = nn.Linear(d_model, width, bias=False)
            self.up = nn.Linear(d_model, width, bias=False)
            self.down = nn.Linear(width, d_model, bias=False)
        else:
            raise ValueError(f"unknown SwiGLU family: {family}")
        self.d_model = d_model
        self.width = width
        self.family = family
        self.rank = rank

    # 역할: 세 변환을 초기화하고 출력 변환인 down에는 깊이 보정을 적용한다.
    def reset_parameters(self, seed: int, name: str, depth_scale: float) -> None:
        for projection_name in ("gate", "up", "down"):
            projection = getattr(self, projection_name)
            layer_scale = depth_scale if projection_name == "down" else 1.0
            if isinstance(projection, FactorizedLinear):
                projection.reset_parameters(seed, f"{name}.{projection_name}", layer_scale)
            else:
                _normal_(
                    projection.weight,
                    layer_scale / math.sqrt(projection.in_features),
                    seed,
                    f"{name}.{projection_name}.weight",
                )

    # 역할: SiLU를 적용한 gate와 up 출력을 곱한 뒤 down으로 출력을 만든다.
    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


# 역할: 모든 토큰이 사용하는 여러 shared expert의 출력을 합친다.
class SharedExpertGroup(nn.Module):
    """Always-active fixed-width SwiGLUs with summed outputs."""

    # 역할: 지정한 폭과 개수의 shared expert를 만든다.
    def __init__(self, d_model: int, width: int, count: int) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            SwiGLU(d_model, width, "SHARED_ONLY") for _ in range(count)
        )

    # 역할: 각 shared expert를 서로 다른 이름의 seed로 초기화한다.
    def reset_parameters(self, seed: int, name: str, depth_scale: float) -> None:
        for index, expert in enumerate(self.experts):
            expert.reset_parameters(seed, f"{name}.expert.{index}", depth_scale)

    # 역할: 모든 shared expert에 같은 입력을 넣고 출력을 합한다.
    def forward(self, x: Tensor) -> Tensor:
        return torch.stack([expert(x) for expert in self.experts]).sum(dim=0)


# 역할: 토큰별 expert를 선택하는 router와 shared·routed expert들을 보관한다.
class RoutedMoE(nn.Module):
    # 역할: router, 항상 사용하는 shared expert, 선택해서 사용하는 routed expert를 만든다.
    def __init__(
        self,
        d_model: int,
        n_experts: int,
        top_k: int,
        routed_width: int,
        family: str,
        rank: int,
        shared_width: int,
        shared_experts: int = 1,
    ) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.shared = (SwiGLU(d_model, shared_width, "SHARED_ONLY") if shared_experts == 1
                       else SharedExpertGroup(d_model, shared_width, shared_experts))
        self.experts = nn.ModuleList(
            SwiGLU(d_model, routed_width, family, rank if family == "LR" else 0)
            for _ in range(n_experts)
        )

    # 역할: router와 shared·routed expert를 모듈별 seed로 초기화한다.
    def reset_parameters(self, seed: int, name: str, depth_scale: float) -> None:
        _normal_(self.router.weight, 1.0 / math.sqrt(self.router.in_features), seed, f"{name}.router.weight")
        self.shared.reset_parameters(seed, f"{name}.shared", depth_scale)
        for index, expert in enumerate(self.experts):
            expert.reset_parameters(seed, f"{name}.expert.{index}", depth_scale)

    # 역할: 유효 토큰을 top-k expert에 보내고 가중 출력, 균형 손실, 선택 통계를 반환한다.
    def forward(self, x: Tensor, valid_mask: Tensor, balance_fractions: Tensor | None = None) -> tuple[Tensor, Tensor, dict[str, Any]]:
        batch, sequence, d_model = x.shape
        flat_x = x.reshape(batch * sequence, d_model)
        flat_valid = valid_mask.reshape(-1).to(device=x.device, dtype=torch.bool)
        valid_positions = flat_valid.nonzero(as_tuple=False).squeeze(-1)
        # padding은 expert 선택과 균형 손실 통계에서 제외한다.
        valid_x = flat_x.index_select(0, valid_positions)

        if valid_x.shape[0] == 0:
            zero = x.sum() * 0.0
            return torch.zeros_like(x), zero, {
                "valid_token_count": 0,
                "selected_count": [0] * self.n_experts,
                "mean_probability": [0.0] * self.n_experts,
                "selected_gate_sums": [],
            }

        probabilities = F.softmax(F.linear(valid_x.float(), self.router.weight.float()), dim=-1)
        top_probabilities, top_experts = torch.topk(probabilities, self.top_k, dim=-1)
        # 선택된 expert끼리 가중치 합이 1이 되도록 다시 나눈다.
        top_gates = top_probabilities / top_probabilities.sum(dim=-1, keepdim=True)

        routed = torch.zeros_like(valid_x)
        selected_counts = torch.bincount(top_experts.reshape(-1), minlength=self.n_experts)
        for expert_index, expert in enumerate(self.experts):
            token_rows, gate_columns = (top_experts == expert_index).nonzero(as_tuple=True)
            if token_rows.numel() == 0:
                continue
            expert_output = expert(valid_x.index_select(0, token_rows))
            weighted = expert_output * top_gates[token_rows, gate_columns].to(expert_output.dtype).unsqueeze(-1)
            routed = routed.index_add(0, token_rows, weighted)

        fractions = selected_counts.to(probabilities.dtype) / (self.top_k * valid_x.shape[0])
        # trainer가 전체 step의 선택 비율을 주면 현재 작은 배치의 비율 대신 사용한다.
        if balance_fractions is not None:
            if balance_fractions.shape != (self.n_experts,):
                raise ValueError("router balance fractions must match expert count")
            fractions = balance_fractions.to(device=x.device, dtype=probabilities.dtype).detach()
        mean_probability = probabilities.mean(dim=0)
        # 선택 횟수는 상수로 두고 확률 쪽으로만 gradient를 흘린다.
        # 많이 선택된 expert의 확률에 더 큰 비용을 줘 선택 쏠림을 줄인다.
        aux_loss = self.n_experts * torch.sum(fractions.detach() * mean_probability)

        dense_routed = torch.zeros_like(flat_x).index_copy(0, valid_positions, routed).view(batch, sequence, d_model)
        stats = {
            "valid_token_count": int(valid_x.shape[0]),
            "selected_count": selected_counts.detach().cpu().tolist(),
            "mean_probability": mean_probability.detach().cpu().tolist(),
            "selected_gate_sums": top_gates.detach().sum(dim=-1).cpu().tolist(),
        }
        return dense_routed, aux_loss, stats


# 역할: 위치에 따라 좌표 쌍을 회전시켜 query와 key에 토큰 위치 정보를 넣는다.
def apply_rope(x: Tensor, positions: Tensor, base: float = 10_000.0) -> Tensor:
    """Rotate adjacent even/odd coordinates; x is [B,H,T,head_dim]."""
    head_dim = x.shape[-1]
    if head_dim % 2:
        raise ValueError("RoPE requires an even head dimension")
    inv_frequency = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=x.device, dtype=torch.float32) / head_dim))
    angles = positions.to(device=x.device, dtype=torch.float32)[:, None] * inv_frequency[None, :]
    cosine = angles.cos()[None, None, :, :].to(dtype=x.dtype)
    sine = angles.sin()[None, None, :, :].to(dtype=x.dtype)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1).flatten(-2)


# 역할: 미래 토큰을 가리고 RoPE 위치 정보를 사용하는 다중 head attention이다.
class CausalSelfAttention(nn.Module):
    # 역할: head 크기를 검사하고 query·key·value·출력 변환층을 만든다.
    def __init__(self, d_model: int, heads: int) -> None:
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        self.d_model = d_model
        self.heads = heads
        self.head_dim = d_model // heads
        if self.head_dim % 2:
            raise ValueError("RoPE requires an even head dimension")
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    # 역할: 변환 행렬을 초기화하고 출력 변환에는 깊이 보정을 적용한다.
    def reset_parameters(self, seed: int, name: str, depth_scale: float) -> None:
        for key in ("q_proj", "k_proj", "v_proj", "o_proj"):
            projection = getattr(self, key)
            scale = depth_scale if key == "o_proj" else 1.0
            _normal_(projection.weight, scale / math.sqrt(self.d_model), seed, f"{name}.{key}.weight")

    # 역할: 위치 정보를 넣고 미래·padding key를 가려 attention 출력을 계산한다.
    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        batch, sequence, _ = x.shape
        q = self.q_proj(x).view(batch, sequence, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, sequence, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, sequence, self.heads, self.head_dim).transpose(1, 2)
        positions = torch.arange(sequence, device=x.device)
        q = apply_rope(q, positions)
        k = apply_rope(k, positions)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        allowed = torch.ones(sequence, sequence, device=x.device, dtype=torch.bool).tril()
        allowed = allowed[None, None, :, :]
        if attention_mask is not None:
            if attention_mask.shape != (batch, sequence):
                raise ValueError("attention_mask must have shape [B,T]")
            allowed = allowed & attention_mask.to(dtype=torch.bool, device=x.device)[:, None, None, :]
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores.float(), dim=-1).to(dtype=scores.dtype)
        attended = torch.matmul(weights, v).transpose(1, 2).contiguous().view(batch, sequence, self.d_model)
        return self.o_proj(attended)


# 역할: 정규화, attention, expert FFN을 잔차 연결로 묶은 transformer 한 층이다.
class TransformerBlock(nn.Module):
    # 역할: 설정에 따라 attention과 shared 전용 또는 routed MoE FFN을 만든다.
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        d_model = config["d_model"]
        self.attention_norm = RMSNorm(d_model)
        self.attention = CausalSelfAttention(d_model, config["heads"])
        self.ffn_norm = RMSNorm(d_model)
        self.family = config["family"]
        if self.family == "SHARED_ONLY":
            self.moe = None
            count = config.get("shared_experts", 1)
            self.shared = (SwiGLU(d_model, config["shared_width"], "SHARED_ONLY") if count == 1
                           else SharedExpertGroup(d_model, config["shared_width"], count))
        else:
            self.moe = RoutedMoE(
                d_model=d_model,
                n_experts=config["routed_experts"],
                top_k=config["top_k"],
                routed_width=config["routed_width"],
                family=self.family,
                rank=config["rank"],
                shared_width=config["shared_width"],
                shared_experts=config.get("shared_experts", 1),
            )
            self.shared = None

    # 역할: attention과 expert를 지정 seed와 깊이 배율로 초기화한다.
    def reset_parameters(self, seed: int, name: str, depth_scale: float) -> None:
        if self.moe is not None:
            self.moe.reset_parameters(seed, f"{name}.moe", depth_scale)
        else:
            assert self.shared is not None
            self.shared.reset_parameters(seed, f"{name}.shared", depth_scale)
        self.attention.reset_parameters(seed, f"{name}.attention", depth_scale)

    # 역할: attention과 FFN 출력을 입력에 더하고 router 손실·통계도 반환한다.
    def forward(self, x: Tensor, attention_mask: Tensor, balance_fractions: Tensor | None = None) -> tuple[Tensor, Tensor | None, dict[str, Any] | None]:
        x = x + self.attention(self.attention_norm(x), attention_mask)
        normalized = self.ffn_norm(x)
        if self.moe is None:
            assert self.shared is not None
            x = x + self.shared(normalized)
            return x, None, None
        if balance_fractions is None:
            routed, aux, stats = self.moe(normalized, attention_mask)
        else:
            routed, aux, stats = self.moe(normalized, attention_mask, balance_fractions)
        # shared expert는 모든 토큰을 처리하고, routed expert의 선택된 출력과 더한다.
        x = x + self.moe.shared(normalized) + routed
        return x, aux, stats


# 역할: 토큰 embedding과 여러 transformer 층으로 다음 토큰 점수를 계산한다.
class MoETransformer(nn.Module):
    """Pre-RMSNorm RoPE decoder with tied embeddings and sparse top-2 MoE."""

    # 역할: 설정을 검증하고 embedding·각 층·최종 정규화를 만든 뒤 초기화한다.
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        validate_config(config)
        self.config = dict(config)
        d_model = config["d_model"]
        global_state = torch.random.get_rng_state()
        try:
            self.embedding = nn.Embedding(config["vocab_size"], d_model)
            self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config["layers"]))
            self.final_norm = RMSNorm(d_model)
            self._initialize(config["seed"])
        finally:
            torch.random.set_rng_state(global_state)

    # 역할: 외부 난수 상태를 보존하며 모듈별 seed와 깊이 배율로 초기화한다.
    def _initialize(self, seed: int) -> None:
        # Named local RNG streams keep shared-shape modules identical across
        # architectures and leave the caller's global RNG state untouched.
        global_state = torch.random.get_rng_state()
        try:
            _normal_(self.embedding.weight, 1.0 / math.sqrt(self.embedding.embedding_dim), seed, "embedding.weight")
            for index, block in enumerate(self.blocks):
                block.reset_parameters(seed, f"blocks.{index}", 1.0 / math.sqrt(2 * len(self.blocks)))
        finally:
            torch.random.set_rng_state(global_state)

    # 역할: 각 층을 거쳐 어휘별 점수, 층 평균 router 손실, 층별 통계를 반환한다.
    def forward(
        self, input_ids: Tensor, attention_mask: Tensor | None = None,
        *, router_balance_fractions: list[Tensor] | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, Any]]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B,T]")
        batch, sequence = input_ids.shape
        if sequence > self.config["context_length"]:
            raise ValueError("sequence length exceeds configured context_length")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        elif attention_mask.shape != (batch, sequence):
            raise ValueError("attention_mask must have shape [B,T]")
        attention_mask = attention_mask.to(device=input_ids.device, dtype=torch.bool)
        x = self.embedding(input_ids)
        layer_losses: list[Tensor] = []
        layer_stats: list[dict[str, Any]] = []
        if router_balance_fractions is not None and len(router_balance_fractions) != len(self.blocks):
            raise ValueError("router balance fractions must match layer count")
        for index, block in enumerate(self.blocks):
            # 층마다 expert 선택이 다르므로 해당 층의 전체 선택 비율을 전달한다.
            fractions = None if router_balance_fractions is None else router_balance_fractions[index]
            x, aux, stats = block(x, attention_mask, fractions)
            if aux is not None and stats is not None:
                layer_losses.append(aux)
                layer_stats.append(stats)
        x = self.final_norm(x)
        logits = F.linear(x, self.embedding.weight)
        if layer_losses:
            router_aux_loss = torch.stack(layer_losses).mean()
        else:
            router_aux_loss = x.new_zeros(())
        routing_stats = {
            "layers": layer_stats,
            "valid_token_count": int(attention_mask.sum().item()),
            "selected_count": [layer["selected_count"] for layer in layer_stats],
            "mean_probability": [layer["mean_probability"] for layer in layer_stats],
        }
        return logits, router_aux_loss, routing_stats
