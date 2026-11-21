import gc
import math

import torch

from src.budget import build_grid, parameter_counts
from src.model import CausalSelfAttention, MoETransformer, TransformerBlock, apply_rope


def _toy_config(family="LR", seed=41):
    return {
        "config_id": f"toy-{family}", "family": family, "cell": "toy",
        "layers": 2, "d_model": 8, "heads": 2,
        "routed_experts": 3 if family != "SHARED_ONLY" else 0,
        "top_k": 2 if family != "SHARED_ONLY" else 0,
        "routed_width": 6 if family != "SHARED_ONLY" else 0,
        "rank": 2 if family != "SHARED_ONLY" else 0,
        "shared_width": 4, "vocab_size": 19, "context_length": 8, "seed": seed,
    }


def _named_parameter(model, name):
    return dict(model.named_parameters())[name]


def test_shapes_causal_mask_no_bias_and_tied_embedding():
    model = MoETransformer(_toy_config("LR")).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    logits_a, aux, stats = model(input_ids)
    changed_future = input_ids.clone()
    changed_future[0, 3] = 9
    logits_b, _, _ = model(changed_future)
    assert logits_a.shape == (1, 4, 19)
    assert aux.ndim == 0
    assert stats["valid_token_count"] == 4
    torch.testing.assert_close(logits_a[:, :3], logits_b[:, :3], atol=1e-6, rtol=1e-6)
    assert all(layer.attention.q_proj.bias is None for layer in model.blocks)
    assert all(layer.attention.k_proj.bias is None for layer in model.blocks)
    assert all(layer.attention.v_proj.bias is None for layer in model.blocks)
    assert all(layer.attention.o_proj.bias is None for layer in model.blocks)
    # The output projection is F.linear(x, embedding.weight), so the embedding
    # matrix is registered exactly once and is tied by construction.
    assert sum(p is model.embedding.weight for p in model.parameters()) == 1
    assert not any(name.startswith("lm_head") for name, _ in model.named_parameters())


def test_seeded_initialization_is_reproducible_and_common_modules_are_family_fixed():
    rng_before = torch.random.get_rng_state().clone()
    lr = MoETransformer(_toy_config("LR", seed=9))
    assert torch.equal(rng_before, torch.random.get_rng_state())
    lr_again = MoETransformer(_toy_config("LR", seed=9))
    dw = MoETransformer(_toy_config("DW", seed=9))
    for name, parameter in lr.named_parameters():
        torch.testing.assert_close(parameter, _named_parameter(lr_again, name), atol=0, rtol=0)
    common_names = [
        "embedding.weight",
        "blocks.0.attention.q_proj.weight",
        "blocks.0.attention.k_proj.weight",
        "blocks.0.attention.v_proj.weight",
        "blocks.0.attention.o_proj.weight",
        "blocks.0.moe.router.weight",
    ]
    for name in common_names:
        torch.testing.assert_close(_named_parameter(lr, name), _named_parameter(dw, name), atol=0, rtol=0)


def test_shared_only_has_no_router_and_zero_auxiliary_loss():
    model = MoETransformer(_toy_config("SHARED_ONLY"))
    logits, aux, stats = model(torch.tensor([[2, 4, 6]]), torch.tensor([[1, 1, 0]]))
    assert logits.shape == (1, 3, 19)
    assert aux.item() == 0.0
    assert stats["valid_token_count"] == 2
    assert stats["layers"] == []
    assert stats["selected_count"] == []
    assert not any("router" in name for name, _ in model.named_parameters())


def test_attention_projection_initialization_uses_depth_scaling_and_shape():
    torch.manual_seed(12)
    attention = CausalSelfAttention(8, 2)
    attention.reset_parameters(seed=2, name="attn", depth_scale=1 / math.sqrt(4))
    assert attention.q_proj.weight.shape == (8, 8)
    x = torch.randn(2, 5, 8)
    output = attention(x)
    assert output.shape == (2, 5, 8)
    assert torch.isfinite(output).all()


def test_rope_matches_hand_computed_adjacent_pair_rotation_at_positions_zero_and_one():
    x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0], [1.0, 0.0, 0.0, 1.0]]]], dtype=torch.float64)
    positions = torch.tensor([0, 1])
    actual = apply_rope(x, positions, base=10_000.0)

    # For position 0, each adjacent even/odd pair is unchanged.
    # At position 1, inverse frequencies are 1 and 1/100 for the two pairs.
    c1, s1 = math.cos(1.0), math.sin(1.0)
    c_low, s_low = math.cos(0.01), math.sin(0.01)
    expected = torch.tensor(
        [[[[1.0, 2.0, 3.0, 4.0], [c1, s1, -s_low, c_low]]]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-7)


def test_transformer_block_uses_pre_norm_and_residuals_in_documented_order():
    class OffsetNorm(torch.nn.Module):
        def __init__(self, offset):
            super().__init__()
            self.offset = offset
            self.seen = None

        def forward(self, x):
            self.seen = x.detach().clone()
            return x + self.offset

    class DoubleAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = None

        def forward(self, x, attention_mask=None):
            self.seen = x.detach().clone()
            return 2 * x

    class AddSeven(torch.nn.Module):
        def forward(self, x):
            return x + 7

    class ControlledMoE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = AddSeven()
            self.seen = None

        def forward(self, x, attention_mask):
            self.seen = x.detach().clone()
            return 5 * x, x.new_tensor(0.25), {"valid_token_count": 1}

    config = {
        "family": "LR", "d_model": 2, "heads": 1,
        "routed_experts": 2, "top_k": 2, "routed_width": 2,
        "rank": 1, "shared_width": 1,
    }
    block = TransformerBlock(config)
    block.attention_norm = OffsetNorm(10)
    block.attention = DoubleAttention()
    block.ffn_norm = OffsetNorm(-5)
    block.moe = ControlledMoE()

    x = torch.tensor([[[1.0, 2.0]]])
    actual, aux, stats = block(x, torch.ones(1, 1, dtype=torch.bool))

    # Hand calculation: n_attn=x+10=[11,12], residual y=x+2*n_attn=[23,26],
    # n_ffn=y-5=[18,21], and residual output=y+(n_ffn+7)+5*n_ffn.
    n_attn = torch.tensor([[[11.0, 12.0]]])
    after_attention_residual = torch.tensor([[[23.0, 26.0]]])
    n_ffn = torch.tensor([[[18.0, 21.0]]])
    expected = after_attention_residual + (n_ffn + 7) + 5 * n_ffn
    torch.testing.assert_close(block.attention_norm.seen, x, atol=0, rtol=0)
    torch.testing.assert_close(block.attention.seen, n_attn, atol=0, rtol=0)
    torch.testing.assert_close(block.ffn_norm.seen, after_attention_residual, atol=0, rtol=0)
    torch.testing.assert_close(block.moe.seen, n_ffn, atol=0, rtol=0)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert aux.item() == 0.25
    assert stats == {"valid_token_count": 1}


def test_all_eleven_model_unique_parameter_counts_match_algebraic_budget():
    for config in build_grid():
        model = MoETransformer(config)
        actual = sum(parameter.numel() for parameter in model.parameters())
        expected = parameter_counts(config)["whole_model_total"]
        assert actual == expected, config["config_id"]
        # There is no separately registered output embedding weight.
        assert sum(parameter is model.embedding.weight for parameter in model.parameters()) == 1
        del model
        gc.collect()


def test_fixed_width_shared_experts_match_budget_and_are_seeded():
    config = _toy_config("LR", seed=23)
    config.update(shared_width=2, shared_experts=2)
    model = MoETransformer(config)
    again = MoETransformer(config)
    assert sum(p.numel() for p in model.parameters()) == parameter_counts(config)["whole_model_total"]
    for left, right in zip(model.parameters(), again.parameters()):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    assert len(model.blocks[0].moe.shared.experts) == 2
    logits, aux, _ = model(torch.tensor([[1, 2, 3]]))
    assert logits.shape == (1, 3, config["vocab_size"])
    assert torch.isfinite(logits).all() and torch.isfinite(aux)
    legacy = MoETransformer(_toy_config("LR", seed=23))
    assert "blocks.0.moe.shared.gate.weight" in legacy.state_dict()
