import copy

import torch

from src.model import RoutedMoE


def _moe():
    moe = RoutedMoE(
        d_model=4, n_experts=3, top_k=2, routed_width=6,
        family="LR", rank=2, shared_width=3,
    ).double()
    generator = torch.Generator().manual_seed(31)
    with torch.no_grad():
        for parameter in moe.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype))
    return moe


def _dense_all_expert_reference(moe, x, valid_mask):
    batch, sequence, width = x.shape
    flat_x = x.reshape(-1, width)
    valid_positions = valid_mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
    valid_x = flat_x.index_select(0, valid_positions)
    # Match the architecture's FP32 router math after materializing its linear.
    probabilities = torch.softmax(torch.nn.functional.linear(valid_x.float(), moe.router.weight.float()), dim=-1)
    top_probs, top_experts = torch.topk(probabilities, moe.top_k, dim=-1)
    gates = top_probs / top_probs.sum(dim=-1, keepdim=True)

    # Deliberately evaluate every expert for every valid token, then mask and sum.
    all_outputs = torch.stack([expert(valid_x) for expert in moe.experts], dim=1)
    selection_mask = torch.nn.functional.one_hot(top_experts, num_classes=moe.n_experts).to(torch.bool)
    selected_outputs = (all_outputs[:, None, :, :] * selection_mask[..., None]).sum(dim=2)
    selected_outputs = selected_outputs * gates.to(selected_outputs.dtype).unsqueeze(-1)
    routed_valid = selected_outputs.sum(dim=1)
    routed_flat = torch.zeros_like(flat_x).index_copy(0, valid_positions, routed_valid)

    counts = torch.bincount(top_experts.reshape(-1), minlength=moe.n_experts)
    fractions = counts.to(probabilities.dtype) / (moe.top_k * valid_x.shape[0])
    mean_probability = probabilities.mean(dim=0)
    aux = moe.n_experts * (fractions.detach() * mean_probability).sum()
    return routed_flat.view(batch, sequence, width), aux, counts, mean_probability, gates


def test_sparse_dropless_dispatch_matches_all_expert_masked_reference_and_gradients():
    atol, rtol = 1e-10, 1e-9
    sparse = _moe()
    dense_reference = copy.deepcopy(sparse)
    generator = torch.Generator(device="cpu").manual_seed(3)
    x_sparse = torch.randn(2, 3, 4, dtype=torch.float64, generator=generator, requires_grad=True)
    x_dense = x_sparse.detach().clone().requires_grad_(True)
    valid = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    probe = torch.randn(x_sparse.shape, dtype=x_sparse.dtype, generator=generator)

    expert_token_counts = [0, 0, 0]
    handles = []
    for i, expert in enumerate(sparse.experts):
        handles.append(expert.register_forward_pre_hook(lambda _module, args, i=i: expert_token_counts.__setitem__(i, expert_token_counts[i] + args[0].shape[0])))
    sparse_output, sparse_aux, stats = sparse(x_sparse, valid)
    for handle in handles:
        handle.remove()
    sparse_objective = (sparse_output * probe).sum() + 0.37 * sparse_aux
    sparse_objective.backward()

    dense_output, dense_aux, counts, probabilities, gates = _dense_all_expert_reference(dense_reference, x_dense, valid)
    dense_objective = (dense_output * probe).sum() + 0.37 * dense_aux
    dense_objective.backward()

    torch.testing.assert_close(sparse_output, dense_output, atol=atol, rtol=rtol)
    torch.testing.assert_close(sparse_aux, dense_aux, atol=atol, rtol=rtol)
    torch.testing.assert_close(x_sparse.grad, x_dense.grad, atol=atol, rtol=rtol)
    for (name_a, param_a), (name_b, param_b) in zip(sparse.named_parameters(), dense_reference.named_parameters()):
        assert name_a == name_b
        torch.testing.assert_close(param_a.grad, param_b.grad, atol=atol, rtol=rtol)

    assert stats["valid_token_count"] == int(valid.sum()) == 5
    assert stats["selected_count"] == counts.tolist()
    assert sum(stats["selected_count"]) == sparse.top_k * int(valid.sum())
    assert sum(expert_token_counts) == sparse.top_k * int(valid.sum())  # exactly selected routes were evaluated
    torch.testing.assert_close(torch.tensor(stats["mean_probability"]), probabilities.detach(), atol=atol, rtol=rtol)
    gate_tolerance = 2 * torch.finfo(gates.dtype).eps
    torch.testing.assert_close(
        gates.sum(dim=-1), torch.ones(5, dtype=gates.dtype),
        atol=gate_tolerance, rtol=gate_tolerance,
    )
    assert all(abs(value - 1.0) <= gate_tolerance for value in stats["selected_gate_sums"])


def test_padding_does_not_change_router_statistics_or_auxiliary_loss():
    moe = _moe()
    first = torch.randn(1, 4, 4, dtype=torch.float64)
    second = first.clone()
    second[:, 2:, :] = torch.randn_like(second[:, 2:, :]) * 100
    mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)
    _, aux_first, stats_first = moe(first, mask)
    _, aux_second, stats_second = moe(second, mask)
    assert stats_first["valid_token_count"] == 2
    assert stats_first["selected_count"] == stats_second["selected_count"]
    assert stats_first["mean_probability"] == stats_second["mean_probability"]
    torch.testing.assert_close(aux_first, aux_second, atol=0.0, rtol=0.0)


def test_uniform_router_has_k_normalized_auxiliary_value_one():
    moe = _moe()
    with torch.no_grad():
        moe.router.weight.zero_()
    x = torch.randn(2, 5, 4, dtype=torch.float64)
    valid = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.bool)
    _, aux, stats = moe(x, valid)
    torch.testing.assert_close(aux, torch.tensor(1.0, dtype=aux.dtype), atol=1e-7, rtol=1e-7)
    assert stats["valid_token_count"] == 5


def test_no_valid_tokens_returns_zero_route_and_loss():
    moe = _moe()
    x = torch.randn(1, 2, 4, dtype=torch.float64)
    route, aux, stats = moe(x, torch.zeros(1, 2, dtype=torch.bool))
    assert torch.equal(route, torch.zeros_like(route))
    assert aux.item() == 0.0
    assert stats["valid_token_count"] == 0
    assert stats["selected_count"] == [0, 0, 0]
