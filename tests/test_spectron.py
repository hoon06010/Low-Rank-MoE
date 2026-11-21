"""Independent checks for the paper Algorithm 1 optimizer integration."""
import copy
import importlib.util

import pytest
import torch

from src.model import FactorizedLinear, MoETransformer
from src.train import TrainingSession, _default_training_config
from test_resume import CONFIG, TRAINING, BATCHES, _identity, _assert_nested_equal, _seed_all


def test_optimizer_configuration_rejects_silent_fallback():
    with pytest.raises(ValueError, match="optimizer"):
        _default_training_config({"optimizer": "spectorn"})


def test_spectron_is_implemented():
    assert importlib.util.find_spec("src.optim") is not None, "Spectron optimizer missing"


def test_scalar_pair_matches_paper_update_and_skips_unused_pair():
    from src.optim import Spectron
    a = torch.nn.Parameter(torch.tensor([[3.]]))
    b = torch.nn.Parameter(torch.tensor([[4.]]))
    opt = Spectron([(a, b)], lr=.08, momentum=.9, ns_steps=5, power_steps=1, weight_decay=0.)
    a.grad, b.grad = torch.tensor([[2.]]), torch.tensor([[-1.]])
    # In dimension one, power iteration gives exact |a| and |b|.
    def direction(g):
        x = .1 * g / (abs(.1 * g) + 1e-7)
        for _ in range(5):
            x = 3.4445*x - 4.775*x**3 + 2.0315*x**5
        return x
    opt.step()
    torch.testing.assert_close(a, torch.tensor([[3. - .01*direction(2.)]]))
    torch.testing.assert_close(b, torch.tensor([[4. - .01*direction(-1.)]]))
    before = copy.deepcopy(opt.state_dict())
    values = (a.clone(), b.clone())
    opt.zero_grad(set_to_none=True)
    opt.step()
    _assert_nested_equal(before, opt.state_dict())
    assert torch.equal(a, values[0]) and torch.equal(b, values[1])


def test_spectron_zero_matrices_and_partial_pair():
    from src.optim import Spectron
    a = torch.nn.Parameter(torch.zeros(2, 3))
    b = torch.nn.Parameter(torch.zeros(4, 2))
    opt = Spectron([(a, b)], lr=.01)
    a.grad, b.grad = torch.zeros_like(a), torch.zeros_like(b)
    opt.step()
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    a.grad = None
    with pytest.raises(ValueError, match="pair"):
        opt.step()


def test_spectron_session_exact_resume_and_adamw_parameter_partition(tmp_path):
    training = {**TRAINING, "optimizer": "spectron", "spectron_weight_decay": 0.}
    def session(path):
        return TrainingSession(MoETransformer(CONFIG), CONFIG, training, copy.deepcopy(BATCHES), _identity(), path)
    _seed_all(94)
    full = session(tmp_path / "full")
    assert full.training_config["recipe_id"] != "adamw-token-cosine-v1"
    assert full.fit()["status"] == "completed"
    _seed_all(94)
    partial = session(tmp_path / "resume")
    assert partial.fit(max_optimizer_steps=2)["status"] == "incomplete"
    resumed = session(tmp_path / "resume")
    resumed.restore_latest()
    assert resumed.fit()["status"] == "completed"
    _assert_nested_equal(full.model.state_dict(), resumed.model.state_dict())
    _assert_nested_equal(full.optimizer.state_dict(), resumed.optimizer.state_dict())
    factors = {id(p) for m in full.model.modules() if isinstance(m, FactorizedLinear) for p in (m.A, m.B)}
    actual = {id(p) for g in full.optimizer.spectron.param_groups for p in g["params"]}
    adam = {id(p) for g in full.optimizer.adamw.param_groups for p in g["params"]}
    assert actual == factors and not actual & adam
    assert actual | adam == {id(p) for p in full.model.parameters()}


@pytest.mark.parametrize("shape", [(2, 7), (7, 2)])
def test_orthogonalization_matches_independent_singular_value_polynomial(shape):
    from src.optim import orthogonalize
    generator = torch.Generator().manual_seed(101)
    matrix = torch.randn(shape, dtype=torch.float64, generator=generator)
    u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    singular = singular / (matrix.norm() + 1e-7)
    for _ in range(5):
        singular = 3.4445*singular - 4.775*singular**3 + 2.0315*singular**5
    torch.testing.assert_close(orthogonalize(matrix), (u*singular)@vh, atol=1e-11, rtol=1e-10)


def test_power_iteration_recovers_from_null_start():
    from src.optim import power_norm
    matrix = torch.diag(torch.tensor([0., 7.]))
    norm, vector = power_norm(matrix, torch.tensor([1., 0.]), steps=3)
    assert norm.item() == pytest.approx(7.)
    assert torch.isfinite(vector).all()


def test_stability_diagnostics_are_saved_with_checkpoint(tmp_path):
    training = {**TRAINING, "optimizer": "spectron"}
    session = TrainingSession(MoETransformer(CONFIG), CONFIG, training, BATCHES, _identity(), tmp_path)
    session.fit(max_optimizer_steps=1)
    payload = session._checkpoint_payload("inspection")
    assert "last_step_diagnostics" in payload
    diagnostics = payload["last_step_diagnostics"]
    assert diagnostics["ce_loss_nats"] >= 0
    assert diagnostics["gradient_norm_before_clip"] >= 0
    assert diagnostics["factor_spectral_estimate_max"] > 0
    assert diagnostics["optimizer_step"] == 1
