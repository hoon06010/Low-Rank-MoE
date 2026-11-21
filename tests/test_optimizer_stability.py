import copy

import pytest
import torch
from torch import nn

from src.model import MoETransformer
from src.train import TrainingSession
from test_resume import CONFIG, TRAINING, BATCHES, _identity


@pytest.mark.parametrize("second_has_targets", [True, False])
def test_global_router_balance_has_same_gradient_after_microbatch_split(tmp_path, second_has_targets):
    config = {**CONFIG, "routed_experts": 4}
    rows = copy.deepcopy(BATCHES[:2])
    rows[1]["attention_mask"][0][-1] = 0
    if not second_has_targets:
        rows[1]["labels"] = [[-100] * 4]
    merged = {key: [row[key][0] for row in rows] for key in rows[0]}
    gradients = []
    for i, batches in enumerate(([merged], rows)):
        model = MoETransformer(config)
        training = {**TRAINING, "total_target_tokens": 4 if second_has_targets else 2,
                    "gradient_accumulation_steps": len(batches), "max_grad_norm": 100.,
                    "router_aux_coefficient": .7, "router_balance_scope": "optimizer_step"}
        session = TrainingSession(model, config, training, batches, _identity(), tmp_path / str(i))
        original = session.optimizer.step
        def record_step():
            gradients.append({name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None})
            original()
        session.optimizer.step = record_step
        assert session.fit()["status"] == "completed"
    assert gradients[0].keys() == gradients[1].keys()
    for name in gradients[0]:
        torch.testing.assert_close(gradients[0][name], gradients[1][name], atol=2e-6, rtol=2e-5, msg=name)


class ConditionalExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.expert = nn.Parameter(torch.tensor([[.2, -.3, .1]]))
        self.shared = nn.Parameter(torch.tensor([[.1, .2, -.1]]))
    def forward(self, ids, mask):
        value = self.shared + (self.expert if ids[0, 0] == 1 else 0.)
        return value.expand(*ids.shape, 3), value.sum()*0., {}


def test_idle_expert_uses_zero_gradient_adamw_momentum_and_decay(tmp_path):
    batches = [{"input_ids": [[i]], "labels": [[0]], "attention_mask": [[1]]} for i in (1, 2)]
    training = {**TRAINING, "total_target_tokens": 2, "warmup_target_tokens": 0}
    model = ConditionalExpert()
    session = TrainingSession(model, CONFIG, training, batches, _identity(), tmp_path)
    session.fit(max_optimizer_steps=1)
    reference = nn.Parameter(model.expert.detach().clone())
    optimizer = torch.optim.AdamW([reference], lr=session.scheduler.lr_for_next_step(1),
                                  weight_decay=training["weight_decay"], betas=(.9,.95), eps=1e-8)
    optimizer.state[reference] = copy.deepcopy(session.optimizer.state[model.expert])
    reference.grad = torch.zeros_like(reference)
    optimizer.step()
    assert session.fit()["status"] == "completed"
    torch.testing.assert_close(model.expert, reference, atol=0, rtol=0)
    assert session.optimizer.state[model.expert]["step"].item() == 2


def test_stability_runner_protocol_accepts_explicit_optimizer_and_rejects_fallback(tmp_path):
    import json
    from scripts import run_colab_mini as runner
    from test_colab_mini import _runner_config
    path = _runner_config(tmp_path / 'unused')
    config = json.loads(path.read_text())
    config["protocol_id"] = "colab-free-mini-stability-v1"
    config["training"].update(optimizer="spectron", recipe_id="mini-spectron-global-v1",
                              router_balance_scope="optimizer_step", idle_expert_policy="zero_grad")
    path.write_text(json.dumps(config))
    assert runner._config(path)["training"]["optimizer"] == "spectron"
    config["training"]["optimizer"] = "spectorn"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="optimizer"):
        runner._config(path)
