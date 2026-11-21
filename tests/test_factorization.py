import torch
from torch.nn import functional as F

from src.model import FactorizedLinear


def test_factorized_forward_and_input_a_b_gradients_match_explicit_ba_reference():
    # Float64 CPU checks use strict tolerances to expose factor ordering errors.
    atol, rtol = 1e-11, 1e-10
    torch.manual_seed(5)
    factor = FactorizedLinear(in_features=5, out_features=4, rank=3).double()
    with torch.no_grad():
        factor.A.copy_(torch.randn_like(factor.A))
        factor.B.copy_(torch.randn_like(factor.B))
    x_factor = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
    x_reference = x_factor.detach().clone().requires_grad_(True)
    probe = torch.randn(2, 3, 4, dtype=torch.float64)

    factor_output = factor(x_factor)
    factor_loss = (factor_output * probe).sum()
    factor_loss.backward()

    # Independent dense reference materializes W=B@A before applying one linear.
    a_reference = factor.A.detach().clone().requires_grad_(True)
    b_reference = factor.B.detach().clone().requires_grad_(True)
    x_reference_factors = x_factor.detach().clone().requires_grad_(True)
    explicit_weight = b_reference @ a_reference
    explicit_output = F.linear(x_reference_factors, explicit_weight)
    (explicit_output * probe).sum().backward()

    torch.testing.assert_close(factor_output, F.linear(x_factor.detach(), factor.B.detach() @ factor.A.detach()), atol=atol, rtol=rtol)
    torch.testing.assert_close(x_factor.grad, x_reference_factors.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(factor.A.grad, a_reference.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(factor.B.grad, b_reference.grad, atol=atol, rtol=rtol)
