from collections import OrderedDict

import torch

from mode_connectivity.xor.xor_curve_fitting import SimpleMLP, state_to_vector
from mode_connectivity.xor.xor_joint_scale_path import (
    assert_function_preserved,
    quadratic_bezier,
    scale_state,
)


def test_positive_scaling_preserves_xor_function_and_is_invertible():
    torch.manual_seed(3)
    model = SimpleMLP(hidden_size=2, output_size=1)
    state = OrderedDict((key, value.detach().clone()) for key, value in model.state_dict().items())
    log_scales = torch.tensor([2.0, -1.5])

    scaled = scale_state(state, log_scales)
    recovered = scale_state(scaled, -log_scales)

    assert_function_preserved(state, scaled)
    for key in state:
        torch.testing.assert_close(state[key], recovered[key], atol=1e-6, rtol=1e-6)


def test_quadratic_bezier_has_exact_endpoints_and_midpoint_formula():
    endpoint_a = torch.tensor([1.0, 2.0])
    midpoint = torch.tensor([4.0, 6.0])
    endpoint_b = torch.tensor([9.0, 12.0])

    torch.testing.assert_close(
        quadratic_bezier(torch.tensor(0.0), endpoint_a, midpoint, endpoint_b),
        endpoint_a,
    )
    torch.testing.assert_close(
        quadratic_bezier(torch.tensor(1.0), endpoint_a, midpoint, endpoint_b),
        endpoint_b,
    )
    torch.testing.assert_close(
        quadratic_bezier(torch.tensor(0.5), endpoint_a, midpoint, endpoint_b),
        0.25 * endpoint_a + 0.5 * midpoint + 0.25 * endpoint_b,
    )


def test_state_vector_remains_differentiable_through_scaling():
    torch.manual_seed(4)
    model = SimpleMLP(hidden_size=2, output_size=1)
    state = model.state_dict()
    log_scales = torch.nn.Parameter(torch.zeros(2))
    vector = state_to_vector(scale_state(state, log_scales))
    vector.square().sum().backward()
    assert log_scales.grad is not None
    assert torch.isfinite(log_scales.grad).all()
