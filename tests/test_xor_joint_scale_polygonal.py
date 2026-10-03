from __future__ import annotations

import math

import torch

from mode_connectivity.xor.xor_curve_fitting import (
    XOR_DATA,
    logits_from_param_vector,
    xor_loss_and_accuracy_from_logits,
)
from mode_connectivity.xor.xor_joint_scale_polygonal import (
    bezier_path_batch,
    evaluate_path,
    logits_from_param_matrix,
    path_losses_and_accuracies,
    polygonal_path,
    polygonal_path_batch,
    stochastic_fitting_grid,
)


def test_vectorized_polygonal_evaluation_matches_scalar_implementation():
    generator = torch.Generator().manual_seed(7)
    for hidden_size in (2, 3, 7):
        parameter_count = hidden_size * 4 + 1
        points = [
            torch.randn(parameter_count, generator=generator) for _ in range(5)
        ]
        locations = torch.linspace(0.0, 1.0, 13)
        batched_parameters = polygonal_path_batch(locations, points)
        scalar_parameters = torch.stack(
            [polygonal_path(location, points) for location in locations]
        )
        torch.testing.assert_close(batched_parameters, scalar_parameters)

        batched_logits = logits_from_param_matrix(
            XOR_DATA, batched_parameters, hidden_size
        )
        scalar_logits = torch.stack(
            [
                logits_from_param_vector(XOR_DATA, parameters, hidden_size, 1)
                for parameters in scalar_parameters
            ]
        )
        torch.testing.assert_close(batched_logits, scalar_logits)

        losses, accuracies = path_losses_and_accuracies(
            batched_parameters, hidden_size
        )
        for index, logits in enumerate(scalar_logits):
            loss, accuracy = xor_loss_and_accuracy_from_logits(logits, 1)
            torch.testing.assert_close(losses[index], loss)
            assert float(accuracies[index]) == accuracy


def test_path_barrier_is_excess_above_endpoint_loss_chord():
    hidden_size = 2
    endpoint_a = torch.tensor(
        [2.0, -2.0, -2.0, 2.0, -1.0, 1.0, 2.0, -2.0, -0.5]
    )
    endpoint_b = endpoint_a.clone()
    metrics = evaluate_path(
        endpoint_a,
        endpoint_b,
        [endpoint_a.clone()],
        hidden_size,
        eval_points=11,
    )

    assert metrics["loss_barrier"] == 0.0
    assert max(abs(value) for value in metrics["excess_loss"]) < 2e-7


def test_vectorized_bezier_matches_scalar_bernstein_sum():
    generator = torch.Generator().manual_seed(11)
    points = [torch.randn(9, generator=generator) for _ in range(6)]
    locations = torch.linspace(0.0, 1.0, 17)
    batched = bezier_path_batch(locations, points)
    scalar = []
    degree = len(points) - 1
    for location in locations:
        value = torch.zeros_like(points[0])
        for index, point in enumerate(points):
            coefficient = (
                torch.tensor(float(math.comb(degree, index)))
                * (1.0 - location) ** (degree - index)
                * location**index
            )
            value = value + coefficient * point
        scalar.append(value)
    torch.testing.assert_close(batched, torch.stack(scalar))


def test_stochastic_grid_is_reproducible_and_keeps_polygon_knots_and_worst():
    first = stochastic_fitting_grid(
        8, 3, "polygonal", 0.37, torch.Generator().manual_seed(23)
    )
    second = stochastic_fitting_grid(
        8, 3, "polygonal", 0.37, torch.Generator().manual_seed(23)
    )
    torch.testing.assert_close(first, second)
    values = first.tolist()
    for required in (0.25, 0.5, 0.75, 0.37):
        assert any(abs(value - required) < 1e-6 for value in values)
