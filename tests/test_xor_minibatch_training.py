import pytest
import torch

from mode_connectivity.xor.xor_curve_fitting import train_xor_network


def test_minibatch_xor_training_is_reproducible_and_differs_from_full_batch():
    minibatch_a, _ = train_xor_network(
        seed=3,
        hidden_size=3,
        max_epochs=20,
        batch_size=2,
        patience=100,
    )
    minibatch_b, _ = train_xor_network(
        seed=3,
        hidden_size=3,
        max_epochs=20,
        batch_size=2,
        patience=100,
    )
    full_batch, _ = train_xor_network(
        seed=3,
        hidden_size=3,
        max_epochs=20,
        batch_size=4,
        patience=100,
    )

    for key, value in minibatch_a.state_dict().items():
        assert torch.equal(value, minibatch_b.state_dict()[key])

    assert any(
        not torch.equal(value, full_batch.state_dict()[key])
        for key, value in minibatch_a.state_dict().items()
    )


@pytest.mark.parametrize("batch_size", [0, 5])
def test_xor_training_rejects_invalid_batch_size(batch_size):
    with pytest.raises(ValueError, match="batch_size must be between"):
        train_xor_network(seed=0, batch_size=batch_size, max_epochs=1)
