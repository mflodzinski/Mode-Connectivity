from __future__ import annotations

from collections import Counter

import torch
from torch.func import functional_call

from mode_connectivity.common.hydra_compat import compose_experiment_config
from mode_connectivity.dense_linear_stage.models import positive_scaled_state
from mode_connectivity.dense_linear_stage.alignment import artifact_aligned_state
from mode_connectivity.dense_linear_stage.calibration import select_method_choices
from mode_connectivity.dense_linear_stage.full_train import (
    METHOD_SHARDS,
    evaluation_tasks,
)
from mode_connectivity.dense_linear_stage.protocol import _stratified_subset
from mode_connectivity.dense_linear_stage.runner import validate_config
from mode_connectivity.dense_linear_stage.tasks import (
    build_dag, calibration_rows,
)
from mode_connectivity.fashion_mnist.model import FashionMLP
from mode_connectivity.alignment.permutation_spec import mlp_permutation_spec


def config(name):
    from omegaconf import OmegaConf

    return OmegaConf.to_container(
        compose_experiment_config(
            default_config_name=f"dense_linear_stage/{name}",
            caller_file=__file__,
            argv=[],
        ),
        resolve=True,
    )




def test_final_alignment_calibrates_and_evaluates_every_seed_pair():
    for name in (
        "final_vgg11", "final_vgg13", "final_vgg16", "final_vgg19",
        "final_fashion_mnist",
    ):
        cfg = config(name)
        validate_config(cfg)
        assert cfg["benchmark_mode"] == "final_alignment"
        assert cfg["calibrate_all_replicates"] is True
        assert cfg["evaluate_all_methods"] is True
        assert len(cfg["stages"]) == 1
        assert len(calibration_rows(cfg)) == 3
        counts = Counter(task["operation"] for task in build_dag(cfg))
        assert counts["calibrate_wm"] == 3
        assert counts["calibrate_base_grid"] == 18
        assert counts["calibrate_base_select"] == 3
        assert counts["calibrate_branch_grid"] == 18
        assert counts["calibrate_branch_select"] == 3
        assert counts["calibrate_base"] == 0
        assert counts["calibrate_branch"] == 0
        assert counts["calibration_evaluation"] == 3
        assert all(
            len(cfg["calibration_candidates"][method]) >= 12
            for method in (
                "wm_scale", "sinkhorn", "sinkhorn_scale_joint",
                "sinkhorn_scale_finetune",
            )
        )


def test_full_train_evaluation_shards_every_pair_and_method_once():
    tasks = evaluation_tasks()
    assert [task["index"] for task in tasks] == list(range(9))
    assert all(len(task["methods"]) == 2 for task in tasks)
    for replicate in range(3):
        methods = [
            method
            for task in tasks
            if task["replicate"] == replicate
            for method in task["methods"]
        ]
        assert methods == [method for shard in METHOD_SHARDS for method in shard]


def test_positive_scaling_preserves_deep_mlp_function():
    torch.manual_seed(4)
    model = FashionMLP(width=7, hidden_layers=3).eval()
    state = dict(model.named_parameters())
    spec = mlp_permutation_spec(3)
    scales = {
        group: torch.randn(state[key].shape[axis])
        for group, ((key, axis), *_) in spec.perm_to_axes.items()
    }
    transformed = positive_scaled_state(state, scales, spec)
    x = torch.randn(13, 1, 28, 28)
    expected = model(x)
    actual = functional_call(model, transformed, (x,))
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


def test_training_report_subset_is_exact_balanced_and_deterministic():
    labels = [label for label in range(10) for _ in range(20)]
    indices = list(range(len(labels)))
    first = _stratified_subset(indices, labels, 100, 92027)
    second = _stratified_subset(indices, labels, 100, 92027)
    assert first == second
    assert len(first) == len(set(first)) == 100
    assert Counter(labels[index] for index in first) == Counter(
        {label: 10 for label in range(10)}
    )


def test_example_budgets_match_full_training_splits():
    for name, train_size in (
        ("final_vgg11", 45_000),
        ("final_fashion_mnist", 55_000),
    ):
        cfg = config(name)
        assert cfg["expected_train_examples"] == train_size
        assert cfg["method_hyperparameters"]["sinkhorn"]["max_examples"] == 1_000_000
        for method in ("wm_scale", "sinkhorn_scale_joint", "sinkhorn_scale_finetune"):
            assert cfg["method_hyperparameters"][method]["max_examples"] == 250_000


def test_permutation_only_and_overall_selections_are_distinct():
    scores = {
        "raw": [0.8, 0.8, 1.0, 0.9],
        "wm": [0.3, 0.3, 0.5, 0.4],
        "wm_scale": [0.2, 0.2, 0.4, 0.3],
        "sinkhorn": [0.1, 0.1, 0.3, 0.2],
        "sinkhorn_scale_joint": [0.02, 0.03, 0.2, 0.1],
        "sinkhorn_scale_finetune": [0.01, 0.02, 0.2, 0.1],
    }
    choices = select_method_choices(scores)
    assert choices["permutation_only"]["method"] == "sinkhorn"
    assert choices["overall"]["method"] == "sinkhorn_scale_finetune"


def test_reused_weight_permutation_materializes_lazily(tmp_path):
    model = FashionMLP(width=7, hidden_layers=3)
    endpoint = tmp_path / "endpoints" / "1" / "epoch_000.pt"
    endpoint.parent.mkdir(parents=True)
    torch.save({"state_dict": model.state_dict()}, endpoint)
    spec = mlp_permutation_spec(3)
    permutation = {
        group: torch.arange(model.state_dict()[axes[0][0]].shape[axes[0][1]])
        for group, axes in spec.perm_to_axes.items()
    }
    cfg = {
        "dataset": "fashion_mnist",
        "hidden_layers": 3,
        "hidden_width": 7,
        "seed_pairs": [[0, 1]],
        "source_roots": {"0": str(tmp_path), "1": str(tmp_path)},
    }
    materialized = artifact_aligned_state(
        cfg, 0, 0, 0, {"method": "wm", "permutation": permutation}
    )
    for key, value in model.state_dict().items():
        torch.testing.assert_close(materialized[key], value)
