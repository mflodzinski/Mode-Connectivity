from __future__ import annotations

from collections import Counter

import torch
from torch.func import functional_call

from mode_connectivity.common.hydra_compat import compose_experiment_config
from mode_connectivity.dense_linear_stage.models import positive_scaled_state
from mode_connectivity.dense_linear_stage.alignment import artifact_aligned_state
from mode_connectivity.dense_linear_stage.calibration import (
    _fixed_choice,
    select_method_choices,
)
from mode_connectivity.dense_linear_stage.protocol import _stratified_subset
from mode_connectivity.dense_linear_stage.runner import validate_config
from mode_connectivity.dense_linear_stage.reuse import _best_grid_rows
from mode_connectivity.dense_linear_stage.tasks import (
    build_dag, calibration_rows, pair_rows, replication_rows, select_tasks,
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


def test_complete_ordered_pair_matrix_and_bundling():
    for name, calibration_bundle, replication_bundle, evaluation_bundle in (
        ("vgg11", 1, 2, 1), ("fashion_mnist", 3, 6, 3)
    ):
        cfg = config(name)
        validate_config(cfg)
        assert cfg["train_report_size"] == 10_000
        rows = pair_rows(cfg)
        assert len(rows) == 12 * 12 * 3
        assert any(row["left_epoch"] == cfg["stages"][0] and row["right_epoch"] == cfg["stages"][-1] for row in rows)
        assert any(row["left_epoch"] == cfg["stages"][-1] and row["right_epoch"] == cfg["stages"][0] for row in rows)
        counts = Counter(task["operation"] for task in build_dag(cfg))
        assert counts["reuse"] == 1
        calibration = calibration_rows(cfg)
        replication = replication_rows(cfg)
        assert len(calibration) == 12 * 12
        assert len(replication) == 12 * 12 * 2
        assert counts["calibrate_base"] == (len(calibration) + calibration_bundle - 1) // calibration_bundle
        assert counts["calibrate_branch"] == counts["calibrate_base"]
        assert counts["replicate_selected_evaluation"] == (len(replication) + replication_bundle - 1) // replication_bundle
        assert counts["calibration_evaluation"] == (len(calibration) + evaluation_bundle - 1) // evaluation_bundle


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


def test_resource_presets_follow_daic_feedback():
    vgg, fashion = config("vgg11"), config("fashion_mnist")
    assert vgg["slurm_resources"]["replicate_selected_evaluation"] == {
        "cpus": 2, "mem": "2GB", "time": "01:00:00"
    }
    assert fashion["slurm_resources"]["replicate_selected_evaluation"] == {
        "cpus": 1, "mem": "2GB", "time": "00:45:00"
    }
    assert vgg["replication_pairs_per_task"] == 2
    assert fashion["replication_pairs_per_task"] == 6


def test_example_budgets_replace_subset_pass_counts():
    for name, train_size in (("vgg11", 45000), ("fashion_mnist", 55000)):
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


def test_pilot_is_representative_and_does_not_freeze_all_cells():
    for name in ("vgg11", "fashion_mnist"):
        cfg = config(name)
        pilot = select_tasks(build_dag(cfg), "pilot")
        counts = Counter(task["operation"] for task in pilot)
        assert 3 <= counts["calibrate_base"] <= 5
        assert counts["calibrate_branch"] == counts["calibrate_base"]
        assert counts["freeze_choices"] == 0


def test_legacy_grid_prior_selects_each_cell_independently(tmp_path):
    for tag, lr, score in (("slow", 0.01, 0.4), ("fast", 0.05, 0.2)):
        path = tmp_path / "base" / tag / "grid_status" / "0_001_200.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            '{"status":"complete","pair":[0,1,200],'
            f'"combo":{{"base_lr":{lr},"tau":1.0,"sinkhorn_l":1.0}},'
            f'"result":{{"selected_score":[{score},0.8,20]}}}}'
        )
    best = _best_grid_rows(tmp_path, "base")
    assert best[(1, 200)][1]["base_lr"] == 0.05


def test_reuse_settings_match_legacy_wm_seeds():
    vgg, fashion = config("vgg11"), config("fashion_mnist")
    assert vgg["alignment_seed"] == 0
    assert fashion["alignment_seed"] == 1729
    assert vgg["reuse"]["use_hyperparameter_priors"] is True
    assert fashion["reuse"]["use_hyperparameter_priors"] is False


def test_only_historically_supported_vgg_hyperparameters_are_fixed():
    vgg, fashion = config("vgg11"), config("fashion_mnist")
    assert vgg["fixed_hyperparameter_values"] == {
        "sinkhorn": {"lr": 0.05, "tau": 1.5, "sinkhorn_l": 1.0},
        "sinkhorn_scale_finetune": {
            "lr": 0.05, "scale_penalty": 0.0001, "weight_decay": 0.01,
        },
    }
    assert _fixed_choice(vgg, "sinkhorn")["index"] == "historical_global_prior"
    assert _fixed_choice(vgg, "wm_scale") is None
    assert fashion["fixed_hyperparameter_values"] == {}


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
