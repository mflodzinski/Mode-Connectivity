"""Checks for the retained CIFAR-10 VGG endpoint-training workflow."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np

from mode_connectivity.alignment.permutation_spec import vgg_features_permutation_spec
from mode_connectivity.common.hydra_compat import compose_experiment_config
from mode_connectivity.training_stage.endpoint_utils import barriers, model
from mode_connectivity.training_stage.protocol import Data, make_subsets, protocol_hash, write_json
from mode_connectivity.training_stage.runner import validate_config
from mode_connectivity.training_stage.scheduling import build_dag, completed, queued_state
from omegaconf import OmegaConf


def config():
    return OmegaConf.to_container(
        compose_experiment_config(
            default_config_name="training_stage/default",
            caller_file=__file__,
            argv=["device=cpu", "workers=0"],
        ),
        resolve=True,
    )


def test_unequal_endpoint_barriers():
    values = barriers([0, 0.5, 1], [3, 2.5, 1])
    assert values["chord"] == 0.5
    assert values["worse"] == 0
    assert barriers([0, 0.5, 1], [3, 1, 1])["chord"] == 0


def test_vgg_model_matches_permutation_spec():
    cfg = config()
    validate_config(cfg)
    net = model(cfg)
    convolution_keys = [
        key for key, value in net.state_dict().items()
        if key.startswith("features.") and key.endswith(".weight") and value.ndim == 4
    ]
    spec = vgg_features_permutation_spec(cfg["model"])
    assert convolution_keys == [
        key for key in spec.axes_to_perm
        if key.startswith("features.") and key.endswith(".weight")
    ]
    assert len(convolution_keys) == 8


def test_subsets_are_reproducible_stratified_and_disjoint():
    cfg = config()
    train_labels = np.tile(np.arange(10), 5000)
    test_labels = np.tile(np.arange(10), 1000)
    first = make_subsets(train_labels, test_labels, cfg)
    assert first == make_subsets(train_labels, test_labels, cfg)
    indices = first["indices"]
    assert not set(indices["train"]) & set(indices["validation"])
    assert not set(indices["alignment"]) & set(indices["train_eval"])
    assert set(indices["selection"]) <= set(indices["validation"])
    assert len(indices["train"]) == 45_000


def test_test_access_is_blocked_during_endpoint_training():
    with patch("mode_connectivity.training_stage.protocol.verify_protocol", return_value={}):
        data = Data(config())
        try:
            data.loader("test_eval")
        except ValueError as error:
            assert "Test access prohibited" in str(error)
        else:
            raise AssertionError("test loader should be unavailable during training")


def test_endpoint_dag_contains_only_prepare_and_six_independent_runs():
    tasks = build_dag(config())
    assert [task["operation"] for task in tasks] == ["prepare"] + ["train"] * 6
    assert [task.get("seed") for task in tasks[1:]] == list(range(6))
    assert all(task["dependencies"] == ["prepare"] for task in tasks[1:])


def test_missing_declared_output_is_not_complete(tmp_path):
    cfg = config()
    cfg["output_root"] = str(tmp_path)
    task = build_dag(cfg)[0]
    write_json(
        tmp_path / "status/prepare.json",
        dict(
            status="complete",
            protocol_hash=protocol_hash(cfg),
            outputs=[str(tmp_path / "missing")],
        ),
    )
    assert not completed(cfg, task)


def test_expired_slurm_job_is_treated_as_inactive():
    expired = subprocess.CompletedProcess(
        args=["squeue"], returncode=1, stdout="", stderr="Invalid job id"
    )
    with patch(
        "mode_connectivity.training_stage.scheduling.subprocess.run",
        return_value=expired,
    ):
        assert queued_state("12891446") == ""
