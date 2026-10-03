"""Behavioral tests: scientific metrics, optimization, recovery, and Slurm DAG."""

import contextlib
import copy
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mode_connectivity.common.hydra_compat import compose_experiment_config
from omegaconf import OmegaConf
from mode_connectivity.training_stage.protocol import (
    Data,
    StopRequested,
    make_subsets,
    rng_state,
    restore_rng,
    seed_all,
    load,
    save,
    write_json,
    protocol_hash,
)
from mode_connectivity.training_stage.geometry import (
    barriers,
    profile,
    interpolated_logits,
    model,
    GitRebasinCifarMLP,
)
from mode_connectivity.alignment.permutation_spec import (
    git_rebasin_cifar_mlp_permutation_spec,
    vgg_features_permutation_spec,
)
from mode_connectivity.training_stage.scheduling import (
    build_dag,
    select_tasks,
    submit,
    completed,
    queued_state,
)
from mode_connectivity.training_stage.checks import smoke
from mode_connectivity.training_stage.training import (
    git_rebasin_augment,
    git_rebasin_lr,
    train,
)
from mode_connectivity.training_stage.runner import validate_config
from mode_connectivity.nonlinear_stage.fitting import (
    fit as fit_nonlinear_path,
    learning_rate as nonlinear_learning_rate,
)
from mode_connectivity.nonlinear_stage.geometry import PathModel, linear_logits
from mode_connectivity.nonlinear_stage.protocol import primary_pairs as nonlinear_pairs
from mode_connectivity.nonlinear_stage.runner import (
    validate_config as validate_nonlinear_config,
)
from mode_connectivity.nonlinear_stage.scheduling import (
    build_calibration_dag,
    build_dag as build_nonlinear_dag,
    select_tasks as select_nonlinear_tasks,
    submit as submit_nonlinear,
)


def config():
    return OmegaConf.to_container(
        compose_experiment_config(
            default_config_name="training_stage/default",
            caller_file=__file__,
            argv=["device=cpu", "workers=0"],
        ),
        resolve=True,
    )


def mlp_config():
    return OmegaConf.to_container(
        compose_experiment_config(
            default_config_name="training_stage/git_rebasin_mlp",
            caller_file=__file__,
            argv=["device=cpu", "workers=0"],
        ),
        resolve=True,
    )


class NeverStop:
    def check(self):
        pass


class TrainingStageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_unequal_endpoint_barriers(self):
        values = barriers([0, 0.5, 1], [3, 2.5, 1])
        self.assertAlmostEqual(values["chord"], 0.5)
        self.assertEqual(values["worse"], 0)
        self.assertEqual(values["git_rebasin"], 1)
        self.assertEqual(barriers([0, 0.5, 1], [3, 1, 1])["chord"], 0)
        with self.assertRaises(FloatingPointError):
            barriers([0, 1], [0, float("nan")])
        with self.assertRaises(ValueError):
            barriers([0.1, 1], [0, 1])

    def test_vgg11_model_and_permutation_spec_match(self):
        cfg = config()
        self.assertEqual(cfg["model"], "VGG11")
        net = model(cfg)
        conv_keys = [
            key
            for key, value in net.state_dict().items()
            if key.startswith("features.")
            and key.endswith(".weight")
            and value.ndim == 4
        ]
        spec = vgg_features_permutation_spec(cfg["model"])
        specified_conv_keys = [
            key
            for key in spec.axes_to_perm
            if key.startswith("features.") and key.endswith(".weight")
        ]
        self.assertEqual(conv_keys, specified_conv_keys)
        self.assertEqual(len(conv_keys), 8)
        self.assertEqual(
            spec.axes_to_perm["classifier.1.weight"],
            ("P_Dense_0", "P_Conv_7"),
        )

    def test_git_rebasin_mlp_architecture_and_schedule(self):
        cfg = mlp_config()
        validate_config(cfg)
        net = model(cfg)
        self.assertIsInstance(net, GitRebasinCifarMLP)
        self.assertEqual(
            [tuple(layer.weight.shape) for layer in [net.Dense_0, net.Dense_1, net.Dense_2, net.Dense_3]],
            [(512, 3072), (512, 512), (512, 512), (10, 512)],
        )
        self.assertEqual(sum(p.numel() for p in net.parameters()), 2_103_818)
        self.assertFalse(any("dropout" in type(m).__name__.lower() for m in net.modules()))
        self.assertTrue(all(layer.bias.count_nonzero() == 0 for layer in [net.Dense_0, net.Dense_1, net.Dense_2, net.Dense_3]))
        output = net(torch.zeros(2, 3, 32, 32))
        torch.testing.assert_close(output.logsumexp(1), torch.zeros(2), atol=1e-6, rtol=0)
        self.assertEqual(
            set(git_rebasin_cifar_mlp_permutation_spec().axes_to_perm),
            set(net.state_dict()),
        )
        self.assertEqual(git_rebasin_lr(0, 500, 100, 0.1), 1e-6)
        self.assertAlmostEqual(git_rebasin_lr(500, 500, 100, 0.1), 0.1)
        self.assertLess(git_rebasin_lr(49_999, 500, 100, 0.1), 1e-9)

    def test_git_rebasin_augmentation_is_seed_local_and_deterministic(self):
        images = torch.arange(2 * 3 * 32 * 32, dtype=torch.uint8).reshape(2, 3, 32, 32)
        first = git_rebasin_augment(images, torch.Generator().manual_seed(11))
        second = git_rebasin_augment(images, torch.Generator().manual_seed(11))
        third = git_rebasin_augment(images, torch.Generator().manual_seed(12))
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        self.assertFalse(torch.equal(first, third))

    def test_subsets_are_reproducible_stratified_and_disjoint(self):
        cfg = config()
        train_labels, test_labels = (
            np.tile(np.arange(10), 5000),
            np.tile(np.arange(10), 1000),
        )
        first = make_subsets(train_labels, test_labels, cfg)
        self.assertEqual(first, make_subsets(train_labels, test_labels, cfg))
        indices = first["indices"]
        for left, right in [("train", "validation"), ("alignment", "train_eval")]:
            self.assertFalse(set(indices[left]) & set(indices[right]))
        self.assertTrue(set(indices["selection"]) <= set(indices["validation"]))
        self.assertEqual(len(indices["train"]), 45000)
        self.assertEqual(
            np.bincount(train_labels[indices["alignment"]]).tolist(), [500] * 10
        )

    def test_test_access_rejected_before_dataset_loading(self):
        with patch(
            "mode_connectivity.training_stage.protocol.verify_protocol", return_value={}
        ):
            data = Data(config())
            with self.assertRaisesRegex(ValueError, "Test access prohibited"):
                data.loader("test_eval")

    def test_rng_roundtrip(self):
        seed_all(9)
        state = rng_state()
        expected = (np.random.rand(4), torch.rand(4))
        restore_rng(state)
        np.testing.assert_array_equal(expected[0], np.random.rand(4))
        torch.testing.assert_close(expected[1], torch.rand(4), atol=0, rtol=0)

    def test_profile_caching_preserves_results_and_endpoints(self):
        a, b = torch.nn.Linear(2, 2).eval(), torch.nn.Linear(2, 2).eval()
        x, y = torch.randn(8, 2), torch.arange(8) % 2
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        before = copy.deepcopy(a.state_dict())
        full = profile(a, b, loader, "cpu", [0, 0.5, 1])
        endpoints = [
            dict(loss=full["losses"][i], error=full["errors"][i]) for i in [0, -1]
        ]
        self.assertEqual(
            full, profile(a, b, loader, "cpu", [0, 0.5, 1], endpoints=endpoints)
        )
        for k, value in a.state_dict().items():
            torch.testing.assert_close(value, before[k], atol=0, rtol=0)
        torch.testing.assert_close(interpolated_logits(a, b, 0.0, x), a(x))
        torch.testing.assert_close(interpolated_logits(a, b, 1.0, x), b(x))

    def test_actual_sinkhorn_and_fixed_scale_gradients(self):
        result = smoke(config())
        self.assertTrue(all(result.values()))

    def test_dag_parallelism_and_pilot_gate(self):
        cfg = config()
        tasks = build_dag(cfg)
        by_id = {t["id"]: t for t in tasks}
        for op in ["base", "scale", "continue", "evaluate"]:
            self.assertEqual(sum(t["operation"] == op for t in tasks), 48)
        for seed in range(6):
            self.assertEqual(by_id[f"train_{seed}"]["dependencies"], ["smoke"])
        self.assertEqual(by_id["scale_1_2"]["dependencies"], ["base_1_2"])
        self.assertEqual(by_id["continue_1_2"]["dependencies"], ["base_1_2"])
        self.assertIn("gate", by_id["base_1_2"]["dependencies"])
        self.assertNotIn("train_4", by_id["base_1_2"]["dependencies"])
        self.assertIn("base_1_15", by_id["evaluate_1_2"]["dependencies"])
        pilot = select_tasks(tasks, "pilot")
        self.assertEqual(sum(t["operation"] == "train" for t in pilot), 6)
        self.assertFalse(
            any(t["operation"] in ["evaluate", "controls", "report"] for t in pilot)
        )
        endpoints = select_tasks(tasks, "endpoints")
        self.assertEqual(len(endpoints), 8)
        self.assertEqual(
            {task["operation"] for task in endpoints}, {"prepare", "smoke", "train"}
        )
        visited = set()
        for task in tasks:
            self.assertTrue(set(task["dependencies"]) <= visited)
            visited.add(task["id"])

    def test_mlp_dag_contains_onset_replication_and_stage_extension(self):
        cfg = mlp_config()
        tasks = build_dag(cfg)
        counts = {op: sum(t["operation"] == op for t in tasks) for op in {t["operation"] for t in tasks}}
        self.assertEqual(counts["train"], 2)
        self.assertEqual(counts["onset"], 101)
        for operation in ["base", "scale", "continue", "evaluate"]:
            self.assertEqual(counts[operation], 16)
        self.assertEqual(len(tasks), 174)
        self.assertEqual(len(select_tasks(tasks, "pilot")), 19)
        by_id = {t["id"]: t for t in tasks}
        self.assertEqual(by_id["onset_0_100"]["dependencies"], ["train_0", "train_1"])
        self.assertIn("onset_0_100", by_id["onset_0_0"]["dependencies"])
        self.assertIn("onset_0_0", by_id["gate"]["dependencies"])

    def test_submission_resources_and_corresponding_arrays(self):
        cfg = config()
        with tempfile.TemporaryDirectory() as temp:
            cfg["output_root"] = temp
            with contextlib.redirect_stdout(io.StringIO()):
                jobs = submit(cfg, build_dag(cfg), dry_run=True)
            self.assertFalse((Path(temp) / "tasks.json").exists())
            self.assertEqual(sum(len(j["tasks"]) for j in jobs), len(build_dag(cfg)))
            for job in jobs:
                self.assertIn("--cpus-per-task=2", job["command"])
                self.assertIn("--mem=4GB", job["command"])
                self.assertTrue(
                    any(
                        c.startswith("--time=0") and int(c[7:9]) < 4
                        for c in job["command"]
                    )
                )
            branch = next(j for j in jobs if "scale_1_2" in j["tasks"])
            self.assertTrue(
                any(c.startswith("--dependency=aftercorr:") for c in branch["command"])
            )
            ev = next(j for j in jobs if "evaluate_1_2" in j["tasks"])
            self.assertTrue(
                any("aftercorr:" in c and "afterok:" in c for c in ev["command"])
            )

    def test_partial_array_resume_rebuilds_dependencies(self):
        cfg = config()
        with tempfile.TemporaryDirectory() as temp:
            cfg["output_root"] = temp
            path = Path(temp) / "status/base_1_2.json"
            write_json(
                path,
                dict(status="complete", protocol_hash=protocol_hash(cfg), outputs=[]),
            )
            with contextlib.redirect_stdout(io.StringIO()):
                jobs = submit(cfg, build_dag(cfg), dry_run=True)
            self.assertFalse(any("base_1_2" in j["tasks"] for j in jobs))
            branch = next(j for j in jobs if "scale_1_2" in j["tasks"])
            self.assertFalse(
                any(c.startswith("--dependency=") for c in branch["command"])
            )

    def test_missing_output_is_not_complete(self):
        cfg = config()
        with tempfile.TemporaryDirectory() as temp:
            cfg["output_root"] = temp
            task = build_dag(cfg)[0]
            write_json(
                Path(temp) / "status/prepare.json",
                dict(
                    status="complete",
                    protocol_hash=protocol_hash(cfg),
                    outputs=[str(Path(temp) / "missing")],
                ),
            )
            self.assertFalse(completed(cfg, task))

    def test_expired_slurm_job_is_not_a_submission_error(self):
        expired = subprocess.CompletedProcess(
            args=["squeue"], returncode=1, stdout="", stderr="Invalid job id"
        )
        with patch(
            "mode_connectivity.training_stage.scheduling.subprocess.run",
            return_value=expired,
        ):
            self.assertEqual(queued_state("12891446"), "")

    def test_epoch_zero_and_training_resume_match_uninterrupted(self):
        cfg = config()
        cfg.update(
            epochs=3, checkpoints=[0, 1, 2, 3], stages=[1, 3], train_batch_size=4
        )
        x = torch.arange(24).reshape(12, 2).float() / 24
        y = torch.arange(12) % 2
        dataset = TensorDataset(x, y)

        class TinyData:
            def __init__(self, *args, **kwargs):
                pass

            def loader(self, name, **kw):
                return DataLoader(
                    dataset, batch_size=4, shuffle=kw.get("shuffle", False)
                )

        class InterruptAfterEpochOne:
            def check(self):
                path = Path(cfg["output_root"]) / "endpoints/0/recovery.pt"
                if path.exists() and load(path)["epoch"] == 1:
                    raise StopRequested("test interruption")

        with tempfile.TemporaryDirectory() as temp, patch(
            "mode_connectivity.training_stage.training.Data", TinyData
        ), patch(
            "mode_connectivity.training_stage.training.model",
            lambda cfg: torch.nn.Sequential(
                torch.nn.Linear(2, 4),
                torch.nn.ReLU(),
                torch.nn.Dropout(),
                torch.nn.Linear(4, 2),
            ),
        ):
            cfg["output_root"] = str(Path(temp) / "full")
            with contextlib.redirect_stdout(io.StringIO()):
                train(cfg, 0, NeverStop())
            full = load(Path(cfg["output_root"]) / "endpoints/0/epoch_003.pt")
            cfg["output_root"] = str(Path(temp) / "resumed")
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(
                StopRequested
            ):
                train(cfg, 0, InterruptAfterEpochOne())
            self.assertEqual(
                load(Path(cfg["output_root"]) / "endpoints/0/epoch_000.pt")["epoch"], 0
            )
            with contextlib.redirect_stdout(io.StringIO()):
                train(cfg, 0, NeverStop())
            resumed = load(Path(cfg["output_root"]) / "endpoints/0/epoch_003.pt")
            for key in full["state_dict"]:
                torch.testing.assert_close(
                    full["state_dict"][key], resumed["state_dict"][key], atol=0, rtol=0
                )

    def test_shell_syntax_and_config(self):
        validate_config(config())
        for script in (
            Path(__file__).resolve().parents[1] / "ops/slurm/training_stage"
        ).glob("*.sh"):
            subprocess.run(["bash", "-n", str(script)], check=True)


class AlignmentRecoveryTests(unittest.TestCase):
    def test_alignment_resume_and_equal_branch_budgets(self):
        from mode_connectivity.training_stage.alignment import optimize
        from mode_connectivity.training_stage.protocol import pair_dir

        cfg = config()
        cfg.update(
            base_passes=3,
            branch_passes=2,
            min_passes=10,
            validation_interval=1,
            alignment_batch_size=4,
            eval_batch_size=4,
        )
        torch.set_num_threads(1)
        seed_all(3)
        models = [
            torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(3072, 8),
                torch.nn.ReLU(),
                torch.nn.Dropout(),
                torch.nn.Linear(8, 10),
            )
            for _ in range(2)
        ]
        dataset = TensorDataset(torch.randn(8, 3, 32, 32), torch.arange(8))

        class TinyData:
            def __init__(self, *args):
                pass

            def loader(self, name, **kw):
                return DataLoader(
                    dataset, batch_size=4, shuffle=kw.get("shuffle", False)
                )

        def tiny_read(path, cfg):
            return copy.deepcopy(models[int(path.parent.name)]).to(cfg["device"])

        class InterruptAfterPassOne:
            def check(self):
                path = pair_dir(cfg, 0, 1, 1) / "base_recovery.pt"
                if path.exists() and load(path)["completed"] == 1:
                    raise StopRequested("test interruption")

        with tempfile.TemporaryDirectory() as temp, patch(
            "mode_connectivity.training_stage.alignment.Data", TinyData
        ), patch(
            "mode_connectivity.training_stage.alignment.read_model", tiny_read
        ), patch(
            "mode_connectivity.training_stage.alignment.weight_artifact",
            return_value={},
        ):
            cfg["output_root"] = str(Path(temp) / "full")
            for seed in [0, 1]:
                save(
                    Path(cfg["output_root"]) / f"endpoints/{seed}/epoch_001.pt",
                    {"seed": seed},
                )
            with contextlib.redirect_stdout(io.StringIO()):
                optimize(cfg, 0, 1, 1, "base", NeverStop())
            full = load(pair_dir(cfg, 0, 1, 1) / "base.pt")
            cfg["output_root"] = str(Path(temp) / "resumed")
            for seed in [0, 1]:
                save(
                    Path(cfg["output_root"]) / f"endpoints/{seed}/epoch_001.pt",
                    {"seed": seed},
                )
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(
                StopRequested
            ):
                optimize(cfg, 0, 1, 1, "base", InterruptAfterPassOne())
            self.assertFalse((pair_dir(cfg, 0, 1, 1) / "base.pt").exists())
            with contextlib.redirect_stdout(io.StringIO()):
                optimize(cfg, 0, 1, 1, "base", NeverStop())
            resumed = load(pair_dir(cfg, 0, 1, 1) / "base.pt")
            self.assertEqual(resumed["updates"], 6)
            self.assertEqual(resumed["examples"], 24)
            for before, after in zip(full["raw_parameters"], resumed["raw_parameters"]):
                torch.testing.assert_close(before, after, atol=0, rtol=0)
            for branch in ["scale", "continue"]:
                with contextlib.redirect_stdout(io.StringIO()):
                    optimize(cfg, 0, 1, 1, branch, NeverStop())
                result = load(pair_dir(cfg, 0, 1, 1) / f"{branch}.pt")
                self.assertEqual(result["updates"], 4)
                self.assertEqual(result["examples"], 16)
                if branch == "scale":
                    for before, after in zip(
                        resumed["raw_parameters"], result["raw_parameters"]
                    ):
                        torch.testing.assert_close(before, after, atol=0, rtol=0)


class NonlinearTrainingStageTests(unittest.TestCase):
    def nonlinear_config(self):
        return OmegaConf.to_container(
            compose_experiment_config(
                default_config_name="nonlinear_stage/default",
                caller_file=__file__,
                argv=[
                    "device=cpu",
                    "workers=0",
                    "source_pairs=[{seeds:[0,1],root:results/source}]",
                ],
            ),
            resolve=True,
        )

    def test_raw_nonlinear_pair_grid_and_pilot_resources(self):
        cfg = self.nonlinear_config()
        validate_nonlinear_config(cfg)
        pairs = nonlinear_pairs(cfg)
        self.assertEqual(len(pairs), 56)
        self.assertEqual(len({p["id"] for p in pairs}), 56)
        self.assertEqual(sum(p["kind"] == "same" for p in pairs), 12)
        self.assertEqual(sum(p["kind"].startswith("final_") for p in pairs), 22)
        self.assertEqual(sum(p["kind"].startswith("within_") for p in pairs), 22)
        pilot = select_nonlinear_tasks(cfg, "pilot")
        self.assertEqual(len(pilot), 28)
        self.assertEqual(sum(t["operation"] == "fit" for t in pilot), 12)
        with tempfile.TemporaryDirectory() as temp:
            cfg["output_root"] = temp
            with contextlib.redirect_stdout(io.StringIO()):
                jobs = submit_nonlinear(cfg, pilot, dry_run=True)
            for job in jobs:
                self.assertIn("--cpus-per-task=2", job["command"])
                self.assertIn("--mem=4GB", job["command"])
                self.assertTrue(any(c.startswith("--time=0") for c in job["command"]))
            validation = next(
                job for job in jobs if "validate_primary_r0_0" in job["tasks"]
            )
            self.assertTrue(
                any(c.startswith("--dependency=aftercorr:") for c in validation["command"])
            )

    def test_bezier_is_the_linear_chord_at_initialization(self):
        left = torch.nn.Sequential(
            torch.nn.Linear(2, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2)
        )
        right = copy.deepcopy(left)
        with torch.no_grad():
            for parameter in right.parameters():
                parameter.add_(0.1 * torch.randn_like(parameter))
        for parameter in list(left.parameters()) + list(right.parameters()):
            parameter.requires_grad_(False)
        path = PathModel(left, right, "bezier")
        x = torch.randn(7, 2)
        torch.testing.assert_close(path(x, 0), left(x))
        torch.testing.assert_close(path(x, 1), right(x))
        for value in [0.11, 0.5, 0.83]:
            torch.testing.assert_close(
                path(x, value), linear_logits(left, right, x, value)
            )
        path(x, 0.37).sum().backward()
        self.assertTrue(all(p.grad is None for p in left.parameters()))
        self.assertTrue(all(p.grad is None for p in right.parameters()))
        self.assertTrue(all(p.grad is not None for p in path.controls.parameters()))

    def test_nonlinear_schedule_and_dynamic_confirmation_dag(self):
        self.assertEqual(nonlinear_learning_rate(0.015, 100, 200), 0.015)
        self.assertAlmostEqual(nonlinear_learning_rate(0.015, 180, 200), 0.00015)
        cfg = self.nonlinear_config()
        pairs = nonlinear_pairs(cfg)
        tasks = build_nonlinear_dag(cfg, [pairs[0]["id"]])
        self.assertEqual(sum(t["operation"] == "freeze" for t in tasks), 56)
        self.assertEqual(sum(t["operation"] == "test" for t in tasks), 56)
        self.assertEqual(
            sum(t["operation"] == "fit" for t in tasks),
            56 + 5 + 5,
        )

    def test_nonlinear_calibration_is_focused_and_equal_budget(self):
        cfg = self.nonlinear_config()
        tasks = build_calibration_dag(cfg)
        fits = [task for task in tasks if task["operation"] == "fit"]
        validations = [task for task in tasks if task["operation"] == "validate"]
        self.assertEqual(len(fits), 6)
        self.assertEqual(len(validations), 6)
        self.assertEqual(len({task["pair_id"] for task in fits}), 1)
        self.assertTrue(
            all(task["fit_overrides"]["fit_subset"] == "curve_fit" for task in fits)
        )
        self.assertTrue(
            all(task["fit_overrides"]["fit_passes"] == 25 for task in fits)
        )
        self.assertEqual(
            {task["fit_overrides"]["seed"] for task in fits}, {cfg["path_seed"]}
        )
        self.assertEqual(tasks[-1]["operation"], "calibration_report")

    def test_nonlinear_recovery_matches_uninterrupted_path(self):
        cfg = self.nonlinear_config()
        cfg.update(
            fit_passes=2,
            validation_interval=1,
            selection_points=3,
            fit_batch_size=4,
            path_weight_decay=0.0,
        )
        pair = nonlinear_pairs(cfg)[1]
        dataset = TensorDataset(torch.randn(12, 2), torch.arange(12) % 2)

        class TinyData:
            def __init__(self, *args, **kwargs):
                pass

            def loader(self, name, **kwargs):
                return DataLoader(dataset, batch_size=4, shuffle=False)

        class StopOnce:
            def __init__(self):
                self.calls = 0

            @property
            def requested(self):
                self.calls += 1
                return self.calls == 1

        def endpoints(*args):
            torch.manual_seed(31)
            left = torch.nn.Sequential(
                torch.nn.Linear(2, 3), torch.nn.ReLU(), torch.nn.Linear(3, 2)
            )
            torch.manual_seed(32)
            right = torch.nn.Sequential(
                torch.nn.Linear(2, 3), torch.nn.ReLU(), torch.nn.Linear(3, 2)
            )
            for net in [left, right]:
                net.eval()
                for parameter in net.parameters():
                    parameter.requires_grad_(False)
            return [left, right], [Path("a"), Path("b")]

        def tiny_artifact(path, cfg, pair, family, restart, noise, selected_pass, score):
            return dict(
                family=family,
                restart=restart,
                noise=noise,
                controls=path.control_state(),
                selected_pass=selected_pass,
                selected_score=list(score),
                pair=pair,
                endpoints=[],
            )

        with tempfile.TemporaryDirectory() as temp, patch(
            "mode_connectivity.nonlinear_stage.fitting.Data", TinyData
        ), patch(
            "mode_connectivity.nonlinear_stage.fitting.read_endpoints", endpoints
        ), patch(
            "mode_connectivity.nonlinear_stage.fitting.artifact", tiny_artifact
        ):
            cfg["output_root"] = str(Path(temp) / "full")
            with contextlib.redirect_stdout(io.StringIO()):
                fit_nonlinear_path(cfg, pair, "bezier", 0, 0.0, NeverStop())
            full = load(
                Path(cfg["output_root"])
                / "paths"
                / pair["id"]
                / "bezier_r0/path.pt"
            )
            cfg["output_root"] = str(Path(temp) / "resumed")
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(
                StopRequested
            ):
                fit_nonlinear_path(cfg, pair, "bezier", 0, 0.0, StopOnce())
            with contextlib.redirect_stdout(io.StringIO()):
                fit_nonlinear_path(cfg, pair, "bezier", 0, 0.0, NeverStop())
            resumed = load(
                Path(cfg["output_root"])
                / "paths"
                / pair["id"]
                / "bezier_r0/path.pt"
            )
            for full_group, resumed_group in zip(
                full["controls"], resumed["controls"]
            ):
                for full_parameter, resumed_parameter in zip(
                    full_group, resumed_group
                ):
                    torch.testing.assert_close(
                        full_parameter, resumed_parameter, atol=0, rtol=0
                    )


if __name__ == "__main__":
    unittest.main()
