"""Run the VGG16 permutation-only endpoint alignment pipeline."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "scripts"))

from src.utils import set_global_seed
from scripts.lib.alignment.c2m3_bridge import (
    get_canonical_combinations,
    local_spec_to_c2m3,
    run_pairwise_frank_wolfe,
    run_synchronized_frank_wolfe,
)
from scripts.lib.alignment.path_checkpoint_sampling import (
    load_sampled_state_dicts,
    sample_curve_checkpoints,
    validate_endpoint_samples,
)
from scripts.lib.alignment.permutation_pipeline import (
    apply_endpoint_permutation_to_state_dict,
    convert_perm_keys_to_apply_format,
    derive_endpoint_permutation_from_factored,
    get_test_batch,
    identity_permutation,
    load_vgg16_cifar10_loaders,
    resolve_device,
    save_checkpoint_with_state_dict,
    serialize_permutation,
    state_dict_to_perm_params,
    to_numpy_permutation,
    verify_functional_equivalence,
    write_json,
    write_permutation_json,
    write_summary_files,
    compose_permutation_sequence,
)
from scripts.lib.alignment.permutation_spec import vgg16_permutation_spec
from scripts.lib.alignment.weight_matching import weight_matching as local_weight_matching
from scripts.lib.core.training_commands import (
    add_curve_args,
    add_optional_arg,
    add_seed_arg,
    add_training_hyperparams,
    build_base_command,
    print_and_format_command,
)


BASELINE_KEYS = [
    "baseline_1_no_permutation",
    "baseline_2_c2m3_direct",
    "baseline_3_greedy_adjacent",
    "baseline_4_c2m3_global",
]

BASELINE_DISPLAY_NAMES = {
    "baseline_1_no_permutation": "Baseline 1 - no permutation",
    "baseline_2_c2m3_direct": "Baseline 2 - direct C2M3 endpoint matching",
    "baseline_3_greedy_adjacent": "Baseline 3 - greedy adjacent matching",
    "baseline_4_c2m3_global": "Baseline 4 - global multi-checkpoint C2M3",
}


def print_stage(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_kv(label: str, value) -> None:
    print(f"{label}: {value}")


@hydra.main(
    version_base=None,
    config_path="../../configs/analysis",
    config_name="permutation_path_alignment_vgg16",
)
def main(cfg: DictConfig):
    validate_config(cfg)
    set_global_seed(cfg.path.seed)

    pipeline_root = Path(to_absolute_path(cfg.output_root)) / cfg.experiment_name
    path_training_dir = pipeline_root / "path_training"
    path_checkpoint_dir = path_training_dir / "checkpoints"
    path_evaluation_dir = path_training_dir / "evaluations"
    sampled_dir = pipeline_root / "sampled_checkpoints"
    summary_dir = pipeline_root / "summary"

    runtime_device = resolve_device(cfg.runtime.device)
    matching_device = "cpu" if runtime_device.type == "mps" else runtime_device.type

    endpoint_a_path = to_absolute_path(cfg.endpoint_a)
    endpoint_b_path = to_absolute_path(cfg.endpoint_b)
    curve_checkpoint_path = path_checkpoint_dir / f"checkpoint-{cfg.path.epochs}.pt"

    print_stage("Permutation Path Alignment Pipeline")
    print_kv("experiment_name", cfg.experiment_name)
    print_kv("pipeline_root", pipeline_root)
    print_kv("endpoint_a", endpoint_a_path)
    print_kv("endpoint_b", endpoint_b_path)
    print_kv("runtime_device", runtime_device)
    print_kv("matching_device", matching_device)
    print_kv("path_pretrained_checkpoint", cfg.path.get("pretrained_checkpoint"))
    print_kv("sampling_precomputed_dir", cfg.sampling.precomputed_dir)
    print_kv("overwrite", cfg.overwrite)

    path_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path_evaluation_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    if cfg.sampling.precomputed_dir:
        print_stage("Stage 1 - Reuse Precomputed Path Samples")
        materialize_precomputed_samples(
            cfg,
            source_dir=Path(to_absolute_path(cfg.sampling.precomputed_dir)),
            sampled_dir=sampled_dir,
            endpoint_a_path=endpoint_a_path,
            endpoint_b_path=endpoint_b_path,
        )
    else:
        print_stage("Stage 1 - Prepare Curve Checkpoint")
        train_path_if_needed(cfg, path_checkpoint_dir, curve_checkpoint_path, endpoint_a_path, endpoint_b_path)
        print_stage("Stage 2 - Sample Path Checkpoints")
        sample_checkpoints_if_needed(
            cfg,
            curve_checkpoint_path=str(curve_checkpoint_path),
            sampled_dir=sampled_dir,
            endpoint_a_path=endpoint_a_path,
            endpoint_b_path=endpoint_b_path,
            device=matching_device,
        )

    sampled_checkpoint_paths = [str(sampled_dir / f"C{i}.pt") for i in range(len(cfg.sampling.ts))]
    print_kv("sampled_checkpoint_paths", sampled_checkpoint_paths)
    sampled_state_dicts = load_sampled_state_dicts(sampled_checkpoint_paths)

    state_a = sampled_state_dicts[0]
    state_b = sampled_state_dicts[-1]
    local_spec = vgg16_permutation_spec()
    c2m3_spec = local_spec_to_c2m3(local_spec)

    loaders, _ = load_vgg16_cifar10_loaders(
        data_path=cfg.data_path,
        batch_size=cfg.evaluation.batch_size,
        num_workers=cfg.evaluation.num_workers,
    )
    equivalence_batch = get_test_batch(loaders, batch_index=cfg.equivalence.batch_index)

    summary_rows = []

    for baseline_key in BASELINE_KEYS:
        baseline_dir = pipeline_root / baseline_key
        baseline_dir.mkdir(parents=True, exist_ok=True)
        print_stage(f"Stage 3 - {BASELINE_DISPLAY_NAMES[baseline_key]}")
        print_kv("baseline_dir", baseline_dir)

        _, equivalence_metrics = run_baseline(
            cfg=cfg,
            baseline_key=baseline_key,
            baseline_dir=baseline_dir,
            local_spec=local_spec,
            c2m3_spec=c2m3_spec,
            state_a=state_a,
            state_b=state_b,
            sampled_state_dicts=sampled_state_dicts,
            endpoint_b_path=endpoint_b_path,
            loaders=loaders,
            equivalence_batch=equivalence_batch,
            runtime_device=runtime_device,
            matching_device=matching_device,
        )

        summary_rows.append(
            {
                "baseline_key": baseline_key,
                "max_abs_logit_diff": equivalence_metrics["max_abs_logit_diff"],
                "mean_abs_logit_diff": equivalence_metrics["mean_abs_logit_diff"],
                "same_argmax_fraction": equivalence_metrics["same_argmax_fraction"],
                "allclose": equivalence_metrics["allclose"],
            }
        )
        print_kv("max_abs_logit_diff", equivalence_metrics["max_abs_logit_diff"])
        print_kv("mean_abs_logit_diff", equivalence_metrics["mean_abs_logit_diff"])
        print_kv("same_argmax_fraction", equivalence_metrics["same_argmax_fraction"])
        print_kv("allclose", equivalence_metrics["allclose"])

    print_stage("Stage 4 - Write Summary")
    write_summary_files(str(summary_dir), summary_rows)
    print(f"Pipeline outputs written to: {pipeline_root}")


def validate_config(cfg: DictConfig) -> None:
    if cfg.model != "VGG16":
        raise ValueError(f"This pipeline is VGG16-only. Received model={cfg.model}.")
    if cfg.dataset != "CIFAR10":
        raise ValueError(f"This pipeline is CIFAR10-only. Received dataset={cfg.dataset}.")
    if cfg.path.curve != "PolyChain":
        raise ValueError(f"This pipeline expects PolyChain path training. Received curve={cfg.path.curve}.")
    if len(cfg.sampling.ts) != 5:
        raise ValueError(f"This pipeline requires exactly five sampled checkpoints. Received {len(cfg.sampling.ts)}.")


def train_path_if_needed(
    cfg: DictConfig,
    checkpoint_dir: Path,
    expected_checkpoint_path: Path,
    endpoint_a_path: str,
    endpoint_b_path: str,
) -> None:
    pretrained_checkpoint = cfg.path.get("pretrained_checkpoint")
    if pretrained_checkpoint:
        source_checkpoint = Path(to_absolute_path(pretrained_checkpoint))
        print_kv("curve_checkpoint_source", source_checkpoint)
        if not source_checkpoint.exists():
            raise FileNotFoundError(f"Configured pretrained path checkpoint does not exist: {source_checkpoint}")

        if expected_checkpoint_path.exists() and not cfg.overwrite:
            print(f"Skipping path checkpoint copy, found existing checkpoint: {expected_checkpoint_path}")
            return

        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if source_checkpoint.resolve() != expected_checkpoint_path.resolve():
            shutil.copy2(source_checkpoint, expected_checkpoint_path)
            print(f"Copied pretrained curve checkpoint to: {expected_checkpoint_path}")
        else:
            print(f"Using pretrained curve checkpoint in place: {expected_checkpoint_path}")
        return

    if expected_checkpoint_path.exists() and not cfg.overwrite:
        print(f"Skipping path training, found existing checkpoint: {expected_checkpoint_path}")
        return

    repo_root = Path(to_absolute_path("external/dnn-mode-connectivity"))
    train_script = repo_root / "train.py"
    print_kv("train_script", train_script)
    print_kv("curve_checkpoint_target", expected_checkpoint_path)

    train_cfg = OmegaConf.create(
        {
            "dataset": cfg.dataset,
            "data_path": cfg.data_path,
            "transform": cfg.transform,
            "model": cfg.model,
            **OmegaConf.to_container(cfg.path, resolve=True),
        }
    )

    cmd = build_base_command(str(train_script), str(checkpoint_dir), train_cfg)
    add_curve_args(
        cmd,
        train_cfg,
        endpoint_a_path,
        endpoint_b_path,
        fix_endpoints=True,
        curve_type=cfg.path.curve,
        num_bends=cfg.path.num_bends,
    )
    add_training_hyperparams(cmd, train_cfg)
    add_seed_arg(cmd, cfg.path.seed)
    add_optional_arg(cmd, train_cfg, "save_freq", "--save_freq")
    add_optional_arg(cmd, train_cfg, "use_test", "--use_test", is_flag=True)
    add_optional_arg(cmd, train_cfg, "skip_eval", "--skip_eval", is_flag=True)

    print_and_format_command(cmd)
    subprocess.run(cmd, check=True)


def sample_checkpoints_if_needed(
    cfg: DictConfig,
    *,
    curve_checkpoint_path: str,
    sampled_dir: Path,
    endpoint_a_path: str,
    endpoint_b_path: str,
    device: str,
) -> None:
    sampled_paths = [sampled_dir / f"C{i}.pt" for i in range(len(cfg.sampling.ts))]
    metadata_path = sampled_dir / "metadata.json"

    if all(path.exists() for path in sampled_paths) and metadata_path.exists() and not cfg.overwrite:
        print(f"Skipping checkpoint sampling, found existing samples under: {sampled_dir}")
        return

    print_kv("curve_checkpoint_path", curve_checkpoint_path)
    print_kv("sampled_dir", sampled_dir)
    print_kv("sampling_ts", [float(t) for t in cfg.sampling.ts])
    sample_curve_checkpoints(
        curve_checkpoint_path,
        output_dir=str(sampled_dir),
        ts=[float(t) for t in cfg.sampling.ts],
        model_name=cfg.model,
        curve_type=cfg.path.curve,
        num_bends=cfg.path.num_bends,
        num_classes=cfg.num_classes,
        device=device,
        source_metadata={
            "endpoint_a": endpoint_a_path,
            "endpoint_b": endpoint_b_path,
            "curve": cfg.path.curve,
            "num_bends": int(cfg.path.num_bends),
        },
    )

    sampled_state_dicts = load_sampled_state_dicts([str(path) for path in sampled_paths])
    validate_endpoint_samples(
        sampled_state_dicts,
        endpoint_a_path=endpoint_a_path,
        endpoint_b_path=endpoint_b_path,
    )

    write_json(
        str(metadata_path),
        {
            "sampled_checkpoint_paths": [str(path) for path in sampled_paths],
            "curve_checkpoint_path": os.path.abspath(curve_checkpoint_path),
            "endpoint_a": os.path.abspath(endpoint_a_path),
            "endpoint_b": os.path.abspath(endpoint_b_path),
            "ts": [float(t) for t in cfg.sampling.ts],
        },
    )


def materialize_precomputed_samples(
    cfg: DictConfig,
    *,
    source_dir: Path,
    sampled_dir: Path,
    endpoint_a_path: str,
    endpoint_b_path: str,
) -> None:
    sampled_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = sampled_dir / "metadata.json"
    expected_paths = [sampled_dir / f"C{i}.pt" for i in range(len(cfg.sampling.ts))]

    if all(path.exists() for path in expected_paths) and metadata_path.exists() and not cfg.overwrite:
        print(f"Using existing sampled checkpoints from: {sampled_dir}")
        return

    print_kv("precomputed_source_dir", source_dir)
    print_kv("sampled_dir", sampled_dir)
    for index in range(len(cfg.sampling.ts)):
        source_path = source_dir / f"C{index}.pt"
        if not source_path.exists():
            raise FileNotFoundError(f"Missing precomputed sample: {source_path}")
        shutil.copy2(source_path, sampled_dir / f"C{index}.pt")

    source_metadata_path = source_dir / "metadata.json"
    metadata = {
        "precomputed_source_dir": str(source_dir.resolve()),
        "endpoint_a": os.path.abspath(endpoint_a_path),
        "endpoint_b": os.path.abspath(endpoint_b_path),
        "ts": [float(t) for t in cfg.sampling.ts],
    }
    if source_metadata_path.exists():
        with open(source_metadata_path, "r") as handle:
            metadata["source_metadata"] = json.load(handle)
    write_json(str(metadata_path), metadata)

    validate_endpoint_samples(
        load_sampled_state_dicts([str(path) for path in expected_paths]),
        endpoint_a_path=endpoint_a_path,
        endpoint_b_path=endpoint_b_path,
    )


def run_baseline(
    *,
    cfg: DictConfig,
    baseline_key: str,
    baseline_dir: Path,
    local_spec,
    c2m3_spec,
    state_a,
    state_b,
    sampled_state_dicts,
    endpoint_b_path: str,
    loaders,
    equivalence_batch,
    runtime_device,
    matching_device: str,
):
    endpoint_q_path = baseline_dir / "endpoint_q.json"
    endpoint_q_converted_path = baseline_dir / "endpoint_q_converted.json"
    b_perm_path = baseline_dir / "b_perm.pt"
    equivalence_path = baseline_dir / "functional_equivalence.json"
    factored_path = baseline_dir / "factored_permutations.json"

    required_paths = [
        endpoint_q_path,
        endpoint_q_converted_path,
        b_perm_path,
        equivalence_path,
    ]
    if baseline_key == "baseline_4_c2m3_global":
        required_paths.append(factored_path)

    if all(path.exists() for path in required_paths) and not cfg.overwrite:
        print(f"Reusing existing artifacts for {baseline_key} from: {baseline_dir}")
        return None, load_json_file(equivalence_path)

    if baseline_key == "baseline_1_no_permutation":
        print("Using identity permutation for endpoint B.")
        endpoint_q = identity_permutation(state_b, local_spec)
        b_perm_state = state_b
    elif baseline_key == "baseline_2_c2m3_direct":
        print("Running pairwise C2M3 Frank-Wolfe on endpoints C0 and C4.")
        endpoint_q = compute_c2m3_direct_endpoint_permutation(
            cfg,
            local_spec=local_spec,
            c2m3_spec=c2m3_spec,
            state_a=state_a,
            state_b=state_b,
            matching_device=matching_device,
        )
        b_perm_state = apply_endpoint_permutation_to_state_dict(state_b, endpoint_q)
    elif baseline_key == "baseline_3_greedy_adjacent":
        print("Running greedy adjacent pairwise matching on C0->C1->C2->C3->C4.")
        endpoint_q = compute_greedy_adjacent_endpoint_permutation(
            cfg,
            local_spec=local_spec,
            sampled_state_dicts=sampled_state_dicts,
        )
        b_perm_state = apply_endpoint_permutation_to_state_dict(state_b, endpoint_q)
    elif baseline_key == "baseline_4_c2m3_global":
        print("Running synchronized C2M3 Frank-Wolfe on all sampled checkpoints.")
        factored_permutations = compute_c2m3_global_factored_permutations(
            cfg,
            local_spec=local_spec,
            c2m3_spec=c2m3_spec,
            sampled_state_dicts=sampled_state_dicts,
            matching_device=matching_device,
        )
        write_json(
            str(factored_path),
            {symbol: serialize_permutation(perms) for symbol, perms in factored_permutations.items()},
        )
        endpoint_q = derive_endpoint_permutation_from_factored(
            factored_permutations,
            fixed_symbol="C0",
            permutee_symbol="C4",
        )
        b_perm_state = apply_endpoint_permutation_to_state_dict(state_b, endpoint_q)
    else:
        raise ValueError(f"Unknown baseline: {baseline_key}")

    write_permutation_json(str(endpoint_q_path), endpoint_q)
    write_json(
        str(endpoint_q_converted_path),
        {key: value.tolist() for key, value in convert_perm_keys_to_apply_format(endpoint_q).items()},
    )

    save_checkpoint_with_state_dict(
        endpoint_b_path,
        str(b_perm_path),
        b_perm_state,
        metadata={
            "baseline_key": baseline_key,
            "source_endpoint_b": endpoint_b_path,
        },
    )

    equivalence_metrics = verify_functional_equivalence(
        state_b,
        b_perm_state,
        equivalence_batch,
        device=runtime_device,
        atol=cfg.equivalence.atol,
        rtol=cfg.equivalence.rtol,
        num_classes=cfg.num_classes,
        permutation_applied=(baseline_key != "baseline_1_no_permutation"),
    )
    write_json(str(equivalence_path), equivalence_metrics)
    print(f"Saved baseline artifacts to: {baseline_dir}")
    return None, equivalence_metrics


def compute_c2m3_direct_endpoint_permutation(
    cfg: DictConfig,
    *,
    local_spec,
    c2m3_spec,
    state_a,
    state_b,
    matching_device: str,
):
    print_kv("pairwise_matching_device", matching_device)
    print_kv("pairwise_fw_max_iter", cfg.matching.pairwise_fw.max_iter)
    permutation, _ = run_pairwise_frank_wolfe(
        state_dict_to_perm_params(state_a, local_spec),
        state_dict_to_perm_params(state_b, local_spec),
        c2m3_spec,
        initialization_method=cfg.matching.pairwise_fw.initialization_method,
        max_iter=cfg.matching.pairwise_fw.max_iter,
        num_trials=cfg.matching.pairwise_fw.num_trials,
        device=matching_device,
    )
    return to_numpy_permutation(permutation)


def compute_greedy_adjacent_endpoint_permutation(
    cfg: DictConfig,
    *,
    local_spec,
    sampled_state_dicts,
):
    adjacent_permutations = []
    for index in range(len(sampled_state_dicts) - 1):
        print(f"Matching adjacent checkpoints: C{index} -> C{index + 1}")
        fixed = state_dict_to_perm_params(sampled_state_dicts[index], local_spec)
        permutee = state_dict_to_perm_params(sampled_state_dicts[index + 1], local_spec)
        permutation = local_weight_matching(
            local_spec,
            fixed,
            permutee,
            max_iter=cfg.matching.greedy_adjacent.max_iter,
            silent=True,
        )
        adjacent_permutations.append(permutation)
    return compose_permutation_sequence(adjacent_permutations)


def compute_c2m3_global_factored_permutations(
    cfg: DictConfig,
    *,
    local_spec,
    c2m3_spec,
    sampled_state_dicts,
    matching_device: str,
):
    symbols = [f"C{index}" for index in range(len(sampled_state_dicts))]
    print_kv("global_matching_symbols", symbols)
    print_kv("global_matching_device", matching_device)
    print_kv("global_fw_max_iter", cfg.matching.global_fw.max_iter)
    params_by_symbol = {
        symbol: state_dict_to_perm_params(state_dict, local_spec)
        for symbol, state_dict in zip(symbols, sampled_state_dicts)
    }
    factored_permutations, _ = run_synchronized_frank_wolfe(
        params_by_symbol,
        c2m3_spec,
        symbols=symbols,
        combinations=get_canonical_combinations(symbols),
        initialization_method=cfg.matching.global_fw.initialization_method,
        max_iter=cfg.matching.global_fw.max_iter,
        device=matching_device,
    )
    return {symbol: to_numpy_permutation(perms) for symbol, perms in factored_permutations.items()}


def load_json_file(path: Path):
    with open(path, "r") as handle:
        return json.load(handle)


if __name__ == "__main__":
    main()
