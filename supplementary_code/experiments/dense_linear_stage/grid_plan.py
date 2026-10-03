"""Print the exact candidate configurations assigned to each Slurm grid shard."""

from __future__ import annotations

import argparse
import json

from mode_connectivity.common.hydra_compat import compose_experiment_config
from mode_connectivity.dense_linear_stage.runner import validate_config
from mode_connectivity.dense_linear_stage.tasks import build_dag
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-name", default="dense_linear_stage/final_vgg11"
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    cfg = OmegaConf.to_container(
        compose_experiment_config(
            default_config_name=args.config_name,
            caller_file=__file__,
            argv=[],
        ),
        resolve=True,
    )
    validate_config(cfg)
    rows = []
    for task in build_dag(cfg):
        if task["operation"] not in (
            "calibrate_base_grid", "calibrate_branch_grid"
        ):
            continue
        item, method = task["item"], task["method"]
        rows.append(dict(
            task=task["id"],
            replicate=int(item["replicate"]),
            seeds=cfg["seed_pairs"][int(item["replicate"])],
            epochs=[int(item["left_epoch"]), int(item["right_epoch"])],
            method=method,
            shard=int(task["shard"]),
            candidates=[
                dict(
                    index=int(index),
                    **cfg["calibration_candidates"][method][int(index)],
                )
                for index in task["candidate_indices"]
            ],
        ))
    if args.json:
        print(json.dumps(rows, indent=2))
        return
    for row in rows:
        print(
            f"{row['task']}  pair r{row['replicate']}={tuple(row['seeds'])}  "
            f"{row['method']} shard {row['shard']}"
        )
        for candidate in row["candidates"]:
            values = ", ".join(
                f"{key}={value}" for key, value in candidate.items()
                if key != "index"
            )
            print(f"  c{candidate['index']}: {values}")


if __name__ == "__main__":
    main()
