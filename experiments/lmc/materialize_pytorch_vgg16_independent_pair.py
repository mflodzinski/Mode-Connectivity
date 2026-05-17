"""Package retained independent pytorch-vgg endpoints into the repo layout.

This script copies the external checkpoint files into the thesis results tree
and emits a small manifest describing both the sources and packaged outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch


from mode_connectivity.common.paths import PROJECT_ROOT as REPO_ROOT

PROJECT_ROOT = REPO_ROOT


def main() -> None:
    source_paths = {
        "seed0": PROJECT_ROOT / "external" / "pytorch-vgg-cifar10" / "save_vgg16_seed0" / "model_final_state_dict.pth",
        "seed1": PROJECT_ROOT / "external" / "pytorch-vgg-cifar10" / "save_vgg16_seed1" / "model_final_state_dict.pth",
    }
    output_root = PROJECT_ROOT / "results" / "vgg16" / "cifar10" / "endpoints" / "pytorch_vgg_independent_existing"

    if output_root.exists():
        for path in output_root.glob("*"):
            if path.is_dir():
                for sub in path.glob("*"):
                    sub.unlink()
                path.rmdir()
            else:
                path.unlink()

    output_root.mkdir(parents=True, exist_ok=True)

    manifest = {"sources": {}, "outputs": {}}
    for seed_name, source_path in source_paths.items():
        if not source_path.exists():
            raise FileNotFoundError(f"Missing source model: {source_path}")
        state_dict = torch.load(source_path, map_location="cpu")
        seed_dir = output_root / seed_name
        seed_dir.mkdir(parents=True, exist_ok=True)

        torch.save(state_dict, seed_dir / "model_final_state_dict.pth")
        torch.save({"epoch": 200, "state_dict": state_dict}, seed_dir / "checkpoint-200.pt")

        manifest["sources"][seed_name] = str(source_path.relative_to(PROJECT_ROOT))
        manifest["outputs"][seed_name] = {
            "raw_state_dict": str((seed_dir / "model_final_state_dict.pth").relative_to(PROJECT_ROOT)),
            "checkpoint": str((seed_dir / "checkpoint-200.pt").relative_to(PROJECT_ROOT)),
        }

    with (output_root / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"Saved independent pair package to {output_root}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
