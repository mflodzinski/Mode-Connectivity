from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARCHITECTURES = ["vgg11", "vgg13", "vgg16", "vgg19"]

FINETUNE_CURVE_PATHS = {
    "vgg11": PROJECT_ROOT
    / "results/vgg11/cifar10/raw_pth_align_sweep_perm_then_scale_only/steps50_tau1p0_lr0p05_l1p0_lossmidpoint_lam0p001_ftscale_only_fixed_hard/interpolation_curve.json",
    "vgg13": PROJECT_ROOT
    / "results/vgg13/cifar10/raw_pth_align_sweep_perm_then_scale_only_cor_def/steps50_tau1p0_lr0p02_l1p0_lossmidpoint_lam0p001_ftscale_only_fixed_hard/interpolation_curve.json",
    "vgg16": PROJECT_ROOT
    / "results/vgg16/cifar10/raw_pth_align_sweep_perm_then_scale_only/steps50_tau1p0_lr0p1_l1p0_lossmidpoint_lam0p001_ftscale_only_fixed_hard/interpolation_curve.json",
    "vgg19": PROJECT_ROOT
    / "results/vgg19/cifar10/raw_pth_align_sweep_perm_then_scale_only/steps50_tau1p0_lr0p1_l1p0_lossmidpoint_lam0p001_ftscale_only_fixed_hard/interpolation_curve.json",
}


def load_json(path: Path) -> dict:
    with open(path, "r") as handle:
        return json.load(handle)


def regenerate_plot(architecture: str) -> Path:
    comparison_root = PROJECT_ROOT / f"results/{architecture}/cifar10/interpolation_comparison_three_way"
    curves_payload = load_json(comparison_root / "curves.json")
    finetune_payload = load_json(FINETUNE_CURVE_PATHS[architecture])

    curves = curves_payload["curves"]
    test_naive = curves["test_naive"]
    test_perm = curves["test_perm"]
    test_scale = curves["test_scale"]
    test_finetune = finetune_payload["test_curve"]

    vgg_name = str(curves_payload["vgg_name"])
    show_legend = vgg_name.upper() == "VGG11"

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(test_naive["lambdas"], test_naive["losses"], color="tab:gray", linewidth=2.0, label="No Alignment")
    ax.plot(test_perm["lambdas"], test_perm["losses"], color="tab:orange", linewidth=2.0, label="Sinkhorn Permutation Only (From Scratch)")
    ax.plot(test_scale["lambdas"], test_scale["losses"], color="tab:purple", linewidth=2.0, label="Sinkhorn Permutation + Scale (From Scratch)")
    ax.plot(
        test_finetune["lambdas"],
        test_finetune["losses"],
        color="tab:green",
        linewidth=2.0,
        label="Permutation Then Scale Finetuning",
    )
    ax.set_xlabel("t (interpolation parameter)")
    ax.set_ylabel("Test Loss")
    ax.set_title(f"{vgg_name}: test loss")
    ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.5)
    if show_legend:
        ax.legend()

    output_path = comparison_root / "compare_test_loss.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    for architecture in ARCHITECTURES:
        output_path = regenerate_plot(architecture)
        print(output_path)


if __name__ == "__main__":
    main()
