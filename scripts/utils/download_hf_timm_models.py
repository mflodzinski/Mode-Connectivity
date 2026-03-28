#!/usr/bin/env python
"""Download timm models from Hugging Face Hub and save local checkpoints."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import timm
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file as load_safetensors_file


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def normalize_model_spec(value: str) -> tuple[str, str]:
    raw_value = value.strip()
    if not raw_value:
        raise ValueError("Encountered an empty model spec.")

    if raw_value.startswith("hf_hub:"):
        repo_id = raw_value[len("hf_hub:") :]
        return raw_value, repo_id

    if raw_value.startswith("https://huggingface.co/"):
        path = raw_value[len("https://huggingface.co/") :].strip("/")
        repo_id = path.split("/tree/")[0].split("/blob/")[0]
        if not repo_id or "/" not in repo_id:
            raise ValueError(f"Could not parse Hugging Face repo id from {value!r}.")
        return f"hf_hub:{repo_id}", repo_id

    if "/" in raw_value:
        return f"hf_hub:{raw_value}", raw_value

    raise ValueError(
        f"Unsupported model spec {value!r}. "
        "Use 'hf_hub:<repo_id>', '<repo_id>', or 'https://huggingface.co/<repo_id>'."
    )


def collect_model_specs(args: argparse.Namespace) -> list[str]:
    specs = list(args.model)
    if args.model_list is not None:
        for line in Path(args.model_list).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            specs.append(line)
    if not specs:
        raise ValueError("No models provided. Use --model and/or --model-list.")
    return specs


def save_checkpoint(
    model: torch.nn.Module,
    model_spec: str,
    repo_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "model.pth"
    metadata_path = output_dir / "metadata.json"

    pretrained_cfg = getattr(model, "pretrained_cfg", None)
    if pretrained_cfg is not None:
        pretrained_cfg = dict(pretrained_cfg)

    checkpoint = {
        "source": "huggingface_timm",
        "model_spec": model_spec,
        "repo_id": repo_id,
        "state_dict": model.state_dict(),
        "pretrained_cfg": pretrained_cfg,
        "num_parameters": sum(p.numel() for p in model.parameters()),
    }
    torch.save(checkpoint, checkpoint_path)

    metadata = {
        "source": "huggingface_timm",
        "model_spec": model_spec,
        "repo_id": repo_id,
        "checkpoint_path": str(checkpoint_path),
        "num_parameters": checkpoint["num_parameters"],
        "pretrained_cfg": pretrained_cfg,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return metadata


def extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state", "model_state_dict", "model"):
            value = payload.get(key)
            if isinstance(value, dict):
                return value
        if all(isinstance(v, torch.Tensor) for v in payload.values()):
            return payload
    raise ValueError("Unsupported checkpoint structure; could not extract a state_dict.")


def maybe_strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key[len("module.") :]: value for key, value in state_dict.items()}
    return state_dict


def remap_legacy_vgg_classifier_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    mapping = {
        "classifier.0.": "pre_logits.fc1.",
        "classifier.3.": "pre_logits.fc2.",
        "classifier.6.": "head.fc.",
    }
    remapped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        for old_prefix, new_prefix in mapping.items():
            if key.startswith(old_prefix):
                new_key = new_prefix + key[len(old_prefix) :]
                break
        remapped[new_key] = value
    return remapped


def adapt_state_dict_to_model(
    state_dict: dict[str, torch.Tensor],
    model: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    target_state = model.state_dict()
    adapted: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        target_tensor = target_state.get(key)
        if target_tensor is not None and tensor.shape != target_tensor.shape:
            if tensor.ndim == 2 and target_tensor.ndim == 4 and tensor.numel() == target_tensor.numel():
                tensor = tensor.reshape(target_tensor.shape)
            elif tensor.ndim == 1 and target_tensor.ndim == 1 and tensor.numel() == target_tensor.numel():
                tensor = tensor.reshape(target_tensor.shape)
        adapted[key] = tensor
    return adapted


def choose_weight_file(snapshot_dir: Path) -> Path:
    candidates = [
        "model.safetensors",
        "pytorch_model.bin",
        "model.bin",
        "weights.safetensors",
        "weights.bin",
        "model.pth",
        "model.pt",
    ]
    for name in candidates:
        path = snapshot_dir / name
        if path.exists():
            return path

    all_candidates = sorted(
        path
        for path in snapshot_dir.rglob("*")
        if path.is_file() and path.suffix in {".safetensors", ".bin", ".pth", ".pt"}
    )
    if not all_candidates:
        raise FileNotFoundError(f"No supported weight file found under {snapshot_dir}.")
    return all_candidates[0]


def build_model_from_snapshot(repo_id: str) -> tuple[torch.nn.Module, dict[str, Any]]:
    snapshot_dir = Path(
        snapshot_download(
            repo_id,
            allow_patterns=["*.json", "*.safetensors", "*.bin", "*.pth", "*.pt"],
        )
    )
    config = json.loads((snapshot_dir / "config.json").read_text())
    architecture = config["architecture"]

    model_kwargs: dict[str, Any] = {}
    for key in ("num_classes", "in_chans", "img_size", "global_pool"):
        if key in config:
            model_kwargs[key] = config[key]

    model = timm.create_model(architecture, pretrained=False, **model_kwargs)

    weights_path = choose_weight_file(snapshot_dir)
    if weights_path.suffix == ".safetensors":
        state_dict = load_safetensors_file(str(weights_path))
    else:
        payload = torch.load(weights_path, map_location="cpu")
        state_dict = extract_state_dict(payload)

    state_dict = maybe_strip_module_prefix(state_dict)
    state_dict = remap_legacy_vgg_classifier_keys(state_dict)
    state_dict = adapt_state_dict_to_model(state_dict, model)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Failed to load Hugging Face timm checkpoint cleanly: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return model, config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Model spec to download. Repeat this flag for multiple models.",
    )
    parser.add_argument(
        "--model-list",
        type=Path,
        default=None,
        help="Optional text file with one model spec per line.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("downloaded_models") / "hf_timm",
        help="Directory under which downloaded checkpoints will be saved.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    specs = collect_model_specs(args)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, Any]] = []
    for raw_spec in specs:
        model_spec, repo_id = normalize_model_spec(raw_spec)
        print(f"Downloading {model_spec}")
        fallback_config = None
        try:
            model = timm.create_model(model_spec, pretrained=True)
        except TypeError as exc:
            if "PretrainedCfg.__init__()" not in str(exc):
                raise
            print("Falling back to manual Hugging Face snapshot loading due to legacy timm config.")
            model, fallback_config = build_model_from_snapshot(repo_id)
        model.eval()

        model_dir = output_root / sanitize_name(repo_id)
        metadata = save_checkpoint(model, model_spec, repo_id, model_dir)
        if fallback_config is not None:
            metadata_path = model_dir / "metadata.json"
            metadata["hf_config"] = fallback_config
            metadata_path.write_text(json.dumps(metadata, indent=2))
        summary.append(metadata)
        print(f"Saved checkpoint to {model_dir / 'model.pth'}")

    summary_path = output_root / "download_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
