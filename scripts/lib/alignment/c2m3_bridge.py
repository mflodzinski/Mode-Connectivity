"""Bridge utilities for using vendored C2M3 Frank-Wolfe matchers locally.

The vendored C2M3 repository is not directly importable in this workspace because
its package-level imports expect extra runtime dependencies (for example
``nn_core`` and full Lightning/W&B training infrastructure). This module builds
the minimal import surface needed to load the vendored Frank-Wolfe matching
implementations unchanged and call them on raw VGG16 state dicts.
"""

from __future__ import annotations

import importlib.util
import random
import sys
import types
from collections import namedtuple
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

from scripts.lib.alignment.weight_matching import weight_matching as local_weight_matching


ROOT = Path(__file__).resolve().parents[3]
_C2M3_MATCHING_ROOT = ROOT / "external" / "cycle-consistent-model-merging" / "src" / "ccmm" / "matching"

C2M3PermutationSpec = namedtuple(
    "PermutationSpec",
    ["perm_to_layers_and_axes", "layer_and_axes_to_perm"],
)

_MODULES_LOADED = False


def local_spec_to_c2m3(local_spec) -> C2M3PermutationSpec:
    """Convert the local VGG16 permutation spec into the C2M3 spec layout."""

    return C2M3PermutationSpec(
        perm_to_layers_and_axes=dict(local_spec.perm_to_axes),
        layer_and_axes_to_perm=dict(local_spec.axes_to_perm),
    )


def get_canonical_combinations(symbols: Iterable[str]) -> List[Tuple[str, str]]:
    """Return all canonical symbol pairs ``(a, b)`` with ``a < b``."""

    ordered = sorted(symbols)
    return [(ordered[i], ordered[j]) for i in range(len(ordered)) for j in range(i + 1, len(ordered))]


def run_pairwise_frank_wolfe(
    fixed_params: Dict[str, torch.Tensor],
    permutee_params: Dict[str, torch.Tensor],
    permutation_spec: C2M3PermutationSpec,
    *,
    initialization_method: str = "identity",
    max_iter: int = 200,
    num_trials: int = 1,
    device: str = "cpu",
):
    """Run the vendored pairwise Frank-Wolfe matcher on raw state dicts."""

    ensure_c2m3_frank_wolfe_imports()
    from ccmm.matching.frank_wolfe_matching import frank_wolfe_weight_matching

    return frank_wolfe_weight_matching(
        ps=permutation_spec,
        fixed=fixed_params,
        permutee=permutee_params,
        initialization_method=initialization_method,
        max_iter=max_iter,
        return_perm_history=False,
        num_trials=num_trials,
        device=device,
        keep_soft_perms=False,
    )


def run_synchronized_frank_wolfe(
    params_by_symbol: Dict[str, Dict[str, torch.Tensor]],
    permutation_spec: C2M3PermutationSpec,
    *,
    symbols: List[str] | None = None,
    combinations: List[Tuple[str, str]] | None = None,
    initialization_method: str = "identity",
    max_iter: int = 200,
    device: str = "cpu",
):
    """Run the vendored synchronized Frank-Wolfe matcher on raw state dicts."""

    ensure_c2m3_frank_wolfe_imports()
    from ccmm.matching.frank_wolfe_sync_matching import frank_wolfe_synchronized_matching

    if symbols is None:
        symbols = sorted(params_by_symbol.keys())
    if combinations is None:
        combinations = get_canonical_combinations(symbols)

    return frank_wolfe_synchronized_matching(
        params=params_by_symbol,
        perm_spec=permutation_spec,
        symbols=symbols,
        combinations=combinations,
        max_iter=max_iter,
        initialization_method=initialization_method,
        keep_soft_perms=False,
        device=device,
        verbose=False,
    )


def ensure_c2m3_frank_wolfe_imports() -> None:
    """Load the vendored Frank-Wolfe modules behind a minimal compatibility shim."""

    global _MODULES_LOADED
    if _MODULES_LOADED:
        return

    _install_pytorch_lightning_shim()
    _install_ccmm_shims()
    _load_module_from_path(
        "ccmm.matching.frank_wolfe_matching",
        _C2M3_MATCHING_ROOT / "frank_wolfe_matching.py",
    )
    _load_module_from_path(
        "ccmm.matching.frank_wolfe_sync_matching",
        _C2M3_MATCHING_ROOT / "frank_wolfe_sync_matching.py",
    )
    _MODULES_LOADED = True


def _install_pytorch_lightning_shim() -> None:
    try:
        import pytorch_lightning  # noqa: F401
        return
    except ImportError:
        pass

    module = types.ModuleType("pytorch_lightning")

    def seed_everything(seed: int) -> int:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return seed

    module.seed_everything = seed_everything
    module.LightningModule = torch.nn.Module
    sys.modules["pytorch_lightning"] = module


def _install_ccmm_shims() -> None:
    if "ccmm" not in sys.modules:
        ccmm_pkg = types.ModuleType("ccmm")
        ccmm_pkg.__path__ = []  # type: ignore[attr-defined]
        sys.modules["ccmm"] = ccmm_pkg

    if "ccmm.matching" not in sys.modules:
        matching_pkg = types.ModuleType("ccmm.matching")
        matching_pkg.__path__ = []  # type: ignore[attr-defined]
        sys.modules["ccmm.matching"] = matching_pkg

    if "ccmm.utils" not in sys.modules:
        utils_pkg = types.ModuleType("ccmm.utils")
        utils_pkg.__path__ = []  # type: ignore[attr-defined]
        sys.modules["ccmm.utils"] = utils_pkg

    sys.modules["ccmm.matching.permutation_spec"] = _build_permutation_spec_module()
    sys.modules["ccmm.matching.utils"] = _build_matching_utils_module()
    sys.modules["ccmm.matching.weight_matching"] = _build_weight_matching_module()
    sys.modules["ccmm.utils.utils"] = _build_utils_module()


def _build_permutation_spec_module() -> types.ModuleType:
    module = types.ModuleType("ccmm.matching.permutation_spec")
    module.PermutationSpec = C2M3PermutationSpec
    return module


def _build_utils_module() -> types.ModuleType:
    module = types.ModuleType("ccmm.utils.utils")
    module.ModelParams = Dict[str, torch.Tensor]

    def to_np(tensor):
        if isinstance(tensor, torch.Tensor):
            if tensor.nelement() == 1:
                return tensor.item()
            return tensor.detach().cpu().numpy()
        return tensor

    def get_model(model):
        while hasattr(model, "model"):
            model = model.model
        return model

    module.to_np = to_np
    module.get_model = get_model
    return module


def _build_matching_utils_module() -> types.ModuleType:
    module = types.ModuleType("ccmm.matching.utils")
    tensor_alias = torch.Tensor
    module.PermutationMatrix = tensor_alias
    module.PermutationIndices = tensor_alias

    def perm_indices_to_perm_matrix(perm_indices):
        if not isinstance(perm_indices, torch.Tensor):
            perm_indices = torch.as_tensor(perm_indices, dtype=torch.long)
        n = len(perm_indices)
        return torch.eye(n, device=perm_indices.device)[perm_indices.long()]

    def perm_matrix_to_perm_indices(perm_matrix):
        return perm_matrix.nonzero()[:, 1].long()

    def perm_rows(x, perm):
        input_dims = "jklm"[: x.dim()]
        output_dims = "iklm"[: x.dim()]
        return torch.einsum(f"ij,{input_dims}->{output_dims}", perm, x)

    def perm_cols(x, perm):
        x_t = x.transpose(1, 0)
        perm_t = perm.transpose(1, 0)
        return perm_rows(x_t, perm_t).transpose(1, 0)

    def perm_tensor_by_perm_matrix(tens, perm, axis):
        if axis == 0:
            return perm_rows(tens, perm)
        if axis == 1:
            return perm_cols(tens, perm.T)
        raise ValueError(f"Unsupported axis for matrix permutation: {axis}")

    def get_permuted_param(param, perms_to_apply, perm_matrices, except_axis=None):
        for axis, perm_id in enumerate(perms_to_apply):
            if axis == except_axis or perm_id is None:
                continue
            perm = perm_matrices[perm_id]
            if not isinstance(perm, torch.Tensor):
                perm = torch.as_tensor(perm)
            if perm.dim() == 1:
                param = torch.index_select(param, axis, perm.long().to(param.device))
            else:
                param = perm_tensor_by_perm_matrix(param, perm.to(param.device), axis)
        return param

    def generalized_inner_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a_dims = "ijkm"[: a.dim()]
        b_dims = "jnkm"[: b.dim()]
        return torch.einsum(f"{a_dims},{b_dims}->in", a, b)

    module.perm_indices_to_perm_matrix = perm_indices_to_perm_matrix
    module.perm_matrix_to_perm_indices = perm_matrix_to_perm_indices
    module.perm_rows = perm_rows
    module.perm_cols = perm_cols
    module.get_permuted_param = get_permuted_param
    module.generalized_inner_product = generalized_inner_product
    return module


def _build_weight_matching_module() -> types.ModuleType:
    module = types.ModuleType("ccmm.matching.weight_matching")
    local_spec_cls = namedtuple("LocalPermutationSpec", ["perm_to_axes", "axes_to_perm"])

    def solve_linear_assignment_problem(sim_matrix, return_matrix=False):
        from scipy.optimize import linear_sum_assignment

        if isinstance(sim_matrix, torch.Tensor):
            sim_matrix = sim_matrix.detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(sim_matrix, maximize=True)
        assert np.array_equal(row_ind, np.arange(len(row_ind)))
        indices = torch.as_tensor(col_ind, dtype=torch.long)
        if return_matrix:
            return _build_matching_utils_module().perm_indices_to_perm_matrix(indices)
        return indices

    def weight_matching(
        ps,
        fixed,
        permutee,
        max_iter=100,
        init_perm=None,
        alternate_diffusion_params=None,
        layer_iteration_order=None,
        verbose=False,
    ):
        del alternate_diffusion_params, layer_iteration_order
        local_spec = local_spec_cls(
            perm_to_axes=ps.perm_to_layers_and_axes,
            axes_to_perm=ps.layer_and_axes_to_perm,
        )
        local_init = None
        if init_perm is not None:
            local_init = {
                name: value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
                for name, value in init_perm.items()
            }
        perms = local_weight_matching(
            local_spec,
            fixed,
            permutee,
            max_iter=max_iter,
            init_perm=local_init,
            silent=not verbose,
        )
        return {name: torch.as_tensor(indices, dtype=torch.long) for name, indices in perms.items()}

    module.solve_linear_assignment_problem = solve_linear_assignment_problem
    module.weight_matching = weight_matching
    return module


def _load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
