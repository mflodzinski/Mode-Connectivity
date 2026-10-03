"""Task graph for per-cell calibration followed by selected replication."""

from __future__ import annotations


def pair_rows(cfg):
    return [
        dict(replicate=replicate, left_epoch=left, right_epoch=right)
        for replicate in range(len(cfg["seed_pairs"]))
        for left in cfg["stages"]
        for right in cfg["stages"]
    ]


def calibration_rows(cfg):
    """Rows whose methods and hyperparameters are calibrated independently."""
    if cfg.get("calibrate_all_replicates", False):
        return pair_rows(cfg)
    return [
        dict(replicate=0, left_epoch=left, right_epoch=right)
        for left in cfg["stages"]
        for right in cfg["stages"]
    ]


def replication_rows(cfg):
    """Every ordered stage pair for the two held-out seed pairs."""
    if cfg.get("calibrate_all_replicates", False):
        return []
    return [row for row in pair_rows(cfg) if int(row["replicate"]) > 0]


def chunks(values, size):
    return [values[index : index + size] for index in range(0, len(values), size)]


def build_dag(cfg):
    tasks = []

    def add(identifier, operation, dependencies=(), **fields):
        task = dict(
            id=identifier,
            operation=operation,
            dependencies=list(dict.fromkeys(dependencies)),
            **fields,
        )
        tasks.append(task)
        return identifier

    add("prepare", "prepare")
    add("reuse", "reuse", ["prepare"])
    base_ids, branch_ids = [], []
    calibration_chunks = chunks(
        calibration_rows(cfg), int(cfg["calibration_pairs_per_task"])
    )
    candidate_shards = cfg.get("calibration_candidate_shards", {})
    if candidate_shards:
        # Final-endpoint mode uses one pair per chunk. Each method grid is split
        # into isolated candidate shards, followed by one selection/full-budget
        # refit task. Indices are global per operation so Slurm arrays can safely
        # group shards with identical dependencies.
        counters = {operation: 0 for operation in (
            "calibrate_wm", "calibrate_base_grid", "calibrate_base_select",
            "calibrate_branch_grid", "calibrate_branch_select",
        )}

        def indexed(operation, dependencies, **fields):
            index = counters[operation]
            counters[operation] += 1
            return add(
                f"{operation}_{index}", operation, dependencies,
                index=index, **fields,
            )

        for items in calibration_chunks:
            if len(items) != 1:
                raise ValueError(
                    "Parallel candidate shards require calibration_pairs_per_task=1."
                )
            item = items[0]
            wm = indexed("calibrate_wm", ["reuse"], item=item)
            grids = []
            for method in ("wm_scale", "sinkhorn"):
                for shard, candidate_indices in enumerate(candidate_shards[method]):
                    dependencies = [wm] if method == "wm_scale" else ["reuse"]
                    grids.append(indexed(
                        "calibrate_base_grid", dependencies,
                        item=item, method=method, shard=shard,
                        candidate_indices=list(candidate_indices),
                    ))
            base = indexed(
                "calibrate_base_select", [wm, *grids], item=item
            )
            branches = []
            for method in ("sinkhorn_scale_joint", "sinkhorn_scale_finetune"):
                for shard, candidate_indices in enumerate(candidate_shards[method]):
                    branches.append(indexed(
                        "calibrate_branch_grid", [base],
                        item=item, method=method, shard=shard,
                        candidate_indices=list(candidate_indices),
                    ))
            branch = indexed(
                "calibrate_branch_select", [base, *branches], item=item
            )
            base_ids.append(base)
            branch_ids.append(branch)
    else:
        for index, items in enumerate(calibration_chunks):
            base = add(
                f"calibrate_base_{index}", "calibrate_base", ["reuse"],
                index=index, chunk=dict(index=index, items=items),
            )
            branch = add(
                f"calibrate_branch_{index}", "calibrate_branch", [base],
                index=index, chunk=dict(index=index, items=items),
            )
            base_ids.append(base)
            branch_ids.append(branch)
    add("freeze_choices", "freeze_choices", branch_ids)

    replication_chunks = chunks(
        replication_rows(cfg), int(cfg["replication_pairs_per_task"])
    )

    endpoints = [
        dict(seed=seed, epoch=epoch)
        for seed in sum(cfg["seed_pairs"], [])
        for epoch in cfg["stages"]
    ]
    endpoint_ids = []
    for index, items in enumerate(chunks(endpoints, int(cfg["endpoint_pairs_per_task"]))):
        endpoint_ids.append(
            add(
                f"endpoint_chunk_{index}", "endpoint_chunk", ["freeze_choices"],
                index=index, chunk=dict(index=index, items=items),
            )
        )

    add("endpoints_complete", "endpoints_complete", endpoint_ids)
    calibration_evaluation_ids = []
    for index, items in enumerate(chunks(
        calibration_rows(cfg), int(cfg["calibration_evaluation_pairs_per_task"])
    )):
        calibration_evaluation_ids.append(add(
            f"calibration_evaluation_{index}", "calibration_evaluation",
            ["endpoints_complete", "freeze_choices"],
            index=index, chunk=dict(index=index, items=items),
        ))
    replication_evaluation_ids = []
    for index, items in enumerate(replication_chunks):
        replication_evaluation_ids.append(add(
            f"replicate_selected_evaluation_{index}", "replicate_selected_evaluation",
            ["endpoints_complete", "freeze_choices"],
            index=index, chunk=dict(index=index, items=items),
        ))
    add(
        "report", "report",
        calibration_evaluation_ids + replication_evaluation_ids,
    )
    return tasks


def ancestors(tasks, targets):
    by_id = {task["id"]: task for task in tasks}
    selected = set()

    def visit(identifier):
        if identifier in selected:
            return
        selected.add(identifier)
        for dependency in by_id[identifier]["dependencies"]:
            visit(dependency)

    for target in targets:
        visit(target)
    return [task for task in tasks if task["id"] in selected]


def select_tasks(tasks, mode):
    if mode in ("all", "main"):
        return tasks
    if mode == "prepare":
        return ancestors(tasks, ["prepare"])
    if mode == "pilot":
        branches = [
            task for task in tasks
            if task["operation"] in ("calibrate_branch", "calibrate_branch_select")
        ]
        all_items = [
            item
            for task in branches
            for item in (
                task["chunk"]["items"] if "chunk" in task else [task["item"]]
            )
        ]
        stages = sorted({int(item["left_epoch"]) for item in all_items})
        early, middle, final = stages[0], stages[len(stages) // 2], stages[-1]
        target_pairs = {
            (early, early), (early, final), (middle, middle),
            (final, early), (final, final),
        }
        targets = [
            task["id"] for task in branches
            if any(
                (int(item["left_epoch"]), int(item["right_epoch"])) in target_pairs
                for item in (
                    task["chunk"]["items"] if "chunk" in task else [task["item"]]
                )
            )
        ]
        return ancestors(tasks, targets)
    if mode == "calibration":
        return ancestors(tasks, ["freeze_choices"])
    if mode == "evaluate":
        return tasks
    raise ValueError(mode)
