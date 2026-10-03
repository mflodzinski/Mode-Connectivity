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
    """Final-endpoint pairs calibrated independently on validation data."""
    return pair_rows(cfg)


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
    base_ids, branch_ids = [], []
    calibration_chunks = chunks(
        calibration_rows(cfg), int(cfg["calibration_pairs_per_task"])
    )
    candidate_shards = cfg["calibration_candidate_shards"]
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
            raise ValueError("Final alignment requires one endpoint pair per calibration task.")
        item = items[0]
        wm = indexed("calibrate_wm", ["prepare"], item=item)
        grids = []
        for method in ("wm_scale", "sinkhorn"):
            for shard, candidate_indices in enumerate(candidate_shards[method]):
                dependencies = [wm] if method == "wm_scale" else ["prepare"]
                grids.append(indexed(
                    "calibrate_base_grid", dependencies,
                    item=item, method=method, shard=shard,
                    candidate_indices=list(candidate_indices),
                ))
        base = indexed("calibrate_base_select", [wm, *grids], item=item)
        branches = []
        for method in ("sinkhorn_scale_joint", "sinkhorn_scale_finetune"):
            for shard, candidate_indices in enumerate(candidate_shards[method]):
                branches.append(indexed(
                    "calibrate_branch_grid", [base],
                    item=item, method=method, shard=shard,
                    candidate_indices=list(candidate_indices),
                ))
        branch = indexed("calibrate_branch_select", [base, *branches], item=item)
        base_ids.append(base)
        branch_ids.append(branch)
    add("freeze_choices", "freeze_choices", branch_ids)

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
    add("report", "report", calibration_evaluation_ids)
    return tasks
