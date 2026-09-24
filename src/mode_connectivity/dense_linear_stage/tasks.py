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
    """Every ordered stage pair for the calibration seed pair (0, 1)."""
    return [
        dict(replicate=0, left_epoch=left, right_epoch=right)
        for left in cfg["stages"]
        for right in cfg["stages"]
    ]


def replication_rows(cfg):
    """Every ordered stage pair for the two held-out seed pairs."""
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
        branches = [task for task in tasks if task["operation"] == "calibrate_branch"]
        all_items = [item for task in branches for item in task["chunk"]["items"]]
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
                for item in task["chunk"]["items"]
            )
        ]
        return ancestors(tasks, targets)
    if mode == "calibration":
        return ancestors(tasks, ["freeze_choices"])
    if mode == "evaluate":
        return tasks
    raise ValueError(mode)
