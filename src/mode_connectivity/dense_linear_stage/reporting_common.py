"""Labels and metric names shared by final-alignment reports."""

METHOD_LABELS = {
    "raw": "Raw",
    "wm": "WM",
    "wm_scale": "WM+S",
    "sinkhorn": "Sinkhorn",
    "sinkhorn_scale_joint": "Sinkhorn+S (joint)",
    "sinkhorn_scale_finetune": "Sinkhorn+S (finetune)",
}
SUBSETS = ("train_report", "test_full")
METRICS = ("loss", "error")
