from datetime import datetime
from pathlib import Path
import bisect
import csv
import json
from typing import Any, Dict, Optional, Union


def _as_float_list(values):
    if values is None:
        return None
    return [float(v) for v in values]


def _tensor_to_list(x):
    if x is None:
        return None

    # Avoid importing torch here; check by duck-typing.
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "tolist"):
        try:
            return x.detach().cpu().tolist()
        except Exception:
            pass

    if hasattr(x, "tolist"):
        try:
            return x.tolist()
        except Exception:
            pass

    return x


def _dataset_info(ds):
    if ds is None:
        return None

    info: Dict[str, Any] = {"class": ds.__class__.__name__}

    try:
        info["len"] = len(ds)
    except Exception:
        pass

    for attr in (
        "root",
        "train",
        "image_file",
        "label_file",
        "images_file",
        "labels_file",
    ):
        if hasattr(ds, attr):
            try:
                info[attr] = str(getattr(ds, attr))
            except Exception:
                pass

    return info


def _unwrap_dataset_index(ds, idx: int):
    """Map a top-level dataset index to an underlying dataset + index.

    Supports common wrappers like torch.utils.data.Subset and ConcatDataset.
    """

    idx = int(idx)

    while True:
        # torch.utils.data.Subset
        if ds is not None and hasattr(ds, "dataset") and hasattr(ds, "indices"):
            try:
                idx = int(ds.indices[idx])
                ds = ds.dataset
                continue
            except Exception:
                pass

        # torch.utils.data.ConcatDataset
        if (
            ds is not None
            and hasattr(ds, "datasets")
            and hasattr(ds, "cumulative_sizes")
        ):
            try:
                cumulative_sizes = list(ds.cumulative_sizes)
                dataset_pos = bisect.bisect_right(cumulative_sizes, idx)
                prev_cum = 0 if dataset_pos == 0 else cumulative_sizes[dataset_pos - 1]
                idx = idx - prev_cum
                ds = ds.datasets[dataset_pos]
                continue
            except Exception:
                pass

        break

    return ds, idx


def _sample_id(ds, idx: int) -> str:
    ds, idx = _unwrap_dataset_index(ds, idx)
    idx = int(idx)

    if ds is None:
        return f"idx={idx}"

    # Preferred: custom dataset provides a stable identifier.
    if hasattr(ds, "sample_id") and callable(getattr(ds, "sample_id")):
        try:
            return str(ds.sample_id(idx))
        except Exception:
            pass

    # Common torchvision datasets (ImageFolder/DatasetFolder)
    for attr in ("samples", "imgs"):
        if hasattr(ds, attr):
            try:
                sample = getattr(ds, attr)[idx]
                if isinstance(sample, (tuple, list)) and sample:
                    return str(sample[0])
            except Exception:
                pass

    return f"{ds.__class__.__name__}#idx={idx}"


def export_results(
    *,
    out_dir: Union[str, Path] = "exports",
    run_name: Optional[str] = None,
    train_loss_history=None,
    train_accuracy_history=None,
    test_loss=None,
    test_accuracy=None,
    confusion_matrix=None,
    wrong_indices=None,
    wrong_details=None,
    train_dataset=None,
    test_dataset=None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = run_name or f"run_{ts}"
    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    wrong_items = None
    if wrong_details is not None:
        wrong_items = []
        derived_indices = []
        for item in wrong_details:
            if not isinstance(item, dict):
                continue
            if "index" not in item:
                continue
            idx = int(item["index"])
            derived_indices.append(idx)
            wrong_items.append(
                {
                    "index": idx,
                    "sample": _sample_id(test_dataset, idx),
                    "target": (
                        int(item.get("target"))
                        if "target" in item and item.get("target") is not None
                        else None
                    ),
                    "pred": (
                        int(item.get("pred"))
                        if "pred" in item and item.get("pred") is not None
                        else None
                    ),
                }
            )

        if wrong_indices is None:
            wrong_indices = derived_indices

    wrong_indices = [int(i) for i in (wrong_indices or [])]
    wrong_samples = [_sample_id(test_dataset, i) for i in wrong_indices]

    train_loss_history_f = _as_float_list(train_loss_history)
    train_accuracy_history_f = _as_float_list(train_accuracy_history)

    results = {
        "created_at": ts,
        "train": {
            "loss_history": train_loss_history_f,
            "accuracy_history": train_accuracy_history_f,
            "final_loss": train_loss_history_f[-1] if train_loss_history_f else None,
            "final_accuracy": (
                train_accuracy_history_f[-1] if train_accuracy_history_f else None
            ),
            "dataset": _dataset_info(train_dataset),
        },
        "test": {
            "loss": float(test_loss) if test_loss is not None else None,
            "accuracy": float(test_accuracy) if test_accuracy is not None else None,
            "dataset": _dataset_info(test_dataset),
        },
        "confusion_matrix": _tensor_to_list(confusion_matrix),
        "wrong": {
            "indices": wrong_indices,
            "samples": wrong_samples,
            "items": wrong_items,
        },
        "extra": extra or {},
    }

    with open(run_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    cm_list = results["confusion_matrix"]
    if cm_list is not None:
        with open(
            run_dir / "confusion_matrix.csv", "w", newline="", encoding="utf-8"
        ) as f:
            csv.writer(f).writerows(cm_list)

    with open(run_dir / "wrong_samples.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(wrong_samples))

    return run_dir
