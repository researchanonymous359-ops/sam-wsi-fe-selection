# save_metrics_grading.py
import pandas as pd
from collections import defaultdict
from pathlib import Path
import numpy as np
import traceback

import torch
from torchmetrics.functional.classification import multiclass_accuracy

from sklearn.metrics import confusion_matrix, cohen_kappa_score

import matplotlib.pyplot as plt
import seaborn as sns


# ----------------------------
# Global state
# ----------------------------
metrics_dict = dict()
perclass_dict = dict()
all_seed_logits_dict = dict()
all_seed_labels_dict = dict()

# Column order for the summary table (in the desired order)
metric_list_qwk = ["QWK", "Balanced Accuracy"]
metric_list_acc = ["Balanced Accuracy", "QWK"]


def _to_numpy_probs_and_labels(y_prob: torch.Tensor, y_true: torch.Tensor):
    """
    y_prob: [N, C] (probabilities or logits)
    y_true: [N]
    """
    if y_prob.ndim == 3 and y_prob.size(1) == 1:
        y_prob = y_prob.squeeze(1)
    if y_true.ndim == 2 and y_true.size(1) == 1:
        y_true = y_true.squeeze(1)

    probs = y_prob.detach().cpu()
    labels = y_true.detach().cpu().long()
    return probs, labels


def _safe_qwk(y_true_np, y_pred_np):
    """
    sklearn's cohen_kappa_score can warn/return NaN when only one class exists.
    """
    try:
        if len(np.unique(y_true_np)) < 2 and len(np.unique(y_pred_np)) < 2:
            return 1.0
        return float(cohen_kappa_score(y_true_np, y_pred_np, weights="quadratic"))
    except Exception:
        return float("nan")


def initialize_grading_metrics(test_dataset_info):
    for tde in sorted(test_dataset_info.keys()):
        metrics_dict[tde] = {"Seed": [], "Metric": [], "Result": []}
        perclass_dict[tde] = defaultdict(list, {"Method": [], "Metric": []})
        all_seed_logits_dict[tde] = []
        all_seed_labels_dict[tde] = []


def make_single_result_grading_metrics(args, seed, trainer_model, test_results, test_dataset_element_name, num_classes):
    """
    Save per-seed single results:
      - QWK (quadratic weighted kappa)
      - Balanced Accuracy (macro)
    """
    try:
        if not hasattr(trainer_model, "y_prob_list") or len(trainer_model.y_prob_list) == 0:
            print(f"[ERROR] y_prob_list is empty for seed={seed}, dataset={test_dataset_element_name}")
            return
        if not hasattr(trainer_model, "label_list") or len(trainer_model.label_list) == 0:
            print(f"[ERROR] label_list is empty for seed={seed}, dataset={test_dataset_element_name}")
            return

        y_prob = torch.cat(trainer_model.y_prob_list, dim=0)  # [N, C] (prob)
        y_true = torch.cat(trainer_model.label_list, dim=0)   # [N]
        y_prob, y_true = _to_numpy_probs_and_labels(y_prob, y_true)

        y_pred = torch.argmax(y_prob, dim=1)

        bal_acc = multiclass_accuracy(y_prob, y_true, num_classes=num_classes, average="macro").item()
        bal_acc = round(bal_acc * 100.0, 3)

        qwk = _safe_qwk(y_true.numpy(), y_pred.numpy())
        qwk = round(qwk * 100.0, 3)

        # Store (QWK + Balanced Accuracy)
        metrics_dict[test_dataset_element_name]["Seed"].extend([int(seed)] * len(metric_list_qwk))
        metrics_dict[test_dataset_element_name]["Metric"].extend(metric_list_qwk)
        metrics_dict[test_dataset_element_name]["Result"].extend([qwk, bal_acc])

        # Store per-class accuracy
        class_names = getattr(trainer_model, "test_class_names_list", [str(i) for i in range(num_classes)])
        perclass_dict[test_dataset_element_name]["Method"].append(int(seed))
        perclass_dict[test_dataset_element_name]["Metric"].append("Per-class Accuracy")

        try:
            acc_all = multiclass_accuracy(y_prob, y_true, num_classes=num_classes, average=None)  # [C]
            for ci, cname in enumerate(class_names):
                val = round(acc_all[ci].item() * 100.0, 3)
                perclass_dict[test_dataset_element_name][cname].append(val)
        except Exception:
            for ci, cname in enumerate(class_names):
                perclass_dict[test_dataset_element_name][cname].append("NaN")

    except Exception as e:
        print(f"[ERROR] make_single_result_grading_metrics failed: seed={seed}, dataset={test_dataset_element_name}, err={e}")
        traceback.print_exc()


def make_whole_result_grading_metrics(
    args,
    test_dataset_element_name,
    num_classes,
    class_names_list,
    save_dir,
    mean_logits,      # passed in as mean_probs (probabilities)
    final_labels,
    slide_names,
):
    """
    ✅ Final output format (reflecting requirements)
    - final_results_summary.csv:
        Seed, Balanced Accuracy, QWK
        20, ...
        40, ...
        Average, "xx.xxx ± yy.yyy", "aa.aaa ± bb.bbb"
        Ensemble, ...
    - final_results_per_class_acc.csv:
        Seed + (class_names)
    - ensemble_all_predictions.csv / ensemble_wrong_predictions.csv
    - final_confusion_matrix_combined.jpg
    """
    try:
        # ----------------------------
        # Resolve save_dir
        # ----------------------------
        if getattr(args, "base_save_dir", None) is not None:
            base_dir = Path(args.base_save_dir)
            save_dir = base_dir / test_dataset_element_name
        else:
            save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # ----------------------------
        # 1) Append Average (mean ± std) (remove separate Average_std row)
        # ----------------------------
        metrics_df = pd.DataFrame(metrics_dict[test_dataset_element_name])  # Seed, Metric, Result

        def _append_average_pm(metric_name: str):
            sub = metrics_df[metrics_df["Metric"] == metric_name].copy()

            # Compute mean/std only from numeric seeds (exclude Average/Ensemble/etc.)
            sub["Seed_num"] = pd.to_numeric(sub["Seed"], errors="coerce")
            sub = sub.dropna(subset=["Seed_num"])

            vals = pd.to_numeric(sub["Result"], errors="coerce").dropna()
            if len(vals) == 0:
                return

            mean_val = float(vals.mean())
            std_val = float(vals.std())  # pandas default ddof=1 -> keeps the usual "±" feel
            pm_str = f"{mean_val:.3f} ± {std_val:.3f}"

            metrics_dict[test_dataset_element_name]["Seed"].append("Average")
            metrics_dict[test_dataset_element_name]["Metric"].append(metric_name)
            metrics_dict[test_dataset_element_name]["Result"].append(pm_str)

        for m in metric_list_qwk:
            _append_average_pm(m)

        # ----------------------------
        # 2) Compute and append ensemble metrics
        # ----------------------------
        probs = mean_logits.detach().cpu()          # [N, C] probabilities
        y_true = final_labels.detach().cpu().long() # [N]
        y_pred = torch.argmax(probs, dim=1)

        bal_acc = multiclass_accuracy(probs, y_true, num_classes=num_classes, average="macro").item()
        bal_acc = round(bal_acc * 100.0, 3)

        qwk = _safe_qwk(y_true.numpy(), y_pred.numpy())
        qwk = round(qwk * 100.0, 3)

        metrics_dict[test_dataset_element_name]["Seed"].extend(["Ensemble"] * 2)
        metrics_dict[test_dataset_element_name]["Metric"].extend(["QWK", "Balanced Accuracy"])
        metrics_dict[test_dataset_element_name]["Result"].extend([qwk, bal_acc])

        # Refresh metrics_df after appending
        metrics_df = pd.DataFrame(metrics_dict[test_dataset_element_name])

        # ----------------------------
        # 3) Save a clean pivot summary
        #    (pivot is safer than pivot_table because Average is a string)
        # ----------------------------
        summary_df = metrics_df.pivot(index="Seed", columns="Metric", values="Result").reset_index()

        # Seed order: numeric seeds -> Average -> Ensemble
        def _seed_sort_key(x):
            try:
                return (0, int(str(x)))
            except Exception:
                pass
            if str(x) == "Average":
                return (1, 0)
            if str(x) == "Ensemble":
                return (2, 0)
            return (9, 0)

        summary_df = summary_df.sort_values(by="Seed", key=lambda s: s.map(_seed_sort_key))

        # Fix column order: Seed, Balanced Accuracy, QWK
        col_order = ["Seed"]
        for c in ["Balanced Accuracy", "QWK"]:
            if c in summary_df.columns:
                col_order.append(c)
        for c in summary_df.columns:
            if c not in col_order:
                col_order.append(c)
        summary_df = summary_df[col_order]

        summary_df.to_csv(save_dir / "final_results_summary.csv", index=False)

        # ----------------------------
        # 4) Save per-class accuracy separately (one row per seed)
        # ----------------------------
        per_df = pd.DataFrame(perclass_dict[test_dataset_element_name]) if len(perclass_dict[test_dataset_element_name]) > 0 else pd.DataFrame()
        if not per_df.empty:
            per_df = per_df[per_df["Metric"] == "Per-class Accuracy"].copy()
            per_df = per_df.drop(columns=["Metric"], errors="ignore")
            per_df = per_df.rename(columns={"Method": "Seed"})

            # Sort by seed
            per_df["Seed"] = per_df["Seed"].astype(int)
            per_df = per_df.sort_values("Seed")

            class_cols = [c for c in class_names_list if c in per_df.columns]
            per_df = per_df[["Seed"] + class_cols]
            per_df.to_csv(save_dir / "final_results_per_class_acc.csv", index=False)

        # ----------------------------
        # 5) Save ensemble prediction CSVs
        # ----------------------------
        rows = []
        for slide_name, pred, gt, p in zip(slide_names, y_pred, y_true, probs):
            name = slide_name[0] if isinstance(slide_name, (tuple, list)) else slide_name
            row = {
                "Slide name": name,
                "GT": class_names_list[int(gt.item())],
                "Pred": class_names_list[int(pred.item())],
                "GT_idx": int(gt.item()),
                "Pred_idx": int(pred.item()),
            }
            p_np = p.numpy()
            p_np = p_np / np.clip(p_np.sum(), 1e-12, None)
            for ci, cname in enumerate(class_names_list[: len(p_np)]):
                row[f"Confidence {cname}"] = float(p_np[ci])
            rows.append(row)

        pred_df = pd.DataFrame(rows)
        pred_df.to_csv(save_dir / "ensemble_all_predictions.csv", index=False)
        pred_df[pred_df["GT_idx"] != pred_df["Pred_idx"]].to_csv(save_dir / "ensemble_wrong_predictions.csv", index=False)

        # ----------------------------
        # 6) Confusion matrix (count + %)
        # ----------------------------
        cm = confusion_matrix(y_true.numpy(), y_pred.numpy(), labels=list(range(num_classes)))
        cm_percent = (cm / np.clip(cm.sum(axis=1, keepdims=True), 1e-12, None)) * 100.0

        plt.figure(figsize=(10, 7))
        ax = sns.heatmap(
            cm,
            annot=False,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names_list,
            yticklabels=class_names_list,
        )
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    f"{cm[i, j]}\n{cm_percent[i, j]:.2f}%",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="black",
                )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Final Ensemble Confusion Matrix (Count + %)")
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_dir / "final_confusion_matrix_combined.jpg", format="jpg")
        plt.close()

        print(f"[Saved] {save_dir / 'final_results_summary.csv'}")
        if not per_df.empty:
            print(f"[Saved] {save_dir / 'final_results_per_class_acc.csv'}")
        print("All grading metrics have been saved successfully!!!")

    except Exception as e:
        print(f"[FATAL ERROR] make_whole_result_grading_metrics failed: {e}")
        traceback.print_exc()