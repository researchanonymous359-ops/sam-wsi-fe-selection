# save_metrics.py (balanced accuracy version)

import pandas as pd
from collections import defaultdict
from pathlib import Path
import numpy as np
import scipy.stats
import json
import traceback

import torch
from torchmetrics.functional.classification import (
    multiclass_auroc,
    multiclass_precision,
    multiclass_recall,
    multiclass_accuracy,
    multiclass_f1_score,
)
import torch.nn.functional as F

from netcal.metrics import ECE

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score


# ----------------------------
# Global state (keep the same keys as before)
# ----------------------------
metrics_dict = dict()
multiclass_metrics_dict = dict()
all_seed_logits_dict = dict()
all_seed_labels_dict = dict()

# Metric names used in seed-level summaries
metric_list = ["Accuracy", "AUROC", "Precision", "Recall", "F1 Score"]


# ----------------------------
# Utils: safe ROC/PR computation (binary)
# ----------------------------
def _safe_roc_auc_binary(y_true, y_score):
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def _safe_pr_auc_binary(y_true, y_score):
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return float("nan")


def initialize_metrics(test_dataset_info):
    """
    Initialize containers for metric storage.
    test_dataset_info: dict like {tde: {...}}
    """
    for test_dataset_element_name in sorted(test_dataset_info.keys()):
        metrics = {"Seed": [], "Metric": [], "Result": []}
        metrics_dict[test_dataset_element_name] = metrics

    for test_dataset_element_name in sorted(test_dataset_info.keys()):
        multiclass_metrics = defaultdict(list, {"Method": [], "Metric": []})
        multiclass_metrics_dict[test_dataset_element_name] = multiclass_metrics

    for test_dataset_element_name in sorted(test_dataset_info.keys()):
        all_seed_logits_dict[test_dataset_element_name] = []
        all_seed_labels_dict[test_dataset_element_name] = []


def make_single_result_metrics(args, seed, trainer_model, test_results, test_dataset_element_name, num_classes):
    """
    Save per-seed single results (Val/Test).
    - Accuracy is computed as class-balanced accuracy (= macro accuracy)
    - AUROC / Precision / Recall / F1 are computed directly via torchmetrics.functional
    - Does not depend on keys inside test_results
    """
    try:
        print(f"\n[INFO] make_single_result_metrics called for seed={seed}, dataset={test_dataset_element_name}")

        # -----------------------
        # Collect y_prob, label and fix shapes
        # -----------------------
        if not hasattr(trainer_model, "y_prob_list") or len(trainer_model.y_prob_list) == 0:
            print(f"[ERROR] y_prob_list is empty for seed={seed}, dataset={test_dataset_element_name}")
            return
        if not hasattr(trainer_model, "label_list") or len(trainer_model.label_list) == 0:
            print(f"[ERROR] label_list is empty for seed={seed}, dataset={test_dataset_element_name}")
            return

        # y_prob_list: [ (1, C), (1, C), ... ]
        # label_list : [ (1,), (1,), ... ]
        y_prob = torch.cat(trainer_model.y_prob_list, dim=0)  # [N, C]
        label = torch.cat(trainer_model.label_list, dim=0)    # [N]

        # Defensive shape handling
        if y_prob.ndim == 3 and y_prob.size(1) == 1:
            y_prob = y_prob.squeeze(1)
        if label.ndim == 2 and label.size(1) == 1:
            label = label.squeeze(1)

        assert y_prob.ndim == 2 and y_prob.shape[1] == num_classes, f"y_prob shape mismatch! {y_prob.shape}"
        assert label.ndim == 1, f"label shape should be [N], but got {label.shape}!"

        # -----------------------
        # ECE (per seed)
        #   - If trainer_model.logits exists, use it directly
        #   - Otherwise compute ECE from y_prob
        # -----------------------
        try:
            ece = ECE(args.n_bins)

            if hasattr(trainer_model, "logits") and trainer_model.logits is not None:
                probs_np = F.softmax(trainer_model.logits, dim=1).detach().cpu().numpy()
                labels_np = trainer_model.labels.detach().cpu().numpy()
            else:
                probs_np = y_prob.detach().cpu().numpy()
                probs_np = probs_np / np.clip(probs_np.sum(axis=1, keepdims=True), 1e-12, None)
                labels_np = label.detach().cpu().numpy()

            _ = ece.measure(probs_np, labels_np)
        except Exception as e:
            print(f"[WARN] Failed to compute ECE (seed={seed}): {e}")

        # -----------------------
        # Seed-level metrics (macro)
        # -----------------------
        try:
            bal_acc = multiclass_accuracy(
                y_prob, label, num_classes=num_classes, average="macro"
            ).item() * 100.0
            auroc = multiclass_auroc(
                y_prob, label, num_classes=num_classes, average="macro"
            ).item() * 100.0
            precision = multiclass_precision(
                y_prob, label, num_classes=num_classes, average="macro"
            ).item() * 100.0
            recall = multiclass_recall(
                y_prob, label, num_classes=num_classes, average="macro"
            ).item() * 100.0
            f1 = multiclass_f1_score(
                y_prob, label, num_classes=num_classes, average="macro"
            ).item() * 100.0

            bal_acc = round(bal_acc, 3)
            auroc = round(auroc, 3)
            precision = round(precision, 3)
            recall = round(recall, 3)
            f1 = round(f1, 3)
        except Exception as e:
            print(f"[ERROR] Failed to compute macro metrics (seed={seed}, dataset={test_dataset_element_name}): {e}")
            traceback.print_exc()
            return

        # Use balanced accuracy as "Accuracy"
        result_list = [bal_acc, auroc, precision, recall, f1]

        # Record seed-level table
        metrics_dict[test_dataset_element_name]["Seed"].extend([int(seed)] * len(metric_list))
        metrics_dict[test_dataset_element_name]["Metric"].extend(metric_list)
        metrics_dict[test_dataset_element_name]["Result"].extend(result_list)

        # -----------------------
        # Collect per-class metrics (only for present classes)
        # -----------------------
        present_classes = np.unique(label.cpu().numpy())
        multiclass_metrics_dict[test_dataset_element_name]["Method"].extend([int(seed)] * len(metric_list))
        multiclass_metrics_dict[test_dataset_element_name]["Metric"].extend(metric_list)

        class_names = getattr(trainer_model, "test_class_names_list", [str(i) for i in range(num_classes)])

        # Pre-compute per-class metrics: Accuracy / AUROC / Precision / Recall / F1
        try:
            acc_all = multiclass_accuracy(y_prob, label, num_classes=num_classes, average=None)  # [C]
            auroc_all = multiclass_auroc(y_prob, label, num_classes=num_classes, average=None)  # [C]
            prec_all = multiclass_precision(y_prob, label, num_classes=num_classes, average=None)  # [C]
            rec_all = multiclass_recall(y_prob, label, num_classes=num_classes, average=None)  # [C]
            f1_all = multiclass_f1_score(y_prob, label, num_classes=num_classes, average=None)  # [C]
        except Exception as e:
            print(f"[WARN] Failed to pre-compute per-class metrics (seed={seed}): {e}")
            traceback.print_exc()
            acc_all = auroc_all = prec_all = rec_all = f1_all = None

        for class_idx, class_name in enumerate(class_names):
            if class_idx in present_classes and acc_all is not None:
                try:
                    acc = round(acc_all[class_idx].item() * 100, 3)
                    auroc_c = round(auroc_all[class_idx].item() * 100, 3)
                    prec = round(prec_all[class_idx].item() * 100, 3)
                    rec = round(rec_all[class_idx].item() * 100, 3)
                    f1_c = round(f1_all[class_idx].item() * 100, 3)
                except Exception as e:
                    print(f"[WARN] Metric calc failed for class {class_name} (idx={class_idx}): {e}")
                    traceback.print_exc()
                    acc = auroc_c = prec = rec = f1_c = -1
            else:
                # Class never appears in this test split
                acc = auroc_c = prec = rec = f1_c = -1

            multiclass_metrics_dict[test_dataset_element_name][class_name].extend(
                [acc, auroc_c, prec, rec, f1_c]
            )

    except Exception:
        print(f"[ERROR] Failed to make result metrics for seed={seed}, dataset={test_dataset_element_name}")
        traceback.print_exc()


def make_whole_result_metrics(
    args,
    test_dataset_element_name,
    num_classes,
    class_names_list,
    save_dir,
    mean_logits,
    final_labels,
    slide_names,
):
    """
    Save final metrics/files using averaged predictions (mean_logits) and labels (final_labels).
    - Accuracy is computed as class-balanced accuracy (= macro accuracy)
    - Does not overwrite class_names_list (Camelyon16 compatibility)
    - Save directory:
        1) If args.base_save_dir exists: base_save_dir / test_dataset_element_name
        2) Else: use save_dir as provided
    """
    try:
        # Prefer the common path created in main
        if getattr(args, "base_save_dir", None) is not None:
            base_dir = Path(args.base_save_dir)
            save_dir = base_dir / test_dataset_element_name
        else:
            save_dir = Path(save_dir)

        save_dir.mkdir(parents=True, exist_ok=True)

        # -------- Append seed-average rows --------
        metrics_df = pd.DataFrame(metrics_dict[test_dataset_element_name])
        for metric in metric_list:
            try:
                vals = metrics_df[metrics_df["Metric"] == metric]["Result"].astype(str)
                numeric_vals = pd.to_numeric(vals, errors="coerce").dropna()
                if len(numeric_vals) == 0:
                    continue
                mean_val_f = numeric_vals.mean()
                std_val_f = numeric_vals.std()
                metrics_dict[test_dataset_element_name]["Seed"].append("Average")
                metrics_dict[test_dataset_element_name]["Metric"].append(metric)
                metrics_dict[test_dataset_element_name]["Result"].append(f"{mean_val_f:.3f} ± {std_val_f:.3f}")
            except Exception as e:
                print(f"[ERROR] Failed to compute average: metric={metric}, error={e}")
                traceback.print_exc()

        # -------- Final (ensemble) predictions --------
        final_preds = torch.argmax(mean_logits, dim=1)

        # Scalar metrics (Accuracy is balanced; others are macro)
        try:
            final_accuracy = multiclass_accuracy(
                mean_logits, final_labels, num_classes=num_classes, average="macro"
            ).item() * 100.0
            final_accuracy = round(final_accuracy, 3)
        except Exception as e:
            print(f"[WARN] Failed to compute ensemble balanced accuracy: {e}")
            final_accuracy = float("nan")

        try:
            final_auroc = round(
                multiclass_auroc(mean_logits, final_labels, num_classes=num_classes, average="macro").item() * 100, 3
            )
        except Exception:
            final_auroc = float("nan")

        final_precision = round(
            multiclass_precision(mean_logits, final_labels, num_classes=num_classes, average="macro").item() * 100, 3
        )
        final_recall = round(
            multiclass_recall(mean_logits, final_labels, num_classes=num_classes, average="macro").item() * 100, 3
        )
        final_f1 = round(
            multiclass_f1_score(mean_logits, final_labels, num_classes=num_classes, average="macro").item() * 100, 3
        )

        # ECE (ensemble)
        ece = ECE(args.n_bins)
        probs_np = mean_logits.detach().cpu().numpy()
        probs_np = probs_np / np.clip(probs_np.sum(axis=1, keepdims=True), 1e-12, None)
        labels_np = final_labels.detach().cpu().numpy()
        ece_score = ece.measure(probs_np, labels_np)

        print("\nFinal Ensemble Results (Balanced Accuracy)")
        print(f"Balanced Accuracy:  {final_accuracy:.4f}")
        print(f"AUROC:              {final_auroc:.4f}")
        print(f"Precision (macro):  {final_precision:.4f}")
        print(f"Recall (macro):     {final_recall:.4f}")
        print(f"F1 Score (macro):   {final_f1:.4f}")
        print(f"ECE:                {ece_score:.4f}\n")

        metrics_dict[test_dataset_element_name]["Seed"].extend(["Ensemble"] * len(metric_list))
        metrics_dict[test_dataset_element_name]["Metric"].extend(metric_list)
        metrics_dict[test_dataset_element_name]["Result"].extend(
            [final_accuracy, final_auroc, final_precision, final_recall, final_f1]
        )

        # -------- Per-class seed-average (mean±std) + ensemble values --------
        multiclass_metrics_df = pd.DataFrame(multiclass_metrics_dict[test_dataset_element_name])

        # Add seed-average rows
        for metric in metric_list:
            multiclass_metrics_dict[test_dataset_element_name]["Method"].append("Average")
            multiclass_metrics_dict[test_dataset_element_name]["Metric"].append(metric)
            for cname in class_names_list:
                try:
                    vals = pd.to_numeric(
                        multiclass_metrics_df[multiclass_metrics_df["Metric"] == metric][cname],
                        errors="coerce",
                    ).replace(-1.0, np.nan)
                    avg = vals.dropna().mean()
                    std = vals.dropna().std()
                    multiclass_metrics_dict[test_dataset_element_name][cname].append(
                        f"{avg:.3f} ± {std:.3f}" if not np.isnan(avg) else "NaN ± NaN"
                    )
                except Exception as e:
                    print(f"[WARN] Failed to save average - metric={metric}, class={cname}: {e}")
                    multiclass_metrics_dict[test_dataset_element_name][cname].append("NaN ± NaN")

        # Add ensemble per-class rows
        multiclass_metrics_dict[test_dataset_element_name]["Method"].extend(["Ensemble"] * len(metric_list))
        multiclass_metrics_dict[test_dataset_element_name]["Metric"].extend(metric_list)

        for class_idx, cname in enumerate(class_names_list):
            for metric in metric_list:
                try:
                    if metric == "Accuracy":
                        value = round(
                            multiclass_accuracy(mean_logits, final_labels, num_classes=num_classes, average=None)[
                                class_idx
                            ].item()
                            * 100,
                            3,
                        )
                    elif metric == "AUROC":
                        value = round(
                            multiclass_auroc(mean_logits, final_labels, num_classes=num_classes, average=None)[
                                class_idx
                            ].item()
                            * 100,
                            3,
                        )
                    elif metric == "Precision":
                        value = round(
                            multiclass_precision(mean_logits, final_labels, num_classes=num_classes, average=None)[
                                class_idx
                            ].item()
                            * 100,
                            3,
                        )
                    elif metric == "Recall":
                        value = round(
                            multiclass_recall(mean_logits, final_labels, num_classes=num_classes, average=None)[
                                class_idx
                            ].item()
                            * 100,
                            3,
                        )
                    elif metric == "F1 Score":
                        value = round(
                            multiclass_f1_score(mean_logits, final_labels, num_classes=num_classes, average=None)[
                                class_idx
                            ].item()
                            * 100,
                            3,
                        )
                    else:
                        value = "NaN"
                except Exception as e:
                    print(f"[WARN] Failed to compute ensemble class metric - class={cname}, metric={metric}: {e}")
                    value = "NaN"

                multiclass_metrics_dict[test_dataset_element_name][cname].append(value)

        # -------- Per-slide prediction CSV --------
        final_preds = torch.argmax(mean_logits, dim=1)
        ensemble_rows = []
        for i, (slide_name, pred, label, probs) in enumerate(zip(slide_names, final_preds, final_labels, mean_logits)):
            try:
                name = slide_name[0] if isinstance(slide_name, (tuple, list)) else slide_name
                row = {
                    "Slide name": name,
                    "GT": class_names_list[label.item()],
                    "Pred": class_names_list[pred.item()],
                }
                p = probs.detach().cpu().numpy()
                p = p / np.clip(p.sum(), 1e-12, None)
                for idx, cname in enumerate(class_names_list[: len(p)]):
                    row[f"Confidence {cname}"] = f"{p[idx]:.4f}"

                # Uncertainty metrics
                entropy_val = scipy.stats.entropy(p).item()
                row["Entropy"] = round(entropy_val, 4)
                top2 = np.sort(p)[-2:] if len(p) >= 2 else np.array([p.max(), 0.0])
                row["Margin"] = round((top2[-1] - top2[-2]).item(), 4)
                msp = float(p.max())
                row["MSP"] = round(msp, 4)
                row["NLC"] = round(-np.log(msp + 1e-12), 4)

                ensemble_rows.append(row)
            except Exception as e:
                print(f"[ERROR] Failed to create prediction row: idx={i}, error={e}")

        pd.DataFrame(ensemble_rows).to_csv(Path(save_dir, "ensemble_all_predictions.csv"), index=False)

        wrong_rows = [r for r in ensemble_rows if r["GT"] != r["Pred"]]
        pd.DataFrame(wrong_rows).to_csv(Path(save_dir, "ensemble_wrong_predictions.csv"), index=False)

        # -------- Confusion Matrix --------
        y_true_np = final_labels.detach().cpu().numpy()
        y_pred_np = final_preds.detach().cpu().numpy()

        labels_idx = list(range(num_classes))
        cm = confusion_matrix(y_true_np, y_pred_np, labels=labels_idx)
        cm_percent = (cm / np.clip(cm.sum(axis=1, keepdims=True), 1e-12, None)) * 100.0
        cm_prob = cm / np.clip(cm.sum(axis=1, keepdims=True), 1e-12, None)

        # Heatmap with count + percent
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
                    fontsize=12,
                    color="black",
                )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Final Ensemble Confusion Matrix (Count + %)")
        plt.xticks(rotation=45)
        plt.savefig(Path(save_dir, "final_confusion_matrix_combined.jpg"), format="jpg")
        plt.close()

        # Pure count/probability heatmaps
        for mat, name, fmt in [
            (cm, "final_confusion_matrix.jpg", "d"),
            (cm_prob, "final_confusion_matrix_prob.jpg", ".2f"),
        ]:
            plt.figure(figsize=(10, 7))
            sns.heatmap(
                mat,
                annot=True,
                fmt=fmt,
                cmap="Blues",
                xticklabels=class_names_list,
                yticklabels=class_names_list,
            )
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(name.replace("_", " ").replace(".jpg", ""))
            plt.savefig(Path(save_dir, name), format="jpg")
            plt.close()

        # -------- Binary results (if needed) --------
        make_binary_result(class_names_list, final_labels, final_preds, save_dir)

        # -------- Save combined final CSV --------
        metrics_df = pd.DataFrame(metrics_dict[test_dataset_element_name])
        multi_df = pd.DataFrame(multiclass_metrics_dict[test_dataset_element_name])

        # Optional dataset-specific column order (kept as-is)
        if test_dataset_element_name in ["data1", "data1+data2", "data3_mixed"]:
            desired_order = ["Method", "Metric"] + ["HP", "SSL", "TSA", "IP", "LP", "TA", "TVA+VA"]
            existing = [c for c in desired_order if c in multi_df.columns]
            if existing:
                multi_df = multi_df[existing + [c for c in multi_df.columns if c not in existing]]

        final_results = pd.concat([multi_df, metrics_df], ignore_index=True)
        final_results.to_csv(Path(save_dir, "final_results.csv"), index=False, float_format="%.3f")

        # (Optional) If an uncertainty summary function exists, call it as-is
        try:
            from uncertainty_save import final_uncertainty_save
            final_uncertainty_save(save_dir, args.seed, test_dataset_element_name, args.label_type)
        except Exception:
            pass

        print("All metrics have been saved successfully!!!")

    except Exception as e:
        print(f"[FATAL ERROR] make_whole_result_metrics failed: {e}")
        traceback.print_exc()


def make_binary_result(class_names_list, final_labels, final_preds, save_dir):
    """
    - If the number of classes is 2, write a standard 2x2 confusion matrix using class_names_list.
    - Otherwise (e.g., 7/8-class), binarize using an Adenoma/Non-Adenoma mapping.
    - Save safely even if only one class appears (1x1 matrix).
    """
    try:
        save_dir = Path(save_dir)
        num_classes = len(class_names_list)

        y_true = final_labels.detach().cpu().numpy()
        y_pred = final_preds.detach().cpu().numpy()

        if num_classes == 2:
            # Binary (e.g., Camelyon16)
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            if cm.shape != (2, 2):
                # Only one class present (1x1) -> save safely
                labels_show = class_names_list[: cm.shape[0]]
            else:
                labels_show = class_names_list

            total = np.clip(cm.sum(), 1e-12, None)
            cm_percent = cm / total * 100.0

            plt.figure(figsize=(7, 6))
            ax = sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=labels_show,
                yticklabels=labels_show,
            )
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(
                        j + 0.5,
                        i + 0.7,
                        f"{cm_percent[i, j]:.2f}%",
                        ha="center",
                        va="bottom",
                        fontsize=12,
                        color="black",
                    )
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("Confusion Matrix (Binary)")
            plt.savefig(Path(save_dir, "binary_confusion_matrix.jpg"), format="jpg")
            plt.close()

            # Print TN, FP, FN, TP when available
            if cm.shape == (2, 2):
                TN, FP, FN, TP = cm.ravel()
                print(f"TN={TN}, FP={FP}, FN={FN}, TP={TP}")
            else:
                print("[WARN] Only one class present in y_true/y_pred; 2x2 CM unavailable.")

        else:
            # 7/8-class -> binarize as Adenoma vs Non-Adenoma
            non_cancer_classes = {"HP", "SSL", "IP", "LP"}
            cancer_classes = {"TSA", "TA", "TVA", "TVA+VA", "Other"}  # include "Other" if needed

            bin_true = np.array([1 if class_names_list[l] in cancer_classes else 0 for l in y_true])
            bin_pred = np.array([1 if class_names_list[p] in cancer_classes else 0 for p in y_pred])

            cm = confusion_matrix(bin_true, bin_pred, labels=[0, 1])
            total = np.clip(cm.sum(), 1e-12, None)
            cm_percent = cm / total * 100.0

            plt.figure(figsize=(7, 6))
            ax = sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Non-Adenoma", "Adenoma"],
                yticklabels=["Non-Adenoma", "Adenoma"],
            )
            for i in range(2):
                for j in range(2):
                    ax.text(
                        j + 0.5,
                        i + 0.7,
                        f"{cm_percent[i, j]:.2f}%",
                        ha="center",
                        va="bottom",
                        fontsize=12,
                        color="black",
                    )
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("Confusion Matrix (Binary: Adenoma vs. Non-Adenoma)")
            plt.savefig(Path(save_dir, "binary_confusion_matrix.jpg"), format="jpg")
            plt.close()

            # Error rates
            if cm.shape == (2, 2):
                TN, FP, FN, TP = cm.ravel()
                total_samples = cm.sum()
                print(f"Type I Error (False Positive): {FP} ({FP / total_samples * 100:.2f}%)")
                print(f"Type II Error (False Negative): {FN} ({FN / total_samples * 100:.2f}%)")

    except Exception as e:
        print(f"[WARN] make_binary_result failed: {e}")
        traceback.print_exc()