# save_metrics.py (balanced accuracy 버전)

import pandas as pd
from collections import defaultdict
from pathlib import Path
import numpy as np
import scipy.stats
import json
import traceback

import torch
from torchmetrics.functional.classification import (
    multiclass_auroc, multiclass_precision, multiclass_recall,
    multiclass_accuracy, multiclass_f1_score
)
import torch.nn.functional as F

from netcal.metrics import ECE

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score


# ----------------------------
# 전역 상태 (기존과 동일한 키 사용)
# ----------------------------
metrics_dict = dict()
multiclass_metrics_dict = dict()
all_seed_logits_dict = dict()
all_seed_labels_dict = dict()

# seed별 요약에서 사용할 메트릭 이름 셋
metric_list = ["Accuracy", "AUROC", "Precision", "Recall", "F1 Score"]


# ----------------------------
# 유틸: 안전한 ROC/PR 계산 (이진)
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
    메트릭 저장용 컨테이너 초기화.
    test_dataset_info: {tde: {...}} 형태
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
    각 시드별 단일 결과(Val/Test) 저장.
    - Accuracy는 class-balanced accuracy(=macro accuracy)로 계산
    - AUROC / Precision / Recall / F1 도 torchmetrics.functional로 직접 계산
    - test_results 딕셔너리의 키에는 의존하지 않음
    """
    try:
        print(f"\n[INFO] make_single_result_metrics called for seed={seed}, dataset={test_dataset_element_name}")

        # -----------------------
        # y_prob, label shape 정리
        # -----------------------
        if not hasattr(trainer_model, "y_prob_list") or len(trainer_model.y_prob_list) == 0:
            print(f"[ERROR] y_prob_list is empty for seed={seed}, dataset={test_dataset_element_name}")
            return
        if not hasattr(trainer_model, "label_list") or len(trainer_model.label_list) == 0:
            print(f"[ERROR] label_list is empty for seed={seed}, dataset={test_dataset_element_name}")
            return

        # y_prob_list: [ (1, C), (1, C), ... ] 형태
        # label_list : [ (1,), (1,), ... ] 형태
        y_prob = torch.cat(trainer_model.y_prob_list, dim=0)  # [N, C]
        label = torch.cat(trainer_model.label_list, dim=0)    # [N]

        # 혹시라도 차원이 꼬였을 경우 방어 코딩
        if y_prob.ndim == 3 and y_prob.size(1) == 1:
            y_prob = y_prob.squeeze(1)
        if label.ndim == 2 and label.size(1) == 1:
            label = label.squeeze(1)

        assert y_prob.ndim == 2 and y_prob.shape[1] == num_classes, f"y_prob shape mismatch! {y_prob.shape}"
        assert label.ndim == 1, f"label shape should be [N], but got {label.shape}!"

        # -----------------------
        # ECE (개별 시드)
        #   - trainer_model.logits가 있다면 그대로 사용
        #   - 없다면 y_prob를 사용하여 ECE 계산
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
            print(f"[WARN] ECE 계산 실패 (seed={seed}): {e}")

        # -----------------------
        # seed-level metric 계산 (macro 기준)
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
            print(f"[ERROR] macro metric 계산 실패 (seed={seed}, dataset={test_dataset_element_name}): {e}")
            traceback.print_exc()
            return

        # Accuracy는 Balanced Accuracy로 사용
        result_list = [bal_acc, auroc, precision, recall, f1]

        # seed-level 테이블에 기록
        metrics_dict[test_dataset_element_name]["Seed"].extend([int(seed)] * len(metric_list))
        metrics_dict[test_dataset_element_name]["Metric"].extend(metric_list)
        metrics_dict[test_dataset_element_name]["Result"].extend(result_list)

        # -----------------------
        # 클래스별 메트릭 수집 (존재 클래스만)
        # -----------------------
        present_classes = np.unique(label.cpu().numpy())
        multiclass_metrics_dict[test_dataset_element_name]["Method"].extend([int(seed)] * len(metric_list))
        multiclass_metrics_dict[test_dataset_element_name]["Metric"].extend(metric_list)

        class_names = getattr(trainer_model, "test_class_names_list", [str(i) for i in range(num_classes)])

        # per-class metrics: Accuracy / AUROC / Precision / Recall / F1
        try:
            acc_all = multiclass_accuracy(
                y_prob, label, num_classes=num_classes, average=None
            )  # [C]
            auroc_all = multiclass_auroc(
                y_prob, label, num_classes=num_classes, average=None
            )  # [C]
            prec_all = multiclass_precision(
                y_prob, label, num_classes=num_classes, average=None
            )  # [C]
            rec_all = multiclass_recall(
                y_prob, label, num_classes=num_classes, average=None
            )  # [C]
            f1_all = multiclass_f1_score(
                y_prob, label, num_classes=num_classes, average=None
            )  # [C]
        except Exception as e:
            print(f"[WARN] per-class metric pre-compute 실패 (seed={seed}): {e}")
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
                # 해당 클래스가 아예 등장하지 않은 경우
                acc = auroc_c = prec = rec = f1_c = -1

            # 딕셔너리에 클래스별 컬럼 생성 + 값 추가
            multiclass_metrics_dict[test_dataset_element_name][class_name].extend(
                [acc, auroc_c, prec, rec, f1_c]
            )

    except Exception as e:
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
    slide_names
):
    """
    시드 평균된 예측(mean_logits)과 라벨(final_labels)로 최종 메트릭/파일 저장.
    - Accuracy는 class-balanced accuracy(=macro accuracy)로 계산
    - class_names_list를 덮어쓰지 않음 (Camelyon16 대응)
    - 저장 경로:
        1) args.base_save_dir 가 있으면: base_save_dir / test_dataset_element_name
        2) 없으면: 인자로 넘어온 save_dir 그대로 사용
    """
    try:
        # 🔥 main에서 만든 공통 경로 우선 사용
        if getattr(args, "base_save_dir", None) is not None:
            base_dir = Path(args.base_save_dir)
            save_dir = base_dir / test_dataset_element_name
        else:
            # fallback: 함수 인자로 받은 경로 사용
            save_dir = Path(save_dir)

        save_dir.mkdir(parents=True, exist_ok=True)

        # -------- Seed 평균 행 추가 --------
        metrics_df = pd.DataFrame(metrics_dict[test_dataset_element_name])
        for metric in metric_list:
            try:
                mean_val = metrics_df[metrics_df['Metric'] == metric]['Result'].astype(str)
                # 숫자/문자 섞일 수 있으므로, 숫자만 골라서 평균/표준편차 계산
                numeric_vals = pd.to_numeric(mean_val, errors='coerce').dropna()
                if len(numeric_vals) == 0:
                    continue
                mean_val_f = numeric_vals.mean()
                std_val_f = numeric_vals.std()
                metrics_dict[test_dataset_element_name]["Seed"].append('Average')
                metrics_dict[test_dataset_element_name]["Metric"].append(metric)
                metrics_dict[test_dataset_element_name]["Result"].append(f"{mean_val_f:.3f} ± {std_val_f:.3f}")
            except Exception as e:
                print(f"[ERROR] 평균 계산 실패: metric={metric}, error={e}")
                traceback.print_exc()

        # -------- 최종(엔상블) 예측 계산 --------
        final_preds = torch.argmax(mean_logits, dim=1)

        # ★ 스칼라 메트릭 (Accuracy는 balanced / 나머지는 macro)
        try:
            final_accuracy = multiclass_accuracy(
                mean_logits, final_labels, num_classes=num_classes, average="macro"
            ).item() * 100.0
            final_accuracy = round(final_accuracy, 3)
        except Exception as e:
            print(f"[WARN] ensemble balanced accuracy 계산 실패: {e}")
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

        # ECE (엔상블)
        ece = ECE(args.n_bins)
        probs_np = mean_logits.detach().cpu().numpy()
        probs_np = probs_np / np.clip(probs_np.sum(axis=1, keepdims=True), 1e-12, None)  # 확률 정규화
        labels_np = final_labels.detach().cpu().numpy()
        ece_score = ece.measure(probs_np, labels_np)

        print("\nFinal Ensemble Results (Balanced Accuracy)")
        print(f"Balanced Accuracy:  {final_accuracy:.4f}")
        print(f"AUROC:              {final_auroc:.4f}")
        print(f"Precision (macro):  {final_precision:.4f}")
        print(f"Recall (macro):     {final_recall:.4f}")
        print(f"F1 Score (macro):   {final_f1:.4f}")
        print(f"ECE:                {ece_score:.4f}\n")

        metrics_dict[test_dataset_element_name]["Seed"].extend(['Ensemble'] * len(metric_list))
        metrics_dict[test_dataset_element_name]["Metric"].extend(metric_list)
        metrics_dict[test_dataset_element_name]["Result"].extend(
            [final_accuracy, final_auroc, final_precision, final_recall, final_f1]
        )

        # -------- 클래스별 평균(시드) + 엔상블 값 --------
        multiclass_metrics_df = pd.DataFrame(multiclass_metrics_dict[test_dataset_element_name])

        # 평균(시드) 행 추가
        for metric in metric_list:
            multiclass_metrics_dict[test_dataset_element_name]["Method"].append('Average')
            multiclass_metrics_dict[test_dataset_element_name]["Metric"].append(metric)
            for cname in class_names_list:
                try:
                    vals = pd.to_numeric(
                        multiclass_metrics_df[multiclass_metrics_df['Metric'] == metric][cname],
                        errors='coerce'
                    ).replace(-1.0, np.nan)
                    avg = vals.dropna().mean()
                    std = vals.dropna().std()
                    multiclass_metrics_dict[test_dataset_element_name][cname].append(
                        f"{avg:.3f} ± {std:.3f}" if not np.isnan(avg) else "NaN ± NaN"
                    )
                except Exception as e:
                    print(f"[WARN] 평균 저장 실패 - metric={metric}, class={cname}: {e}")
                    multiclass_metrics_dict[test_dataset_element_name][cname].append("NaN ± NaN")

        # 엔상블 클래스별 값
        multiclass_metrics_dict[test_dataset_element_name]["Method"].extend(['Ensemble'] * len(metric_list))
        multiclass_metrics_dict[test_dataset_element_name]["Metric"].extend(metric_list)

        for class_idx, cname in enumerate(class_names_list):
            for metric in metric_list:
                try:
                    if metric == "Accuracy":
                        value = round(multiclass_accuracy(
                            mean_logits, final_labels, num_classes=num_classes, average=None
                        )[class_idx].item() * 100, 3)
                    elif metric == "AUROC":
                        value = round(multiclass_auroc(
                            mean_logits, final_labels, num_classes=num_classes, average=None
                        )[class_idx].item() * 100, 3)
                    elif metric == "Precision":
                        value = round(multiclass_precision(
                            mean_logits, final_labels, num_classes=num_classes, average=None
                        )[class_idx].item() * 100, 3)
                    elif metric == "Recall":
                        value = round(multiclass_recall(
                            mean_logits, final_labels, num_classes=num_classes, average=None
                        )[class_idx].item() * 100, 3)
                    elif metric == "F1 Score":
                        value = round(multiclass_f1_score(
                            mean_logits, final_labels, num_classes=num_classes, average=None
                        )[class_idx].item() * 100, 3)
                    else:
                        value = "NaN"
                except Exception as e:
                    print(f"[WARN] ensemble class metric 계산 실패 - class={cname}, metric={metric}: {e}")
                    value = "NaN"

                multiclass_metrics_dict[test_dataset_element_name][cname].append(value)

        # -------- 슬라이드별 예측 CSV --------
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
                p = p / np.clip(p.sum(), 1e-12, None)  # softmax 안전
                for idx, cname in enumerate(class_names_list[:len(p)]):
                    row[f"Confidence {cname}"] = f"{p[idx]:.4f}"

                # 불확실성 지표
                entropy_val = scipy.stats.entropy(p).item()
                row["Entropy"] = round(entropy_val, 4)
                top2 = np.sort(p)[-2:] if len(p) >= 2 else np.array([p.max(), 0.0])
                row["Margin"] = round((top2[-1] - top2[-2]).item(), 4)
                msp = float(p.max())
                row["MSP"] = round(msp, 4)
                row["NLC"] = round(-np.log(msp + 1e-12), 4)

                ensemble_rows.append(row)
            except Exception as e:
                print(f"[ERROR] prediction row 생성 실패: idx={i}, error={e}")

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

        # 카운트+퍼센트 통합 heatmap
        plt.figure(figsize=(10, 7))
        ax = sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
                         xticklabels=class_names_list, yticklabels=class_names_list)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j + 0.5, i + 0.5, f"{cm[i, j]}\n{cm_percent[i, j]:.2f}%",
                        ha='center', va='center', fontsize=12, color="black")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Final Ensemble Confusion Matrix (Count + %)")
        plt.xticks(rotation=45)
        plt.savefig(Path(save_dir, "final_confusion_matrix_combined.jpg"), format="jpg")
        plt.close()

        # 순수 카운트/확률 히트맵
        for mat, name, fmt in [
            (cm, "final_confusion_matrix.jpg", "d"),
            (cm_prob, "final_confusion_matrix_prob.jpg", ".2f"),
        ]:
            plt.figure(figsize=(10, 7))
            sns.heatmap(mat, annot=True, fmt=fmt, cmap="Blues",
                        xticklabels=class_names_list, yticklabels=class_names_list)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(name.replace("_", " ").replace(".jpg", ""))
            plt.savefig(Path(save_dir, name), format="jpg")
            plt.close()

        # -------- 이진 결과(필요 시) --------
        make_binary_result(class_names_list, final_labels, final_preds, save_dir)

        # -------- 최종 CSV 묶어 저장 --------
        metrics_df = pd.DataFrame(metrics_dict[test_dataset_element_name])
        multi_df = pd.DataFrame(multiclass_metrics_dict[test_dataset_element_name])

        # 필요 시 특정 데이터셋 전용 열 순서(세그인 전용) 유지
        if test_dataset_element_name in ["data1", "data1+data2", "data3_mixed"]:
            desired_order = ['Method', 'Metric'] + ['HP', 'SSL', 'TSA', 'IP', 'LP', 'TA', 'TVA+VA']
            existing = [c for c in desired_order if c in multi_df.columns]
            if existing:
                multi_df = multi_df[existing + [c for c in multi_df.columns if c not in existing]]

        final_results = pd.concat([multi_df, metrics_df], ignore_index=True)
        final_results.to_csv(Path(save_dir, "final_results.csv"), index=False, float_format="%.3f")

        # (선택) 불확실성 요약 함수가 있다면 그대로 호출
        try:
            from uncertainty_save import final_uncertainty_save
            final_uncertainty_save(save_dir, args.seed, test_dataset_element_name, args.label_type)
        except Exception:
            pass

        print("All metrics have been saved successfully!!!")

    except Exception as e:
        print(f"[FATAL ERROR] make_whole_result_metrics 실패: {e}")
        traceback.print_exc()


def make_binary_result(class_names_list, final_labels, final_preds, save_dir):
    """
    - 클래스가 2개면 그대로 2x2 혼동행렬 작성 (라벨은 class_names_list 사용)
    - 그 외(세그인 7/8클래스)는 Adenoma/Non-Adenoma 매핑으로 이진화
    - 1x1 행렬(한 클래스만 등장)일 때도 에러 없이 저장
    """
    try:
        save_dir = Path(save_dir)
        num_classes = len(class_names_list)

        y_true = final_labels.detach().cpu().numpy()
        y_pred = final_preds.detach().cpu().numpy()

        if num_classes == 2:
            # Camelyon16 등 이진
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            if cm.shape != (2, 2):
                # 한 클래스만 존재한 경우(1x1) → 안전 저장
                labels_show = class_names_list[:cm.shape[0]]
            else:
                labels_show = class_names_list

            total = np.clip(cm.sum(), 1e-12, None)
            cm_percent = cm / total * 100.0

            plt.figure(figsize=(7, 6))
            ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                             xticklabels=labels_show, yticklabels=labels_show)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j + 0.5, i + 0.7, f"{cm_percent[i, j]:.2f}%",
                            ha='center', va='bottom', fontsize=12, color="black")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("Confusion Matrix (Binary)")
            plt.savefig(Path(save_dir, "binary_confusion_matrix.jpg"), format="jpg")
            plt.close()

            # TN, FP, FN, TP 출력(가능할 때만)
            if cm.shape == (2, 2):
                TN, FP, FN, TP = cm.ravel()
                print(f"TN={TN}, FP={FP}, FN={FN}, TP={TP}")
            else:
                print("[WARN] Only one class present in y_true/y_pred; 2x2 CM unavailable.")

        else:
            # 세그인 7/8 클래스 → Adenoma vs Non-Adenoma 매핑
            non_cancer_classes = {"HP", "SSL", "IP", "LP"}
            cancer_classes = {"TSA", "TA", "TVA", "TVA+VA", "Other"}  # Other는 필요 시 포함

            bin_true = np.array([1 if class_names_list[l] in cancer_classes else 0 for l in y_true])
            bin_pred = np.array([1 if class_names_list[p] in cancer_classes else 0 for p in y_pred])

            cm = confusion_matrix(bin_true, bin_pred, labels=[0, 1])
            total = np.clip(cm.sum(), 1e-12, None)
            cm_percent = cm / total * 100.0

            plt.figure(figsize=(7, 6))
            ax = sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-Adenoma", "Adenoma"],
                yticklabels=["Non-Adenoma", "Adenoma"]
            )
            for i in range(2):
                for j in range(2):
                    ax.text(j + 0.5, i + 0.7, f"{cm_percent[i, j]:.2f}%",
                            ha='center', va='bottom', fontsize=12, color="black")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("Confusion Matrix (Binary: Adenoma vs. Non-Adenoma)")
            plt.savefig(Path(save_dir, "binary_confusion_matrix.jpg"), format="jpg")
            plt.close()

            # 에러율 출력
            if cm.shape == (2, 2):
                TN, FP, FN, TP = cm.ravel()
                total_samples = cm.sum()
                print(f"Type I Error (False Positive): {FP} ({FP / total_samples * 100:.2f}%)")
                print(f"Type II Error (False Negative): {FN} ({FN / total_samples * 100:.2f}%)")

    except Exception as e:
        print(f"[WARN] make_binary_result failed: {e}")
        traceback.print_exc()
