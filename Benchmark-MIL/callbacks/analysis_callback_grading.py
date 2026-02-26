# callbacks/analysis_callback_grading.py
import pytorch_lightning as pl
import torch
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import gc
from sklearn.metrics import confusion_matrix, cohen_kappa_score


class SaveGradingAnalysisResultsCallback(pl.Callback):
    """
    Grading 전용 분석 콜백
    - test_outputs에서 (probs, label, name) 수집
    - confusion matrix 저장
    - all_predictions / wrong_predictions 저장
    - QWK(Quadratic Weighted Kappa) + Balanced Accuracy summary 저장
    """

    def on_test_epoch_end(self, trainer, pl_module):
        # -------------------------
        # 1. 저장 경로
        # -------------------------
        save_dir = self._get_save_dir(pl_module)
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Callback][Grading] Saving analysis results to {save_dir}")

        # -------------------------
        # 2. 테스트 결과 수집
        # -------------------------
        outputs = getattr(pl_module, "test_outputs", None)
        if not outputs:
            print("[Callback][Grading] Warning: No test outputs found.")
            return

        names = [x["name"] for x in outputs]  # list[str]
        probs = torch.cat([x["probs"] for x in outputs], dim=0).numpy()   # [N,C]
        labels = torch.cat([x["label"] for x in outputs], dim=0).numpy()  # [N]
        preds = np.argmax(probs, axis=1)

        # -------------------------
        # 3. Confusion Matrix 저장
        # -------------------------
        self._save_confusion_matrix(pl_module, save_dir, labels, preds)

        # -------------------------
        # 4. Prediction CSV 저장
        # -------------------------
        self._save_prediction_csv(pl_module, save_dir, names, probs, labels, preds)

        # -------------------------
        # 5. Summary metric 저장 (QWK + Balanced Acc)
        # -------------------------
        self._save_grading_summary(pl_module, save_dir, labels, preds)

        # -------------------------
        # 6. 메모리 정리
        # -------------------------
        pl_module.test_outputs.clear()
        gc.collect()

    # ----------------------------------------------------------------------
    # 저장 경로 생성 (base_save_dir 우선)
    # ----------------------------------------------------------------------
    def _get_save_dir(self, pl_module):
        """
        classification callback과 동일한 규칙:
        base_save_dir가 있으면:
          base_save_dir / test_dataset_element_name / seed_{seed}
        """
        base_save_dir = getattr(pl_module, "base_save_dir", None)
        if base_save_dir is not None:
            base_path = Path(base_save_dir)
            save_dir = base_path / pl_module.test_dataset_element_name / f"seed_{pl_module.seed}"
            return save_dir

        # fallback (기존 방식)
        train_dataset_name_str = (
            "_".join(pl_module.args.train_dataset_name)
            if isinstance(pl_module.args.train_dataset_name, list)
            else pl_module.args.train_dataset_name
        )

        save_dir_parts = [pl_module.args.output_dir, train_dataset_name_str]

        save_dir_parts.extend(
            [
                pl_module.test_dataset_element_name,
                pl_module.args.resolution_str,
                str(pl_module.args.patch_size),
                pl_module.args.mil_model,
                pl_module.args.feature_extractor,
                f"seed_{pl_module.seed}",
            ]
        )
        return Path(*save_dir_parts)

    # ----------------------------------------------------------------------
    # Confusion Matrix
    # ----------------------------------------------------------------------
    def _save_confusion_matrix(self, pl_module, save_dir, labels, preds):
        class_names = pl_module.test_class_names_list
        cm = confusion_matrix(labels, preds, labels=range(len(class_names)))

        # row-normalized
        cm_prob = cm / np.clip(cm.sum(axis=1, keepdims=True), 1e-12, None)
        cm_prob = np.nan_to_num(cm_prob)

        for matrix, suffix, fmt in [
            (cm, "confusion_matrix.jpg", "d"),
            (cm_prob, "confusion_matrix_prob.jpg", ".2f"),
        ]:
            plt.figure(figsize=(10, 7))
            sns.heatmap(
                matrix,
                annot=True,
                fmt=fmt,
                cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names,
            )
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"Seed: {pl_module.seed}")
            plt.tight_layout()
            plt.savefig(save_dir / suffix, format="jpg")
            plt.close()

    # ----------------------------------------------------------------------
    # Prediction CSV
    # ----------------------------------------------------------------------
    def _save_prediction_csv(self, pl_module, save_dir, names, probs, labels, preds):
        class_names = pl_module.test_class_names_list
        predictions = []

        for name, prob_arr, label, pred in zip(names, probs, labels, preds):
            # 안전 정규화 (혹시 합이 1이 아닐 수 있음)
            prob_arr = prob_arr / np.clip(prob_arr.sum(), 1e-12, None)

            entropy_val = float(scipy.stats.entropy(prob_arr))
            top_two = np.sort(prob_arr)[-2:] if len(prob_arr) >= 2 else np.array([prob_arr.max(), 0.0])
            margin = float(top_two[-1] - top_two[-2])

            row = {
                "Slide name": name,
                "GT": class_names[int(label)],
                "Pred": class_names[int(pred)],
                "GT_idx": int(label),
                "Pred_idx": int(pred),
                "Entropy": round(entropy_val, 4),
                "Margin": round(margin, 4),
                "MSP": round(float(prob_arr.max()), 4),
            }

            for idx, cname in enumerate(class_names):
                row[f"Confidence {cname}"] = float(prob_arr[idx])

            predictions.append(row)

        df = pd.DataFrame(predictions)
        df.to_csv(save_dir / "all_predictions.csv", index=False)

        df_wrong = df[df["GT_idx"] != df["Pred_idx"]]
        df_wrong.to_csv(save_dir / "wrong_predictions.csv", index=False)

    # ----------------------------------------------------------------------
    # Summary metric (QWK + Balanced Acc)
    # ----------------------------------------------------------------------
    def _save_grading_summary(self, pl_module, save_dir, labels, preds):
        """
        - QWK: quadratic weighted kappa
        - Balanced Acc: macro accuracy (per-class recall 평균)
        """
        num_classes = len(pl_module.test_class_names_list)

        # Balanced Accuracy (macro accuracy) = mean per-class recall
        cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
        per_class_recall = np.diag(cm) / np.clip(cm.sum(axis=1), 1e-12, None)
        bal_acc = float(np.nanmean(per_class_recall)) * 100.0

        # QWK (quadratic)
        try:
            if len(np.unique(labels)) < 2 and len(np.unique(preds)) < 2:
                qwk = 1.0
            else:
                qwk = float(cohen_kappa_score(labels, preds, weights="quadratic"))
        except Exception:
            qwk = float("nan")

        summary = {
            "Seed": [pl_module.seed],
            "Dataset": [pl_module.test_dataset_element_name],
            "QWK": [round(qwk * 100.0, 3)],
            "BalancedAccuracy": [round(bal_acc, 3)],
            "NumSamples": [int(len(labels))],
        }
        pd.DataFrame(summary).to_csv(save_dir / "grading_test_summary.csv", index=False)

        print(
            f"[Callback][Grading][seed={pl_module.seed}] "
            f"QWK={summary['QWK'][0]:.3f}  "
            f"BalancedAcc={summary['BalancedAccuracy'][0]:.3f}  "
            f"N={summary['NumSamples'][0]}"
        )
