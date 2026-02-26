# callbacks/analysis_callback.py
import pytorch_lightning as pl
import torch
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import gc
from sklearn.metrics import confusion_matrix
from utils import save_attention_map  # utils.py에 해당 함수가 있다고 가정


class SaveAnalysisResultsCallback(pl.Callback):
    def on_test_epoch_end(self, trainer, pl_module):
        """
        테스트 에폭이 끝날 때 CSV 저장 및 시각화를 수행합니다.
        """
        # -------------------------
        # 1. 저장 경로 설정
        # -------------------------
        save_dir = self._get_save_dir(pl_module)
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Callback] Saving analysis results to {save_dir}")

        # -------------------------
        # 2. 테스트 결과 수집
        # -------------------------
        outputs = getattr(pl_module, "test_outputs", None)
        if not outputs:
            print("[Callback] Warning: No test outputs found.")
            return

        # probs: [N, C], labels: [N]
        names = [x["name"] for x in outputs]
        probs = torch.cat([x["probs"] for x in outputs], dim=0).numpy()
        labels = torch.cat([x["label"] for x in outputs], dim=0).numpy()
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
        # 5. Attention Map 저장 (있는 경우만)
        #    MILTrainerModule / DTFDTrainerModule 공통 지원
        # -------------------------
        if getattr(pl_module, "attention_func", False):
            coords_list = [x["coords"] for x in outputs]
            attn_list = [x["attn"] for x in outputs]
            self._save_attention_maps(
                pl_module, save_dir, names, coords_list, attn_list, labels, preds
            )

        # -------------------------
        # 6. 메모리 정리
        # -------------------------
        pl_module.test_outputs.clear()
        gc.collect()

    # ----------------------------------------------------------------------
    # 저장 경로 생성 (base_save_dir 우선)
    # ----------------------------------------------------------------------
    def _get_save_dir(self, pl_module):

        # 1) base_save_dir가 있으면 그걸 기준으로 사용
        base_save_dir = getattr(pl_module, "base_save_dir", None)
        if base_save_dir is not None:
            base_path = Path(base_save_dir)
            save_dir = (
                base_path
                / pl_module.test_dataset_element_name
                / f"seed_{pl_module.seed}"
            )
            return save_dir

        # 2) fallback: 기존 방식(args 기반)
        train_dataset_name_str = (
            "_".join(pl_module.args.train_dataset_name)
            if isinstance(pl_module.args.train_dataset_name, list)
            else pl_module.args.train_dataset_name
        )

        save_dir_parts = [pl_module.args.output_dir, train_dataset_name_str]

        # mixup 사용 시 경로에 mixup 정보 추가
        if getattr(pl_module.args, "use_mixed", False):
            save_dir_parts.append(f"mixup_{pl_module.args.mixup_ratio}")

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
        cm_prob = cm / cm.sum(axis=1, keepdims=True)
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
            plt.savefig(save_dir / suffix, format="jpg")
            plt.close()

    # ----------------------------------------------------------------------
    # Prediction CSV
    # ----------------------------------------------------------------------
    def _save_prediction_csv(self, pl_module, save_dir, names, probs, labels, preds):
        class_names = pl_module.test_class_names_list
        predictions = []

        for name, prob_arr, label, pred in zip(names, probs, labels, preds):
            entropy_val = scipy.stats.entropy(prob_arr).item()
            top_two = np.sort(prob_arr)[-2:]
            margin = round(top_two[-1] - top_two[-2], 4)

            row = {
                "Slide name": name,
                "GT": class_names[label],
                "Pred": class_names[pred],
                "Entropy": round(entropy_val, 4),
                "Margin": margin,
            }
            # 클래스별 confidence
            for idx, cname in enumerate(class_names):
                row[f"Confidence {cname}"] = f"{prob_arr[idx]:.4f}"

            predictions.append(row)

        df = pd.DataFrame(predictions)
        df.to_csv(save_dir / "all_predictions.csv", index=False)

        # 오답만 따로 저장
        df_wrong = df[df["GT"] != df["Pred"]]
        df_wrong.to_csv(save_dir / "wrong_predictions.csv", index=False)

    # ----------------------------------------------------------------------
    # Attention Maps
    # ----------------------------------------------------------------------
    def _save_attention_maps(
        self, pl_module, save_dir, names, coords_list, attn_list, labels, preds
    ):
        save_path = save_dir / "attention_maps"
        print(f"[Callback] Saving {len(names)} attention maps...")

        class_names = pl_module.test_class_names_list
        cnt = 0

        for name, coords, attn, label, pred in zip(
            names, coords_list, attn_list, labels, preds
        ):
            if attn is None:
                continue

            cnt += 1
            label_name = class_names[label]
            pred_name = class_names[pred]

            # 예외처리 그대로 유지
            if label_name == "TVA+VA":
                label_name = "TVA"

            # attn: [1, N] or [N] -> numpy
            attn_np = attn.squeeze().numpy()

            save_attention_map(
                slide_name=name,
                label=label_name,
                pred=pred_name,
                coords=coords,
                attention_map=attn_np,
                patch_size=pl_module.args.patch_size,
                downsample=pl_module.args.downsample,
                patch_path=pl_module.patch_path,
                save_path=save_path,
            )

            if cnt % 50 == 0:
                gc.collect()
