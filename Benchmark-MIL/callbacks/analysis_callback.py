# callbacks/analysis_callback.py
# -*- coding: utf-8 -*-

import gc
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scipy.stats
import seaborn as sns
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from utils import save_attention_map  # assumed to exist in utils.py


class SaveAnalysisResultsCallback(pl.Callback):
    """
    Callback to save analysis results (confusion matrix, prediction CSV, attention maps) at test end
    """

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """
        Save CSVs and visualizations at the end of the test epoch.
        """
        # -------------------------
        # 1. Setup save directory
        # -------------------------
        save_dir = self._get_save_dir(pl_module)
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Callback] Saving analysis results to {save_dir}")

        # -------------------------
        # 2. Collect test outputs
        # -------------------------
        outputs: Optional[List[Dict[str, Any]]] = getattr(pl_module, "test_outputs", None)
        if not outputs:
            print("[Callback] Warning: No test outputs found.")
            return

        # outputs item expected keys:
        # - "name": str
        # - "probs": torch.Tensor [B, C] or [1, C]
        # - "label": torch.Tensor [B] or [1]
        # - (optional) "coords", "attn"
        names = [x["name"] for x in outputs]

        probs_t = torch.cat([x["probs"] for x in outputs], dim=0)
        labels_t = torch.cat([x["label"] for x in outputs], dim=0)

        # Move to CPU in case tensors are on GPU
        probs = probs_t.detach().cpu().numpy()
        labels = labels_t.detach().cpu().numpy()
        preds = np.argmax(probs, axis=1)

        # -------------------------
        # 3. Save confusion matrices
        # -------------------------
        self._save_confusion_matrix(pl_module, save_dir, labels, preds)

        # -------------------------
        # 4. Save prediction CSVs
        # -------------------------
        self._save_prediction_csv(pl_module, save_dir, names, probs, labels, preds)

        # -------------------------
        # 5. Save attention maps (if available)
        #    Common support for MILTrainerModule / DTFDTrainerModule
        # -------------------------
        if bool(getattr(pl_module, "attention_func", False)):
            coords_list = [x.get("coords", None) for x in outputs]
            attn_list = [x.get("attn", None) for x in outputs]
            self._save_attention_maps(pl_module, save_dir, names, coords_list, attn_list, labels, preds)

        # -------------------------
        # 6. Cleanup memory
        # -------------------------
        try:
            pl_module.test_outputs.clear()
        except Exception:
            # might not be a list in some setups
            setattr(pl_module, "test_outputs", [])
        gc.collect()

    # ----------------------------------------------------------------------
    # Build save directory (prefer base_save_dir)
    # ----------------------------------------------------------------------
    def _get_save_dir(self, pl_module: "pl.LightningModule") -> Path:
        # 1) Prefer base_save_dir if provided
        base_save_dir = getattr(pl_module, "base_save_dir", None)
        if base_save_dir is not None:
            base_path = Path(base_save_dir)
            save_dir = base_path / pl_module.test_dataset_element_name / f"seed_{pl_module.seed}"
            return save_dir

        # 2) Fallback: previous args-based scheme
        train_dataset_name_str = (
            "_".join(pl_module.args.train_dataset_name)
            if isinstance(pl_module.args.train_dataset_name, list)
            else pl_module.args.train_dataset_name
        )

        save_dir_parts = [pl_module.args.output_dir, train_dataset_name_str]

        # Append mixup info if enabled
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
    def _save_confusion_matrix(
        self,
        pl_module: "pl.LightningModule",
        save_dir: Path,
        labels: np.ndarray,
        preds: np.ndarray,
    ) -> None:
        class_names: List[str] = pl_module.test_class_names_list
        num_classes = len(class_names)

        cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))

        # row-normalized confusion matrix
        row_sum = cm.sum(axis=1, keepdims=True)
        cm_prob = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=(row_sum != 0))

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
            plt.savefig(save_dir / suffix, format="jpg", dpi=200)
            plt.close()

    # ----------------------------------------------------------------------
    # Prediction CSV
    # ----------------------------------------------------------------------
    def _save_prediction_csv(
        self,
        pl_module: "pl.LightningModule",
        save_dir: Path,
        names: List[str],
        probs: np.ndarray,
        labels: np.ndarray,
        preds: np.ndarray,
    ) -> None:
        class_names: List[str] = pl_module.test_class_names_list
        predictions: List[Dict[str, Any]] = []

        for name, prob_arr, label, pred in zip(names, probs, labels, preds):
            # entropy
            entropy_val = float(scipy.stats.entropy(prob_arr))

            # margin = top1 - top2
            sorted_probs = np.sort(prob_arr)
            top1 = float(sorted_probs[-1])
            top2 = float(sorted_probs[-2]) if len(sorted_probs) >= 2 else 0.0
            margin = round(top1 - top2, 4)

            row: Dict[str, Any] = {
                "Slide name": name,
                "GT": class_names[int(label)],
                "Pred": class_names[int(pred)],
                "Entropy": round(entropy_val, 4),
                "Margin": margin,
            }

            # per-class confidence
            for idx, cname in enumerate(class_names):
                row[f"Confidence {cname}"] = f"{prob_arr[idx]:.4f}"

            predictions.append(row)

        df = pd.DataFrame(predictions)
        df.to_csv(save_dir / "all_predictions.csv", index=False, encoding="utf-8-sig")

        # Save wrong predictions only
        df_wrong = df[df["GT"] != df["Pred"]]
        df_wrong.to_csv(save_dir / "wrong_predictions.csv", index=False, encoding="utf-8-sig")

    # ----------------------------------------------------------------------
    # Attention Maps
    # ----------------------------------------------------------------------
    def _save_attention_maps(
        self,
        pl_module: "pl.LightningModule",
        save_dir: Path,
        names: List[str],
        coords_list: List[Any],
        attn_list: List[Any],
        labels: np.ndarray,
        preds: np.ndarray,
    ) -> None:
        save_path = save_dir / "attention_maps"
        save_path.mkdir(parents=True, exist_ok=True)
        print(f"[Callback] Saving {len(names)} attention maps...")

        class_names: List[str] = pl_module.test_class_names_list
        cnt = 0

        for name, coords, attn, label, pred in zip(names, coords_list, attn_list, labels, preds):
            if attn is None:
                continue

            cnt += 1
            label_name = class_names[int(label)]
            pred_name = class_names[int(pred)]

            # Keep the original exception handling
            if label_name == "TVA+VA":
                label_name = "TVA"

            # convert to numpy safely: attn: [1, N] or [N] -> numpy
            if isinstance(attn, torch.Tensor):
                attn_np = attn.detach().squeeze().cpu().numpy()
            else:
                attn_np = np.asarray(attn).squeeze()

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