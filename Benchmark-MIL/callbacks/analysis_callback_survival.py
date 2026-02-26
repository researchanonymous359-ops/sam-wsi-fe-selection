# callbacks/analysis_callback_survival.py
import pytorch_lightning as pl
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import gc

# KM / log-rank (있으면 사용)
try:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    _HAS_LIFELINES = True
except Exception:
    _HAS_LIFELINES = False

from utils import save_attention_map


class SaveSurvivalAnalysisResultsCallback(pl.Callback):
    """
    Discrete-time (bin/hazard) survival 테스트 결과 저장용 callback.

    pl_module.test_outputs item expected keys:
      - name: str
      - time: float
      - event: int (1=event, 0=censored)
      - risk: float (higher = riskier)
      - expected_time: float (higher = longer survival prediction)
      - cum_event_prob: float (1 - S_end)
      - coords: optional
      - attn: optional
    """

    def on_test_epoch_end(self, trainer, pl_module):
        save_dir = self._get_save_dir(pl_module)
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"[SurvivalCallback] Saving survival analysis results to {save_dir}")

        outputs = getattr(pl_module, "test_outputs", None)
        if not outputs:
            print("[SurvivalCallback] Warning: No test outputs found.")
            return

        # -------------------------
        # 1) gather
        # -------------------------
        names = [str(x.get("name")) for x in outputs]

        times = self._cat_1d(outputs, "time")
        events = self._cat_1d(outputs, "event").astype(int)

        risks = self._cat_1d(outputs, "risk")
        expected_times = self._cat_1d(outputs, "expected_time", allow_missing=True)
        cum_event_probs = self._cat_1d(outputs, "cum_event_prob", allow_missing=True)

        # -------------------------
        # 2) prediction csv 저장 (+ 메타 컬럼 추가)
        # -------------------------
        data = {
            "name": names,
            "time": times,
            "event": events,
            "risk": risks,
            "dataset": str(getattr(pl_module, "test_dataset_element_name", "NA")),
            "seed": int(getattr(pl_module, "seed", -1)),
            "endpoint": str(getattr(pl_module, "survival_endpoint", "NA")),
            "stage": "test",
        }
        if expected_times is not None:
            data["expected_time"] = expected_times
        if cum_event_probs is not None:
            data["cum_event_prob"] = cum_event_probs

        df = pd.DataFrame(data)
        df.to_csv(save_dir / "survival_predictions.csv", index=False)

        # -------------------------
        # 3) epoch-level c-index 저장 (호환성 강화)
        #   - 기존: cindex_test.txt (0~1)
        #   - 추가: cindex_test_percent.txt (0~100)
        #   - 추가: prediction.csv 기반 재계산본(디버깅용)
        # -------------------------
        test_cindex = getattr(pl_module, "test_cindex_epoch", None)
        if test_cindex is not None:
            with open(save_dir / "cindex_test.txt", "w") as f:
                f.write(str(float(test_cindex)) + "\n")
            with open(save_dir / "cindex_test_percent.txt", "w") as f:
                f.write(str(float(test_cindex) * 100.0) + "\n")

        val_cindex = getattr(pl_module, "val_cindex_epoch", None)
        if val_cindex is not None:
            with open(save_dir / "cindex_val.txt", "w") as f:
                f.write(str(float(val_cindex)) + "\n")
            with open(save_dir / "cindex_val_percent.txt", "w") as f:
                f.write(str(float(val_cindex) * 100.0) + "\n")

        # prediction.csv 기반 재계산(정의 일치 확인용)
        # save_metrics_survival._compute_cindex와 동일 의미(높은 risk = 더 위험)
        try:
            from save_metrics_survival import _compute_cindex as _compute_cindex_local

            cidx01 = _compute_cindex_local(times, events, risks)
            if np.isfinite(cidx01):
                with open(save_dir / "cindex_test_recomputed.txt", "w") as f:
                    f.write(str(float(cidx01)) + "\n")
                with open(save_dir / "cindex_test_recomputed_percent.txt", "w") as f:
                    f.write(str(float(cidx01) * 100.0) + "\n")
        except Exception as e:
            print(f"[SurvivalCallback] Warning: recompute c-index failed: {e}")

        # -------------------------
        # 4) cutpoints / bin centers 저장
        # -------------------------
        cutpoints = getattr(pl_module, "cutpoints", None)
        bin_centers = getattr(pl_module, "bin_centers", None)
        if isinstance(cutpoints, torch.Tensor):
            np.savetxt(save_dir / "bin_cutpoints.txt", cutpoints.detach().cpu().numpy())
        if isinstance(bin_centers, torch.Tensor):
            np.savetxt(save_dir / "bin_centers.txt", bin_centers.detach().cpu().numpy())

        # risk 방향 메타 저장 (나중에 KM 해석 실수 방지)
        risk_higher_is_riskier = bool(getattr(pl_module, "risk_higher_is_riskier", True))
        with open(save_dir / "risk_definition.txt", "w") as f:
            f.write(f"risk_higher_is_riskier={risk_higher_is_riskier}\n")

        # -------------------------
        # 5) optional: KM plot + log-rank
        # -------------------------
        if _HAS_LIFELINES:
            self._save_km_plots(save_dir, df, risk_higher_is_riskier=risk_higher_is_riskier)

        # -------------------------
        # 6) optional: attention maps
        # -------------------------
        if getattr(pl_module, "attention_func", False):
            coords_list = [x.get("coords", None) for x in outputs]
            attn_list = [x.get("attn", None) for x in outputs]
            self._save_attention_maps(pl_module, save_dir, names, coords_list, attn_list)

        # -------------------------
        # 7) cleanup
        # -------------------------
        pl_module.test_outputs.clear()
        gc.collect()

    # ----------------------------------------------------------------------
    # 저장 경로 생성
    # ----------------------------------------------------------------------
    def _get_save_dir(self, pl_module):
        base_save_dir = getattr(pl_module, "base_save_dir", None)
        if base_save_dir is not None:
            base_path = Path(base_save_dir)
            save_dir = base_path / pl_module.test_dataset_element_name / f"seed_{pl_module.seed}"
            return save_dir

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
                "survival",
            ]
        )
        return Path(*save_dir_parts)

    # ----------------------------------------------------------------------
    # helpers
    # ----------------------------------------------------------------------
    def _cat_1d(self, outputs, key: str, allow_missing: bool = False) -> np.ndarray | None:
        vals = []
        for x in outputs:
            if key not in x:
                if allow_missing:
                    return None
                raise KeyError(f"[SurvivalCallback] '{key}' not found in test_outputs item.")
            v = x.get(key, None)
            if v is None:
                if allow_missing:
                    return None
                raise KeyError(f"[SurvivalCallback] '{key}' is None in test_outputs item.")

            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().view(-1)
                vals.append(v)
            else:
                vals.append(torch.tensor([float(v)], dtype=torch.float32))
        return torch.cat(vals, dim=0).numpy()

    # ----------------------------------------------------------------------
    # Kaplan-Meier plots (median split + optional quantile split)
    # ----------------------------------------------------------------------
    def _save_km_plots(self, save_dir: Path, df: pd.DataFrame, risk_higher_is_riskier: bool = True):
        if len(df) < 10:
            return

        # 1) Median split
        self._save_km_plot_one(
            save_dir=save_dir,
            df=df,
            split_mode="median",
            thr=float(df["risk"].median()),
            risk_higher_is_riskier=risk_higher_is_riskier,
            filename="km_median_split.jpg",
            thr_filename="risk_threshold_median.txt",
            p_filename="logrank_pvalue_median.txt",
        )

        # 2) Optional: Quartile split (top/bottom 25%) if enough samples
        q_low = float(df["risk"].quantile(0.25))
        q_high = float(df["risk"].quantile(0.75))
        low = df[df["risk"] <= q_low]
        high = df[df["risk"] >= q_high]
        if len(low) >= 10 and len(high) >= 10:
            df_q = pd.concat([low, high], axis=0).copy()
            with open(save_dir / "risk_threshold_quartile.txt", "w") as f:
                f.write(f"q25={q_low}\nq75={q_high}\n")

            self._save_km_plot_one(
                save_dir=save_dir,
                df=df_q,
                split_mode="quartile",
                thr=None,
                risk_higher_is_riskier=risk_higher_is_riskier,
                filename="km_quartile_split.jpg",
                thr_filename=None,
                p_filename="logrank_pvalue_quartile.txt",
                quartile=(q_low, q_high),
            )

    def _save_km_plot_one(
        self,
        save_dir: Path,
        df: pd.DataFrame,
        split_mode: str,
        thr: float | None,
        risk_higher_is_riskier: bool,
        filename: str,
        thr_filename: str | None,
        p_filename: str,
        quartile: tuple[float, float] | None = None,
    ):
        kmf = KaplanMeierFitter()
        plt.figure(figsize=(8, 6))

        if split_mode == "median":
            assert thr is not None
            df = df.copy()
            if risk_higher_is_riskier:
                df["group"] = np.where(df["risk"] >= thr, "high_risk", "low_risk")
            else:
                df["group"] = np.where(df["risk"] >= thr, "low_risk", "high_risk")

            groups = ["low_risk", "high_risk"]

            if thr_filename is not None:
                with open(save_dir / thr_filename, "w") as f:
                    f.write(str(thr) + "\n")

        elif split_mode == "quartile":
            assert quartile is not None
            q_low, q_high = quartile
            df = df.copy()
            if risk_higher_is_riskier:
                df["group"] = np.where(df["risk"] >= q_high, "high_risk", "low_risk")
            else:
                df["group"] = np.where(df["risk"] >= q_high, "low_risk", "high_risk")
            groups = ["low_risk", "high_risk"]
        else:
            return

        for g in groups:
            sub = df[df["group"] == g]
            if len(sub) == 0:
                continue
            kmf.fit(durations=sub["time"], event_observed=sub["event"], label=g)
            kmf.plot()

        # log-rank p-value
        p = None
        try:
            low = df[df["group"] == "low_risk"]
            high = df[df["group"] == "high_risk"]
            if len(low) >= 2 and len(high) >= 2:
                res = logrank_test(
                    low["time"], high["time"],
                    event_observed_A=low["event"],
                    event_observed_B=high["event"],
                )
                p = float(res.p_value)
        except Exception:
            p = None

        title = f"Kaplan–Meier ({split_mode} split by risk)"
        if p is not None:
            title += f" | log-rank p={p:.3e}"
            with open(save_dir / p_filename, "w") as f:
                f.write(str(p) + "\n")

        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Survival probability")
        plt.tight_layout()
        plt.savefig(save_dir / filename, format="jpg")
        plt.close()

    # ----------------------------------------------------------------------
    # Attention Maps
    # ----------------------------------------------------------------------
    def _save_attention_maps(self, pl_module, save_dir, names, coords_list, attn_list):
        save_path = save_dir / "attention_maps"
        save_path.mkdir(parents=True, exist_ok=True)
        print(f"[SurvivalCallback] Saving {len(names)} attention maps...")

        cnt = 0
        for name, coords, attn in zip(names, coords_list, attn_list):
            if attn is None or coords is None:
                continue

            cnt += 1
            attn_np = attn.squeeze().detach().cpu().numpy()

            save_attention_map(
                slide_name=name,
                label="NA",
                pred="NA",
                coords=coords,
                attention_map=attn_np,
                patch_size=pl_module.args.patch_size,
                downsample=pl_module.args.downsample,
                patch_path=pl_module.patch_path,
                save_path=save_path,
            )

            if cnt % 50 == 0:
                gc.collect()
