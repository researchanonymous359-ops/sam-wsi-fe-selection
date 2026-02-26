# uncertainty_save.py (전체 교체본)

import numpy as np
from scipy.stats import entropy as shannon_entropy
from pathlib import Path
import pandas as pd
import torch

from torchmetrics.functional.classification import (
    multiclass_auroc,
)
from netcal.metrics import ECE
from sklearn.metrics import accuracy_score


def AULC(accs, uncertainties):
    """
    accs: 0/1 (정답/오답) 1D array
    uncertainties: 1D array (낮을수록 확실)
    """
    idxs = np.argsort(uncertainties)
    error_s = accs[idxs]  # 여기 accs는 0/1(정답)인데 아래에서 error로 쓰고 있어 기존 구현을 유지
    mean_error = error_s.mean() if len(error_s) else 0.0
    if mean_error == 0.0:
        return 0.0, np.array([])
    error_csum = np.cumsum(error_s)
    Fs = error_csum / np.arange(1, len(error_s) + 1)
    s = 1.0 / len(Fs)
    return -1.0 + s * Fs.sum() / mean_error, Fs


def rAULC(uncertainties, accs_bool):
    """
    accs_bool: (GT==Pred) -> True/False
    """
    accs = accs_bool.astype("float")
    perf_aulc, _ = AULC(accs, -accs)  # 이상적 불확실성: 정답이면 낮음(=불확실성 낮음)
    curr_aulc, _ = AULC(accs, uncertainties)
    if perf_aulc == 0.0:
        return np.nan
    return curr_aulc / perf_aulc


def _infer_class_cols(df: pd.DataFrame):
    """
    'Confidence XXX' 컬럼을 자동 탐색해 (클래스명 리스트, 열 리스트) 반환
    """
    conf_cols = [c for c in df.columns if c.startswith("Confidence ")]
    if not conf_cols:
        raise ValueError("No 'Confidence *' columns found in prediction CSV.")
    # 클래스명은 'Confidence ' 이후
    class_names = [c[len("Confidence "):] for c in conf_cols]
    return class_names, conf_cols


def _ensure_uncertainty(df: pd.DataFrame, conf_cols):
    """
    df에 'Uncertainty' 없으면 Shannon entropy로 생성
    """
    if "Uncertainty" in df.columns:
        return df
    probs = df[conf_cols].to_numpy(dtype=float)
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
    df = df.copy()
    df["Uncertainty"] = shannon_entropy(probs, base=2, axis=1)
    return df


def _map_labels(df: pd.DataFrame, class_names):
    """
    GT/Pred 문자열을 class_names 인덱스로 매핑
    """
    name_to_idx = {n: i for i, n in enumerate(class_names)}
    df = df.copy()
    df["GT"] = df["GT"].map(name_to_idx)
    df["Pred"] = df["Pred"].map(name_to_idx)
    # 매핑 실패가 있으면 드롭
    df = df.dropna(subset=["GT", "Pred"]).reset_index(drop=True)
    df["GT"] = df["GT"].astype(int)
    df["Pred"] = df["Pred"].astype(int)
    return df, name_to_idx


def _multiclass_auroc_safe(y_probs_np, y_true_np, num_classes: int):
    """
    torchmetrics.multiclass_auroc 사용. 빈배열/한 클래스만 존재시 NaN 반환.
    """
    if len(y_true_np) == 0:
        return np.nan
    if len(np.unique(y_true_np)) < 2:
        return np.nan
    y_probs = torch.tensor(y_probs_np, dtype=torch.float32)
    y_true = torch.tensor(y_true_np, dtype=torch.long)
    try:
        return float(multiclass_auroc(y_probs, y_true, num_classes=num_classes).cpu().numpy())
    except Exception:
        return np.nan


def _slice_by_quantile(df: pd.DataFrame, q: float):
    """
    'Uncertainty' 기준으로 q 분위수 이하만 남김 (q in [0,1]).
    비어있으면 None 반환.
    """
    if len(df) == 0:
        return None
    thr = df["Uncertainty"].quantile(q)
    sub = df[df["Uncertainty"] <= thr].reset_index(drop=True)
    if len(sub) == 0:
        return None
    return sub


def final_uncertainty_save(dir_path, seeds, dataset, label_type, bins=30):
    """
    dir_path: 저장 폴더 (seed_* 하위에 all_predictions.csv, 최상위에 ensemble_all_predictions.csv 기대)
    seeds: 리스트 (예: [20,40])
    dataset/label_type: 기존 시그니처 유지(로직에서 동적 처리)
    """
    dir_path = Path(dir_path)

    # 수집 벡터
    test_size_fracs = [round(1 - 0.05 * i, 2) for i in range(11)]  # 1.00, 0.95, ..., 0.50
    seed_col, size_col = [], []
    acc_col, auc_col, ece_col, ment_col, raulc_col = [], [], [], [], []

    # -----------------------
    # per-seed 루프
    # -----------------------
    for seed in seeds:
        csv_path = dir_path / f"seed_{seed}" / "all_predictions.csv"
        if not csv_path.exists():
            print(f"[WARN] per-seed CSV not found: {csv_path}")
            continue

        df_full = pd.read_csv(csv_path)
        # 클래스/열 파악
        class_names, conf_cols = _infer_class_cols(df_full)
        num_classes = len(class_names)
        # Uncertainty 생성
        df_full = _ensure_uncertainty(df_full, conf_cols)
        # 라벨 매핑
        df_full, _ = _map_labels(df_full, class_names)

        # 모든 p에 대해 슬라이싱 & 메트릭
        for p in test_size_fracs:
            df = _slice_by_quantile(df_full, p)
            if df is None:
                # 비어있으면 스킵
                continue

            y_probs = df[conf_cols].to_numpy(dtype=float)
            y_probs = y_probs / np.clip(y_probs.sum(axis=1, keepdims=True), 1e-12, None)
            y_true = df["GT"].to_numpy(dtype=int)
            y_pred = df["Pred"].to_numpy(dtype=int)

            # 메트릭
            seed_col.append(seed)
            size_col.append(int(round(p * 100)))

            acc_col.append(round(accuracy_score(y_true, y_pred) * 100, 2))
            auc_val = _multiclass_auroc_safe(y_probs, y_true, num_classes=num_classes)
            auc_col.append(round(auc_val * 100, 2) if not np.isnan(auc_val) else np.nan)

            ece = ECE(bins)
            ece_col.append(round(float(ece.measure(y_probs, y_true)), 4))

            ment_col.append(round(float(shannon_entropy(y_probs, base=2, axis=1).mean()), 4))
            raulc_col.append(round(float(rAULC(df["Uncertainty"].to_numpy(), (y_true == y_pred))), 2))

    # -----------------------
    # 각 p별 평균행 (Average)
    # -----------------------
    for p in test_size_fracs:
        p_pct = int(round(p * 100))
        mask = np.array(size_col) == p_pct
        if not mask.any():
            continue

        seed_col.append("Average")
        size_col.append(p_pct)

        acc_vals = np.array([a for m, a in zip(mask, acc_col) if m], dtype=float)
        auc_vals = np.array([a for m, a in zip(mask, auc_col) if m], dtype=float)
        ece_vals = np.array([a for m, a in zip(mask, ece_col) if m], dtype=float)
        ment_vals = np.array([a for m, a in zip(mask, ment_col) if m], dtype=float)
        raulc_vals = np.array([a for m, a in zip(mask, raulc_col) if m], dtype=float)

        def _avgstd_str(x, fmt=":.2f"):
            x = x[~np.isnan(x)]
            if len(x) == 0:
                return "NaN ± NaN"
            return f"{np.mean(x):{fmt}} ± {np.std(x):{fmt}}"

        acc_col.append(_avgstd_str(acc_vals, ":.2f"))
        auc_col.append(_avgstd_str(auc_vals, ":.2f"))
        ece_col.append(_avgstd_str(ece_vals, ":.4f"))
        ment_col.append(_avgstd_str(ment_vals, ":.4f"))
        raulc_col.append(_avgstd_str(raulc_vals, ":.2f"))

    # -----------------------
    # Ensemble (최상위 CSV)
    # -----------------------
    ens_path = dir_path / "ensemble_all_predictions.csv"
    if ens_path.exists():
        df_full = pd.read_csv(ens_path)
        class_names, conf_cols = _infer_class_cols(df_full)
        num_classes = len(class_names)
        df_full = _ensure_uncertainty(df_full, conf_cols)
        df_full, _ = _map_labels(df_full, class_names)

        for p in test_size_fracs:
            p_pct = int(round(p * 100))
            df = _slice_by_quantile(df_full, p)
            if df is None:
                continue

            seed_col.append("Ensemble")
            size_col.append(p_pct)

            y_probs = df[conf_cols].to_numpy(dtype=float)
            y_probs = y_probs / np.clip(y_probs.sum(axis=1, keepdims=True), 1e-12, None)
            y_true = df["GT"].to_numpy(dtype=int)
            y_pred = df["Pred"].to_numpy(dtype=int)

            acc_col.append(round(accuracy_score(y_true, y_pred) * 100, 2))
            auc_val = _multiclass_auroc_safe(y_probs, y_true, num_classes=num_classes)
            auc_col.append(round(auc_val * 100, 2) if not np.isnan(auc_val) else np.nan)

            ece = ECE(bins)
            ece_col.append(round(float(ece.measure(y_probs, y_true)), 4))
            ment_col.append(round(float(shannon_entropy(y_probs, base=2, axis=1).mean()), 4))
            raulc_col.append(round(float(rAULC(df["Uncertainty"].to_numpy(), (y_true == y_pred))), 2))
    else:
        print(f"[WARN] Ensemble CSV not found: {ens_path}")

    # -----------------------
    # 저장
    # -----------------------
    out = pd.DataFrame({
        "SEED": seed_col,
        "TEST SET SIZE": size_col,
        "ACC": acc_col,
        "AUC": auc_col,
        "ECE": ece_col,
        "mEntropy": ment_col,
        "rAULC": raulc_col,
    })
    out_path = dir_path / "uncertainty_result.csv"
    out.to_csv(out_path, index=False)
    print(f"Final results saved to {out_path}")
