# save_metrics_survival.py
from __future__ import annotations

import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# ✅ 내부 구현(이미 학습 로그에서 쓰고 있는 것)으로 c-index 계산
from pl_model.forward_fn_survival import concordance_index as _torch_concordance_index


# ----------------------------
# 전역 상태
# ----------------------------
surv_metrics_dict = dict()         # dataset -> {"Seed":[], "Metric":[], "Result":[]}
all_seed_risk_dict = dict()        # dataset -> [risk_tensor(N,), ...]
all_seed_time_dict = dict()        # dataset -> [time_tensor(N,), ...]
all_seed_event_dict = dict()       # dataset -> [event_tensor(N,), ...]
all_seed_names_dict = dict()       # dataset -> [names_list(N), ...]

surv_metric_list = ["C-index"]


# ----------------------------
# util
# ----------------------------
def _to_1d_cpu_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().view(-1)
    return torch.tensor([float(x)], dtype=torch.float32)


def _compute_cindex_torch(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor, signature: str = "risk_time_event") -> float:
    """
    ✅ pl_model.forward_fn_survival.concordance_index를 그대로 사용
    risk: higher => riskier (shorter survival)
    """
    # 안전 처리
    risk = risk.detach().cpu().view(-1)
    time = time.detach().cpu().view(-1)
    event = event.detach().cpu().view(-1)

    # event 값이 한쪽만 있으면 c-index 자체가 정의가 애매/불가능 -> nan
    ev = event.numpy()
    if len(np.unique(ev)) < 2:
        return float("nan")

    c = _torch_concordance_index(risk, time, event, signature=signature)
    if not torch.is_tensor(c) or not torch.isfinite(c):
        return float("nan")
    return float(c.detach().cpu())


def initialize_survival_metrics(test_dataset_info):
    for tde in sorted(test_dataset_info.keys()):
        surv_metrics_dict[tde] = {"Seed": [], "Metric": [], "Result": []}
        all_seed_risk_dict[tde] = []
        all_seed_time_dict[tde] = []
        all_seed_event_dict[tde] = []
        all_seed_names_dict[tde] = []


# ----------------------------
# seed 단일 결과 저장
# ----------------------------
def make_single_result_survival_metrics(
    args,
    seed,
    trainer_model,
    test_results,
    test_dataset_element_name,
):
    """
    trainer_model 에 다음이 있어야 함:
      - risk_list, time_list, event_list, names (선택)
    """
    try:
        print(f"\n[INFO] make_single_result_survival_metrics called for seed={seed}, dataset={test_dataset_element_name}")

        # 1) 가장 신뢰도 높은 값: trainer_model.test_cindex_epoch (이미 로그로 찍히는 값)
        #    있으면 그걸 그대로 쓰자 (외부 라이브러리 의존 X)
        cidx_epoch = getattr(trainer_model, "test_cindex_epoch", None)
        if cidx_epoch is not None and np.isfinite(float(cidx_epoch)):
            cindex = float(cidx_epoch) * 100.0
        else:
            # 2) fallback: risk/time/event으로 내부 concordance_index로 계산
            for k in ["risk_list", "time_list", "event_list"]:
                if not hasattr(trainer_model, k) or len(getattr(trainer_model, k)) == 0:
                    print(f"[ERROR] {k} is empty for seed={seed}, dataset={test_dataset_element_name}")
                    return

            risk = torch.cat([_to_1d_cpu_tensor(x) for x in trainer_model.risk_list], dim=0)
            time = torch.cat([_to_1d_cpu_tensor(x) for x in trainer_model.time_list], dim=0)
            event = torch.cat([_to_1d_cpu_tensor(x) for x in trainer_model.event_list], dim=0)

            cindex_raw = _compute_cindex_torch(
                risk=risk, time=time, event=event,
                signature=str(getattr(args, "survival_cindex_signature", "risk_time_event")).lower()
            )
            cindex = cindex_raw * 100.0

        cindex = round(float(cindex), 3) if np.isfinite(float(cindex)) else float("nan")

        surv_metrics_dict[test_dataset_element_name]["Seed"].append(int(seed))
        surv_metrics_dict[test_dataset_element_name]["Metric"].append("C-index")
        surv_metrics_dict[test_dataset_element_name]["Result"].append(cindex)

        # ensemble용 raw 저장도 같이 (가능하면 저장)
        # epoch cindex를 썼더라도 ensemble은 risk_mean으로 계산해야 하므로 raw를 저장해둔다.
        if hasattr(trainer_model, "risk_list") and len(trainer_model.risk_list) > 0:
            risk = torch.cat([_to_1d_cpu_tensor(x) for x in trainer_model.risk_list], dim=0)
            time = torch.cat([_to_1d_cpu_tensor(x) for x in trainer_model.time_list], dim=0)
            event = torch.cat([_to_1d_cpu_tensor(x) for x in trainer_model.event_list], dim=0)

            names = getattr(trainer_model, "names", None)
            if names is None or len(names) == 0:
                names = [f"sample_{i}" for i in range(len(risk))]

            all_seed_risk_dict[test_dataset_element_name].append(risk.clone())
            all_seed_time_dict[test_dataset_element_name].append(time.clone())
            all_seed_event_dict[test_dataset_element_name].append(event.clone())
            all_seed_names_dict[test_dataset_element_name].append(list(names))

            # seed별 prediction csv 저장
            if getattr(args, "base_save_dir", None) is not None:
                save_dir = Path(args.base_save_dir) / test_dataset_element_name / f"seed_{seed}"
                save_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(
                    {
                        "name": list(names),
                        "time": time.numpy(),
                        "event": event.numpy().astype(int),
                        "risk": risk.numpy(),
                    }
                ).to_csv(save_dir / "survival_predictions.csv", index=False)

    except Exception as e:
        print(f"[ERROR] Failed to make survival metrics for seed={seed}, dataset={test_dataset_element_name}: {e}")
        traceback.print_exc()


# ----------------------------
# 전체(ensemble) 결과 저장
# ----------------------------
def make_whole_result_survival_metrics(
    args,
    test_dataset_element_name,
    save_dir,
):
    try:
        # save dir 결정
        if getattr(args, "base_save_dir", None) is not None:
            base_dir = Path(args.base_save_dir)
            save_dir = base_dir / test_dataset_element_name
        else:
            save_dir = Path(save_dir) / test_dataset_element_name
        save_dir.mkdir(parents=True, exist_ok=True)

        # seed별 요약 테이블
        metrics_df = pd.DataFrame(surv_metrics_dict[test_dataset_element_name])

        # Average 행 추가
        for metric in surv_metric_list:
            numeric_vals = pd.to_numeric(
                metrics_df[metrics_df["Metric"] == metric]["Result"],
                errors="coerce"
            ).dropna()
            if len(numeric_vals) == 0:
                continue
            mean_val = numeric_vals.mean()
            std_val = numeric_vals.std()
            surv_metrics_dict[test_dataset_element_name]["Seed"].append("Average")
            surv_metrics_dict[test_dataset_element_name]["Metric"].append(metric)
            surv_metrics_dict[test_dataset_element_name]["Result"].append(f"{mean_val:.3f} ± {std_val:.3f}")

        # seed raw 모으기
        risks_list = all_seed_risk_dict[test_dataset_element_name]
        times_list = all_seed_time_dict[test_dataset_element_name]
        events_list = all_seed_event_dict[test_dataset_element_name]
        names_list = all_seed_names_dict[test_dataset_element_name]

        if len(risks_list) == 0:
            print(f"[ERROR] No seed risks stored for dataset={test_dataset_element_name}")
            # 그래도 metrics csv는 저장
            final_metrics_df = pd.DataFrame(surv_metrics_dict[test_dataset_element_name])
            final_metrics_df.to_csv(save_dir / "final_survival_results.csv", index=False)
            return

        # time/event/names는 첫 seed 기준
        time = times_list[0]
        event = events_list[0]
        names = names_list[0]

        # risk stack: (S, N)
        risk_stack = torch.stack(risks_list, dim=0)  # (S,N)
        risk_mean = risk_stack.mean(dim=0)
        risk_std = risk_stack.std(dim=0)

        # ensemble c-index (torch 내부 구현)
        cindex_raw = _compute_cindex_torch(
            risk=risk_mean, time=time, event=event,
            signature=str(getattr(args, "survival_cindex_signature", "risk_time_event")).lower()
        )
        cindex_ens = cindex_raw * 100.0
        cindex_ens = round(float(cindex_ens), 3) if np.isfinite(float(cindex_ens)) else float("nan")

        surv_metrics_dict[test_dataset_element_name]["Seed"].append("Ensemble")
        surv_metrics_dict[test_dataset_element_name]["Metric"].append("C-index")
        surv_metrics_dict[test_dataset_element_name]["Result"].append(cindex_ens)

        # 최종 metrics csv 저장
        final_metrics_df = pd.DataFrame(surv_metrics_dict[test_dataset_element_name])
        final_metrics_df.to_csv(save_dir / "final_survival_results.csv", index=False)

        # ensemble prediction csv 저장
        pred_df = pd.DataFrame(
            {
                "name": list(names),
                "time": time.numpy(),
                "event": event.numpy().astype(int),
                "risk_mean": risk_mean.numpy(),
                "risk_std": risk_std.numpy(),
            }
        )
        pred_df.to_csv(save_dir / "ensemble_survival_predictions.csv", index=False)

        print("\nFinal Ensemble Results (Survival)")
        print(f"C-index (Ensemble): {cindex_ens}")
        print("All survival metrics have been saved successfully!!!")

    except Exception as e:
        print(f"[FATAL ERROR] make_whole_result_survival_metrics failed: {e}")
        traceback.print_exc()
