# pl_model/forward_fn_survival.py
# ✅ Discrete-time survival utilities (hazard bins) + MICCAI-friendly options

from __future__ import annotations

from typing import Optional, Tuple, Dict, Any, Literal

import torch
import torch.nn.functional as F


# ============================================================
# Concordance index (Harrell's C-index)
# ============================================================
@torch.no_grad()
def concordance_index(
    risk: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
    *,
    signature: Literal["risk_time_event", "time_risk_event"] = "risk_time_event",
) -> torch.Tensor:
    """
    Harrell's C-index (pairwise) for a batch.

    Assumption:
      - risk가 클수록 더 빨리 event가 발생(짧은 time)해야 "좋은" 예측.

    signature:
      - "risk_time_event": (risk, time, event)  [default]
      - "time_risk_event": (time, risk, event)  (외부 구현체/legacy 호환용)
    """
    # ---- signature handling ----
    sig = str(signature).lower()
    if sig == "time_risk_event":
        time, risk = risk, time  # swap
    elif sig != "risk_time_event":
        raise ValueError(f"Unknown signature: {signature}")

    # ---- shape normalize ----
    if risk.ndim == 2 and risk.size(1) == 1:
        risk = risk.view(-1)
    risk = risk.view(-1)
    time = time.view(-1)
    event = event.view(-1)

    n = time.numel()
    if n < 2:
        return torch.tensor(float("nan"), device=time.device)

    # comparable pairs: i is uncensored event and time_i < time_j
    ti = time.unsqueeze(1)   # (n,1)
    tj = time.unsqueeze(0)   # (1,n)
    ei = event.unsqueeze(1)  # (n,1)

    comparable = (ei == 1) & (ti < tj)
    denom = comparable.sum()
    if denom == 0:
        return torch.tensor(float("nan"), device=time.device)

    ri = risk.unsqueeze(1)
    rj = risk.unsqueeze(0)

    concordant = (ri > rj) & comparable
    tied = (ri == rj) & comparable

    c = (concordant.sum().float() + 0.5 * tied.sum().float()) / denom.float()
    return c


# ============================================================
# Discrete-time hazard loss (masked BCE) with censor handling
# ============================================================
def hazard_bce_loss(
    logits: torch.Tensor,          # (B,K) or (K,)
    time: torch.Tensor,            # (B,)
    event: torch.Tensor,           # (B,)  (1=event, 0=censored)
    cutpoints: torch.Tensor,       # (K-1,)
    *,
    censor_include_current_bin: bool = False,
) -> torch.Tensor:
    """
    Discrete-time hazard loss (masked BCE).

    Let y = bucketize(time, cutpoints) in [0, K-1].

    For uncensored (event=1):
      - bins < y : target=0, mask=1
      - bin == y : target=1, mask=1
      - bins > y : mask=0

    For censored (event=0):
      - If censor_include_current_bin=True (강한 가정):
          bins <= y : target=0, mask=1  (y bin도 '끝까지 생존'으로 학습)
      - If censor_include_current_bin=False (MICCAI 엄밀):
          bins <  y : target=0, mask=1
          bins >= y : mask=0  (y bin 자체는 모른다고 보고 제외)
    """
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)

    B, K = logits.shape
    time = time.view(-1)
    event = event.view(-1).to(logits.dtype)

    # y in [0, K-1]
    y = torch.bucketize(time, cutpoints, right=False).clamp(0, K - 1)

    targets = torch.zeros((B, K), device=logits.device, dtype=logits.dtype)
    mask = torch.zeros((B, K), device=logits.device, dtype=logits.dtype)

    # vectorized mask 만들기
    ar = torch.arange(K, device=logits.device).view(1, K)  # (1,K)
    yv = y.view(B, 1)                                      # (B,1)

    is_event = (event.view(B, 1) == 1.0)

    # uncensored: mask <= y
    mask_event = (ar <= yv).to(logits.dtype)

    # censored:
    if censor_include_current_bin:
        # mask <= y (y bin 포함)
        mask_cens = (ar <= yv).to(logits.dtype)
    else:
        # mask < y (y bin 제외)
        mask_cens = (ar < yv).to(logits.dtype)

    mask = torch.where(is_event, mask_event, mask_cens)

    # target: only for uncensored at bin y -> 1
    # (censored는 모두 0)
    idx = torch.arange(B, device=logits.device)
    y_int = y.to(torch.long)
    targets[idx, y_int] = event  # event=1이면 1, event=0이면 0 (어차피 mask에서 제외됨)

    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    denom = mask.sum().clamp_min(1.0)
    return (bce * mask).sum() / denom


# ============================================================
# logits -> hazards -> expected_time / risk / cum_event_prob
# ============================================================
def logits_to_expected_time_and_risk(
    logits: torch.Tensor,          # (B,K) or (K,)
    bin_centers: torch.Tensor,     # (K,)
    *,
    risk_type: Literal["neg_expected_time", "cumhaz", "cumevent"] = "cumhaz",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    logits -> hazards -> pmf -> expected_time
    또한 risk를 여러 방식으로 정의 가능.

    Returns:
      expected_time: (B,)
      risk: (B,)
      cum_event_prob: (B,) = 1 - S_end

    risk_type:
      - "neg_expected_time": risk = -E[T]
      - "cumhaz":           risk = sum_{k} -log(1-h_k)  (누적 hazard, 큼=위험 큼)
      - "cumevent":         risk = 1 - S_end            (누적 사건 확률, 큼=위험 큼)
    """
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)

    hazards = torch.sigmoid(logits)  # (B,K)
    one_minus = (1.0 - hazards).clamp(min=1e-6, max=1.0)

    # survival prev (S_{k-1})
    surv_prev = torch.cumprod(one_minus, dim=1)
    surv_prev = torch.cat(
        [
            torch.ones((hazards.size(0), 1), device=hazards.device, dtype=hazards.dtype),
            surv_prev[:, :-1],
        ],
        dim=1,
    )

    pmf = hazards * surv_prev  # (B,K)
    centers = bin_centers.view(1, -1).to(dtype=logits.dtype, device=logits.device)

    expected_time = (pmf * centers).sum(dim=1)

    s_end = torch.prod(one_minus, dim=1)
    cum_event_prob = 1.0 - s_end

    rt = str(risk_type).lower()
    if rt == "neg_expected_time":
        risk = -expected_time
    elif rt == "cumhaz":
        # 누적 hazard = sum -log(1-h)
        risk = (-torch.log(one_minus)).sum(dim=1)
    elif rt == "cumevent":
        risk = cum_event_prob
    else:
        raise ValueError(f"Unknown risk_type: {risk_type}")

    return expected_time, risk, cum_event_prob


# ============================================================
# (선택) Forward wrapper (Discrete-time ONLY)
# ============================================================
def get_forward_func_survival(mil_model: str):
    """
    survival은 discrete-time hazard logits 예측을 표준으로 둔다.
    - non-DTFD: survival_forward_discrete 반환
    - DTFD-MIL: trainer에서 2-tier forward를 자체 처리할 수 있으니 None
    """
    mil_model = str(mil_model)
    if mil_model == "DTFD-MIL":
        return None
    return survival_forward_discrete


def _extract_logits_from_model_output(out) -> torch.Tensor:
    if isinstance(out, (tuple, list)):
        out = out[0]
    if not torch.is_tensor(out):
        raise TypeError(f"Model output must be Tensor or (Tensor,...). Got: {type(out)}")
    return out


def survival_forward_discrete(
    data: torch.Tensor,
    classifier,
    loss_func=None,  # unused (compat)
    num_classes: int = 1,  # unused (compat)
    event: Optional[torch.Tensor] = None,
    time: Optional[torch.Tensor] = None,
    return_cindex: bool = False,
    *,
    cutpoints: Optional[torch.Tensor] = None,     # (K-1,)
    bin_centers: Optional[torch.Tensor] = None,   # (K,)
    censor_include_current_bin: bool = False,
    risk_type: Literal["neg_expected_time", "cumhaz", "cumevent"] = "cumhaz",
    cindex_signature: Literal["risk_time_event", "time_risk_event"] = "risk_time_event",
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
    """
    Discrete-time survival forward.

    Returns:
      logits (원본 형태 유지),
      loss (가능하면),
      extra dict
    """
    out = classifier(data)
    logits = _extract_logits_from_model_output(out)

    # normalize to (B,K)
    if logits.dim() == 1:
        logits_mat = logits.unsqueeze(0)
    elif logits.dim() == 2:
        logits_mat = logits
    else:
        logits_mat = logits.view(logits.size(0), -1)

    loss = None
    extra: Dict[str, Any] = {}

    if bin_centers is not None:
        expected_time, risk, cum_event_prob = logits_to_expected_time_and_risk(
            logits_mat, bin_centers, risk_type=risk_type
        )
        extra["expected_time"] = expected_time
        extra["risk"] = risk
        extra["cum_event_prob"] = cum_event_prob

        if return_cindex and (event is not None) and (time is not None):
            extra["cindex"] = concordance_index(
                risk, time.view(-1), event.view(-1), signature=cindex_signature
            )

    if (event is not None) and (time is not None) and (cutpoints is not None):
        loss = hazard_bce_loss(
            logits=logits_mat,
            time=time.view(-1),
            event=event.view(-1),
            cutpoints=cutpoints,
            censor_include_current_bin=censor_include_current_bin,
        )

    return logits, loss, extra
