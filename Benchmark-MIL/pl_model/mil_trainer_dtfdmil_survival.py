# pl_model/mil_trainer_dtfdmil_survival.py

from __future__ import annotations

from typing import List, Dict, Any, Optional, Literal

import random
import numpy as np
import torch
import pytorch_lightning as pl

from pl_model.optimizers import Lookahead, Lion
from pl_model.forward_fn import dtfdmil_forward_1st_tier
from pl_model.forward_fn_survival import (
    concordance_index,
    logits_to_expected_time_and_risk,
    hazard_bce_loss,
)


class DTFDSurvivalTrainerModule(pl.LightningModule):
    """
    DTFD-MIL + Discrete-time (bin/hazard) survival trainer.

    ✅ CoxPH 완전 제거 버전
    - hazard bins로 학습: masked BCE (censor 엄밀 옵션 포함)
    - C-index는 epoch-level로 계산 (안정형 risk: 기본 cumhaz)
    - save_metrics_survival.py 호환:
        * risk_list/time_list/event_list/names
        * test_cindex_epoch
        * (추가) risks/events/times 집계 텐서
    """

    def __init__(
        self,
        args,
        seed,
        test_dataset_element_name,
        resolution_str,
        classifier_list,          # [classifier, attention, dimReduction, UClassifier]
        loss_func_list=None,      # optional: [aux_loss0]
        survival_endpoint: str = "OS",
        patch_path=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["classifier_list", "loss_func_list"])

        self.args = args
        self.seed = int(seed)
        self.test_dataset_element_name = test_dataset_element_name
        self.patch_path = patch_path
        self.base_save_dir = getattr(self.args, "base_save_dir", None)
        self.resolution_str = resolution_str

        self.survival_endpoint = str(survival_endpoint).upper()
        if self.survival_endpoint not in {"OS", "PFI"}:
            raise ValueError(f"survival_endpoint must be OS or PFI, got: {survival_endpoint}")

        # ✅ manual opt 유지
        self.automatic_optimization = False

        # modules
        self.classifier = classifier_list[0]
        self.attention = classifier_list[1]
        self.dimReduction = classifier_list[2]
        self.UClassifier = classifier_list[3]

        self.loss_func0 = None
        if loss_func_list is not None and len(loss_func_list) >= 1:
            self.loss_func0 = loss_func_list[0]

        self.accumulate_grad_batches = int(getattr(args, "accumulate_grad_batches", 1))

        # patch drop
        self.mil_patch_drop_min = float(getattr(self.args, "mil_patch_drop_min", 0.0))
        self.mil_patch_drop_max = float(getattr(self.args, "mil_patch_drop_max", 0.0))
        self._use_patch_drop = self.mil_patch_drop_max > 0.0

        self.attention_func = bool(getattr(self.args, "attention", False))

        # bins
        self.num_bins = int(getattr(self.args, "survival_num_bins", 20))
        if self.num_bins < 2:
            raise ValueError(f"survival_num_bins must be >= 2, got {self.num_bins}")

        self.cutpoints: Optional[torch.Tensor] = None
        self.bin_centers: Optional[torch.Tensor] = None

        # policies
        self.censor_include_current_bin: bool = bool(
            getattr(self.args, "survival_censor_include_current_bin", False)
        )
        self.risk_type: Literal["neg_expected_time", "cumhaz", "cumevent"] = str(
            getattr(self.args, "survival_risk_type", "cumhaz")
        ).lower()  # type: ignore

        self.cindex_signature: Literal["risk_time_event", "time_risk_event"] = str(
            getattr(self.args, "survival_cindex_signature", "risk_time_event")
        ).lower()  # type: ignore

        # epoch buffers (for stable epoch-level c-index)
        self._train_risk_buf: List[torch.Tensor] = []
        self._train_time_buf: List[torch.Tensor] = []
        self._train_event_buf: List[torch.Tensor] = []

        self._val_risk_buf: List[torch.Tensor] = []
        self._val_time_buf: List[torch.Tensor] = []
        self._val_event_buf: List[torch.Tensor] = []

        self._test_risk_buf: List[torch.Tensor] = []
        self._test_time_buf: List[torch.Tensor] = []
        self._test_event_buf: List[torch.Tensor] = []

        self.val_cindex_epoch: Optional[float] = None
        self.test_cindex_epoch: Optional[float] = None

        # callback 호환용 (SaveSurvivalAnalysisResultsCallback)
        self.test_outputs: List[Dict[str, Any]] = []

        # save_metrics_survival.py 호환용
        self.risk_list: List[torch.Tensor] = []
        self.event_list: List[torch.Tensor] = []
        self.time_list: List[torch.Tensor] = []
        self.names: List[str] = []

        # (선택) 집계 텐서 (디버깅/호환)
        self.risks: Optional[torch.Tensor] = None
        self.events: Optional[torch.Tensor] = None
        self.times: Optional[torch.Tensor] = None

    # -----------------------------------------
    # Patch drop
    # -----------------------------------------
    def _random_patch_subsample(self, feats: torch.Tensor) -> torch.Tensor:
        if feats.dim() != 2 or feats.size(0) <= 1:
            return feats

        drop_min = max(0.0, min(1.0, self.mil_patch_drop_min))
        drop_max = max(0.0, min(1.0, self.mil_patch_drop_max))
        if drop_max <= 0.0:
            return feats
        if drop_min > drop_max:
            drop_min, drop_max = drop_max, drop_min

        drop_ratio = random.uniform(drop_min, drop_max)
        keep_ratio = 1.0 - drop_ratio
        num_keep = max(1, int(round(keep_ratio * feats.size(0))))
        if num_keep >= feats.size(0):
            return feats

        idx = torch.randperm(feats.size(0), device=feats.device)[:num_keep]
        return feats[idx]

    # -----------------------------------------
    # Build cutpoints fast path
    # -----------------------------------------
    def _try_get_train_times_fast(self) -> Optional[np.ndarray]:
        if self.trainer is None:
            return None
        dm = getattr(self.trainer, "datamodule", None)
        if dm is None:
            return None

        for attr in ["train_dataset", "dataset_train", "train_ds", "ds_train"]:
            ds = getattr(dm, attr, None)
            if ds is None:
                continue

            if hasattr(ds, "times"):
                t = getattr(ds, "times")
                if torch.is_tensor(t):
                    t = t.detach().cpu().numpy()
                else:
                    t = np.asarray(t)
                return t.astype(np.float64)

            if hasattr(ds, "df"):
                df = getattr(ds, "df")
                for col in ["time", "OS_time", "PFI_time", "days", "duration"]:
                    if hasattr(df, "__getitem__") and col in df.columns:
                        return df[col].to_numpy(dtype=np.float64)

        return None

    @torch.no_grad()
    def _build_cutpoints_from_train(self) -> torch.Tensor:
        times_fast = self._try_get_train_times_fast()
        if times_fast is None:
            if self.trainer is None:
                raise RuntimeError("Trainer not attached.")
            train_loader = self.trainer.datamodule.train_dataloader()

            times: List[float] = []
            for batch in train_loader:
                time = batch[-1]
                if torch.is_tensor(time):
                    t = time.detach().cpu().view(-1).tolist()
                    times.extend([float(x) for x in t if np.isfinite(x)])
                else:
                    times.append(float(time))
            times_np = np.asarray(times, dtype=np.float64)
        else:
            times_np = np.asarray(times_fast, dtype=np.float64)

        times_np = times_np[np.isfinite(times_np)]
        if times_np.size < 10:
            raise RuntimeError(f"[DTFD-Survival][Bins] Too few train samples for cutpoints: n={times_np.size}")

        qs = np.linspace(0.0, 1.0, self.num_bins + 1, dtype=np.float64)[1:-1]
        cut = np.quantile(times_np, qs, method="linear")
        cut = np.unique(cut)

        if cut.size < self.num_bins - 1:
            tmin, tmax = float(times_np.min()), float(times_np.max())
            if tmax <= tmin:
                tmax = tmin + 1.0
            cut = np.linspace(tmin, tmax, self.num_bins + 1, dtype=np.float64)[1:-1]

        return torch.tensor(cut, dtype=torch.float32)

    def _build_bin_centers(self, cutpoints: torch.Tensor) -> torch.Tensor:
        K = self.num_bins
        c = cutpoints.detach().cpu().numpy().astype(np.float64)

        # enforce size K-1
        if c.size != K - 1:
            if c.size > K - 1:
                c = c[: K - 1]
            else:
                tmin = float(np.min(c)) if c.size > 0 else 0.0
                tmax = float(np.max(c)) + 1.0 if c.size > 0 else 1.0
                c = np.linspace(tmin, tmax, K + 1, dtype=np.float64)[1:-1]

        e0 = 0.0
        if c.size >= 1:
            if c.size >= 2:
                last_w = float(c[-1] - c[-2])
                last_w = max(last_w, 1e-6)
            else:
                last_w = max(float(c[-1] - e0), 1.0)
            eK = float(c[-1] + last_w)
        else:
            eK = 1.0

        edges = np.concatenate([[e0], c, [eK]], axis=0)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return torch.tensor(centers, dtype=torch.float32)

    def on_fit_start(self):
        if self.cutpoints is None:
            cut = self._build_cutpoints_from_train()
            self.cutpoints = cut.to(self.device)
            self.bin_centers = self._build_bin_centers(cut).to(self.device)

            print(f"[DTFD-Survival][Bins] K={self.num_bins}, cutpoints[:5]={self.cutpoints.detach().cpu().numpy()[:5]}")
            print(f"[DTFD-Survival][Bins] centers[:5]={self.bin_centers.detach().cpu().numpy()[:5]}")
            print(f"[DTFD-Survival] censor_include_current_bin={self.censor_include_current_bin}, risk_type={self.risk_type}")
            print(f"[DTFD-Survival] cindex_signature={self.cindex_signature}")

    def _logits_to_expected_time_and_risk(self, logits: torch.Tensor):
        if self.bin_centers is None:
            raise RuntimeError("bin_centers not built.")
        return logits_to_expected_time_and_risk(logits, self.bin_centers, risk_type=self.risk_type)

    # -----------------------------------------
    # epoch buffer helpers
    # -----------------------------------------
    def _append_epoch_buffers(self, stage: str, risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor):
        r = risk.detach().cpu().view(-1)
        t = time.detach().cpu().view(-1)
        e = event.detach().cpu().view(-1)

        if stage == "train":
            self._train_risk_buf.append(r); self._train_time_buf.append(t); self._train_event_buf.append(e)
        elif stage == "val":
            self._val_risk_buf.append(r); self._val_time_buf.append(t); self._val_event_buf.append(e)
        elif stage == "test":
            self._test_risk_buf.append(r); self._test_time_buf.append(t); self._test_event_buf.append(e)

    def _compute_epoch_cindex(self, stage: str) -> Optional[float]:
        if stage == "train":
            buf = (self._train_risk_buf, self._train_time_buf, self._train_event_buf)
        elif stage == "val":
            buf = (self._val_risk_buf, self._val_time_buf, self._val_event_buf)
        else:
            buf = (self._test_risk_buf, self._test_time_buf, self._test_event_buf)

        if len(buf[0]) == 0:
            return None

        risks = torch.cat(buf[0], dim=0)
        times = torch.cat(buf[1], dim=0)
        events = torch.cat(buf[2], dim=0)

        cidx = concordance_index(risks, times, events, signature=self.cindex_signature)
        if torch.is_tensor(cidx):
            if not torch.isfinite(cidx):
                return None
            return float(cidx.detach().cpu())
        try:
            cidx = float(cidx)
            if not np.isfinite(cidx):
                return None
            return cidx
        except Exception:
            return None

    def _clear_epoch_buffers(self, stage: str):
        if stage == "train":
            self._train_risk_buf.clear(); self._train_time_buf.clear(); self._train_event_buf.clear()
        elif stage == "val":
            self._val_risk_buf.clear(); self._val_time_buf.clear(); self._val_event_buf.clear()
        else:
            self._test_risk_buf.clear(); self._test_time_buf.clear(); self._test_event_buf.clear()

    # -----------------------------------------
    # Forward (single bag) -> pseudo_feat
    # -----------------------------------------
    def forward_single_bag(self, feats: torch.Tensor, train: bool = False, get_attention: bool = False):
        dummy0 = (lambda x, y: x.mean() * 0.0)

        if get_attention:
            loss0, pseudo_feat, tAA, patch_indices = dtfdmil_forward_1st_tier(
                self.args,
                feats,
                self.classifier,
                self.attention,
                self.dimReduction,
                loss_func0=self.loss_func0 if self.loss_func0 is not None else dummy0,
                label=None,
                get_attention=True,
            )
        else:
            loss0, pseudo_feat = dtfdmil_forward_1st_tier(
                self.args,
                feats,
                self.classifier,
                self.attention,
                self.dimReduction,
                loss_func0=self.loss_func0 if self.loss_func0 is not None else dummy0,
                label=None,
                get_attention=False,
            )

        if train and self.loss_func0 is not None:
            self.manual_backward(loss0, retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.dimReduction.parameters(), self.args.grad_clipping)
            torch.nn.utils.clip_grad_norm_(self.attention.parameters(), self.args.grad_clipping)
            torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), self.args.grad_clipping)

        if get_attention:
            return loss0, pseudo_feat, tAA, patch_indices
        return loss0, pseudo_feat

    def _pseudo_to_bag_logits(self, pseudo_feat: torch.Tensor) -> torch.Tensor:
        out = self.UClassifier(pseudo_feat)
        if isinstance(out, (tuple, list)):
            out = out[0]

        # 기대: (n_pseudo, K) 또는 (K,)
        if out.dim() == 1:
            logits = out
        elif out.dim() == 2:
            logits = out.mean(dim=0)
        else:
            raise RuntimeError(f"UClassifier output shape unexpected: {tuple(out.shape)}")
        return logits.view(-1)  # (K,)

    @staticmethod
    def _normalize_name_list(name, B: int) -> List[str]:
        """
        batch에서 들어오는 name이 str/tuple/list/tensor 등 무엇이든
        길이 B의 str 리스트로 정규화
        """
        if isinstance(name, (list, tuple)):
            lst = list(name)
            if len(lst) != B:
                # 흔치 않지만 안전 처리
                if len(lst) == 1:
                    lst = lst * B
                else:
                    lst = (lst + [lst[-1]])[:B]
            return [str(x[0] if isinstance(x, (list, tuple)) and len(x) > 0 else x) for x in lst]

        # torch tensor 등
        if torch.is_tensor(name):
            if name.numel() == 1:
                return [str(name.item())] * B
            flat = name.detach().cpu().view(-1).tolist()
            if len(flat) == 1:
                flat = flat * B
            return [str(x) for x in flat[:B]]

        # scalar
        return [str(name)] * B

    # -----------------------------------------
    # training_step (manual opt)
    # -----------------------------------------
    def training_step(self, batch, batch_idx):
        name, coords, feats, event, time = batch
        optimizer0, optimizer1 = self.optimizers()

        B = feats.size(0)
        bag_logits_list = []
        loss0_total = torch.tensor(0.0, device=self.device)

        for i in range(B):
            bag = feats[i]
            if self._use_patch_drop:
                bag = self._random_patch_subsample(bag)

            if self.loss_func0 is not None:
                loss0_i, pseudo_feat = self.forward_single_bag(bag, train=True, get_attention=False)
                loss0_total = loss0_total + loss0_i
            else:
                _, pseudo_feat = self.forward_single_bag(bag, train=False, get_attention=False)

            bag_logits_list.append(self._pseudo_to_bag_logits(pseudo_feat))

        logits_mat = torch.stack(bag_logits_list, dim=0)  # (B,K)

        if self.cutpoints is None:
            raise RuntimeError("cutpoints not built.")
        loss1 = hazard_bce_loss(
            logits=logits_mat,
            time=time.view(-1),
            event=event.view(-1),
            cutpoints=self.cutpoints,
            censor_include_current_bin=self.censor_include_current_bin,
        )

        self.manual_backward(loss1)
        torch.nn.utils.clip_grad_norm_(self.UClassifier.parameters(), self.args.grad_clipping)

        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            optimizer0.step()
            optimizer1.step()
            optimizer0.zero_grad()
            optimizer1.zero_grad()

        total_loss = loss1 + (loss0_total / max(1, B) if self.loss_func0 is not None else 0.0)

        _, risk, _ = self._logits_to_expected_time_and_risk(logits_mat)
        self._append_epoch_buffers("train", risk, time, event)

        self.log("train_loss", total_loss, on_step=True, on_epoch=False, sync_dist=False, batch_size=B, prog_bar=True, logger=False)
        self.log("Loss/train", total_loss, on_step=False, on_epoch=True, sync_dist=False, batch_size=B, prog_bar=False, logger=True)
        return total_loss

    def on_train_epoch_end(self):
        # scheduler step
        sch0, sch1 = self.lr_schedulers()
        sch0.step()
        sch1.step()

        cidx = self._compute_epoch_cindex("train")
        if cidx is not None:
            self.log("CIndex/train", cidx, sync_dist=False)
            self.log("train_cindex", cidx, prog_bar=True, sync_dist=False)
        self._clear_epoch_buffers("train")

    # -----------------------------------------
    # validation_step
    # -----------------------------------------
    def validation_step(self, batch, batch_idx):
        name, coords, feats, event, time = batch
        B = feats.size(0)

        bag_logits_list = []
        for i in range(B):
            bag = feats[i]
            _, pseudo_feat = self.forward_single_bag(bag, train=False, get_attention=False)
            bag_logits_list.append(self._pseudo_to_bag_logits(pseudo_feat))

        logits_mat = torch.stack(bag_logits_list, dim=0)

        if self.cutpoints is None:
            raise RuntimeError("cutpoints not built.")
        loss1 = hazard_bce_loss(
            logits=logits_mat,
            time=time.view(-1),
            event=event.view(-1),
            cutpoints=self.cutpoints,
            censor_include_current_bin=self.censor_include_current_bin,
        )

        _, risk, _ = self._logits_to_expected_time_and_risk(logits_mat)
        self._append_epoch_buffers("val", risk, time, event)

        self.log("val_loss", loss1, on_step=False, on_epoch=True, sync_dist=False, batch_size=B, prog_bar=True)
        self.log("Loss/val", loss1, on_step=False, on_epoch=True, sync_dist=False, batch_size=B, prog_bar=False)

    def on_validation_epoch_end(self):
        cidx = self._compute_epoch_cindex("val")
        self.val_cindex_epoch = cidx
        if cidx is not None:
            print(f"[VAL][seed={self.seed}] C-index: {cidx:.4f}")
            self.log("CIndex/val", cidx, sync_dist=False)
            self.log("val_cindex", cidx, prog_bar=True, sync_dist=False)
        self._clear_epoch_buffers("val")

    # -----------------------------------------
    # test
    # -----------------------------------------
    def on_test_epoch_start(self):
        # callback / metrics 버퍼 초기화
        self.test_outputs = []

        self.risk_list = []
        self.event_list = []
        self.time_list = []
        self.names = []

        self.risks = None
        self.events = None
        self.times = None

        self.test_cindex_epoch = None
        self._clear_epoch_buffers("test")

    def test_step(self, batch, batch_idx):
        name, coords, feats, event, time = batch
        B = feats.size(0)

        bag_logits_list = []
        attn_weights = None
        want_attn = bool(getattr(self.args, "attention", False))

        # name 정규화 (반드시 길이 B)
        name_list = self._normalize_name_list(name, B)

        for i in range(B):
            bag = feats[i]

            if want_attn and i == 0:
                _, pseudo_feat, tAA, patch_indices = self.forward_single_bag(bag, train=False, get_attention=True)
                # ⚠️ tAA shape은 모델마다 다름 -> callback에서 해석하도록 raw 저장
                attn_weights = tAA.detach().cpu() if tAA is not None else None
            else:
                _, pseudo_feat = self.forward_single_bag(bag, train=False, get_attention=False)

            bag_logits_list.append(self._pseudo_to_bag_logits(pseudo_feat))

        logits_mat = torch.stack(bag_logits_list, dim=0)

        if self.cutpoints is None:
            raise RuntimeError("cutpoints not built.")
        loss1 = hazard_bce_loss(
            logits=logits_mat,
            time=time.view(-1),
            event=event.view(-1),
            cutpoints=self.cutpoints,
            censor_include_current_bin=self.censor_include_current_bin,
        )

        expected_time, risk, cum_event_prob = self._logits_to_expected_time_and_risk(logits_mat)
        self._append_epoch_buffers("test", risk, time, event)

        self.log("Loss/final_test", loss1, on_epoch=True, sync_dist=False, batch_size=B)

        # callback 저장용 구조
        for i in range(B):
            self.test_outputs.append(
                {
                    "name": name_list[i],
                    "coords": None,
                    "risk": float(risk[i].detach().cpu()),
                    "expected_time": float(expected_time[i].detach().cpu()),
                    "cum_event_prob": float(cum_event_prob[i].detach().cpu()),
                    "event": int(event[i].detach().cpu()),
                    "time": float(time[i].detach().cpu()),
                    "attn": attn_weights if (want_attn and i == 0) else None,
                }
            )

        # save_metrics_survival 호환 버퍼
        self.risk_list.append(risk.detach().cpu().view(-1))
        self.event_list.append(event.detach().cpu().view(-1))
        self.time_list.append(time.detach().cpu().view(-1))
        self.names.extend(name_list)

    def on_test_epoch_end(self):
        cidx = self._compute_epoch_cindex("test")
        self.test_cindex_epoch = cidx
        if cidx is not None:
            self.log("CIndex/final_test", cidx, sync_dist=False)
            self.log("final_test_cindex", cidx, prog_bar=True, sync_dist=False)

        # ✅ 집계 텐서도 만들어서 (save_metrics / 디버깅) 어디서든 쓸 수 있게
        if len(self.risk_list) > 0:
            self.risks = torch.cat(self.risk_list, dim=0).detach().cpu()
        if len(self.event_list) > 0:
            self.events = torch.cat(self.event_list, dim=0).detach().cpu()
        if len(self.time_list) > 0:
            self.times = torch.cat(self.time_list, dim=0).detach().cpu()

        self._clear_epoch_buffers("test")

    # -----------------------------------------
    # Optimizer / Scheduler
    # -----------------------------------------
    def configure_optimizers(self):
        trainable_parameters = []
        trainable_parameters += list(self.classifier.parameters())
        trainable_parameters += list(self.attention.parameters())
        trainable_parameters += list(self.dimReduction.parameters())

        params_optimizer_0 = [{"params": filter(lambda p: p.requires_grad, trainable_parameters)}]
        params_optimizer_1 = [{"params": filter(lambda p: p.requires_grad, self.UClassifier.parameters())}]

        def create_optimizer(params, opt_name, lr, wd):
            opt_name = (opt_name or "adam").lower()
            if opt_name == "adam":
                return torch.optim.Adam(params, lr=lr, weight_decay=wd)
            elif opt_name == "adamw":
                return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
            elif opt_name == "radam":
                return torch.optim.RAdam(params, lr=lr, weight_decay=wd)
            elif opt_name == "lookahead_radam":
                base = torch.optim.RAdam(params, lr=lr, weight_decay=wd)
                return Lookahead(base, k=5, alpha=0.5)
            elif opt_name == "lion":
                return Lion(params, lr=lr, betas=(0.9, 0.99), weight_decay=wd)
            else:
                return torch.optim.Adam(params, lr=lr, weight_decay=wd)

        opt_name = getattr(self.args, "opt", "adam")
        lr = float(getattr(self.args, "lr", 1e-4))
        wd = float(getattr(self.args, "weight_decay", 0.0))

        optimizer0 = create_optimizer(params_optimizer_0, opt_name, lr, wd)
        optimizer1 = create_optimizer(params_optimizer_1, opt_name, lr, wd)

        scheduler0 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer0, T_max=int(self.args.epochs), eta_min=5e-6)
        scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=int(self.args.epochs), eta_min=5e-6)

        # manual opt라서 step은 on_train_epoch_end에서 직접 호출
        return [optimizer0, optimizer1], [scheduler0, scheduler1]
