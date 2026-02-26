# pl_model/mil_trainer_survival.py

from __future__ import annotations

from typing import Optional, Dict, Any, List, Literal

import random
import numpy as np
import torch
import pytorch_lightning as pl

from pl_model.optimizers import Lookahead, Lion
from pl_model.forward_fn_survival import (
    concordance_index,
    logits_to_expected_time_and_risk,
    hazard_bce_loss,
)


class MILSurvivalTrainerModule(pl.LightningModule):
    """
    Discrete-time survival (hazard bins) trainer ONLY
    - loss: masked BCE over hazard logits (censor 엄밀 옵션 포함)
    - c-index: epoch-level risk score 기반
    """

    def __init__(
        self,
        args,
        seed,
        test_dataset_element_name,
        resolution_str,
        classifier,  # MIL backbone (bag -> hazard logits K)
        survival_endpoint: str = "OS",
        patch_path=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["classifier"])

        self.args = args
        self.seed = seed
        self.resolution_str = resolution_str
        self.test_dataset_element_name = test_dataset_element_name
        self.patch_path = patch_path

        self.survival_endpoint = str(survival_endpoint).upper()
        if self.survival_endpoint not in {"OS", "PFI"}:
            raise ValueError(f"survival_endpoint must be OS or PFI, got: {survival_endpoint}")

        self.base_save_dir = getattr(self.args, "base_save_dir", None)
        self.classifier = classifier

        # ✅ callback/metrics와 risk 방향 메타를 통일하기 위해 명시
        #    (risk 값이 클수록 위험도가 높다 = 생존이 짧다)
        self.risk_higher_is_riskier: bool = True

        # bins
        self.num_bins = int(getattr(self.args, "survival_num_bins", 20))
        if self.num_bins < 2:
            raise ValueError(f"survival_num_bins must be >= 2, got {self.num_bins}")

        self.cutpoints: Optional[torch.Tensor] = None
        self.bin_centers: Optional[torch.Tensor] = None

        # ---- options ----
        self.censor_include_current_bin: bool = bool(
            getattr(self.args, "survival_censor_include_current_bin", False)
        )

        self.risk_type: Literal["neg_expected_time", "cumhaz", "cumevent"] = str(
            getattr(self.args, "survival_risk_type", "cumhaz")
        ).lower()  # type: ignore

        self.cindex_signature: Literal["risk_time_event", "time_risk_event"] = str(
            getattr(self.args, "survival_cindex_signature", "risk_time_event")
        ).lower()  # type: ignore

        # patch drop
        self.mil_patch_drop_min = float(getattr(self.args, "mil_patch_drop_min", 0.0))
        self.mil_patch_drop_max = float(getattr(self.args, "mil_patch_drop_max", 0.0))
        self._use_patch_drop = self.mil_patch_drop_max > 0.0

        # epoch buffers
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

        # save_metrics_survival compat (seed별 metric 계산용)
        self.test_outputs: List[Dict[str, Any]] = []
        self.risk_list: List[torch.Tensor] = []
        self.event_list: List[torch.Tensor] = []
        self.time_list: List[torch.Tensor] = []
        self.names: List[str] = []

    # -----------------------------
    # patch subsample
    # -----------------------------
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

    # -----------------------------
    # build cutpoints fast path
    # -----------------------------
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
                # batch: name, coords, feats, event, time
                time = batch[-1]
                t = time.detach().cpu().view(-1).tolist()
                times.extend([float(x) for x in t if np.isfinite(x)])
            times_np = np.asarray(times, dtype=np.float64)
        else:
            times_np = np.asarray(times_fast, dtype=np.float64)

        times_np = times_np[np.isfinite(times_np)]
        if times_np.size < 10:
            raise RuntimeError(f"[Survival][Bins] Too few train samples for cutpoints: n={times_np.size}")

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
        c = cutpoints.detach().cpu().numpy().astype(np.float64)

        # edges: [0, c1, c2, ..., c_{K-1}, end]
        e0 = 0.0
        if c.size >= 2:
            last_w = float(c[-1] - c[-2])
            last_w = max(last_w, 1e-6)
            eK = float(c[-1] + last_w)
        elif c.size == 1:
            eK = float(c[-1] + max(float(c[-1] - e0), 1.0))
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

            print(
                f"[Survival][Bins] K={self.num_bins}, cutpoints[:5]={self.cutpoints.detach().cpu().numpy()[:5]}"
            )
            print(
                f"[Survival][Bins] centers[:5]={self.bin_centers.detach().cpu().numpy()[:5]}"
            )
            print(
                f"[Survival] censor_include_current_bin={self.censor_include_current_bin}, risk_type={self.risk_type}"
            )
            print(f"[Survival] cindex_signature={self.cindex_signature}")

    # -----------------------------
    # forward
    # -----------------------------
    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        out = self.classifier(feats)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if not torch.is_tensor(out):
            raise TypeError(f"classifier output must be Tensor, got {type(out)}")
        return out

    # -----------------------------
    # logits -> expected_time/risk
    # -----------------------------
    def _logits_to_expected_time_and_risk(self, logits: torch.Tensor):
        if self.bin_centers is None:
            raise RuntimeError("bin_centers not built.")
        return logits_to_expected_time_and_risk(logits, self.bin_centers, risk_type=self.risk_type)

    # -----------------------------
    # epoch buffers
    # -----------------------------
    def _append_epoch_buffers(self, stage: str, risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor):
        r = risk.detach().cpu().view(-1)
        t = time.detach().cpu().view(-1)
        e = event.detach().cpu().view(-1)

        if stage == "train":
            self._train_risk_buf.append(r)
            self._train_time_buf.append(t)
            self._train_event_buf.append(e)
        elif stage == "val":
            self._val_risk_buf.append(r)
            self._val_time_buf.append(t)
            self._val_event_buf.append(e)
        elif stage == "test":
            self._test_risk_buf.append(r)
            self._test_time_buf.append(t)
            self._test_event_buf.append(e)

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
        if not torch.isfinite(cidx):
            return None
        return float(cidx.detach().cpu())

    def _clear_epoch_buffers(self, stage: str):
        if stage == "train":
            self._train_risk_buf.clear()
            self._train_time_buf.clear()
            self._train_event_buf.clear()
        elif stage == "val":
            self._val_risk_buf.clear()
            self._val_time_buf.clear()
            self._val_event_buf.clear()
        else:
            self._test_risk_buf.clear()
            self._test_time_buf.clear()
            self._test_event_buf.clear()

    # -----------------------------
    # training
    # -----------------------------
    def training_step(self, batch, batch_idx):
        name, coords, feats, event, time = batch
        B = feats.size(0)

        logits_list = []
        for i in range(B):
            bag = feats[i]
            if self._use_patch_drop:
                bag = self._random_patch_subsample(bag)
            logits = self.forward(bag).view(-1)
            logits_list.append(logits)
        logits_mat = torch.stack(logits_list, dim=0)  # (B,K)

        if self.cutpoints is None:
            raise RuntimeError("cutpoints not built.")
        loss = hazard_bce_loss(
            logits=logits_mat,
            time=time.view(-1),
            event=event.view(-1),
            cutpoints=self.cutpoints,
            censor_include_current_bin=self.censor_include_current_bin,
        )

        _, risk, _ = self._logits_to_expected_time_and_risk(logits_mat)
        self._append_epoch_buffers("train", risk, time, event)

        self.log("Loss/train", loss, on_step=False, on_epoch=True, logger=True, prog_bar=False, batch_size=B)
        self.log("train_loss", loss, on_step=True, on_epoch=False, logger=False, prog_bar=True, batch_size=B)
        return loss

    def on_train_epoch_end(self):
        cidx = self._compute_epoch_cindex("train")
        if cidx is not None:
            self.log("CIndex/train", cidx, sync_dist=False)
            self.log("train_cindex", cidx, prog_bar=True, sync_dist=False)
        self._clear_epoch_buffers("train")

    # -----------------------------
    # validation
    # -----------------------------
    def validation_step(self, batch, batch_idx):
        name, coords, feats, event, time = batch
        B = feats.size(0)

        logits_list = []
        for i in range(B):
            bag = feats[i]
            logits = self.forward(bag).view(-1)
            logits_list.append(logits)
        logits_mat = torch.stack(logits_list, dim=0)  # (B,K)

        if self.cutpoints is None:
            raise RuntimeError("cutpoints not built.")
        loss = hazard_bce_loss(
            logits=logits_mat,
            time=time.view(-1),
            event=event.view(-1),
            cutpoints=self.cutpoints,
            censor_include_current_bin=self.censor_include_current_bin,
        )

        _, risk, _ = self._logits_to_expected_time_and_risk(logits_mat)
        self._append_epoch_buffers("val", risk, time, event)

        self.log("Loss/val", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=False, batch_size=B)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False, batch_size=B)

    def on_validation_epoch_end(self):
        cidx = self._compute_epoch_cindex("val")
        self.val_cindex_epoch = cidx
        if cidx is not None:
            print(f"[VAL][seed={self.seed}] C-index: {cidx:.4f}")
            self.log("CIndex/val", cidx, sync_dist=False)
            self.log("val_cindex", cidx, prog_bar=True, sync_dist=False)
        self._clear_epoch_buffers("val")

    # -----------------------------
    # test
    # -----------------------------
    def on_test_epoch_start(self):
        self.test_outputs = []
        self.risk_list = []
        self.event_list = []
        self.time_list = []
        self.names = []
        self.test_cindex_epoch = None
        self._clear_epoch_buffers("test")

    def test_step(self, batch, batch_idx):
        name, coords, feats, event, time = batch
        B = feats.size(0)

        logits_list = []
        for i in range(B):
            bag = feats[i]
            logits = self.forward(bag).view(-1)
            logits_list.append(logits)
        logits_mat = torch.stack(logits_list, dim=0)  # (B,K)

        if self.cutpoints is None:
            raise RuntimeError("cutpoints not built.")
        loss = hazard_bce_loss(
            logits=logits_mat,
            time=time.view(-1),
            event=event.view(-1),
            cutpoints=self.cutpoints,
            censor_include_current_bin=self.censor_include_current_bin,
        )

        expected_time, risk, cum_event_prob = self._logits_to_expected_time_and_risk(logits_mat)
        self._append_epoch_buffers("test", risk, time, event)

        self.log("Loss/final_test", loss, on_epoch=True, sync_dist=False, batch_size=B)

        if isinstance(name, (list, tuple)):
            name_list = list(name)
        else:
            name_list = [name] * B

        # callback 저장용 dict
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
                }
            )

        # save_metrics_survival용 버퍼
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
        self._clear_epoch_buffers("test")

    # -----------------------------
    # optimizer
    # -----------------------------
    def configure_optimizers(self):
        params = [{"params": filter(lambda p: p.requires_grad, self.classifier.parameters())}]
        opt_name = (self.args.opt or "adam").lower()

        if opt_name == "adam":
            optimizer = torch.optim.Adam(
                params, lr=self.args.lr, betas=(0.5, 0.9), weight_decay=self.args.weight_decay
            )
        elif opt_name == "adamw":
            optimizer = torch.optim.AdamW(params, lr=self.args.lr, weight_decay=self.args.weight_decay)
        elif opt_name == "lion":
            optimizer = Lion(params, lr=self.args.lr, betas=(0.9, 0.99), weight_decay=self.args.weight_decay)
        elif opt_name == "lookahead_radam":
            base = torch.optim.RAdam(params, lr=self.args.lr, weight_decay=self.args.weight_decay)
            optimizer = Lookahead(base, k=5, alpha=0.5)
        elif opt_name == "radam":
            optimizer = torch.optim.RAdam(params, lr=self.args.lr, weight_decay=self.args.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs, eta_min=5e-6)
        return [optimizer], [scheduler]
