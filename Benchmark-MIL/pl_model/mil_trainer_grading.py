# pl_model/mil_trainer_grading.py
import inspect
import random
from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassCalibrationError,
    MulticlassCohenKappa,
)

from pl_model.optimizers import Lookahead, Lion


class MILGradingTrainerModule(pl.LightningModule):
    """
    Grading 전용 Trainer (일반 MIL용)
    - validation monitor: QWK (Quadratic Weighted Kappa)
    - save_metrics 계열과 호환: y_prob_list/label_list/names/logits/labels 유지
    - ✅ DTFD grading과 동일한 철학: metric update 전에 항상 slide-level shape 정규화
    """

    def __init__(
        self,
        args,
        seed,
        test_dataset_element_name,
        test_class_names_list,
        num_classes,
        resolution_str,
        classifier,
        loss_func,
        metrics=None,               # main.py에서 넘겨줘도 grading에서는 안정성 위해 직접 생성
        forward_func: Callable = None,
        attention_func: Optional[Callable] = None,
        patch_path=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["classifier", "loss_func", "metrics", "attention_func", "forward_func"])

        self.args = args
        self.seed = seed
        self.num_classes = int(num_classes)
        self.resolution_str = resolution_str
        self.test_dataset_element_name = test_dataset_element_name
        self.test_class_names_list = test_class_names_list
        self.patch_path = patch_path

        self.base_save_dir = getattr(self.args, "base_save_dir", None)

        self.classifier = classifier
        self.loss_func = loss_func

        if forward_func is None:
            raise ValueError("forward_func must be provided (e.g., from pl_model.forward_fn.get_forward_func)")
        self.forward_func = forward_func

        self.attention_forward = attention_func
        self.attention_func = attention_func is not None

        self.automatic_optimization = True

        # Patch Drop (optional)
        self.mil_patch_drop_min = float(getattr(self.args, "mil_patch_drop_min", 0.0))
        self.mil_patch_drop_max = float(getattr(self.args, "mil_patch_drop_max", 0.0))
        self._use_patch_drop = self.mil_patch_drop_max > 0.0

        self.use_weighted_sampler = bool(getattr(self.args, "use_weighted_sampler", False))

        # -----------------------------
        # Metrics (Grading 전용)
        # -----------------------------
        self.train_acc = MulticlassAccuracy(num_classes=self.num_classes, average="micro")

        self.val_acc = MulticlassAccuracy(num_classes=self.num_classes, average="micro")
        self.val_bacc = MulticlassAccuracy(num_classes=self.num_classes, average="macro")
        self.val_qwk = MulticlassCohenKappa(num_classes=self.num_classes, weights="quadratic")

        self.test_acc = MulticlassAccuracy(num_classes=self.num_classes, average="micro")
        self.test_bacc = MulticlassAccuracy(num_classes=self.num_classes, average="macro")
        self.test_qwk = MulticlassCohenKappa(num_classes=self.num_classes, weights="quadratic")
        self.test_ece = MulticlassCalibrationError(num_classes=self.num_classes, n_bins=int(getattr(args, "n_bins", 15)))

        # Callback 분석용 버퍼
        self.test_outputs = []

        # save_metrics_grading.py 호환용 버퍼
        self.y_prob_list = []
        self.label_list = []
        self.names = []
        self.logits = None
        self.labels = None

        # ---- forward_fn signature 호환 처리용 캐시 ----
        self._forward_accepts_args = self._func_accepts_kw(self.forward_func, "args")
        self._attn_accepts_args = self._func_accepts_kw(self.attention_forward, "args") if self.attention_forward else False
        # =========================
        # DEBUG PRINTS (Grading)
        # =========================
        def _fname(f):
            if f is None:
                return "None"
            return getattr(f, "__qualname__", getattr(f, "__name__", str(f)))

        print(
            f"[DEBUG][GradingTrainer:init] seed={self.seed} "
            f"train_mode={getattr(self.args, 'train_mode', None)} "
            f"mil_model={getattr(self.args, 'mil_model', None)} "
            f"num_classes={self.num_classes}"
        )
        print(
            f"[DEBUG][GradingTrainer:init] forward_func={_fname(self.forward_func)} "
            f"accepts_args={self._forward_accepts_args}"
        )
        print(
            f"[DEBUG][GradingTrainer:init] attention_func={_fname(self.attention_forward)} "
            f"accepts_args={self._attn_accepts_args}"
        )
        print(
            f"[DEBUG][GradingTrainer:init] grading_loss={getattr(self.args, 'grading_loss', None)} "
            f"grading_alpha={getattr(self.args, 'grading_alpha', None)} "
            f"grading_power={getattr(self.args, 'grading_power', None)} "
            f"(src) cost_type={getattr(self.args, 'grading_cost_type', None)} "
            f"lambda={getattr(self.args, 'grading_cost_lambda', None)} "
            f"gamma={getattr(self.args, 'grading_cost_gamma', None)} "
            f"normalize={bool(getattr(self.args, 'grading_cost_normalize', False))}"
        )
    # --------------------------------
    # helpers
    # --------------------------------
    @staticmethod
    def _func_accepts_kw(func: Optional[Callable], kw: str) -> bool:
        if func is None:
            return False
        try:
            sig = inspect.signature(func)
            if kw in sig.parameters:
                return True
            # **kwargs를 받는지
            for p in sig.parameters.values():
                if p.kind == inspect.Parameter.VAR_KEYWORD:
                    return True
            return False
        except Exception:
            # signature introspection 실패 시 안전하게 False 처리하고 try/except로 커버
            return False

    @staticmethod
    def _to_slide_logits(y_logit: torch.Tensor) -> torch.Tensor:
        """
        metric / argmax / softmax 전에 logits shape 정규화

        - (C,)      -> (1,C)
        - (1,C)     -> (1,C)
        - (G,C)     -> (1,C) by mean over G  (혹시 모델이 group logits를 내는 경우 방어)
        - (1,G,C)   -> (1,C) by mean over G
        """
        if y_logit is None:
            raise ValueError("y_logit is None")

        if y_logit.ndim == 1:
            return y_logit.unsqueeze(0)

        if y_logit.ndim == 2:
            if y_logit.size(0) > 1:
                return y_logit.mean(dim=0, keepdim=True)
            return y_logit

        if y_logit.ndim == 3 and y_logit.size(0) == 1:
            # (1,G,C)
            return y_logit.mean(dim=1)

        raise ValueError(f"Unexpected y_logit shape: {tuple(y_logit.shape)}")

    @staticmethod
    def _to_slide_prob(y_prob: torch.Tensor) -> torch.Tensor:
        """
        - (C,) -> (1,C)
        - (G,C) -> (1,C) mean
        - (1,G,C) -> (1,C) mean over G
        """
        if y_prob is None:
            return None
        if y_prob.ndim == 1:
            return y_prob.unsqueeze(0)
        if y_prob.ndim == 2:
            if y_prob.size(0) > 1:
                return y_prob.mean(dim=0, keepdim=True)
            return y_prob
        if y_prob.ndim == 3 and y_prob.size(0) == 1:
            return y_prob.mean(dim=1)
        return y_prob

    @staticmethod
    def _get_label_idx_and_target(label: torch.Tensor):
        """
        label:
          - hard: (1,) or (B,)
          - soft: (1,C) or (B,C)
        returns:
          label_idx: (B,)  (metric용)
          target_label: loss용 label (hard/soft 그대로)
        """
        target_label = label
        if isinstance(label, torch.Tensor) and label.ndim > 1:
            label_idx = torch.argmax(label, dim=1).long()
        else:
            label_idx = label.view(-1).long()
            target_label = label.view(-1).long()
        return label_idx, target_label

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

    # --------------------------------
    # Forward wrappers (signature-safe)
    # --------------------------------
    def forward(self, feats, label=None):
        """
        forward_func 지원 형태가 2종일 수 있음:

        (A) grading 전용 형태:
            y_logit, loss, y_prob = forward_func(feats, classifier, loss_func, num_classes, label=..., args=args)

        (B) 기존 classification 형태:
            y_logit, loss, y_prob = forward_func(feats, classifier, loss_func, num_classes, label=...)
            (args 미지원)

        ✅ 둘 다 호환되게 호출한다.
        """
        if self._forward_accepts_args:
            return self.forward_func(
                feats,
                self.classifier,
                self.loss_func,
                self.num_classes,
                label=label,
                args=self.args,
            )

        # fallback: args 없이
        return self.forward_func(
            feats,
            self.classifier,
            self.loss_func,
            self.num_classes,
            label=label,
        )

    def get_attention_maps(self, feats, label=None):
        if self.attention_forward is None:
            raise RuntimeError("attention_forward is None, but get_attention_maps() called.")

        if self._attn_accepts_args:
            return self.attention_forward(
                feats,
                self.classifier,
                self.loss_func,
                self.num_classes,
                label=label,
                args=self.args,
            )

        return self.attention_forward(
            feats,
            self.classifier,
            self.loss_func,
            self.num_classes,
            label=label,
        )

    # --------------------------------
    # training / validation / test
    # --------------------------------
    def training_step(self, batch, batch_idx):
        name, coords, feats, label = batch
        feats = feats.squeeze(0)

        if self._use_patch_drop:
            feats = self._random_patch_subsample(feats)

        label_idx, target_label = self._get_label_idx_and_target(label)

        y_logit, loss, y_prob = self.forward(feats, label=target_label)
        y_logit = self._to_slide_logits(y_logit)

        self.train_acc.update(y_logit.detach(), label_idx)

        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        self.log("Loss/train", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        return loss

    def on_train_epoch_end(self):
        train_acc = self.train_acc.compute().item() * 100.0
        self.log("train/acc", train_acc, sync_dist=False)
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        name, coords, feats, label = batch
        feats = feats.squeeze(0)

        label_idx, target_label = self._get_label_idx_and_target(label)

        y_logit, loss, y_prob = self.forward(feats, label=target_label)
        y_logit = self._to_slide_logits(y_logit)

        self.val_acc.update(y_logit.detach(), label_idx)
        self.val_bacc.update(y_logit.detach(), label_idx)

        pred_idx = torch.argmax(y_logit.detach(), dim=1)
        self.val_qwk.update(pred_idx, label_idx)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)
        self.log("Loss/val", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)

    def on_validation_epoch_end(self):
        val_acc = self.val_acc.compute().item() * 100.0
        val_bacc = self.val_bacc.compute().item() * 100.0
        val_qwk = self.val_qwk.compute().item()

        print(
            f"[VAL][seed={self.seed}] "
            f"QWK: {val_qwk:.4f} | ACC: {val_acc:.2f}% | Balanced ACC: {val_bacc:.2f}%"
        )

        # ✅ main.py EarlyStopping/Checkpoint 모니터용
        self.log("QWK/val", val_qwk, sync_dist=False)
        self.log("val_qwk", val_qwk, prog_bar=True, sync_dist=False)

        # 참고용
        self.log("ACC/val", val_acc, sync_dist=False)
        self.log("ACC_balanced/val", val_bacc, sync_dist=False)
        self.log("val_bacc", val_bacc, prog_bar=True, sync_dist=False)

        if getattr(self.args, "use_weighted_sampler", False):
            self.log("Loss/val_monitor", 1.0 - float(val_qwk), sync_dist=False)

        self.val_acc.reset()
        self.val_bacc.reset()
        self.val_qwk.reset()

    # --------------------------
    # Test
    # --------------------------
    def on_test_epoch_start(self):
        self.test_outputs = []
        self.y_prob_list = []
        self.label_list = []
        self.names = []
        self.logits = None
        self.labels = None

        self.test_acc.reset()
        self.test_bacc.reset()
        self.test_qwk.reset()
        self.test_ece.reset()

    def test_step(self, batch, batch_idx):
        name, coords, feats, label = batch
        feats = feats.squeeze(0)

        label_idx, target_label = self._get_label_idx_and_target(label)

        attn_map = None
        if self.attention_forward is not None:
            out = self.get_attention_maps(feats, label=target_label)
            y_logit, loss, y_prob, attn_map = out[:4]
        else:
            y_logit, loss, y_prob = self.forward(feats, label=target_label)

        # ✅ logits/ probs를 slide-level로 통일
        y_logit = self._to_slide_logits(y_logit)

        # y_prob이 없거나 shape 이상하면 softmax로 대체
        if not isinstance(y_prob, torch.Tensor):
            y_prob = torch.softmax(y_logit.detach(), dim=1)
        else:
            y_prob = self._to_slide_prob(y_prob)
            if y_prob.ndim != 2 or y_prob.size(0) != 1:
                # 최후 방어
                y_prob = torch.softmax(y_logit.detach(), dim=1)

        self.test_acc.update(y_logit.detach(), label_idx)
        self.test_bacc.update(y_logit.detach(), label_idx)

        pred_idx = torch.argmax(y_logit.detach(), dim=1)
        self.test_qwk.update(pred_idx, label_idx)

        self.test_ece.update(y_prob.detach(), label_idx)

        self.log("Loss/final_test", loss, on_epoch=True, sync_dist=False)

        slide_name = name[0] if isinstance(name, (list, tuple)) else name

        self.test_outputs.append(
            {
                "name": slide_name,
                "coords": coords,
                "probs": y_prob.detach().cpu(),           # (1,C)
                "label": label_idx.detach().cpu(),        # (1,)
                "attn": attn_map.detach().cpu() if (attn_map is not None and hasattr(attn_map, "detach")) else None,
            }
        )

        self.y_prob_list.append(y_prob.detach().cpu())
        self.label_list.append(label_idx.detach().cpu())
        self.names.append(slide_name)

    def on_test_epoch_end(self):
        test_acc = self.test_acc.compute().item() * 100.0
        test_bacc = self.test_bacc.compute().item() * 100.0
        test_qwk = self.test_qwk.compute().item()
        test_ece = self.test_ece.compute().item()

        print(
            f"[TEST][seed={self.seed}] "
            f"QWK: {test_qwk:.4f} | ACC: {test_acc:.2f}% | Balanced ACC: {test_bacc:.2f}% | ECE: {test_ece:.4f}"
        )

        self.log("QWK/final_test", test_qwk, sync_dist=False)
        self.log("ACC/final_test", test_acc, sync_dist=False)
        self.log("ACC_balanced/final_test", test_bacc, sync_dist=False)
        self.log("ECE/final_test", test_ece, sync_dist=False)

        self.test_acc.reset()
        self.test_bacc.reset()
        self.test_qwk.reset()
        self.test_ece.reset()

        # save_metrics 호환용 속성
        self.logits = torch.cat(self.y_prob_list, dim=0).detach().cpu() if len(self.y_prob_list) else None
        self.labels = torch.cat(self.label_list, dim=0).detach().cpu() if len(self.label_list) else None

    # --------------------------------
    # Optimizer
    # --------------------------------
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
