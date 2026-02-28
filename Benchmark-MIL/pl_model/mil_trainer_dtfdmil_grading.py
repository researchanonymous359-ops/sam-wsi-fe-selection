# pl_model/mil_trainer_dtfdmil_grading.py
import random

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassCalibrationError,
    MulticlassCohenKappa,
)

from pl_model.forward_fn import dtfdmil_forward_1st_tier, dtfdmil_forward_2nd_tier
from pl_model.optimizers import Lookahead, Lion


class DTFDGradingTrainerModule(pl.LightningModule):

    def __init__(
        self,
        args,
        seed,
        test_dataset_element_name,
        test_class_names_list,
        resolution_str,
        classifier_list,
        loss_func_list,
        metrics,  # metric collection passed from main.py (acc/f1/etc.)
        patch_path=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["classifier_list", "loss_func_list", "metrics"])

        self.args = args
        self.seed = seed
        self.test_dataset_element_name = test_dataset_element_name
        self.test_class_names_list = test_class_names_list
        self.patch_path = patch_path

        self.base_save_dir = getattr(self.args, "base_save_dir", None)

        # Dataset-specific label override (data1 + 8-class label types)
        if "data1" in self.test_dataset_element_name:
            if getattr(self.args, "label_type", "") in ["label1_8_class", "label2_8_probs"]:
                self.test_class_names_list = ["HP", "IP", "LP", "SSL", "TA", "TSA", "TVA+VA", "Other"]
            else:
                self.test_class_names_list = ["HP", "IP", "LP", "SSL", "TA", "TSA", "TVA+VA"]

        self.num_classes = len(self.test_class_names_list)
        self.resolution_str = resolution_str

        # DTFD-MIL uses manual optimization
        self.automatic_optimization = False

        # Unpack classifier_list
        self.classifier = classifier_list[0]
        self.attention = classifier_list[1]
        self.dimReduction = classifier_list[2]
        self.UClassifier = classifier_list[3]

        # Loss functions
        self.loss_func0 = loss_func_list[0]
        self.loss_func1 = loss_func_list[1]

        self.accumulate_grad_batches = int(getattr(args, "accumulate_grad_batches", 1))

        # -------------------------
        # Metrics
        # -------------------------
        # Train: reuse the provided metric collection (acc/f1/precision/recall/etc.)
        self.train_metrics = metrics.clone(prefix="train/")

        # Val/Test: for QWK it is safer to update with pred_idx(int) vs label_idx(int)
        self.val_acc_metric = MulticlassAccuracy(num_classes=self.num_classes, average="micro")
        self.val_bacc_metric = MulticlassAccuracy(num_classes=self.num_classes, average="macro")
        self.val_qwk_metric = MulticlassCohenKappa(num_classes=self.num_classes, weights="quadratic")

        self.test_acc_metric = MulticlassAccuracy(num_classes=self.num_classes, average="micro")
        self.test_bacc_metric = MulticlassAccuracy(num_classes=self.num_classes, average="macro")
        self.test_qwk_metric = MulticlassCohenKappa(num_classes=self.num_classes, weights="quadratic")
        self.test_ece_metric = MulticlassCalibrationError(
            num_classes=self.num_classes, n_bins=int(getattr(args, "n_bins", 15))
        )

        # Buffer for analysis callback
        self.test_outputs = []

        # Buffers for save_metrics_grading.py compatibility
        self.y_prob_list = []
        self.label_list = []
        self.names = []
        self.logits = None
        self.labels = None

        # Patch drop config
        self.mil_patch_drop_min = float(getattr(self.args, "mil_patch_drop_min", 0.0))
        self.mil_patch_drop_max = float(getattr(self.args, "mil_patch_drop_max", 0.0))
        self._use_patch_drop = self.mil_patch_drop_max > 0.0

        # Whether attention is enabled
        self.attention_func = bool(getattr(self.args, "attention", False))

        # -------------------------
        # Debug flags
        # -------------------------
        self.debug_grading = bool(getattr(self.args, "debug_grading", False))
        self._debug_print_every = int(getattr(self.args, "debug_print_every", 1))  # reserved

        if self.debug_grading:
            print(
                f"[DEBUG][DTFDGrading:init] seed={self.seed} "
                f"train_mode={getattr(self.args,'train_mode',None)} "
                f"mil_model={getattr(self.args,'mil_model',None)} "
                f"num_classes={self.num_classes} "
                f"accumulate_grad_batches={self.accumulate_grad_batches} "
                f"patch_drop=({self.mil_patch_drop_min},{self.mil_patch_drop_max}) "
                f"attention={getattr(self.args,'attention',False)} "
                f"grad_clipping={getattr(self.args,'grad_clipping',None)}"
            )

    # ---------------------------------------------------------
    # Core: unify 2nd-tier logits to slide-level (1, C) before metric updates
    # ---------------------------------------------------------
    def _to_slide_logits(self, Y_logits: torch.Tensor) -> torch.Tensor:
        """
        Convert logits to slide-level shape (1, C):
        - (C,)      -> (1, C)
        - (1, C)    -> (1, C)
        - (G, C)    -> (1, C) by mean over G
        - (1, G, C) -> (1, C) by mean over G (defensive)
        """
        if Y_logits is None:
            raise ValueError("Y_logits is None")

        if Y_logits.dim() == 1:
            return Y_logits.unsqueeze(0)

        if Y_logits.dim() == 2:
            if Y_logits.size(0) > 1:
                return Y_logits.mean(dim=0, keepdim=True)
            return Y_logits

        # Defensive handling for unexpected shapes such as (1, G, C)
        if Y_logits.dim() == 3 and Y_logits.size(0) == 1:
            return Y_logits.mean(dim=1)

        raise ValueError(f"Unexpected Y_logits shape: {tuple(Y_logits.shape)}")

    # -----------------------------
    # Patch-drop
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
    # Forward (DTFD 1st/2nd tier)
    # -----------------------------
    def forward(self, feats, label=None, train: bool = False, get_attention: bool = False):
        if get_attention:
            loss0, slide_pseudo_feat, tAA, patch_indices = dtfdmil_forward_1st_tier(
                self.args,
                feats,
                self.classifier,
                self.attention,
                self.dimReduction,
                self.loss_func0,
                label,
                get_attention=True,
            )
        else:
            loss0, slide_pseudo_feat = dtfdmil_forward_1st_tier(
                self.args,
                feats,
                self.classifier,
                self.attention,
                self.dimReduction,
                self.loss_func0,
                label,
            )

        if train:
            self.manual_backward(loss0, retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.dimReduction.parameters(), self.args.grad_clipping)
            torch.nn.utils.clip_grad_norm_(self.attention.parameters(), self.args.grad_clipping)
            torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), self.args.grad_clipping)

        loss1, gSlideLogits = dtfdmil_forward_2nd_tier(
            slide_pseudo_feat, self.UClassifier, self.loss_func1, label
        )

        if train:
            self.manual_backward(loss1)
            torch.nn.utils.clip_grad_norm_(self.UClassifier.parameters(), self.args.grad_clipping)

        if get_attention:
            return loss0, loss1, gSlideLogits, tAA, patch_indices
        return loss0, loss1, gSlideLogits

    # -----------------------------
    # Training
    # -----------------------------
    def training_step(self, batch, batch_idx):
        name, coords, feats, label = batch
        feats = feats.squeeze(0)

        if self._use_patch_drop:
            feats = self._random_patch_subsample(feats)

        # Normalize label shape
        if isinstance(label, torch.Tensor) and label.ndim == 3:
            label = label.squeeze(0)

        if isinstance(label, torch.Tensor) and label.ndim > 1:
            loss0, loss1, Y_logits = self.forward(feats, label, train=True)
            label_idx = torch.argmax(label, dim=1)  # (1,)
        else:
            loss0, loss1, Y_logits = self.forward(feats, label.long(), train=True)
            label_idx = label.view(-1).long()       # unify to (1,)

        optimizer0, optimizer1 = self.optimizers()

        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            optimizer0.step()
            optimizer1.step()
            optimizer0.zero_grad()
            optimizer1.zero_grad()

        total_loss = loss0 + loss1

        # Unify to slide-level logits before metric update (avoid shape mismatch)
        Y_logits_slide = self._to_slide_logits(Y_logits)
        self.train_metrics.update(Y_logits_slide.detach(), label_idx)

        self.log(
            "train_loss",
            total_loss,
            on_step=True,
            on_epoch=False,
            sync_dist=False,
            batch_size=self.args.batch_size,
            prog_bar=True,
            logger=False,
        )
        self.log(
            "Loss/train",
            total_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=False,
            batch_size=self.args.batch_size,
            prog_bar=False,
            logger=True,
        )
        return total_loss

    def on_train_epoch_end(self):
        # Step schedulers
        sch0, sch1 = self.lr_schedulers()
        sch0.step()
        sch1.step()

        self.log_dict(self.train_metrics.compute(), sync_dist=False)
        self.train_metrics.reset()

    # -----------------------------
    # Validation
    # -----------------------------
    def validation_step(self, batch, batch_idx):
        name, coords, feats, label = batch
        feats = feats.squeeze(0)

        if isinstance(label, torch.Tensor) and label.ndim == 3:
            label = label.squeeze(0)

        if isinstance(label, torch.Tensor) and label.ndim > 1:
            loss0, loss1, Y_logits = self.forward(feats, label, train=False)
            label_idx = torch.argmax(label, dim=1)  # (1,)
        else:
            loss0, loss1, Y_logits = self.forward(feats, label.long(), train=False)
            label_idx = label.view(-1).long()       # unify to (1,)

        total_loss = loss0 + loss1

        # Unify (G, C) -> (1, C)
        Y_logits_slide = self._to_slide_logits(Y_logits)

        # Update val metrics (shape-safe)
        self.val_acc_metric.update(Y_logits_slide.detach(), label_idx)
        self.val_bacc_metric.update(Y_logits_slide.detach(), label_idx)

        pred_idx = torch.argmax(Y_logits_slide.detach(), dim=1)  # (1,)
        self.val_qwk_metric.update(pred_idx, label_idx)          # (1,) vs (1,)

        self.log(
            "val_loss",
            total_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=False,
            batch_size=self.args.batch_size,
            prog_bar=True,
        )
        self.log(
            "Loss/val",
            total_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=False,
            batch_size=self.args.batch_size,
            prog_bar=False,
        )

    def on_validation_epoch_end(self):
        # Compute
        val_acc = float(self.val_acc_metric.compute().item()) * 100.0
        val_bacc = float(self.val_bacc_metric.compute().item()) * 100.0
        val_qwk = float(self.val_qwk_metric.compute().item())  # 0~1

        print(
            f"[VAL][grading][seed={self.seed}] "
            f"QWK={val_qwk:.4f} | BalancedAcc={val_bacc:.2f}% | ACC={val_acc:.2f}%"
        )

        # ==========================================================
        # MUST log monitor keys for EarlyStopping/Checkpoint
        # ==========================================================
        self.log("QWK/val", val_qwk, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
        self.log("val_qwk", val_qwk, on_step=False, on_epoch=True, prog_bar=True,  logger=True, sync_dist=False)

        # Additional reference logs
        self.log("ACC/val", val_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
        self.log("ACC_balanced/val", val_bacc, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
        self.log("val_bacc", val_bacc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)

        # Reset
        self.val_acc_metric.reset()
        self.val_bacc_metric.reset()
        self.val_qwk_metric.reset()

    # -----------------------------
    # Test
    # -----------------------------
    def on_test_epoch_start(self):
        self.test_outputs = []
        self.y_prob_list = []
        self.label_list = []
        self.names = []
        self.logits = None
        self.labels = None

        self.test_acc_metric.reset()
        self.test_bacc_metric.reset()
        self.test_qwk_metric.reset()
        self.test_ece_metric.reset()

    def test_step(self, batch, batch_idx):
        name, coords, feats, label = batch
        feats = feats.squeeze(0)

        if isinstance(label, torch.Tensor) and label.ndim == 3:
            label = label.squeeze(0)

        attn_weights = None
        target_label = label if (isinstance(label, torch.Tensor) and label.ndim > 1) else label.long()

        if getattr(self.args, "attention", False):
            loss0, loss1, Y_logits, tAA, patch_indices = self.forward(
                feats, label=target_label, train=False, get_attention=True
            )
            attn_weights = tAA.detach().cpu()
        else:
            loss0, loss1, Y_logits = self.forward(feats, label=target_label, train=False)

        # Unify logits and compute probs
        Y_logits_slide = self._to_slide_logits(Y_logits)
        Y_prob = F.softmax(Y_logits_slide, dim=1)

        if isinstance(label, torch.Tensor) and label.ndim > 1:
            label_idx = torch.argmax(label, dim=1)  # (1,)
        else:
            label_idx = label.view(-1).long()       # (1,)

        total_loss = loss0 + loss1

        # Update test metrics
        self.test_acc_metric.update(Y_logits_slide.detach(), label_idx)
        self.test_bacc_metric.update(Y_logits_slide.detach(), label_idx)

        pred_idx = torch.argmax(Y_logits_slide.detach(), dim=1)
        self.test_qwk_metric.update(pred_idx, label_idx)

        self.test_ece_metric.update(Y_prob.detach(), label_idx)

        self.log("Loss/final_test", total_loss, on_step=False, on_epoch=True, sync_dist=False)

        slide_name = name[0] if isinstance(name, (list, tuple)) else name

        # Store for analysis callback
        self.test_outputs.append(
            {
                "name": slide_name,
                "coords": coords,
                "probs": Y_prob.detach().cpu(),     # (1, C)
                "label": label_idx.detach().cpu(),  # (1,)
                "attn": attn_weights,
            }
        )
        self.y_prob_list.append(Y_prob.detach().cpu())
        self.label_list.append(label_idx.detach().cpu())
        self.names.append(slide_name)

    def on_test_epoch_end(self):
        test_acc = float(self.test_acc_metric.compute().item())
        test_bacc = float(self.test_bacc_metric.compute().item())
        test_qwk = float(self.test_qwk_metric.compute().item())
        test_ece = float(self.test_ece_metric.compute().item())

        # Log so they appear in Lightning's table
        self.log("final_test/acc", test_acc, sync_dist=False)
        self.log("final_test/balanced_acc", test_bacc, sync_dist=False)
        self.log("final_test/qwk", test_qwk, sync_dist=False)
        self.log("final_test/ece", test_ece, sync_dist=False)

        print(
            f"[TEST][grading][seed={self.seed}] "
            f"QWK={test_qwk:.4f} | BAcc={test_bacc*100:.2f}% | ACC={test_acc*100:.2f}% | ECE={test_ece:.4f}"
        )

        # For save_metrics compatibility
        self.logits = torch.cat(self.y_prob_list, dim=0).detach().cpu() if len(self.y_prob_list) else None
        self.labels = torch.cat(self.label_list, dim=0).detach().cpu() if len(self.label_list) else None

        # Reset
        self.test_acc_metric.reset()
        self.test_bacc_metric.reset()
        self.test_qwk_metric.reset()
        self.test_ece_metric.reset()

    # -----------------------------
    # Optimizers / schedulers
    # -----------------------------
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
        optimizer0 = create_optimizer(params_optimizer_0, opt_name, self.args.lr, self.args.weight_decay)
        optimizer1 = create_optimizer(params_optimizer_1, opt_name, self.args.lr, self.args.weight_decay)

        scheduler0 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer0, T_max=self.args.epochs, eta_min=5e-6)
        scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=self.args.epochs, eta_min=5e-6)

        return [optimizer0, optimizer1], [scheduler0, scheduler1]