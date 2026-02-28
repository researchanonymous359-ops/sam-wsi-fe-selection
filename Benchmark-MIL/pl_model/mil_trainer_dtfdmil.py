# pl_model/mil_trainer_dtfdmil.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import random
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassCalibrationError

from pl_model.optimizers import Lookahead, Lion
from pl_model.forward_fn import dtfdmil_forward_1st_tier, dtfdmil_forward_2nd_tier


class DTFDTrainerModule(pl.LightningModule):
    def __init__(
        self,
        args,
        seed,
        test_dataset_element_name,
        test_class_names_list,
        resolution_str,
        classifier_list,
        loss_func_list,
        metrics,
        patch_path=None,
    ):
        super(DTFDTrainerModule, self).__init__()
        self.save_hyperparameters(ignore=["classifier_list", "loss_func_list", "metrics"])

        self.args = args
        self.seed = seed
        self.test_dataset_element_name = test_dataset_element_name
        self.test_class_names_list = test_class_names_list
        self.patch_path = patch_path

        # 🔥 Common path created in main.py (results/TCGA-RCC/x10/256/DTFD-MIL-AFS/hibou_b/)
        self.base_save_dir = getattr(self.args, "base_save_dir", None)

        # Data1 case: fix colon 7/8-class label names (maintaining existing logic)
        if "data1" in self.test_dataset_element_name:
            if getattr(self.args, "label_type", "") in ["label1_8_class", "label2_8_probs"]:
                self.test_class_names_list = [
                    "HP", "IP", "LP", "SSL", "TA", "TSA", "TVA+VA", "Other"
                ]
            else:
                self.test_class_names_list = ["HP", "IP", "LP", "SSL", "TA", "TSA", "TVA+VA"]

        self.num_classes = len(self.test_class_names_list)
        self.resolution_str = resolution_str

        # Manual Optimization is required for DTFD-MIL
        self.automatic_optimization = False

        # Destructure classifier_list
        self.classifier = classifier_list[0]
        self.attention = classifier_list[1]
        self.dimReduction = classifier_list[2]
        self.UClassifier = classifier_list[3]

        # Loss functions (Tier-1, Tier-2)
        self.loss_func0 = loss_func_list[0]  # 1st-tier loss
        self.loss_func1 = loss_func_list[1]  # 2nd-tier loss (slide-level)

        self.accumulate_grad_batches = args.accumulate_grad_batches

        # Metrics setup (using TorchMetrics)
        balanced_acc = MulticlassAccuracy(num_classes=self.num_classes, average="macro")

        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = MetricCollection(
            {
                "acc": metrics.clone(),
                "balanced_acc": balanced_acc.clone(),
            },
            prefix="val/",
        )

        self.test_metrics = MetricCollection(
            {
                "acc": metrics.clone(),
                "balanced_acc": balanced_acc.clone(),
                "ece": MulticlassCalibrationError(
                    num_classes=self.num_classes,
                    n_bins=args.n_bins,
                ),
            },
            prefix="final_test/",
        )

        # Data buffer to pass to the Callback
        self.test_outputs = []

        # 🔥 Buffer for compatibility with save_metrics.py (for seed-wise / ensemble calculations)
        self.y_prob_list = []
        self.label_list = []
        self.names = []
        self.logits = None
        self.labels = None

        # Patch Drop settings
        self.mil_patch_drop_min = float(getattr(self.args, "mil_patch_drop_min", 0.0))
        self.mil_patch_drop_max = float(getattr(self.args, "mil_patch_drop_max", 0.0))
        self._use_patch_drop = self.mil_patch_drop_max > 0.0

        self.use_weighted_sampler = bool(getattr(self.args, "use_weighted_sampler", False))

        # 🔥 Flag indicating whether to use attention
        self.attention_func = bool(getattr(self.args, "attention", False))

    # ------------------------------------------------------------------
    # Random subsampling function for patches within a bag (MIL Patch Drop)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, feats, label=None, train=False, get_attention=False):
        """
        Returns:
          - Common for train/val/test: loss0, loss1, gSlideLogits
          - If get_attention=True: additionally tAA, patch_indices
        """
        # ---------- 1st tier ----------
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
            # Manual Backward (Tier 1)
            self.manual_backward(loss0, retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.dimReduction.parameters(), self.args.grad_clipping)
            torch.nn.utils.clip_grad_norm_(self.attention.parameters(), self.args.grad_clipping)
            torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), self.args.grad_clipping)

        # ---------- 2nd tier ----------
        loss1, gSlideLogits = dtfdmil_forward_2nd_tier(
            slide_pseudo_feat, self.UClassifier, self.loss_func1, label
        )

        if train:
            # Manual Backward (Tier 2)
            self.manual_backward(loss1)
            torch.nn.utils.clip_grad_norm_(self.UClassifier.parameters(), self.args.grad_clipping)

        # ❗ Do not apply softmax here (use raw logits)
        if get_attention:
            return loss0, loss1, gSlideLogits, tAA, patch_indices
        return loss0, loss1, gSlideLogits

    # ------------------------------------------------------------------
    # training_step
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        name, coords, feats, label = batch
        feats = feats.squeeze(0)  # [1, N, D] -> [N, D]

        if self._use_patch_drop:
            feats = self._random_patch_subsample(feats)

        # Handle Label Shape (e.g., [1, 1, C] -> [1, C])
        if label.ndim == 3:
            label = label.squeeze(0)

        # Forward & Backward (Manual Optimization)
        if isinstance(label, torch.Tensor) and label.ndim > 1:
            # Soft label handling
            loss0, loss1, Y_logits = self.forward(feats, label, train=True)
            label_idx = torch.argmax(label, dim=1)
        else:
            # Standard Hard label
            loss0, loss1, Y_logits = self.forward(feats, label.long(), train=True)
            label_idx = label.long()

        optimizer0, optimizer1 = self.optimizers()
        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            optimizer0.step()
            optimizer1.step()
            optimizer0.zero_grad()
            optimizer1.zero_grad()

        total_loss = loss0 + loss1

        # 🔥 Detach and update metrics (saves memory/increases speed)
        self.train_metrics.update(Y_logits.detach(), label_idx)

        # 🔥 For progress bar (do not record in logger -> reduces I/O)
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
        # 🔥 For internal recording: log only epoch averages to the logger (same pattern as MILTrainer)
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
        # Schedulers Step (CosineAnnealingLR acts on optimizer0/1)
        sch0, sch1 = self.lr_schedulers()
        sch0.step()
        sch1.step()

        # Metrics Logging
        self.log_dict(self.train_metrics.compute(), sync_dist=False)
        self.train_metrics.reset()

    # ------------------------------------------------------------------
    # validation_step
    # ------------------------------------------------------------------
    def validation_step(self, batch, batch_idx):
        name, coords, feats, label = batch
        feats = feats.squeeze(0)

        # Handle Label Shape
        if label.ndim == 3:
            label = label.squeeze(0)

        if isinstance(label, torch.Tensor) and label.ndim > 1:
            loss0, loss1, Y_logits = self.forward(feats, label, train=False)
            label_idx = torch.argmax(label, dim=1)
        else:
            loss0, loss1, Y_logits = self.forward(feats, label.long(), train=False)
            label_idx = label.long()

        total_loss = loss0 + loss1

        # 🔥 Update metric after detaching
        self.val_metrics.update(Y_logits.detach(), label_idx)

        # Display on Progress Bar (val_loss)
        self.log(
            "val_loss",
            total_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=False,
            batch_size=self.args.batch_size,
            prog_bar=True,
        )
        # For Checkpoint
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
        metrics = self.val_metrics.compute()
        bal_acc = metrics["val/balanced_acc"].item() * 100.0

        print(f"[VAL][seed={self.seed}] Balanced Accuracy (macro): {bal_acc:.2f}%")

        # Name for Monitor (ACC_balanced/val) and Progress Bar (val_bacc)
        self.log("ACC_balanced/val", bal_acc, sync_dist=False)
        self.log("val_bacc", bal_acc, prog_bar=True, sync_dist=False)

        # Alternative value for monitoring Loss when using Weighted Sampler (Loss/val_monitor)
        if getattr(self.args, "use_weighted_sampler", False):
            self.log("Loss/val_monitor", 100.0 - bal_acc, sync_dist=False)

        self.log_dict(metrics, sync_dist=False)
        self.val_metrics.reset()

    # ------------------------------------------------------------------
    # Test Hooks & Step
    # ------------------------------------------------------------------
    def on_test_epoch_start(self):
        # Initialize per seed / dataset
        self.test_outputs = []
        self.y_prob_list = []
        self.label_list = []
        self.names = []
        self.logits = None
        self.labels = None

    def test_step(self, batch, batch_idx):
        name, coords, feats, label = batch
        feats = feats.squeeze(0)

        if label.ndim == 3:
            label = label.squeeze(0)

        # DTFD requires tAA and patch_indices to obtain the Attention Map
        attn_weights = None
        target_label = label if label.ndim > 1 else label.long()

        if self.args.attention:
            loss0, loss1, Y_logits, tAA, patch_indices = self.forward(
                feats, label=target_label, train=False, get_attention=True
            )
            # DTFD: tAA is the attention score of the pseudo bag
            attn_weights = tAA.detach().cpu()
        else:
            loss0, loss1, Y_logits = self.forward(feats, label=target_label, train=False)

        # 🔥 1) If there is a group dimension -> aggregate at the slide level
        #  - Y_logits: Might be (G, C) (G = numGroup)
        if Y_logits.dim() == 1:
            # If (C,) -> (1, C)
            Y_logits_slide = Y_logits.unsqueeze(0)
        elif Y_logits.dim() == 2 and Y_logits.size(0) > 1:
            # If (G, C) -> Create representative slide logit by averaging groups
            Y_logits_slide = Y_logits.mean(dim=0, keepdim=True)
        else:
            # If already in (1, C) shape, use as is
            Y_logits_slide = Y_logits

        # 🔥 2) Apply softmax only once for calibration in test (save probability)
        Y_prob = F.softmax(Y_logits_slide, dim=1)

        # Label index (handles both soft/hard labels) -> Convert to 1D tensor
        if label.ndim > 1:
            label_idx = torch.argmax(label, dim=1)
        else:
            label_idx = label.view(-1)  # scalar -> (1,)

        total_loss = loss0 + loss1

        # 🔥 3) Update metrics with slide-level probabilities (detach)
        self.test_metrics.update(Y_prob.detach(), label_idx)
        self.log("Loss/final_test", total_loss, on_epoch=True, sync_dist=False)

        # 🔥 4) Save data for Callback analysis (move to CPU)
        slide_name = name[0] if isinstance(name, (list, tuple)) else name

        self.test_outputs.append(
            {
                "name": slide_name,
                "coords": coords,
                "probs": Y_prob.detach().cpu(),     # (1, C) slide level
                "label": label_idx.detach().cpu(),  # (1,)
                "attn": attn_weights,               # For DTFD, tAA (Tier-1 attention scores)
            }
        )

        # 🔥 5) Simultaneously fill buffer for compatibility with save_metrics.py
        self.y_prob_list.append(Y_prob.detach().cpu())    # (1, C)
        self.label_list.append(label_idx.detach().cpu())  # (1,)
        self.names.append(slide_name)

    def on_test_epoch_end(self):
        # 1) Aggregate metrics
        self.log_dict(self.test_metrics.compute(), sync_dist=False)
        self.test_metrics.reset()

        # 2) Fill attributes for compatibility with save_metrics.py
        if len(self.y_prob_list) > 0:
            # (N, C) probability tensor
            self.logits = torch.cat(self.y_prob_list, dim=0).detach().cpu()
        else:
            self.logits = None

        if len(self.label_list) > 0:
            # (N,) label tensor
            self.labels = torch.cat(self.label_list, dim=0).detach().cpu()
        else:
            self.labels = None
        # Use self.names as a list directly
        # self.test_outputs is emptied after being used in callbacks.analysis_callback

    # ------------------------------------------------------------------
    # Optimizer / Scheduler
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        # ---- 1. Parameter Grouping ----
        trainable_parameters = []
        trainable_parameters += list(self.classifier.parameters())
        trainable_parameters += list(self.attention.parameters())
        trainable_parameters += list(self.dimReduction.parameters())

        params_optimizer_0 = [{"params": filter(lambda p: p.requires_grad, trainable_parameters)}]
        params_optimizer_1 = [{"params": filter(lambda p: p.requires_grad, self.UClassifier.parameters())}]

        # ---- 2. Optimizer Creation Helper ----
        def create_optimizer(params, opt_name, lr, wd):
            opt_name = opt_name.lower()
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

        opt_name = self.args.opt or "adam"
        optimizer0 = create_optimizer(
            params_optimizer_0, opt_name, self.args.lr, self.args.weight_decay
        )
        optimizer1 = create_optimizer(
            params_optimizer_1, opt_name, self.args.lr, self.args.weight_decay
        )

        # ---- 3. Scheduler ----
        scheduler0 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer0, T_max=self.args.epochs, eta_min=5e-6
        )
        scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer1, T_max=self.args.epochs, eta_min=5e-6
        )

        return [optimizer0, optimizer1], [scheduler0, scheduler1]