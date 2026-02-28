# pl_model/mil_trainer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import random
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassCalibrationError

from pl_model.optimizers import Lookahead, Lion


class MILTrainerModule(pl.LightningModule):
    def __init__(
        self,
        args,
        seed,
        test_dataset_element_name,  # For creating logging paths
        test_class_names_list,      # Class names for analysis
        num_classes,
        resolution_str,
        classifier,
        loss_func,
        metrics,                    # (Default metrics passed from main.py)
        forward_func="general",
        attention_func=None,
        patch_path=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["classifier", "loss_func", "metrics", "attention_func"])

        self.args = args
        self.seed = seed
        self.num_classes = num_classes
        self.resolution_str = resolution_str
        self.test_dataset_element_name = test_dataset_element_name
        self.test_class_names_list = test_class_names_list
        self.patch_path = patch_path

        # 🔥 Common base path created in main.py (results/.../mil/fe/)
        self.base_save_dir = getattr(self.args, "base_save_dir", None)

        # Model components
        self.classifier = classifier
        self.loss_func = loss_func
        self.forward_func = forward_func

        # Attention function:
        #   - Internal callable: self.attention_forward
        #   - Bool flag for callback compatibility: self.attention_func
        self.attention_forward = attention_func
        self.attention_func = attention_func is not None

        # Optimization settings (Standard MIL uses automatic optimization)
        self.automatic_optimization = True

        # Patch Drop settings (Same method as DTFD)
        self.mil_patch_drop_min = float(getattr(self.args, "mil_patch_drop_min", 0.0))
        self.mil_patch_drop_max = float(getattr(self.args, "mil_patch_drop_max", 0.0))
        self._use_patch_drop = self.mil_patch_drop_max > 0.0

        # Flag for using Weighted Sampler (for changing val monitoring criteria)
        self.use_weighted_sampler = bool(getattr(self.args, "use_weighted_sampler", False))

        # Metrics setup (using TorchMetrics)
        balanced_acc = MulticlassAccuracy(num_classes=num_classes, average="macro")

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
                "ece": MulticlassCalibrationError(num_classes=num_classes, n_bins=args.n_bins),
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

    # --------------------------------
    # Forward & Attention wrappers
    # --------------------------------
    def forward(self, feats, label=None):
        """
        forward_func assumes the following format:
          y_logit, loss, y_prob = forward_func(feats, classifier, loss_func, num_classes, label=label)
        """
        return self.forward_func(
            feats,
            self.classifier,
            self.loss_func,
            self.num_classes,
            label=label,
        )

    def get_attention_maps(self, feats, label=None):
        """
        attention_forward assumes the following format:
          y_logit, loss, y_prob, attn_map = attention_forward(...)
        """
        return self.attention_forward(
            feats,
            self.classifier,
            self.loss_func,
            self.num_classes,
            label=label,
        )

    # --------------------------------
    # Patch Subsampling (MIL Patch Drop)
    # --------------------------------
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
    # training / validation / test step
    # --------------------------------
    def training_step(self, batch, batch_idx):
        name, coords, feats, label = batch
        feats = feats.squeeze(0)

        if self._use_patch_drop:
            feats = self._random_patch_subsample(feats)

        # Handle Label
        target_label = label
        label_idx = label
        if isinstance(label, torch.Tensor) and label.ndim > 1:
            # soft-label (one-hot or prob)
            label_idx = torch.argmax(label, dim=1)
        else:
            target_label = label.long()
            label_idx = label.long()

        # forward_func returns (y_logit, loss, y_prob)
        y_logit, loss, y_prob = self.forward(feats, label=target_label)

        # train metric is updated based on logits
        self.train_metrics.update(y_logit.detach(), label_idx)

        # 🔥 For Progress Bar (do not log to logger -> reduces I/O)
        self.log(
            "train_loss",
            loss,
            on_step=True,        # Update pbar every step
            on_epoch=False,
            sync_dist=False,
            batch_size=self.args.batch_size,
            prog_bar=True,
            logger=False,        # Do not record in CSVLogger, etc.
        )
        # 🔥 For internal recording: log only epoch averages to the logger (same meaning as original Loss/train)
        self.log(
            "Loss/train",
            loss,
            on_step=False,       # Remove per-step logging
            on_epoch=True,       # Log only epoch averages
            sync_dist=False,
            batch_size=self.args.batch_size,
            prog_bar=False,
            logger=True,
        )

        return loss

    def on_train_epoch_end(self):
        # Aggregate metrics per epoch
        self.log_dict(self.train_metrics.compute(), sync_dist=False)
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        name, coords, feats, label = batch
        feats = feats.squeeze(0)

        target_label = label
        label_idx = label
        if isinstance(label, torch.Tensor) and label.ndim > 1:
            label_idx = torch.argmax(label, dim=1)
        else:
            target_label = label.long()
            label_idx = label.long()

        y_logit, loss, y_prob = self.forward(feats, label=target_label)

        # val metric also based on logits
        self.val_metrics.update(y_logit.detach(), label_idx)

        # Progress Bar
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=False,
            batch_size=self.args.batch_size,
            prog_bar=True,
        )
        # For Checkpoint
        self.log(
            "Loss/val",
            loss,
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

        # For Monitor + Progress bar
        self.log("ACC_balanced/val", bal_acc, sync_dist=False)
        self.log("val_bacc", bal_acc, prog_bar=True, sync_dist=False)

        # Alternative value for monitoring Loss when using Weighted Sampler
        if getattr(self.args, "use_weighted_sampler", False):
            self.log("Loss/val_monitor", 100.0 - bal_acc, sync_dist=False)

        self.log_dict(metrics, sync_dist=False)
        self.val_metrics.reset()

    # --------------------------
    # Test Hooks
    # --------------------------
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

        target_label = label
        label_idx = label
        if isinstance(label, torch.Tensor) and label.ndim > 1:
            label_idx = torch.argmax(label, dim=1)
        else:
            target_label = label.long()
            label_idx = label.long()

        # Handle cases requiring Attention Map
        attn_map = None
        if self.attention_forward is not None:
            y_logit, loss, y_prob, attn_map = self.get_attention_maps(feats, label=target_label)
        else:
            y_logit, loss, y_prob = self.forward(feats, label=target_label)

        # test_metrics has ECE, so update based on probability
        self.test_metrics.update(y_prob.detach(), label_idx)
        self.log(
            "Loss/final_test",
            loss,
            on_epoch=True,
            sync_dist=False,
            batch_size=self.args.batch_size,
        )

        # Save data for Callback analysis
        slide_name = name[0] if isinstance(name, (list, tuple)) else name

        self.test_outputs.append(
            {
                "name": slide_name,
                "coords": coords,                      # Coordinates are generally CPU tensor/list
                "probs": y_prob.detach().cpu(),      # (1, C) or (G, C)
                "label": label_idx.detach().cpu(),   # (1,)
                "attn": attn_map.detach().cpu() if attn_map is not None else None,
            }
        )

        # 🔥 Buffer for compatibility with save_metrics.py
        self.y_prob_list.append(y_prob.detach().cpu())
        self.label_list.append(label_idx.detach().cpu())
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

    # --------------------------------
    # Optimizer & Scheduler
    # --------------------------------
    def configure_optimizers(self):
        params = [
            {
                "params": filter(lambda p: p.requires_grad, self.classifier.parameters())
            }
        ]
        opt_name = (self.args.opt or "adam").lower()

        if opt_name == "adam":
            optimizer = torch.optim.Adam(
                params,
                lr=self.args.lr,
                betas=(0.5, 0.9),
                weight_decay=self.args.weight_decay,
            )
        elif opt_name == "adamw":
            optimizer = torch.optim.AdamW(
                params,
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
            )
        elif opt_name == "lion":
            optimizer = Lion(
                params,
                lr=self.args.lr,
                betas=(0.9, 0.99),
                weight_decay=self.args.weight_decay,
            )
        elif opt_name == "lookahead_radam":
            base = torch.optim.RAdam(
                params,
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
            )
            optimizer = Lookahead(base, k=5, alpha=0.5)
        elif opt_name == "radam":
            optimizer = torch.optim.RAdam(
                params,
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.args.epochs, eta_min=5e-6
        )
        return [optimizer], [scheduler]