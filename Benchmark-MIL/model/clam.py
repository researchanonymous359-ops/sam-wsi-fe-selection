# model/clam.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def initialize_weights(module: nn.Module) -> None:
    """Initialize weights for Linear and BatchNorm1d layers."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


# ===========================
# Attention Network (no gate)
# ===========================
class Attn_Net(nn.Module):
    def __init__(self, L: int = 1024, D: int = 256, dropout: bool = False, n_classes: int = 1):
        super().__init__()
        layers = [
            nn.Linear(L, D),
            nn.Tanh(),
        ]
        if dropout:
            layers.append(nn.Dropout(0.25))
        layers.append(nn.Linear(D, n_classes))
        self.module = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [N, L]
        Returns:
            A: [N, n_classes] (unnormalized attention scores)
            x: [N, L] (input features, returned for convenience)
        """
        return self.module(x), x


# ===============================
# Attention Network (gated)
# ===============================
class Attn_Net_Gated(nn.Module):
    def __init__(self, L: int = 1024, D: int = 256, dropout: bool = False, n_classes: int = 1):
        super().__init__()
        attn_a = [
            nn.Linear(L, D),
            nn.Tanh(),
        ]
        attn_b = [
            nn.Linear(L, D),
            nn.Sigmoid(),
        ]
        if dropout:
            attn_a.append(nn.Dropout(0.25))
            attn_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*attn_a)
        self.attention_b = nn.Sequential(*attn_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [N, L]
        Returns:
            A: [N, n_classes] (unnormalized attention scores)
            x: [N, L] (input features, returned for convenience)
        """
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a * b                    # [N, D]
        A = self.attention_c(A)      # [N, n_classes]
        return A, x


# ===========================
# CLAM-SB (Single Branch)
# ===========================
class CLAM_SB(nn.Module):
    def __init__(
        self,
        gate: bool = True,
        size_arg: str = "small",
        dropout: bool = False,
        k_sample: int = 8,
        n_classes: int = 2,
        instance_loss_fn: str = "ce",   # SVM removed; only CE is used
        subtyping: bool = False,
        embed_dim: int = 1024,
        bag_weight: float = 0.7,
    ):
        super().__init__()

        self.size_dict = {
            "small": [embed_dim, 512, 256],
            "big": [embed_dim, 512, 384],
        }
        size = self.size_dict[size_arg]

        self.bag_weight = bag_weight
        self.k_sample = k_sample
        self.n_classes = n_classes
        self.subtyping = subtyping

        # Feature -> attention input projection
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))

        # Attention network
        if gate:
            attn_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        else:
            attn_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)

        fc.append(attn_net)
        self.attention_net = nn.Sequential(*fc)

        # Bag-level classifier
        self.classifiers = nn.Linear(size[1], n_classes)

        # Instance-level classifiers (positive / negative)
        inst_cls = [nn.Linear(size[1], 2) for _ in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(inst_cls)

        # Instance loss: unified to CrossEntropy
        self.instance_loss_fn = nn.CrossEntropyLoss()

        initialize_weights(self)

    @staticmethod
    def create_positive_targets(k: int, device: torch.device) -> torch.Tensor:
        """Create positive targets of shape [k] with label 1."""
        return torch.full((k,), 1, device=device, dtype=torch.long)

    @staticmethod
    def create_negative_targets(k: int, device: torch.device) -> torch.Tensor:
        """Create negative targets of shape [k] with label 0."""
        return torch.full((k,), 0, device=device, dtype=torch.long)

    # In-class instance-level evaluation
    def inst_eval(self, A: torch.Tensor, h: torch.Tensor, classifier: nn.Module):
        """
        Args:
            A: [N] or [1, N] attention scores for a single class
            h: [N, D] instance features
        Returns:
            instance_loss: scalar tensor
            preds: [2k] predicted instance labels
            all_targets: [2k] ground-truth instance labels
        """
        device = h.device

        if A.dim() == 1:
            A = A.view(1, -1)  # [1, N]

        # Select top-k positive / negative instances
        top_p_ids = torch.topk(A, self.k_sample, dim=1)[1][-1]    # [k]
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]   # [k]

        top_p = h.index_select(0, top_p_ids)  # [k, D]
        top_n = h.index_select(0, top_n_ids)  # [k, D]

        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_instances = torch.cat([top_p, top_n], dim=0)        # [2k, D]
        all_targets = torch.cat([p_targets, n_targets], dim=0)  # [2k]

        logits = classifier(all_instances)  # [2k, 2]
        instance_loss = self.instance_loss_fn(logits, all_targets)
        preds = torch.argmax(logits, dim=1)

        return instance_loss, preds, all_targets

    # Out-of-class instance-level evaluation
    def inst_eval_out(self, A: torch.Tensor, h: torch.Tensor, classifier: nn.Module):
        """
        Args:
            A: [N] or [1, N] attention scores for a single class
            h: [N, D] instance features
        Returns:
            instance_loss: scalar tensor
            preds: [k] predicted instance labels
            targets: [k] ground-truth instance labels (all 0)
        """
        device = h.device
        if A.dim() == 1:
            A = A.view(1, -1)

        top_p_ids = torch.topk(A, self.k_sample, dim=1)[1][-1]
        top_p = h.index_select(0, top_p_ids)  # [k, D]

        targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)  # [k, 2]
        instance_loss = self.instance_loss_fn(logits, targets)
        preds = torch.argmax(logits, dim=1)

        return instance_loss, preds, targets

    def forward(
        self,
        h: torch.Tensor,
        label: torch.Tensor = None,
        instance_eval: bool = False,
        return_features: bool = False,
        attention_only: bool = False,
    ):
        """
        Args:
            h: [N, embed_dim] instance features in a bag
            label: scalar long tensor (bag label). Optional.
            instance_eval: whether to compute instance-level loss/preds
            return_features: whether to return bag-level features
            attention_only: if True, return raw attention only
        Returns:
            logits: [1, n_classes]
            Y_prob: [1, n_classes]
            Y_hat: [1]
            A_raw: [1, N] raw attention scores (before softmax)
            results_dict: dict with optional instance-level outputs / features
        """
        device = h.device

        A, h = self.attention_net(h)  # A: [N, 1], h: [N, H]
        A = A.transpose(1, 0)         # [1, N]

        if attention_only:
            return A

        A_raw = A
        A = F.softmax(A, dim=1)       # [1, N]

        total_inst_loss = torch.tensor(0.0, device=device)
        all_preds, all_targets = [], []

        if instance_eval and label is not None:
            # Convert scalar label to one-hot: [n_classes]
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze(0)

            for i in range(self.n_classes):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]

                if inst_label == 1:
                    # In-class branch
                    instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                else:
                    # Out-of-class branch (optional, only if subtyping)
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                    else:
                        continue

                total_inst_loss = total_inst_loss + instance_loss
                all_preds.extend(preds.detach().cpu().numpy())
                all_targets.extend(targets.detach().cpu().numpy())

            if self.subtyping and len(self.instance_classifiers) > 0:
                total_inst_loss = total_inst_loss / len(self.instance_classifiers)

        # Bag-level representation
        M = torch.mm(A, h)            # [1, H]
        logits = self.classifiers(M)  # [1, n_classes]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)

        results_dict = {}
        if instance_eval:
            results_dict["instance_loss"] = total_inst_loss
            results_dict["inst_labels"] = np.array(all_targets)
            results_dict["inst_preds"] = np.array(all_preds)

        if return_features:
            results_dict["features"] = M

        return logits, Y_prob, Y_hat, A_raw, results_dict


# ===========================
# CLAM-MB (Multi Branch)
# ===========================
class CLAM_MB(CLAM_SB):
    def __init__(
        self,
        gate: bool = True,
        size_arg: str = "small",
        dropout: bool = False,
        k_sample: int = 8,
        n_classes: int = 2,
        instance_loss_fn: str = "ce",   # same: only CE is used
        subtyping: bool = False,
        embed_dim: int = 1024,
        bag_weight: float = 0.7,
    ):
        nn.Module.__init__(self)

        self.size_dict = {
            "small": [embed_dim, 512, 256],
            "big": [embed_dim, 512, 384],
        }
        size = self.size_dict[size_arg]

        self.bag_weight = bag_weight
        self.k_sample = k_sample
        self.n_classes = n_classes
        self.subtyping = subtyping

        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))

        if gate:
            attn_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
        else:
            attn_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)

        fc.append(attn_net)
        self.attention_net = nn.Sequential(*fc)

        # Per-class bag classifier heads (each outputs a scalar)
        bag_classifiers = [nn.Linear(size[1], 1) for _ in range(n_classes)]
        self.classifiers = nn.ModuleList(bag_classifiers)

        inst_cls = [nn.Linear(size[1], 2) for _ in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(inst_cls)

        # Instance loss: unified to CrossEntropy
        self.instance_loss_fn = nn.CrossEntropyLoss()

        initialize_weights(self)

    def forward(
        self,
        h: torch.Tensor,
        label: torch.Tensor = None,
        instance_eval: bool = False,
        return_features: bool = False,
        attention_only: bool = False,
    ):
        """
        Args:
            h: [N, embed_dim]
        Returns:
            logits: [1, n_classes]
            Y_prob: [1, n_classes]
            Y_hat: [1]
            A_raw: [K, N] raw attention scores (before softmax)
            results_dict: dict with optional instance-level outputs / features
        """
        device = h.device

        A, h = self.attention_net(h)   # A: [N, K], h: [N, H]
        A = A.transpose(1, 0)          # [K, N]

        if attention_only:
            return A

        A_raw = A
        A = F.softmax(A, dim=1)        # [K, N]

        total_inst_loss = torch.tensor(0.0, device=device)
        all_preds, all_targets = [], []

        if instance_eval and label is not None:
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze(0)  # [K]

            for i in range(self.n_classes):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]

                if inst_label == 1:
                    instance_loss, preds, targets = self.inst_eval(A[i], h, classifier)
                else:
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A[i], h, classifier)
                    else:
                        continue

                total_inst_loss = total_inst_loss + instance_loss
                all_preds.extend(preds.detach().cpu().numpy())
                all_targets.extend(targets.detach().cpu().numpy())

            if self.subtyping and len(self.instance_classifiers) > 0:
                total_inst_loss = total_inst_loss / len(self.instance_classifiers)

        # Bag-level logits (per-class branch)
        M = torch.mm(A, h)  # [K, H]
        logits = torch.empty(1, self.n_classes, device=device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])

        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)

        results_dict = {}
        if instance_eval:
            results_dict["instance_loss"] = total_inst_loss
            results_dict["inst_labels"] = np.array(all_targets)
            results_dict["inst_preds"] = np.array(all_preds)

        if return_features:
            results_dict["features"] = M

        return logits, Y_prob, Y_hat, A_raw, results_dict