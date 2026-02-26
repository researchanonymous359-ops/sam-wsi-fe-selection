import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def initialize_weights(module):
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
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super().__init__()
        layers = [
            nn.Linear(L, D),
            nn.Tanh(),
        ]
        if dropout:
            layers.append(nn.Dropout(0.25))
        layers.append(nn.Linear(D, n_classes))
        self.module = nn.Sequential(*layers)

    def forward(self, x):
        # x: [N, L]
        # return: [N, n_classes], x
        return self.module(x), x


# ===============================
# Attention Network (gated)
# ===============================
class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
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

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a * b           # [N, D]
        A = self.attention_c(A)  # [N, n_classes]
        return A, x


# ===========================
# CLAM-SB (Single Branch)
# ===========================
class CLAM_SB(nn.Module):
    def __init__(
        self,
        gate=True,
        size_arg="small",
        dropout=False,
        k_sample=8,
        n_classes=2,
        instance_loss_fn="ce",   # 🔥 SVM 제거, CE만 사용
        subtyping=False,
        embed_dim=1024,
        bag_weight=0.7,
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

        # Feature → attention input
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

        # Instance-level classifier (positive / negative)
        inst_cls = [nn.Linear(size[1], 2) for _ in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(inst_cls)

        # 🔥 Instance loss: CrossEntropy로 통일
        self.instance_loss_fn = nn.CrossEntropyLoss()

        initialize_weights(self)

    @staticmethod
    def create_positive_targets(k, device):
        return torch.full((k,), 1, device=device, dtype=torch.long)

    @staticmethod
    def create_negative_targets(k, device):
        return torch.full((k,), 0, device=device, dtype=torch.long)

    # In-class instance-level evaluation
    def inst_eval(self, A, h, classifier):
        """
        A: [N] or [1, N]
        h: [N, D]
        """
        device = h.device

        if A.dim() == 1:
            A = A.view(1, -1)  # [1, N]

        # top-k positive / negative
        top_p_ids = torch.topk(A, self.k_sample, dim=1)[1][-1]   # [k]
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]  # [k]

        top_p = h.index_select(0, top_p_ids)       # [k, D]
        top_n = h.index_select(0, top_n_ids)       # [k, D]

        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_instances = torch.cat([top_p, top_n], dim=0)          # [2k, D]
        all_targets = torch.cat([p_targets, n_targets], dim=0)    # [2k]

        logits = classifier(all_instances)  # [2k, 2]
        instance_loss = self.instance_loss_fn(logits, all_targets)
        preds = torch.argmax(logits, dim=1)

        return instance_loss, preds, all_targets

    # Out-of-class instance-level evaluation
    def inst_eval_out(self, A, h, classifier):
        device = h.device
        if A.dim() == 1:
            A = A.view(1, -1)

        top_p_ids = torch.topk(A, self.k_sample, dim=1)[1][-1]
        top_p = h.index_select(0, top_p_ids)  # [k, D]

        targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)           # [k, 2]
        instance_loss = self.instance_loss_fn(logits, targets)
        preds = torch.argmax(logits, dim=1)

        return instance_loss, preds, targets

    def forward(
        self,
        h,
        label=None,
        instance_eval=False,
        return_features=False,
        attention_only=False,
    ):
        """
        h: [N, embed_dim]
        label: scalar long tensor (bag label)
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
            # label: scalar → one-hot: [n_classes]
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze(0)

            for i in range(self.n_classes):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]

                if inst_label == 1:
                    # in-class branch
                    instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                else:
                    # out-of-class branch (optional, only if subtyping)
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
        M = torch.mm(A, h)           # [1, H]
        logits = self.classifiers(M) # [1, n_classes]
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
        gate=True,
        size_arg="small",
        dropout=False,
        k_sample=8,
        n_classes=2,
        instance_loss_fn="ce",   # 🔥 동일하게 CE만 사용
        subtyping=False,
        embed_dim=1024,
        bag_weight=0.7,
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

        bag_classifiers = [nn.Linear(size[1], 1) for _ in range(n_classes)]
        self.classifiers = nn.ModuleList(bag_classifiers)

        inst_cls = [nn.Linear(size[1], 2) for _ in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(inst_cls)

        # 🔥 마찬가지로 CE만 사용
        self.instance_loss_fn = nn.CrossEntropyLoss()

        initialize_weights(self)

    def forward(
        self,
        h,
        label=None,
        instance_eval=False,
        return_features=False,
        attention_only=False,
    ):
        """
        h: [N, embed_dim]
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
        M = torch.mm(A, h)   # [K, H]
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
