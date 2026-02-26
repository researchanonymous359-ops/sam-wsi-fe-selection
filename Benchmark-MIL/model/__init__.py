# model/__init__.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import switch_dim

# ---- Classification Trainers (existing) ----
from pl_model.mil_trainer import MILTrainerModule
from pl_model.mil_trainer_dtfdmil import DTFDTrainerModule

# ---- Survival Trainers (Discrete-time ONLY) ----
from pl_model.mil_trainer_survival import MILSurvivalTrainerModule
from pl_model.mil_trainer_dtfdmil_survival import DTFDSurvivalTrainerModule

# ---- Grading Trainers (NEW: 분리) ----
from pl_model.mil_trainer_grading import MILGradingTrainerModule
from pl_model.mil_trainer_dtfdmil_grading import DTFDGradingTrainerModule


# --------------------------------------------------------
# 0. Task / Mode helpers
# --------------------------------------------------------
def _get_train_mode(args) -> str:
    """
    args.train_mode / args.task 를 함께 고려해서
    classification / grading / survival 중 하나로 정규화.
    """
    train_mode = str(getattr(args, "train_mode", "classification")).lower().strip()

    # aliases
    if train_mode in ["survival", "time_to_event", "time-to-event", "tte"]:
        return "survival"
    if train_mode in ["grading", "grade", "ordinal", "isup", "qwk"]:
        return "grading"
    if train_mode in ["classification", "cls", "ce"]:
        return "classification"

    # fallback: task 값으로 추정
    task = str(getattr(args, "task", "classification")).lower().strip()
    if task in ["survival", "time_to_event", "time-to-event", "tte"]:
        return "survival"
    if task in ["grading", "grade", "ordinal", "isup", "qwk"]:
        return "grading"
    return "classification"


def _is_survival_mode(args) -> bool:
    return _get_train_mode(args) == "survival"


def _is_grading_mode(args) -> bool:
    return _get_train_mode(args) == "grading"


def _get_survival_num_bins(args) -> int:
    if hasattr(args, "survival_num_bins"):
        return int(getattr(args, "survival_num_bins"))
    if hasattr(args, "survival_n_bins"):
        return int(getattr(args, "survival_n_bins"))
    return 20


# --------------------------------------------------------
# 1. Pooling Layers & Simple Models
# --------------------------------------------------------
class LinearClassifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)


class MeanPooling(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MeanPooling, self).__init__()
        self.classifier = LinearClassifier(in_dim, out_dim)

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(0)
        h = torch.mean(x, dim=0, keepdim=True)
        return self.classifier(h)


class MaxPooling(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MaxPooling, self).__init__()
        self.classifier = LinearClassifier(in_dim, out_dim)

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(0)
        h, _ = torch.max(x, dim=0, keepdim=True)
        return self.classifier(h)


# --------------------------------------------------------
# 2. Get MIL Model Factory Function
# --------------------------------------------------------
def get_mil_model(args, num_classes, loss_weight=None, num_feats: int | None = None):
    """
    ✅ discrete-time survival only
    - Classification/Grading: out_dim = num_classes, loss = CE
      (grading_loss는 MILGradingTrainerModule 내부에서 args로 분기 처리)
    - Survival: out_dim = survival_num_bins(K), loss = None (trainer에서 hazard loss)
    """
    model_name = args.mil_model
    in_dim = num_feats if num_feats is not None else getattr(args, "num_feats")

    train_mode = _get_train_mode(args)
    is_survival = (train_mode == "survival")

    if is_survival:
        out_dim = _get_survival_num_bins(args)  # ✅ K
        loss_func = None
    else:
        out_dim = num_classes
        loss_func = nn.CrossEntropyLoss(weight=loss_weight)

    # --- Simple Pooling ---
    if model_name == "meanpooling":
        return MeanPooling(in_dim, out_dim), loss_func

    elif model_name == "maxpooling":
        return MaxPooling(in_dim, out_dim), loss_func

    # --- ABMIL ---
    elif model_name == "ABMIL":
        from model.abmil import ABMILClassifier, AttentionPooling
        pooling = AttentionPooling(feature_dim=in_dim, mid_dim=128, out_dim=1, flatten=True)
        model = ABMILClassifier(pooling, num_feats=in_dim, num_classes=out_dim)
        return model, loss_func

    # --- GABMIL ---
    elif model_name == "GABMIL":
        from model.gabmil import GABMILClassifier, GatedAttentionPooling
        pooling = GatedAttentionPooling(feature_dim=in_dim, mid_dim=128, out_dim=1, flatten=True)
        model = GABMILClassifier(pooling, num_feats=in_dim, num_classes=out_dim)
        return model, loss_func

    # --- DSMIL ---
    elif model_name == "DSMIL":
        from model.dsmil import MILNet, IClassifier, BClassifier
        i_classifier = IClassifier(
            feature_extractor=nn.Identity(),
            feature_size=in_dim,
            output_class=out_dim,
        )
        b_classifier = BClassifier(input_size=in_dim, output_class=out_dim)
        model = MILNet(i_classifier, b_classifier)
        return model, loss_func

    # --- CLAM ---
    elif model_name in ["CLAM-SB", "CLAM-MB"]:
        if is_survival:
            raise NotImplementedError("CLAM survival은 별도 구현 필요 (instance_loss 구조가 CE 기반).")
        from model.clam import CLAM_SB, CLAM_MB
        CLAM = CLAM_SB if model_name == "CLAM-SB" else CLAM_MB
        model = CLAM(
            gate=True,
            size_arg="small",
            dropout=True,
            k_sample=8,
            n_classes=out_dim,
            instance_loss_fn="svm",
            subtyping=True,
            embed_dim=in_dim,
        )
        return model, loss_func

    # --- TransMIL ---
    elif model_name == "TransMIL":
        from model.transmil import TransMIL
        model = TransMIL(n_classes=out_dim, input_size=in_dim)
        return model, loss_func

    # --- Transformer ---
    elif model_name == "Transformer":
        from model.transformer import TransformerMIL
        model = TransformerMIL(n_classes=out_dim, input_size=in_dim, heads=8, dropout=0.1)
        return model, loss_func

    # --- RRTMIL ---
    elif model_name == "RRTMIL":
        from model.rrt import RRTMIL
        model = RRTMIL(in_dim=in_dim, num_classes=out_dim)
        return model, loss_func

    # --- ILRA ---
    elif model_name == "ILRA":
        from model.ilra import ILRA
        model = ILRA(
            n_classes=out_dim,
            input_size=in_dim,
            embed_dim=256,
            num_heads=8,
            num_layers=2,
            topk=64,
            ln=True,
        )
        return model, loss_func

    # --- WiKG ---
    elif model_name == "WiKG":
        from model.wikg import WiKG
        model = WiKG(
            dim_in=in_dim,
            dim_hidden=512,
            topk=args.topk,
            n_classes=out_dim,
            agg_type=args.agg_type,
            dropout=args.dropout,
            pool=args.pool,
        )
        return model, loss_func

    # --- DTFD-MIL ---
    elif model_name == "DTFD-MIL":
        from model.dtfdmil.network import DimReduction, Classifier_1fc
        from model.dtfdmil.attention import Attention_with_Classifier

        mDim = 512
        dimReduction = DimReduction(n_channels=in_dim, m_dim=mDim, numLayer_Res=0)

        attention = Attention_with_Classifier(L=mDim, D=128, K=1, num_cls=out_dim, droprate=0)
        classifier = Classifier_1fc(n_channels=mDim, n_classes=out_dim)
        UClassifier = Classifier_1fc(n_channels=mDim, n_classes=out_dim)

        model_list = [classifier, attention, dimReduction, UClassifier]

        if is_survival:
            return model_list, None

        loss_func0 = nn.CrossEntropyLoss(weight=loss_weight)
        loss_func1 = nn.CrossEntropyLoss(weight=loss_weight)
        return model_list, [loss_func0, loss_func1]

    else:
        raise NotImplementedError(f"Model {model_name} not implemented")


def get_metrics(num_classes):
    from torchmetrics import MetricCollection
    from torchmetrics.classification import (
        MulticlassAccuracy,
        MulticlassPrecision,
        MulticlassRecall,
        MulticlassF1Score,
    )

    return MetricCollection(
        {
            "acc": MulticlassAccuracy(num_classes=num_classes, average="micro"),
            "f1_macro": MulticlassF1Score(num_classes=num_classes, average="macro"),
            "pre_macro": MulticlassPrecision(num_classes=num_classes, average="macro"),
            "rec_macro": MulticlassRecall(num_classes=num_classes, average="macro"),
        }
    )


# --------------------------------------------------------
# 3. Get Lightning Module (Trainer)
# --------------------------------------------------------
def get_model_module(
    args,
    seed,
    test_dataset_element_name,
    resolution_str,
    mil_model,
    num_feats,
    test_class_names_list,
    num_classes,
    loss_weight=None,
    get_attention=False,
    patch_path=None,
):
    """
    ✅ 모델 + Trainer 결합
    - classification / grading / survival(discrete-time) 3-way 강제 분기
    - grading은 반드시 grading forward_fn을 사용 (cost-sensitive CE 적용 가능)
    """
    train_mode = _get_train_mode(args)
    is_survival = (train_mode == "survival")
    is_grading = (train_mode == "grading")

    # 1) 모델 생성
    model, loss_func = get_mil_model(args, num_classes, loss_weight, num_feats=num_feats)

    # =========================================
    # 2) SURVIVAL path
    # =========================================
    if is_survival:
        if mil_model == "DTFD-MIL":
            return DTFDSurvivalTrainerModule(
                args=args,
                seed=seed,
                test_dataset_element_name=test_dataset_element_name,
                resolution_str=resolution_str,
                classifier_list=model,
                loss_func_list=loss_func,  # None
                patch_path=patch_path,
            )

        return MILSurvivalTrainerModule(
            args=args,
            seed=seed,
            test_dataset_element_name=test_dataset_element_name,
            resolution_str=resolution_str,
            classifier=model,
            patch_path=patch_path,
        )

    # =========================================
    # 3) GRADING path  ✅ (USE forward_fn_grading)
    # =========================================
    if is_grading:
        if mil_model == "DTFD-MIL":
            return DTFDGradingTrainerModule(
                args=args,
                seed=seed,
                test_dataset_element_name=test_dataset_element_name,
                test_class_names_list=test_class_names_list,
                resolution_str=resolution_str,
                classifier_list=model,
                loss_func_list=loss_func,
                metrics=get_metrics(num_classes),
                patch_path=patch_path,
            )

        # ✅ grading 전용 forward/attention 함수 사용
        from pl_model.forward_fn_grading import (
            get_forward_func_grading,
            get_attention_func_grading,
        )

        forward_func = get_forward_func_grading(mil_model)

        attention_func = None
        if get_attention:
            attention_func = get_attention_func_grading(mil_model)

        return MILGradingTrainerModule(
            args=args,
            seed=seed,
            test_dataset_element_name=test_dataset_element_name,
            test_class_names_list=test_class_names_list,
            num_classes=num_classes,
            resolution_str=resolution_str,
            classifier=model,
            loss_func=loss_func,  # ✅ base CE loss func (필요시 fallback용). grading loss는 forward_fn_grading에서 처리
            metrics=get_metrics(num_classes),
            forward_func=forward_func,
            attention_func=attention_func,
            patch_path=patch_path,
        )

    # =========================================
    # 4) CLASSIFICATION path
    # =========================================
    from pl_model.forward_fn import get_forward_func, get_attention_func

    forward_func = get_forward_func(mil_model)
    attention_func = None
    if get_attention and mil_model != "DTFD-MIL":
        attention_func = get_attention_func(mil_model)

    if mil_model == "DTFD-MIL":
        return DTFDTrainerModule(
            args=args,
            seed=seed,
            test_dataset_element_name=test_dataset_element_name,
            test_class_names_list=test_class_names_list,
            resolution_str=resolution_str,
            classifier_list=model,
            loss_func_list=loss_func,
            metrics=get_metrics(num_classes),
            patch_path=patch_path,
        )

    return MILTrainerModule(
        args=args,
        seed=seed,
        test_dataset_element_name=test_dataset_element_name,
        test_class_names_list=test_class_names_list,
        num_classes=num_classes,
        resolution_str=resolution_str,
        classifier=model,
        loss_func=loss_func,
        metrics=get_metrics(num_classes),
        forward_func=forward_func,
        attention_func=attention_func,
        patch_path=patch_path,
    )
