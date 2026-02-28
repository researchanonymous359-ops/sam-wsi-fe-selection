# option.py
import argparse
from pathlib import Path

# ==========================================
# 1. Configuration Constants
# ==========================================

BASE_FEAT_DIMS = {
    "resnet50": 1024,
    "resnet50-tr-supervised-imagenet1k": 1024,
    "conch_v1": 512,
    "conch_v15": 768,
    "hibou_b": 768,
    "vit-ssl-dino-p16": 384,
    "musk": 1024,
    "phikon_v2": 1024,
    "uni_v1": 1024,
    "uni_v2": 1536,
    "virchow2": 1280,
}

FOLDER_NAME_ALIASES = {
    "vit-ssl-dino-p16": "lunit-vits16",
    "vit_ssl_dino_p16": "lunit-vits16",
}
REVERSE_FOLDER_ALIASES = {v: k for k, v in FOLDER_NAME_ALIASES.items()}


# ==========================================
# 2. Argument Parsing Functions
# ==========================================

def add_common_arguments(parser):
    """Define arguments commonly used across classification/grading tasks."""

    # --- [System & Hardware] ---
    group_sys = parser.add_argument_group("System & Hardware")
    group_sys.add_argument(
        "--seed",
        type=lambda s: [int(item) for item in s.split(",")],
        default=None,
        help="Random seeds (comma separated)",
    )
    group_sys.add_argument(
        "--gpu-id",
        type=lambda s: [int(item) for item in s.split(",")],
        default=None,
        help="GPU IDs to use",
    )
    group_sys.add_argument("--num-workers", type=int, default=2, help="Number of dataloader workers (default: 2)")
    group_sys.add_argument("--output-dir", type=str, default="results", help="Output directory (default: results)")
    group_sys.add_argument("--precision", type=int, default=32, help="Training precision (default: 32)")

    # --- [Dataset] ---
    group_data = parser.add_argument_group("Dataset")
    group_data.add_argument("--dataset-root", type=str, default="./", help="Root path of datasets (default: ./)")
    group_data.add_argument(
        "--train-dataset-name",
        type=str,
        nargs="+",
        default=["data1"],
        help="List of training dataset names (default: ['data1'])",
    )
    group_data.add_argument(
        "--test-dataset-name",
        type=str,
        nargs="+",
        default=["data1"],
        help="List of test dataset names (default: ['data1'])",
    )
    group_data.add_argument(
        "--feature-extractor",
        type=str,
        default="resnet50",
        help="Name of the feature extractor folder (default: resnet50)",
    )
    group_data.add_argument(
        "--resolution-list",
        type=str,
        nargs="+",
        default=["x5", "x10"],
        help="List of resolutions (default: ['x5','x10'])",
    )
    group_data.add_argument("--patch-size", type=int, default=256, help="Patch size used for extraction (default: 256)")
    group_data.add_argument(
        "--patch-path",
        type=str,
        default="patch_data/x5/256/test",
        help="Path to patch images (default: patch_data/x5/256/test)",
    )

    # --- [Task / Train Mode] ---
    group_task = parser.add_argument_group("Task")
    group_task.add_argument(
        "--train-mode",
        type=str,
        default="classification",
        choices=["classification", "grading"],
        help=(
            "Training objective (default: classification).\n"
            "- classification: standard multi-class CE\n"
            "- grading: ordinal-aware cost-sensitive CE (distance penalty)  ✅ auto-enabled"
        ),
    )

    # --- [Grading / Ordinal Loss Options] ---
    group_grade = parser.add_argument_group("Grading (Ordinal / Cost-sensitive CE)")

    group_grade.add_argument(
        "--grading-cost-type",
        type=str,
        default="sq",
        choices=["abs", "sq"],
        help="Distance type d(y,k): abs=|y-k|, sq=(y-k)^2 (default: sq)",
    )

    group_grade.add_argument(
        "--grading-cost-gamma",
        type=float,
        default=1.0,
        help="Exponent multiplier for distance (default: 1.0)",
    )

    group_grade.add_argument(
        "--grading-cost-lambda",
        type=float,
        default=0.5,
        help="Penalty weight alpha (default: 0.5). Set to 0 to recover standard CE.",
    )

    group_grade.add_argument(
        "--grading-cost-normalize",
        action="store_true",
        help="Normalize penalty so mean cost ≈ 1 (default: False)",
    )

    group_grade.add_argument(
        "--grading-cost-eps",
        type=float,
        default=1e-8,
        help="Epsilon for numerical stability (default: 1e-8)",
    )

    # --- [Model Structure] ---
    group_model = parser.add_argument_group("Model")
    group_model.add_argument(
        "--mil-model",
        type=str,
        default=None,
        choices=[
            "meanpooling",
            "maxpooling",
            "ABMIL",
            "GABMIL",
            "DSMIL",
            "CLAM-SB",
            "CLAM-MB",
            "TransMIL",
            "Transformer",
            "DTFD-MIL",
            "WiKG",
            "RRTMIL",
            "ILRA",
        ],
        help="MIL architecture to use",
    )
    group_model.add_argument("--attention", action="store_true", default=False, help="Enable attention map generation")
    group_model.add_argument(
        "--downsample",
        type=int,
        default=1,
        choices=[1, 2, 4, 8, 16],
        help="Downsample factor for attention maps (default: 1)",
    )

    # --- [Training Strategy] ---
    group_train = parser.add_argument_group("Training")
    group_train.add_argument("--epochs", type=int, default=200, help="Total number of epochs (default: 200)")
    group_train.add_argument("--batch-size", type=int, default=1, help="Batch size (default: 1)")
    group_train.add_argument("--accumulate-grad-batches", type=int, default=1, help="Gradient accumulation (default: 1)")
    group_train.add_argument("--patience", type=int, default=10, help="Early stopping patience (default: 10)")

    group_train.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    group_train.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay (default: 1e-2)")
    group_train.add_argument("--opt", type=str, default="adam", help="Optimizer (default: adam)")
    group_train.add_argument(
        "--loss-weight",
        type=lambda s: [float(item) for item in s.split(",")],
        default=None,
        help="Manual loss weights (classification only)",
    )
    group_train.add_argument("--auto-loss-weight", action="store_true", help="Automatically compute class weights")

    group_train.add_argument("--mil-patch-drop-min", type=float, default=0.0, help="Min patch drop ratio (default: 0.0)")
    group_train.add_argument("--mil-patch-drop-max", type=float, default=0.0, help="Max patch drop ratio (default: 0.0)")
    group_train.add_argument("--use-weighted-sampler", action="store_true", help="Use WeightedRandomSampler (default: False)")
    group_train.add_argument("--sampler-power", type=float, default=1.0, help="Power for inverse-frequency weighting (default: 1.0)")

    group_train.add_argument("--n_bins", type=int, default=30, help="Number of bins for ECE (default: 30)")


def add_dtfd_mil_arguments(parser):
    group = parser.add_argument_group("DTFD-MIL Specific")
    group.add_argument("--distill", type=str, default="MaxMinS", choices=["MaxMinS", "MaxS", "AFS"])
    group.add_argument("--total-instance", type=int, default=4)
    group.add_argument("--numGroup", type=int, default=4)
    group.add_argument("--grad-clipping", type=int, default=5)
    group.add_argument("--lr-decay-ratio", type=float, default=0.2)


def add_wikg_arguments(parser):
    group = parser.add_argument_group("WiKG Specific")
    group.add_argument("--topk", type=int, default=6)
    group.add_argument("--agg_type", type=str, default="bi-interaction", choices=["gcn", "sage", "bi-interaction"])
    group.add_argument("--dropout", type=float, default=0.3)
    group.add_argument("--pool", type=str, default="mean", choices=["mean", "max", "attn"])


def add_mamba_arguments(parser):
    group = parser.add_argument_group("Mamba")
    group.add_argument("--mambamil_dim", type=int, default=128)
    group.add_argument("--mambamil_rate", type=int, default=10)
    group.add_argument("--mambamil_state_dim", type=int, default=16)
    group.add_argument("--mambamil_layer", type=int, default=1)
    group.add_argument("--mambamil_inner_layernorms", default=False, action="store_true")
    group.add_argument(
        "--mambamil_type",
        type=str,
        default=None,
        choices=["Mamba", "SRMamba", "SimpleMamba"],
    )
    group.add_argument("--pscan", default=True)
    group.add_argument("--cuda_pscan", default=False, action="store_true")
    group.add_argument("--pos_emb_dropout", type=float, default=0.0)
    group.add_argument("--mamba_2d", default=False, action="store_true")
    group.add_argument("--mamba_2d_pad_token", "-p", type=str, default="trainable", choices=["zero", "trainable"])
    group.add_argument("--mamba_2d_patch_size", type=int, default=1)
    group.add_argument("--mamba_2d_pos_emb_type", default=None, choices=[None, "linear"])


# ==========================================
# 3. Post-Processing Logic (Classification/Grading only)
# ==========================================

def post_process_args(args):
    """
    Validate inputs and apply automatic configurations before running main.
    - Normalize feature-extractor folder name
    - Auto-compute num_feats
    - Sanity-check grading args (when train_mode == grading)
    - Auto-map grading_cost_* to grading_* for forward_fn_grading.py compatibility
    """
    args = _resolve_feature_extractor_path(args)
    args = _auto_set_num_feats(args)

    # For grading mode, map arguments before sanity checks so loss is auto-enabled.
    args = _auto_map_grading_args_for_forward_fn(args)
    args = _sanity_check_grading_args(args)

    return args


def _resolve_feature_extractor_path(args):
    dataset_list = (getattr(args, "train_dataset_name", []) or []) + (getattr(args, "test_dataset_name", []) or [])
    if not dataset_list:
        return args

    root = Path(args.dataset_root)
    base_ds = None
    for ds_name in dataset_list:
        if (root / ds_name).exists():
            base_ds = root / ds_name
            break
    if base_ds is None:
        return args

    existing_dirs = {d.name for d in base_ds.iterdir() if d.is_dir()}
    input_fe = args.feature_extractor

    if input_fe.lower() in FOLDER_NAME_ALIASES:
        alias = FOLDER_NAME_ALIASES[input_fe.lower()]
        if alias in existing_dirs:
            print(f"[Auto-Config] Feature Extractor Alias: '{input_fe}' -> '{alias}'")
            args.feature_extractor = alias
            return args

    if input_fe in existing_dirs:
        return args

    candidates = {
        input_fe,
        input_fe.lower(),
        input_fe.upper(),
        input_fe.replace("-", "_"),
        input_fe.replace("_", "-"),
    }
    for cand in candidates:
        if cand in existing_dirs:
            print(f"[Auto-Config] Feature Extractor Resolved: '{input_fe}' -> '{cand}'")
            args.feature_extractor = cand
            return args

    lower_map = {d.lower(): d for d in existing_dirs}
    if input_fe.lower() in lower_map:
        resolved = lower_map[input_fe.lower()]
        print(f"[Auto-Config] Feature Extractor Resolved (Case-insensitive): '{input_fe}' -> '{resolved}'")
        args.feature_extractor = resolved

    return args


def _auto_set_num_feats(args):
    current_fe = args.feature_extractor
    logical_fe = REVERSE_FOLDER_ALIASES.get(current_fe, current_fe)
    base_dim = BASE_FEAT_DIMS.get(logical_fe)

    if base_dim is None:
        for known_key, dim in BASE_FEAT_DIMS.items():
            if known_key in logical_fe or logical_fe.replace("_", "-") == known_key.replace("_", "-"):
                base_dim = dim
                break

    if base_dim is None:
        raise ValueError(
            f"[Error] Unknown feature extractor '{current_fe}' (logical: '{logical_fe}'). "
            f"Please add its dimension to BASE_FEAT_DIMS in option.py"
        )

    num_res = len(args.resolution_list)
    if num_res == 0:
        raise ValueError("resolution-list cannot be empty.")

    total_dim = base_dim * num_res
    args.num_feats = total_dim
    print(f"[Auto-Config] Input Dimension Set: {current_fe} ({base_dim}) x {num_res} resolutions = {total_dim}")
    return args


def _sanity_check_grading_args(args):
    if getattr(args, "train_mode", "classification") != "grading":
        return args

    lam = float(getattr(args, "grading_cost_lambda", 1.0))
    if lam < 0:
        raise ValueError("--grading-cost-lambda must be >= 0.")

    gamma = float(getattr(args, "grading_cost_gamma", 1.0))
    if gamma <= 0:
        raise ValueError("--grading-cost-gamma must be > 0.")

    cost_type = str(getattr(args, "grading_cost_type", "sq")).lower()
    if cost_type not in {"abs", "sq"}:
        raise ValueError("--grading-cost-type must be in {abs, sq}.")

    print(
        f"[Auto-Config][Grading] "
        f"cost_type={cost_type} gamma={gamma} lambda={lam} "
        f"normalize={bool(getattr(args, 'grading_cost_normalize', False))}"
    )
    return args


def _auto_map_grading_args_for_forward_fn(args):
    if getattr(args, "train_mode", "classification") != "grading":
        return args

    # Ensure grading loss is enabled by default when train_mode=grading
    args.grading_loss = "cost_sensitive_ce"
    args.grading_alpha = float(getattr(args, "grading_cost_lambda", 0.5))

    cost_type = str(getattr(args, "grading_cost_type", "sq")).lower()
    base_power = 1.0 if cost_type == "abs" else 2.0
    args.grading_power = base_power * float(getattr(args, "grading_cost_gamma", 1.0))

    print(f"[Auto-Config][Grading] alpha={args.grading_alpha}, power={args.grading_power}")
    return args


# Backward compatibility entrypoint name used by main.py
auto_adjust_for_camelyon = post_process_args