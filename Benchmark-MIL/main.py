# main.py
import argparse
from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from option import (
    add_common_arguments,
    add_dtfd_mil_arguments,
    add_wikg_arguments,
    add_mamba_arguments,
    auto_adjust_for_camelyon,
)

from utils import seed_everything, get_loss_weight
from dataset import get_class_names
from model import get_model_module

# classification / grading datamodule (same)
from dataset.classification_grading_dataset import CombinedPatchFeaturesWSIDataModule
# survival datamodule
from dataset.survival_analysis_dataset import SurvivalWSIDataModule

# callbacks
from callbacks.analysis_callback import SaveAnalysisResultsCallback
from callbacks.analysis_callback_survival import SaveSurvivalAnalysisResultsCallback
from callbacks.analysis_callback_grading import SaveGradingAnalysisResultsCallback

from save_metrics import (
    initialize_metrics,
    make_single_result_metrics,
    make_whole_result_metrics,
)
from save_metrics_survival import (
    initialize_survival_metrics,
    make_single_result_survival_metrics,
    make_whole_result_survival_metrics,
)
from save_metrics_grading import (
    initialize_grading_metrics,
    make_single_result_grading_metrics,
    make_whole_result_grading_metrics,
)

import warnings
warnings.filterwarnings("ignore")


def is_bracs_dataset(dataset_names):
    if dataset_names is None:
        return False
    if isinstance(dataset_names, (str, Path)):
        dataset_names = [dataset_names]
    return any("bracs" in str(d).lower() for d in dataset_names)


def make_experiment_base_dir(args):
    resolution_str = "_".join(args.resolution_list)
    dataset_name_str = (
        "_".join(args.train_dataset_name)
        if isinstance(args.train_dataset_name, list)
        else args.train_dataset_name
    )

    mil_name = (
        f"{args.mil_model}-{args.distill}"
        if getattr(args, "distill", None) is not None and args.mil_model == "DTFD-MIL"
        else args.mil_model
    )

    parts = [
        args.output_dir,
        dataset_name_str,
        resolution_str,
        str(args.patch_size),
        mil_name,
        args.feature_extractor,
        args.train_mode,
    ]
    if args.train_mode == "survival":
        parts.append(str(args.survival_endpoint))
        parts.append(f"bins_{int(args.survival_num_bins)}")

    base_dir = Path(*parts)
    base_dir.mkdir(parents=True, exist_ok=True)
    print(f"[PATH] Base experiment dir: {base_dir}")
    return base_dir


def get_monitor_cfg(args):
    # -------------------------
    # ✅ grading: 무조건 QWK로 monitor
    # -------------------------
    if args.train_mode == "grading":
        # ✅ EarlyStopping / ModelCheckpoint 둘 다 이 metric이 "반드시" 로그로 존재해야 함
        # trainer가 "QWK/val" + "val_qwk" 둘 다 로깅하도록 grading trainer에서 보장한다(아래 파일 참고)
        return "QWK/val", "max", "best-{epoch:02d}-{val_qwk:.4f}"

    # -------------------------
    # classification
    # -------------------------
    if args.train_mode == "classification":
        if getattr(args, "use_weighted_sampler", False):
            return "ACC_balanced/val", "max", "best-{epoch:02d}-{val_bacc:.2f}"
        return "Loss/val", "min", "best-{epoch:02d}-{val_loss:.4f}"

    # -------------------------
    # survival
    # -------------------------
    monitor = getattr(args, "monitor_survival_metric", "cindex").lower()
    if monitor == "cindex":
        return "CIndex/val", "max", "best-{epoch:02d}-{val_cindex:.4f}"
    return "Loss/val", "min", "best-{epoch:02d}-{val_loss:.4f}"


def _ensure_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _ensure_default_args(args):
    if getattr(args, "seed", None) is None:
        args.seed = [0]

    if getattr(args, "gpu_id", None) is None:
        args.gpu_id = [0]
    elif isinstance(args.gpu_id, int):
        args.gpu_id = [args.gpu_id]

    args.task = args.train_mode

    if args.train_mode == "survival":
        if not hasattr(args, "survival_num_bins"):
            if hasattr(args, "survival_n_bins"):
                args.survival_num_bins = getattr(args, "survival_n_bins")
            else:
                args.survival_num_bins = 20

    if not hasattr(args, "accumulate_grad_batches"):
        args.accumulate_grad_batches = 1

    return args


def _print_grading_cfg(args):
    if args.train_mode != "grading":
        return
    print(
        "[Grading Config] "
        f"grading_loss={getattr(args, 'grading_loss', None)} | "
        f"grading_alpha={getattr(args, 'grading_alpha', None)} | "
        f"grading_power={getattr(args, 'grading_power', None)} | "
        f"(src) cost_type={getattr(args, 'grading_cost_type', None)} "
        f"lambda={getattr(args, 'grading_cost_lambda', None)} gamma={getattr(args, 'grading_cost_gamma', None)} "
        f"normalize={bool(getattr(args, 'grading_cost_normalize', False))}"
    )


def main(args):
    torch.set_float32_matmul_precision("medium")
    args = _ensure_default_args(args)

    _print_grading_cfg(args)

    # ------------------------------
    # 0) 공통 경로 / patch_path
    # ------------------------------
    exp_base_dir = make_experiment_base_dir(args)
    args.base_save_dir = str(exp_base_dir)

    if is_bracs_dataset(args.train_dataset_name):
        base_dataset_for_patch = args.train_dataset_name[0]
    elif is_bracs_dataset(args.test_dataset_name):
        base_dataset_for_patch = args.test_dataset_name[0]
    else:
        base_dataset_for_patch = args.test_dataset_name[0]

    args.patch_path = f"{args.dataset_root}/{base_dataset_for_patch}/{args.patch_path}"

    # ------------------------------
    # 1) test_dataset_info + metrics init (mode별)
    # ------------------------------
    test_dataset_info = {}
    survival_K = int(getattr(args, "survival_num_bins", 0)) if args.train_mode == "survival" else None

    for tde in args.test_dataset_name:
        if args.train_mode in {"classification", "grading"}:
            names = get_class_names(tde)
            num_classes = len(names)
        else:
            names = [f"bin_{i}" for i in range(survival_K)]
            num_classes = survival_K

        test_dataset_info[tde] = {
            "test_class_names_list": names,
            "test_num_classes": num_classes,
            "test_dataset_element": tde,
        }

    if args.train_mode == "classification":
        initialize_metrics(test_dataset_info)
        all_seed_logits_dict = {k: [] for k in test_dataset_info}
        all_seed_labels_dict = {k: [] for k in test_dataset_info}
        all_seed_names_dict = {k: [] for k in test_dataset_info}

    elif args.train_mode == "grading":
        initialize_grading_metrics(test_dataset_info)
        all_seed_logits_dict = {k: [] for k in test_dataset_info}
        all_seed_labels_dict = {k: [] for k in test_dataset_info}
        all_seed_names_dict = {k: [] for k in test_dataset_info}

    else:
        initialize_survival_metrics(test_dataset_info)
        all_seed_logits_dict = None
        all_seed_labels_dict = None
        all_seed_names_dict = None

    # ------------------------------
    # 2) seed loop
    # ------------------------------
    for seed in args.seed:
        seed_everything(seed)
        args.resolution_str = "_".join(args.resolution_list)

        # --------------------------------------
        # Train DataModule (mode별)
        # --------------------------------------
        if args.train_mode == "survival":
            train_ds_name = (
                args.train_dataset_name[0]
                if isinstance(args.train_dataset_name, list)
                else args.train_dataset_name
            )

            train_dm = SurvivalWSIDataModule(
                dataset_root=args.dataset_root,
                dataset_name=train_ds_name,
                feature_extractor=args.feature_extractor,
                resolutions=args.resolution_list,
                patch_size=args.patch_size,
                num_workers=args.num_workers,
                survival_endpoint=args.survival_endpoint,
                survival_event_key=args.survival_event_key,
                survival_time_key=args.survival_time_key,
                drop_no_survival=True,
            )
            train_dm.setup()

            loss_weight = None
            combined_train_name = str(train_ds_name)

        else:
            train_dm = CombinedPatchFeaturesWSIDataModule(
                dataset_root=args.dataset_root,
                dataset_mode="train",
                train_dataset_name=args.train_dataset_name,
                resolutions=args.resolution_list,
                patch_size=args.patch_size,
                feature_extractor=args.feature_extractor,
                num_workers=args.num_workers,
            )
            train_dm.use_weighted_sampler = getattr(args, "use_weighted_sampler", False)
            train_dm.sampler_power = getattr(args, "sampler_power", 1.0)
            train_dm.setup()

            loss_weight = get_loss_weight(args, train_dm)
            combined_train_name = "_".join(train_dm.train_dataset_name)

        # --------------------------------------
        # Test DataModules (mode별)
        # --------------------------------------
        test_dm_dict = {}
        for tde in sorted(test_dataset_info):
            if args.train_mode == "survival":
                dm = SurvivalWSIDataModule(
                    dataset_root=args.dataset_root,
                    dataset_name=tde,
                    feature_extractor=args.feature_extractor,
                    resolutions=args.resolution_list,
                    patch_size=args.patch_size,
                    num_workers=args.num_workers,
                    survival_endpoint=args.survival_endpoint,
                    survival_event_key=args.survival_event_key,
                    survival_time_key=args.survival_time_key,
                    drop_no_survival=True,
                )
                dm.setup()
            else:
                dm = CombinedPatchFeaturesWSIDataModule(
                    dataset_root=args.dataset_root,
                    dataset_mode="test",
                    train_dataset_name=[tde],
                    resolutions=args.resolution_list,
                    patch_size=args.patch_size,
                    feature_extractor=args.feature_extractor,
                    num_workers=args.num_workers,
                )
                dm.setup()

            test_dm_dict[tde] = dm

        # --------------------------------------
        # Model init
        # --------------------------------------
        if args.train_mode in {"classification", "grading"}:
            train_class_names = get_class_names(args.train_dataset_name[0])
            train_num_classes = len(train_class_names)
        else:
            K = int(args.survival_num_bins)
            train_class_names = [f"bin_{i}" for i in range(K)]
            train_num_classes = K

        model = get_model_module(
            args=args,
            seed=seed,
            test_dataset_element_name=combined_train_name,
            resolution_str=args.resolution_str,
            mil_model=args.mil_model,
            num_feats=args.num_feats,
            test_class_names_list=train_class_names,
            num_classes=train_num_classes,
            loss_weight=loss_weight,
            get_attention=args.attention,
            patch_path=args.patch_path,
        )

        # ---- train save dir (mode별)
        if args.train_mode == "classification":
            train_subdir = "ce"
        elif args.train_mode == "grading":
            train_subdir = "grading"
        else:
            train_subdir = "survival"

        save_seed_dir = exp_base_dir / "train" / train_subdir / f"seed_{seed}"
        save_seed_dir.mkdir(parents=True, exist_ok=True)
        args.save_seed_dir = str(save_seed_dir)

        model.exp_base_dir = str(exp_base_dir)
        model.seed = seed

        # ---- callbacks/logger/trainer
        monitor_metric, monitor_mode, filename_format = get_monitor_cfg(args)

        checkpoint_cb = ModelCheckpoint(
            monitor=monitor_metric,
            dirpath=save_seed_dir,
            filename=filename_format,
            save_top_k=1,
            verbose=True,
            mode=monitor_mode,
        )
        earlystop_cb = EarlyStopping(
            monitor=monitor_metric,
            min_delta=0.00,
            patience=args.patience,
            verbose=True,
            mode=monitor_mode,
        )

        if args.train_mode == "classification":
            analysis_cb = SaveAnalysisResultsCallback()
        elif args.train_mode == "grading":
            analysis_cb = SaveGradingAnalysisResultsCallback()
        else:
            analysis_cb = SaveSurvivalAnalysisResultsCallback()

        logger = CSVLogger(save_seed_dir)
        gpu_ids = _ensure_list(args.gpu_id)

        trainer = pl.Trainer(
            default_root_dir=save_seed_dir,
            max_epochs=args.epochs,
            log_every_n_steps=50,
            num_sanity_val_steps=0,
            precision=args.precision,
            accelerator="gpu",
            devices=gpu_ids,
            logger=logger,
            callbacks=[checkpoint_cb, earlystop_cb, analysis_cb],
            strategy="ddp_find_unused_parameters_true" if len(gpu_ids) > 1 else "auto",
            accumulate_grad_batches=int(getattr(args, "accumulate_grad_batches", 1)),
        )

        # ---- fit
        trainer.fit(model, train_dm)

        # ---- test per dataset (best ckpt)
        for tde, test_dm in test_dm_dict.items():
            model.test_dataset_element_name = tde
            model.test_class_names_list = test_dataset_info[tde]["test_class_names_list"]

            print(f"\n[Test] Testing on {tde} ...")
            test_results = trainer.test(model, test_dm, ckpt_path="best")

            if args.train_mode in {"classification", "grading"}:
                probs = torch.cat(model.y_prob_list, dim=0).detach().cpu()
                labels = torch.cat(model.label_list, dim=0).detach().cpu()
                names = [n[0] if isinstance(n, (tuple, list)) else n for n in model.names]

                all_seed_logits_dict[tde].append(probs)
                all_seed_labels_dict[tde].append(labels)
                all_seed_names_dict[tde].append(names)

                if args.train_mode == "classification":
                    make_single_result_metrics(
                        args=args,
                        seed=seed,
                        trainer_model=model,
                        test_results=test_results,
                        test_dataset_element_name=tde,
                        num_classes=test_dataset_info[tde]["test_num_classes"],
                    )
                else:
                    make_single_result_grading_metrics(
                        args=args,
                        seed=seed,
                        trainer_model=model,
                        test_results=test_results,
                        test_dataset_element_name=tde,
                        num_classes=test_dataset_info[tde]["test_num_classes"],
                    )

            else:
                make_single_result_survival_metrics(
                    args=args,
                    seed=seed,
                    trainer_model=model,
                    test_results=test_results,
                    test_dataset_element_name=tde,
                )

    # ------------------------------
    # 3) ensemble (mode별)
    # ------------------------------
    for tde in test_dataset_info:
        if args.train_mode in {"classification", "grading"}:
            if len(all_seed_logits_dict[tde]) == 0:
                print(f"[{tde}] No predictions collected. Skipping ensemble.")
                continue

            mean_probs = torch.mean(torch.stack(all_seed_logits_dict[tde]), dim=0)
            final_labels = all_seed_labels_dict[tde][0]
            slide_names = all_seed_names_dict[tde][0]

            if args.train_mode == "classification":
                make_whole_result_metrics(
                    args=args,
                    test_dataset_element_name=tde,
                    num_classes=test_dataset_info[tde]["test_num_classes"],
                    class_names_list=test_dataset_info[tde]["test_class_names_list"],
                    save_dir=exp_base_dir,
                    mean_logits=mean_probs,
                    final_labels=final_labels,
                    slide_names=slide_names,
                )
            else:
                make_whole_result_grading_metrics(
                    args=args,
                    test_dataset_element_name=tde,
                    num_classes=test_dataset_info[tde]["test_num_classes"],
                    class_names_list=test_dataset_info[tde]["test_class_names_list"],
                    save_dir=exp_base_dir,
                    mean_logits=mean_probs,
                    final_labels=final_labels,
                    slide_names=slide_names,
                )

        else:
            make_whole_result_survival_metrics(
                args=args,
                test_dataset_element_name=tde,
                save_dir=exp_base_dir,
            )

    print("All seeds processing finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)

    temp_args, _ = parser.parse_known_args()

    if temp_args.mil_model == "DTFD-MIL":
        add_dtfd_mil_arguments(parser)
    elif temp_args.mil_model == "WiKG":
        add_wikg_arguments(parser)

    add_mamba_arguments(parser)

    args = parser.parse_args()
    args = auto_adjust_for_camelyon(args)

    # patch drop sanity
    if args.mil_patch_drop_min < 0.0 or args.mil_patch_drop_max < 0.0:
        raise ValueError("mil_patch_drop_min / mil_patch_drop_max must be >= 0.0")
    if args.mil_patch_drop_max < args.mil_patch_drop_min:
        raise ValueError("mil_patch_drop_max must be >= mil_patch_drop_min")
    if args.mil_patch_drop_max > 1.0:
        raise ValueError("mil_patch_drop_max must be <= 1.0")

    main(args)
