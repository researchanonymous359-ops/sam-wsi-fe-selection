# dataset/classification_grading_dataset.py
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
import pytorch_lightning as pl
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataset import get_class_names
import h5py
import pickle

# ===============================
# 이름 정규화 & 공용 유틸
# ===============================
def normalize_dataset_name(name: str) -> str:
    """
    다양한 별칭/소문자 입력을 정규화
    """
    key = name.strip().lower().replace("-", "_")
    alias = {
        "camelyon": "camelyon16",
        "camelyon_16": "camelyon16",
        "3d_certain": "3D_certain",
        "3dcertain": "3D_certain",
        "leica_certain": "Leica_certain",
        "leicacertain": "Leica_certain",
        "bracs": "bracs",
        "bracs_wsi": "bracs",

        # 🔥 TCGA 계열
        "tcga_nsclc": "TCGA-NSCLC",
        "tcga_rcc": "TCGA-RCC",
        "tcga_glioma": "TCGA-GLIOMA",

        # 🔥 HISAI SKIN 계열
        "histai_skin_b1": "histai-skin-b1",
        "histai_skin": "histai-skin-b1",
        "histai_skin1": "histai-skin-b1",
        "histai_skin_b": "histai-skin-b1",
        "histai_skin_v1": "histai-skin-b1",
        "histai_skin_v01": "histai-skin-b1",

        # ✅ PANDA
        "panda": "panda",
        "panda_wsi": "panda",
        "prostate_panda": "panda",

        # ✅ UBC-OCEAN
        "ubc_ocean": "UBC-OCEAN",
        "ubc-ocean": "UBC-OCEAN",
        "ocean": "UBC-OCEAN",
        "ubcocean": "UBC-OCEAN",

        # ✅ CPTAC (grading)
        "cptac_ccrcc": "cptac_ccrcc",
        "cptac_ccrcc_wsi": "cptac_ccrcc",
        "cptac_hnscc": "cptac_hnscc",
        "cptac_hnscc_wsi": "cptac_hnscc",
        "cptac_ucec": "cptac_ucec",
        "cptac_ucec_wsi": "cptac_ucec",
    }

    if name in {
        "camelyon16",
        "3D_certain",
        "Leica_certain",
        "bracs",
        "TCGA-NSCLC",
        "TCGA-RCC",
        "TCGA-GLIOMA",
        "histai-skin-b1",
        "panda",
        "UBC-OCEAN",
        # CPTAC
        "cptac_ccrcc",
        "cptac_hnscc",
        "cptac_ucec",
    }:
        return name

    return alias.get(key, name)


def resolve_slide_pkl(
    base_dir: Path,
    class_name: Optional[str],
    slide_name: str,
) -> Path:
    candidates = []
    if class_name is not None:
        r = base_dir / class_name
        candidates.append(r / slide_name / f"{slide_name}.pkl")
        candidates.append(r / f"{slide_name}.pkl")
    else:
        candidates.append(base_dir / slide_name / f"{slide_name}.pkl")
        candidates.append(base_dir / f"{slide_name}.pkl")

    for p in candidates:
        if p.exists():
            return p

    msg = f"[ERROR] .pkl not found for slide='{slide_name}' under base='{base_dir}'"
    if class_name is not None:
        msg += f", class='{class_name}'"
    raise FileNotFoundError(msg)


def try_resolve_alt_res(
    original_base: Path,
    original_res: str,
    alt_res: str,
    class_name: Optional[str],
    slide_name: str,
) -> Optional[Path]:
    alt_base = Path(str(original_base).replace(f"/{original_res}/", f"/{alt_res}/"))
    try:
        return resolve_slide_pkl(alt_base, class_name, slide_name)
    except FileNotFoundError:
        return None


# ===============================
# Base: 공통 Dataset 부모 클래스
# ===============================
class _BaseWSIDataset(Dataset):
    def __init__(
        self,
        dataset_root: str,
        dataset_info: dict,
        resolutions: List[str],
        patch_size: int,
        data_split: str,
        feature_extractor: str = "resnet50-tr-supervised-imagenet1k",
    ):
        super().__init__()
        self.dataset_root = dataset_root
        self.dataset_info = dataset_info
        self.patch_size = patch_size
        self.data_split = data_split
        self.feature_extractor = feature_extractor

        # 🔥 Sampler용 라벨 캐시
        self.labels: List[int] = []

        # 해상도 정렬 (내림차순)
        self.resolutions = sorted(resolutions, key=lambda x: int(x[1:]), reverse=True)
        self.resolution_values = [int(res[1:]) for res in self.resolutions]
        self.highest_res = self.resolutions[0]
        self.highest_res_value = self.resolution_values[0]
        self.downsampling_factors = {
            res: self.highest_res_value // res_val
            for res, res_val in zip(self.resolutions, self.resolution_values)
        }


# ===============================
# Dataset: 3D_certain / Leica (PKL 로더)
# ===============================
class ThreeDCertainWSIDataset(_BaseWSIDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.data: List[Tuple[str, str, str]] = []  # (class, slide, data_element)
        self.label_mapping: Dict[str, Dict[str, int]] = {}

        for data_element_name in self.dataset_info.keys():
            base = Path(
                f"{self.dataset_root}/{data_element_name}/{self.feature_extractor}/{self.highest_res}/{self.patch_size}/{self.data_split}"
            )
            self.dataset_info[data_element_name]["highest_res_slide_file_path"] = base

            label_mapping = self._create_label_mapping(data_element_name)
            self.dataset_info[data_element_name]["label_mapping"] = label_mapping
            self.label_mapping[data_element_name] = label_mapping

            data_list = self._load_data(data_element_name)
            self.dataset_info[data_element_name]["data"] = data_list
            self.data += data_list

        self.data = [(c, s, d, 0) for (c, s, d) in self.data]

        self.labels = []
        for c, _, d_name, _ in self.data:
            mapping = self.label_mapping[d_name]
            label_key = "TVA+VA" if c in ["TVA", "VA"] else c
            self.labels.append(mapping[label_key])

        print(f"[{self.__class__.__name__}] Dataset '{self.data_split}' initialized:")
        if self.data_split != "test":
            print(f"  Total samples: {len(self.data)}")

    def _create_label_mapping(self, data_element_name) -> dict:
        mapping = {}
        class_names_list = self.dataset_info[data_element_name]["class_names_list"]
        for i, class_name in enumerate(class_names_list):
            mapping[class_name] = i
        if "TVA+VA" in mapping:
            mapping["TVA"] = mapping["TVA+VA"]
            mapping["VA"] = mapping["TVA+VA"]
        return mapping

    def _load_data(self, data_element_name) -> List[Tuple[str, str, str]]:
        data = []
        base: Path = self.dataset_info[data_element_name]["highest_res_slide_file_path"]
        if not base.exists():
            print(f"[WARN] Base path does not exist: {base}")
            return data

        for class_dir in base.iterdir():
            if not class_dir.is_dir():
                continue
            for entry in class_dir.iterdir():
                if entry.is_file() and entry.suffix == ".pkl":
                    slide_name = entry.stem
                elif entry.is_dir():
                    slide_name = entry.name
                else:
                    continue
                data.append((class_dir.name, slide_name, data_element_name))
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        class_name, slide_name, data_element_name, _ = self.data[index]
        base: Path = self.dataset_info[data_element_name]["highest_res_slide_file_path"]

        slide_path = resolve_slide_pkl(base, class_name, slide_name)

        with open(slide_path, "rb") as f:
            slide_data = pickle.load(f)

        highest_res_feature_map: Dict[str, np.ndarray] = slide_data.get("features", {})
        other_res_feature_maps: Dict[str, Dict[str, np.ndarray]] = {}

        for other_res in self.resolutions[1:]:
            alt_path = try_resolve_alt_res(
                base, self.highest_res, other_res, class_name, slide_name
            )
            if alt_path is not None:
                with open(alt_path, "rb") as f:
                    res_data = pickle.load(f)
                    other_res_feature_maps[other_res] = res_data.get("features", {})

        slide_feature_list = []
        slide_coordinates = []
        train_split = (self.data_split == "train")

        for coord_str, feat in highest_res_feature_map.items():
            x, y = map(int, coord_str.split("-"))
            if len(self.resolutions) == 1:
                slide_feature_list.append(torch.from_numpy(feat).float())
                if not train_split:
                    slide_coordinates.append(coord_str)
            else:
                patch_feats = [feat]
                if not train_split:
                    patch_coords = {self.highest_res: coord_str}
                for orr in self.resolutions[1:]:
                    x_r, y_r = (
                        x // self.downsampling_factors[orr],
                        y // self.downsampling_factors[orr],
                    )
                    coord_r = f"{x_r}-{y_r}"
                    patch_feats.append(
                        other_res_feature_maps.get(orr, {}).get(coord_r, feat)
                    )
                    if not train_split:
                        patch_coords[orr] = coord_r

                slide_feature_list.append(
                    torch.from_numpy(np.concatenate(patch_feats)).float()
                )
                if not train_split:
                    slide_coordinates.append(patch_coords)

        slide_features_tensor = torch.stack(slide_feature_list, dim=0)

        label_map = self.label_mapping[data_element_name]
        label_key = "TVA+VA" if class_name in ["TVA", "VA"] else class_name
        label = torch.tensor(label_map[label_key], dtype=torch.long)

        coords_return = slide_coordinates if not train_split else None
        return slide_name, coords_return, slide_features_tensor, label


class LeicaCertainWSIDataset(ThreeDCertainWSIDataset):
    pass


# ===============================
# Generic H5 Dataset (H5 로더 통합 최적화)
# ===============================
class _BaseH5WSIDataset(_BaseWSIDataset):
    """
    Camelyon16, BRACS, TCGA, HISAI-SKIN, PANDA, UBC-OCEAN, CPTAC 등 H5 기반 데이터셋 공통 부모 클래스.

    ✅ CPTAC처럼 다음 형태도 지원:
      .../<FE>/<res>/<split>/<class>/<slide_id>/<uuid>.h5
    즉, class 폴더 바로 아래가 아니라 더 깊게 .h5가 있어도 로딩 가능하도록
    self.data에 '실제 h5 경로(Path)'를 저장하는 구조로 변경.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # ✅ (class_key, h5_path, dataset_key)
        self.data: List[Tuple[str, Path, str]] = []

        self._scan_all_files()

        print(
            f"[{self.__class__.__name__}] split={self.data_split} items={len(self.data)} "
            f"resolutions={self.resolutions}"
        )

    def _pick_existing_base(self, data_element_name: str) -> Optional[Path]:
        """
        데이터셋마다 경로에 patch_size가 포함되거나 포함되지 않을 수 있어서 둘 다 시도.
        """
        base_with_ps = Path(
            f"{self.dataset_root}/{data_element_name}/{self.feature_extractor}/{self.highest_res}/{self.patch_size}/{self.data_split}"
        )
        base_no_ps = Path(
            f"{self.dataset_root}/{data_element_name}/{self.feature_extractor}/{self.highest_res}/{self.data_split}"
        )

        if base_with_ps.exists():
            return base_with_ps
        if base_no_ps.exists():
            return base_no_ps
        return None

    def _scan_all_files(self):
        for data_element_name in self.dataset_info.keys():
            base = self._pick_existing_base(data_element_name)
            if base is None:
                print(
                    f"[WARN] Base path not found (tried both):\n"
                    f" - {self.dataset_root}/{data_element_name}/{self.feature_extractor}/{self.highest_res}/{self.patch_size}/{self.data_split}\n"
                    f" - {self.dataset_root}/{data_element_name}/{self.feature_extractor}/{self.highest_res}/{self.data_split}"
                )
                continue

            self.dataset_info[data_element_name]["base_path"] = base

            cls_list = self.dataset_info[data_element_name]["class_names_list"]
            mapping = {name: i for i, name in enumerate(cls_list)}
            self.dataset_info[data_element_name]["label_mapping"] = mapping

            if not base.exists():
                print(f"[WARN] Base path not found: {base}")
                continue

            for class_dir in sorted(base.iterdir()):
                if not class_dir.is_dir():
                    continue

                class_key = class_dir.name

                # 폴더명이 숫자(0,1,2,...) 또는 문자열(G1,G2,...) 모두 대응
                if class_key.isdigit():
                    lbl = int(class_key)
                else:
                    lbl = mapping.get(class_key, -1)

                # ✅ 핵심: 깊은 폴더 구조까지 모두 스캔
                h5_files = sorted(class_dir.glob("**/*.h5"))
                if len(h5_files) == 0:
                    continue

                for h5_path in h5_files:
                    self.data.append((class_key, h5_path, data_element_name))
                    if lbl != -1:
                        self.labels.append(lbl)

    def _load_h5_features(self, file_path: Path, require_coords: bool = True):
        with h5py.File(str(file_path), "r") as f:
            if "features" in f:
                feats = np.array(f["features"])
            else:
                ds_name = list(f.keys())[0]
                feats = np.array(f[ds_name])

            coords_list = None
            if require_coords:
                if "coords" in f:
                    coords = np.array(f["coords"])
                    coords_list = [f"{int(x)}-{int(y)}" for x, y in coords]
                elif "x" in f and "y" in f:
                    xs, ys = np.array(f["x"]), np.array(f["y"])
                    coords_list = [f"{int(x)}-{int(y)}" for x, y in zip(xs, ys)]
                else:
                    coords_list = [str(i) for i in range(len(feats))]
        return feats, coords_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        class_key, h5_path, data_element_name = self.data[index]

        need_coords = (self.data_split != "train")
        feats_high, coords_high = self._load_h5_features(h5_path, require_coords=need_coords)
        feat_list = [feats_high]

        # 멀티 해상도 병합: 동일한 상대경로를 가정하고 res 부분만 치환
        for res in self.resolutions[1:]:
            alt_path_str = str(h5_path).replace(f"/{self.highest_res}/", f"/{res}/")
            alt_path = Path(alt_path_str)

            if alt_path.exists():
                feats_r, _ = self._load_h5_features(alt_path, require_coords=False)
                if len(feats_r) == len(feats_high):
                    feat_list.append(feats_r)

        feat_mat = np.concatenate(feat_list, axis=1) if len(feat_list) > 1 else feats_high
        slide_features_tensor = torch.from_numpy(feat_mat).float()

        # ✅ slide_name은 기본적으로 uuid stem. (원하면 parent 폴더 포함하도록 커스텀 가능)
        slide_name = h5_path.stem

        cls_list = self.dataset_info[data_element_name]["class_names_list"]
        mapping = self.dataset_info[data_element_name]["label_mapping"]

        if class_key.isdigit():
            idx = int(class_key)
            if 0 <= idx < len(cls_list):
                label = torch.tensor(idx, dtype=torch.long)
            else:
                raise ValueError(f"Invalid class index {class_key}")
        else:
            if class_key not in mapping:
                raise KeyError(
                    f"[{self.__class__.__name__}] Unknown class folder '{class_key}'. "
                    f"Expected one of {list(mapping.keys())}"
                )
            label = torch.tensor(mapping[class_key], dtype=torch.long)

        return slide_name, coords_high, slide_features_tensor, label


# ===============================
# 구체적인 H5 Dataset 클래스들 (Base 상속)
# ===============================
class Camelyon16WSIDataset(_BaseH5WSIDataset): pass
class BRACSWSIDataset(_BaseH5WSIDataset): pass
class TCGABinaryWSIDataset(_BaseH5WSIDataset): pass
class HistaiSkinWSIDataset(_BaseH5WSIDataset): pass
class PANDAWSIDataset(_BaseH5WSIDataset): pass
class UBCOCEANWSIDataset(_BaseH5WSIDataset): pass

# ✅ CPTAC (grading)
class CPTACCCRCCWSIDataset(_BaseH5WSIDataset): pass   # G1,G2,G3,G4
class CPTACHNSCCWSIDataset(_BaseH5WSIDataset): pass   # G1,G2,G3
class CPTACUCECWSIDataset(_BaseH5WSIDataset): pass    # G1,G2,G3


# ===============================
# DataModule
# ===============================
class CombinedPatchFeaturesWSIDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_root: str,
        dataset_mode: str,
        train_dataset_name: List[str],
        resolutions: List[str],
        patch_size: int,
        feature_extractor: str = "resnet50-tr-supervised-imagenet1k",
        num_workers: int = 1,
    ):
        super().__init__()
        self.dataset_root = dataset_root
        self.train_dataset_name = [normalize_dataset_name(n) for n in train_dataset_name]
        self.dataset_mode = dataset_mode
        self.resolutions = resolutions
        self.patch_size = patch_size
        self.feature_extractor = feature_extractor
        self.num_workers = num_workers

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.use_weighted_sampler: bool = False
        self.sampler_power: float = 1.0
        self._train_sample_weights: Optional[torch.Tensor] = None

        # Info Dictionaries 초기화
        self.train_camelyon16_info = {}
        self.val_camelyon16_info = {}
        self.test_camelyon16_info = {}

        self.train_3d_certain_info = {}
        self.val_3d_certain_info = {}
        self.test_3d_certain_info = {}

        self.train_leica_certain_info = {}
        self.val_leica_certain_info = {}
        self.test_leica_certain_info = {}

        self.train_bracs_info = {}
        self.val_bracs_info = {}
        self.test_bracs_info = {}

        self.train_tcga_nsclc_info = {}
        self.val_tcga_nsclc_info = {}
        self.test_tcga_nsclc_info = {}

        self.train_tcga_rcc_info = {}
        self.val_tcga_rcc_info = {}
        self.test_tcga_rcc_info = {}

        self.train_tcga_glioma_info = {}
        self.val_tcga_glioma_info = {}
        self.test_tcga_glioma_info = {}

        self.train_histai_skin_b1_info = {}
        self.val_histai_skin_b1_info = {}
        self.test_histai_skin_b1_info = {}

        self.train_panda_info = {}
        self.val_panda_info = {}
        self.test_panda_info = {}

        self.train_ubc_ocean_info = {}
        self.val_ubc_ocean_info = {}
        self.test_ubc_ocean_info = {}

        # ✅ CPTAC
        self.train_cptac_ccrcc_info = {}
        self.val_cptac_ccrcc_info = {}
        self.test_cptac_ccrcc_info = {}

        self.train_cptac_hnscc_info = {}
        self.val_cptac_hnscc_info = {}
        self.test_cptac_hnscc_info = {}

        self.train_cptac_ucec_info = {}
        self.val_cptac_ucec_info = {}
        self.test_cptac_ucec_info = {}

        self._setup_dataset_info_dicts()

    def _setup_dataset_info_dicts(self):
        def _init_info(name, train_dict, val_dict, test_dict):
            if name in self.train_dataset_name:
                train_dict[name] = {}
                val_dict[name] = {}
                test_dict[name] = {}

        _init_info("camelyon16", self.train_camelyon16_info, self.val_camelyon16_info, self.test_camelyon16_info)
        _init_info("3D_certain", self.train_3d_certain_info, self.val_3d_certain_info, self.test_3d_certain_info)
        _init_info("Leica_certain", self.train_leica_certain_info, self.val_leica_certain_info, self.test_leica_certain_info)
        _init_info("bracs", self.train_bracs_info, self.val_bracs_info, self.test_bracs_info)
        _init_info("TCGA-NSCLC", self.train_tcga_nsclc_info, self.val_tcga_nsclc_info, self.test_tcga_nsclc_info)
        _init_info("TCGA-RCC", self.train_tcga_rcc_info, self.val_tcga_rcc_info, self.test_tcga_rcc_info)
        _init_info("TCGA-GLIOMA", self.train_tcga_glioma_info, self.val_tcga_glioma_info, self.test_tcga_glioma_info)
        _init_info("histai-skin-b1", self.train_histai_skin_b1_info, self.val_histai_skin_b1_info, self.test_histai_skin_b1_info)
        _init_info("panda", self.train_panda_info, self.val_panda_info, self.test_panda_info)
        _init_info("UBC-OCEAN", self.train_ubc_ocean_info, self.val_ubc_ocean_info, self.test_ubc_ocean_info)

        # ✅ CPTAC
        _init_info("cptac_ccrcc", self.train_cptac_ccrcc_info, self.val_cptac_ccrcc_info, self.test_cptac_ccrcc_info)
        _init_info("cptac_hnscc", self.train_cptac_hnscc_info, self.val_cptac_hnscc_info, self.test_cptac_hnscc_info)
        _init_info("cptac_ucec", self.train_cptac_ucec_info, self.val_cptac_ucec_info, self.test_cptac_ucec_info)

        # Class Names 설정
        for name in self.train_dataset_name:
            cls_list = get_class_names(name)

            def _set_cls(info_dict, key):
                if info_dict:
                    info_dict[key]["class_names_list"] = cls_list

            if name == "camelyon16":
                _set_cls(self.train_camelyon16_info, name); _set_cls(self.val_camelyon16_info, name); _set_cls(self.test_camelyon16_info, name)
            elif name == "3D_certain":
                _set_cls(self.train_3d_certain_info, name); _set_cls(self.val_3d_certain_info, name); _set_cls(self.test_3d_certain_info, name)
            elif name == "Leica_certain":
                _set_cls(self.train_leica_certain_info, name); _set_cls(self.val_leica_certain_info, name); _set_cls(self.test_leica_certain_info, name)
            elif name == "bracs":
                _set_cls(self.train_bracs_info, name); _set_cls(self.val_bracs_info, name); _set_cls(self.test_bracs_info, name)
            elif name == "TCGA-NSCLC":
                _set_cls(self.train_tcga_nsclc_info, name); _set_cls(self.val_tcga_nsclc_info, name); _set_cls(self.test_tcga_nsclc_info, name)
            elif name == "TCGA-RCC":
                _set_cls(self.train_tcga_rcc_info, name); _set_cls(self.val_tcga_rcc_info, name); _set_cls(self.test_tcga_rcc_info, name)
            elif name == "TCGA-GLIOMA":
                _set_cls(self.train_tcga_glioma_info, name); _set_cls(self.val_tcga_glioma_info, name); _set_cls(self.test_tcga_glioma_info, name)
            elif name == "histai-skin-b1":
                _set_cls(self.train_histai_skin_b1_info, name); _set_cls(self.val_histai_skin_b1_info, name); _set_cls(self.test_histai_skin_b1_info, name)
            elif name == "panda":
                _set_cls(self.train_panda_info, name); _set_cls(self.val_panda_info, name); _set_cls(self.test_panda_info, name)
            elif name == "UBC-OCEAN":
                _set_cls(self.train_ubc_ocean_info, name); _set_cls(self.val_ubc_ocean_info, name); _set_cls(self.test_ubc_ocean_info, name)
            elif name == "cptac_ccrcc":
                _set_cls(self.train_cptac_ccrcc_info, name); _set_cls(self.val_cptac_ccrcc_info, name); _set_cls(self.test_cptac_ccrcc_info, name)
            elif name == "cptac_hnscc":
                _set_cls(self.train_cptac_hnscc_info, name); _set_cls(self.val_cptac_hnscc_info, name); _set_cls(self.test_cptac_hnscc_info, name)
            elif name == "cptac_ucec":
                _set_cls(self.train_cptac_ucec_info, name); _set_cls(self.val_cptac_ucec_info, name); _set_cls(self.test_cptac_ucec_info, name)

    @staticmethod
    def _wsi_collate_fn(batch):
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            raise RuntimeError("Empty batch encountered in _wsi_collate_fn (all samples were None).")

        names, coords, feats, labels = zip(*batch)
        feats = torch.stack(feats, dim=0)

        if isinstance(labels[0], torch.Tensor):
            labels = torch.stack(labels, dim=0)
        else:
            labels = torch.tensor(labels, dtype=torch.long)

        return list(names), list(coords), feats, labels

    def _build_train_sample_weights(self):
        if self.train_dataset is None:
            return

        print(
            f"[Sampler] WeightedRandomSampler 활성화: "
            f"데이터셋 길이={len(self.train_dataset)}, sampler_power={self.sampler_power}"
        )

        all_labels = []
        if isinstance(self.train_dataset, ConcatDataset):
            for ds in self.train_dataset.datasets:
                all_labels.extend(ds.labels)
        else:
            all_labels = self.train_dataset.labels

        labels_tensor = torch.tensor(all_labels, dtype=torch.long)
        if len(labels_tensor) == 0:
            print("[Sampler] Warning: No labels found. Skipping sampler build.")
            self.use_weighted_sampler = False
            return

        num_classes = int(labels_tensor.max().item()) + 1
        class_counts = torch.bincount(labels_tensor, minlength=num_classes).float()

        print(f"[Sampler] class_counts: {class_counts.tolist()}")

        class_weights = torch.zeros_like(class_counts)
        nonzero_mask = class_counts > 0
        class_weights[nonzero_mask] = 1.0 / (class_counts[nonzero_mask] ** self.sampler_power)

        self._train_sample_weights = class_weights[labels_tensor]
        print(f"[Sampler] class_weights: {class_weights.tolist()}")

    def setup(self, stage: Optional[str] = None) -> None:
        def get_ds_instance(name, info, split):
            args = (self.dataset_root, info, self.resolutions, self.patch_size, split, self.feature_extractor)

            if name == "camelyon16":      return Camelyon16WSIDataset(*args)
            if name == "3D_certain":      return ThreeDCertainWSIDataset(*args)
            if name == "Leica_certain":   return LeicaCertainWSIDataset(*args)
            if name == "bracs":           return BRACSWSIDataset(*args)
            if name == "TCGA-NSCLC":      return TCGABinaryWSIDataset(*args)
            if name == "TCGA-RCC":        return TCGABinaryWSIDataset(*args)
            if name == "TCGA-GLIOMA":     return TCGABinaryWSIDataset(*args)
            if name == "histai-skin-b1":  return HistaiSkinWSIDataset(*args)
            if name == "panda":           return PANDAWSIDataset(*args)
            if name == "UBC-OCEAN":       return UBCOCEANWSIDataset(*args)

            # ✅ CPTAC (grading)
            if name == "cptac_ccrcc":     return CPTACCCRCCWSIDataset(*args)
            if name == "cptac_hnscc":     return CPTACHNSCCWSIDataset(*args)
            if name == "cptac_ucec":      return CPTACUCECWSIDataset(*args)

            return None

        all_names = [
            "camelyon16",
            "3D_certain",
            "Leica_certain",
            "bracs",
            "TCGA-NSCLC",
            "TCGA-RCC",
            "TCGA-GLIOMA",
            "histai-skin-b1",
            "panda",
            "UBC-OCEAN",
            # ✅ CPTAC
            "cptac_ccrcc",
            "cptac_hnscc",
            "cptac_ucec",
        ]

        # ========== TRAIN ==========
        if self.dataset_mode == "train":
            datasets = []
            val_datasets = []

            for name in all_names:
                if name in self.train_dataset_name:
                    attr_name = name.lower().replace("-", "_")
                    tr_info = getattr(self, f"train_{attr_name}_info")
                    val_info = getattr(self, f"val_{attr_name}_info")

                    ds_tr = get_ds_instance(name, tr_info, "train")
                    ds_val = get_ds_instance(name, val_info, "val")
                    if ds_tr is None or ds_val is None:
                        raise ValueError(f"Dataset instance creation failed for '{name}'")

                    datasets.append(ds_tr)
                    val_datasets.append(ds_val)

            if not datasets:
                raise ValueError(f"No valid datasets found for train mode with {self.train_dataset_name}")

            self.train_dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
            self.val_dataset = val_datasets[0] if len(val_datasets) == 1 else ConcatDataset(val_datasets)

            if getattr(self, "use_weighted_sampler", False):
                self._build_train_sample_weights()

        # ========== VAL / TEST ==========
        else:
            datasets = []
            split_name = self.dataset_mode

            for name in all_names:
                if name in self.train_dataset_name:
                    attr_name = name.lower().replace("-", "_")
                    test_info = getattr(self, f"test_{attr_name}_info")
                    ds_te = get_ds_instance(name, test_info, split_name)
                    if ds_te is None:
                        raise ValueError(f"Dataset instance creation failed for '{name}'")
                    datasets.append(ds_te)

            if not datasets:
                raise ValueError(f"No valid datasets found for {split_name} mode with {self.train_dataset_name}")

            self.test_dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)

    # ==========================
    # Dataloaders
    # ==========================
    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is None. Check setup().")

        loader_kwargs = {
            "batch_size": 1,
            "num_workers": self.num_workers,
            "pin_memory": True,
            "persistent_workers": True if self.num_workers > 0 else False,
            "collate_fn": self._wsi_collate_fn,
        }
        if self.num_workers > 0:
            loader_kwargs["prefetch_factor"] = 1

        if getattr(self, "use_weighted_sampler", False):
            if self._train_sample_weights is None:
                self._build_train_sample_weights()

            sampler = WeightedRandomSampler(
                weights=self._train_sample_weights,
                num_samples=len(self._train_sample_weights),
                replacement=True,
            )
            return DataLoader(self.train_dataset, sampler=sampler, shuffle=False, **loader_kwargs)

        return DataLoader(self.train_dataset, shuffle=True, **loader_kwargs)

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("val_dataset is None.")
        loader_kwargs = {
            "batch_size": 1,
            "shuffle": False,
            "num_workers": self.num_workers,
            "pin_memory": True,
            "persistent_workers": True if self.num_workers > 0 else False,
            "collate_fn": self._wsi_collate_fn,
        }
        if self.num_workers > 0:
            loader_kwargs["prefetch_factor"] = 1
        return DataLoader(self.val_dataset, **loader_kwargs)

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("test_dataset is None.")
        loader_kwargs = {
            "batch_size": 1,
            "shuffle": False,
            "num_workers": self.num_workers,
            "pin_memory": True,
            "persistent_workers": True if self.num_workers > 0 else False,
            "collate_fn": self._wsi_collate_fn,
        }
        if self.num_workers > 0:
            loader_kwargs["prefetch_factor"] = 1
        return DataLoader(self.test_dataset, **loader_kwargs)

