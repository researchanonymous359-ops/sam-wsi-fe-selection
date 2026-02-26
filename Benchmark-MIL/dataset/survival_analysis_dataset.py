# dataset/survival_analysis_dataset.py

from __future__ import annotations

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pytorch_lightning as pl
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import h5py
from tqdm import tqdm  # 진행률 표시를 위해 추가 권장


# ===============================
# Utils
# ===============================
def _sort_resolutions_desc(resolutions: List[str]) -> List[str]:
    return sorted(resolutions, key=lambda x: int(x[1:]), reverse=True)


def _infer_endpoint_for_dataset(dataset_name: str, survival_endpoint: Optional[str]) -> str:
    """
    너의 규칙:
    - TCGA-HNSC -> PFI
    - TCGA-OV, TCGA-STAD -> OS
    """
    if survival_endpoint is not None:
        return survival_endpoint

    name = str(dataset_name).upper()
    if "HNSC" in name:
        return "PFI"
    return "OS"


def _endpoint_to_keys(endpoint: str) -> Tuple[str, str]:
    endpoint = endpoint.upper()
    if endpoint not in {"OS", "PFI"}:
        raise ValueError(f"endpoint must be one of [OS, PFI], got={endpoint}")
    return endpoint, f"{endpoint}_time"


def _read_h5_features(file_path: Path, require_coords: bool) -> Tuple[np.ndarray, Optional[List[str]]]:
    """
    - 'features' dataset 우선, 없으면 top-level 첫 dataset 사용
    - coords는 train에서는 필요없으니 require_coords=False 권장
    """
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


def _read_h5_survival(file_path: Path, event_key: str, time_key: str) -> Tuple[Optional[int], Optional[int]]:
    """
    survival 정보 읽기.
    없거나 invalid면 (None, None) 반환.
    """
    try:
        with h5py.File(str(file_path), "r") as f:
            if "survival" not in f:
                return None, None
            g = f["survival"]
            if event_key not in g or time_key not in g:
                return None, None

            # 저장 방식이 scalar dataset일 수도 있고, numpy scalar일 수도 있음
            ev = g[event_key][()]
            tm = g[time_key][()]

            # numpy scalar -> python int
            ev = int(ev) if ev is not None else None
            tm = int(tm) if tm is not None else None

            # 유효성 최소 체크
            if ev not in (0, 1):
                return None, None
            if tm is None or tm < 0:
                return None, None

            return ev, tm
    except Exception:
        return None, None


def _find_h5_files_under(split_dir: Path) -> List[Path]:
    """
    split_dir 바로 아래에 *.h5가 있거나,
    split_dir/*/*.h5 형태로 class_dir가 있을 수도 있으니 둘 다 수집.
    """
    if not split_dir.exists():
        return []

    direct = sorted(split_dir.glob("*.h5"))
    nested = sorted(split_dir.glob("*/*.h5"))
    # 중복 제거
    all_files = list(dict.fromkeys(direct + nested))
    return all_files


# ===============================
# Survival H5 Dataset
# ===============================
class SurvivalH5WSIDataset(Dataset):
    """
    반환 형식 (기존 _wsi_collate_fn 스타일과 유사하게 유지):
      return slide_name, coords_or_None, feats_tensor, y
    여기서 y는 survival에서는 label 대신
      y = {"event": tensor(int64), "time": tensor(int64)}
    를 사용.

    - train split: coords=None (I/O 절약)
    - val/test split: coords 리스트 반환
    
    ✅ [Improved] self.times 리스트를 미리 구축하여 Trainer의 Bin 생성 속도를 최적화함.
    """

    def __init__(
        self,
        dataset_root: str,
        dataset_name: str,
        feature_extractor: str,
        resolutions: List[str],
        patch_size: int,          # 현재 H5 구조에서는 경로에 patch_size가 안 들어가므로 unused일 수 있음
        data_split: str,          # "train" | "val" | "test"
        survival_endpoint: Optional[str] = None,
        survival_event_key: Optional[str] = None,
        survival_time_key: Optional[str] = None,
        drop_no_survival: bool = True,
        verbose: bool = True,
    ):
        super().__init__()
        self.dataset_root = dataset_root
        self.dataset_name = dataset_name
        self.feature_extractor = feature_extractor
        self.patch_size = patch_size
        self.data_split = data_split
        self.drop_no_survival = drop_no_survival

        self.resolutions = _sort_resolutions_desc(resolutions)
        self.highest_res = self.resolutions[0]

        endpoint = _infer_endpoint_for_dataset(dataset_name, survival_endpoint)
        auto_event_key, auto_time_key = _endpoint_to_keys(endpoint)

        self.survival_event_key = survival_event_key or auto_event_key
        self.survival_time_key = survival_time_key or auto_time_key
        self.survival_endpoint = endpoint

        # 스캔
        self.base_dir = Path(self.dataset_root) / self.dataset_name / self.feature_extractor / self.highest_res / self.data_split
        self.h5_files: List[Path] = _find_h5_files_under(self.base_dir)

        # ✅ [Optimization] 메타데이터(times, events) 미리 캐싱
        # Trainer에서 Binning할 때 이 self.times를 참조하면 순식간에 끝남.
        self.valid_indices = []
        self.times = []
        self.events = []
        
        # 필터링 및 메타데이터 로드
        if verbose:
             print(f"[SurvivalH5WSIDataset] Scanning {len(self.h5_files)} files for metadata...")
        
        for i, p in enumerate(self.h5_files):
            ev, tm = _read_h5_survival(p, self.survival_event_key, self.survival_time_key)
            
            if ev is None or tm is None:
                if not self.drop_no_survival:
                     # 데이터는 쓰되, dummy값 (-1) 등을 넣는 정책이라면 여기서 처리
                     # 현재 로직상 drop_no_survival=False면 None 반환하게 되어있음 -> Dataset index 관리가 복잡해짐.
                     # 여기서는 drop_no_survival=False여도 일단 유효한 것만 리스트에 넣는 방식(Subset)이 안전함.
                     # 하지만 기존 __getitem__ 로직(None 반환)을 유지하려면 valid_indices 대신 전체 리스트 사용해야 함.
                     # 여기서는 "Valid한 데이터만 추려서 self.h5_files를 재구성"하는 방식(기존 코드 유지)을 따르되, times를 채움.
                     pass 
            else:
                self.valid_indices.append(i)
                self.times.append(float(tm))
                self.events.append(int(ev))
        
        # 파일 리스트 갱신 (drop_no_survival=True인 경우)
        if self.drop_no_survival:
            self.h5_files = [self.h5_files[i] for i in self.valid_indices]
            # self.times, self.events는 이미 valid한 것만 들어있음
            
            dropped_count = len(self.valid_indices) - len(self.times) # 0이어야 함
            if verbose:
                print(
                    f"[SurvivalH5WSIDataset] {self.dataset_name} split={self.data_split} "
                    f"endpoint={self.survival_endpoint} keys=({self.survival_event_key},{self.survival_time_key}) "
                    f"files={len(self.h5_files)} (dropped invalid)"
                )
        else:
             # drop하지 않는 경우, times에는 valid한 것만 들어있음.
             # __len__은 전체 파일 수.
             # 이 경우 Trainer에서 self.times 길이랑 len(dataset)이랑 다를 수 있어 주의 필요하나,
             # 보통 Binning은 valid한 time만 가지고 하므로 self.times만 넘겨주면 됨.
             if verbose:
                print(
                    f"[SurvivalH5WSIDataset] {self.dataset_name} split={self.data_split} "
                    f"endpoint={self.survival_endpoint} keys=({self.survival_event_key},{self.survival_time_key}) "
                    f"files={len(self.h5_files)} (keep all)"
                )

    def __len__(self) -> int:
        return len(self.h5_files)

    def __getitem__(self, index: int):
        h5_path = self.h5_files[index]
        slide_name = h5_path.stem

        need_coords = (self.data_split != "train")
        feats_high, coords_high = _read_h5_features(h5_path, require_coords=need_coords)
        feat_list = [feats_high]

        # multi-res concat (길이 같을 때만 concat)
        for res in self.resolutions[1:]:
            alt_path = Path(str(h5_path).replace(f"/{self.highest_res}/", f"/{res}/"))
            if alt_path.exists():
                feats_r, _ = _read_h5_features(alt_path, require_coords=False)
                if len(feats_r) == len(feats_high):
                    feat_list.append(feats_r)

        feat_mat = np.concatenate(feat_list, axis=1) if len(feat_list) > 1 else feats_high
        slide_features_tensor = torch.from_numpy(feat_mat).float()

        # survival
        # 이미 __init__에서 읽었지만, 안전하게 다시 읽거나 캐싱된 값 써도 됨.
        # 여기서는 파일 I/O 정합성을 위해 재료딩 (scalar read라 빠름)
        ev, tm = _read_h5_survival(h5_path, self.survival_event_key, self.survival_time_key)
        
        if (ev is None or tm is None) and self.drop_no_survival:
            return None

        # drop_no_survival=False이고 값이 없으면 dummy 처리
        if ev is None: ev = -1
        if tm is None: tm = -1

        y = {
            "event": torch.tensor(int(ev), dtype=torch.long),
            "time": torch.tensor(int(tm), dtype=torch.long),
        }

        coords_return = coords_high if need_coords else None
        return slide_name, coords_return, slide_features_tensor, y


# ===============================
# Collate
# ===============================
def survival_wsi_collate_fn(batch):
    """
    sample = (name, coords, feats, y_dict)
      y_dict = {"event": tensor, "time": tensor}

    return
      names: List[str]
      coords: List[Optional[List[str]]]
      feats:  Tensor[B, N, D]
      events: Tensor[B]
      times:  Tensor[B]
    """
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        # 빈 배치 방지: return None or raise
        # Lightning에서는 return None하면 해당 step skip되는 경우가 많음
        # 하지만 에러 디버깅을 위해 raise가 나을 수도 있음.
        # 여기서는 안전하게 dummy 반환 또는 에러
        raise RuntimeError("Empty batch encountered in survival_wsi_collate_fn (all samples were None).")

    names, coords, feats, y_dicts = zip(*batch)
    feats = torch.stack(feats, dim=0)

    events = torch.stack([y["event"] for y in y_dicts], dim=0).view(-1)
    times = torch.stack([y["time"] for y in y_dicts], dim=0).view(-1)

    return list(names), list(coords), feats, events, times


# ===============================
# DataModule
# ===============================
class SurvivalWSIDataModule(pl.LightningDataModule):
    """
    survival 전용 datamodule.
    - train/val/test 각각 SurvivalH5WSIDataset 생성
    - drop_no_survival=True로 survival 없는 파일은 아예 제외
    """

    def __init__(
        self,
        dataset_root: str,
        dataset_name: str,                 # ex) "TCGA-HNSC"
        feature_extractor: str,            # ex) "conch_v1"
        resolutions: List[str],            # ex) ["x5"] or ["x10","x5"]
        patch_size: int = 256,
        num_workers: int = 2,
        survival_endpoint: Optional[str] = None,    # OS / PFI (None이면 dataset_name으로 자동)
        survival_event_key: Optional[str] = None,   # None이면 endpoint로 자동(OS or PFI)
        survival_time_key: Optional[str] = None,    # None이면 endpoint로 자동(OS_time or PFI_time)
        drop_no_survival: bool = True,
    ):
        super().__init__()
        self.dataset_root = dataset_root
        self.dataset_name = dataset_name
        self.feature_extractor = feature_extractor
        self.resolutions = resolutions
        self.patch_size = patch_size
        self.num_workers = num_workers

        self.survival_endpoint = survival_endpoint
        self.survival_event_key = survival_event_key
        self.survival_time_key = survival_time_key
        self.drop_no_survival = drop_no_survival

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = SurvivalH5WSIDataset(
            dataset_root=self.dataset_root,
            dataset_name=self.dataset_name,
            feature_extractor=self.feature_extractor,
            resolutions=self.resolutions,
            patch_size=self.patch_size,
            data_split="train",
            survival_endpoint=self.survival_endpoint,
            survival_event_key=self.survival_event_key,
            survival_time_key=self.survival_time_key,
            drop_no_survival=self.drop_no_survival,
            verbose=True,
        )
        self.val_dataset = SurvivalH5WSIDataset(
            dataset_root=self.dataset_root,
            dataset_name=self.dataset_name,
            feature_extractor=self.feature_extractor,
            resolutions=self.resolutions,
            patch_size=self.patch_size,
            data_split="val",
            survival_endpoint=self.survival_endpoint,
            survival_event_key=self.survival_event_key,
            survival_time_key=self.survival_time_key,
            drop_no_survival=self.drop_no_survival,
            verbose=True,
        )
        self.test_dataset = SurvivalH5WSIDataset(
            dataset_root=self.dataset_root,
            dataset_name=self.dataset_name,
            feature_extractor=self.feature_extractor,
            resolutions=self.resolutions,
            patch_size=self.patch_size,
            data_split="test",
            survival_endpoint=self.survival_endpoint,
            survival_event_key=self.survival_event_key,
            survival_time_key=self.survival_time_key,
            drop_no_survival=self.drop_no_survival,
            verbose=True,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is None. Call setup() first.")
        loader_kwargs = dict(
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=survival_wsi_collate_fn,
            shuffle=True,
        )
        if self.num_workers > 0:
            loader_kwargs["prefetch_factor"] = 1
        return DataLoader(self.train_dataset, **loader_kwargs)

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("val_dataset is None. Call setup() first.")
        loader_kwargs = dict(
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=survival_wsi_collate_fn,
            shuffle=False,
        )
        if self.num_workers > 0:
            loader_kwargs["prefetch_factor"] = 1
        return DataLoader(self.val_dataset, **loader_kwargs)

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("test_dataset is None. Call setup() first.")
        loader_kwargs = dict(
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=survival_wsi_collate_fn,
            shuffle=False,
        )
        if self.num_workers > 0:
            loader_kwargs["prefetch_factor"] = 1
        return DataLoader(self.test_dataset, **loader_kwargs)


# ===============================
# (선택) 여러 TCGA cohort를 concat해서 쓰고 싶을 때
# ===============================
class MultiCohortSurvivalWSIDataModule(pl.LightningDataModule):
    """
    예: TCGA-HNSC + TCGA-OV + TCGA-STAD 같이 합쳐 학습할 때.
    """

    def __init__(
        self,
        dataset_root: str,
        dataset_names: List[str],
        feature_extractor: str,
        resolutions: List[str],
        patch_size: int = 256,
        num_workers: int = 2,
        drop_no_survival: bool = True,
    ):
        super().__init__()
        self.dataset_root = dataset_root
        self.dataset_names = dataset_names
        self.feature_extractor = feature_extractor
        self.resolutions = resolutions
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.drop_no_survival = drop_no_survival

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        tr_list, va_list, te_list = [], [], []
        for name in self.dataset_names:
            tr_list.append(SurvivalH5WSIDataset(
                self.dataset_root, name, self.feature_extractor,
                self.resolutions, self.patch_size, "train",
                survival_endpoint=None,  # auto rule
                drop_no_survival=self.drop_no_survival,
                verbose=True,
            ))
            va_list.append(SurvivalH5WSIDataset(
                self.dataset_root, name, self.feature_extractor,
                self.resolutions, self.patch_size, "val",
                survival_endpoint=None,
                drop_no_survival=self.drop_no_survival,
                verbose=True,
            ))
            te_list.append(SurvivalH5WSIDataset(
                self.dataset_root, name, self.feature_extractor,
                self.resolutions, self.patch_size, "test",
                survival_endpoint=None,
                drop_no_survival=self.drop_no_survival,
                verbose=True,
            ))

        self.train_dataset = tr_list[0] if len(tr_list) == 1 else ConcatDataset(tr_list)
        self.val_dataset = va_list[0] if len(va_list) == 1 else ConcatDataset(va_list)
        self.test_dataset = te_list[0] if len(te_list) == 1 else ConcatDataset(te_list)

    def train_dataloader(self):
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is None. Call setup().")
        kwargs = dict(
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=survival_wsi_collate_fn,
            shuffle=True,
        )
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = 1
        return DataLoader(self.train_dataset, **kwargs)

    def val_dataloader(self):
        if self.val_dataset is None:
            raise RuntimeError("val_dataset is None. Call setup().")
        kwargs = dict(
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=survival_wsi_collate_fn,
            shuffle=False,
        )
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = 1
        return DataLoader(self.val_dataset, **kwargs)

    def test_dataloader(self):
        if self.test_dataset is None:
            raise RuntimeError("test_dataset is None. Call setup().")
        kwargs = dict(
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=survival_wsi_collate_fn,
            shuffle=False,
        )
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = 1
        return DataLoader(self.test_dataset, **kwargs)