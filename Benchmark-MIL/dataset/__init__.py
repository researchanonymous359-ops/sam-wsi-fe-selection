# dataset/__init__.py
# ============================================================
# 1. Class Label Definitions
# ============================================================

def get_camelyon16_CLASS_LABELS():
    return ["normal", "tumor"]


def get_default_CLASS_LABELS():
    return ["0", "1"]


# Seegene colon 계열 (3D / Leica) 7-class
def get_seegene_7_class_LABELS():
    return ["HP", "IP", "LP", "SSL", "TA", "TSA", "TVA+VA"]


# BRACS 7-class
def get_bracs_7_class_LABELS():
    return [
        "Type_ADH",
        "Type_DCIS",
        "Type_FEA",
        "Type_IC",
        "Type_N",
        "Type_PB",
        "Type_UDH",
    ]


# TCGA
def get_tcga_nsclc_CLASS_LABELS():
    return ["LUAD", "LUSC"]


def get_tcga_rcc_CLASS_LABELS():
    return ["KIRC", "KIRP"]


def get_tcga_glioma_CLASS_LABELS():
    return ["LGG", "GBM"]


# HISAI SKIN
def get_histai_skin_3_class_LABELS():
    return [
        "Benign_nevus",
        "Melanoma",
        "Non_melanoma_cancer",
    ]


# PANDA
def get_panda_6_class_LABELS():
    return ["0", "1", "2", "3", "4", "5"]


# UBC-OCEAN
def get_ubc_ocean_5_class_LABELS():
    return ["CC", "EC", "HGSC", "LGSC", "MC"]


# ✅ CPTAC (grading)
def get_cptac_ccrcc_4_class_LABELS():
    return ["G1", "G2", "G3", "G4"]


def get_cptac_hnscc_3_class_LABELS():
    return ["G1", "G2", "G3"]


def get_cptac_ucec_3_class_LABELS():
    return ["G1", "G2", "G3"]


# ============================================================
# 2. Dataset Name Normalization (🔥 핵심)
# ============================================================

def _normalize_dataset_name(name: str) -> str:
    """
    모든 dataset name을 다음 규칙으로 canonical name으로 변환
    - 대소문자 무시
    - '-' '_' 통일
    - alias 허용
    """
    if name is None:
        return None

    if isinstance(name, (list, tuple)):
        name = name[0]

    raw = str(name).strip()
    key = raw.lower().replace("_", "-")

    # ---- Canonical Mapping Table ----
    CANONICAL_MAP = {
        # Camelyon
        "camelyon": "camelyon16",
        "camelyon16": "camelyon16",

        # 3D / Leica
        "3d-certain": "3D_certain",
        "leica-certain": "Leica_certain",

        # BRACS
        "bracs": "bracs",

        # TCGA
        "tcga-nsclc": "tcga-nsclc",
        "tcga-rcc": "tcga-rcc",
        "tcga-glioma": "tcga-glioma",
        "tcga-kirc": "tcga-rcc",
        "tcga-kirp": "tcga-rcc",
        "tcga-lgg": "tcga-glioma",
        "tcga-gbm": "tcga-glioma",

        # HISAI SKIN
        "histai-skin": "histai-skin-b1",
        "histai-skin-b1": "histai-skin-b1",

        # PANDA
        "panda": "panda",
        "panda-wsi": "panda",
        "prostate-panda": "panda",

        # UBC-OCEAN
        "ubc-ocean": "UBC-OCEAN",
        "ubcocean": "UBC-OCEAN",
        "ocean": "UBC-OCEAN",

        # ✅ CPTAC (grading)
        "cptac-ccrcc": "cptac_ccrcc",
        "cptac-ccrcc-wsi": "cptac_ccrcc",
        "cptac-ccrcc-grading": "cptac_ccrcc",
        "cptac-hnscc": "cptac_hnscc",
        "cptac-hnscc-wsi": "cptac_hnscc",
        "cptac-hnscc-grading": "cptac_hnscc",
        "cptac-ucec": "cptac_ucec",
        "cptac-ucec-wsi": "cptac_ucec",
        "cptac-ucec-grading": "cptac_ucec",
    }

    # 1차: 정확 매칭
    if key in CANONICAL_MAP:
        return CANONICAL_MAP[key]

    # 2차: 포함 관계 (관대하게)
    for k, v in CANONICAL_MAP.items():
        if k in key:
            return v

    # fallback: 원래 문자열 (최소한 crash는 방지)
    return raw


# ============================================================
# 3. Main Entry
# ============================================================

def get_class_names(dataset_name):
    if dataset_name is None:
        print("Not specify dataset, use default dataset with label 0, 1 instead.")
        return get_default_CLASS_LABELS()

    canon = _normalize_dataset_name(dataset_name)

    if canon in {"3D_certain", "Leica_certain"}:
        return get_seegene_7_class_LABELS()

    if canon == "camelyon16":
        return get_camelyon16_CLASS_LABELS()

    if canon == "bracs":
        return get_bracs_7_class_LABELS()

    if canon == "tcga-nsclc":
        return get_tcga_nsclc_CLASS_LABELS()

    if canon == "tcga-rcc":
        return get_tcga_rcc_CLASS_LABELS()

    if canon == "tcga-glioma":
        return get_tcga_glioma_CLASS_LABELS()

    if canon == "histai-skin-b1":
        return get_histai_skin_3_class_LABELS()

    if canon == "panda":
        return get_panda_6_class_LABELS()

    if canon == "UBC-OCEAN":
        return get_ubc_ocean_5_class_LABELS()

    # ✅ CPTAC (grading)
    if canon == "cptac_ccrcc":
        return get_cptac_ccrcc_4_class_LABELS()

    if canon == "cptac_hnscc":
        return get_cptac_hnscc_3_class_LABELS()

    if canon == "cptac_ucec":
        return get_cptac_ucec_3_class_LABELS()

    raise NotImplementedError(
        f"Unknown dataset name: {dataset_name} (normalized: {canon})"
    )
