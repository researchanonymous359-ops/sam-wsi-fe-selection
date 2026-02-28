# SAM-Cluster and SAM-LP: Structure-Aware Evaluation Metrics for Selecting WSI Feature Extractors without MIL Training

Selecting an appropriate Feature Extractor (FE) for a new whole-slide image (WSI) dataset typically requires repeatedly training Multiple Instance Learning (MIL) models, leading to massive computational overhead. This repository introduces an efficient pre-evaluation framework that ranks candidate foundation models **without full MIL training**. 

By leveraging the Segment Anything Model (SAM) to capture inherent morphological boundaries and spatial heterogeneity, we introduce two novel WSI-aware metrics:
* **SAM-Cluster** (Unsupervised)
* **SAM-LP** (Supervised)

## 🌟 Key Features

* **Zero-Training Evaluation:** Rank candidate FEs (e.g., Virchow2, UNI, CONCH, Phikon) in minutes rather than weeks.
* **Structure-Aware Metrics:** Incorporates histological boundaries into evaluation metrics using SAM, moving beyond standard i.i.d. image assumptions.
* **High Rank Correlation:** Proven to correlate strongly with actual downstream MIL performance across 8 public WSI datasets.
* **Cost-Efficient:** Operates reliably on a small random subset of WSIs (e.g., 30 slides), drastically reducing computational scaling costs to $\mathcal{O}(1)$ relative to the number of candidate FEs.

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
Extract patches and embeddings from your WSI dataset using your candidate foundation models. In our pipeline, we utilized **Trident** for efficient WSI patching and feature extraction, and **SAM 1** to generate zero-shot segmentation masks for the sampled WSIs. *Note: SAM processing only needs to be executed once per WSI, regardless of how many FEs you evaluate.*

---

## 🧠 Methodology

### SAM-Cluster (Unsupervised)
Evaluates intra-region consistency and inter-region separability. It uses SAM masks as pseudo-structural labels. A higher score indicates superior preservation of histological boundaries.

$$S_{\text{SAM-Cluster}}(\phi) = 1 - \frac{S_{inter}}{S_{intra}}$$

### SAM-LP (Supervised)
Mitigates the "Signal Dilution" problem found in conventional mean-pooling Linear Probing. SAM-LP constructs structure-aware prototypes and uses the Expectation-Maximization (EM) algorithm to iteratively optimize attention weights and evaluate diagnostic task alignment.

$$S_{\text{SAM-LP}}(\phi) = \frac{1}{N} \sum_{i=1}^{N} \log P(Y_i \mid \tilde{z}_i^* ; \theta^*)$$

---

## 📊 Establishing Ground Truth via Downstream MIL Benchmarks

To rigorously evaluate our proposed pre-evaluation metrics without bias, we first establish a robust **Ground Truth (GT)**. This GT is defined as the actual downstream performance of each Feature Extractor (FE) when fully trained end-to-end. We achieve this by conducting exhaustive initial training across multiple benchmark tasks under a strictly unified experimental setup.



| Task | Evaluation Metric | Datasets | Pathology / Target |
| :--- | :--- | :--- | :--- |
| **1️⃣ Grade Scoring** | Quadratic Weighted Kappa (QWK) | • PANDA | • Prostate cancer |
| **2️⃣ Subtype Classification** | Balanced Accuracy | • Histai_skin-b1<br>• TCGA_GLIOMA<br>• TCGA_NSCLC<br>• TCGA_RCC<br>• UBC-OCEAN<br>• bracs<br>• camelyon16 | • Skin<br>• Brain tumor (glioma)<br>• Non-small cell lung cancer<br>• Renal cell carcinoma<br>• Bladder cancer<br>• Breast cancer<br>• Lymph node metastasis (Breast cancer) |

### 🔁 Experimental Protocol for GT Generation

To ensure the Ground Truth is highly reliable and allows for a fair comparison against our zero-training metrics, all initial full-training experiments strictly follow identical data splits and evaluation procedures.

* **Data Splits:** Standardized Train/Validation/Test splits are provided in:
  ```bash
  Benchmark-MIL/dataset/data_split
  ```
* **Random Seeds:** To minimize variance, every FE-Aggregator combination is fully trained and evaluated **5 times** using the following seeds:
  `20`, `40`, `60`, `80`, `100`

*Note: The final GT suitability ranking of a candidate Feature Extractor is determined by its average downstream performance across these seeds and aggregators.*

### 🏗️ Supported Architectures for GT Training

#### MIL Models
The repository exhaustively trains candidate FEs using a wide range of Multiple Instance Learning (MIL) aggregators to establish an unbiased baseline GT:
> `meanpooling`, `ABMIL`, `DSMIL`, `CLAM`, `TransMIL`, `DTFD-MIL-AFS`, `WiKG`, `RRTMIL`, `ILRA`

#### Feature Extractors
The following foundation models are fully trained in the initial GT setup:
> `conch_v1`, `conch_v15`, `lunit-vits16`, `musk`, `phikon_v2`, `resnet50`, `uni_v1`, `uni_v2`, `virchow2`, `hibou_b`

---

## 📁 Data Format: Required `.h5` Structure



Each `.h5` file strictly corresponds to **one Whole Slide Image (WSI)**. Ensure your data adheres to the following structural requirements.

### Mandatory Keys

**1. `features`**
Patch-level embeddings extracted from a foundation model.
* **Shape:** `(n_patches, d)`
* **Type:** `float16`, `float32`, or `float64`
* **Description:** Used by MIL aggregators and all evaluation metrics.

**2. `label`**
Slide-level ground truth label.
* **Shape:** scalar or `(1,)`
* **Type:** `int` (recommended)
* **Task-Specific Rules:**
    * *Grade Scoring (QWK):* Must represent **ordinal classes** (e.g., 0–5 for ISUP grade) where the ordering preserves clinical meaning.
    * *Subtype Classification (Balanced Accuracy):* Must represent categorical class IDs (`0` to `C-1`). 
    * *Note:* If labels are initially stored as strings, they must be converted to integer IDs prior to evaluation.

### Required for SAM-Based Metrics

If evaluating using structure-aware metrics (SAM-Cluster or SAM-LP), the following key is mandatory:

**3. `sam_region`**
SAM-derived region ID for each WSI patch.
* **Shape:** `(n_patches,)`
* **Type:** `int32` or `int64`
* **Constraint:** `features.shape[0]` must exactly equal `sam_region.shape[0]`.

### Optional Keys (Recommended)

**4. `coords`**
Patch spatial coordinates within the original WSI.
* **Shape:** `(n_patches, 2)` or `(n_patches, 4)`
* **Description:** Highly recommended for downstream visualization, spatial aggregators, and debugging.

### Example `.h5` Structure

```text
slide_0001.h5
 ├── features      float32   (5231, 768)
 ├── label         int64     ()
 ├── sam_region    int32     (5231,)
 ├── coords        int32     (5231, 2)   # optional
 └── attrs:
       └── slide_id: "slide_0001"
```

---

## 🔍 Pre-Evaluation Metric Assessment

Once the Ground Truth (GT) is established via full MIL training, we validate the effectiveness of pre-evaluation metrics (EffDim, NESum, Self-Cluster, SAM-Cluster, LogME, Linear Probing, SAM-LP). 

We compute the rank correlation (e.g., Spearman's $\rho$) between:
1. The suitability score generated by the pre-evaluation metric **prior to MIL training**.
2. The averaged **GT downstream MIL performance**.

This correlation verifies that our proposed metrics can reliably select the optimal Feature Extractor using only a small data subset, bypassing exhaustive training.

---

## ⚡ Computational Efficiency & Scalability


Traditional FE selection relies on exhaustive MIL training, which scales poorly. Our framework optimizes this process:
* **Single-Pass Prior:** SAM masks are computed only **once** per WSI, regardless of how many FEs are evaluated.
* **Zero Backpropagation:** Metric computation (including SAM-Cluster and SAM-LP) relies on lightweight algebraic operations over pre-extracted 1D embeddings.
* **Subset Reliability:** Evaluation cost scales only with the number of sampled WSIs (e.g., $N=30$), reducing the computational burden from weeks of GPU training to mere minutes.

---

## ✅ Consistency Checklist

Before executing the benchmark pipelines, verify that:
* [ ] Each WSI is mapped to exactly one `.h5` file.
* [ ] For SAM metrics, every patch must be assigned a region ID (i.e., the length of the `sam_region` array must equal the number of patches: `features.shape[0] == sam_region.shape[0]`).
* [ ] Label mappings are globally consistent across the dataset.
* [ ] Ordinal relationships are strictly preserved for datasets evaluated with QWK.

---
```