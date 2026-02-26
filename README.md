## 📊 Benchmark Tasks and Evaluation

This repository supports three major benchmark tasks. All experiments are conducted under a unified experimental setup to ensure fair comparison.

| Task | Metric | Datasets | Pathology / Target |
| :--- | :--- | :--- | :--- |
| **1️⃣ Grade Scoring** | Quadratic Weighted Kappa (QWK) | • PANDA<br>| • Prostate cancer |
| **2️⃣ Subtype Classification** | Balanced Accuracy | • Histai_skin-b1<br>• TCGA_GLIOMA<br>• TCGA_NSCLC<br>• TCGA_RCC<br>• UBC-OCEAN<br>• bracs<br>• camelyon16 | • Skin<br>• Brain tumor (glioma)<br>• Non-small cell lung cancer<br>• Renal cell carcinoma<br>• Bladder cancer<br>• Breast cancer<br>• Lymph node metastasis (Breast cancer) |

---

### 🔁 Experimental Protocol

All experiments were conducted with consistent splits and evaluation protocols.

* **Data Splits:** Train/Validation/Test splits are provided in the directory below:
    ```bash
    Benchmark-MIL/dataset/data_split
    ```
* **Seeds:** All experiments were repeated **5 times** using the following seeds:
    `20`, `40`, `60`, `80`, `100`

### 🏗️ Supported Models & Features

#### MIL Models
The repository supports a wide range of Multiple Instance Learning (MIL) models:
> `meanpooling`, `maxpooling`, `ABMIL`, `DSMIL`, `CLAM-SB`, `CLAM-MB`, `TransMIL`, `Transformer`, `DTFD-MIL-AFS`, `WiKG`, `RRTMIL`, `ILRA`

#### Feature Extractors
Supported feature extractors include:
> `conch_v1`, `conch_v15`, `lunit-vits16`, `musk`, `phikon_v2`, `resnet50`, `uni_v1`, `uni_v2`, `virchow2`, `hibou_b`
