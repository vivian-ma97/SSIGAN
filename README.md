# SSIGAN: Subglacial Wetness Detection via GAN-based Anomaly Scoring

## 📘 Project Overview

**SSIGAN** is an unsupervised generative framework for identifying subglacial wetness conditions from radar echograms. It leverages GAN-based anomaly detection to distinguish basal water bodies from dry bedrock using radar A-scope signals.

---

## 📂 Data and Preprocessing

### 1. 📥 Data Source
- **Dataset:** [CReSIS 2009 Antarctica TO Gambit Campaign](https://data.cresis.ku.edu/data/rds/2009_Antarctica_TO_Gambit/)
- **Provider:** Center for Remote Sensing of Ice Sheets (CReSIS)
- **Note:** Please comply with the [CReSIS Data Access Policy](https://data.cresis.ku.edu/data/).

### 2. 📊 Dataset Overview
- Total radar echograms (B-scope files): **923**
- Valid data used in this project: **892**
- Invalid files: 31 (excluded due to corrupted traces or unusable signals)

### 3. ✂️ Input Preparation
- Crop each radar A-scope waveform to **200 samples** based on the manually selected pick boundary (ice-bed interface).
- Options for preprocessing:
  - **(a)** Write your own cropping script (MATLAB/Python).
  - **(b)** Use the provided script in `/Data_preparation/` to generate `.mat` or `.pkl` files.
- Save all processed files in: `pick_data/`

---

## 🧠 Anomaly Detection with SSIGAN

### Step 1: Load Pretrained GAN Model
Place your pretrained weights (e.g., `.h5` files) in:

```
Save_model/
```

---

### Step 2: Generate Anomaly Scores
Run:

```bash
python test_our.py
```

This will:
- Load cropped waveforms from `pick_data/`
- Generate anomaly scores using the trained GAN
- Save results to: `Anomaly_score/`

---

### Step 3: Normalize Anomaly Scores (R1)

Go to the `Anomaly_score/` folder and run:

```matlab
step1_caculate_anomaly_score_norm.m
step2_save_norm_result.m
```

Results will be saved in:

```
Normalized_Anomaly_score/
```

---

### Step 4: Compute Corrected Bed Reflectivity (R2, a.k.a. CBRP)

Navigate to the `CBRP/` folder and run:

```matlab
Step_2_caculate_CBRP.m
```

This script:
- Computes CBRP based on an attenuation rate of **12.76 dB/km** (AGAP region)
- Saves results and `max_CBRP` in:

```
final_CBRP_chu/
```

📖 If using this method, please cite:

> Chu, W., et al. (2021). *Multisystem synthesis of radar sounding observations of the Amundsen Sea sector from the 2004–2005 field season*.  
> *Journal of Geophysical Research: Earth Surface, 126*, e2021JF006296.  
> https://doi.org/10.1029/2021JF006296

---

### Step 5: Generate Final Metric R3

Run:

```matlab
step3_caculate_r3.m
```

This script combines R1 and R2 using:

\[
R_3 = \sqrt{(R_1 - R_{1,\text{ref}})^2 + (R_2 - R_{2,\text{ref}})^2}
\]

---


## 📄 License

This repository will be released under an open-source license upon final acceptance of the associated publication. **Notice:** This work is currently under peer review.  
 Please do not use, redistribute,the code or data until the paper has been officially accepted.   All materials will be made publicly available after final acceptance.

---


## 📬 Contact

For questions or collaborations, please contact the project maintainer. You can reach me at qianma@tongji.edu.cn — I am also available to provide remote assistance with code execution. Please don’t hesitate to get in touch.
