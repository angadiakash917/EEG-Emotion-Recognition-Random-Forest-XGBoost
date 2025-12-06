# 📊 EEG Emotion Recognition Using Machine Learning (RF & XGBoost)

### 🔍 Overview

This project focuses on classifying emotional states from EEG signals using machine learning models. The dataset contains numerical EEG-derived feature sets from two channels, and the classification is extended to multiple emotion categories. The proposed pipeline evaluates performance, identifies important features, and analyses cross-channel contributions.

---

## 🎯 Project Objectives

* Classify emotional states from EEG signals
* Expand classification from 3 to up to 6–8 emotion classes
* Compare Random Forest (RF) and XGBoost (XGB)
* Study the contribution of cross-channel EEG features
* Evaluate performance using confusion matrices and feature importance

---

## 🧠 Emotion Classes Supported

| Class    | Description                            |
| -------- | -------------------------------------- |
| NEGATIVE | Low-valence unpleasant state           |
| NEUTRAL  | Baseline emotional state               |
| POSITIVE | High-valence pleasant state            |
| HAPPY    | Positive emotion with moderate arousal |
| SAD      | Low valence, low arousal               |
| ANGER    | Negative valence, high arousal         |
| FEAR     | Threat-based high arousal              |
| SURPRISE | High arousal; context dependent        |

---

## 🗂️ Dataset Details

* Pre-processed EEG feature dataset (`emotions/emotions.csv`)
* Features include:

  * Statistical features (mean, std, min, max)
  * FFT spectral features
  * Correlation and covariance matrices
  * Eigenvalue and log-matrix features
* Data split: **70% training / 30% testing**

---

## 🧪 Methodology

### 🔹 Processing Flow

```
📁 Load EEG features
🔁 Label Encoding (6–8 classes)
✂ Train-test split
🌳 Train Random Forest & XGBoost
📈 Confusion Matrix & Accuracy
⭐ Evaluate Top Features
🔍 Channel-wise performance
```

---

## 🧮 Machine Learning Models

### Random Forest (RF)

* Robust to noise
* Good interpretability
* Baseline ensemble method

### XGBoost (XGB)

* Boosted decision trees
* Higher accuracy
* Handles high-dimensional features efficiently

---

## 📊 Results Summary

* XGBoost slightly outperforms Random Forest
* Both models achieve high accuracy
* Cross-channel combination improves performance
* Feature importance provides interpretability

---

## 🔀 Cross-Channel Analysis

* Models evaluated using:

  * Channel A only
  * Channel B only
  * Combined A+B
* Combined model shows best accuracy
* Multi-channel EEG contributes stronger signal features

---

## 📌 Comparative Notes

| Model     | Pros                   | Cons                    |
| --------- | ---------------------- | ----------------------- |
| RF        | Stable, interpretable  | Slightly lower accuracy |
| XGB       | Best performance       | Higher training cost    |
| SVM       | Good in small datasets | Sensitive to tuning     |
| DL models | Very powerful          | Require large dataset   |

---

## 💡 Conclusion

* Ensemble ML models (RF & XGB) are effective for EEG-based emotion classification
* Cross-channel data improves recognition accuracy
* Interpretable feature importance makes the method scalable

---

## 👥 Individual Contributions

### **Akash A. (S20230020277)**

* EEG feature preprocessing, selection, and model training
* Confusion matrix and feature importance visualizations
* Experimental benchmarking and channel-based evaluation
* Wrote Methodology, Results, and Discussion
* Ensured reproducibility and code documentation

### **Sujan A. (S20230020279)**

* Literature review and dataset interpretation
* Wrote Introduction, Related Work, Abstract & parts of Conclusion
* Helped design model evaluation framework
* Managed IEEE formatting and structured reporting
* Contributed to PPT and comparative analysis write-up

---

## 📦 Repository Structure

```
├── emotions/
│   └── emotions.csv
├── models/
│   ├── rf_model.ipynb
│   └── xgb_model.ipynb
├── results/
│   ├── confusion_matrix.png
│   ├── feature_importance_rf.png
│   └── feature_importance_xgb.png
├── README.md
└── requirements.txt
```

---

## 🛠️ Installation & Usage

```bash
git clone <your-repo-link>
cd EEG-Emotion-Classification
pip install -r requirements.txt
python main.py
```


