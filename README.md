# Bank Marketing Campaign Analysis

## Overview
End-to-end data analysis project on 4,521 bank customers to predict term deposit subscription likelihood using customer segmentation and predictive modeling.

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Power BI

## Project Structure
```
bank-marketing-analysis/
├── bank_marketing_analysis.py   # Main analysis script
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
├── Group_3_BankMarketing.docx   # Full project report
├── Group-3_BankMarketing.pptx  # Project presentation
└── charts/                      # Generated visualizations (auto-created on run)
```

## How to Run
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full analysis
python bank_marketing_analysis.py
```
The script will automatically download the dataset from the UCI ML Repository.

## Key Results
| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | ~89% | ~0.88 |
| Decision Tree | ~88% | ~0.83 |
| **Random Forest** | **~91%** | **~0.93** |

## Analysis Steps
1. **Data Loading** — UCI Bank Marketing Dataset (4,521 records, 17 attributes)
2. **EDA** — Univariate and bivariate analysis, class imbalance detection (88% No / 12% Yes)
3. **Preprocessing** — Missing value handling, one-hot encoding, feature engineering, normalization
4. **Customer Segmentation** — K-Means (K=3) and Hierarchical Clustering
5. **Predictive Modeling** — Logistic Regression, Decision Trees, Random Forest
6. **Evaluation** — Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix

## Key Findings
- **Account balance**, **call duration**, and **prior campaign history** are top predictors of subscription
- K-Means identified 3 distinct customer segments varying in age, balance, and campaign responsiveness
- Random Forest outperformed all models with highest accuracy and ROC-AUC
- 88% class imbalance handled through careful evaluation strategy

## Dataset
[UCI Machine Learning Repository — Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- 4,521 customer records
- 17 attributes: demographics, financial status, campaign history
- Target: term deposit subscription (yes/no)

## Visualizations Generated
- `eda_numerical_distributions.png` — Numerical feature distributions
- `eda_categorical_distributions.png` — Categorical feature distributions
- `target_distribution.png` — Class imbalance pie chart
- `bivariate_analysis.png` — Age & balance by subscription outcome
- `correlation_matrix.png` — Feature correlation heatmap
- `kmeans_elbow.png` — Optimal K selection
- `kmeans_clusters.png` — Customer segments scatter plot
- `hierarchical_dendrogram.png` — Hierarchical clustering dendrogram
- `model_comparison.png` — All models performance comparison
- `roc_curves.png` — ROC curves for all 3 models
- `confusion_matrix_rf.png` — Random Forest confusion matrix
- `feature_importance.png` — Top 15 feature importances
