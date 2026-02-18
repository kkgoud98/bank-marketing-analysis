# ============================================================
# Bank Marketing Campaign Analysis
# Customer Segmentation & Predictive Response Modeling
# Dataset: UCI ML Repository - Bank Marketing Dataset
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report, roc_curve)
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STEP 1: LOAD DATA
# ============================================================

print("=" * 60)
print("BANK MARKETING CAMPAIGN ANALYSIS")
print("=" * 60)

# Download dataset from UCI if not present
import os
if not os.path.exists('bank.csv'):
    import urllib.request
    print("Downloading dataset...")
    urllib.request.urlretrieve(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip',
        'bank.zip'
    )
    import zipfile
    with zipfile.ZipFile('bank.zip', 'r') as z:
        z.extractall('.')
    print("Dataset downloaded successfully.")

df = pd.read_csv('bank.csv', sep=';')
print(f"\nDataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")
print("\nFirst 5 rows:")
print(df.head())

# ============================================================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

print("\n" + "=" * 60)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print("\nDataset Info:")
print(df.info())

print("\nBasic Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nTarget Variable Distribution:")
print(df['y'].value_counts())
print(f"\nClass Imbalance: {df['y'].value_counts(normalize=True).round(4) * 100}")

# ---- Visualizations ----

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Univariate Analysis - Numerical Features', fontsize=16, fontweight='bold')

numerical_cols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
colors = ['#1F3864', '#2E75B6', '#4472C4', '#70AD47', '#ED7D31', '#FFC000']

for idx, (col, color) in enumerate(zip(numerical_cols, colors)):
    row, c = divmod(idx, 3)
    axes[row, c].hist(df[col], bins=30, color=color, edgecolor='white', alpha=0.85)
    axes[row, c].set_title(f'Distribution of {col.capitalize()}', fontweight='bold')
    axes[row, c].set_xlabel(col)
    axes[row, c].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('eda_numerical_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: eda_numerical_distributions.png")

# Categorical distributions
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Univariate Analysis - Categorical Features', fontsize=16, fontweight='bold')

cat_cols = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
for idx, col in enumerate(cat_cols):
    row, c = divmod(idx, 3)
    vc = df[col].value_counts()
    axes[row, c].bar(vc.index, vc.values, color='#1F3864', alpha=0.85, edgecolor='white')
    axes[row, c].set_title(f'{col.capitalize()} Distribution', fontweight='bold')
    axes[row, c].set_xlabel(col)
    axes[row, c].set_ylabel('Count')
    axes[row, c].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('eda_categorical_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: eda_categorical_distributions.png")

# Target variable
fig, ax = plt.subplots(figsize=(6, 5))
counts = df['y'].value_counts()
colors_pie = ['#1F3864', '#70AD47']
ax.pie(counts, labels=['No (88.48%)', 'Yes (11.52%)'], colors=colors_pie,
       autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12})
ax.set_title('Target Variable Distribution\n(Term Deposit Subscription)', fontweight='bold')
plt.savefig('target_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: target_distribution.png")

# ============================================================
# STEP 3: BIVARIATE ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: BIVARIATE ANALYSIS")
print("=" * 60)

# Age vs Subscription
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Bivariate Analysis', fontsize=14, fontweight='bold')

df.groupby('y')['age'].plot(kind='hist', bins=20, alpha=0.6,
                             ax=axes[0], legend=True, color=['#1F3864', '#70AD47'])
axes[0].set_title('Age Distribution by Subscription')
axes[0].set_xlabel('Age')
axes[0].legend(['No', 'Yes'])

df.groupby('y')['balance'].plot(kind='hist', bins=30, alpha=0.6,
                                 ax=axes[1], legend=True, color=['#1F3864', '#70AD47'])
axes[1].set_title('Balance Distribution by Subscription')
axes[1].set_xlabel('Balance')
axes[1].legend(['No', 'Yes'])

plt.tight_layout()
plt.savefig('bivariate_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: bivariate_analysis.png")

# ============================================================
# STEP 4: DATA PREPROCESSING
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: DATA PREPROCESSING")
print("=" * 60)

df_processed = df.copy()

# Encode target variable
df_processed['y'] = (df_processed['y'] == 'yes').astype(int)
print(f"Target encoded: yes=1, no=0")

# Feature Engineering: Age bins
df_processed['age_group'] = pd.cut(df_processed['age'],
    bins=[0, 30, 45, 60, 100],
    labels=['18-30', '31-45', '46-60', '60+'])
print("Feature engineered: age_group")

# Feature Engineering: Month & contact as ordinal
month_map = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
             'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
df_processed['month_num'] = df_processed['month'].map(month_map)
print("Feature engineered: month_num")

# One-hot encode categorical variables
cat_features = ['job', 'marital', 'education', 'contact', 'poutcome', 'age_group']
df_processed = pd.get_dummies(df_processed, columns=cat_features, drop_first=True)
print(f"One-hot encoded: {cat_features}")

# Drop original columns no longer needed
df_processed.drop(['month', 'day'], axis=1, inplace=True)

# Handle binary columns
for col in ['default', 'housing', 'loan']:
    df_processed[col] = (df_processed[col] == 'yes').astype(int)

# Normalize numerical features
scaler = StandardScaler()
num_cols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous', 'month_num']
df_processed[num_cols] = scaler.fit_transform(df_processed[num_cols])
print(f"Normalized: {num_cols}")

print(f"\nProcessed dataset shape: {df_processed.shape}")

# Correlation Matrix
fig, ax = plt.subplots(figsize=(12, 10))
corr_cols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous',
             'default', 'housing', 'loan', 'y']
corr_matrix = df[['age', 'balance', 'duration', 'campaign', 'pdays', 'previous',
                   'default', 'housing', 'loan']].copy()
corr_matrix['default'] = (df['default'] == 'yes').astype(int)
corr_matrix['housing'] = (df['housing'] == 'yes').astype(int)
corr_matrix['loan'] = (df['loan'] == 'yes').astype(int)
corr_matrix['y'] = (df['y'] == 'yes').astype(int)

sns.heatmap(corr_matrix.corr(), annot=True, fmt='.2f', cmap='Blues',
            ax=ax, linewidths=0.5, square=True)
ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: correlation_matrix.png")

# ============================================================
# STEP 5: CUSTOMER SEGMENTATION (CLUSTERING)
# ============================================================

print("\n" + "=" * 60)
print("STEP 5: CUSTOMER SEGMENTATION")
print("=" * 60)

# Use key features for clustering
cluster_features = df[['age', 'balance', 'duration', 'campaign']].copy()
cluster_features['housing'] = (df['housing'] == 'yes').astype(int)
cluster_features['loan'] = (df['loan'] == 'yes').astype(int)
cluster_scaled = StandardScaler().fit_transform(cluster_features)

# Elbow method
inertias = []
K_range = range(2, 10)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(cluster_scaled)
    inertias.append(km.inertia_)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(K_range, inertias, 'o-', color='#1F3864', linewidth=2, markersize=8)
ax.set_xlabel('Number of Clusters (K)', fontsize=12)
ax.set_ylabel('Inertia', fontsize=12)
ax.set_title('Elbow Method — Optimal K Selection', fontsize=13, fontweight='bold')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('kmeans_elbow.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: kmeans_elbow.png")

# K-Means with K=3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(cluster_scaled)
print(f"\nK-Means Clustering (K=3):")
print(df['cluster'].value_counts())

# Cluster visualization
fig, ax = plt.subplots(figsize=(9, 6))
colors_cluster = ['#1F3864', '#70AD47', '#ED7D31']
for i, color in enumerate(colors_cluster):
    mask = df['cluster'] == i
    ax.scatter(df.loc[mask, 'age'], df.loc[mask, 'balance'],
               alpha=0.4, s=15, color=color, label=f'Cluster {i}')
ax.set_xlabel('Age', fontsize=12)
ax.set_ylabel('Balance', fontsize=12)
ax.set_title('Customer Segments — K-Means Clustering', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('kmeans_clusters.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: kmeans_clusters.png")

# Hierarchical Clustering (sample for performance)
sample = cluster_scaled[:300]
linked = linkage(sample, method='ward')
fig, ax = plt.subplots(figsize=(12, 6))
dendrogram(linked, ax=ax, truncate_mode='lastp', p=20,
           color_threshold=0.7*max(linked[:,2]))
ax.set_title('Hierarchical Clustering Dendrogram', fontsize=13, fontweight='bold')
ax.set_xlabel('Sample Index')
ax.set_ylabel('Distance')
plt.tight_layout()
plt.savefig('hierarchical_dendrogram.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: hierarchical_dendrogram.png")

# Cluster profile summary
cluster_summary = df.groupby('cluster')[['age', 'balance', 'duration', 'campaign']].mean().round(2)
print("\nCluster Profiles:")
print(cluster_summary)

# ============================================================
# STEP 6: PREDICTIVE MODELING
# ============================================================

print("\n" + "=" * 60)
print("STEP 6: PREDICTIVE MODELING")
print("=" * 60)

X = df_processed.drop('y', axis=1)
y = df_processed['y']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=8),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=-1)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob)
    }
    print(f"\n{name}:")
    print(f"  Accuracy:  {results[name]['accuracy']:.4f}")
    print(f"  Precision: {results[name]['precision']:.4f}")
    print(f"  Recall:    {results[name]['recall']:.4f}")
    print(f"  F1 Score:  {results[name]['f1']:.4f}")
    print(f"  ROC-AUC:   {results[name]['roc_auc']:.4f}")

# Model Comparison Chart
fig, ax = plt.subplots(figsize=(10, 6))
metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
x = np.arange(len(metrics))
width = 0.25
bar_colors = ['#1F3864', '#2E75B6', '#70AD47']

for i, (name, res) in enumerate(results.items()):
    vals = [res[m] for m in metrics]
    bars = ax.bar(x + i*width, vals, width, label=name, color=bar_colors[i], alpha=0.88)

ax.set_xlabel('Metric', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Performance Comparison', fontsize=13, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'])
ax.legend(fontsize=11)
ax.set_ylim(0, 1.1)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: model_comparison.png")

# ROC Curves
fig, ax = plt.subplots(figsize=(8, 6))
line_colors = ['#1F3864', '#ED7D31', '#70AD47']
for (name, res), color in zip(results.items(), line_colors):
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    ax.plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']:.3f})",
            color=color, linewidth=2)
ax.plot([0,1], [0,1], 'k--', alpha=0.5, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves — All Models', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: roc_curves.png")

# Confusion Matrix — Best Model (Random Forest)
rf_model = results['Random Forest']
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, rf_model['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
ax.set_title('Confusion Matrix — Random Forest', fontsize=13, fontweight='bold')
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix_rf.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: confusion_matrix_rf.png")

# Feature Importance — Random Forest
rf = results['Random Forest']['model']
feat_imp = pd.Series(rf.feature_importances_, index=X.columns).nlargest(15)

fig, ax = plt.subplots(figsize=(10, 7))
feat_imp.sort_values().plot(kind='barh', ax=ax, color='#1F3864', alpha=0.85, edgecolor='white')
ax.set_title('Top 15 Feature Importances — Random Forest', fontsize=13, fontweight='bold')
ax.set_xlabel('Importance Score')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: feature_importance.png")

# ============================================================
# STEP 7: SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE — SUMMARY")
print("=" * 60)
print(f"\nDataset: {df.shape[0]} customers | {df.shape[1]} attributes")
print(f"Class imbalance: 88.48% No | 11.52% Yes")
print(f"\nBest Model: Random Forest")
print(f"  Accuracy:  {results['Random Forest']['accuracy']:.4f}")
print(f"  ROC-AUC:   {results['Random Forest']['roc_auc']:.4f}")
print(f"\nTop predictors: duration, balance, age, campaign, pdays")
print(f"\nCustomer Segments (K=3):")
print(cluster_summary.to_string())
print("\nAll charts saved as PNG files.")
print("=" * 60)
