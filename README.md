
# Oral Cancer Prediction & Clustering Analysis

This project applies **data mining and machine learning** techniques to an oral cancer dataset.  
The pipeline follows a **CRISP-DM style workflow** (Data Understanding → Preprocessing → Data Mining → Evaluation) and also mirrors the **clustering analysis steps** from the Loan PDF (Transformation → K-Means → Profiling).

---

## 📌 Project Workflow

### 1. Data Understanding
- Load dataset (`oral_cancer_prediction_dataset_modified.csv`)
- Explore shape, columns, and first rows
- Summary statistics for all features
- Check for missing values and duplicate rows
- Visualizations:
  - Target distribution
  - Numeric feature histograms
  - Boxplots for outlier detection

### 2. Data Preprocessing
- **Cleaning**:
  - Remove duplicates
  - Handle missing values (SimpleImputer: median/mode)
  - Drop rows with missing target labels
- **Feature Engineering**:
  - Age binning (Young, Middle, Old)
  - Tumor size discretization (quartiles)
- **Transformation**:
  - Standard scaling (normalize numeric features)
  - One-hot encode categorical features
- **Reduction**:
  - PCA (10 components for modeling, 2 components for visualization)

### 3. Data Mining & Machine Learning
- **Clustering**:
  - K-Means clustering on PCA-reduced features
  - Elbow method (WCSS) and Silhouette score to choose best `k`
  - Scatter plot of PCA components with cluster colors
  - Cluster profiling:
    - Cluster sizes
    - Mean values of numeric features
    - Oral cancer diagnosis distribution across clusters
- **Classification**:
  - Logistic Regression
  - Random Forest
  - Cross-validation
  - GridSearchCV hyperparameter tuning
  - ROC curve for Random Forest

### 4. Evaluation & Interpretation
- Classification reports (accuracy, precision, recall, F1)
- Cross-validation results
- Hyperparameter tuning results
- ROC curve (AUC)
- Cluster profiling results

---

## 📊 Visualizations
- Target distribution bar plot
- Histograms for feature distributions
- Boxplots for outlier detection
- PCA scatter plot by diagnosis
- PCA scatter plot with K-Means clusters
- Elbow and Silhouette method plots
- ROC curve for Random Forest

---

## 📂 Output Files
The script saves the following files:

- `oral_cancer_with_clusters.csv` → dataset with cluster labels
- `cluster_profile_numeric.csv` → average numeric features per cluster
- `cluster_target_counts.csv` → oral cancer diagnosis counts per cluster

---

## ⚙️ Requirements
- Python 3.8+
- Libraries:
  - pandas, numpy, matplotlib, seaborn
  - scikit-learn

Install requirements:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
````

---

## 🚀 How to Run

1. Place the dataset `oral_cancer_prediction_dataset_modified.csv` in the `data/` folder or update the path in the script.
2. Run the notebook or script:

   ```bash
   python oral_cancer_pipeline.py
   ```
3. Review console outputs, plots, and saved CSV files.

---

## 📌 Notes

* `roc_curve` requires binary target encoding. If using "Yes/No", specify `pos_label="Yes"`.
* PCA is used for both **dimensionality reduction** (model speed) and **visualization**.
* Clustering analysis replicates the loan clustering PDF structure (Transformation → K-Means → Profiling).

---

## 🧾 License

This project is for **educational and research purposes**.

```

---

Would you like me to also generate this README.md as an **actual file** (`README.md`) in your `/mnt/data/` folder so you can download it directly?
```
