
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


# 🦷 Oral Cancer Risk Predictor

## Live Demo

Check out the live version of this app: [Mini Project NIA App](https://thulanithisarani-mini-projectnia-app-lzpp4q.streamlit.app/)


A web application that predicts the risk of oral cancer based on patient information using machine learning. Built with **Streamlit**, **Pandas**, and **Scikit-learn**.

---

## 🔹 Features

- Predicts **high or low risk** of oral cancer.
- Provides **probability scores** for better insight.
- Interactive **form inputs** for patient data.
- Scales numerical inputs and encodes categorical features.
- Uses a **Random Forest Classifier** for predictions.
- Fully responsive layout with sidebar info and instructions.




## 🚀 Running the App

1. Ensure your dataset `oral_cancer_prediction_dataset_modified.csv` is in the project folder.

2. Start the Streamlit app:

   ```bash
   streamlit run app.py
   ```

3. The app will open in your default browser. Fill out the form and click **Predict Oral Cancer Risk** to see results.


## 📊 Input Details

* **Age:** Patient age 
* **Cancer Stage:** Stage of cancer 
* **Survival Rate (5-Year, %):** Probability of 5-year survival (0–100%)
* Other patient-related categorical and numerical features as in the dataset.

---

## ⚠️ Disclaimer

This tool is **not a medical diagnostic device**. Results are based on a machine learning model and should **not replace professional medical advice**. Always consult a healthcare professional for accurate diagnosis and treatment.

---

## 📝 Technologies Used

* Python 3.x
* [Streamlit](https://streamlit.io/)
* [Pandas](https://pandas.pydata.org/)
* [NumPy](https://numpy.org/)
* [Scikit-learn](https://scikit-learn.org/)
* Random Forest Classifier

---


