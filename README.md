# AI/ML Model Training & Data Refinement Project

## Overview
This notebook covers a complete machine learning workflow — from generating and analyzing data to training, evaluating, and refining models.  
It focuses on practical, real-world ML tasks, including model comparison, evaluation metrics, and statistical data cleaning using differential statistics.

---

## Steps Covered

### 1. Import Libraries
All essential libraries for data manipulation, visualization, and modeling are imported.  
Includes:
- numpy  
- pandas  
- matplotlib  
- seaborn  
- scikit-learn  
- scipy  

---

### 2. Create and Prepare Dataset
A synthetic dataset is created using `make_classification()` from scikit-learn.  
It simulates a binary classification problem with multiple informative and redundant features to represent realistic data complexity.

---

### 3. Data Splitting
The dataset is split into **training** and **test** sets (80–20) using `train_test_split()` to ensure unbiased model evaluation.

---

### 4. Train Multiple Models
Three machine learning algorithms are trained:
- Logistic Regression  
- Random Forest Classifier  
- Support Vector Machine (SVM)  

Each model is fit on the training data and later tested for performance.

---

### 5. Model Comparison
All models are compared using **accuracy**.  
Results are stored in a dataframe and visualized as a bar chart for easier interpretation.

---

### 6. Visualizing Accuracy Comparison
A bar chart displays model performance side-by-side.  
This helps identify which algorithm gives the best initial accuracy.

---

### 7. Model Evaluation
Each model is further evaluated using:
- **Confusion Matrix** — shows True Positives, True Negatives, False Positives, and False Negatives.  
- **AUC Score** — measures how well the model separates classes.  
- **ROC Curve** — visualizes the trade-off between sensitivity and specificity.

The best model (based on accuracy or AUC) is highlighted and visualized using a confusion matrix heatmap.

---

### 8. Outlier Detection & Normalization
Before fine-tuning, data is refined using statistical methods.

**Steps performed:**
1. **Outlier Detection:**  
   - Z-score method identifies points beyond 3 standard deviations.  
   - A summary table shows outlier count per feature.

2. **Data Visualization (Before Normalization):**  
   - Histograms reveal skewness and uneven distributions.

3. **Outlier Removal:**  
   - Removes extreme data points beyond acceptable Z-scores.  
   - Keeps only clean, meaningful samples.

4. **Normalization:**  
   - Uses `StandardScaler` to scale all features (mean = 0, std = 1).  
   - Makes data more Gaussian-like.

5. **Data Visualization (After Normalization):**  
   - Histograms show smoother, centered distributions.

6. **Skewness Check:**  
   - Compares skewness before and after normalization.  
   - A drop in skew value indicates better, more balanced data.

---

## Results
At the end of this notebook:
- Multiple ML models are trained and compared.
- Accuracy, AUC, and confusion matrices are visualized.
- Data is cleaned and normalized statistically.
- A strong foundation for model optimization and tuning is set.

---

## Key Learnings
- Always visualize and statistically analyze data before training.  
- Removing or normalizing outliers improves model consistency.  
- AUC and confusion matrices provide more insight than accuracy alone.  
- Comparing multiple models gives a fair idea of performance trends.

---

## Tech Stack
- **Language:** Python  
- **Libraries:** numpy, pandas, matplotlib, seaborn, scikit-learn, scipy  

---

## How to Run
1. Clone or download this repository.  
2. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn scipy


Author: Sarveshwararam M
Focus: Practical AI/ML workflow with differential statistics and model evaluation
