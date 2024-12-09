# phase-3---Telcom---Churn---Prediction
To create a **README-style document** summarizing each step of the uploaded Jupyter Notebook (`phase 3.ipynb`), I'll first review the file to extract its key steps. Let me process the notebook to provide a detailed summary.

### README Summary for "Phase 3" Notebook

This README outlines the steps, methodologies, and objectives followed in the Jupyter Notebook.

---

#### **1. Business Understanding**
- **Objective**: Develop a predictive model to address customer churn for SyriaTel, minimizing financial losses and improving retention strategies.
- **Significance**: Customer attrition impacts revenue and market reputation. This model aims to identify at-risk customers for timely interventions.

---

#### **2. Data Understanding**
- **Dataset Overview**: Data from Kaggle, with 3,333 records and 21 features covering customer demographics, service usage, and churn labels.
- **Importance**: The data provides insights into customer behavior, enabling the development of actionable churn predictions.

---

#### **3. Environment Setup**
- **Tools Used**:
  - Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, and imbalanced-learn.
  - Gradient Boosting Frameworks: XGBoost or LightGBM.
  - SHAP for interpretability.
- **Development Environment**: Jupyter Notebook with GitHub for version control.

---

#### **4. Steps in the Notebook**

1. **Data Import and Initial Exploration**:
   - Loaded dataset (`bigml_59c28831336c6604c800002a.csv`).
   - Performed `.info()` and `.head()` for structure and preview.

2. **Data Preprocessing**:
   - Encoded categorical variables using label encoding (e.g., "international plan" and "voice mail plan").
   - Scaled numerical features using `StandardScaler` for model compatibility.
   - Split data into training and test sets (80-20 split).

3. **Class Imbalance Handling**:
   - Used **SMOTE-Tomek** for oversampling and cleaning the minority class, balancing the dataset effectively.

4. **Model Training**:
   - **Models Used**:
     - Logistic Regression
     - Decision Trees
     - Random Forest Classifier
   - Hyperparameter tuning using `GridSearchCV` for optimal performance.

5. **Model Evaluation**:
   - Metrics computed: Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
   - Confusion matrix visualized to understand class predictions.

6. **Advanced Techniques**:
   - Gradient boosting techniques (e.g., XGBoost) were explored to enhance prediction accuracy and ROC-AUC performance.

---

#### **5. Observations and Outcomes**
- **Initial Results**: Logistic Regression faced convergence warnings due to feature scaling or imbalanced data.
- **Improvements**:
  - SMOTE-Tomek improved model performance by addressing imbalance.
  - Decision Trees and Random Forests performed well but required hyperparameter tuning.
- **Evaluation Highlights**:
  - Precision and recall for "Churn" class were key focus areas.
  - ROC Curve plotted to assess classifier discrimination.

---

#### **6. Recommendations**
- Adopt the model with the highest ROC-AUC and balanced metrics.
- Deploy the model in production for real-time churn prediction.
- Periodically retrain the model as customer behavior evolves.

---

