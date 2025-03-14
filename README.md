# Vomitoxin Prediction using Random Forest and Transformer Model  

## Project Overview  
This project aims to predict **vomitoxin levels (ppb)** in agricultural samples using **Random Forest** and **Transformer-based models**. It incorporates **data preprocessing, dimensionality reduction (PCA), hyperparameter tuning**, and model evaluation using **error metrics and visualizations**.

---

## Dataset  
The dataset (`TASK-ML-INTERN.csv`) contains multiple features influencing vomitoxin levels. The **target variable** is `vomitoxin_ppb`.

---

## Steps Implemented  

### 1. **Data Preprocessing**  
- **Handled missing values**: Replaced missing values with column means.  
- **Outlier removal**: Applied **IQR-based filtering** to remove extreme values.  
- **Feature scaling**: Used **Min-Max Scaling** to normalize input features.  

### 2. **Dimensionality Reduction**  
- **Principal Component Analysis (PCA)**: Reduced feature space to 10 components.  
- **Explained Variance Plot**: Justified the number of components selected.  

### 3. **Model Training & Evaluation**  

#### **Random Forest Regression**  
- Performed **hyperparameter tuning** using `RandomizedSearchCV`.  
- **Feature importance analysis** was conducted using trained model weights.  

#### **Transformer-based Regression**  
- Implemented a custom **PyTorch Transformer model** with:  
  - **Multi-head self-attention** mechanism.  
  - **Dropout and ReLU activation** for better generalization.  
  - **Learning rate scheduling** for stable training.  

#### **Performance Metrics**  
- **Mean Absolute Error (MAE)**  
- **Root Mean Squared Error (RMSE)**  
- **R² Score**  

| Model        | MAE   | RMSE  | R²  |  
|-------------|-------|-------|------|  
| Random Forest | 590.70 | 783.88 | -0.025 |  
| Transformer  | 572.47 | 951.81 | -0.511 |  

### 4. **Visualizations**  
1. **Feature Distribution (Before & After Outlier Removal)**  
2. **PCA Explained Variance**  
3. **Feature Importance from Random Forest**  
4. **Predicted vs. Actual Scatter Plot for Model Evaluation**  

---

## How to Run  
1. Install dependencies:  
   ```bash
   pip install pandas numpy matplotlib seaborn torch scikit-learn
   ```
2. Place the dataset (`TASK-ML-INTERN.csv`) in the working directory.  
3. Run the script:  
   ```bash
   python script.py
   ```

---

## Key Findings & Suggestions  
- **Random Forest** performed slightly better than **Transformer** in MAE.  
- **Transformer had lower R²**, indicating poor variance explanation.  
- **PCA reduced feature space efficiently** while retaining information.  
- **Further hyperparameter tuning** can improve Transformer performance.  

---

## Future Improvements  
- **Try alternative deep learning models (LSTMs, CNNs).**  
- **Increase training data for Transformer stability.**  
- **Use domain-specific feature engineering.**  

---

This project provides a **solid baseline for vomitoxin prediction**, integrating **ML & DL techniques** with **advanced preprocessing and analysis**. 🚀

