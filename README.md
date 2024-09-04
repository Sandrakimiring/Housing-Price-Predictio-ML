# California Housing Prices Prediction using ML
A machine learning model predicting house prices based on features like location, size, and age using linear regression. The model preprocesses data for improved accuracy and aims to assist in real estate valuation.

The project involves developing a machine learning model to predict housing prices in California based on a variety of features. The dataset used for this project is the **California Housing Prices Dataset** from Kaggle. The goal is to create a reliable model that accurately estimates house prices given specific input data.

---

## Project Overview

- **Dataset:** [California Housing Prices Dataset](https://www.kaggle.com/camnugent/california-housing-prices)
- **Project Goal:** To predict housing prices in California using machine learning techniques.
- **Model Performance:** Achieved an R-squared score of 85.00% on the test data.

---

## Repository Structure

- **`notebooks/`**
  - **`california_housing_prediction.ipynb:`** Jupyter Notebook containing the entire workflow, from data preprocessing and exploratory data analysis to model training and prediction.
  
- **`models/`**
  - **`random_forest_model.pkl:`** Serialized model using the RandomForestRegressor for predictions.

- **`data/`**
  - **`train.csv:`** Training data used for model development.
  - **`test.csv:`** Test data used for generating predictions.
  - **`submission.csv:`** Final predictions on the test data.

---

## Tools and Libraries Used

- **Pandas:** For data manipulation and preprocessing.
- **NumPy:** For numerical operations and array handling.
- **Matplotlib/Seaborn:** For data visualization and exploratory data analysis.
- **Scikit-learn:** For building and evaluating machine learning models.
- **XGBoost:** For advanced boosting algorithms used in model training.

---

## Steps Followed in the Project

### 1. Data Loading and Exploration
- The dataset is loaded into Pandas DataFrames for analysis.
- Key features of the data, such as distributions, missing values, and correlations, are explored through visualizations and summary statistics.

### 2. Data Preprocessing
- **Missing Values:** Addressed using imputation or by removing features with excessive missing data.
- **Feature Engineering:** Created new features based on existing ones to capture more information.
- **Encoding:** Categorical variables are converted to numerical values using one-hot encoding.
- **Scaling:** Numerical features are standardized to ensure consistency across features.

### 3. Model Selection and Training
- Several machine learning models were considered, including:
  - **Linear Regression**
  - **DecisionTreeRegressor**
  - **RandomForestRegressor**
  - **GradientBoostingRegressor**
  - **XGBRegressor**
  
- **Cross-Validation:** Used to evaluate the performance of each model, with **RandomForestRegressor** being selected for its accuracy and robustness.

### 4. Model Evaluation and Prediction
- **Training:** The RandomForestRegressor model was trained on the training dataset.
- **Prediction:** The model was used to predict housing prices on the test dataset, with results saved to `submission.csv`.

### 5. Model Saving
- The trained RandomForestRegressor model was saved as a pickle file (`random_forest_model.pkl`) for future use.

---

## How to Use This Project

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Sandrakimiring/california-housing-prediction.git
