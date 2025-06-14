**Dajubhai MiniMart Analysis**

# 📊 ITS69304 Individual Assignment — Suprem Khatri

This repository contains the **Jupyter Notebook** for dajubhai minimart analysis. The notebook demonstrates a full data science workflow: from data cleaning to model building using a dataset named `DajuBhaiMinimart.csv`.

---

## 📂 Files
- `ITS69304_SupremKhatri_Individual_Assignment.ipynb` — Main notebook with code and analysis.
- `DajuBhaiMinimart.csv` — Dataset (assumed provided externally).

---

## 📝 Notebook Overview

### 1️⃣ **Importing Libraries**
The following libraries are used:
```python
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import plotly.io as pio
pio.renderers.default = 'notebook'
```

---

### 2️⃣ **Data Loading**
```python
df = pd.read_csv('DajuBhaiMinimart.csv')
df.head()
```
- Reads and displays initial data.

---

### 3️⃣ **Data Cleaning**
- Checks data info:
  ```python
  df.info()
  ```
- Checks for missing values:
  ```python
  df.isnull().sum()
  ```
- Drops unnecessary columns:
  ```python
  df.drop('CustomerID', axis=1, inplace=True)
  ```
- Renames columns where necessary.

---

### 4️⃣ **Data Visualization**
- Uses **Seaborn** and **Plotly** to visualize relationships and distributions.
- Example plots:
  ```python
  sns.countplot(x='ProductCategory', data=df)
  px.histogram(df, x='Sales', color='ProductCategory')
  ```

---

### 5️⃣ **Feature Engineering**
- Creates or modifies features to improve model input.
- Handles categorical encoding, scaling, etc.

---

### 6️⃣ **Model Training**
- Trains models such as:
  - Gradient Boosting Regressor
- Example code:
  ```python
  from sklearn.ensemble import GradientBoostingRegressor
  model = GradientBoostingRegressor()
  model.fit(X_train, y_train)
  ```

---

### 7️⃣ **Hyperparameter Tuning & Cross-Validation**
- Performs grid search / cross-validation:
  ```python
  from sklearn.model_selection import GridSearchCV
  params = {'n_estimators': [100, 200], 'learning_rate': [0.1, 0.05]}
  grid = GridSearchCV(GradientBoostingRegressor(), params, cv=5)
  grid.fit(X_train, y_train)
  ```

---

## 🚀 How to Run

✅ Clone or download this repository.

✅ Install requirements:
```bash
pip install pandas matplotlib seaborn plotly scikit-learn
```

✅ Run the notebook:
```bash
jupyter notebook ITS69304_SupremKhatri_Individual_Assignment.ipynb
```

---

## ⚠️ Notes

- Ensure `DajuBhaiMinimart.csv` is present in the notebook directory.
- All visualizations are rendered using Plotly + Seaborn + Matplotlib.
- The notebook suppresses warnings for cleaner output.
---

## 💡 Future Work
- Extend analysis with additional models (e.g. XGBoost, Random Forest).
- Improve feature engineering with domain knowledge.
- Deploy model via Flask / Streamlit for interactive use.
