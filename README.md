# Boston Housing Price Prediction

## Project Overview

This project uses the Boston Housing dataset to build a **Multiple Linear Regression** model for predicting housing prices based on different features describing housing and socioeconomic conditions.

The project demonstrates a basic machine learning workflow, including:

* Data exploration
* Data visualization
* Feature and target selection
* Train/test splitting
* Multiple Linear Regression
* Model evaluation
* Interpretation of model performance

## Dataset

The dataset contains information about housing in the Boston area.

The features are used as independent variables (`X`), while the housing price is used as the dependent variable (`y`).

The dataset used in this project is stored in:

```text
regression-datasets-housing.csv
```

## Project Structure

```text
Python_projects/
│
├── Boston Housing Project.ipynb
├── regression-datasets-housing.csv
├── README.md
└── .gitignore
```

The `.ipynb_checkpoints/` directory is not included in version control because it contains automatically generated Jupyter Notebook checkpoint files.

## Methodology

### 1. Feature and Target Selection

The dataset is divided into:

* `X` — independent variables/features
* `y` — dependent variable/target

Example:

```python
X = boston[:, :-1]
y = boston[:, -1]
```

The last column is used as the target, while all preceding columns are used as features.

### 2. Train-Test Split

The data is divided into:

* **80% training data**
* **20% testing data**

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)
```

### 3. Multiple Linear Regression

A Linear Regression model is trained using the training data:

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)
```

The model is then used to predict housing prices for the test dataset:

```python
y_pred = model.predict(X_test)
```

## Model Evaluation

The model is evaluated using the **R² score**.

```python
print("Training R²: {:.2f}".format(model.score(X_train, y_train)))
print("Testing R²: {:.2f}".format(model.score(X_test, y_test)))
```

### Results

The current model produced:

| Dataset  | R² Score |
| -------- | -------: |
| Training |     0.75 |
| Testing  |     0.67 |

The training R² of **0.75** indicates that the model explains approximately 75% of the variation in the training data.

The testing R² of **0.67** indicates that the model explains approximately 67% of the variation in unseen test data.

The difference between the training and testing scores is relatively moderate, suggesting that the model does not show severe overfitting.

## Visualizations

The project can include visualizations such as:


* Distribution of housing prices
* Distribution of features

These visualizations help explore the relationships in the data and evaluate the performance of the regression model.

## Technologies Used

* Python
* Jupyter Notebook
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

## Conclusion

A Multiple Linear Regression model was developed to predict Boston housing prices using multiple features.

The model achieved an R² score of **0.75 on the training data** and **0.67 on the testing data**. The results indicate that the model provides a reasonable predictive relationship between the selected housing features and the target variable, while also generalizing reasonably well to unseen data.

## Future Improvements

Possible improvements include:

* Feature selection using `SelectKBest`
* Feature scaling
* Testing other regression algorithms
* Hyperparameter tuning
* Cross-validation
* Comparing additional evaluation metrics such as Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
* Further analysis of regression assumptions and residuals
