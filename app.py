import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline

# ---------------------------------------------------------
# Create the original dataset
# ---------------------------------------------------------

data = {
    "Advertising_Spend": [2000, 3000, 2500, 4000, 3500,
                          None, 5000, 4500, 3000, 3800],
    "Store_Size": [1500, 2000, 1800, 2200, None,
                   2100, 2500, 2400, 2000, 2300],
    "Customers": [200, 250, 230, 300, 280,
                  260, 320, 310, 270, None],
    "Promotion": ["Yes", "No", "Yes", "Yes", "No",
                  "No", "Yes", "Yes", "No", "Yes"],
    "Sales": [40000, 50000, 45000, 60000, 52000,
              48000, 65000, 63000, 51000, 59000]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

# ---------------------------------------------------------
# Question 2: Data preprocessing
# ---------------------------------------------------------

numerical_features = [
    "Advertising_Spend",
    "Store_Size",
    "Customers"
]

# Median imputation is used because:
# 1. The dataset is very small, so removing rows would lose
#    too much information.
# 2. The median is less affected by extreme values than mean.
for column in numerical_features:
    df[column] = df[column].fillna(df[column].median())

# Convert categorical Promotion values to numerical values.
df["Promotion"] = df["Promotion"].map({"No": 0, "Yes": 1})

# Standardize numerical predictor variables.
scaler = StandardScaler()

df_scaled = df.copy()

df_scaled[numerical_features] = scaler.fit_transform(
    df[numerical_features]
)

print("\nPreprocessed and Scaled Dataset:")
print(df_scaled.round(3))

# ---------------------------------------------------------
# Question 3: Build regression models using full dataset
# ---------------------------------------------------------

y = df_scaled["Sales"]

# 1. Simple Linear Regression
X_simple = df_scaled[["Advertising_Spend"]]

simple_model = LinearRegression()
simple_model.fit(X_simple, y)

simple_predictions = simple_model.predict(X_simple)

# 2. Multiple Linear Regression
X_multiple = df_scaled[
    ["Advertising_Spend", "Store_Size",
     "Customers", "Promotion"]
]

multiple_model = LinearRegression()
multiple_model.fit(X_multiple, y)

multiple_predictions = multiple_model.predict(X_multiple)

# 3. Polynomial Regression, degree 2
poly_features = PolynomialFeatures(
    degree=2,
    include_bias=False
)

X_poly = poly_features.fit_transform(X_simple)

polynomial_model = LinearRegression()
polynomial_model.fit(X_poly, y)

polynomial_predictions = polynomial_model.predict(X_poly)

# ---------------------------------------------------------
# Predict sales for the new data point
# ---------------------------------------------------------

new_store = pd.DataFrame({
    "Advertising_Spend": [4200],
    "Store_Size": [2100],
    "Customers": [290],
    "Promotion": [1]
})

# Apply SAME scaler used on training data.
new_store_scaled = new_store.copy()

new_store_scaled[numerical_features] = scaler.transform(
    new_store[numerical_features]
)

new_simple = simple_model.predict(
    new_store_scaled[["Advertising_Spend"]]
)

new_multiple = multiple_model.predict(
    new_store_scaled[
        ["Advertising_Spend", "Store_Size",
         "Customers", "Promotion"]
    ]
)

new_poly_values = poly_features.transform(
    new_store_scaled[["Advertising_Spend"]]
)

new_polynomial = polynomial_model.predict(new_poly_values)

print("\nPredicted Sales for New Store:")
print(f"Simple Linear: ${new_simple[0]:,.2f}")
print(f"Multiple Linear: ${new_multiple[0]:,.2f}")
print(f"Polynomial: ${new_polynomial[0]:,.2f}")

# ---------------------------------------------------------
# Plot actual versus predicted values
# ---------------------------------------------------------

models_to_plot = {
    "Simple Linear Regression": simple_predictions,
    "Multiple Linear Regression": multiple_predictions,
    "Polynomial Regression": polynomial_predictions
}

for title, predictions in models_to_plot.items():

    plt.figure(figsize=(7, 5))

    plt.scatter(y, predictions)

    minimum = min(y.min(), predictions.min())
    maximum = max(y.max(), predictions.max())

    # Perfect prediction reference line.
    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        "--"
    )

    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.title(f"Actual vs Predicted - {title}")

    plt.show()

# ---------------------------------------------------------
# Question 4: Training/testing evaluation
# ---------------------------------------------------------

# Use unscaled cleaned data.
X = df[
    ["Advertising_Spend", "Store_Size",
     "Customers", "Promotion"]
]

y = df["Sales"]

# With only 10 observations, 40% testing gives four test
# observations instead of an extremely small two-row set.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.40,
    random_state=42
)

# Simple linear regression.
simple_test_model = LinearRegression()

simple_test_model.fit(
    X_train[["Advertising_Spend"]],
    y_train
)

# Multiple regression.
multiple_test_model = make_pipeline(
    StandardScaler(),
    LinearRegression()
)

multiple_test_model.fit(X_train, y_train)

# Polynomial regression.
polynomial_test_model = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(
        degree=2,
        include_bias=False
    ),
    LinearRegression()
)

polynomial_test_model.fit(
    X_train[["Advertising_Spend"]],
    y_train
)

# Ridge Regression.
ridge_model = make_pipeline(
    StandardScaler(),
    Ridge(alpha=1.0)
)

ridge_model.fit(X_train, y_train)

# Lasso Regression.
lasso_model = make_pipeline(
    StandardScaler(),
    Lasso(
        alpha=100,
        max_iter=100000
    )
)

lasso_model.fit(X_train, y_train)

# ---------------------------------------------------------
# Function for model evaluation
# ---------------------------------------------------------

def evaluate_model(model_name, model,
                   x_train, x_test):

    train_predictions = model.predict(x_train)
    test_predictions = model.predict(x_test)

    train_rmse = np.sqrt(
        mean_squared_error(
            y_train,
            train_predictions
        )
    )

    test_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            test_predictions
        )
    )

    test_mae = mean_absolute_error(
        y_test,
        test_predictions
    )

    test_r2 = r2_score(
        y_test,
        test_predictions
    )

    return {
        "Model": model_name,
        "Train RMSE": train_rmse,
        "Test RMSE": test_rmse,
        "Test MAE": test_mae,
        "Test R2": test_r2
    }

results = []

results.append(
    evaluate_model(
        "Simple Linear",
        simple_test_model,
        X_train[["Advertising_Spend"]],
        X_test[["Advertising_Spend"]]
    )
)

results.append(
    evaluate_model(
        "Multiple Linear",
        multiple_test_model,
        X_train,
        X_test
    )
)

results.append(
    evaluate_model(
        "Polynomial Degree 2",
        polynomial_test_model,
        X_train[["Advertising_Spend"]],
        X_test[["Advertising_Spend"]]
    )
)

results.append(
    evaluate_model(
        "Ridge",
        ridge_model,
        X_train,
        X_test
    )
)

results.append(
    evaluate_model(
        "Lasso",
        lasso_model,
        X_train,
        X_test
    )
)

results_df = pd.DataFrame(results)

print("\nModel Evaluation:")
print(results_df.round(3))