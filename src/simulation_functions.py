"""
simulation_functions.py

Reusable functions for the project:

Robustness of Statistical and Machine-Learning
Regression Models under Violations of Classical
Regression Assumptions.

Models:
    - OLS
    - Ridge
    - Huber
    - Random Forest
    - XGBoost

Data-generating processes:
    - Baseline
    - Heteroscedasticity
    - Outliers
    - Multicollinearity
    - Nonlinearity

Author:
    Maryam Sattar
"""

import numpy as np
import pandas as pd

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    HuberRegressor
)

from sklearn.ensemble import (
    RandomForestRegressor
)

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from xgboost import XGBRegressor


# ============================================================
# 1. MODEL DEFINITIONS
# ============================================================

def get_models():
    """
    Return the five regression models used in the study.

    Returns
    -------
    dict
        Dictionary containing model names and model objects.
    """

    models = {

        "OLS": LinearRegression(),

        "Ridge": Ridge(
            alpha=1.0
        ),

        "Huber": HuberRegressor(
            max_iter=1000
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ),

        "XGBoost": XGBRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1,
            objective="reg:squarederror"
        )
    }

    return models


# ============================================================
# 2. MODEL EVALUATION
# ============================================================

def evaluate_models(
    X_train,
    X_test,
    y_train,
    y_test
):
    """
    Train all five models and calculate predictive metrics.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training predictors.

    X_test : pandas.DataFrame
        Test predictors.

    y_train : pandas.Series
        Training response.

    y_test : pandas.Series
        Test response.

    Returns
    -------
    pandas.DataFrame
        RMSE, MAE and R2 for each model.
    """

    models = get_models()

    results = []

    for name, model in models.items():

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        r2 = r2_score(
            y_test,
            predictions
        )

        results.append({

            "Model": name,

            "RMSE": rmse,

            "MAE": mae,

            "R2": r2
        })

    return pd.DataFrame(results)


# ============================================================
# 3. BASELINE DATA
# ============================================================

def generate_baseline_data(
    n=1000,
    seed=None
):
    """
    Generate baseline data satisfying the classical
    linear regression structure.

    Data-generating process:

        Y = 3 + 2X1 + 1.5X2 - X3 + epsilon

    where:

        X1, X2, X3 ~ N(0, 1)
        epsilon ~ N(0, 1)
    """

    if seed is not None:
        np.random.seed(seed)

    X1 = np.random.normal(
        0,
        1,
        n
    )

    X2 = np.random.normal(
        0,
        1,
        n
    )

    X3 = np.random.normal(
        0,
        1,
        n
    )

    error = np.random.normal(
        0,
        1,
        n
    )

    Y = (
        3
        + 2 * X1
        + 1.5 * X2
        - X3
        + error
    )

    return pd.DataFrame({

        "X1": X1,

        "X2": X2,

        "X3": X3,

        "Y": Y
    })


# ============================================================
# 4. HETEROSCEDASTIC DATA
# ============================================================

def generate_heteroscedastic_data(
    n=1000,
    seed=None
):
    """
    Generate data with heteroscedastic errors.

    Error standard deviation depends on |X1|:

        sigma_i = 0.5 + 1.5|X1_i|

    Therefore, the error variance changes
    across observations.
    """

    if seed is not None:
        np.random.seed(seed)

    X1 = np.random.normal(
        0,
        1,
        n
    )

    X2 = np.random.normal(
        0,
        1,
        n
    )

    X3 = np.random.normal(
        0,
        1,
        n
    )

    error_sd = (
        0.5
        + 1.5 * np.abs(X1)
    )

    error = np.random.normal(
        0,
        error_sd,
        n
    )

    Y = (
        3
        + 2 * X1
        + 1.5 * X2
        - X3
        + error
    )

    return pd.DataFrame({

        "X1": X1,

        "X2": X2,

        "X3": X3,

        "Y": Y
    })


# ============================================================
# 5. OUTLIER DATA
# ============================================================

def create_outlier_data(
    outlier_percentage,
    n=1000,
    seed=42
):
    """
    Generate data with response-variable outliers.

    Parameters
    ----------
    outlier_percentage : float
        Proportion of observations contaminated.

        Examples:
            0.00 = 0%
            0.02 = 2%
            0.05 = 5%
            0.10 = 10%

    n : int
        Number of observations.

    seed : int
        Random seed.

    Notes
    -----
    Outliers are introduced by adding large random
    deviations to Y.
    """

    np.random.seed(seed)

    X1 = np.random.normal(
        0,
        1,
        n
    )

    X2 = np.random.normal(
        0,
        1,
        n
    )

    X3 = np.random.normal(
        0,
        1,
        n
    )

    error = np.random.normal(
        0,
        1,
        n
    )

    Y = (
        3
        + 2 * X1
        + 1.5 * X2
        - X3
        + error
    )

    number_of_outliers = int(
        outlier_percentage * n
    )

    if number_of_outliers > 0:

        indices = np.random.choice(
            n,
            size=number_of_outliers,
            replace=False
        )

        Y[indices] += np.random.normal(
            0,
            15,
            number_of_outliers
        )

    return pd.DataFrame({

        "X1": X1,

        "X2": X2,

        "X3": X3,

        "Y": Y
    })


# ============================================================
# 6. MULTICOLLINEAR DATA
# ============================================================

def generate_multicollinear_data(
    n=1000,
    seed=None
):
    """
    Generate data with strong correlation between X1 and X2.

        X2 = X1 + small random noise

    The outcome follows:

        Y = 3 + 2X1 + 1.5X2 - X3 + epsilon
    """

    if seed is not None:
        np.random.seed(seed)

    X1 = np.random.normal(
        0,
        1,
        n
    )

    X2 = (
        X1
        + np.random.normal(
            0,
            0.1,
            n
        )
    )

    X3 = np.random.normal(
        0,
        1,
        n
    )

    error = np.random.normal(
        0,
        1,
        n
    )

    Y = (
        3
        + 2 * X1
        + 1.5 * X2
        - X3
        + error
    )

    return pd.DataFrame({

        "X1": X1,

        "X2": X2,

        "X3": X3,

        "Y": Y
    })


# ============================================================
# 7. NONLINEAR DATA
# ============================================================

def generate_nonlinear_data(
    n=1000,
    seed=None
):
    """
    Generate data with a nonlinear relationship.

        Y = 3 + 2X1^2 + 1.5X2 - X3 + epsilon
    """

    if seed is not None:
        np.random.seed(seed)

    X1 = np.random.normal(
        0,
        1,
        n
    )

    X2 = np.random.normal(
        0,
        1,
        n
    )

    X3 = np.random.normal(
        0,
        1,
        n
    )

    error = np.random.normal(
        0,
        1,
        n
    )

    Y = (
        3
        + 2 * (X1 ** 2)
        + 1.5 * X2
        - X3
        + error
    )

    return pd.DataFrame({

        "X1": X1,

        "X2": X2,

        "X3": X3,

        "Y": Y
    })


# ============================================================
# 8. TRAIN/TEST SPLIT
# ============================================================

def prepare_train_test(
    data,
    target="Y",
    test_size=0.20,
    random_state=42
):
    """
    Separate predictors and target and create a train/test split.
    """

    from sklearn.model_selection import train_test_split

    X = data.drop(
        columns=[target]
    )

    y = data[target]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )


# ============================================================
# 9. MULTICOLLINEARITY COEFFICIENT ANALYSIS
# ============================================================

def get_linear_model_coefficients(
    X_train,
    y_train
):
    """
    Compare OLS and Ridge regression coefficients.

    Useful for studying coefficient instability
    under multicollinearity.
    """

    ols = LinearRegression()

    ridge = Ridge(
        alpha=1.0
    )

    ols.fit(
        X_train,
        y_train
    )

    ridge.fit(
        X_train,
        y_train
    )

    coefficient_results = pd.DataFrame({

        "Variable": X_train.columns,

        "OLS": ols.coef_,

        "Ridge": ridge.coef_
    })

    return coefficient_results


# ============================================================
# 10. REPEATED BASELINE EXPERIMENT
# ============================================================

def run_baseline_experiment(
    seed,
    n=1000,
    test_size=0.20
):
    """
    Run one complete baseline simulation.
    """

    data = generate_baseline_data(
        n=n,
        seed=seed
    )

    X_train, X_test, y_train, y_test = (
        prepare_train_test(
            data,
            target="Y",
            test_size=test_size,
            random_state=seed
        )
    )

    results = evaluate_models(
        X_train,
        X_test,
        y_train,
        y_test
    )

    results["Seed"] = seed

    return results


# ============================================================
# 11. REPEATED BASELINE SIMULATIONS
# ============================================================

def run_repeated_baseline(
    n_simulations=30,
    n=1000
):
    """
    Run multiple baseline simulations and return
    both individual and summary results.
    """

    all_results = []

    for seed in range(n_simulations):

        results = run_baseline_experiment(
            seed=seed,
            n=n
        )

        all_results.append(
            results
        )

    repeated_results = pd.concat(
        all_results,
        ignore_index=True
    )

    mean_results = (
        repeated_results
        .groupby("Model")[[
            "RMSE",
            "MAE",
            "R2"
        ]]
        .mean()
        .reset_index()
    )

    std_results = (
        repeated_results
        .groupby("Model")[[
            "RMSE",
            "MAE",
            "R2"
        ]]
        .std()
        .reset_index()
    )

    summary_results = mean_results.copy()

    summary_results["RMSE_SD"] = (
        std_results["RMSE"]
    )

    summary_results["MAE_SD"] = (
        std_results["MAE"]
    )

    summary_results["R2_SD"] = (
        std_results["R2"]
    )

    return (
        repeated_results,
        summary_results
    )


# ============================================================
# 12. OUTLIER EXPERIMENT
# ============================================================

def run_outlier_experiment(
    outlier_percentage,
    seed,
    n=1000,
    test_size=0.20
):
    """
    Run one outlier experiment.
    """

    data = create_outlier_data(
        outlier_percentage=outlier_percentage,
        n=n,
        seed=42
    )

    X_train, X_test, y_train, y_test = (
        prepare_train_test(
            data,
            target="Y",
            test_size=test_size,
            random_state=seed
        )
    )

    results = evaluate_models(
        X_train,
        X_test,
        y_train,
        y_test
    )

    results["Outlier_Percentage"] = (
        outlier_percentage * 100
    )

    results["Seed"] = seed

    return results


# ============================================================
# 13. REPEATED OUTLIER EXPERIMENT
# ============================================================

def run_repeated_outlier_experiment(
    outlier_levels=None,
    n_simulations=10,
    n=1000
):
    """
    Run repeated experiments for several
    outlier contamination levels.
    """

    if outlier_levels is None:

        outlier_levels = [
            0.00,
            0.02,
            0.05,
            0.10
        ]

    all_results = []

    for level in outlier_levels:

        for seed in range(n_simulations):

            results = run_outlier_experiment(
                outlier_percentage=level,
                seed=seed,
                n=n
            )

            all_results.append(
                results
            )

    repeated_results = pd.concat(
        all_results,
        ignore_index=True
    )

    summary_results = (
        repeated_results
        .groupby([
            "Outlier_Percentage",
            "Model"
        ])[
            ["RMSE", "MAE", "R2"]
        ]
        .mean()
        .reset_index()
    )

    return (
        repeated_results,
        summary_results
    )