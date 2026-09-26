Robustness of Statistical and Machine-Learning Regression Models
Overview
This project investigates how statistical and machine-learning regression models behave when classical regression assumptions are violated.
The study uses controlled simulation experiments to examine model performance under:
Baseline conditions
Heteroscedasticity
Outliers
Multicollinearity
Nonlinearity
A real-world application using the California Housing dataset is also included.
The project compares five regression models:
1.Ordinary Least Squares (OLS)
2.Ridge Regression
3.Huber Regression
4.Random Forest
5.XGBoost
Research Question
How do different regression and machine-learning models behave when classical regression assumptions are progressively violated, and how does their predictive performance change across different data-generating conditions?
Objectives
The main objectives are to:
Establish a baseline comparison of statistical and machine-learning regression models.
Examine model behavior under heteroscedasticity.
Investigate the effect of increasing outlier contamination.
Study model performance under multicollinearity.
Evaluate model behavior under nonlinear relationships.
Compare simulation results with a real-world housing dataset.
Identify conditions under which model performance changes substantially.
Models
OLS
Ordinary Least Squares provides the classical linear regression benchmark.
Ridge Regression
Ridge Regression applies L2 regularization and is included to investigate performance when predictors are correlated.
Huber Regression
Huber Regression uses a robust loss function designed to reduce the influence of large residuals.
Random Forest
Random Forest is an ensemble tree-based method capable of representing nonlinear relationships and interactions.
XGBoost
XGBoost is a gradient-boosting tree-based method designed for flexible predictive modelling.
Experimental Design
The simulation study uses controlled data-generating processes.
Baseline
The baseline response is generated using:
Y = 3 + 2X1 + 1.5X2 - X3 + error
Heteroscedasticity
The error variance changes according to the magnitude of X1.
Outliers
Response contamination is introduced at:
0%, 2%, 5%, and 10%
Multicollinearity
Correlated predictors are generated so that one predictor is strongly related to another.
Nonlinearity
A quadratic relationship is introduced through:
Y = 3 + 2(X1²) + 1.5X2 - X3 + error
Evaluation Metrics
The models are evaluated using:
RMSE: Root Mean Squared Error
MAE: Mean Absolute Error
R²: Coefficient of Determination
Lower RMSE and MAE indicate lower prediction error, while higher R² indicates greater explained variation.
Main Findings
The experiments demonstrate that model performance depends on the underlying data-generating process.
Baseline
OLS, Ridge, and Huber produced very similar performance. Random Forest and XGBoost produced somewhat higher prediction errors under the reported baseline setting.
Heteroscedasticity
OLS, Ridge, and Huber remained relatively close. Random Forest and XGBoost produced higher errors in the reported experiment.
Outliers
RMSE increased for all five models as outlier contamination increased from 0% to 10%.
Huber produced slightly lower RMSE than OLS and Ridge at each contamination level, but the advantage was modest.
Multicollinearity
OLS, Ridge, and Huber again produced very similar results. Ridge had slightly lower RMSE and MAE and slightly higher R² in the reported experiment.
Nonlinearity
The nonlinear simulation did not result in Random Forest or XGBoost outperforming the linear models. This demonstrates that model performance depends on the specific nonlinear data-generating mechanism rather than on nonlinearity alone.
California Housing
On the California Housing dataset, Random Forest produced the lowest RMSE and MAE and the highest R² among the five tested models. XGBoost also performed better than OLS, Ridge, and Huber on the reported test set.
California Housing Results
Model	RMSE	MAE	R²
OLS	69,297.72	50,413.43	0.6488
Ridge	69,250.31	50,391.14	0.6493
Huber	76,534.64	55,733.46	0.5717
Random Forest	48,757.92	31,663.84	0.8262
XGBoost	56,343.41	38,938.38	0.7679
Outlier Results
RMSE under increasing response-outlier contamination:
Outliers	OLS	Ridge	Huber	Random Forest	XGBoost
0%	1.0463	1.0463	1.0463	1.2148	1.1458
2%	2.4340	2.4340	2.4259	2.6114	2.5737
5%	3.2668	3.2667	3.2519	3.4519	3.4053
10%	4.7499	4.7499	4.7302	5.0150	4.9404
Repository Structure
ML-Regression-Model-Comparison/
│
├── data/
│   └── housing.csv
│
├── notebooks/
│   ├── 01_baseline_simulation.ipynb
│   ├── 02_heteroscedasticity.ipynb
│   ├── 03_outliers.ipynb
│   ├── 04_multicollinearity.ipynb
│   ├── 05_nonlinearity.ipynb
│   └── 06_real_world_california_housing.ipynb
│
├── src/
│   └── simulation_functions.py
│
├── results/
│   ├── tables/
│   └── figures/
│
├── report/
│   └── research_report.pdf
│
├── README.md
├── requirements.txt
└── LICENSE
How to Run
1. Clone the repository
git clone https://github.com/Maryamsattar2025/ML-Regression-Model-Comparison.git
2. Open the project
cd ML-Regression-Model-Comparison
3. Install the required packages
pip install -r requirements.txt
4. Open Jupyter
jupyter notebook
Open the notebooks in the following order:
01_baseline_simulation.ipynb
02_heteroscedasticity.ipynb
03_outliers.ipynb
04_multicollinearity.ipynb
05_nonlinearity.ipynb
06_real_world_california_housing.ipynb
Results
The results/ directory contains the tables and figures generated during the experiments.
results/
├── tables/
└── figures/
Limitations
The simulations represent specific data-generating mechanisms and therefore do not cover every possible form of assumption violation.
Model performance may also change with different hyperparameter settings, sample sizes, contamination mechanisms, and nonlinear functional forms.
The supplied heteroscedasticity, multicollinearity, and nonlinearity results represent the reported experimental outputs rather than complete repeated-simulation mean ± SD summaries. Future versions can extend these experiments using a larger number of repeated simulations.
The California Housing experiment is based on the reported train-test evaluation. Repeated cross-validation could provide a more robust estimate of generalization performance.
Future Work
Possible extensions include:
More extensive repeated simulations
Stronger levels of assumption violations
Multiple types of outliers
More severe multicollinearity
Multiple nonlinear functional forms
Hyperparameter tuning
Cross-validation
Additional regression and machine-learning models
Prediction intervals and uncertainty quantification
Model interpretability analysis

Research Context
This project is motivated by the broader question of how statistical and machine-learning models behave under model misspecification and departures from classical assumptions.
It provides a bridge between classical regression diagnostics, robust statistical modelling, simulation-based analysis, and modern machine learning.
Author
Maryam Sattar
M.Sc. Statistics
M.Phil. Statistics
Research interests include:
Regression modelling
Robust statistics
Machine learning
Simulation-based inference
Model misspecification
Uncertainty quantification
Statistical machine learning

