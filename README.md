# Machine Learning Applications in Option Pricing, Implied Volatility, and Bid-Ask Spread Prediction

This project applies machine learning to three option-related tasks:

1. Synthetic American call option pricing
2. Implied volatility prediction
3. Bid-ask spread prediction

Rather than treating these as generic regression problems, the workflow is designed around the financial structure of each task, combining finance-informed feature engineering, stratified sampling, model comparison, and interpretability analysis.

## Key technical features

- **Finance-informed feature engineering**
  - Pricing task: `sqrt(T)`, `rT`, `divT`, `v*sqrt(T)`, intrinsic value, ITM flag
  - Real-world tasks: moneyness, log-moneyness, moneyness², liquidity proxies (`log_volume`, `log_openInterest`)
- **Stratified train-test splitting for regression**
  - Quantile-based binning of the target variable to preserve distribution across splits
- **Model pipelines and leakage control**
  - Scaling embedded in pipelines for scale-sensitive models
- **Broad model comparison**
  - Linear Regression, SVR, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM, Neural Network
- **Hyperparameter tuning**
  - Randomized search for traditional models in the synthetic pricing task
- **Model diagnostics**
  - Train vs. test comparison for overfitting checks
  - Residual analysis and actual-vs-predicted plots
- **Interpretability**
  - SHAP analysis
  - Permutation feature importance
  - Moneyness-bucket diagnostics

## Project structure

- **Part 1: Synthetic American option pricing**
  - Learn a smooth pricing function generated from a binomial framework
  - Compare traditional ML models with a feedforward neural network
  - Diagnose model behaviour across OTM / ATM / ITM regions

- **Part 2A: Implied volatility modelling**
  - Predict cross-sectional implied volatility using strike/maturity-related variables
  - Capture nonlinear smile effects through moneyness transformations

- **Part 2B: Bid-ask spread modelling**
  - Model spread as a market microstructure outcome
  - Link predictions to liquidity, moneyness, maturity, and trading-friction intuition

## Main findings

- **Synthetic pricing**
  - Neural Network performed best, followed by SVR
  - Engineered features allowed even linear models to perform reasonably well

- **Implied volatility**
  - Random Forest performed best
  - Moneyness, log-moneyness, and moneyness² were the dominant predictors
  - Short-dated options showed stronger curvature in the IV surface, while longer-dated options appeared flatter

- **Bid-ask spread**
  - XGBoost performed best
  - Spread was driven mainly by moneyness, nonlinear moneyness effects, and liquidity variables
  - Results were consistent with market microstructure intuition around hedging difficulty, trading activity, and spread formation

## Tools and libraries

- Python
- pandas, numpy
- matplotlib, seaborn, plotly
- scikit-learn
- xgboost, lightgbm
- tensorflow / keras
- shap
- statsmodels

## Notes

This project was completed as part of coursework in Machine Learning for Finance and combines predictive modelling with financial interpretation rather than treating option data as a purely generic machine learning dataset.