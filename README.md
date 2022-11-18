# probabilistic-regression

Probabilistic Lifetime Value prediction usign [Kaggle](https://www.kaggle.com/c/acquire-valued-shoppers-challenge/data) dataset.

The most part of lifetime values is zeros and common distribution of non-zero part is log-normal.

## Implemented methods
- Deep Neural Network with Zero-Inflated LogNormal predictive distribution
- Catboost regressor with custom Zero-Inflated LogNormal objective function
