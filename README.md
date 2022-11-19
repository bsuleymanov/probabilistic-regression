# probabilistic-regression

Probabilistic Lifetime Value prediction usign [Kaggle](https://www.kaggle.com/c/acquire-valued-shoppers-challenge/data) dataset.

Most of the lifetime values are zero, and the overall distribution of the non-zero part is lognormal.

## Implemented methods
- Deep Neural Network with Zero-Inflated LogNormal predictive distribution
- Catboost regressor with custom Zero-Inflated LogNormal objective function

To customize your pipeline, change the corresponding config:

![](https://raw.githubusercontent.com/bsuleymanov/probabilistic-regression/main/images/config.png?token=GHSAT0AAAAAABZRMBP2BIMBTZRWQ2TPZMAKY3XZ4OA)
