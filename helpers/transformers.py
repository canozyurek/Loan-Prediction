from sklearn.base import BaseEstimator, TransformerMixin

class DateTransformer(BaseEstimator, TransformerMixin):
  def fit(self, X, y=None):
    return self

  def transform(self, X, y = None):
    for i in self.columns:
      X[f"{i}_year"] = X.i.dt.year
      X[f"{i}_month"] = X.i.dt.month
      X[f"{i}_day"] = X.i.dt.day
      X[f"{i}_dow"] = X.i.dt.dayofweek
      X[f"{i}_quarter"] = X.i.dt.quarter
      X = X.drop(i, axis=1)
      X[i] = X[i].astype(str)
    return X