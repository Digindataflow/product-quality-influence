import shap
from joblib import load

def display_shap_plot(model_name, train_X, test_X):
    estimator = load(model_name) if isinstance(model_name, str) else model_name
    explainer = shap.Explainer(lambda x: estimator.predict(x), train_X, feature_names=train_X.columns)
    shap_values = explainer(test_X)
    shap.plots.beeswarm(shap_values)
