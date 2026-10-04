"""Ajuste de modelos."""

from sklearn.linear_model import LinearRegression, LogisticRegression

from .data import load_csv
from .metrics import (
    corr_pair,
    empirical_risk_logistic,
    empirical_risk_squared,
    misclassification,
)


def fit_linear(df, features, target="y"):
    """Ajusta una regresión lineal con intercepto."""
    model = LinearRegression(fit_intercept=True)
    model.fit(df[features], df[target])
    return model


def linear_summary(df, features, target="y", label=None):
    """Ajusta el modelo y devuelve intercepto, coeficientes y riesgo empírico."""
    model = fit_linear(df, features, target)
    preds = model.predict(df[features])

    row = {
        "modelo": label or " + ".join(features),
        "intercepto": float(model.intercept_),
    }
    for name, coef in zip(features, model.coef_):
        row["coef_" + name] = float(coef)
    row["riesgo_empirico"] = empirical_risk_squared(df[target], preds)

    # Estas dos no van en las tablas, pero las necesitamos en otros ejercicios
    row["_model"] = model
    row["_pred"] = preds
    return row


def compare_x1_vs_x1x2(name):
    """Ajusta x1 -> y y (x1, x2) -> y sobre un archivo y resume el resultado."""
    d = load_csv(name)

    only_x1 = linear_summary(d, ["x1"])
    both = linear_summary(d, ["x1", "x2"])

    return {
        "archivo": name,
        "corr(x1,x2)": corr_pair(d, "x1", "x2"),
        "R_x1": only_x1["riesgo_empirico"],
        "R_x1,x2": both["riesgo_empirico"],
        "beta1": both["coef_x1"],
        "beta2": both["coef_x2"],
    }


def fit_logistic(df, features, target, C=1e6, max_iter=20000):
    """Regresión logística (lbfgs) casi sin regularización, como pide el enunciado."""
    model = LogisticRegression(solver="lbfgs", C=C, max_iter=max_iter)
    model.fit(df[features], df[target])
    return model


def logistic_summary(df, features, target, label, threshold=0.5):
    """Ajusta el modelo logístico y devuelve riesgo, errores y proporción de errores."""
    model = fit_logistic(df, features, target)
    p_hat = model.predict_proba(df[features])[:, 1]
    errors, proportion = misclassification(df[target], p_hat, threshold)

    return {
        "modelo": label,
        "n_variables": len(features),
        "riesgo_logistico": empirical_risk_logistic(df[target], p_hat),
        "errores": errors,
        "prop_errores": proportion,
        "converge": bool(model.n_iter_[0] < model.max_iter),
        "_model": model,
        "_p_hat": p_hat,
    }