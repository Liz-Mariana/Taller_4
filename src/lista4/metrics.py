"""Riesgos empíricos y correlaciones."""

import numpy as np


def empirical_risk_squared(y_true, y_pred):
    """Riesgo empírico con pérdida cuadrática: promedio de (y_i - h(x_i))^2."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def empirical_risk_logistic(y_true, p_hat, eps=1e-15):
    """Riesgo empírico logístico: -promedio de y*log(p) + (1-y)*log(1-p)."""
    y_true = np.asarray(y_true, dtype=float)
    # Se recorta p a [eps, 1-eps] solo para evitar log(0)
    p = np.clip(np.asarray(p_hat, dtype=float), eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def misclassification(y_true, p_hat, threshold=0.5):
    """Número y proporción de errores de clasificación con la regla p >= umbral."""
    y_true = np.asarray(y_true)
    y_class = (np.asarray(p_hat) >= threshold).astype(int)
    errors = int(np.sum(y_class != y_true))
    return errors, errors / len(y_true)


def corr_pair(df, a, b):
    """Correlación de Pearson entre las columnas a y b de un DataFrame."""
    return float(df[a].corr(df[b]))