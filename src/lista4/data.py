"""Carga de datos."""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_csv(name):
    """Lee un CSV de la carpeta data/. El nombre puede ir con o sin '.csv'."""
    if not name.endswith(".csv"):
        name = name + ".csv"
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"No se encontró {path}")
    return pd.read_csv(path)