"""
used_car_model.py

Módulo utilitario para un proyecto de predicción de precios de coches usados.
Incluye: carga de datos, preprocesado básico, ingeniería de características, entrenamiento
(con scikit-learn), evaluación y guardado/carga de modelo.

Referencias:
- Notebook del proyecto (si necesitas reutilizar código/experimentación): /mnt/data/3_Entrenamiento_Evaluacion.ipynb

Dependencias:
- pandas, numpy, scikit-learn, joblib

Uso básico (desde terminal):
python used_car_model.py --data data.csv --target price --out model.joblib

Funciones exportadas:
- load_data(path)
- preprocess(df, dropna=True)
- train_model(X, y, model_type='random_forest', random_state=42)
- evaluate_model(model, X_test, y_test)
- save_model(model, path)
- load_model(path)
- predict(model, df)

"""

from __future__ import annotations
import argparse
import json
from typing import Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib


def load_data(path: str) -> pd.DataFrame:
    """Carga un CSV o parquet y devuelve un DataFrame.

    Args:
        path: Ruta al archivo (.csv o .parquet).
    Returns:
        pd.DataFrame
    """
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
    """Preprocesado mínimo:
    - Elimina columnas totalmente vacías
    - Rellena missing en numéricos con mediana y en categóricos con 'missing'
    - Crea columnas derivadas sencillas (edad del coche a partir del año si existe)

    Ajusta/añade transformaciones según tus columnas reales.
    """
    df = df.copy()

    # eliminar columnas totalmente vacías
    df.dropna(axis=1, how='all', inplace=True)

    # ejemplo: si existe columna 'year', crear 'car_age'
    if 'year' in df.columns:
        try:
            current_year = pd.Timestamp.now().year
            df['car_age'] = current_year - pd.to_numeric(df['year'], errors='coerce')
        except Exception:
            df['car_age'] = np.nan

    # rellenado simple
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    for c in num_cols:
        if df[c].isnull().any():
            df[c] = df[c].fillna(df[c].median())

    for c in cat_cols:
        if df[c].isnull().any():
            df[c] = df[c].fillna('missing')

    if dropna:
        df.dropna(inplace=True)

    return df


def build_preprocessor(df: pd.DataFrame) -> Tuple[ColumnTransformer, list]:
    """Construye un ColumnTransformer según tipos detectados en df.

    Returns:
        transformer, feature_columns (lista de columnas usadas por el modelo)
    """
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    # evita usar columnas objetivo o identificadores si las hay; el caller debe manejar eso
    num_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    cat_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols),
        ],
        remainder='drop'
    )

    # feature_columns no se puede conocer exactamente hasta aplicar OneHot; devolvemos lista base
    feature_columns = num_cols + cat_cols
    return preprocessor, feature_columns


def train_model(X: pd.DataFrame, y: pd.Series, model_type: str = 'random_forest', random_state: int = 42):
    """Entrena un pipeline (preprocesador + modelo).

    Args:
        X: DataFrame con features
        y: Serie target
        model_type: 'random_forest' o 'ridge'
    Returns:
        pipeline entrenado
    """
    preprocessor, _ = build_preprocessor(X)

    if model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1)
    elif model_type == 'ridge':
        model = Ridge(random_state=random_state)
    else:
        raise ValueError(f"model_type desconocido: {model_type}")

    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

    pipeline.fit(X, y)
    return pipeline


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Devuelve métricas RMSE y R2"""
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)
    return {'rmse': float(rmse), 'r2': float(r2)}


def save_model(model, path: str) -> None:
    """Guarda el pipeline completo con joblib"""
    joblib.dump(model, path)


def load_model(path: str):
    """Carga un pipeline guardado"""
    return joblib.load(path)


def predict(model, df: pd.DataFrame) -> np.ndarray:
    """Predice a partir de un DataFrame ya preprocesado/compatible con el pipeline"""
    return model.predict(df)


def _parse_args():
    p = argparse.ArgumentParser(description='Entrena un modelo simple para precios de coches usados')
    p.add_argument('--data', required=True, help='Ruta al CSV/Parquet con datos')
    p.add_argument('--target', required=True, help='Nombre de la columna objetivo, e.g. price')
    p.add_argument('--out', default='model.joblib', help='Ruta de salida para el modelo')
    p.add_argument('--model', default='random_forest', choices=['random_forest', 'ridge'], help='Tipo de modelo')
    p.add_argument('--test-size', type=float, default=0.2, help='Tamaño del test set.')
    p.add_argument('--random-state', type=int, default=42)
    return p.parse_args()


def main():
    args = _parse_args()
    print(f"Cargando datos desde {args.data}...")
    df = load_data(args.data)
    if args.target not in df.columns:
        raise ValueError(f"La columna target '{args.target}' no está en el dataset")

    print("Preprocesando datos...")
    df = preprocess(df)

    X = df.drop(columns=[args.target])
    y = df[args.target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state)

    print(f"Entrenando modelo ({args.model})...")
    pipeline = train_model(X_train, y_train, model_type=args.model, random_state=args.random_state)

    print("Evaluando modelo...")
    metrics = evaluate_model(pipeline, X_test, y_test)
    print("Métricas:", json.dumps(metrics, indent=2))

    print(f"Guardando modelo en {args.out}...")
    save_model(pipeline, args.out)
    print("Hecho.")


if __name__ == '__main__':
    main()
