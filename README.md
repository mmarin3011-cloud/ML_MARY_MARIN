# 📘 Proyecto de Predicción de Precios de Coches Usados

Este proyecto desarrolla un **modelo de machine learning** para predecir el precio de coches usados a partir de información técnica y comercial.
Incluye un pipeline completo: carga de datos, limpieza, preprocesado, entrenamiento, evaluación y despliegue del modelo.

---

## 📁 Estructura del proyecto

```
1_Fuentes.ipynb                # Obtención y exploración inicial de datos
2_Limpieza.ipynb               # Limpieza y estandarización del dataset
3_Entrenamiento_Evaluacion.ipynb # Entrenamiento y test del modelo
used_car_model.py              # Módulo Python con las funciones principales
data/                          # Datos de entrada (CSV/Parquet)
models/                        # Modelos entrenados (joblib)
```

---

## 🚀 Funcionalidades principales

El módulo `used_car_model.py` incluye:

* **Carga de datos** (CSV o Parquet)
* **Preprocesado automático**

  * Eliminación de columnas vacías
  * Imputación de valores faltantes
  * Generación de nuevas columnas (p. ej., edad del coche)
* **Construcción del pipeline de ML**

  * Codificación OneHot
  * Estandarización
  * Modelos disponibles:

    * Random Forest
    * Ridge Regression
* **Entrenamiento y evaluación**

  * Métricas: RMSE y R²
* **Guardado y carga del modelo**
* **Predicciones a partir de nuevos datos**

---

## ▶️ Uso desde línea de comandos

Ejemplo básico:

```bash
python used_car_model.py \
    --data data/coches.csv \
    --target price \
    --model random_forest \
    --out models/model.joblib
```

Parámetros principales:

* `--data`: https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho
* `--target`: Selling_price
* `--model`: Modelo a entrenar (`random_forest`)
* `--out`: Models/mnejor_modelo.joblib
* `--test-size`: Proporción para test (por defecto 0.2)

---

## 🛠️ Requisitos

```
pandas
numpy
scikit-learn
joblib
```

---

## 📄 Licencia

Este proyecto puede usarse y modificarse libremente para fines académicos o personales.
