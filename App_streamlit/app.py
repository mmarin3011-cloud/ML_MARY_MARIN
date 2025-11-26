import joblib
import streamlit as st
import pandas as pd

st.title("🔮 Predicción del Precio de Vehículos Usados")
st.write("Selecciona un vehículo y ajusta sus características para obtener una predicción basada en el modelo RandomForest.")

# -------------------------------------------------------
# 1. Cargar el modelo RandomForest entrenado
# -------------------------------------------------------
MODEL_PATH = "/Users/marymarin/Documents/ML_Mary_Marin/Models/mejor_modelo.pkl"

modelo = joblib.load(MODEL_PATH)

# -------------------------------------------------------
# 2. Cargar dataset real para poblar el formulario
# -------------------------------------------------------
df = pd.read_csv("/Users/marymarin/Documents/ML_Mary_Marin/Data/test/X_test.csv")

nombres_unicos = sorted(df["name"].unique())

# -------------------------------------------------------
# 3. Formulario basado en vehículos reales
# -------------------------------------------------------
name = st.selectbox("Selecciona el vehículo exacto", nombres_unicos)

row = df[df["name"] == name].iloc[0]

st.subheader("Ajusta las características:")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Años de uso", min_value=0, max_value=40, value=int(row["age"]))
    km_driven = st.number_input("Kilometraje", min_value=0, value=int(row["km_driven"]))
    fuel = st.selectbox("Combustible", sorted(df["fuel"].unique()),
                        index=list(sorted(df["fuel"].unique())).index(row["fuel"]))

with col2:
    transmission = st.selectbox("Transmisión", sorted(df["transmission"].unique()),
                                index=list(sorted(df["transmission"].unique())).index(row["transmission"]))
    seller_type = st.selectbox("Tipo de vendedor", sorted(df["seller_type"].unique()),
                               index=list(sorted(df["seller_type"].unique())).index(row["seller_type"]))
    owner = st.selectbox("Número de dueños", sorted(df["owner"].unique()),
                         index=list(sorted(df["owner"].unique())).index(row["owner"]))

brand = st.selectbox("Marca", sorted(df["brand"].unique()),
                     index=list(sorted(df["brand"].unique())).index(row["brand"]))

# -------------------------------------------------------
# 4. Construir DataFrame EXACTO como en entrenamiento
# -------------------------------------------------------
input_df = pd.DataFrame([{
    "name": name,
    "brand": brand,
    "age": age,
    "km_driven": km_driven,
    "fuel": fuel,
    "transmission": transmission,
    "seller_type": seller_type,
    "owner": owner,
    "km_per_year": km_driven / age if age > 0 else 0
}])

# -------------------------------------------------------
# 5. Botón de predicción
# -------------------------------------------------------
if st.button("Predecir precio"):
    try:
        pred = modelo.predict(input_df)[0]
        st.success(f"💰 Precio estimado: **${(pred*0.011):,.2f}**")
    except Exception as e:
        st.error(f"⚠️ Error durante la predicción: {e}")
