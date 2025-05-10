import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
import joblib
import os

def entrenar_modelo(df, exportar_modelo=True, ruta_modelo="modelo_calidad_aire.pkl", ruta_imagen="grafica_resultado.png"):
    """
    Entrena un modelo Random Forest para predecir la calidad del aire y genera una gráfica
    con las métricas y estimación de contaminantes según la predicción.

    Args:
        df (pd.DataFrame): Dataset con columnas: ['dia', 'hora', 'estacion', 'lluvia', 'temperatura', 'radiacion_solar', 'calidad_aire']
        exportar_modelo (bool): Si se desea guardar el modelo entrenado.
        ruta_modelo (str): Ruta para guardar el modelo entrenado.
        ruta_imagen (str): Ruta donde se guardará la gráfica.

    Returns:
        modelo (RandomForestClassifier): Modelo entrenado.
        reporte (str): Reporte de clasificación.
        ruta_imagen (str): Ruta de la imagen de la gráfica.
    """

    # Variables de entrada y salida
    X = df[["dia", "hora", "estacion", "lluvia", "temperatura", "radiacion_solar"]]
    y = df["calidad_aire"]

    # Si estación es categórica, convertirla
    if X["estacion"].dtype == object:
        X = pd.get_dummies(X, columns=["estacion"])

    # División
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Modelo
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)

    # Predicciones
    y_pred = modelo.predict(X_test)

    # Métricas
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    reporte = classification_report(y_test, y_pred)

    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)

    # Estimación de contaminantes según calidad del aire
    promedio_contaminantes = {
        1: {"PM2.5": 5, "PM10": 10, "NO2": 15, "O3": 20},
        2: {"PM2.5": 10, "PM10": 20, "NO2": 25, "O3": 30},
        3: {"PM2.5": 20, "PM10": 40, "NO2": 35, "O3": 45},
        4: {"PM2.5": 30, "PM10": 60, "NO2": 50, "O3": 60},
        5: {"PM2.5": 50, "PM10": 80, "NO2": 70, "O3": 80},
    }

    # Cálculo promedio por clase predicha
    clase_predicha = int(pd.Series(y_pred).mode()[0])
    estimado = promedio_contaminantes.get(clase_predicha, {"PM2.5": 0, "PM10": 0, "NO2": 0, "O3": 0})

    # Gráfica
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(estimado.keys()), y=list(estimado.values()), palette="viridis")
    plt.title(f"Predicción: Calidad Aire {clase_predicha}\nEstimación de contaminantes")
    plt.xlabel("Contaminante")
    plt.ylabel("Concentración estimada (µg/m³)")
    plt.ylim(0, max(estimado.values()) + 10)
    plt.grid(axis="y")

    # Agregar métricas en el gráfico
    texto_metricas = f"Accuracy: {acc:.2f}\nPrecision: {prec:.2f}\nRecall: {rec:.2f}"
    plt.text(4.2, max(estimado.values()), texto_metricas, fontsize=10, verticalalignment="top")

    # Guardar imagen
    os.makedirs(os.path.dirname(ruta_imagen), exist_ok=True)
    plt.tight_layout()
    plt.savefig(ruta_imagen)
    plt.close()

    # Guardar modelo
    if exportar_modelo:
        joblib.dump(modelo, ruta_modelo)

    return modelo, reporte, ruta_imagen
