import joblib
import pandas as pd


modelo = joblib.load('modelo_calidad_aire.pkl')


print("\nCaracterísticas que espera el modelo:")
print(modelo.feature_names_in_)


columnas_modelo = modelo.feature_names_in_
datos_prueba = pd.DataFrame(columns=columnas_modelo)


datos_ejemplo = {
    'MES': 5,
    'DIA': 15,
    'HORA': 14,
    'Temperatura': 25.0,
    'Lluvia': 0.0,
    'Humedad': 65.0,
    'Direccion_viento': 180.0,
    'Velocidad_viento': 2.0,
    'Radiacion_solar': 500.0
}


datos_prueba.loc[0] = 0

for columna, valor in datos_ejemplo.items():
    datos_prueba.loc[0, columna] = valor


datos_prueba.loc[0, 'Estacion_La Ciudadela'] = 1


try:
    prediccion = modelo.predict(datos_prueba)
    print("\nDatos de entrada:")
    print(datos_prueba)
    print("\nPredicción:", prediccion[0])
    print("\nColumnas utilizadas en orden:")
    print(datos_prueba.columns.tolist())
except Exception as e:
    print("Error en la predicción:", str(e))