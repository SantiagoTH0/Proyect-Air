import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_squared_error, classification_report, 
                           accuracy_score, precision_score, recall_score,
                           confusion_matrix)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def clasificar_calidad_aire(row):
    if (row['PM25'] <= 12 and row['PM10'] <= 54 and 
        row['NO2'] <= 53 and row['O3'] <= 100):
        return 'Optimo'
    elif (row['PM25'] <= 35.4 and row['PM10'] <= 154 and 
          row['NO2'] <= 100 and row['O3'] <= 150):
        return 'Moderado'
    elif (row['PM25'] <= 55.4 and row['PM10'] <= 254 and 
          row['NO2'] <= 360 and row['O3'] <= 200):
        return 'Contaminado'
    else:
        return 'Muy Contaminado'

def cargar_y_preprocesar_excel(path_excel= "c:\\Users\\estef\\Documents\\Proyect\\Proyect-Air\\data\\Reporte_Calidad_de_Aire.xlsx"):
    df = pd.read_excel(path_excel)

    df['TIEMPO'] = pd.to_datetime(df['TIEMPO'], errors='coerce')
    df['AÑO'] = df['TIEMPO'].dt.year
    df['MES'] = df['TIEMPO'].dt.month
    df['DIA'] = df['TIEMPO'].dt.day
    df['HORA'] = df['TIEMPO'].dt.hour

    df.columns = df.columns.str.normalize('NFKD')\
                          .str.encode('ascii', errors='ignore')\
                          .str.decode('ascii')\
                          .str.strip()

    mediciones = {
        'PM10': 'PM10',
        'PM2,5': 'PM25',
        'NO2': 'NO2',
        'O3': 'O3',
        'TEMPERATURA': 'Temperatura',
        'LLUVIA': 'Lluvia',
        'HUMEDAD': 'Humedad',
        'DIR. VIENTO': 'Direccion_viento',
        'VEL.VIENTO': 'Velocidad_viento',
        'RAD. SOLAR': 'Radiacion_solar'
    }

    for col in df.columns:
        if col != 'TIEMPO':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    for medicion_original, medicion_nueva in mediciones.items():
        columnas_medicion = [col for col in df.columns if medicion_original in col]
        if columnas_medicion:
            df[medicion_nueva] = df[columnas_medicion].mean(axis=1)

    columnas_requeridas = ['PM10', 'PM25', 'NO2', 'O3', 'Temperatura', 'Lluvia',
                          'Humedad', 'Direccion_viento', 'Velocidad_viento', 'Radiacion_solar']
    df = df.dropna(subset=columnas_requeridas)

    df['Calidad_Aire'] = df.apply(clasificar_calidad_aire, axis=1)
    
    return df

def entrenar_modelo_calidad_aire(df):
    feature_columns = ['PM10', 'PM25', 'NO2', 'O3', 'Temperatura', 'Lluvia',
                      'Humedad', 'Direccion_viento', 'Velocidad_viento', 'Radiacion_solar']
    
    X = df[feature_columns]
    y = df['Calidad_Aire']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    conf_matrix = confusion_matrix(y_test, predictions)
    
    print("\nMétricas del Modelo:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, predictions))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=model.classes_,
                yticklabels=model.classes_)
    plt.title('Matriz de Confusión')
    plt.ylabel('Valor Real')
    plt.xlabel('Predicción')
    plt.show()
    
    plt.figure(figsize=(12, 6))
    predictions_df = pd.DataFrame({
        'Real': y_test,
        'Predicción': predictions
    }).value_counts().unstack()
    predictions_df.plot(kind='bar')
    plt.title('Comparación de Valores Reales vs Predicciones')
    plt.xlabel('Categorías de Calidad del Aire')
    plt.ylabel('Cantidad de Muestras')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return model, X_test, y_test, predictions

if __name__ == "__main__":
    df = cargar_y_preprocesar_excel()
    model, X_test, y_test, predictions = entrenar_modelo_calidad_aire(df)