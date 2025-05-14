import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from normalizar_datos import normalizar_datos

def entrenar_modelo():
    try:
        
        datos = normalizar_datos()
        
        
        columnas_requeridas = [
            'MES', 'DIA', 'HORA', 'Estacion', 'Temperatura', 
            'Lluvia', 'Humedad', 'Direccion_viento', 
            'Velocidad_viento', 'Radiacion_solar', 'Calidad_Aire'
        ]
        
        for columna in columnas_requeridas:
            if columna not in datos.columns:
                raise ValueError(f"Columna {columna} no encontrada en el dataset")

        
        X = datos.drop('Calidad_Aire', axis=1)
        y = datos['Calidad_Aire']

       
       
        X = pd.get_dummies(X, columns=['Estacion'])

      
    
       
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        
        modelo = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        modelo.fit(X_train, y_train)

        
        y_pred = modelo.predict(X_test)
        print("\nReporte de Clasificación:")
        print(classification_report(y_test, y_pred))
        
        print("\nMatriz de Confusión:")
        print(confusion_matrix(y_test, y_pred))

        
        joblib.dump(modelo, 'modelo_calidad_aire.pkl')
        print("\nModelo guardado como 'modelo_calidad_aire.pkl'")

        return True

    except Exception as e:
        print(f"Error durante el entrenamiento: {str(e)}")
        return False

if __name__ == "__main__":
    entrenar_modelo()