import pandas as pd

def eliminar_columnas_contaminantes(df):
    columnas_a_eliminar = ['PM25', 'PM10', 'NO2', 'O3']
    columnas_existentes = [col for col in columnas_a_eliminar if col in df.columns]
    
    df_sin_contaminantes = df.drop(columns=columnas_existentes)
    
    print("Columnas eliminadas:", columnas_existentes)
    return df_sin_contaminantes
