import pandas as pd

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

def agregar_calidad_aire(df):
    df['Calidad_Aire'] = df.apply(clasificar_calidad_aire, axis=1)
    return df
