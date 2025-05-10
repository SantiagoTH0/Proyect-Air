import pandas as pd

def normalizar_datos(path_excel= "c:\\Users\\estef\\Documents\\Proyect\\Proyect-Air\\data\\Reporte_Calidad_de_Aire.xlsx"):
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

    return df
