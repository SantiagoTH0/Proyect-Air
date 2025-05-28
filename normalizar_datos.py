import pandas as pd
import os
from variable_calidadAire import agregar_calidad_aire
from eliminar_Contaminantes import eliminar_columnas_contaminantes


def normalizar_datos(path_excel="data/Reporte_Calidad_de_Aire.xlsx"):
    try:
        print(f"Intentando leer archivo: {path_excel}")  
        
        # Verificar si estamos en entorno de producción (Render)
        is_production = os.environ.get('RENDER', False)
        
        if is_production or not os.path.exists(path_excel):
            print("Usando datos de prueba para el entorno de producción")
            # Crear un DataFrame de prueba con datos simulados
            return crear_datos_prueba()
        
        df = pd.read_excel(path_excel)
        print(f"Datos leídos. Shape inicial: {df.shape}")  

        dfs_estaciones = []
        
     
        tiempo_base = pd.to_datetime(df['TIEMPO'], errors='coerce')
        
        
        estaciones = {
            'EST. LAGOS I F/BLANCA': 'Lagos I',
            'EST. LACIUDADELA': 'La Ciudadela',
            'EST. SANTA CRUZ GIRÓN': 'Santa Cruz Girón',
            'EST. SAN FRANCISCO': 'San Francisco',
            'EST. LAGOS DEL CACIQUE': 'Lagos del Cacique'
        }

        
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

        
        for estacion_original, estacion_normalizada in estaciones.items():
            
            df_estacion = pd.DataFrame()
            df_estacion['TIEMPO'] = tiempo_base
            df_estacion['AÑO'] = tiempo_base.dt.year
            df_estacion['MES'] = tiempo_base.dt.month
            df_estacion['DIA'] = tiempo_base.dt.day
            df_estacion['HORA'] = tiempo_base.dt.hour
            df_estacion['Estacion'] = estacion_normalizada
            
            
            cols_estacion = [col for col in df.columns if estacion_original in col]
            
            
            for medicion_original, medicion_nueva in mediciones.items():
                cols_medicion = [col for col in cols_estacion if medicion_original in col]
                if cols_medicion:
                    df_estacion[medicion_nueva] = df[cols_medicion].mean(axis=1)
            
            
            for col in df_estacion.columns:
                if col not in ['TIEMPO', 'Estacion']:
                    df_estacion[col] = pd.to_numeric(df_estacion[col], errors='coerce')
            
            
            mediciones_requeridas = ['PM10', 'PM25', 'NO2', 'O3']
            columnas_existentes = [col for col in mediciones_requeridas if col in df_estacion.columns]
            
            if columnas_existentes:
                
                df_estacion = df_estacion.dropna(subset=columnas_existentes)
            
            
            if not df_estacion.empty:
                dfs_estaciones.append(df_estacion)

        
        if dfs_estaciones:
            df_normalizado = pd.concat(dfs_estaciones, ignore_index=True)
            
            print(f"Registros después de eliminar datos vacíos: {df_normalizado.shape}")  
            
            
            df_normalizado = agregar_calidad_aire(df_normalizado)
            
            
            df_final = eliminar_columnas_contaminantes(df_normalizado)
            
            print(f"Shape final después de normalización: {df_final.shape}")  
            return df_final
        else:
            print("No se encontraron datos válidos después de la normalización")
            return None
            
    except Exception as e:
        print(f"Error en normalizar_datos: {str(e)}")  
        # En caso de error, devolver datos de prueba
        return crear_datos_prueba()


def crear_datos_prueba():
    """Crea un conjunto de datos de prueba para usar cuando el archivo Excel no está disponible"""
    # Crear un DataFrame con datos simulados
    data = {
        'MES': [1, 2, 3, 4, 5, 6] * 10,
        'DIA': [15, 16, 17, 18, 19, 20] * 10,
        'HORA': [8, 10, 12, 14, 16, 18] * 10,
        'Estacion': ['La Ciudadela', 'Lagos I', 'Lagos del Cacique', 'San Francisco', 'Santa Cruz Girón'] * 12,
        'Temperatura': [25.5, 26.2, 24.8, 27.0, 23.5, 26.8] * 10,
        'Lluvia': [0.0, 2.5, 0.0, 1.5, 3.0, 0.0] * 10,
        'Humedad': [65, 70, 60, 75, 80, 55] * 10,
        'Direccion_viento': [180, 220, 90, 270, 45, 135] * 10,
        'Velocidad_viento': [2.1, 1.5, 3.0, 2.8, 1.2, 2.5] * 10,
        'Radiacion_solar': [800, 750, 900, 850, 600, 700] * 10,
        'Calidad_Aire': [1, 2, 2, 3, 1, 2] * 10
    }
    
    df = pd.DataFrame(data)
    print(f"Datos de prueba creados. Shape: {df.shape}")
    return df
