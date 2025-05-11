from Entrenar_Modelo import entrenar_modelo
from normalizar_datos import normalizar_datos


datos_normalizados = normalizar_datos()

if datos_normalizados is not None:
    
    modelo, reporte, ruta_imagen = entrenar_modelo(
        df=datos_normalizados,
        exportar_modelo=True,
        ruta_modelo="modelo_calidad_aire.pkl",
        ruta_imagen="static/grafica_resultado.png"
    )
    
 
    print("\nReporte de Clasificación:")
    print(reporte)
else:
    print("Error: No se pudieron cargar los datos normalizados")