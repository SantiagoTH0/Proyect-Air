from flask import Flask, render_template, request, url_for
from datetime import datetime
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from flask import render_template
from normalizar_datos import normalizar_datos
from flask import send_file
import io

app = Flask(__name__)

# Configurar carpeta para archivos estáticos
app.static_folder = 'static'
os.makedirs(app.static_folder, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/modelo', methods=['GET', 'POST'])
def modelo():
    return render_template('modelo.html')

@app.route('/predecir', methods=['GET', 'POST'])
def predecir():
    if request.method == 'POST':
        try:
            # Obtener datos del formulario
            mes = int(request.form['mes'])
            dia = int(request.form['dia'])
            hora = int(request.form['hora'])
            estacion = request.form['estacion']
            temperatura = float(request.form['temperatura'])
            lluvia = float(request.form['lluvia'])
            humedad = float(request.form['humedad'])
            direccion_viento = float(request.form['direccion_viento'])
            velocidad_viento = float(request.form['velocidad_viento'])
            radiacion_solar = float(request.form['radiacion_solar'])

            # Crear DataFrame con el orden exacto de columnas que espera el modelo
            datos = pd.DataFrame({
                'MES': [mes],
                'DIA': [dia],
                'HORA': [hora],
                'Temperatura': [temperatura],
                'Lluvia': [lluvia],
                'Humedad': [humedad],
                'Direccion_viento': [direccion_viento],
                'Velocidad_viento': [velocidad_viento],
                'Radiacion_solar': [radiacion_solar],
                'Estacion_La Ciudadela': [1 if estacion == 'La Ciudadela' else 0],
                'Estacion_Lagos I': [1 if estacion == 'Lagos I' else 0],
                'Estacion_Lagos del Cacique': [1 if estacion == 'Lagos del Cacique' else 0],
                'Estacion_San Francisco': [1 if estacion == 'San Francisco' else 0],
                'Estacion_Santa Cruz Girón': [1 if estacion == 'Santa Cruz Girón' else 0]
            })

            # Cargar el modelo
            ruta_modelo = os.path.join(os.path.dirname(__file__), "modelo_calidad_aire.pkl")
            modelo = joblib.load(ruta_modelo)

            # Realizar predicción
            prediccion = modelo.predict(datos)[0]
            print("Predicción realizada:", prediccion)

            # Generar gráfica para esta predicción
            ruta_imagen = os.path.join(app.static_folder, 'grafica_resultado.png')
            generar_grafica_prediccion(prediccion, ruta_imagen)

            return render_template('modelo.html', 
                                prediccion=prediccion,
                                ruta_grafica='grafica_resultado.png')

        except Exception as e:
            import traceback
            print(f"Error detallado: {str(e)}")
            print("Traceback completo:")
            print(traceback.format_exc())
            return render_template('modelo.html', 
                                error=f"Error en la predicción: {str(e)}")
    
    return render_template('modelo.html')

def generar_grafica_prediccion(prediccion, ruta_imagen):
    # Estimación de contaminantes según calidad del aire
    promedio_contaminantes = {
        1: {"PM2.5": 5, "PM10": 10, "NO2": 15, "O3": 20},
        2: {"PM2.5": 10, "PM10": 20, "NO2": 25, "O3": 30},
        3: {"PM2.5": 20, "PM10": 40, "NO2": 35, "O3": 45},
        4: {"PM2.5": 30, "PM10": 60, "NO2": 50, "O3": 60},
        5: {"PM2.5": 50, "PM10": 80, "NO2": 70, "O3": 80},
    }

    estimado = promedio_contaminantes.get(prediccion, {"PM2.5": 0, "PM10": 0, "NO2": 0, "O3": 0})

    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(estimado.keys()), y=list(estimado.values()), palette="viridis")
    plt.title(f"Predicción: Calidad Aire {prediccion}\nEstimación de contaminantes")
    plt.xlabel("Contaminante")
    plt.ylabel("Concentración estimada (µg/m³)")
    plt.ylim(0, max(estimado.values()) + 10)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(ruta_imagen)
    plt.close()

from flask import render_template
from normalizar_datos import normalizar_datos

@app.route('/visualizar-normalizacion')
def visualizar_normalizacion():
    try:
        # Obtener los datos normalizados
        datos = normalizar_datos()
        
        # Verificar si hay datos y convertir fechas a string
        if datos is not None and not datos.empty:
            datos['TIEMPO'] = datos['TIEMPO'].astype(str)
            print("Datos cargados exitosamente. Shape:", datos.shape)  # Debug
            return render_template('test_normalizacion.html', datos=datos)
        else:
            print("No se pudieron cargar los datos")  # Debug
            return render_template('test_normalizacion.html', datos=None)
    except Exception as e:
        print(f"Error en visualizar_normalizacion: {str(e)}")  # Debug
        return render_template('test_normalizacion.html', datos=None)

@app.route('/descargar-excel')
def descargar_excel():
    try:
        # Obtener los datos normalizados
        datos = normalizar_datos()
        
        if datos is not None and not datos.empty:
            # Crear un buffer en memoria para el archivo Excel
            output = io.BytesIO()
            
            # Convertir todas las fechas a string antes de guardar
            if 'TIEMPO' in datos.columns:
                datos['TIEMPO'] = datos['TIEMPO'].astype(str)
            
            # Usar openpyxl como motor de Excel
            datos.to_excel(output, index=False, engine='openpyxl')
            
            # Preparar el buffer para la lectura
            output.seek(0)
            
            # Enviar el archivo
            return send_file(
                output,
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                as_attachment=True,
                download_name='datos_normalizados.xlsx'
            )
        else:
            return "No hay datos disponibles para descargar", 404
            
    except Exception as e:
        print(f"Error al generar Excel: {str(e)}")  # Para debugging
        return "Error al generar el archivo Excel", 500

if __name__ == '__main__':
    app.run(debug=True)
