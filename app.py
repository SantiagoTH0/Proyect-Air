from flask import Flask, render_template, request, url_for
from datetime import datetime
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

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
            fecha = datetime.strptime(request.form['fecha'], '%Y-%m-%d')
            hora = datetime.strptime(request.form['hora'], '%H:%M').hour
            estacion = request.form['estacion']
            lluvia = float(request.form['lluvia'])
            temperatura = float(request.form['temperatura'])
            radiacion = float(request.form['radiacion'])

            # Crear DataFrame con los datos de entrada
            datos = pd.DataFrame({
                'dia': [fecha.day],
                'hora': [hora],
                'estacion': [estacion],
                'lluvia': [lluvia],
                'temperatura': [temperatura],
                'radiacion_solar': [radiacion]
            })

            # Cargar el modelo
            ruta_modelo = os.path.join(os.path.dirname(__file__), "modelo_calidad_aire.pkl")
            modelo = joblib.load(ruta_modelo)

            # Si estación es categórica, convertirla
            if datos["estacion"].dtype == object:
                datos = pd.get_dummies(datos, columns=["estacion"])

            # Realizar predicción
            prediccion = modelo.predict(datos)[0]

            # Generar gráfica para esta predicción
            ruta_imagen = os.path.join(app.static_folder, 'grafica_resultado.png')
            generar_grafica_prediccion(prediccion, ruta_imagen)

            return render_template('modelo.html', 
                                prediccion=prediccion,
                                ruta_grafica='grafica_resultado.png')

        except Exception as e:
            print(f"Error: {str(e)}")  # Para debugging
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

if __name__ == '__main__':
    app.run(debug=True)
