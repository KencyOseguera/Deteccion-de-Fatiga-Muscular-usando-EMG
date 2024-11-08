# Detección de Fatiga Muscular Basado en Señales EMG

## Descripción

Este programa utiliza señales EMG (electromiografía) de una sola sesión de una persona para analizar el nivel de fatiga muscular. A través del procesamiento de estas señales, el programa emplea un modelo de aprendizaje automático (Random Forest) para clasificar los segmentos de datos en dos categorías: **fatiga** y **no fatiga**. Este tipo de análisis es útil en aplicaciones de rehabilitación, entrenamiento deportivo y estudios biomédicos relacionados con la fatiga muscular.

## Funcionalidad

El programa lleva a cabo las siguientes tareas:

1. **Carga de Datos EMG**: Lee los datos de las señales EMG en formato de un archivo `.txt`.
2. **Extracción de Características**: Divide los datos en segmentos de 1 segundo (asumiendo una frecuencia de muestreo de 200 Hz). A partir de cada segmento, se calculan dos características importantes:
   - **RMS (Root Mean Square)**: Indica la amplitud media de la señal.
   - **MDF (Mediana de la Frecuencia)**: Representa la frecuencia donde la mitad de la potencia espectral de la señal está concentrada.
3. **Etiquetado de Datos**: Los segmentos de la señal se etiquetan como **fatiga** o **no fatiga**. En este ejemplo, los últimos segmentos de la señal se consideran como fatiga (1), y los primeros, como no fatiga (0).
4. **Entrenamiento del Modelo**: Utiliza un clasificador de Bosques Aleatorios para aprender patrones en los datos y distinguir entre los estados de fatiga.
5. **Evaluación del Modelo**: Evalúa el rendimiento del modelo con métricas de clasificación (precisión, exhaustividad, puntuación F1) y genera una **matriz de confusión** para visualizar los aciertos y errores en la clasificación.
6. **Generación de Reportes**: Crea un archivo `reporte_fatiga_muscular.txt` que documenta las métricas de rendimiento, una interpretación de resultados y una visualización de la matriz de confusión.

## Requisitos para ejecutar el programa

- Python 3.7 o superior
- Bibliotecas de Python:
  - `numpy`
  - `pandas`
  - `scipy`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`

Para instalar las bibliotecas necesarias, puedes usar:

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn
```

## Estructura del Código

El programa consta de las siguientes secciones principales:

1. **Carga de Datos**: 
   ```python
   data = np.loadtxt('./dataset_emg.txt')
   ```

2. **Extracción de Características**: 
   Calcula RMS y MDF para cada canal de datos, dividiendo en segmentos y obteniendo valores que representan el estado de la señal.
   
   ```python
   def compute_rms(segment): ...
   def compute_mdf(segment, fs=200): ...
   ```

3. **Entrenamiento y Pruebas**:
   Divide los datos en conjuntos de entrenamiento y prueba, entrena un clasificador de Bosques Aleatorios, y genera predicciones.
   
   ```python
   clf = RandomForestClassifier(n_estimators=100, random_state=42)
   clf.fit(X_train, y_train)
   ```

4. **Evaluación y Reporte**:
   Utiliza el reporte de clasificación y la matriz de confusión para evaluar el modelo, y exporta los resultados a un archivo `.txt`.
   
   ```python
   report = classification_report(y_test, y_pred, target_names=["No Fatiga", "Fatiga"], output_dict=True)
   ```

5. **Visualización**:
   Genera una gráfica de barras para mostrar las métricas de clasificación y un mapa de calor de la matriz de confusión.

## Ejecución del Programa

1. Coloca el archivo de datos de señales EMG (`dataset_emg.txt`) en la misma carpeta que el código.
2. Ejecuta el script de Python:

   ```bash
   python analisis_fatiga_muscular.py
   ```

3. Tras la ejecución, se generarán los siguientes archivos de salida:
   - `reporte_fatiga_muscular.txt`: Reporte con el análisis de métricas y la matriz de confusión.
   - `metricas_clasificacion.png`: Gráfica de barras con las métricas de clasificación.
   - `matriz_confusion.png`: Visualización de la matriz de confusión.

## Ejemplo de Salida

El archivo de reporte (`reporte_fatiga_muscular.txt`) incluye:

- Métricas de precisión, exhaustividad y puntuación F1 para la clasificación de fatiga/no fatiga.
- Interpretación de los resultados, explicando las implicaciones de cada métrica.
- Matriz de confusión, para visualizar la efectividad del modelo en cada clase.

---

Este README explica la finalidad y funcionamiento del programa, detallando cada paso en el análisis de señales EMG para detectar fatiga muscular.
