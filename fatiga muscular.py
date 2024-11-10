import numpy as np #biblioteca para realizar cálculos matemáticos y científicos
import pandas as pd
from scipy.signal import welch
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el dataset
data = np.loadtxt('./dataset_emg.txt') 

# Definir funciones para calcular RMS y MDF (Mediana de la Frecuencia)
def compute_rms(segment):
    return np.sqrt(np.mean(np.square(segment)))

def compute_mdf(segment, fs=200):
    freqs, psd = welch(segment, fs=fs)
    cumulative_power = np.cumsum(psd)
    return freqs[np.where(cumulative_power >= np.sum(psd) / 2)[0][0]]

# Dividir los datos en segmentos (ventanas de 1 segundo, asumiendo 200 Hz)
fs = 200  # Frecuencia de muestreo
window_size = fs // 2  # Ventana de 1 segundo
n_channels = data.shape[1] 
n_segments = data.shape[0] // window_size

# Extracción de características RMS y MDF por ventana y canal
rms_features = []
mdf_features = []

for i in range(n_segments):
    segment = data[i*window_size:(i+1)*window_size, :]
    rms_values = [compute_rms(segment[:, ch]) for ch in range(n_channels)]
    mdf_values = [compute_mdf(segment[:, ch], fs) for ch in range(n_channels)]
    rms_features.append(rms_values)
    mdf_features.append(mdf_values)

# Convertir las características a DataFrame
rms_df = pd.DataFrame(rms_features, columns=[f'rms_ch{ch+1}' for ch in range(n_channels)])
mdf_df = pd.DataFrame(mdf_features, columns=[f'mdf_ch{ch+1}' for ch in range(n_channels)])

# Combinar RMS y MDF en un solo conjunto de datos
features_df = pd.concat([rms_df, mdf_df], axis=1)

# Etiquetar los últimos segmentos como "fatiga" (1) y los primeros como "no fatiga" (0)
fatigue_labels = np.zeros(n_segments)
fatigue_labels[-(n_segments // 3):] = 1 

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(features_df, fatigue_labels, test_size=0.3, random_state=42)

# Crear y entrenar un modelo de Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predicciones
y_pred = clf.predict(X_test)

# Evaluación del modelo
report = classification_report(y_test, y_pred, target_names=["No Fatiga", "Fatiga"], output_dict=True)

# Generar el reporte detallado
with open('reporte_fatiga_muscular.txt', 'w') as f:
    # Introducción
    f.write("Reporte de Análisis de Fatiga Muscular\n\n")
    f.write("Este reporte analiza el nivel de fatiga muscular a través de señales EMG utilizando un modelo de clasificación de Bosque Aleatorio (Random Forest). Se evalúan dos características por canal de señal: RMS (Root Mean Square) y MDF (Mediana de la Frecuencia).\n\n")

    # Metodología
    f.write("Metodología\n")
    f.write("1. Se calculan las características RMS y MDF para cada ventana de 1 segundo de los datos EMG.\n")
    f.write("2. Las ventanas se etiquetan según la presencia o ausencia de fatiga muscular.\n")
    f.write("3. Se entrena un modelo de Arbol Aleatorio y se evalúa su precisión, exhaustividad y puntuación F1 en el conjunto de prueba.\n\n")

    # Resultados
    f.write("Resultados\n")
    f.write("Las métricas de clasificación son las siguientes:\n")
    f.write(f"- Precisión (Fatiga): {report['Fatiga']['precision']:.2f}\n")
    f.write(f"- Exhaustividad (Fatiga): {report['Fatiga']['recall']:.2f}\n")
    f.write(f"- Puntuación F1 (Fatiga): {report['Fatiga']['f1-score']:.2f}\n\n")

    # Interpretación de Resultados
    f.write("Interpretación de Resultados\n")
    f.write("La precisión refleja la cantidad de verdaderos positivos respecto a todas las predicciones de fatiga, mientras que la exhaustividad indica la proporción de verdaderos positivos respecto a todos los casos reales de fatiga. La puntuación F1 es una combinación equilibrada de ambas.\n\n")

    # Informe general
    f.write("Informe General\n")
    f.write(f"Exactitud del modelo: {report['accuracy']:.2f}\n")
    f.write(f"Promedio macro de precisión: {report['macro avg']['precision']:.2f}\n")
    f.write(f"Promedio macro de exhaustividad: {report['macro avg']['recall']:.2f}\n")
    f.write(f"Promedio macro de puntuación F1: {report['macro avg']['f1-score']:.2f}\n")
    f.write(f"Promedio ponderado de precisión: {report['weighted avg']['precision']:.2f}\n")
    f.write(f"Promedio ponderado de exhaustividad: {report['weighted avg']['recall']:.2f}\n")
    f.write(f"Promedio ponderado de puntuación F1: {report['weighted avg']['f1-score']:.2f}\n\n")

    # Matriz de Confusión
    cm = confusion_matrix(y_test, y_pred)
    f.write("Matriz de Confusión:\n")
    f.write(f"{cm}\n\n")
    f.write("Esta matriz permite observar los aciertos y errores del modelo al clasificar la fatiga muscular.\n\n")


# Gráfico de Barras para Métricas de Clasificación
report_dict = classification_report(y_test, y_pred, target_names=["No Fatiga", "Fatiga"], output_dict=True)

# Filtrar solo las clases de interés
labels = ["No Fatiga", "Fatiga"]
precision = [report_dict[label]['precision'] for label in labels]
recall = [report_dict[label]['recall'] for label in labels]
f1_score = [report_dict[label]['f1-score'] for label in labels]

# Crear gráfica de barras con etiquetas de valor
x = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(10, 6))
width = 0.2

# Graficar las barras
rects1 = ax.bar(x - width, precision, width, label='Precisión')
rects2 = ax.bar(x, recall, width, label='Exhaustividad')
rects3 = ax.bar(x + width, f1_score, width, label='Puntuación F1')

# Añadir etiquetas de valor encima de cada barra
ax.bar_label(rects1, padding=3, fmt='%.2f')  # Valores de precisión
ax.bar_label(rects2, padding=3, fmt='%.2f')  # Valores de exhaustividad
ax.bar_label(rects3, padding=3, fmt='%.2f')  # Valores de puntuación F1

# Configuración de la gráfica
ax.set_ylabel('Scores')
ax.set_title('Métricas de Clasificación')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.tight_layout()
# Guardar la figura en archivo
plt.savefig("metricas_clasificacion.png", dpi=300, bbox_inches='tight')  # Guardado con alta resolución

# Reorganización de las etiquetas de los ejes
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
            xticklabels=["No Fatiga", "Fatiga"], 
            yticklabels=["No Fatiga", "Fatiga"])

# Ajustes de los ejes y título
plt.xlabel('Predicción')
plt.ylabel('Realidad')
plt.title('Matriz de Confusión')
plt.gca().invert_yaxis()  # Invertir eje
plt.savefig("matriz_confusion.png", dpi=300, bbox_inches='tight')  # Guardar
plt.show()
