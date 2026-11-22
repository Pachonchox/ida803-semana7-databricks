# Predicción de edad con una CNN en Databricks

Proyecto académico del módulo Procesamiento de imágenes, semana 7. Estima edades de 20 a 50 años a partir de imágenes etiquetadas.

**[Abrir el notebook con resultados](Prediccion_de_edad.ipynb)**

## Resultados de la ejecución en Databricks

La prueba usó **6.682 imágenes** después de la auditoría de duplicados y etiquetas en conflicto. La selección se hizo con el MAE de validación.

| Modelo | MAE (años) | RMSE (años) | Dentro de ±5 años |
| --- | ---: | ---: | ---: |
| Mediana de entrenamiento | 7,82 | 9,03 | 34,77 % |
| CNN base | 7,57 | 8,95 | 34,94 % |
| CNN optimizada | 7,46 | 8,79 | 35,12 % |

La CNN optimizada bajó el MAE de prueba un 4,6 % frente a la predicción constante. Los errores son mayores en los extremos del rango de edades. La [gráfica de evaluación](evaluacion.png) y las [métricas completas](metricas.json) se exportaron de esa ejecución.

## API y despliegue

La [API de FastAPI](api.py) carga el modelo exportado y acepta imágenes JPEG o PNG en `POST /predict`. La [prueba local de rendimiento](rendimiento_api.json) usó 120 solicitudes por escenario. El [Dockerfile](Dockerfile) espera `modelo_edad.keras`, generado por el notebook, en el directorio del proyecto. El [manifiesto Kubernetes](kubernetes.yaml) define el servicio y el escalado por CPU. Las pruebas se realizaron en proceso con el cliente de pruebas de FastAPI.

## Reproducir en un workspace propio

1. Importa el notebook `.ipynb` en Databricks y selecciona el entorno base ML v5 o uno compatible con TensorFlow 2.20.
2. Coloca `20-50.zip` en un volumen de Unity Catalog al que tengas acceso y configura el parámetro `ruta_volumen`.
3. Ejecuta todas las celdas. La ejecución exporta el modelo, la gráfica y las métricas en una carpeta nueva bajo `resultados/` en el volumen.

[Notebook de referencia citado en la pauta](https://www.kaggle.com/code/gcdatkin/age-prediction-from-images-cnn-regression/notebook)
