# Proyecto de Neumonía - Curso Desarrollo de Proyectos de Inteligencia Artificial

Este proyecto consiste en una aplicación de escritorio diseñada para apoyar el diagnóstico de neumonía mediante el procesamiento de imágenes de rayos X (DICOM y JPG). El sistema ha sido refactorizado para garantizar una arquitectura modular y escalable.

## Requisitos del Sistema
* **Versión de Python:** 3.10 o superior.
* **Dependencias principales:** OpenCV, Pydicom, Pillow, Pytest, TensorFlow.

## Estructura del Proyecto
```
UAO-Neumonia/
│
├── src/                        
│   ├── read_img.py             
│   ├── preprocess_img.py       
│   ├── load_model.py           
│   ├── grad_cam.py             
│   └── integrator.py           
│
├── tests/                      
│   ├── test_read_img.py        
│   ├── test_preprocess_img.py   
│   ├── test_load_model.py      
│   └── test_grad_cam.py        
│
├── docs/                       
│   ├── flujo_app.png 
│   ├── arquitectura_modulos.png
│
├── models/                     
│   └── WilhemNet86.h5          
│
├── detector_neumonia.py        
├── requirements.txt            
├── Dockerfile                  
├── LICENSE                     
└── README.md                   
```

## Uso de la herramienta:

A continuación le explicaremos cómo empezar a utilizarla.

Requerimientos necesarios para el funcionamiento:

- Instale UV para Windows siguiendo las siguientes instrucciones: https://docs.astral.sh/uv/getting-started/installation/
- Abra la terminal y ejecute las siguientes instrucciones:
  ```
  uv sync
  uv run detector_neumonia.py
  ```
  
  En caso de errores por la librería Tensorflow en Windows, ejecutar:
  ```
  uv pip install tensorflow==2.15.0
  ```

Uso de la Interfaz Gráfica:

- Ingrese la cédula del paciente en la caja de texto
- Presione el botón 'Cargar Imagen', seleccione la imagen del explorador de archivos del computador (Imagenes de prueba en https://drive.google.com/drive/folders/1WOuL0wdVC6aojy8IfssHcqZ4Up14dy0g?usp=drive_link)
- Presione el botón 'Predecir' y espere unos segundos hasta que observe los resultados
- Presione el botón 'Guardar' para almacenar la información del paciente en un archivo excel con extensión .csv
- Presione el botón 'PDF' para descargar un archivo PDF con la información desplegada en la interfaz
- Presión el botón 'Borrar' si desea cargar una nueva imagen

---

## Ejecución de Tests

Para ejecutar los tests del proyecto, abra la terminal en el directorio del proyecto y ejecute el siguiente comando:

  ```bash
  uv run pytest
  ```

---

## Arquitectura de archivos propuesta.

## detector_neumonia.py

Contiene el diseño de la interfaz gráfica utilizando Tkinter.

Los botones llaman métodos contenidos en otros scripts.

## integrator.py

Es un módulo que integra los demás scripts y retorna solamente lo necesario para ser visualizado en la interfaz gráfica.
Retorna la clase, la probabilidad y una imagen el mapa de calor generado por Grad-CAM.

## read_img.py

Script que lee la imagen en formato DICOM para visualizarla en la interfaz gráfica. Además, la convierte a arreglo para su preprocesamiento.

## preprocess_img.py

Script que recibe el arreglo proveniento de read_img.py, realiza las siguientes modificaciones:

- resize a 512x512
- conversión a escala de grises
- ecualización del histograma con CLAHE
- normalización de la imagen entre 0 y 1
- conversión del arreglo de imagen a formato de batch (tensor)

## load_model.py

Script que lee el archivo binario del modelo de red neuronal convolucional previamente entrenado llamado 'WilhemNet86.h5'.

## grad_cam.py

Script que recibe la imagen y la procesa, carga el modelo, obtiene la predicción y la capa convolucional de interés para obtener las características relevantes de la imagen.

---

## Acerca del Modelo

La red neuronal convolucional implementada (CNN) es basada en el modelo implementado por F. Pasa, V.Golkov, F. Pfeifer, D. Cremers & D. Pfeifer
en su artículo Efcient Deep Network Architectures for Fast Chest X-Ray Tuberculosis Screening and Visualization.

Está compuesta por 5 bloques convolucionales, cada uno contiene 3 convoluciones; dos secuenciales y una conexión 'skip' que evita el desvanecimiento del gradiente a medida que se avanza en profundidad.
Con 16, 32, 48, 64 y 80 filtros de 3x3 para cada bloque respectivamente.

Después de cada bloque convolucional se encuentra una capa de max pooling y después de la última una capa de Average Pooling seguida por tres capas fully-connected (Dense) de 1024, 1024 y 3 neuronas respectivamente.

Para regularizar el modelo utilizamos 3 capas de Dropout al 20%; dos en los bloques 4 y 5 conv y otra después de la 1ra capa Dense.

## Acerca de Grad-CAM

Es una técnica utilizada para resaltar las regiones de una imagen que son importantes para la clasificación. Un mapeo de activaciones de clase para una categoría en particular indica las regiones de imagen relevantes utilizadas por la CNN para identificar esa categoría.

Grad-CAM realiza el cálculo del gradiente de la salida correspondiente a la clase a visualizar con respecto a las neuronas de una cierta capa de la CNN. Esto permite tener información de la importancia de cada neurona en el proceso de decisión de esa clase en particular. Una vez obtenidos estos pesos, se realiza una combinación lineal entre el mapa de activaciones de la capa y los pesos, de esta manera, se captura la importancia del mapa de activaciones para la clase en particular y se ve reflejado en la imagen de entrada como un mapa de calor con intensidades más altas en aquellas regiones relevantes para la red con las que clasificó la imagen en cierta categoría.

## licencia:
Este proyecto se distribuye bajo la licencia MIT. Consulte el archivo LICENSE para más información.

## Integrantes:

ALEXANDER CALAMBAS RAMIREZ
OSCAR PORTELA OSPINA
SEBASTIAN TORRES CABRERA
ANGELO PARRA CORTEZ
