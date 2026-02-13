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

### Uso de la Interfaz Gráfica:

La interfaz implementa un **flujo de trabajo secuencial** que guía al usuario paso a paso, habilitando botones progresivamente para prevenir errores:

#### Flujo de Trabajo:

**1. Estado Inicial (al abrir la aplicación):**
   - Solo el botón **"Cargar Imagen"** está habilitado
   - Todos los demás controles están deshabilitados

**2. Cargar Imagen:**
   - Presione el botón **"Cargar Imagen"**
   - Seleccione una imagen de rayos X del explorador de archivos (formatos: `.dcm`, `.jpg`, `.jpeg`, `.png`)
   - Imágenes de prueba disponibles en: https://drive.google.com/drive/folders/1WOuL0wdVC6aojy8IfssHcqZ4Up14dy0g?usp=drive_link
   - La imagen se visualizará en el panel izquierdo
   - Se habilitará automáticamente el campo **"Cédula Paciente"**

**3. Ingresar Cédula del Paciente:**
   - El campo de cédula ahora está habilitado (con foco automático)
   - Ingrese el número de identificación del paciente
   - El botón **"Predecir"** se habilitará automáticamente al ingresar texto

**4. Ejecutar Predicción:**
   - Presione el botón **"Predecir"**
   - Espere unos segundos mientras el sistema:
     - Preprocesa la imagen (resize, CLAHE, normalización)
     - Ejecuta el modelo CNN WilhemNet86
     - Genera el mapa de calor Grad-CAM
   - Se mostrarán los resultados:
     - **Panel derecho:** Imagen con mapa de calor superpuesto
     - **Resultado:** Diagnóstico (bacteriana / normal / viral)
     - **Probabilidad:** Porcentaje de confianza del modelo
   - Se habilitarán los botones **"Guardar"**, **"PDF"** y **"Borrar"**

**5. Guardar Resultados:**
   - Presione el botón **"Guardar"**
   - Los datos se almacenarán en `historial.csv` (formato: cédula-diagnóstico-probabilidad)
   - Se mostrará un mensaje con la ruta del archivo guardado

**6. Generar Reporte PDF:**
   - Presione el botón **"PDF"**
   - Se generará un archivo PDF con la captura de pantalla de los resultados
   - Se mostrará un mensaje con la ruta del archivo generado
   - **Nota para macOS:** Si aparece un error, verifique los permisos de captura de pantalla en:
     - *Configuración del Sistema > Privacidad y Seguridad > Grabación de Pantalla*
     - Active los permisos para Terminal o Python

**7. Limpiar / Nueva Consulta:**
   - Presione el botón **"Borrar"**
   - Se limpiará toda la información de la interfaz
   - El flujo se reiniciará al estado inicial (solo "Cargar Imagen" habilitado)
   - Listo para procesar un nuevo paciente

#### Notas Importantes:
- Los botones se habilitan **progresivamente** según el flujo
- No puede predecir sin antes cargar una imagen e ingresar cédula
- No puede guardar o generar PDF sin antes ejecutar una predicción
- El botón "Borrar" resetea completamente la aplicación al estado inicial

---

## Ejecución de Tests

Para ejecutar los tests del proyecto, abra la terminal en el directorio del proyecto y ejecute el siguiente comando:

  ```bash
  uv run pytest
  ```

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

- **ALEXANDER CALAMBAS RAMIREZ** 
- **OSCAR PORTELA OSPINA** 
- **SEBASTIAN TORRES CABRERA** 
- **ANGELO PARRA CORTEZ** 

---

## Documentación Detallada de Scripts

### 1. detector_neumonia.py (Aplicación Principal)

**Descripción:** Interfaz gráfica de usuario (GUI) desarrollada con Tkinter para el diagnóstico de neumonía.

**Funciones principales:**

- `predict(array)`: Pipeline completo de predicción
  - Preprocesa la imagen usando preprocess_img.preprocess_image()
  - Carga el modelo usando load_model.load_model()
  - Realiza la predicción (clasificación en 3 clases: bacteriana, normal, viral)
  - Calcula la probabilidad de la predicción
  - Genera el mapa de calor Grad-CAM usando grad_cam.grad_cam()
  - Retorna: (etiqueta, probabilidad, heatmap)

**Clase App:**

Métodos:
- `__init__()`: Inicializa la interfaz gráfica con dimensiones 815x560 píxeles
- `load_img_file()`: Abre explorador de archivos, carga imagen DICOM/JPG/PNG usando read_img.load_image_file()
- `run_model()`: Ejecuta predict() y muestra resultados en la GUI
- `save_results_csv()`: Guarda cédula, diagnóstico y probabilidad en historial.csv
- `create_pdf()`: Captura pantalla y genera reporte PDF
- `delete()`: Limpia la interfaz para nuevo análisis

**Componentes GUI:**
- Campos de entrada: Cédula del paciente
- Visualización: Imagen original + Grad-CAM superpuesto
- Botones: Cargar Imagen, Predecir, Guardar, PDF, Borrar

---

### 2. src/read_img.py (Lectura de Imágenes Médicas)

**Descripción:** Módulo especializado en lectura unificada de archivos médicos DICOM y estándar (JPG/PNG).

**Funciones:**

- `load_image_file(filepath)`: Función principal unificada
  - Detecta automáticamente el tipo de archivo por extensión
  - Llama a _handle_dicom() o _handle_standard() según corresponda
  - Retorna: (array_numpy, imagen_pil) para procesamiento y visualización

- `_handle_dicom(path)`: Procesamiento de archivos DICOM (.dcm)
  - Lee usando pydicom
  - Normaliza valores Min-Max a escala 0-255 (8 bits)
  - Convierte a RGB para visualización (repite canal en las 3 dimensiones)
  - Retorna: (array_normalizado, imagen_pil)

- `_handle_standard(path)`: Procesamiento de imágenes estándar (.jpg, .jpeg, .png)
  - Lee con OpenCV (cv2.imread)
  - Normalización Min-Max a escala 0-255
  - Conversión de BGR a RGB (OpenCV lee en BGR)
  - Retorna: (array_rgb, imagen_pil)

**Manejo de errores:**
- FileNotFoundError: Archivo no encontrado
- ValueError: Formato de archivo no soportado

**Formatos soportados:** .dcm, .jpg, .jpeg, .png

---

### 3. src/preprocess_img.py (Preprocesamiento de Imágenes)

**Descripción:** Pipeline modular de preprocesamiento para preparar imágenes para el modelo CNN.

**Funciones especializadas:**

- `resize_image(img, target_size=(512, 512))`: Redimensiona imagen
  - Usa cv2.resize con interpolación INTER_AREA
  - Retorna: array redimensionado

- `convert_to_grayscale(img)`: Convierte a escala de grises
  - Detecta si la imagen ya está en grises
  - Convierte RGB a escala de grises usando cv2.cvtColor(COLOR_RGB2GRAY)
  - Retorna: array en escala de grises

- `apply_clahe(img, clip_limit=2.0, tile_grid_size=(4, 4))`: Ecualización de histograma adaptativo
  - CLAHE (Contrast Limited Adaptive Histogram Equalization)
  - Mejora el contraste local de la imagen
  - Parámetros: clip_limit controla el nivel de contraste, tile_grid_size define el tamaño de las regiones
  - Retorna: imagen con contraste mejorado

- `normalize_image(img)`: Normaliza valores
  - Escala valores de [0, 255] a [0.0, 1.0]
  - Convierte a tipo float64
  - Retorna: array normalizado

- `expand_dims_for_model(img)`: Expande dimensiones para formato de batch
  - Imágenes en escala de grises: (H, W) → (H, W, 1) → (1, H, W, 1)
  - Imágenes a color: (H, W, C) → (1, H, W, C)
  - Primera dimensión = batch size (siempre 1 para inferencia)
  - Retorna: tensor en formato batch

- `preprocess_image(img)`: Pipeline completo de preprocesamiento
  - Ejecuta secuencialmente: resize → grayscale → CLAHE → normalize → expand_dims
  - Retorna: tensor (1, 512, 512, 1) listo para el modelo

**Flujo completo:** Imagen original → 512x512 → Escala de grises → CLAHE → Normalización [0,1] → Tensor batch

---

### 4. src/load_model.py (Carga del Modelo CNN)

**Descripción:** Módulo para carga segura del modelo pre-entrenado WilhemNet86 con validaciones de integridad.

**Funciones:**

- `load_model(model_filename='conv_MLP_84.h5')`: Carga del modelo con validaciones completas
  - **Validación 1:** Verifica que el archivo existe en la ruta del proyecto
  - **Validación 2:** Verifica extensión válida (.h5, .keras, .pb)
  - **Validación 3:** Verifica que el archivo no está vacío (tamaño > 0 bytes)
  - **Validación 4:** Intenta cargar el modelo usando tf.keras.models.load_model(compile=False)
  - **Validación 5:** Verifica que el modelo tiene el método 'predict' requerido
  - Manejo robusto de errores con mensajes descriptivos
  - Retorna: modelo de Keras listo para inferencia

- `model_fun()`: Función de compatibilidad con código legacy
  - Wrapper para load_model() con firma simplificada
  - Mantiene compatibilidad con detector_neumonia.py original
  - Retorna: modelo de Keras

**Excepciones manejadas:**
- FileNotFoundError: Modelo no encontrado
- ValueError: Archivo corrupto o extensión inválida
- OSError: Problemas de permisos o lectura

**Ubicación del modelo:** Raíz del proyecto / conv_MLP_84.h5 (112 MB)

---

### 5. src/grad_cam.py (Visualización Explicativa)

**Descripción:** Implementación de Grad-CAM (Gradient-weighted Class Activation Mapping) para explicabilidad visual de predicciones usando TensorFlow 2.x API moderna.

**Funciones:**

- `generate_heatmap(model, preprocessed_img, last_conv_layer_name="conv10_thisone")`: Genera mapa de calor
  - **Paso 1:** Verifica que la capa convolucional existe en el modelo
  - **Paso 2:** Crea modelo auxiliar que retorna activaciones + predicciones
  - **Paso 3:** Convierte imagen a tensor TensorFlow si es necesario
  - **Paso 4:** Usa tf.GradientTape para calcular gradientes de la clase predicha respecto a activaciones
  - **Paso 5:** Promedia gradientes espacialmente (Global Average Pooling)
  - **Paso 6:** Pondera cada canal de activaciones por su gradiente correspondiente
  - **Paso 7:** Promedia canales ponderados para crear heatmap
  - **Paso 8:** Aplica ReLU (mantiene solo influencias positivas)
  - **Paso 9:** Normaliza heatmap a rango [0, 1]
  - Retorna: heatmap normalizado (H, W)

  **Tecnología:** tf.GradientTape (compatible con eager execution y Keras 3.x)

- `superimpose_heatmap(heatmap, original_img, target_size=(512, 512), alpha=0.8)`: Superpone heatmap sobre imagen
  - Redimensiona heatmap al tamaño objetivo
  - Convierte heatmap a escala 0-255 (uint8)
  - Aplica colormap JET (azul=baja activación, rojo=alta activación)
  - Redimensiona imagen original
  - Aplica transparencia al heatmap (alpha=0.8 → 80% visible)
  - Superpone heatmap sobre imagen usando cv2.add()
  - Convierte de BGR a RGB
  - Retorna: imagen RGB con heatmap superpuesto

- `grad_cam(array, model=None)`: Pipeline completo de Grad-CAM
  - Preprocesa imagen usando preprocess_img.preprocess_image()
  - Carga modelo si no se provee
  - Genera heatmap usando generate_heatmap()
  - Superpone heatmap usando superimpose_heatmap()
  - Retorna: imagen RGB con visualización explicativa

**Interpretación:** Las áreas brillantes (rojas/amarillas) indican regiones de mayor importancia para la clasificación del modelo.

---

### 6. tests/test_read_img.py (Tests de Lectura)

**Descripción:** Suite de pruebas unitarias para validar la lectura de imágenes médicas.

**Tests implementados (3):**

- `test_load_standard_image()`: Valida carga de JPG/PNG
  - Verifica retorno de NumPy array y PIL Image
  - Verifica dimensiones correctas

- `test_load_dicom_simulated()`: Simula lectura de DICOM con mocks
  - Usa mock de pydicom
  - Verifica normalización a 8-bit
  - Valida conversión a RGB

- `test_invalid_format()`: Manejo de formatos inválidos
  - Verifica excepción ValueError para archivos no soportados

**Cobertura:** Lectura DICOM, lectura estándar, manejo de errores

---

### 7. tests/test_preprocess_img.py (Tests de Preprocesamiento)

**Descripción:** Suite de pruebas para validar el pipeline de preprocesamiento.

**Tests implementados (6):**

- `test_resize_image()`: Verifica cambio de dimensiones a 512x512
- `test_convert_to_grayscale_color_image()`: Valida conversión RGB a escala de grises
- `test_apply_clahe()`: Verifica aplicación de CLAHE (mantiene forma y tipo)
- `test_normalize_image()`: Valida normalización a rango [0, 1]
- `test_expand_dims_for_model_grayscale()`: Verifica expansión de dimensiones (1, 512, 512, 1)
- `test_preprocess_image_pipeline()`: Valida pipeline completo
  - Verifica forma final (1, 512, 512, 1)
  - Verifica tipo float64
  - Verifica rango [0, 1]

**Cobertura:** Todas las funciones de preprocesamiento + pipeline integrado

---

### 8. tests/test_load_model.py (Tests de Carga de Modelo)

**Descripción:** Suite de pruebas para validar la carga segura del modelo CNN.

**Tests implementados (8):**

- `test_load_model_success()`: Carga exitosa del modelo real
  - Requiere archivo conv_MLP_84.h5
  - Verifica modelo no es None
  - Verifica existencia de método 'predict'

- `test_load_model_file_not_found()`: Error cuando modelo no existe
- `test_load_model_invalid_extension()`: Validación de extensiones (.h5, .keras, .pb)
- `test_load_model_empty_file()`: Detección de archivos vacíos
- `test_model_fun_compatibility()`: Compatibilidad con función legacy
- `test_load_model_corrupted_file()`: Manejo de archivos corruptos
- `test_load_model_returns_none()`: Validación cuando load retorna None
- `test_load_model_no_predict_method()`: Validación de método 'predict'

**Cobertura:** Carga exitosa, validaciones de integridad, manejo de errores

---

### 9. tests/test_grad_cam.py (Tests de Grad-CAM)

**Descripción:** Suite de pruebas para validar la generación de mapas de calor explicativos.

**Tests implementados (8):**

- `test_superimpose_heatmap_output_structure()`: Valida estructura de salida
  - Tipo uint8
  - Dimensiones (512, 512, 3)
  - Rango [0, 255]

- `test_superimpose_heatmap_with_custom_size()`: Tamaños personalizados
- `test_superimpose_heatmap_alpha_parameter()`: Efecto del parámetro de transparencia
- `test_generate_heatmap_with_real_model()`: Generación con modelo real
  - Requiere conv_MLP_84.h5
  - Valida heatmap normalizado [0, 1]
  - Verifica dimensiones 2D

- `test_generate_heatmap_invalid_layer()`: Error cuando capa no existe
- `test_grad_cam_full_pipeline()`: Pipeline completo con mocks
- `test_grad_cam_with_provided_model()`: Uso de modelo pre-cargado
- `test_heatmap_normalization()`: Validación de normalización

**Cobertura:** Generación de heatmap, superposición, pipeline completo, manejo de errores

---

**Comando para ejecutar tests:**
```bash
uv run pytest tests/ -v
```
