# Proyecto de Detección de Neumonía con IA

Este proyecto es una aplicación web diseñada para apoyar el diagnóstico de neumonía a partir de imágenes de rayos X. Utiliza una red neuronal convolucional (CNN) para clasificar las imágenes en tres categorías (Bacteriana, Normal, Viral) y emplea Grad-CAM para generar un mapa de calor que resalta las áreas de la imagen más relevantes para la predicción.

## Características

- **Clasificación de Neumonía:** Diagnóstico asistido por IA para neumonía bacteriana, viral o estado normal.
- **Visualización Explicativa (XAI):** Generación de mapas de calor (Grad-CAM) para interpretar las decisiones del modelo.
- **Interfaz Web Moderna:** UI intuitiva y fácil de usar, accesible desde cualquier navegador web moderno.
- **Soporte Multiformato:** Compatible con imágenes médicas `.dcm` (DICOM), `.jpg`, `.jpeg` y `.png`.
- **Carga Dinámica de Modelos:** Permite al usuario cargar su propio modelo `.h5`.
- **Generación de Reportes:** Exporta resultados a un archivo `.csv` y genera un reporte en formato `.pdf`.
- **Arquitectura Modular:** Código desacoplado en módulos para facilitar el mantenimiento y la escalabilidad.
- **Contenerización:** Listo para ser ejecutado en un entorno aislado con Docker.

## Instalación y Uso

A continuación se detallan los pasos para poner en marcha la aplicación.

### Requisitos Previos

- **Python 3.10** o superior.
- **uv:** Una herramienta rápida para la gestión de paquetes de Python. Instálala siguiendo la [guía oficial de instalación de uv](https://docs.astral.sh/uv/getting-started/installation/).

### Pasos de Ejecución

1.  **Clona el repositorio y navega al directorio del proyecto.**

2.  **Sincroniza las dependencias del proyecto:**
    Abra una terminal en el directorio raíz del proyecto y ejecute:
    ```bash
    uv sync
    ```
    > **Nota:** Si encuentras errores con TensorFlow en Windows, puedes instalar una versión compatible explícitamente:
    > `uv pip install tensorflow==2.15.0`

3.  **Ejecuta la aplicación web:**
    ```bash
    uv run python web.py
    ```

4.  **Abre tu navegador:**
    Accede a [http://127.0.0.1:8000](http://127.0.0.1:8000) para empezar a utilizar la aplicación.

### Flujo de la Interfaz Web

1.  **Cargar Modelo:** Haz clic en **"Cargar modelo"** para subir un archivo de modelo `.h5`. Sin un modelo, la predicción no funcionará.
2.  **Ingresar Cédula:** Introduce el número de identificación del paciente.
3.  **Cargar Imagen:** Selecciona una imagen de rayos X (`.dcm`, `.jpg`, etc.).
    - Puedes encontrar imágenes de prueba en [este enlace de Google Drive](https://drive.google.com/drive/folders/1WOuL0wdVC6aojy8IfssHcqZ4Up14dy0g?usp=drive_link).
4.  **Predecir:** Haz clic en **"Predecir"**. El sistema procesará la imagen, mostrará el resultado (diagnóstico y probabilidad) y el mapa de calor Grad-CAM.
5.  **Guardar y Exportar:**
    - **Guardar (CSV):** Almacena los resultados en `historial.csv`.
    - **Descargar PDF:** Genera un reporte en PDF con la información del diagnóstico.
    - **Borrar:** Limpia la interfaz para iniciar un nuevo análisis.

## Estructura del Proyecto

El proyecto está organizado en una arquitectura modular para separar responsabilidades:

```
UAO-Neumonia-Proyecto-1/
│
├── src/
│   ├── __init__.py
│   ├── grad_cam.py         # Lógica de Grad-CAM
│   ├── integrator.py       # Orquesta el pipeline de predicción
│   ├── load_model.py       # Utilidades para la carga de modelos
│   ├── preprocess_img.py   # Funciones de preprocesamiento de imágenes
│   └── read_img.py         # Lectura de imágenes DICOM y estándar
│
├── tests/
│   ├── test_grad_cam.py
│   ├── test_load_model.py
│   ├── test_preprocess_img.py
│   └── test_read_img.py
│
├── web.py                  # Aplicación principal (servidor web y UI)
├── detector_neumonia.py    # UI de escritorio original (Tkinter)
├── requirements.txt
├── Dockerfile              # Dockerfile para la interfaz web
├── Dockerfile-gui          # Dockerfile para la interfaz web
├── LICENSE
└── README.md
```

## Documentación de Scripts

### 1. `web.py` (Aplicación Principal)

Interfaz web de usuario para el diagnóstico de neumonía.

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

Módulo que integra todos los pasos del pipeline: carga de imagen, preprocesamiento, predicción y generación de Grad-CAM.

## Acerca del Modelo

La red neuronal convolucional implementada (CNN) es basada en el modelo implementado por F. Pasa, V.Golkov, F. Pfeifer, D. Cremers & D. Pfeifer en su artículo Efcient Deep Network Architectures for Fast Chest X-Ray Tuberculosis Screening and Visualization.

Está compuesta por 5 bloques convolucionales, cada uno contiene 3 convoluciones; dos secuenciales y una conexión 'skip' que evita el desvanecimiento del gradiente a medida que se avanza en profundidad. Con 16, 32, 48, 64 y 80 filtros de 3x3 para cada bloque respectivamente.

Después de cada bloque convolucional se encuentra una capa de max pooling y después de la última una capa de Average Pooling seguida por tres capas fully-connected (Dense) de 1024, 1024 y 3 neuronas respectivamente.

Para regularizar el modelo utilizamos 3 capas de Dropout al 20%; dos en los bloques 4 y 5 conv y otra después de la 1ra capa Dense.

## Acerca de Grad-CAM

Es una técnica utilizada para resaltar las regiones de una imagen que son importantes para la clasificación. Un mapeo de activaciones de clase para una categoría en particular indica las regiones de imagen relevantes utilizadas por la CNN para identificar esa categoría.

Grad-CAM realiza el cálculo del gradiente de la salida correspondiente a la clase a visualizar con respecto a las neuronas de una cierta capa de la CNN. Esto permite tener información de la importancia de cada neurona en el proceso de decisión de esa clase en particular. Una vez obtenidos estos pesos, se realiza una combinación lineal entre el mapa de activaciones de la capa y los pesos, de esta manera, se captura la importancia del mapa de activaciones para la clase en particular y se ve reflejado en la imagen de entrada como un mapa de calor con intensidades más altas en aquellas regiones relevantes para la red con las que clasificó la imagen en cierta categoría.

## Ejecución de Tests

Para asegurar la calidad y el correcto funcionamiento de los módulos, puedes ejecutar la suite de pruebas automatizadas.

```bash
uv run pytest tests/ -v
```

## Docker

Puedes ejecutar la aplicación web dentro de un contenedor Docker para un despliegue aislado, consistente y multiplataforma.

### Requisitos Previos

- **Docker Desktop** instalado y corriendo en tu sistema
  - macOS: Descarga desde [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/)
  - Windows: Descarga desde [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/)
  - Linux: Instala Docker Engine siguiendo la [documentación oficial](https://docs.docker.com/engine/install/)

### Instrucciones de Uso

#### 1. Navegar al directorio del proyecto

Abre una terminal y navega a la ruta donde se encuentra el `Dockerfile`:

```bash
cd /ruta/a/UAO-Neumonia-Proyecto-1
```

Verifica que estés en el directorio correcto:
```bash
ls Dockerfile
```

#### 2. Construir la imagen Docker

Ejecuta el siguiente comando para construir la imagen (esto puede tomar varios minutos la primera vez):

```bash
docker build -t neumonia-detector:latest .
```

**Nota:** El punto (`.`) al final es importante, indica que el contexto de construcción es el directorio actual.

#### 3. Ejecutar el contenedor

**Opción A - Ejecutar en primer plano (ver logs en tiempo real):**
```bash
docker run --rm -p 8000:8000 --name neumonia-app neumonia-detector:latest
```

**Opción B - Ejecutar en segundo plano (modo detached):**
```bash
docker run -d -p 8000:8000 --name neumonia-app neumonia-detector:latest
```

#### 4. Acceder a la aplicación

Abre tu navegador web y visita:
```
http://localhost:8000
```

La interfaz web del detector de neumonía estará disponible.

### Comandos Útiles

**Ver contenedores en ejecución:**
```bash
docker ps
```

**Ver todos los contenedores (incluidos los detenidos):**
```bash
docker ps -a
```

**Ver logs del contenedor:**
```bash
docker logs neumonia-app

# Ver logs en tiempo real
docker logs -f neumonia-app
```

**Detener el contenedor:**
```bash
docker stop neumonia-app
```

**Reiniciar el contenedor:**
```bash
docker restart neumonia-app
```

**Eliminar el contenedor:**
```bash
# Primero detenerlo si está corriendo
docker stop neumonia-app

# Luego eliminarlo
docker rm neumonia-app
```

**Eliminar la imagen:**
```bash
docker rmi neumonia-detector:latest
```

### Características del Contenedor

- **Puerto expuesto:** 8000
- **Plataforma:** Linux (compatible con macOS ARM, Windows, Linux)
- **TensorFlow:** Versión optimizada para CPU (`tensorflow-cpu-aws` en ARM)
- **OpenCV:** `opencv-python-headless` (sin GUI, optimizado para contenedores)
- **Tamaño de imagen:** ~2.5 GB (incluye TensorFlow y todas las dependencias)

### Solución de Problemas

**El contenedor no inicia:**
1. Verifica que Docker Desktop esté corriendo
2. Revisa los logs: `docker logs neumonia-app`
3. Verifica que el puerto 8000 no esté ocupado: `lsof -i :8000` (macOS/Linux)

**Error "port is already allocated":**
- Otro proceso está usando el puerto 8000
- Usa un puerto diferente: `docker run -d -p 8080:8000 --name neumonia-app neumonia-detector:latest`
- Accede en: `http://localhost:8080`

**Reconstruir después de cambios en el código:**
```bash
# Detener y eliminar contenedor anterior
docker stop neumonia-app && docker rm neumonia-app

# Reconstruir imagen sin caché
docker build --no-cache -t neumonia-detector:latest .

# Ejecutar nuevo contenedor
docker run -d -p 8000:8000 --name neumonia-app neumonia-detector:latest
```

## Contribuidores

- **ALEXANDER CALAMBAS RAMIREZ**
- **OSCAR PORTELA OSPINA**
- **SEBASTIAN TORRES CABRERA**
- **ANGELO PARRA CORTEZ**

## Licencia

Este proyecto se distribuye bajo la licencia MIT. Consulta el archivo `LICENSE` para más información.
