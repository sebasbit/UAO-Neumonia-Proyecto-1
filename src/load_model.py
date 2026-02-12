# Módulo para la carga del modelo de red neuronal convolucional WilhemNet86.
# Este módulo proporciona funciones para cargar el modelo pre-entrenado
# utilizado en la clasificación de neumonía (bacteriana, normal, viral).

import os
import tensorflow as tf


def load_model(model_filename='conv_MLP_84.h5'):
    # Carga el modelo pre-entrenado de Keras/TensorFlow con validación de integridad.
    #
    # Args:
    #     model_filename (str): Nombre del archivo del modelo. Por defecto 'conv_MLP_84.h5'.
    #
    # Returns:
    #     tf.keras.Model: Modelo de Keras cargado y listo para predicción.
    #
    # Raises:
    #     FileNotFoundError: Si el archivo del modelo no existe.
    #     ValueError: Si el archivo no es un modelo válido de Keras/TensorFlow.
    #     OSError: Si hay problemas de permisos o lectura del archivo.
    #
    # Example:
    #     model = load_model()
    #     prediction = model.predict(preprocessed_image)

    # Construir ruta absoluta del modelo (asume que está en la raíz del proyecto)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, model_filename)

    # Validación 1: Verificar que el archivo existe
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"El archivo del modelo no fue encontrado en: {model_path}\n"
            f"Asegúrate de que '{model_filename}' está en la raíz del proyecto."
        )

    # Validación 2: Verificar que el archivo tiene extensión válida
    valid_extensions = ['.h5', '.keras', '.pb']
    file_extension = os.path.splitext(model_path)[1].lower()
    if file_extension not in valid_extensions:
        raise ValueError(
            f"Extensión de archivo no válida: {file_extension}\n"
            f"Extensiones válidas: {', '.join(valid_extensions)}"
        )

    # Validación 3: Verificar que el archivo no está vacío
    file_size = os.path.getsize(model_path)
    if file_size == 0:
        raise ValueError(
            f"El archivo del modelo está vacío: {model_path}\n"
            f"Tamaño: {file_size} bytes"
        )

    # Intentar cargar el modelo con manejo de errores
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        raise ValueError(
            f"Error al cargar el modelo desde {model_path}:\n"
            f"El archivo puede estar corrupto o no ser un modelo válido de Keras.\n"
            f"Error original: {str(e)}"
        )

    # Validación 4: Verificar que el modelo cargado es válido
    if model is None:
        raise ValueError(
            f"El modelo cargado es None. Verifica la integridad del archivo: {model_path}"
        )

    # Validación 5: Verificar que el modelo tiene la estructura esperada (opcional)
    if not hasattr(model, 'predict'):
        raise ValueError(
            f"El objeto cargado no es un modelo de Keras válido.\n"
            f"No tiene el método 'predict' requerido."
        )

    return model


def model_fun():
    # Función de compatibilidad con el código legacy de detector_neumonia.py.
    # Esta función mantiene la misma firma que la versión original para
    # no romper la compatibilidad con código existente.
    #
    # Returns:
    #     tf.keras.Model: Modelo de Keras cargado.
    #
    # Example:
    #     model = model_fun()
    #     prediction = model.predict(image_array)

    return load_model()
