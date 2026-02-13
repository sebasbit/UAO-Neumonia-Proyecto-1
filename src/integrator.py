import numpy as np

from src import read_img, preprocess_img, load_model, grad_cam

_model = None  # Variable global para cachear el modelo y evitar recargas innecesarias.


def get_model():
    """
    Carga el modelo de Keras y lo almacena en una variable global.
    Si el modelo ya está cargado, lo devuelve directamente.

    Returns:
        tf.keras.Model: El modelo de red neuronal cargado.
    """
    global _model
    if _model is None:
        _model = load_model.load_model()
    return _model


def load_and_prepare_image(filepath: str):
    """
    Encapsula la lógica de lectura de `read_img` para ser usada por la GUI.
    Esta función carga un archivo de imagen (DICOM o estándar) y devuelve
    el array de la imagen para el modelo y una imagen para visualización.

    Args:
        filepath (str): La ruta al archivo de imagen.

    Returns:
        tuple: Una tupla conteniendo (array_para_modelo, imagen_para_mostrar).
               - array_para_modelo (np.array): La imagen como array NumPy, lista para preprocesar.
               - imagen_para_mostrar (PIL.Image): La imagen lista para ser mostrada en una GUI.
    """
    return read_img.load_image_file(filepath)


def predict_pneumonia(image_array: np.ndarray):
    """
    Orquesta el flujo completo de predicción de neumonía.

    Esta función toma un array de imagen, lo preprocesa, obtiene una predicción
    del modelo y genera un mapa de calor Grad-CAM para la explicabilidad.

    Args:
        image_array (np.array): El array de la imagen original obtenido de
                                load_and_prepare_image.

    Returns:
        tuple: Una tupla conteniendo (clase_predicha, probabilidad, imagen_heatmap).
               - clase_predicha (str): El nombre de la clase ('Bacteriana', 'Normal', 'Viral').
               - probabilidad (float): La confianza de la predicción (de 0.0 a 1.0).
               - imagen_heatmap (np.array): La imagen original con el heatmap Grad-CAM superpuesto.
    """
    model = get_model()
    preprocessed_tensor = preprocess_img.preprocess_image(image_array)
    prediction = model.predict(preprocessed_tensor)
    class_index = np.argmax(prediction, axis=1)[0]
    probability = float(np.max(prediction))
    class_labels = {0: 'Bacteriana', 1: 'Normal', 2: 'Viral'}
    predicted_class = class_labels.get(class_index, 'Desconocido')
    heatmap = grad_cam.generate_heatmap(model, preprocessed_tensor)
    superimposed_image = grad_cam.superimpose_heatmap(heatmap, image_array)
    return predicted_class, probability, superimposed_image
