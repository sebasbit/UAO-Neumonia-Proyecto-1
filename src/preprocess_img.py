import cv2
import numpy as np


def resize_image(image_array, size=(512, 512)):
    """
    Redimensiona una imagen a un tamaño específico.

    Args:
        image_array (np.array): La imagen de entrada como un arreglo NumPy.
        size (tuple): El tamaño deseado (ancho, alto).

    Returns:
        np.array: La imagen redimensionada.
    """
    return cv2.resize(image_array, size)


def convert_to_grayscale(image_array):
    """
    Convierte una imagen a escala de grises.

    Args:
        image_array (np.array): La imagen de entrada como un arreglo NumPy.

    Returns:
        np.array: La imagen en escala de grises.
    """
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:  # Check if it's a color image
        return cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    return image_array


def apply_clahe(image_array, clip_limit=2.0, tile_grid_size=(4, 4)):
    """
    Aplica Contraste Adaptativo de Histograma Limitado (CLAHE) a una imagen.

    Args:
        image_array (np.array): La imagen de entrada como un arreglo NumPy (debe ser en escala de grises).
        clip_limit (float): Umbral para limitar el contraste.
        tile_grid_size (tuple): Tamaño de la cuadrícula para el histograma adaptativo.

    Returns:
        np.array: La imagen con CLAHE aplicado.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image_array)


def normalize_image(image_array):
    """
    Normaliza los valores de píxel de la imagen a un rango de 0-1.

    Args:
        image_array (np.array): La imagen de entrada como un arreglo NumPy.

    Returns:
        np.array: La imagen normalizada.
    """
    return image_array / 255.0


def expand_dims_for_model(image_array):
    """
    Expande las dimensiones de la imagen para que sea compatible con la entrada del modelo
    (añade dimensión de batch y de canal si es necesario).

    Args:
        image_array (np.array): La imagen de entrada como un arreglo NumPy.

    Returns:
        np.array: La imagen con dimensiones expandidas.
    """
    if len(image_array.shape) == 2:
        # (H, W) -> (H, W, 1) -> (1, H, W, 1)
        image_array = np.expand_dims(image_array, axis=-1)
        image_array = np.expand_dims(image_array, axis=0)
    elif len(image_array.shape) == 3:
        # (H, W, C) -> (1, H, W, C)
        image_array = np.expand_dims(image_array, axis=0)
    return image_array


def preprocess_image(image_array):
    """
    Procesar el preprocesamiento para una imagen.

    Args:
        image_array (np.array): La imagen de entrada como un arreglo NumPy.

    Returns:
        np.array: La imagen preprocesada lista para el modelo.
    """
    resized_img = resize_image(image_array, size=(512, 512))
    gray_img = convert_to_grayscale(resized_img)
    clahe_img = apply_clahe(gray_img)
    normalized_img = normalize_image(clahe_img)
    final_img = expand_dims_for_model(normalized_img)
    return final_img
