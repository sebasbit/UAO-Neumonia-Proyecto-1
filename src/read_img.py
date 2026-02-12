import os
import numpy as np

import pydicom
import cv2
from PIL import Image


def load_image_file(filepath: str):
    """
    Función unificada para cargar archivos médicos.
    Retorna un array de NumPy para procesamiento del modelo y 
    un objeto PIL para la interfaz grafica.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No se encuentra el archivo: {filepath}")

    extension = os.path.splitext(filepath)[1].lower()

    if extension == '.dcm':
        return _handle_dicom(filepath)
    elif extension in ['.jpg', '.jpeg', '.png']:
        return _handle_standard(filepath)
    else:
        raise ValueError(f"Formato no soportado: {extension}")


def _handle_dicom(path):
    ds = pydicom.dcmread(path)
    img_array = ds.pixel_array

    # Normalización Min-Max para visualización 8-bit
    min_val = np.min(img_array.astype(float))
    max_val = np.max(img_array.astype(float))
    if max_val - min_val == 0:
        img_norm = np.zeros_like(img_array)
    else:
        img_norm = (((img_array.astype(float) - min_val) / (max_val - min_val)) * 255).astype(np.uint8)
    img_rgb = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2RGB)
    return img_rgb, Image.fromarray(img_array)


def _handle_standard(path):
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise ValueError("Error al decodificar la imagen.")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_array = np.asarray(img_rgb)

    min_val = np.min(img_array.astype(float))
    max_val = np.max(img_array.astype(float))
    if max_val - min_val == 0:
        img_norm = np.zeros_like(img_array)
    else:
        img_norm = (((img_array.astype(float) - min_val) / (max_val - min_val)) * 255).astype(np.uint8)

    return img_norm, Image.fromarray(img_array)
