import os
import numpy as np

import cv2
import pytest
from PIL import Image
from unittest.mock import MagicMock, patch

from src.read_img import load_image_file

@pytest.fixture
def create_temp_image():
    """Crea una imagen JPG de prueba."""
    path = "test_sample.jpg"
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(path, img)
    yield path
    if os.path.exists(path):
        os.remove(path)

# Prueba 1: Validación de imagen estándar (JPG/PNG) y preparación para GUI
def test_load_standard_image(create_temp_image):
    arr, img_pil = load_image_file(create_temp_image)
    
    # Valida conversión a NumPy (IA)
    assert isinstance(arr, np.ndarray)
    # Valida preparación para GUI (PIL Image)
    assert isinstance(img_pil, Image.Image) 
    assert arr.shape == (100, 100)

# Prueba 2: Lectura de DICOM simulado
@patch('pydicom.dcmread')
def test_load_dicom_simulated(mock_dcmread):
    # Simulamos el comportamiento de un archivo DICOM
    mock_ds = MagicMock()
    mock_ds.pixel_array = np.random.randint(0, 4000, (128, 128), dtype=np.uint16)
    mock_dcmread.return_value = mock_ds
    
    # Se crea un archivo vacío con extensión .dcm para engañar al validador de ruta
    path = "test_fake.dcm"
    with open(path, "w") as f: f.write("dummy data")
    
    try:
        arr, img_pil = load_image_file(path)
        assert arr.shape == (128, 128)
        assert isinstance(img_pil, Image.Image)
        # Verifica que se haya normalizado a 8-bit (0-255)
        assert arr.max() <= 255
    finally:
        if os.path.exists(path):
            os.remove(path)

# Prueba 3: Manejo de errores
def test_invalid_format():
    path = "archivo_invalido.txt"
    with open(path, "w") as f: f.write("test")
    with pytest.raises(ValueError, match="Formato de archivo no soportado"):
        load_image_file(path)
    os.remove(path)