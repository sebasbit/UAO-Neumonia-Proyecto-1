import numpy as np
import pytest

from src import preprocess_img


@pytest.fixture
def dummy_image_color():
    """Devuelve una imagen de color simulada (3 canales)."""
    return np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)


@pytest.fixture
def dummy_image_grayscale():
    """Devuelve una imagen en escala de grises simulada (2 canales)."""
    return np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)


def test_resize_image(dummy_image_color):
    """Verifica que resize_image cambie correctamente las dimensiones de la imagen."""
    resized_img = preprocess_img.resize_image(dummy_image_color, size=(512, 512))
    assert resized_img.shape[:2] == (512, 512)
    assert resized_img.dtype == np.uint8  # Should maintain original dtype


def test_convert_to_grayscale_color_image(dummy_image_color):
    """Verifica la conversión de una imagen a color a escala de grises."""
    grayscale_img = preprocess_img.convert_to_grayscale(dummy_image_color)
    assert len(grayscale_img.shape) == 2 or (len(grayscale_img.shape) == 3 and grayscale_img.shape[2] == 1)
    assert grayscale_img.dtype == np.uint8


def test_apply_clahe(dummy_image_grayscale):
    """Verifica que apply_clahe devuelve una imagen con la misma forma y tipo."""
    clahe_img = preprocess_img.apply_clahe(dummy_image_grayscale)
    assert clahe_img.shape == dummy_image_grayscale.shape
    assert clahe_img.dtype == np.uint8  # CLAHE output is usually uint8


def test_normalize_image(dummy_image_grayscale):
    """Verifica que normalize_image escala los valores de píxel a 0-1 y cambia el tipo de dato a flotante."""
    normalized_img = preprocess_img.normalize_image(dummy_image_grayscale)
    assert normalized_img.min() >= 0.0
    assert normalized_img.max() <= 1.0


def test_expand_dims_for_model_grayscale(dummy_image_grayscale):
    """Verifica la expansión de dimensiones para una imagen en escala de grises (H, W) -> (1, H, W, 1)."""
    expanded_img = preprocess_img.expand_dims_for_model(dummy_image_grayscale)
    assert expanded_img.shape == (1, dummy_image_grayscale.shape[0], dummy_image_grayscale.shape[1], 1)


def test_preprocess_image_pipeline(dummy_image_color):
    """
    Verifica el pipeline de preprocesamiento con una imagen a color.
    Comprueba la forma final y el tipo de dato.
    """
    processed_img = preprocess_img.preprocess_image(dummy_image_color)
    assert processed_img.shape == (1, 512, 512, 1)
    assert processed_img.dtype == np.float64
    assert processed_img.min() >= 0.0
    assert processed_img.max() <= 1.0
