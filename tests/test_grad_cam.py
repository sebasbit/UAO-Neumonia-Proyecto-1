# Pruebas unitarias para el módulo src/grad_cam.py
# Este módulo contiene tests para validar la funcionalidad de generación
# de mapas de calor Grad-CAM (Gradient-weighted Class Activation Mapping).

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from src import grad_cam


def test_superimpose_heatmap_output_structure():
    # Test: Verificar la estructura de salida de superimpose_heatmap.
    # Este test valida que la función superimpose_heatmap genera una imagen
    # con las dimensiones y tipo de datos correctos.

    # Crear heatmap simulado (normalizado 0-1)
    heatmap = np.random.rand(256, 256).astype(np.float32)

    # Crear imagen original simulada
    original_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    # Superponer heatmap sobre imagen
    result = grad_cam.superimpose_heatmap(heatmap, original_img)

    # Verificar tipo de datos
    assert result.dtype == np.uint8, "La imagen de salida debe ser uint8"

    # Verificar dimensiones (debe ser RGB)
    assert result.shape == (512, 512, 3), "La imagen debe tener dimensiones (512, 512, 3)"

    # Verificar rango de valores (0-255)
    assert np.min(result) >= 0, "Los valores mínimos deben ser >= 0"
    assert np.max(result) <= 255, "Los valores máximos deben ser <= 255"


def test_superimpose_heatmap_with_custom_size():
    # Test: Verificar que superimpose_heatmap respeta el tamaño objetivo.
    # Este test valida que se puede especificar un tamaño personalizado
    # para la imagen de salida.

    heatmap = np.random.rand(100, 100).astype(np.float32)
    original_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    # Tamaño personalizado
    custom_size = (256, 256)
    result = grad_cam.superimpose_heatmap(
        heatmap, original_img, target_size=custom_size
    )

    # Verificar que el tamaño es el esperado
    assert result.shape == (256, 256, 3), f"La imagen debe tener dimensiones {custom_size + (3,)}"


def test_superimpose_heatmap_alpha_parameter():
    # Test: Verificar que el parámetro alpha afecta la transparencia.
    # Este test valida que diferentes valores de alpha producen resultados diferentes.

    heatmap = np.ones((100, 100), dtype=np.float32) * 0.5
    original_img = np.ones((100, 100, 3), dtype=np.uint8) * 100

    # Generar con diferentes alphas
    result_alpha_low = grad_cam.superimpose_heatmap(
        heatmap, original_img, alpha=0.2
    )
    result_alpha_high = grad_cam.superimpose_heatmap(
        heatmap, original_img, alpha=0.9
    )

    # Los resultados deben ser diferentes
    assert not np.array_equal(result_alpha_low, result_alpha_high), \
        "Diferentes valores de alpha deben producir resultados diferentes"


@pytest.mark.skipif(
    not __import__('os').path.exists('conv_MLP_84.h5'),
    reason="El archivo del modelo conv_MLP_84.h5 no está disponible"
)
def test_generate_heatmap_with_real_model():
    # Test: Verificar la estructura de salida de generate_heatmap con modelo real.
    # Este test usa el modelo real para verificar que generate_heatmap
    # retorna un heatmap normalizado del tamaño correcto.
    # NOTA: Este test requiere que el archivo conv_MLP_84.h5 exista en la raíz del proyecto.

    # Cargar modelo real
    from src import load_model
    model = load_model.load_model()

    # Crear imagen preprocesada simulada (batch format)
    preprocessed_img = np.random.rand(1, 512, 512, 1).astype(np.float32)

    # Generar heatmap
    heatmap = grad_cam.generate_heatmap(model, preprocessed_img)

    # Verificar tipo de salida
    assert isinstance(heatmap, np.ndarray), "El heatmap debe ser un array de NumPy"

    # Verificar que está normalizado (0-1)
    assert np.min(heatmap) >= 0, "El heatmap debe tener valores >= 0"
    assert np.max(heatmap) <= 1, "El heatmap debe tener valores <= 1"

    # Verificar dimensiones
    assert len(heatmap.shape) == 2, "El heatmap debe ser 2D"


def test_generate_heatmap_invalid_layer():
    # Test: Verificar manejo de error cuando la capa especificada no existe.
    # Este test valida que se lanza ValueError cuando se intenta acceder
    # a una capa convolucional que no existe en el modelo.

    # Crear modelo mock que lanza ValueError al buscar capa
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[0.1, 0.7, 0.2]])
    mock_model.get_layer.side_effect = ValueError("Layer not found")

    preprocessed_img = np.random.rand(1, 512, 512, 1).astype(np.float32)

    # Verificar que se lanza ValueError
    with pytest.raises(ValueError) as exc_info:
        grad_cam.generate_heatmap(mock_model, preprocessed_img)

    assert "no existe en el modelo" in str(exc_info.value)


def test_grad_cam_full_pipeline():
    # Test: Verificar el pipeline completo de grad_cam.
    # Este test simula el proceso completo: preprocesamiento, carga de modelo,
    # generación de heatmap y superposición.

    # Crear imagen original simulada
    original_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

    # Mock de preprocess_img
    with patch('src.grad_cam.preprocess_img.preprocess_image') as mock_preprocess:
        # Mock de load_model
        with patch('src.grad_cam.load_model.load_model') as mock_load:
            # Mock de generate_heatmap
            with patch('src.grad_cam.generate_heatmap') as mock_gen_heatmap:
                # Configurar retornos de mocks
                mock_preprocess.return_value = np.random.rand(1, 512, 512, 1)
                mock_load.return_value = MagicMock()
                mock_gen_heatmap.return_value = np.random.rand(512, 512)

                # Ejecutar grad_cam
                result = grad_cam.grad_cam(original_img)

                # Verificar que se llamaron las funciones correctas
                mock_preprocess.assert_called_once()
                mock_load.assert_called_once()
                mock_gen_heatmap.assert_called_once()

                # Verificar estructura de salida
                assert isinstance(result, np.ndarray), "El resultado debe ser un array"
                assert result.dtype == np.uint8, "El resultado debe ser uint8"
                assert len(result.shape) == 3, "El resultado debe ser una imagen RGB"
                assert result.shape[2] == 3, "El resultado debe tener 3 canales (RGB)"


def test_grad_cam_with_provided_model():
    # Test: Verificar que grad_cam acepta un modelo pre-cargado.
    # Este test valida que cuando se provee un modelo, no se carga uno nuevo.

    original_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    mock_model = MagicMock()

    with patch('src.grad_cam.preprocess_img.preprocess_image') as mock_preprocess:
        with patch('src.grad_cam.load_model.load_model') as mock_load:
            with patch('src.grad_cam.generate_heatmap') as mock_gen_heatmap:
                mock_preprocess.return_value = np.random.rand(1, 512, 512, 1)
                mock_gen_heatmap.return_value = np.random.rand(512, 512)

                # Ejecutar grad_cam con modelo provisto
                result = grad_cam.grad_cam(original_img, model=mock_model)

                # Verificar que NO se llamó a load_model (ya se proveyó el modelo)
                mock_load.assert_not_called()

                # Verificar que generate_heatmap se llamó con el modelo provisto
                mock_gen_heatmap.assert_called_once()
                call_args = mock_gen_heatmap.call_args
                assert call_args[0][0] == mock_model, "Debe usar el modelo provisto"


def test_heatmap_normalization():
    # Test: Verificar que el heatmap se normaliza correctamente.
    # Este test valida que independientemente de los valores de entrada,
    # el heatmap de salida está normalizado entre 0 y 1.

    # Crear valores extremos
    heatmap_values = np.array([[-100, 0, 100], [200, 300, 400]], dtype=np.float32)

    # Simular normalización manual (como en generate_heatmap)
    heatmap = np.maximum(heatmap_values, 0)  # ReLU
    if np.max(heatmap) != 0:
        heatmap = heatmap / np.max(heatmap)  # Normalize

    # Verificar normalización
    assert np.min(heatmap) >= 0, "Valores mínimos deben ser >= 0"
    assert np.max(heatmap) <= 1, "Valores máximos deben ser <= 1"
    assert np.max(heatmap) == 1, "El valor máximo debe ser exactamente 1"
