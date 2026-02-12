# Pruebas unitarias para el módulo src/load_model.py
# Este módulo contiene tests para validar la funcionalidad de carga del modelo
# de red neuronal convolucional WilhemNet86.

import pytest
from unittest.mock import patch, MagicMock
from src import load_model


@pytest.mark.skipif(
    not __import__('os').path.exists('conv_MLP_84.h5'),
    reason="El archivo del modelo conv_MLP_84.h5 no está disponible"
)
def test_load_model_success():
    # Test: Verificar que el modelo se carga correctamente sin errores.
    # Este test valida que:
    # - La función load_model() se ejecuta sin excepciones
    # - El modelo retornado tiene el método 'predict' necesario para inferencia
    # - El modelo retornado no es None
    # NOTA: Este test requiere que el archivo conv_MLP_84.h5 exista en la raíz del proyecto.

    # Intentar cargar el modelo
    model = load_model.load_model()

    # Verificar que el modelo no es None
    assert model is not None, "El modelo cargado no debe ser None"

    # Verificar que el modelo tiene el método 'predict'
    assert hasattr(model, 'predict'), "El modelo debe tener el método 'predict'"

    # Verificar que es un objeto callable
    assert callable(model.predict), "El método 'predict' debe ser callable"


def test_load_model_file_not_found():
    # Test: Verificar que se lanza FileNotFoundError cuando el modelo no existe.
    # Este test valida el manejo de errores cuando se intenta cargar un modelo
    # que no existe en el sistema de archivos.

    with pytest.raises(FileNotFoundError) as exc_info:
        load_model.load_model(model_filename='modelo_inexistente.h5')

    # Verificar que el mensaje de error es descriptivo
    assert "no fue encontrado" in str(exc_info.value)


def test_load_model_invalid_extension():
    # Test: Verificar que se lanza ValueError para extensiones de archivo inválidas.
    # Este test valida que el módulo rechaza archivos con extensiones no soportadas.

    # Crear un archivo temporal con extensión inválida
    invalid_file = 'test_invalid.txt'

    # Usar patch para simular que el archivo existe pero tiene extensión incorrecta
    with patch('os.path.exists', return_value=True):
        with pytest.raises(ValueError) as exc_info:
            load_model.load_model(model_filename=invalid_file)

        # Verificar que el mensaje menciona extensión inválida
        assert "Extensión de archivo no válida" in str(exc_info.value)


def test_load_model_empty_file():
    # Test: Verificar que se lanza ValueError cuando el archivo del modelo está vacío.
    # Este test valida que el módulo detecta archivos vacíos que no pueden ser
    # modelos válidos de Keras/TensorFlow.

    # Usar patch para simular archivo existente pero vacío
    with patch('os.path.exists', return_value=True):
        with patch('os.path.getsize', return_value=0):
            with pytest.raises(ValueError) as exc_info:
                load_model.load_model(model_filename='empty_model.h5')

            # Verificar mensaje descriptivo
            assert "está vacío" in str(exc_info.value)


@pytest.mark.skipif(
    not __import__('os').path.exists('conv_MLP_84.h5'),
    reason="El archivo del modelo conv_MLP_84.h5 no está disponible"
)
def test_model_fun_compatibility():
    # Test: Verificar compatibilidad con la función legacy model_fun().
    # Este test valida que la función model_fun() (usada por código legacy)
    # funciona correctamente y retorna un modelo válido.
    # NOTA: Este test requiere que el archivo conv_MLP_84.h5 exista en la raíz del proyecto.

    # Llamar a la función de compatibilidad
    model = load_model.model_fun()

    # Verificar que retorna un modelo válido
    assert model is not None
    assert hasattr(model, 'predict')


def test_load_model_corrupted_file():
    # Test: Verificar manejo de archivos corruptos o inválidos.
    # Este test simula un archivo que existe pero no es un modelo válido de Keras,
    # verificando que se lanza un ValueError apropiado.

    # Crear un archivo temporal "corrupto" (no es un modelo real)
    corrupted_file = 'corrupted_model.h5'

    with patch('os.path.exists', return_value=True):
        with patch('os.path.getsize', return_value=1000):  # Archivo no vacío
            with patch('tensorflow.keras.models.load_model', side_effect=Exception("Corrupted")):
                with pytest.raises(ValueError) as exc_info:
                    load_model.load_model(model_filename=corrupted_file)

                # Verificar mensaje de error
                assert "Error al cargar el modelo" in str(exc_info.value)


def test_load_model_returns_none():
    # Test: Verificar manejo cuando tf.keras.models.load_model retorna None.
    # Este test cubre el caso edge donde load_model de Keras retorna None.

    with patch('os.path.exists', return_value=True):
        with patch('os.path.getsize', return_value=1000):
            with patch('tensorflow.keras.models.load_model', return_value=None):
                with pytest.raises(ValueError) as exc_info:
                    load_model.load_model()

                assert "modelo cargado es None" in str(exc_info.value)


def test_load_model_no_predict_method():
    # Test: Verificar validación de que el modelo tiene método 'predict'.
    # Este test valida que se rechaza un objeto que no tiene el método predict.

    # Crear un mock que no tiene método predict
    fake_model = MagicMock()
    del fake_model.predict

    with patch('os.path.exists', return_value=True):
        with patch('os.path.getsize', return_value=1000):
            with patch('tensorflow.keras.models.load_model', return_value=fake_model):
                with pytest.raises(ValueError) as exc_info:
                    load_model.load_model()

                assert "No tiene el método 'predict'" in str(exc_info.value)
