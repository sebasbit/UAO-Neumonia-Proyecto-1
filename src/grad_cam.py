# Módulo para la generación de mapas de calor Grad-CAM.
# Este módulo implementa la técnica Gradient-weighted Class Activation Mapping (Grad-CAM)
# para visualizar las regiones de una imagen que son importantes para la clasificación
# del modelo de red neuronal convolucional.
# Grad-CAM proporciona explicabilidad visual mostrando qué áreas de la imagen médica
# influyeron más en la predicción del modelo.

import cv2
import numpy as np
import tensorflow as tf

from src import preprocess_img, load_model


def generate_heatmap(model, preprocessed_img, last_conv_layer_name="conv10_thisone"):
    # Genera el mapa de calor Grad-CAM para una imagen preprocesada.
    # Esta implementación usa tf.GradientTape (TensorFlow 2.x API) para compatibilidad
    # con eager execution y es más moderna que la implementación con K.gradients.
    #
    # Args:
    #     model: Modelo de Keras/TensorFlow cargado.
    #     preprocessed_img: Imagen preprocesada en formato batch (1, H, W, C).
    #     last_conv_layer_name (str): Nombre de la última capa convolucional.
    #                                  Por defecto "conv10_thisone" para WilhemNet86.
    #
    # Returns:
    #     numpy.ndarray: Heatmap normalizado (0-1) de tamaño (H, W).
    #
    # Raises:
    #     ValueError: Si la capa especificada no existe en el modelo.

    # Verificar que la capa convolucional existe
    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
    except ValueError:
        raise ValueError(
            f"La capa '{last_conv_layer_name}' no existe en el modelo.\n"
            f"Capas disponibles: {[layer.name for layer in model.layers]}"
        )

    # Crear modelo auxiliar que retorna las activaciones de la última capa conv y las predicciones
    grad_model = tf.keras.models.Model(
        inputs=[model.input],
        outputs=[last_conv_layer.output, model.output]
    )

    # Convertir a tensor si no lo es
    if not isinstance(preprocessed_img, tf.Tensor):
        preprocessed_img = tf.convert_to_tensor(preprocessed_img, dtype=tf.float32)

    # Usar GradientTape para calcular gradientes
    with tf.GradientTape() as tape:
        # Obtener activaciones de la última capa conv y predicciones
        conv_outputs, predictions = grad_model(preprocessed_img)

        # Manejar el caso donde predictions puede ser una lista
        if isinstance(predictions, list):
            predictions = predictions[0]

        # Obtener la clase predicha
        pred_index = tf.argmax(predictions[0])

        # Obtener el score de la clase predicha
        class_channel = predictions[:, pred_index]

    # Calcular gradientes de la clase predicha con respecto a las activaciones
    grads = tape.gradient(class_channel, conv_outputs)

    # Promediar los gradientes espacialmente (Global Average Pooling)
    # Shape: (batch, height, width, channels) -> (channels,)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Convertir a numpy para facilitar operaciones
    conv_outputs = conv_outputs[0].numpy()  # Remover dimensión batch
    pooled_grads = pooled_grads.numpy()

    # Ponderar cada canal de las activaciones por su gradiente correspondiente
    # Esto indica qué filtros son más importantes para la clase predicha
    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    # Crear el heatmap promediando todos los canales ponderados
    heatmap = np.mean(conv_outputs, axis=-1)

    # Aplicar ReLU para mantener solo influencias positivas
    heatmap = np.maximum(heatmap, 0)

    # Normalizar el heatmap a rango [0, 1]
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)

    return heatmap


def superimpose_heatmap(heatmap, original_img, target_size=(512, 512), alpha=0.8):
    # Superpone el heatmap Grad-CAM sobre la imagen original.
    #
    # Args:
    #     heatmap (numpy.ndarray): Heatmap normalizado (0-1) generado por Grad-CAM.
    #     original_img (numpy.ndarray): Imagen original (puede ser cualquier tamaño).
    #     target_size (tuple): Tamaño objetivo (ancho, alto) para redimensionar. Default (512, 512).
    #     alpha (float): Factor de transparencia del heatmap (0-1). Default 0.8.
    #                   Valores más altos hacen el heatmap más visible.
    #
    # Returns:
    #     numpy.ndarray: Imagen RGB con heatmap superpuesto.

    # Redimensionar heatmap al tamaño objetivo
    heatmap_resized = cv2.resize(heatmap, target_size)

    # Convertir heatmap a escala 0-255 (uint8)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    # Aplicar colormap (JET: azul=baja activación, rojo=alta activación)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Redimensionar imagen original al tamaño objetivo
    original_resized = cv2.resize(original_img, target_size)

    # Aplicar transparencia al heatmap
    transparency = heatmap_colored * alpha
    transparency = transparency.astype(np.uint8)

    # Superponer heatmap sobre imagen original
    superimposed = cv2.add(transparency, original_resized)
    superimposed = superimposed.astype(np.uint8)

    # Convertir de BGR a RGB para visualización correcta
    superimposed_rgb = superimposed[:, :, ::-1]

    return superimposed_rgb


def grad_cam(array, model=None):
    # Genera visualización Grad-CAM completa para una imagen.
    # Esta función es compatible con el código legacy de detector_neumonia.py.
    # Realiza todo el pipeline: preprocesamiento, carga de modelo (si no se provee),
    # generación de heatmap y superposición.
    #
    # Args:
    #     array (numpy.ndarray): Imagen original como array de NumPy.
    #     model (opcional): Modelo pre-cargado. Si es None, se carga automáticamente.
    #
    # Returns:
    #     numpy.ndarray: Imagen RGB con heatmap Grad-CAM superpuesto.
    #
    # Example:
    #     from src import grad_cam, load_model
    #     img_array = load_image_file("imagen.dcm")[0]
    #     heatmap_img = grad_cam.grad_cam(img_array)

    # 1. Preprocesar imagen
    preprocessed_img = preprocess_img.preprocess_image(array)

    # 2. Cargar modelo si no se provee
    if model is None:
        model = load_model.load_model()

    # 3. Generar heatmap Grad-CAM
    heatmap = generate_heatmap(model, preprocessed_img)

    # 4. Superponer heatmap sobre imagen original
    superimposed_img = superimpose_heatmap(heatmap, array)

    return superimposed_img
