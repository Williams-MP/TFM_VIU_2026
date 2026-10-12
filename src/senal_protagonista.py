"""
Señal protagonista de una ventana (apartado 5.1.3 de la memoria).

Para cada señal se compara su valor máximo en la ventana con su propio
percentil 99, calculado solo con los valores distintos de cero del
conjunto de entrenamiento (con tantos ceros, el percentil sobre todos los
valores sería casi siempre 0). La señal con el mayor cociente es la
protagonista. Si ninguna supera su percentil, la ventana queda sin
protagonista.
"""

import numpy as np


def calcular_percentiles_referencia(X_train_desescalado, percentil=99):
    """
    Calcula, para cada variable, el percentil de referencia de sus valores
    distintos de cero en entrenamiento.

    Parámetros:
        X_train_desescalado (np.ndarray): ventanas de entrenamiento con
            valores reales, forma (n, 12, 13).
        percentil (float): percentil a calcular (0-100).

    Devuelve:
        np.ndarray: un valor por variable, forma (13,). Las variables sin
        ningún valor distinto de cero quedan como NaN.
    """
    X_train_flat = X_train_desescalado.reshape(-1, X_train_desescalado.shape[-1])
    n_columnas = X_train_flat.shape[1]
    percentiles_referencia = np.full(n_columnas, np.nan)

    for i in range(n_columnas):
        valores_columna = X_train_flat[:, i]
        valores_no_cero = np.abs(valores_columna[valores_columna != 0])
        if len(valores_no_cero) > 0:
            percentiles_referencia[i] = np.percentile(valores_no_cero, percentil)

    return percentiles_referencia


def identificar_senal_protagonista(matriz_desescalada, percentiles_referencia, nombres_features):
    """
    Identifica la señal que más supera su percentil de referencia en una
    ventana.

    Parámetros:
        matriz_desescalada (np.ndarray): una ventana con valores reales,
            forma (12, 13).
        percentiles_referencia (np.ndarray): salida de
            calcular_percentiles_referencia(), forma (13,).
        nombres_features (list[str]): nombre de cada variable, en orden.

    Devuelve:
        dict con:
            columna_protagonista (str): nombre de la señal.
            proporcion_exceso (float): su máximo dividido por su percentil.
            protagonista_valido (bool): False si ninguna señal supera su
                percentil (proporción menor que 1).
    """
    # Las variables sin referencia (NaN o 0) no pueden ser protagonistas.
    referencia_segura = np.where(
        np.isnan(percentiles_referencia) | (percentiles_referencia == 0), np.nan, percentiles_referencia
    )
    maximos_por_columna = np.abs(matriz_desescalada).max(axis=0)
    proporciones = maximos_por_columna / referencia_segura
    proporciones = np.nan_to_num(proporciones, nan=-1.0)

    indice_protagonista = int(np.argmax(proporciones))
    proporcion_exceso = float(proporciones[indice_protagonista])

    return {
        "columna_protagonista": nombres_features[indice_protagonista],
        "proporcion_exceso": proporcion_exceso,
        "protagonista_valido": proporcion_exceso >= 1,
    }
