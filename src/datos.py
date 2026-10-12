"""
Escalado de las ventanas y su preparación para el modelo de lenguaje.

El escalado (apartado 4.2.2 de la memoria) se aplica en
01_preparacion_datos: logaritmo con signo, sign(x)·log(1+|x|), seguido de
mínimo-máximo en los 11 contadores, y solo mínimo-máximo en V01. V02 no se
escala. Los escaladores se ajustan solo con el conjunto de entrenamiento.
"""

import numpy as np


def escalar(X, escalador_contadores, escalador_gap, idx_contadores, idx_gap):
    """
    Escala ventanas con valores reales, igual que en 01_preparacion_datos.

    Parámetros:
        X (np.ndarray): ventanas con valores reales, forma (n, 12, 13).
        escalador_contadores: MinMaxScaler ajustado sobre los 11 contadores,
            ya en escala logarítmica.
        escalador_gap: MinMaxScaler ajustado sobre V01.
        idx_contadores (list[int]): posiciones de los contadores.
        idx_gap (int): posición de V01.

    Devuelve:
        np.ndarray: la misma forma que X, con los valores escalados.
    """
    forma_original = X.shape
    X_2d = X.reshape(-1, forma_original[-1]).copy()

    # Logaritmo con signo: comprime los valores altos y mantiene el 0 como 0.
    contadores_log = np.sign(X_2d[:, idx_contadores]) * np.log1p(np.abs(X_2d[:, idx_contadores]))
    X_2d[:, idx_contadores] = escalador_contadores.transform(contadores_log)

    X_2d[:, [idx_gap]] = escalador_gap.transform(X_2d[:, [idx_gap]])

    return X_2d.reshape(forma_original)


def desescalar(X_esc, escalador_contadores, escalador_gap, idx_contadores, idx_gap):
    """
    Deshace el escalado para recuperar los valores reales de las ventanas.
    Es la operación inversa de escalar().

    Parámetros:
        X_esc (np.ndarray): ventanas escaladas, forma (n, 12, 13).
        escalador_contadores, escalador_gap, idx_contadores, idx_gap:
            los mismos que en escalar().

    Devuelve:
        np.ndarray: la misma forma que X_esc, con los valores reales de los
        contadores y de V01.
    """
    forma_original = X_esc.shape
    X_2d = X_esc.reshape(-1, forma_original[-1]).copy()

    # Inversa del mínimo-máximo y, después, del logaritmo con signo.
    contadores_log = escalador_contadores.inverse_transform(X_2d[:, idx_contadores])
    X_2d[:, idx_contadores] = np.sign(contadores_log) * (np.expm1(np.abs(contadores_log)))

    X_2d[:, [idx_gap]] = escalador_gap.inverse_transform(X_2d[:, [idx_gap]])

    return X_2d.reshape(forma_original)


def redondear_ventana(matriz_desescalada):
    """
    Redondea al entero más cercano los valores de una o varias ventanas ya
    desescaladas. Los contadores son enteros, pero el escalado y su inversa
    dejan pequeños errores de coma flotante; se redondean antes de mostrar
    la ventana al modelo de lenguaje.

    Parámetros:
        matriz_desescalada (np.ndarray): forma (12, 13) o (n, 12, 13).

    Devuelve:
        np.ndarray: la misma forma, con valores enteros (int64).
    """
    return np.rint(matriz_desescalada).astype(np.int64)


def formatear_ventana_como_texto(matriz_desescalada, nombres_features):
    """
    Convierte una ventana con valores reales en una tabla de texto alineada,
    lista para insertarla en el prompt del modelo de lenguaje.

    La tabla tiene una fila por señal y una columna por reporte (ev1 ... ev12).
    Los nombres se muestran completos: recortarlos hacía que dos señales
    distintas quedaran con el mismo nombre en la tabla y el modelo las
    confundía. Los valores enteros se escriben sin decimales.

    Parámetros:
        matriz_desescalada (np.ndarray): una ventana, forma (12, n_senales).
        nombres_features (list[str]): nombre de cada columna, en el mismo
            orden que la matriz.

    Devuelve:
        str: la tabla, con una fila de cabecera y una fila por señal.
    """
    def formatear_valor(valor):
        return f"{valor:12.0f}" if float(valor).is_integer() else f"{valor:12.1f}"

    ancho_nombre = max(len(nombre) for nombre in nombres_features) + 2
    n_eventos = matriz_desescalada.shape[0]

    cabecera = "senal".ljust(ancho_nombre) + "".join(f"ev{i+1}".rjust(12) for i in range(n_eventos))
    lineas = [cabecera]
    for indice_columna, nombre in enumerate(nombres_features):
        valores = "".join(formatear_valor(matriz_desescalada[i, indice_columna]) for i in range(n_eventos))
        lineas.append(nombre.ljust(ancho_nombre) + valores)
    return "\n".join(lineas)
