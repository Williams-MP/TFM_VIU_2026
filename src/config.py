"""
Configuración común a todos los notebooks de Colab.

Define dónde están los datos, el nombre y el orden de las 13 variables de
cada ventana y las dimensiones del modelo. Por el acuerdo de
confidencialidad, las variables se identifican con nombres genéricos
(S01ERR ... V02) y no se describe qué mide cada una.
"""

import json
import os

# ============================================================
# Rutas
# ============================================================

# Carpeta con los ficheros Parquet de ventanas (subcarpetas train/, val/
# y test/; formato descrito en el README). Se puede cambiar sin tocar el
# código definiendo la variable de entorno TFM_RUTA_DATOS antes de importar
# este módulo.
RUTA_DRIVE = os.environ.get("TFM_RUTA_DATOS", "/content/drive/MyDrive/TFM_VIU_2026/datos")

# Carpeta donde 01_preparacion_datos guarda las ventanas escaladas y los
# escaladores, y de la que leen el resto de notebooks.
RUTA_PREPARADOS = f"{RUTA_DRIVE}/preparados"

# Carpeta raíz de los resultados. Cada notebook escribe en su propia
# subcarpeta y solo lee las de los notebooks anteriores.
RUTA_RESULTADOS = f"{RUTA_DRIVE}/resultados"

# Resultados de 02_entrenamiento_detectores: checkpoints, errores por señal
# de cada modelo y registro de entrenamientos.
RUTA_DETECTORES = f"{RUTA_RESULTADOS}/detectores"

# Resultados de 03_analisis_detectores: concordancia de alertas, alertas de
# consenso, umbral POT y alertas comunes (entrada de los Experimentos 2 y 3).
RUTA_ANALISIS_DETECTORES = f"{RUTA_RESULTADOS}/analisis_detectores"

# Repetición de los 10 entrenamientos (sección 7 de 02_entrenamiento_detectores),
# que 04_experimento_1 compara con el entrenamiento original.
RUTA_DETECTORES_REPRODUCIBILIDAD = f"{RUTA_DETECTORES}_reproducibilidad"

# Figuras, tablas y registro de resultados de 04_experimento_1.
RUTA_EXPERIMENTO_1 = f"{RUTA_RESULTADOS}/experimento_1"

# ============================================================
# Variables de cada ventana
# ============================================================

# Los 11 contadores, en el orden en que aparecen en cada reporte de la
# ventana (posiciones 0 a 10).
NOMBRES_CONTADORES = [
    "S01ERR", "S01REC01", "S01REC02", "S03", "S04", "S04L01",
    "S03H01", "S03L01", "S02ERR", "S02REC01", "S02REC02",
]

# Las 13 variables: los 11 contadores más las dos derivadas.
# V01: minutos desde el reporte anterior del mismo dispositivo.
# V02: 1 si S01ERR cambia en más de 10^6 respecto al reporte anterior.
NOMBRES_FEATURES_COMPLETO = NOMBRES_CONTADORES + ["V01", "V02"]

# Posición de cada grupo de variables dentro del vector de 13.
IDX_CONTADORES = list(range(0, 11))
IDX_GAP = 11    # V01
IDX_SALTO = 12  # V02

# Familia de cada variable, con nombres genéricos (apartado 5.1.3 de la
# memoria). Un error y sus recuperaciones están en la misma familia.
FAMILIA_DE_VARIABLE = {
    "S01ERR": "FAM_1", "S01REC01": "FAM_1", "S01REC02": "FAM_1",
    "S02ERR": "FAM_1", "S02REC01": "FAM_1", "S02REC02": "FAM_1",
    "S03": "FAM_2", "S03H01": "FAM_2", "S03L01": "FAM_2",
    "S04": "FAM_3", "S04L01": "FAM_3",
    "V01": "FAM_4", "V02": "FAM_4",
}
ORDEN_FAMILIAS = ["FAM_1", "FAM_2", "FAM_3", "FAM_4"]

# Dimensiones de la entrada del modelo: 12 reportes por ventana, 13 variables.
N_FEATS = 13
N_WINDOW = 12

# ============================================================
# Nombres que recibe el modelo de lenguaje
# ============================================================


def cargar_nombres_para_llm(ruta_fichero=None):
    """
    Devuelve los nombres de los 11 contadores que se muestran al modelo de
    lenguaje en los Experimentos 2 y 3.

    En los experimentos de la memoria, el modelo recibió los nombres
    descriptivos de los contadores. Esos nombres están protegidos por el
    acuerdo de confidencialidad y no se incluyen en el repositorio: se leen
    de un fichero JSON privado, fuera del control de versiones, con la forma
    {"S01ERR": "nombre descriptivo", ...}.

    Si el fichero no existe, se devuelven los nombres genéricos. El código
    funciona igual, pero los prompts no son idénticos a los de la memoria y
    las respuestas del modelo pueden cambiar.

    Parámetros:
        ruta_fichero (str o None): ruta del JSON privado. Por defecto,
            la variable de entorno TFM_NOMBRES_LLM o, si no está definida,
            {RUTA_DRIVE}/privado/nombres_senales.json.

    Devuelve:
        list[str]: un nombre por contador, en el orden de NOMBRES_CONTADORES.
    """
    if ruta_fichero is None:
        ruta_fichero = os.environ.get("TFM_NOMBRES_LLM", f"{RUTA_DRIVE}/privado/nombres_senales.json")

    if not os.path.exists(ruta_fichero):
        print("Aviso: no se encuentra el fichero privado de nombres; "
              "el modelo de lenguaje recibirá los nombres genéricos.")
        return list(NOMBRES_CONTADORES)

    with open(ruta_fichero, encoding="utf-8") as f:
        correspondencia = json.load(f)

    # Se comprueba que el fichero cubre todos los contadores, para no mezclar
    # en un mismo prompt nombres genéricos y descriptivos sin darse cuenta.
    faltan = [nombre for nombre in NOMBRES_CONTADORES if nombre not in correspondencia]
    if faltan:
        raise ValueError(f"El fichero de nombres no incluye: {faltan}")

    return [correspondencia[nombre] for nombre in NOMBRES_CONTADORES]
