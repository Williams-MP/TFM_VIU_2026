"""
Representaciones de la ventana en el prompt: cómo ve el modelo los 12 reportes.

Cada representación recibe la ventana (12, 11) en enteros y los nombres de las
señales tal como los verá el modelo, y devuelve el texto que se inserta en el
prompt. La descripción sustituye a {datos} en las instrucciones que lo usan.
"""

from dataclasses import dataclass
from typing import Callable

from config import NOMBRES_CONTADORES
from datos import formatear_ventana_como_texto


@dataclass(frozen=True)
class Representacion:
    funcion: Callable      # (ventana, nombres) -> texto
    version: str           # se sube al cambiar el código de la función
    descripcion: str       # texto que sustituye a {datos} en una instrucción que lo contenga


def tabla_senales(v, nombres):
    """Una fila por señal y una columna por reporte (ev1 a ev12)."""
    return formatear_ventana_como_texto(v, nombres)


def tabla_reportes(v, nombres):
    """Una fila por reporte (ev1 a ev12) y una columna por señal."""
    anchos = [len(n) + 2 for n in nombres]
    lineas = ["reporte".ljust(9) + "".join(n.rjust(a) for n, a in zip(nombres, anchos))]
    for i in range(v.shape[0]):
        lineas.append(f"ev{i+1}".ljust(9) + "".join(f"{int(x)}".rjust(a) for x, a in zip(v[i], anchos)))
    return "\n".join(lineas)


# Posición de cada contador en la ventana (12, 11), por su nombre corto
POSICION = {nombre: i for i, nombre in enumerate(NOMBRES_CONTADORES)}
_P = POSICION

# Experimento 3: valores calculados aquí (no por el modelo) en cada reporte, y la tolerancia con la que se
# comprueba lo que escribe el modelo (redondea a entero). Cada función recibe la fila del reporte y devuelve
# el valor, o None si no se puede calcular (sin pérdidas RTP no hay porcentaje).
CALCULOS = {
    "sin_recuperar": (lambda f: f[_P["S01ERR"]] - f[_P["S01REC01"]] - f[_P["S01REC02"]], 0),
    "pct_rafaga": (lambda f: 100 * f[_P["S02ERR"]] / f[_P["S01ERR"]] if f[_P["S01ERR"]] else None, 1),
    "pct_recuperado": (lambda f: 100 * (f[_P["S01REC01"]] + f[_P["S01REC02"]]) / f[_P["S01ERR"]]
                       if f[_P["S01ERR"]] else None, 1),
}
FILAS_CALCULADAS = ["sin_recuperar", "pct_rafaga", "pct_recuperado"]


def tabla_senales_calculos(v, nombres):
    """
    La tabla de señales en filas con tres filas más, calculadas aquí: lo que queda sin recuperar, el porcentaje
    de la pérdida en ráfaga y el porcentaje recuperado. Valores redondeados a entero; «-» en los porcentajes de
    los reportes sin pérdidas RTP.
    """
    ancho = max(len(n) for n in nombres) + 2
    lineas = [tabla_senales(v, nombres)]
    for nombre in FILAS_CALCULADAS:
        funcion = CALCULOS[nombre][0]
        celdas = []
        for fila in v.astype(float):
            valor = funcion(fila)
            celdas.append("-".rjust(12) if valor is None else f"{int(round(valor)):12d}")
        lineas.append(nombre.ljust(ancho) + "".join(celdas))
    return "\n".join(lineas)


REPRESENTACIONES = {
    "tabla_senales": Representacion(tabla_senales, "v1",
                                    "una tabla (filas: señales; columnas ev1 a ev12: reportes en orden cronológico)"),
    "tabla_reportes": Representacion(tabla_reportes, "v1",
                                     "una tabla (filas ev1 a ev12: reportes en orden cronológico; columnas: señales)"),
    "tabla_senales_calculos": Representacion(tabla_senales_calculos, "v1",
                                             "una tabla (filas: señales y tres valores calculados; columnas ev1 a "
                                             "ev12: reportes en orden cronológico)"),
}


def mostrar(nombre, ventana, nombres):
    """Texto de la representación `nombre` para una ventana."""
    if nombre not in REPRESENTACIONES:
        raise KeyError(f"Representación desconocida: {nombre}. Disponibles: {list(REPRESENTACIONES)}")
    return REPRESENTACIONES[nombre].funcion(ventana, nombres)
