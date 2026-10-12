"""
Representaciones de la ventana en el prompt: cómo ve el modelo los 12 reportes.

Cada representación recibe la ventana (12, 11) en enteros y los nombres de las
señales tal como los verá el modelo, y devuelve el texto que se inserta en el
prompt. La descripción sustituye a {datos} en las instrucciones que lo usan.
"""

from dataclasses import dataclass
from typing import Callable

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


REPRESENTACIONES = {
    "tabla_senales": Representacion(tabla_senales, "v1",
                                    "una tabla (filas: señales; columnas ev1 a ev12: reportes en orden cronológico)"),
    "tabla_reportes": Representacion(tabla_reportes, "v1",
                                     "una tabla (filas ev1 a ev12: reportes en orden cronológico; columnas: señales)"),
}


def mostrar(nombre, ventana, nombres):
    """Texto de la representación `nombre` para una ventana."""
    if nombre not in REPRESENTACIONES:
        raise KeyError(f"Representación desconocida: {nombre}. Disponibles: {list(REPRESENTACIONES)}")
    return REPRESENTACIONES[nombre].funcion(ventana, nombres)
