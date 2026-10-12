"""
Huellas cortas (SHA-256, 12 caracteres) para identificar textos y ventanas.

Son las mismas funciones que usaban los notebooks 35_ a 39_, para que las
huellas nuevas y las guardadas en ejecuciones anteriores sean comparables.
"""

import hashlib

import numpy as np


def huella_texto(*textos):
    """Huella de uno o varios textos (separados por un carácter que no aparece en ellos)."""
    return hashlib.sha256("\x1f".join(textos).encode("utf-8")).hexdigest()[:12]


def huella_valores(ventana):
    """Huella de los valores enteros de una ventana."""
    return hashlib.sha256(np.ascontiguousarray(ventana, dtype=np.int64).tobytes()).hexdigest()[:12]
