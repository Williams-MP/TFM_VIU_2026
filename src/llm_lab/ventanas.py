"""
Ventanas en valores reales enteros, tal como las recibe el modelo de lenguaje.

Se deshace el escalado de la preparación de datos y se redondea al entero más
cercano. Solo se usan los 11 contadores: V01 y V02 no se muestran al modelo.
"""

import joblib
import numpy as np

from datos import desescalar, redondear_ventana
from .huellas import huella_valores


class FuenteVentanas:
    """
    Acceso a las ventanas de los conjuntos preparados.

    Parámetros:
        ruta_preparados (str): carpeta con ventanas_escaladas.npz y los dos escaladores.
        idx_contadores (list[int]): posición de los 11 contadores en cada reporte.
        idx_gap (int): posición de V01.
        conjuntos (tuple[str]): conjuntos que se cargan en memoria.
    """

    def __init__(self, ruta_preparados, idx_contadores, idx_gap, conjuntos=("val", "test")):
        self.idx_contadores = list(idx_contadores)
        self.idx_gap = idx_gap
        self.escalador_contadores = joblib.load(f"{ruta_preparados}/escalador_contadores.pkl")
        self.escalador_gap = joblib.load(f"{ruta_preparados}/escalador_gap.pkl")
        with np.load(f"{ruta_preparados}/ventanas_escaladas.npz") as datos:
            self.escaladas = {c: datos[c] for c in conjuntos}

    def ventana(self, conjunto, indice, avisar=True):
        """Ventana (12, 11) en enteros. Avisa si el redondeo oculta decimales (no debería ocurrir)."""
        real = desescalar(self.escaladas[conjunto][indice:indice + 1], self.escalador_contadores,
                          self.escalador_gap, self.idx_contadores, self.idx_gap)[0][:, self.idx_contadores]
        residuo = np.abs(real - np.rint(real)).max()
        if avisar and residuo > 1e-3:
            print(f"AVISO: residuo de redondeo alto en {conjunto}[{indice}]: {residuo:.3f}")
        return redondear_ventana(real)

    def huella(self, conjunto, indice):
        """Huella de los valores de una ventana (para comprobar que es la misma que se anotó)."""
        return huella_valores(self.ventana(conjunto, indice, avisar=False))
