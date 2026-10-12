"""
Experimento 2: entradas (ventanas y referencia del autor) y, más adelante, el análisis.

Las entradas se guardan en la carpeta privada con solo lo necesario: identificador, conjunto,
índice, huella de la ventana, referencia del autor y grupo. Las notas del autor no se copian.
"""

import json
import os

import numpy as np
import pandas as pd

from .huellas import huella_valores

VARIANTES_PRUEBA = ["exp2_N0_senales_en_filas", "exp2_N0_reportes_en_filas",
                    "exp2_N1_senales_en_filas", "exp2_N1_reportes_en_filas"]   # las 4 que se usaron en prueba


def copiar_entradas(fuente, origen_seleccion_dev, origen_etiquetas_dev, origen_referencia_prueba, destino):
    """
    Crea ventanas_desarrollo.csv y ventanas_prueba.csv en `destino` a partir de los ficheros de trabajo,
    comprobando que cada ventana coincide con la de los datos preparados. Solo lee los orígenes.
    """
    os.makedirs(destino, exist_ok=True)
    seleccion = json.load(open(origen_seleccion_dev, encoding="utf-8"))
    etiquetas = pd.read_csv(origen_etiquetas_dev).set_index("id")["etiqueta"].to_dict()
    dev = []
    for w in seleccion["ventanas"]:
        v = fuente.ventana("val", w["indice"], avisar=False)
        id_ = f"val_{w['indice']}"
        assert np.array_equal(v, np.array(w["valores"])), f"{id_}: no coincide con los datos preparados"
        dev.append({"id": id_, "conjunto": "val", "indice": int(w["indice"]), "huella_ventana": huella_valores(v),
                    "referencia": etiquetas.get(id_, "sin etiqueta"), "grupo": "desarrollo"})
    prueba = []
    for r in pd.read_csv(origen_referencia_prueba).itertuples():
        v = fuente.ventana("test", int(r.indice_test), avisar=False)
        assert huella_valores(v) == r.huella_ventana, f"{r.id}: no coincide con los datos preparados"
        prueba.append({"id": r.id, "conjunto": "test", "indice": int(r.indice_test), "huella_ventana": r.huella_ventana,
                       "referencia": r.veredicto, "grupo": r.grupo})
    pd.DataFrame(dev).to_csv(f"{destino}/ventanas_desarrollo.csv", index=False)
    pd.DataFrame(prueba).to_csv(f"{destino}/ventanas_prueba.csv", index=False)
    return len(dev), len(prueba)


def cargar_ventanas(ruta_csv, fuente=None):
    """Ventanas de un conjunto como lista de dicts. Con `fuente`, comprueba de nuevo cada huella."""
    ventanas = pd.read_csv(ruta_csv).to_dict("records")
    for w in ventanas:
        w["indice"] = int(w["indice"])
        if fuente is not None:
            assert huella_valores(fuente.ventana(w["conjunto"], w["indice"], avisar=False)) == w["huella_ventana"], w["id"]
    return ventanas
