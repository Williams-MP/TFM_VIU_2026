"""
Comprobación de que los prompts guardados en ficheros (llm_lab.prompts) montan exactamente
los mismos mensajes que recibió el modelo en las ejecuciones de la memoria.

Para cada llamada guardada: se lee la ventana real de los datos preparados, se comprueba su
huella y se vuelve a montar el prompt con la variante equivalente. Solo lee.
"""

import json

import pandas as pd

from .huellas import huella_valores


def verificar_ejecucion(prompts, fuente, ruta_json, ruta_csv, version=1):
    """
    Una fila por llamada guardada con: ventana igual (huella) y mensajes iguales.

    prompts: llm_lab.prompts.Prompts.
    fuente: llm_lab.ventanas.FuenteVentanas con los conjuntos de las llamadas.
    ruta_json / ruta_csv: ejecución de 37_ (fase1_* o fase2_*): el JSON tiene los mensajes; el CSV, el
        conjunto y el índice de cada ventana.
    version: versión de las variantes exp2_* que corresponde a la ejecución (v1 = la de 37_).
    """
    with open(ruta_json, encoding="utf-8") as f:
        registro = json.load(f)
    ubicacion = (pd.read_csv(ruta_csv).drop_duplicates("id").set_index("id")[["conjunto", "indice"]]
                 .to_dict("index"))
    filas = []
    for llamada in registro["llamadas"]:
        u = ubicacion[llamada["id"]]
        ventana = fuente.ventana(u["conjunto"], int(u["indice"]), avisar=False)
        ref = f"exp2_{llamada['metodo']}_{llamada['orientacion']}@v{version}"
        filas.append({"id": llamada["id"], "variante": ref,
                      "ventana_igual": huella_valores(ventana) == llamada.get("huella_ventana", huella_valores(ventana)),
                      "mensajes_iguales": prompts.montar_mensajes(ref, ventana) == llamada["mensajes"]})
    return pd.DataFrame(filas)
