"""
Ejecución de variantes sobre un conjunto de ventanas, con registro completo.

Cada ejecución guarda dos ficheros con el mismo nombre base en su carpeta (privada):
- <base>.csv: una fila por llamada (ventana × variante), con la respuesta interpretada.
- <base>.json: trazabilidad completa: definición de cada variante con sus textos, modelo,
  parámetros y entorno y, por cada llamada, los mensajes exactos, el texto que recibió el
  modelo y la respuesta en bruto.

Las dos llevan los nombres reales y las respuestas del modelo: solo en la carpeta privada.

Las ventanas del conjunto de prueba solo se usan con la variante ya cerrada: hay que
confirmarlo expresamente y cada uso queda anotado en registro_uso_prueba.json.
"""

import json
import os
import time
from datetime import datetime

import pandas as pd

from .modelos import entorno
from .respuestas import campos_deteccion


def _guardar(carpeta, base, filas, registro):
    pd.DataFrame(filas).to_csv(f"{carpeta}/{base}.csv", index=False)
    with open(f"{carpeta}/{base}.json", "w", encoding="utf-8") as f:
        json.dump(registro, f, ensure_ascii=False, indent=1, default=str)


def anotar_uso_prueba(carpeta, variantes, huellas, modelo):
    """Deja constancia de cada uso de las ventanas de prueba."""
    ruta = f"{carpeta}/registro_uso_prueba.json"
    uso = json.load(open(ruta, encoding="utf-8")) if os.path.exists(ruta) else []
    uso.append({"fecha": datetime.now().strftime("%Y-%m-%d %H:%M"), "variantes": variantes,
                "huellas": huellas, "modelo": modelo})
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(uso, f, ensure_ascii=False, indent=1)
    print(f"Uso nº {len(uso)} de las ventanas de prueba registrado")


def ejecutar(prompts, modelo, fuente, ventanas, variantes, carpeta, prefijo,
             confirmo_prueba=False, reanudar=None):
    """
    Ejecuta cada ventana con cada variante y guarda los resultados tras cada ventana.

    Parámetros:
        prompts: llm_lab.prompts.Prompts.
        modelo: ModeloLLM con los pesos cargados.
        fuente: FuenteVentanas.
        ventanas: lista de dicts con al menos id, conjunto e indice. Las claves extra
            (referencia, grupo...) se copian a la tabla de resultados.
        variantes: referencias de variante ("nombre@v2"...).
        carpeta: carpeta (privada) de resultados.
        prefijo: inicio del nombre base (p. ej. "exp2_test").
        confirmo_prueba: obligatorio si hay ventanas del conjunto de prueba.
        reanudar: nombre base de una ejecución interrumpida, para completar lo que falte.

    Devuelve el nombre base de la ejecución: <prefijo>_<modelo>_<fecha>.
    """
    os.makedirs(carpeta, exist_ok=True)
    definiciones = {}
    for ref in variantes:
        r = prompts.resolver(ref)
        if r["tipo_respuesta"] != "deteccion":
            raise NotImplementedError("De momento solo se ejecutan variantes de detección (Experimento 2)")
        definiciones[r["id"]] = {"huella": prompts.huella_variante(ref), **{k: v for k, v in r.items() if k != "id"}}
    ids_variantes = list(definiciones)
    huellas = {v: d["huella"] for v, d in definiciones.items()}

    if any(v["conjunto"] == "test" for v in ventanas):
        if not confirmo_prueba:
            raise PermissionError("Hay ventanas de prueba: úsalas solo con la variante cerrada (confirmo_prueba=True)")
        anotar_uso_prueba(carpeta, ids_variantes, huellas, modelo.clave)

    if reanudar:
        base = reanudar
        filas = pd.read_csv(f"{carpeta}/{base}.csv").to_dict("records")
        with open(f"{carpeta}/{base}.json", encoding="utf-8") as f:
            registro = json.load(f)
        registro["sesiones"].append({"inicio": datetime.now().strftime("%Y-%m-%d %H:%M"), "entorno": entorno()})
        print(f"Reanudando {base}: {len(filas)} llamadas ya hechas")
    else:
        base = f"{prefijo}_{modelo.clave}_{datetime.now():%Y%m%d_%H%M%S}"
        filas = []
        registro = {"base": base, "variantes": ids_variantes, "definiciones": definiciones,
                    "modelo": modelo.parametros, "nombres_senales": list(prompts.nombres_llm),
                    "sesiones": [{"inicio": datetime.now().strftime("%Y-%m-%d %H:%M"), "entorno": entorno()}],
                    "llamadas": []}
    hechas = {(f["id"], f["variante"]) for f in filas}
    total, inicio = len(ventanas) * len(ids_variantes), time.time()

    for k, w in enumerate(ventanas, start=1):
        ventana = fuente.ventana(w["conjunto"], w["indice"])
        for ref, id_variante in zip(variantes, ids_variantes):   # todas las variantes de una ventana seguidas
            if (w["id"], id_variante) in hechas:
                continue
            definicion = definiciones[id_variante]
            mensajes = prompts.montar_mensajes(ref, ventana)
            r = modelo.consultar(mensajes, max_new_tokens=definicion["max_new_tokens"])
            filas.append({**w, "variante": id_variante, "huella_variante": definicion["huella"], "modelo": modelo.clave,
                          **campos_deteccion(r["respuesta_bruta"]), "p_interna": r["p_interna"],
                          "tokens_entrada": r["tokens_entrada"], "tokens_salida": r["tokens_salida"],
                          "cortada": r["tokens_salida"] >= definicion["max_new_tokens"], "segundos": r["segundos"],
                          "coincide_tokens": r["coincide_tokens"], "respuesta_bruta": r["respuesta_bruta"],
                          "sesion": len(registro["sesiones"])})
            registro["llamadas"].append({"id": w["id"], "variante": id_variante, "huella_ventana": w.get("huella_ventana"),
                                         "mensajes": mensajes, "texto_formateado": r["texto_formateado"],
                                         "respuesta_bruta": r["respuesta_bruta"], "p_interna": r["p_interna"]})
        _guardar(carpeta, base, filas, registro)  # tras cada ventana: un corte de Colab no pierde lo hecho
        restantes = (time.time() - inicio) / k * (len(ventanas) - k) / 60
        print(f"{k}/{len(ventanas)} ventanas ({len(filas)}/{total} llamadas) | quedan ~{restantes:.0f} min")

    registro["fin"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    _guardar(carpeta, base, filas, registro)
    if not all(f["coincide_tokens"] for f in filas):
        print("AVISO: en alguna llamada el texto formateado no reproduce exactamente los tokens de entrada")
    print(f"Guardado: {base}.csv y .json")
    return base
