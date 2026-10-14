"""
Experimento 3: explicación de las alertas con el modelo de lenguaje.

El prompt es una variante guardada en la carpeta privada (llm_lab.prompts), con los nombres cortos de las
señales; al montarlo se ponen los nombres que ve el modelo. La ventana va como tabla de señales en filas con
tres filas más calculadas en Python (representaciones.tabla_senales_calculos): lo que queda sin recuperar, el
porcentaje de la pérdida en ráfaga y el porcentaje recuperado. Así el modelo no tiene que hacer esas
cuentas, que es donde fallaba en las pruebas de desarrollo.

Comprobaciones automáticas de cada respuesta:
- Fidelidad (H3): el formato pide citar los valores como «evN: señal=valor». comprobar_citas() comprueba cada
  cita en el reporte exacto de la ventana (no basta con que el valor aparezca en otro reporte). También se
  comprueban las citas escritas como «evN (señal=valor)» o «señal=valor en evN», que el modelo usa a veces.
- Aritmética: los valores calculados que el modelo copia («evN: sin_recuperar=…») se comparan con los reales.
  Es una medida aparte: no cuenta como error de fidelidad de H3.
- Impacto: el nivel que da el modelo para vídeo y audio se compara con el de la escala declarada (peor reporte de
  S03 y S04: 0 ninguno, 1-10 bajo, 11-360 medio, > 360 alto). También es una medida aparte de H3.

Confidencialidad: las respuestas del modelo y los errores de cita llevan los nombres que ve el modelo, así que
las ejecuciones, las pruebas y el informe HTML se guardan solo en la carpeta privada. Lo que se puede compartir
son las cifras (resumen_ejecucion, evaluar_h3), sin textos.
"""

import glob
import html
import json
import os
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd

from .huellas import huella_texto
from .modelos import entorno
from .representaciones import CALCULOS, POSICION
from .respuestas import leer_explicacion

COLUMNAS_TEXTO = ["errores_cita", "senales_a_cero", "errores_calculo"]   # llevan nombres reales: solo privado


# ---------------------------------------------------------------- citas y cálculos
_PATRON_REPORTE = re.compile(r"\bev\s?(\d{1,2})\s*:", re.I)
_PAR_PEGADO = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(-?\d(?:[\d.,]*\d)?)")
_SEPARADOR = re.compile(r"\s*(?:,|;|\by\b)\s*")


def _a_entero(texto):
    # «9.000», «9,000» o «9000» -> 9000 (los separadores de miles se ignoran)
    limpio = texto.rstrip(".,").replace(".", "").replace(",", "")
    return int(limpio) if limpio.lstrip("-").isdigit() else None


def _a_numero(texto):
    # «2.710» o «2,710» (miles) -> 2710; «38.8» o «38,8» (decimal) -> 38.8
    t = texto.rstrip(".,")
    if re.fullmatch(r"-?\d{1,3}([.,]\d{3})+", t):
        return float(re.sub(r"[.,]", "", t))
    try:
        return float(t.replace(",", "."))
    except ValueError:
        return None


# Además del formato pedido («evN: señal=valor»), el modelo escribe a veces «evN (señal=valor, …)» o
# «señal=valor en evN». Las dos formas atribuyen el valor a un único reporte sin ambigüedad, así que también se
# comprueban. No se atribuyen: un valor para varios reportes («=0 en ev7, ev8») ni un número sin señal («ev4 (35)»).
_PATRON_REPORTE_PARENTESIS = re.compile(r"\bev\s?(\d{1,2})\s*\(", re.I)
_PATRON_PAR_EN_REPORTE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(-?\d(?:[\d.,]*\d)?)\s+en\s+(?:el\s+)?"
                                    r"ev\s?(\d{1,2})\b(?!\s*(?:,|\by\b|\be\b)\s*ev\s?\d)", re.I)


def _pares_con_posicion(texto):
    # (reporte, nombre, valor en texto, posición del par) de las tres formas, sin repetir ningún par
    pares = []
    for patron in (_PATRON_REPORTE, _PATRON_REPORTE_PARENTESIS):
        for m in patron.finditer(texto or ""):
            if patron is _PATRON_REPORTE_PARENTESIS:
                cierre = texto.find(")", m.end())
                if "respectivamente" in texto[m.end():cierre if cierre >= 0 else len(texto)].lower():
                    continue   # «en ev11 y ev12 (a=957 y 618 respectivamente)»: los valores son de varios reportes
            pos = m.end()
            while True:
                par = _PAR_PEGADO.match(texto, pos)
                if not par:
                    break
                pares.append((int(m.group(1)), par.group(1), par.group(2), par.start(1)))
                sep = _SEPARADOR.match(texto, par.end())
                if not sep:
                    break
                pos = sep.end()
    ya = {p[3] for p in pares}
    for m in _PATRON_PAR_EN_REPORTE.finditer(texto or ""):
        if m.start(1) not in ya:
            pares.append((int(m.group(3)), m.group(1), m.group(2), m.start(1)))
    return sorted(pares, key=lambda p: p[3])


def _pares_tras_marcas(texto):
    """
    Pares (reporte, nombre, valor en texto) atribuidos a un único reporte: «evN: a=5, b=7» (el formato pedido),
    «evN (a=5, b=7)» y «a=5 en evN». Un «nombre=valor» que no va pegado a su reporte (p. ej. «con pct_rafaga=99
    (ej: ev1: …)») no se atribuye a ninguno: así no se confunde con el reporte citado antes en otra frase.
    """
    return [p[:3] for p in _pares_con_posicion(texto)]


def citas(texto):
    """Citas «evN: señal=valor» de un texto, como lista de (reporte, señal, valor). Admite «ev3: a=5, b=7»."""
    return [(reporte, nombre, _a_entero(valor)) for reporte, nombre, valor in _pares_tras_marcas(texto)]


def _como_texto(datos):
    return json.dumps(datos, ensure_ascii=False) if isinstance(datos, dict) else str(datos or "")


def comprobar_citas(datos, ventana, nombres):
    """
    Comprueba cada cita «evN: señal=valor» de la respuesta en el reporte exacto de la ventana.
    nombres: los nombres de las señales tal como los ve el modelo.
    Devuelve el número de citas, las correctas y la lista de errores (con el valor real). sin_errores = ninguna
    cita equivocada. Las señales nombradas que valen 0 en toda la ventana se listan aparte, para revisarlas.
    """
    texto = _como_texto(datos)
    posicion = {n: j for j, n in enumerate(nombres)}
    errores, correctas = [], 0
    lista = [c for c in citas(texto) if c[1] not in CALCULOS]   # los cálculos se comprueban aparte
    for reporte, senal, valor in lista:
        if senal not in posicion:
            errores.append(f"ev{reporte}: {senal} no es una señal de la ventana")
        elif not 1 <= reporte <= ventana.shape[0]:
            errores.append(f"ev{reporte} no existe")
        elif valor is None or int(ventana[reporte - 1, posicion[senal]]) != valor:
            errores.append(f"ev{reporte}: {senal}={valor}, pero vale {int(ventana[reporte - 1, posicion[senal]])}")
        else:
            correctas += 1
    nombradas = {n for n in nombres if re.search(rf"\b{re.escape(n)}\b", texto)}
    a_cero = sorted(n for n in nombradas if not ventana[:, posicion[n]].any())
    return {"citas": len(lista), "citas_correctas": correctas, "errores_cita": errores,
            # Nombrar una señal que vale 0 no es por sí solo un error («no hay errores en X»): se informa aparte
            "senales_a_cero": a_cero, "sin_errores": not errores}


def comprobar_calculos(datos, ventana):
    """
    Comprueba cada valor calculado «evN: nombre=valor» (nombres de CALCULOS) contra el real del reporte.
    sin_recuperar debe ser exacto; los porcentajes admiten ±1 punto (más el redondeo).
    Devuelve el número de cálculos, los correctos, los errores y el detalle por tipo.
    """
    errores, correctos, por_tipo = [], 0, {n: [0, 0] for n in CALCULOS}   # [correctos, total]
    for reporte, nombre, crudo in _pares_tras_marcas(_como_texto(datos)):
        if nombre not in CALCULOS:
            continue
        funcion, tolerancia = CALCULOS[nombre]
        por_tipo[nombre][1] += 1
        valor = _a_numero(crudo)
        real = funcion(ventana[reporte - 1].astype(float)) if 1 <= reporte <= ventana.shape[0] else None
        if real is None or valor is None:
            errores.append(f"ev{reporte}: {nombre}={crudo} no se puede calcular en ese reporte")
        elif abs(valor - real) <= tolerancia + 0.5 * (tolerancia > 0):
            correctos += 1
            por_tipo[nombre][0] += 1
        else:
            errores.append(f"ev{reporte}: {nombre}={crudo}, pero vale {real:.0f}")
    return {"calculos": sum(t for _, t in por_tipo.values()), "calculos_correctos": correctos,
            "errores_calculo": errores,
            **{f"{n}_correctos": c for n, (c, _) in por_tipo.items()},
            **{f"{n}_total": t for n, (_, t) in por_tipo.items()}}


# ---------------------------------------------------------------- campos de la respuesta
def tipo_evolucion(datos):
    """Tipo de evolución al principio de «evolucion» («sostenido. El problema…» -> "sostenido"); None si no está."""
    texto = str((datos or {}).get("evolucion", "")).strip().lower()
    m = re.match(r"\W*(aislado|sostenido|cr[oó]nico|ninguno)", texto)
    return m.group(1).replace("ó", "o") if m else None


_NIVELES = ["ninguno", "bajo", "medio", "alto"]


def partes_impacto(datos):
    """
    Impacto en vídeo y en audio («vídeo: medio; audio: bajo» -> ("medio", "bajo", "medio")): nivel de cada uno y
    el mayor de los dos. Si la respuesta da un solo nivel sin separar, se toma como el mayor. None si no se reconoce.
    """
    texto = str((datos or {}).get("impacto", "")).strip().lower().replace("í", "i")
    nivel = lambda clave: (re.search(rf"{clave}\W*(ninguno|bajo|medio|alto)", texto) or [None, None])[1]
    video, audio = nivel("video"), nivel("audio")
    niveles = [n for n in (video, audio) if n]
    if niveles:
        mayor = max(niveles, key=_NIVELES.index)
    else:
        unico = re.match(r"\W*(ninguno|bajo|medio|alto)", texto)
        mayor = unico.group(1) if unico else None
    return video, audio, mayor


# Escala de impacto declarada en el prompt (supuesto del trabajo), por canal y con su peor reporte:
# 0 = ninguno; 1-10 = bajo (p90 de los valores distintos de 0 en entrenamiento); 11-360 = medio; > 360 = alto.
ESCALA_IMPACTO = ((0, "ninguno"), (10, "bajo"), (360, "medio"))


def nivel_impacto(valor):
    """Nivel de la escala de impacto para el peor valor de un canal."""
    for limite, nivel in ESCALA_IMPACTO:
        if valor <= limite:
            return nivel
    return "alto"


def comprobar_impacto(datos, ventana):
    """
    Nivel de impacto que corresponde a la ventana según la escala (vídeo: S03; audio: S04, su peor reporte) y si
    coincide con el que da el modelo. Es una medida aparte: no cuenta como error de fidelidad de H3.
    """
    video, audio, _ = partes_impacto(datos)
    esperado_video = nivel_impacto(int(ventana[:, POSICION["S03"]].max()))
    esperado_audio = nivel_impacto(int(ventana[:, POSICION["S04"]].max()))
    return {"impacto_video_esperado": esperado_video, "impacto_video_coherente": video == esperado_video,
            "impacto_audio_esperado": esperado_audio, "impacto_audio_coherente": audio == esperado_audio}


def analizar(respuesta_bruta, ventana, nombres):
    """Respuesta interpretada: JSON (con estado), citas, cálculos, tipo de evolución e impacto."""
    datos, estado = leer_explicacion(respuesta_bruta)
    fidelidad = comprobar_citas(datos if datos else respuesta_bruta, ventana, nombres)
    calculos = comprobar_calculos(datos if datos else respuesta_bruta, ventana)
    video, audio, mayor = partes_impacto(datos)
    return {"estado_json": estado, "datos": datos, **fidelidad, **calculos, "evolucion_tipo": tipo_evolucion(datos),
            "impacto_video": video, "impacto_audio": audio, "impacto_nivel": mayor, **comprobar_impacto(datos, ventana)}


def fila_resultado(w, a, cortada=None, tokens_salida=None):
    """Fila de la tabla de resultados de una ventana: cifras y campos cortos (los textos van aparte)."""
    return {"id": w["id"], "referencia": w.get("referencia"), "grupo": w.get("grupo"),
            "estado_json": a["estado_json"], "cortada": cortada, "tokens_salida": tokens_salida,
            "citas": a["citas"], "citas_correctas": a["citas_correctas"], "sin_errores": a["sin_errores"],
            "calculos": a["calculos"], "calculos_correctos": a["calculos_correctos"],
            **{f"{n}_{s}": a[f"{n}_{s}"] for n in CALCULOS for s in ("correctos", "total")},
            "evolucion_tipo": a["evolucion_tipo"], "impacto_video": a["impacto_video"],
            "impacto_audio": a["impacto_audio"], "impacto_nivel": a["impacto_nivel"],
            **{k: a[k] for k in ("impacto_video_esperado", "impacto_video_coherente",
                                 "impacto_audio_esperado", "impacto_audio_coherente")},
            "origen": str(a["datos"].get("origen", "")) if a["datos"] else None,
            "errores_cita": " | ".join(a["errores_cita"]), "senales_a_cero": ", ".join(a["senales_a_cero"]),
            "errores_calculo": " | ".join(a["errores_calculo"])}


# ---------------------------------------------------------------- tipos de error de cita
TIPOS_ERROR = ["reporte vecino", "otra señal, mismo reporte", "otra señal, reporte vecino",
               "valor de otra celda", "no está en la ventana", "señal o reporte inexistente"]


def clasificar_errores(respuesta_bruta, ventana, nombres):
    """
    Cada cita equivocada con su probable origen, comparando el valor citado con el resto de la ventana:
    - «reporte vecino»: es el valor de esa misma señal en el reporte anterior o el siguiente (se equivoca de columna);
    - «otra señal, mismo reporte»: es el valor de otra señal en ese reporte (se equivoca de fila);
    - «otra señal, reporte vecino»: otra señal en el reporte anterior o el siguiente;
    - «valor de otra celda»: está en la ventana, pero lejos;  «no está en la ventana»: no aparece en ninguna celda.
    Es una aproximación: con valores pequeños (0, 1, 2…) varias celdas pueden coincidir por azar; se toma la
    primera explicación de la lista. Devuelve una lista de dicts (reporte, señal, citado, real, tipo).
    """
    datos, _ = leer_explicacion(respuesta_bruta)
    texto = json.dumps(datos, ensure_ascii=False) if datos else str(respuesta_bruta or "")
    posicion = {n: j for j, n in enumerate(nombres)}
    filas = []
    for reporte, senal, valor in citas(texto):
        if senal in CALCULOS:
            continue
        if senal not in posicion or not 1 <= reporte <= ventana.shape[0] or valor is None:
            filas.append({"reporte": reporte, "senal": senal, "citado": valor, "real": None,
                          "tipo": "señal o reporte inexistente"})
            continue
        i, j = reporte - 1, posicion[senal]
        real = int(ventana[i, j])
        if real == valor:
            continue
        vecinos = [k for k in (i - 1, i + 1) if 0 <= k < ventana.shape[0]]
        otras = [c for c in range(ventana.shape[1]) if c != j]
        if any(int(ventana[k, j]) == valor for k in vecinos):
            tipo = "reporte vecino"
        elif any(int(ventana[i, c]) == valor for c in otras):
            tipo = "otra señal, mismo reporte"
        elif any(int(ventana[k, c]) == valor for k in vecinos for c in otras):
            tipo = "otra señal, reporte vecino"
        elif (ventana == valor).any():
            tipo = "valor de otra celda"
        else:
            tipo = "no está en la ventana"
        filas.append({"reporte": reporte, "senal": senal, "citado": valor, "real": real, "tipo": tipo})
    return filas


# ---------------------------------------------------------------- una ventana
def probar(prompts, ref, modelo, ventana, w, registro):
    """
    Ejecuta la variante en una ventana y añade la prueba (prompt, mensajes, respuesta y comprobaciones) al fichero
    .jsonl `registro` (privado). Solo muestra cifras: la respuesta se lee en el registro o en el informe HTML.
    Devuelve el análisis de la respuesta.
    """
    r_var = prompts.resolver(ref)
    mensajes = prompts.montar_mensajes(ref, ventana)
    r = modelo.consultar(mensajes, max_new_tokens=r_var["max_new_tokens"])
    a = analizar(r["respuesta_bruta"], ventana, prompts.nombres_llm)
    cortada = r["tokens_salida"] >= r_var["max_new_tokens"]
    print(f"{w['id']} · {r['tokens_entrada']} tokens de entrada · {r['tokens_salida']} de salida"
          f"{' (CORTADA)' if cortada else ''} · {r['segundos']:.0f} s · JSON: {a['estado_json']}")
    print(f"Citas correctas: {a['citas_correctas']} de {a['citas']} · cálculos correctos: "
          f"{a['calculos_correctos']} de {a['calculos']} · señales nombradas a 0: {len(a['senales_a_cero'])}")
    print(f"Evolución: {a['evolucion_tipo']} · impacto vídeo: {a['impacto_video']} · audio: {a['impacto_audio']}")
    os.makedirs(os.path.dirname(registro), exist_ok=True)
    texto = prompts.texto_prompt(ref)
    fila = {"fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "id": w["id"], "modelo": modelo.clave,
            "variante": r_var["id"], "huella_prompt": huella_texto(texto), "prompt": texto, "mensajes": mensajes,
            "max_new_tokens": r_var["max_new_tokens"], "respuesta_bruta": r["respuesta_bruta"],
            "tokens_entrada": r["tokens_entrada"], "tokens_salida": r["tokens_salida"], "cortada": cortada,
            "segundos": r["segundos"], **{k: a[k] for k in a if k != "datos"}}
    with open(registro, "a", encoding="utf-8") as f:
        f.write(json.dumps(fila, ensure_ascii=False) + "\n")
    print(f"Respuesta completa (privada) en {os.path.basename(registro)}")
    return a


# ---------------------------------------------------------------- todas las ventanas
def ejecutar(prompts, ref, modelo, fuente, ventanas, carpeta, prefijo="exp3", confirmo_prueba=False, reanudar=None):
    """
    Aplica la variante a todas las ventanas y guarda, tras cada una, <prefijo>_<modelo>_<fecha>.json (prompt
    exacto, mensajes y respuesta de cada ventana) en la carpeta (privada). El formato es el de las ejecuciones del
    repositorio anterior, para poder copiarlas (copiar_ejecucion).
    reanudar: nombre base de una ejecución interrumpida, para completar las ventanas que falten.
    Devuelve la ruta del JSON.
    """
    os.makedirs(carpeta, exist_ok=True)
    r_var = prompts.resolver(ref)
    texto = prompts.texto_prompt(ref)
    if any(w["conjunto"] == "test" for w in ventanas) and not confirmo_prueba:
        raise PermissionError("Hay ventanas de prueba: úsalas solo con el prompt cerrado (confirmo_prueba=True)")
    if reanudar:
        ruta = f"{carpeta}/{reanudar}.json"
        registro = json.load(open(ruta, encoding="utf-8"))
        assert registro["huella_prompt"] == huella_texto(texto), "la ejecución que se reanuda usaba otro prompt"
        registro.setdefault("sesiones", []).append({"inicio": datetime.now().strftime("%Y-%m-%d %H:%M"),
                                                    "entorno": entorno()})
        print(f"Reanudando {reanudar}: {len(registro['llamadas'])} ventanas ya hechas")
    else:
        ruta = f"{carpeta}/{prefijo}_{modelo.clave}_{datetime.now():%Y%m%d_%H%M%S}.json"
        registro = {"fecha": datetime.now().strftime("%Y-%m-%d %H:%M"), "modelo": modelo.parametros,
                    "variante": r_var["id"], "huella_variante": prompts.huella_variante(ref),
                    "prompt": texto, "huella_prompt": huella_texto(texto), "max_new_tokens": r_var["max_new_tokens"],
                    "sesiones": [{"inicio": datetime.now().strftime("%Y-%m-%d %H:%M"), "entorno": entorno()}],
                    "llamadas": []}
    hechas = {c["id"] for c in registro["llamadas"]}
    pendientes = [w for w in ventanas if w["id"] not in hechas]
    print(f"Prompt: huella {registro['huella_prompt']} · tokens máx.: {r_var['max_new_tokens']} · "
          f"ventanas pendientes: {len(pendientes)} de {len(ventanas)}")
    inicio = time.time()
    for k, w in enumerate(pendientes, start=1):
        v = fuente.ventana(w["conjunto"], w["indice"], avisar=False)
        mensajes = prompts.montar_mensajes(ref, v)
        r = modelo.consultar(mensajes, max_new_tokens=r_var["max_new_tokens"])
        a = analizar(r["respuesta_bruta"], v, prompts.nombres_llm)
        registro["llamadas"].append({"id": w["id"], "huella_ventana": w.get("huella_ventana"), "mensajes": mensajes,
                                     "texto_formateado": r["texto_formateado"], "respuesta_bruta": r["respuesta_bruta"],
                                     "tokens_salida": r["tokens_salida"], "segundos": r["segundos"]})
        with open(ruta, "w", encoding="utf-8") as f:   # tras cada ventana: un corte no pierde lo hecho
            json.dump(registro, f, ensure_ascii=False, indent=1)
        restantes = (time.time() - inicio) / k * (len(pendientes) - k) / 60
        print(f"{k}/{len(pendientes)} {w['id']}: JSON {a['estado_json']}"
              f"{' · CORTADA' if r['tokens_salida'] >= r_var['max_new_tokens'] else ''} · citas "
              f"{a['citas_correctas']}/{a['citas']} · cálculos {a['calculos_correctos']}/{a['calculos']} · "
              f"{r['segundos']:.0f} s · quedan ~{restantes:.0f} min")
    print("Guardado:", os.path.basename(ruta))
    return ruta


def copiar_ejecucion(ruta_origen, carpeta, prompts, ref, fuente, ventanas):
    """
    Copia a `carpeta` una ejecución hecha con el repositorio anterior, después de comprobar que es la misma
    variante: misma huella del prompt y, ventana a ventana, exactamente los mismos mensajes que se montan ahora con
    los datos preparados. Las ventanas que no estén en `ventanas` se rechazan. Devuelve la ruta de la copia.
    """
    registro = json.load(open(ruta_origen, encoding="utf-8"))
    huella = huella_texto(prompts.texto_prompt(ref))
    assert registro["huella_prompt"] == huella, (f"{os.path.basename(ruta_origen)}: huella {registro['huella_prompt']}, "
                                                 f"pero la variante {ref} da {huella}")
    por_id = {w["id"]: w for w in ventanas}
    for c in registro["llamadas"]:
        assert c["id"] in por_id, f"{c['id']} no está en las ventanas indicadas"
        w = por_id[c["id"]]
        v = fuente.ventana(w["conjunto"], w["indice"], avisar=False)
        assert c["mensajes"] == prompts.montar_mensajes(ref, v), f"{c['id']}: los mensajes no coinciden"
    # El JSON antiguo no guardaba los tokens de salida (para saber si se cortó la respuesta): están en su CSV
    ruta_csv = ruta_origen[:-5] + ".csv"
    if os.path.exists(ruta_csv):
        tokens = pd.read_csv(ruta_csv).set_index("id")["tokens_salida"].to_dict()
        for c in registro["llamadas"]:
            c.setdefault("tokens_salida", int(tokens[c["id"]]) if c["id"] in tokens else None)
    os.makedirs(carpeta, exist_ok=True)
    destino = f"{carpeta}/{os.path.basename(ruta_origen)}"
    registro["copiada_de"] = ruta_origen
    registro.setdefault("variante", prompts.resolver(ref)["id"])
    registro.setdefault("huella_variante", prompts.huella_variante(ref))
    with open(destino, "w", encoding="utf-8") as f:
        json.dump(registro, f, ensure_ascii=False, indent=1)
    print(f"Copiada: {os.path.basename(destino)} · {len(registro['llamadas'])} ventanas · huella {huella} · "
          f"mensajes idénticos")
    return destino


# ---------------------------------------------------------------- evaluación
def ultima_ejecucion(carpeta, prefijo, modelo):
    """Ruta del JSON más reciente «<prefijo>_<modelo>_*.json» de la carpeta."""
    rutas = sorted(glob.glob(f"{carpeta}/{prefijo}_{modelo}_*.json"))
    if not rutas:
        raise FileNotFoundError(f"No hay ejecuciones {prefijo}_{modelo}_* en {carpeta}")
    return rutas[-1]


def cargar_ejecucion(ruta, prompts, fuente, ventanas):
    """
    Registro de una ejecución y su tabla de resultados, recalculada con las comprobaciones de este módulo a partir
    de las respuestas guardadas (vale igual para una ejecución nueva que para una copiada).
    Devuelve (registro, tabla). La tabla lleva columnas de texto con nombres reales (COLUMNAS_TEXTO): no la
    compartas tal cual.
    """
    registro = json.load(open(ruta, encoding="utf-8"))
    por_id = {w["id"]: w for w in ventanas}
    maximo = registro["max_new_tokens"]
    filas = []
    for c in registro["llamadas"]:
        w = por_id[c["id"]]
        v = fuente.ventana(w["conjunto"], w["indice"], avisar=False)
        a = analizar(c["respuesta_bruta"], v, prompts.nombres_llm)
        tokens = c.get("tokens_salida")
        fila = fila_resultado(w, a, cortada=None if tokens is None else tokens >= maximo, tokens_salida=tokens)
        tipos = [t["tipo"] for t in clasificar_errores(c["respuesta_bruta"], v, prompts.nombres_llm)]
        filas.append({**fila, **{f"error: {t}": tipos.count(t) for t in TIPOS_ERROR}})
    return registro, pd.DataFrame(filas)


def resumen_ejecucion(tabla):
    """
    Cifras automáticas de una ejecución: JSON válidos, cortadas, citas (fidelidad de H3) y cálculos (por tipo), con
    el intervalo de Clopper-Pearson de cada proporción. Sin textos: se puede compartir.
    """
    from .exp2_estadisticas import intervalo_clopper_pearson as ic
    n = len(tabla)
    filas = [("Respuestas con JSON válido", int((tabla["estado_json"] != "no_valido").sum()), n)]
    if tabla["cortada"].notna().all():   # las ejecuciones copiadas no guardaban los tokens de salida
        filas.append(("Respuestas cortadas por el límite de tokens", int(tabla["cortada"].astype(bool).sum()), n))
    filas += [("Respuestas sin errores al citar la tabla", int(tabla["sin_errores"].sum()), n),
              ("Respuestas sin ninguna cita comprobable", int((tabla["citas"] == 0).sum()), n),
              ("Citas correctas", int(tabla["citas_correctas"].sum()), int(tabla["citas"].sum())),
              ("Valores calculados copiados correctamente", int(tabla["calculos_correctos"].sum()),
               int(tabla["calculos"].sum()))]
    for nombre in CALCULOS:
        filas.append((f"  · {nombre}", int(tabla[f"{nombre}_correctos"].sum()), int(tabla[f"{nombre}_total"].sum())))
    for canal in ("video", "audio"):   # medida aparte de H3: nivel de impacto coherente con la escala declarada
        if f"impacto_{canal}_coherente" in tabla:
            filas.append((f"Nivel de impacto en {'vídeo' if canal == 'video' else 'audio'} coherente con la escala",
                          int(tabla[f"impacto_{canal}_coherente"].sum()), n))
    salida = []
    for medida, k, total in filas:
        inf, sup = ic(k, total) if total else (np.nan, np.nan)
        salida.append({"medida": medida, "k": k, "n": total, "proporcion": round(k / total, 3) if total else np.nan,
                       "IC 95 %": f"[{inf:.2f}–{sup:.2f}]" if total else ""})
    return pd.DataFrame(salida)


def plantilla_puntuaciones(ids):
    """Texto de la celda de puntuación, listo para copiar: una línea por ventana, con los campos vacíos."""
    lineas = ["PUNTUACIONES = {"]
    for id_ in ids:
        lineas.append(f'    "{id_}": dict(utilidad=None, legibilidad=None, error_fidelidad=None),')
    lineas.append("}")
    return "\n".join(lineas)


def evaluar_h3(puntuaciones, tabla, media_minima=4.0, proporcion_minima=0.8):
    """
    H3 con la rúbrica (Y. Liu, Tao, Meng, Wang et al., 2024): media >= 4 en utilidad y en legibilidad, y al menos
    el 80 % de las explicaciones sin errores de fidelidad. Una explicación no tiene errores de fidelidad si la
    comprobación automática no encuentra citas equivocadas, el JSON se pudo leer y el autor no anotó ningún error
    (error_fidelidad = 0). Devuelve (tabla por ventana, resumen); ninguna de las dos lleva textos.
    """
    from scipy import stats
    from .exp2_estadisticas import intervalo_clopper_pearson as ic
    p = pd.DataFrame.from_dict(puntuaciones, orient="index").rename_axis("id").reset_index()
    p = p[[c for c in ("id", "utilidad", "legibilidad", "error_fidelidad") if c in p.columns]]
    faltan = p[p[["utilidad", "legibilidad", "error_fidelidad"]].isna().any(axis=1)]["id"].tolist()
    if faltan:
        print(f"AVISO: faltan puntuaciones en {len(faltan)} ventanas: {', '.join(faltan)} (quedan fuera del resumen)")
    p = p.dropna(subset=["utilidad", "legibilidad", "error_fidelidad"])
    sin_puntuar = sorted(set(tabla["id"]) - set(p["id"]))
    if sin_puntuar:
        print(f"AVISO: {len(sin_puntuar)} ventanas de la ejecución sin puntuar: {', '.join(sin_puntuar)} (quedan fuera)")
    t = tabla[["id", "referencia", "grupo", "estado_json", "sin_errores", "citas", "calculos",
               "calculos_correctos"]].merge(p, on="id", how="inner", validate="one_to_one")
    t["fiel"] = t["sin_errores"].astype(bool) & (t["estado_json"] != "no_valido") & (t["error_fidelidad"].astype(int) == 0)
    n = len(t)
    filas = []
    for medida in ("utilidad", "legibilidad"):
        x = t[medida].astype(float)
        if n > 1 and x.std() > 0:
            inf, sup = stats.t.interval(0.95, n - 1, loc=x.mean(), scale=stats.sem(x))
        else:   # todas las puntuaciones iguales (o una sola): el intervalo se reduce a la media
            inf = sup = x.mean() if n else np.nan
        filas.append({"criterio": f"Media de {medida} >= {media_minima:g}", "valor": round(x.mean(), 2),
                      "IC 95 %": f"[{inf:.2f}–{sup:.2f}]", "n": n, "cumple": bool(x.mean() >= media_minima)})
    k = int(t["fiel"].sum())
    inf, sup = ic(k, n) if n else (np.nan, np.nan)
    filas.append({"criterio": f"Explicaciones sin errores de fidelidad >= {proporcion_minima:.0%}",
                  "valor": f"{k} de {n} ({k / n:.0%})" if n else "", "IC 95 %": f"[{inf:.2f}–{sup:.2f}]", "n": n,
                  "cumple": bool(n and k / n >= proporcion_minima)})
    resumen = pd.DataFrame(filas)
    resumen.loc[len(resumen)] = {"criterio": "H3 (se cumplen los tres criterios)", "valor": "", "IC 95 %": "", "n": n,
                                 "cumple": bool(resumen["cumple"].all())}
    return t, resumen


def grafica_puntuaciones(t):
    """Número de explicaciones con cada puntuación (1 a 5) en utilidad y en legibilidad, con la media."""
    import plotly.graph_objects as go
    from .exp2_estadisticas import _estilo, TINTA_SUAVE, FONDO
    fig = go.Figure()
    for medida, color in (("utilidad", "#2a78d6"), ("legibilidad", "#1baf7a")):
        cuentas = t[medida].astype(int).value_counts().reindex(range(1, 6), fill_value=0)
        fig.add_bar(x=[str(i) for i in range(1, 6)], y=cuentas.values,
                    name=f"{medida.capitalize()} (media {t[medida].mean():.2f})",
                    marker_color=color, marker_line=dict(color=FONDO, width=1.5), text=cuentas.values,
                    textposition="outside", textfont=dict(color=TINTA_SUAVE, size=11))
    _estilo(fig, f"Puntuaciones de las {len(t)} explicaciones según la rúbrica (1 = peor, 5 = mejor)", 460)
    fig.update_layout(barmode="group", bargap=0.25)
    fig.update_xaxes(title_text="Puntuación")
    fig.update_yaxes(title_text="Explicaciones", rangemode="tozero")
    return fig


# ---------------------------------------------------------------- informe privado
def informe_html(registro, tabla, prompts, fuente, ventanas, ruta, titulo):
    """
    HTML autónomo (privado) para revisar y puntuar: un desplegable y flechas para elegir la ventana y, para cada una,
    la gráfica y la tabla con los datos que vio el modelo, las comprobaciones (citas y cálculos, con el tipo de cada
    error, y el nivel de impacto frente a la escala) y cada campo de la respuesta. Funciona sin conexión.
    """
    from visualizacion import graficar_ventana_y_tabla
    e = lambda x: "" if x is None or (isinstance(x, float) and np.isnan(x)) else html.escape(str(x))
    por_id = {w["id"]: w for w in ventanas}
    filas = tabla.set_index("id")
    ids = [c["id"] for c in registro["llamadas"]]
    nombres = prompts.nombres_llm
    opciones = "".join(f"<option value='{e(i)}'>{e(i)} · {e(por_id[i].get('grupo'))} · {e(por_id[i].get('referencia'))}"
                       f"{'' if filas.loc[i, 'sin_errores'] else ' · con errores de cita'}</option>" for i in ids)
    partes = ["<!doctype html><html lang='es'><head><meta charset='utf-8'><title>" + e(titulo) + "</title><style>"
              "body{font-family:system-ui,sans-serif;margin:16px;color:#1f2328;background:#fff}"
              "header{position:sticky;top:0;background:#fff;padding:8px 0;border-bottom:2px solid #d0d7de;z-index:10}"
              "select{font-size:15px;padding:4px;max-width:100%}section{display:none;padding:12px 0}"
              "table{border-collapse:collapse;font-size:13px;width:100%}"
              "td,th{border:1px solid #d0d7de;padding:4px 6px;vertical-align:top;text-align:left}th{background:#E6F1FB;width:16%}"
              ".valores th{width:auto}.valores td,.valores th{padding:2px 5px}"
              ".ok{background:#E3F4E6}.ko{background:#FCE4E4}pre{white-space:pre-wrap;margin:0;font-family:inherit}"
              "</style></head><body>",
              f"<header><b>{e(titulo)}</b> · prompt {e(registro.get('huella_prompt'))} · "
              f"<button onclick='mover(-1)'>&larr;</button> <select id='sel' onchange='ver(this.value)'>{opciones}</select> "
              f"<button onclick='mover(1)'>&rarr;</button></header>"]
    primera = True
    for c in registro["llamadas"]:
        w, f = por_id[c["id"]], filas.loc[c["id"]]
        v = fuente.ventana(w["conjunto"], w["indice"], avisar=False)
        partes.append(f"<section id='s_{e(c['id'])}'><h2>{e(c['id'])} · grupo: {e(w.get('grupo'))} · "
                      f"referencia: {e(w.get('referencia'))}</h2>")
        fig = graficar_ventana_y_tabla(v, nombres, c["id"], mostrar=False)
        partes.append(fig.to_html(full_html=False, include_plotlyjs=primera))   # plotly.js solo una vez, incrustado
        primera = False
        # Los valores, también en una tabla HTML: en la de la figura los números grandes pueden quedar cortados
        partes.append("<table class='valores'><tr><th>señal</th>" + "".join(f"<th>ev{k + 1}</th>" for k in range(v.shape[0]))
                      + "</tr>" + "".join(f"<tr><td>{e(n)}</td>" + "".join(
                          f"<td style='text-align:right{';background:#FFF3D6' if x else ''}'>{int(x)}</td>" for x in v[:, j])
                          + "</tr>" for j, n in enumerate(nombres)) + "</table><br>")
        tipos = clasificar_errores(c["respuesta_bruta"], v, nombres)
        detalle = "<br>".join(f"ev{t['reporte']}: {e(t['senal'])}={e(t['citado'])} (vale {e(t['real'])}) → {e(t['tipo'])}"
                              for t in tipos)
        impacto = (f"vídeo: {e(f['impacto_video'])} (escala: {e(f['impacto_video_esperado'])}); "
                   f"audio: {e(f['impacto_audio'])} (escala: {e(f['impacto_audio_esperado'])})")
        coherente = bool(f["impacto_video_coherente"]) and bool(f["impacto_audio_coherente"])
        partes.append("<table>"
                      f"<tr class='{'ok' if f['sin_errores'] else 'ko'}'><th>Citas correctas</th>"
                      f"<td>{f['citas_correctas']} de {f['citas']}<br>{detalle}</td></tr>"
                      f"<tr class='{'ok' if f['calculos_correctos'] == f['calculos'] else 'ko'}'><th>Valores calculados"
                      f"</th><td>{f['calculos_correctos']} de {f['calculos']}<br>{e(f['errores_calculo'])}</td></tr>"
                      f"<tr class='{'ok' if coherente else 'ko'}'><th>Impacto frente a la escala</th><td>{impacto}</td></tr>")
        datos, estado = leer_explicacion(c["respuesta_bruta"])
        if datos:
            for clave, valor in datos.items():
                valor = "\n".join(map(str, valor)) if isinstance(valor, list) else valor
                partes.append(f"<tr><th>{e(clave)}</th><td><pre>{e(valor)}</pre></td></tr>")
        else:
            partes.append(f"<tr class='ko'><th>Respuesta (JSON {e(estado)})</th><td><pre>{e(c['respuesta_bruta'])}</pre></td></tr>")
        partes.append("</table></section>")
    partes.append("<script>const sel=document.getElementById('sel');"
                  "function ver(id){document.querySelectorAll('section').forEach(s=>s.style.display='none');"
                  "const s=document.getElementById('s_'+id);s.style.display='block';sel.value=id;"
                  "window.dispatchEvent(new Event('resize'));}"
                  "function mover(d){const i=sel.selectedIndex+d;if(i>=0&&i<sel.options.length)ver(sel.options[i].value);}"
                  "ver(sel.value);</script></body></html>")
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as f:
        f.write("".join(partes))
    return ruta
