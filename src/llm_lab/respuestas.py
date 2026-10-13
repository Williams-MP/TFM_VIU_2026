"""
Lectura de las respuestas del modelo.

- Detección (Experimento 2): JSON con reasoning, affected_signals, anomalous y anomaly_score.
- Explicación (Experimento 3): JSON con cadena, evolucion, origen, reportes_origen, impacto y explicacion.
"""

import json
import re
import unicodedata


def _limpiar(texto):
    # Quita los bloques ```json ... ``` que a veces añade el modelo
    return (texto or "").strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()


def _sin_tildes(texto):
    return "".join(c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn")


def leer_json(texto):
    """Respuesta como diccionario; None si no es un JSON válido."""
    try:
        return json.loads(_limpiar(texto))
    except json.JSONDecodeError:
        return None


def campos_deteccion(respuesta_bruta):
    """Campos de una respuesta de detección para la tabla de resultados."""
    datos = leer_json(respuesta_bruta) or {}
    senales = datos.get("affected_signals") or []
    senales = senales if isinstance(senales, list) else [str(senales)]
    return {"json_valido": bool(datos), "veredicto": datos.get("anomalous"), "score_verbal": datos.get("anomaly_score"),
            "senales": ";".join(map(str, senales)), "razonamiento": datos.get("reasoning")}


def leer_explicacion(texto):
    """
    Respuesta de explicación como (datos, estado):
      "valido"    JSON correcto tal cual;
      "reparado"  solo fallaban las comillas de reportes_origen (p. ej. [ev1, ev2]) y se han añadido;
      "no_valido" no se puede interpretar.
    Las claves se normalizan sin tildes y en minúsculas ("explicación" -> "explicacion").
    """
    limpio, estado = _limpiar(texto), "valido"
    try:
        datos = json.loads(limpio)
    except json.JSONDecodeError:
        # Única reparación: poner comillas a los elementos de reportes_origen
        reparado = re.sub(r'("reportes_origen"\s*:\s*\[)([^\]]*)(\])',
                          lambda m: m.group(1) + ", ".join(json.dumps(x.strip().strip("\"'"))
                                                           for x in m.group(2).split(",") if x.strip()) + m.group(3),
                          limpio)
        try:
            datos, estado = json.loads(reparado), "reparado"
        except json.JSONDecodeError:
            return {}, "no_valido"
    if not isinstance(datos, dict):
        return {}, "no_valido"
    return {_sin_tildes(str(k)).lower().strip(): v for k, v in datos.items()}, estado
