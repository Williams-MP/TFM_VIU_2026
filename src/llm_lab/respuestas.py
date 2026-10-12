"""
Lectura de las respuestas del modelo.

Detección (Experimento 2): JSON con reasoning, affected_signals, anomalous y anomaly_score.
"""

import json


def _limpiar(texto):
    # Quita los bloques ```json ... ``` que a veces añade el modelo
    return (texto or "").strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()


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
