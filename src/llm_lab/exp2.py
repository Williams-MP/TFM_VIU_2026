"""
Experimento 2: entradas (ventanas y referencia del autor) y análisis de las ejecuciones.

Las entradas se guardan en la carpeta privada con solo lo necesario: identificador, conjunto,
índice, huella de la ventana, referencia del autor y grupo. Las notas del autor no se copian.
"""

import json
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

from .huellas import huella_valores

VARIANTES_PRUEBA = ["exp2_N0_senales_en_filas", "exp2_N0_reportes_en_filas",
                    "exp2_N1_senales_en_filas", "exp2_N1_reportes_en_filas"]   # las 4 que se usaron en prueba
MAPEO_REFERENCIA = {"anomalia": 1, "anomalia_mitigada": 0, "mitigada": 0, "normal": 0}   # dudosa: fuera
PATRON_VARIANTE = re.compile(r"exp2_(N0\+conocimiento|N0|N1)_(senales_en_filas|reportes_en_filas)(@v\d+)?$")


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


# ---------------------------------------------------------------- resultados
def cargar(ruta_csv, modelo, ventanas):
    """
    Resultados de una ejecución en formato común: id, metodo, orientacion, veredicto, p_interna,
    json_valido, cortada, referencia, ref_binaria, grupo y modelo. Vale para el CSV de 37_ y para el de ejecutar().
    Solo se quedan las variantes N0 y N1 (las de la prueba).
    """
    df = pd.read_csv(ruta_csv)
    if "variante" in df.columns:   # ejecución nueva (llm_lab)
        partes = df["variante"].str.extract(PATRON_VARIANTE)
        df["metodo"], df["orientacion"] = partes[0], partes[1]
    if "cortada" not in df.columns:   # 37_ no guardaba esta columna (ninguna respuesta llegó al límite)
        df["cortada"] = False
    df = df[df["metodo"].isin(["N0", "N1"])].copy()
    info = pd.DataFrame(ventanas)[["id", "referencia", "grupo"]]
    df = df.drop(columns=[c for c in ("referencia", "grupo") if c in df.columns]).merge(info, on="id", how="left",
                                                                                       validate="many_to_one")
    assert df["referencia"].notna().all(), "hay resultados de ventanas que no están en el conjunto"
    df["veredicto"] = pd.to_numeric(df["veredicto"], errors="coerce")
    df["ref_binaria"] = df["referencia"].map(MAPEO_REFERENCIA)
    df["modelo"] = modelo
    df["acierta"] = np.where(df["ref_binaria"].notna() & df["veredicto"].isin([0, 1]),
                             df["veredicto"] == df["ref_binaria"], np.nan)
    for c in ("senales", "razonamiento", "respuesta_bruta"):   # textos del modelo (solo para revisar; no se comparten)
        if c not in df.columns:
            df[c] = None
    return df[["modelo", "id", "metodo", "orientacion", "veredicto", "p_interna", "json_valido", "cortada",
               "referencia", "ref_binaria", "grupo", "acierta", "senales", "razonamiento", "respuesta_bruta"]]


def metricas(df):
    """Una fila por modelo y variante, con el criterio de la memoria (sin las dudosas)."""
    filas = []
    for (modelo, metodo, orientacion), g in df.groupby(["modelo", "metodo", "orientacion"], sort=False):
        b = g[g["ref_binaria"].notna()]
        con_veredicto = b[b["veredicto"].isin([0, 1])]
        y, pred = con_veredicto["ref_binaria"].astype(int), con_veredicto["veredicto"].astype(int)
        p = con_veredicto["p_interna"]
        con_p = p.notna()
        filas.append({
            "modelo": modelo, "metodo": metodo, "orientacion": orientacion, "ventanas": len(g),
            "json_valido": round(g["json_valido"].astype(bool).mean(), 3), "cortadas": int(g["cortada"].astype(bool).sum()),
            "evaluables": len(b), "sin_veredicto": len(b) - len(con_veredicto),
            # Las ventanas sin veredicto cuentan como fallo: el modelo no respondió
            "aciertos": int((y == pred).sum()), "acierto": round((y == pred).sum() / max(len(b), 1), 3),
            "sensibilidad": round(((pred == 1) & (y == 1)).sum() / max((b["ref_binaria"] == 1).sum(), 1), 3),
            "especificidad": round(((pred == 0) & (y == 0)).sum() / max((b["ref_binaria"] == 0).sum(), 1), 3),
            "auroc_p_interna": (round(roc_auc_score(y[con_p], p[con_p]), 3) if y[con_p].nunique() == 2 else np.nan),
            "auroc_n": int(con_p.sum()),
        })
    return pd.DataFrame(filas)


def mcnemar_exacto(acierta_a, acierta_b):
    """Prueba exacta de McNemar con las ventanas en que uno acierta y el otro no."""
    a, b = np.asarray(acierta_a, dtype=bool), np.asarray(acierta_b, dtype=bool)
    solo_a, solo_b = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(solo_a, solo_a + solo_b, 0.5).pvalue if solo_a + solo_b else 1.0
    return {"solo_a": solo_a, "solo_b": solo_b, "p_mcnemar": round(p, 4)}


def _pareadas(df, clave_a, clave_b, columnas):
    # Ventanas evaluables en las dos condiciones; sin veredicto = fallo
    a = df[(df[columnas] == pd.Series(clave_a, index=columnas)).all(axis=1)].set_index("id")
    b = df[(df[columnas] == pd.Series(clave_b, index=columnas)).all(axis=1)].set_index("id")
    ids = a.index[a["ref_binaria"].notna()].intersection(b.index)
    return a.loc[ids, "acierta"].fillna(False), b.loc[ids, "acierta"].fillna(False)


def n0_frente_a_n1(df):
    """N0 frente a N1 para cada modelo y orientación (como en la memoria)."""
    filas = []
    for (modelo, orientacion), _ in df.groupby(["modelo", "orientacion"], sort=False):
        a, b = _pareadas(df, (modelo, "N0", orientacion), (modelo, "N1", orientacion), ["modelo", "metodo", "orientacion"])
        filas.append({"modelo": modelo, "orientacion": orientacion, "ventanas": len(a), "aciertos_N0": int(a.sum()),
                      "aciertos_N1": int(b.sum()), **{k.replace("_a", "_N0").replace("_b", "_N1"): v
                                                       for k, v in mcnemar_exacto(a, b).items()}})
    return pd.DataFrame(filas)


def modelos_pareados(df, modelo_a, modelo_b):
    """Para cada variante, el modelo A frente al B sobre las mismas ventanas."""
    filas = []
    for (metodo, orientacion), _ in df.groupby(["metodo", "orientacion"], sort=False):
        a, b = _pareadas(df, (modelo_a, metodo, orientacion), (modelo_b, metodo, orientacion),
                         ["modelo", "metodo", "orientacion"])
        r = mcnemar_exacto(a, b)
        filas.append({"metodo": metodo, "orientacion": orientacion, "ventanas": len(a),
                      f"aciertos_{modelo_a}": int(a.sum()), f"aciertos_{modelo_b}": int(b.sum()),
                      f"solo_{modelo_a}": r["solo_a"], f"solo_{modelo_b}": r["solo_b"], "p_mcnemar": r["p_mcnemar"]})
    return pd.DataFrame(filas)


def dudosas(df):
    """Aciertos contando las dudosas primero como anomalía y después como normales (Tabla 13 de la memoria)."""
    filas = []
    for (modelo, metodo, orientacion), g in df.groupby(["modelo", "metodo", "orientacion"], sort=False):
        fila = {"modelo": modelo, "metodo": metodo, "orientacion": orientacion}
        for nombre, valor in (("sin_dudosas", None), ("dudosas_anomalia", 1), ("dudosas_normal", 0)):
            ref = g["referencia"].map({**MAPEO_REFERENCIA, **({"dudosa": valor} if valor is not None else {})})
            b = ref.notna()
            fila[nombre] = f"{int((g.loc[b, 'veredicto'] == ref[b]).sum())}/{int(b.sum())}"
        filas.append(fila)
    return pd.DataFrame(filas)


# ---------------------------------------------------------------- todas las ejecuciones
PATRON_EJECUCION = re.compile(r"^exp2_(dev|test)_(.+)_(\d{8}_\d{6})\.csv$")


def ultimas_ejecuciones(carpeta, conjunto):
    """{modelo: nombre base} con la ejecución más reciente de cada modelo en ese conjunto."""
    import os
    ultimas = {}
    for nombre in sorted(os.listdir(carpeta)):
        m = PATRON_EJECUCION.match(nombre)
        if m and m.group(1) == conjunto:
            ultimas[m.group(2)] = nombre[:-4]   # sorted: la última fecha gana
    return ultimas


def todos_pareados(df, modelos):
    """Cada par de modelos, en cada variante, sobre las mismas ventanas (McNemar exacto)."""
    from itertools import combinations
    filas = []
    for a, b in combinations(modelos, 2):
        t = modelos_pareados(df, a, b)
        filas += [{"modelo_a": a, "modelo_b": b, "metodo": r["metodo"], "orientacion": r["orientacion"],
                   "ventanas": r["ventanas"], "aciertos_a": r[f"aciertos_{a}"], "aciertos_b": r[f"aciertos_{b}"],
                   "solo_a": r[f"solo_{a}"], "solo_b": r[f"solo_{b}"], "p_mcnemar": r["p_mcnemar"]}
                  for r in t.to_dict("records")]
    return pd.DataFrame(filas)


# ---------------------------------------------------------------- revisión de respuestas
def detalle_ventana(df, id_ventana):
    """Respuestas de todos los modelos y variantes para una ventana, en tabla."""
    g = df[df["id"] == id_ventana].sort_values(["metodo", "orientacion", "modelo"])
    return g[["modelo", "metodo", "orientacion", "veredicto", "p_interna", "senales", "razonamiento"]]


def informe_html(df, ventanas_valores, nombres, ruta, titulo):
    """
    HTML autónomo para revisar las respuestas: por ventana, la gráfica y la tabla (interactivas)
    y, debajo, el veredicto, la probabilidad, las señales y el razonamiento de cada modelo y variante.
    Las filas que aciertan van en verde y las que fallan en rojo. Funciona sin conexión.

    ventanas_valores: {id: matriz (12, 11) con los valores que recibió el modelo}.
    """
    import html
    from visualizacion import graficar_ventana_y_tabla
    e = lambda x: "" if x is None or (isinstance(x, float) and np.isnan(x)) else html.escape(str(x))
    partes = ["<!doctype html><html lang='es'><head><meta charset='utf-8'><title>" + e(titulo) + "</title><style>"
              "body{font-family:system-ui,sans-serif;margin:16px;color:#1f2328;background:#fff}"
              "section{border-top:2px solid #d0d7de;padding:12px 0}table{border-collapse:collapse;font-size:13px;width:100%}"
              "td,th{border:1px solid #d0d7de;padding:4px 6px;vertical-align:top;text-align:left}th{background:#E6F1FB}"
              ".ok{background:#E3F4E6}.ko{background:#FCE4E4}nav a{margin-right:8px}</style></head><body>",
              f"<h1>{e(titulo)}</h1>"]
    ids = list(dict.fromkeys(df["id"]))
    resumen = df.groupby("id")["acierta"].mean()
    partes.append("<nav>" + " ".join(f"<a href='#{e(i)}'>{e(i)} ({resumen.get(i, np.nan):.0%})</a>"
                                      if pd.notna(resumen.get(i)) else f"<a href='#{e(i)}'>{e(i)}</a>" for i in ids) + "</nav>")
    primera = True
    for id_ in ids:
        g = df[df["id"] == id_].sort_values(["metodo", "orientacion", "modelo"])
        info = g.iloc[0]
        partes.append(f"<section id='{e(id_)}'><h2>{e(id_)} · referencia: {e(info['referencia'])} · grupo: {e(info['grupo'])}</h2>")
        fig = graficar_ventana_y_tabla(ventanas_valores[id_], nombres, id_, mostrar=False)
        partes.append(fig.to_html(full_html=False, include_plotlyjs=primera))   # plotly.js solo una vez, incrustado
        primera = False
        partes.append("<table><tr><th>Modelo</th><th>Variante</th><th>Veredicto</th><th>P interna</th>"
                      "<th>Señales</th><th>Razonamiento</th></tr>")
        for r in g.itertuples():
            clase = "" if pd.isna(r.acierta) else ("ok" if r.acierta else "ko")
            veredicto = "sin veredicto" if pd.isna(r.veredicto) else int(r.veredicto)
            p = "" if pd.isna(r.p_interna) else f"{r.p_interna:.3f}"
            partes.append(f"<tr class='{clase}'><td>{e(r.modelo)}</td><td>{e(r.metodo)} | {e(r.orientacion)}</td>"
                          f"<td>{veredicto}</td><td>{p}</td><td>{e(r.senales)}</td><td>{e(r.razonamiento)}</td></tr>")
        partes.append("</table></section>")
    partes.append("</body></html>")
    with open(ruta, "w", encoding="utf-8") as f:
        f.write("".join(partes))
    return ruta


def metricas_por_grupo(df):
    """Acierto por modelo, variante y grupo de ventanas (con alerta, alta actividad sin alerta, al azar)."""
    b = df[df["ref_binaria"].notna()]
    t = (b.groupby(["modelo", "metodo", "orientacion", "grupo"])["acierta"]
         .agg(aciertos=lambda s: int(s.fillna(False).sum()), ventanas="size").reset_index())
    t["acierto"] = (t["aciertos"] / t["ventanas"]).round(3)
    return t


def comparacion_principal(df, modelo_a, modelo_b, metodo, orientacion):
    """La comparación fijada antes de la prueba: modelo B frente a A en una variante, McNemar exacto."""
    t = modelos_pareados(df[(df["metodo"] == metodo) & (df["orientacion"] == orientacion)], modelo_a, modelo_b)
    return t.assign(modelo_a=modelo_a, modelo_b=modelo_b)
