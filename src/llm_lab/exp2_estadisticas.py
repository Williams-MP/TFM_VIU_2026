"""
Experimento 2: aciertos de los tres modelos en desarrollo y en prueba, por tipo de ventana según
la etiqueta del autor: anomalía, anomalía mitigada y funcionamiento normal.

Lee los resúmenes `comparacion_<conjunto>_<fecha>.json` que guarda el notebook 06 en
resultados/experimento_2/<conjunto>/: una fila por llamada con el veredicto, la probabilidad interna,
la referencia del autor y el grupo. No contienen textos del modelo ni nombres de señales, así que
todo lo que sale de aquí se puede compartir.

Criterio (el de la memoria): en una ventana etiquetada como anomalía, acierta el modelo que dice
«anomalía»; en una mitigada o normal, el que dice «normal». Las dudosas quedan fuera. Una ventana
sin veredicto cuenta como fallo.

Medidas:
- aciertos por tipo de ventana (anomalías, mitigadas, normales) y en total;
- acierto equilibrado = (acierto en anomalías + acierto en ventanas sin anomalía) / 2: no depende
  de cuántas anomalías haya, y responder siempre «anomalía» da 0,5 en cualquier conjunto;
- comparaciones ventana a ventana con la prueba exacta de McNemar (llm_lab.exp2.mcnemar_exacto).
"""

import glob
import json
import os
from itertools import combinations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import beta

from .exp2 import MAPEO_REFERENCIA, mcnemar_exacto

MODELOS = ["qwen2.5-7b", "qwen2.5-14b", "qwen3.5-9b"]
NOMBRE_MODELO = {"qwen2.5-7b": "Qwen2.5-7B", "qwen2.5-14b": "Qwen2.5-14B", "qwen3.5-9b": "Qwen3.5-9B"}
VARIANTES = [("N0", "senales_en_filas"), ("N0", "reportes_en_filas"),
             ("N1", "senales_en_filas"), ("N1", "reportes_en_filas")]
NOMBRE_CONJUNTO = {"dev": "Desarrollo", "test": "Prueba", "juntos": "Desarrollo + prueba"}

# De dónde sale cada conjunto y cómo se eligió (va al pie de las gráficas)
ORIGEN_CONJUNTO = {
    "dev": ("<b>Desarrollo</b> (29 ventanas): partición de validación. Elegidas a mano por el autor, ordenando las "
            "ventanas por los valores de cada contador; sin información de TranAD. Se usaron para ajustar los prompts. "
            "Etiquetas del autor."),
    "test": ("<b>Prueba</b> (40 ventanas): partición de prueba. Sorteadas con semilla 2026: 20 alertas de "
             "consenso de TranAD y TranAD+, 10 sin alerta de alta actividad y 10 sin alerta al azar. "
             "Clasificadas a ciegas por el autor, sin conocer el grupo."),
}

# Colores por modelo: los tres primeros de la paleta categórica de referencia (validada para
# distinguirse entre sí también con daltonismo). El color sigue al modelo en todas las gráficas.
COLOR_MODELO = {"qwen2.5-7b": "#2a78d6", "qwen2.5-14b": "#eb6834", "qwen3.5-9b": "#1baf7a"}
TINTA, TINTA_SUAVE, REJILLA, FONDO = "#0b0b0b", "#52514e", "#e8e7e3", "#fcfcfb"
# Botón de la cámara de las gráficas: PNG a 3 veces la resolución de pantalla (para la memoria)
CONFIG_PNG = {"toImageButtonOptions": {"format": "png", "scale": 3}, "displaylogo": False}

# Tipo de ventana según la etiqueta del autor. «Sin anomalía» = mitigada + normal.
TIPO_VENTANA = {"anomalia": "anomalia", "normal": "normal", "anomalia_mitigada": "mitigada", "dudosa": "dudosa"}
TIPOS = {"anomalia": "anomalía", "mitigada": "anomalía mitigada", "normal": "funcionamiento normal"}
RESPUESTA_CORRECTA = {"anomalia": "«anomalía»", "mitigada": "«normal»", "normal": "«normal»"}
ETIQUETA_CORTA = {"normal": "normal", "anomalia_mitigada": "mitigada"}


def variante_corta(metodo, orientacion):
    return f"{metodo}<br>{orientacion.replace('_en_filas', '').replace('senales', 'señales')} en filas"


def nombre_variante(metodo, orientacion):
    return f"{metodo}, {orientacion.replace('_en_filas', '').replace('senales', 'señales')} en filas"


# ---------------------------------------------------------------- carga
def ultima_comparacion(carpeta, conjunto):
    """Ruta del resumen más reciente de ese conjunto guardado por el notebook 06 (carpeta = la de ese conjunto)."""
    rutas = sorted(glob.glob(f"{carpeta}/comparacion_{conjunto}_*.json"))   # la fecha va en el nombre
    if not rutas:
        raise FileNotFoundError(f"No hay comparacion_{conjunto}_*.json en {carpeta}: "
                                f"ejecuta antes el 06 con CONJUNTO = \"{conjunto}\"")
    return rutas[-1]


def cargar_llamadas(rutas):
    """
    Une las llamadas de varios resúmenes del 06 en una tabla.

    Parámetros:
        rutas: {conjunto: ruta del JSON}, p. ej. {"dev": ..., "test": ...}.

    Devuelve un DataFrame con una fila por llamada y las columnas añadidas:
        conjunto, ref_binaria (1/0/NaN si dudosa), acierta (bool) y tipo (anomalia, mitigada,
        normal o dudosa, según la etiqueta del autor).
    """
    partes = []
    for conjunto, ruta in rutas.items():
        with open(ruta, encoding="utf-8") as f:
            resumen = json.load(f)
        assert resumen["conjunto"] == conjunto, f"{os.path.basename(ruta)} es del conjunto {resumen['conjunto']}"
        partes.append(pd.DataFrame(resumen["llamadas"]).assign(conjunto=conjunto))
    df = pd.concat(partes, ignore_index=True)
    df["ref_binaria"] = df["referencia"].map(MAPEO_REFERENCIA)          # dudosa -> NaN
    df["acierta"] = df["veredicto"].notna() & (df["veredicto"] == df["ref_binaria"])   # sin veredicto = fallo
    df["tipo"] = df["referencia"].map(TIPO_VENTANA)
    faltan = set(MODELOS) - set(df["modelo"])
    if faltan:
        print(f"AVISO: no hay llamadas de {sorted(faltan)}")
    return df


def comprobar_etiquetas(df, rutas_ventanas):
    """
    Comprueba, ventana a ventana, que la referencia de los resúmenes es exactamente la etiqueta del autor
    guardada en las entradas del experimento (ventanas_desarrollo.csv y ventanas_prueba.csv, columnas
    id y referencia, creadas por el notebook 05).

    Parámetros:
        rutas_ventanas: {conjunto: ruta del CSV de ventanas}, p. ej. {"dev": ..., "test": ...}.

    Si alguna ventana falta o no coincide, se detiene. Devuelve el recuento de etiquetas de cada conjunto.
    """
    autor = {c: pd.read_csv(r).set_index("id")["referencia"].astype(str).str.strip()
             for c, r in rutas_ventanas.items()}
    filas = []
    for conjunto, etiquetas in autor.items():
        usadas = df[df["conjunto"] == conjunto].drop_duplicates("id").set_index("id")["referencia"]
        if usadas.empty:
            continue
        faltan = sorted(set(etiquetas.index) - set(usadas.index))
        sobran = sorted(set(usadas.index) - set(etiquetas.index))
        distintas = [i for i in usadas.index.intersection(etiquetas.index) if usadas[i] != etiquetas[i]]
        assert not (faltan or sobran or distintas), (f"{conjunto}: la referencia no es la del autor | faltan {faltan} | "
                                                     f"sobran {sobran} | distintas {distintas}")
        # Además, todas las llamadas de cada ventana (3 modelos × 4 variantes) llevan la misma referencia
        assert (df[df["conjunto"] == conjunto].groupby("id")["referencia"].nunique() == 1).all()
        filas.append({"conjunto": NOMBRE_CONJUNTO[conjunto], "ventanas": len(usadas),
                      "coinciden_con_el_autor": int((usadas == etiquetas[usadas.index]).sum()),
                      **etiquetas.value_counts().to_dict()})
    return pd.DataFrame(filas).fillna(0).set_index("conjunto")


def resumen_conjuntos(df):
    """Ventanas de cada conjunto según la etiqueta del autor (comprobación de la carga)."""
    ventanas = df.drop_duplicates(["conjunto", "id"])
    t = ventanas.pivot_table(index="conjunto", columns="referencia", values="id", aggfunc="count", fill_value=0)
    t["ventanas"] = t.sum(axis=1)
    t["evaluables (sin dudosas)"] = ventanas.groupby("conjunto")["ref_binaria"].count()
    return t.join(df.groupby("conjunto").size().rename("llamadas"))


# ---------------------------------------------------------------- tablas
def _de_tipo(g, tipo):
    # Ventanas de un tipo según la etiqueta del autor; "total" = todas menos las dudosas
    return g[g["ref_binaria"].notna()] if tipo == "total" else g[g["tipo"] == tipo]


def _k(g, tipo):
    v = _de_tipo(g, tipo)
    return int(v["acierta"].sum()), len(v)


def tabla_por_variante(df):
    """Una fila por conjunto, modelo y variante: aciertos por tipo de ventana y en total (sin las dudosas)."""
    filas = []
    for (conjunto, modelo, metodo, orientacion), g in df.groupby(["conjunto", "modelo", "metodo", "orientacion"]):
        ev = g[g["ref_binaria"].notna()]
        (a_an, n_an), (a_mi, n_mi), (a_no, n_no) = _k(ev, "anomalia"), _k(ev, "mitigada"), _k(ev, "normal")
        sens = a_an / n_an if n_an else float("nan")
        espec = (a_mi + a_no) / (n_mi + n_no) if n_mi + n_no else float("nan")
        filas.append({
            "conjunto": conjunto, "modelo": modelo, "metodo": metodo, "orientacion": orientacion,
            "evaluables": len(ev), "aciertos": int(ev["acierta"].sum()),
            "aciertos_anomalias": a_an, "anomalias": n_an,
            "aciertos_mitigadas": a_mi, "mitigadas": n_mi,
            "aciertos_normales": a_no, "normales": n_no,
            "aciertos_sin_anomalia": a_mi + a_no, "sin_anomalia": n_mi + n_no,
            "acierto_equilibrado": round((sens + espec) / 2, 3),
            # Con todas las ventanas (también las dudosas): cuánto tiende a decir «anomalía»
            "respuestas_anomalia": int((g["veredicto"] == 1).sum()), "ventanas": len(g),
            "sin_veredicto": int(g["veredicto"].isna().sum()),
        })
    t = pd.DataFrame(filas)
    orden_variante = {v: k for k, v in enumerate(VARIANTES)}
    t["_v"] = [orden_variante[(m, o)] for m, o in zip(t["metodo"], t["orientacion"])]
    t["_m"] = t["modelo"].map({m: k for k, m in enumerate(MODELOS)})
    return t.sort_values(["conjunto", "_v", "_m"]).drop(columns=["_v", "_m"]).reset_index(drop=True)


def tabla_modelos(tabla, metodo, orientacion):
    """Una variante: medidas en filas (por conjunto) y modelos en columnas."""
    t = tabla[(tabla["metodo"] == metodo) & (tabla["orientacion"] == orientacion)]
    filas = []
    for conjunto in ["dev", "test"]:
        c = t[t["conjunto"] == conjunto].set_index("modelo")
        if c.empty:
            continue
        n = c.iloc[0]
        medidas = {
            f"Aciertos en total (de {n['evaluables']})": c["aciertos"],
            f"Aciertos en anomalías (de {n['anomalias']})": c["aciertos_anomalias"],
            f"Aciertos en anomalías mitigadas (de {n['mitigadas']})": c["aciertos_mitigadas"],
            f"Aciertos en funcionamiento normal (de {n['normales']})": c["aciertos_normales"],
            "Acierto equilibrado": c["acierto_equilibrado"],
            f"Respuestas «anomalía» (de {n['ventanas']})": c["respuestas_anomalia"],
        }
        for medida, serie in medidas.items():
            filas.append({"conjunto": NOMBRE_CONJUNTO[conjunto], "medida": medida,
                          **{NOMBRE_MODELO[m]: serie.get(m) for m in MODELOS if m in serie.index}})
    # dtype object: los recuentos se muestran como enteros y el acierto equilibrado con decimales
    return pd.DataFrame(filas, dtype=object).set_index(["medida", "conjunto"])


# Medidas de la comparación ventana a ventana: (nombre, qué ventanas entran)
MEDIDAS_PARES = [
    ("Aciertos en total", lambda g: g["ref_binaria"].notna()),
    ("Aciertos en anomalías", lambda g: g["tipo"] == "anomalia"),
    ("Aciertos en anomalías mitigadas", lambda g: g["tipo"] == "mitigada"),
    ("Aciertos en funcionamiento normal", lambda g: g["tipo"] == "normal"),
]


def _por_ventana(df, modelo, metodo, orientacion, filtro):
    g = df[(df["modelo"] == modelo) & (df["metodo"] == metodo) & (df["orientacion"] == orientacion)]
    return g[filtro(g)].set_index(["conjunto", "id"])["acierta"]


def pares(df, modelo_a, modelo_b, metodo, orientacion):
    """
    Modelo B frente a modelo A, ventana a ventana: aciertos en total y por tipo de ventana. Para cada
    conjunto y para los dos juntos: ventanas que acierta solo uno de los dos y p de McNemar exacto.
    «Desarrollo + prueba» suma ventanas distintas (no se repite ninguna).
    """
    filas = []
    for medida, filtro in MEDIDAS_PARES:
        a = _por_ventana(df, modelo_a, metodo, orientacion, filtro)
        b = _por_ventana(df, modelo_b, metodo, orientacion, filtro)
        comunes = a.index.intersection(b.index)
        a, b = a.loc[comunes], b.loc[comunes]
        for conjunto in ["dev", "test", "juntos"]:
            sel = slice(None) if conjunto == "juntos" else a.index.get_level_values("conjunto") == conjunto
            aa, bb = a[sel], b[sel]
            r = mcnemar_exacto(aa, bb)
            filas.append({"medida": medida, "conjunto": NOMBRE_CONJUNTO[conjunto], "ventanas": len(aa),
                          NOMBRE_MODELO[modelo_a]: int(aa.sum()), NOMBRE_MODELO[modelo_b]: int(bb.sum()),
                          f"solo {NOMBRE_MODELO[modelo_b]}": r["solo_b"], f"solo {NOMBRE_MODELO[modelo_a]}": r["solo_a"],
                          "p_mcnemar": r["p_mcnemar"]})
    return pd.DataFrame(filas)


def pares_todas_las_variantes(df, medida):
    """Una medida de MEDIDAS_PARES: cada par de modelos en cada variante, por conjunto."""
    filas = []
    for metodo, orientacion in VARIANTES:
        for a, b in combinations(MODELOS, 2):
            t = pares(df, a, b, metodo, orientacion)
            for r in t[t["medida"] == medida].to_dict("records"):
                filas.append({"variante": nombre_variante(metodo, orientacion), "modelo_a": NOMBRE_MODELO[a],
                              "modelo_b": NOMBRE_MODELO[b], "conjunto": r["conjunto"], "ventanas": r["ventanas"],
                              "aciertos_a": r[NOMBRE_MODELO[a]], "aciertos_b": r[NOMBRE_MODELO[b]],
                              "solo_a": r[f"solo {NOMBRE_MODELO[a]}"], "solo_b": r[f"solo {NOMBRE_MODELO[b]}"],
                              "p_mcnemar": r["p_mcnemar"]})
    return pd.DataFrame(filas)


def tabla_por_tipo(df):
    """Aciertos por tipo de ventana, modelo y variante, en desarrollo, en prueba y juntos. Celdas «k de n»."""
    filas = []
    for conjunto in ["dev", "test", "juntos"]:
        d = df if conjunto == "juntos" else df[df["conjunto"] == conjunto]
        for (modelo, metodo, orientacion), g in d.groupby(["modelo", "metodo", "orientacion"]):
            for tipo, nombre in TIPOS.items():
                k, n = _k(g, tipo)
                filas.append({"variante": nombre_variante(metodo, orientacion), "modelo": NOMBRE_MODELO[modelo],
                              "conjunto": NOMBRE_CONJUNTO[conjunto], "tipo": f"Aciertos en {nombre}",
                              "celda": f"{k} de {n}"})
    t = pd.DataFrame(filas).pivot_table(index=["variante", "modelo"], columns=["conjunto", "tipo"],
                                        values="celda", aggfunc="first")
    orden = pd.MultiIndex.from_product([[nombre_variante(*v) for v in VARIANTES], [NOMBRE_MODELO[m] for m in MODELOS]])
    columnas = [(NOMBRE_CONJUNTO[c], f"Aciertos en {n}") for c in ["dev", "test", "juntos"] for n in TIPOS.values()]
    return t.reindex(orden.intersection(t.index, sort=False))[[c for c in columnas if c in t.columns]]


def intervalo_clopper_pearson(k, n, confianza=0.95):
    """
    Intervalo de confianza exacto (Clopper y Pearson, 1934) de una proporción de aciertos k/n.
    Con k = 0 el límite inferior es 0 y con k = n el superior es 1. Devuelve (inferior, superior).
    """
    if n == 0:
        return float("nan"), float("nan")
    a = 1 - confianza
    inferior = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    superior = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return float(inferior), float(superior)


def _con_intervalo(k, n):
    # «k de n [inf–sup]», con coma decimal como en la memoria
    inf, sup = intervalo_clopper_pearson(k, n)
    return f"{k} de {n} [{inf:.2f}–{sup:.2f}]".replace(".", ",")


def tabla_resumen(df, metodo, orientacion, modelo=MODELOS[2], intervalos=True):
    """
    Tabla para la memoria: una variante, aciertos de los tres modelos por tipo de ventana (según la
    etiqueta del autor) y conjunto, y la comparación ventana a ventana de `modelo` con cada uno de los
    otros dos: «a / b (p)» = ventanas que acierta solo `modelo` / solo el otro, y p de McNemar exacto.
    intervalos: añade a cada «k de n» su intervalo de confianza al 95 % (Clopper-Pearson).
    """
    otros = [m for m in MODELOS if m != modelo]
    filas = []
    for medida, filtro in MEDIDAS_PARES[1:] + MEDIDAS_PARES[:1]:          # tipos de ventana primero, total al final
        aciertos = {m: _por_ventana(df, m, metodo, orientacion, filtro) for m in MODELOS}
        for conjunto in ["dev", "test", "juntos"]:
            sel = {m: (a if conjunto == "juntos" else a[a.index.get_level_values("conjunto") == conjunto])
                   for m, a in aciertos.items()}
            fila = {"tipo de ventana": medida.replace("Aciertos en ", "").capitalize(),
                    "conjunto": NOMBRE_CONJUNTO[conjunto]}
            for m in MODELOS:
                # «k de n» y no «k/n»: Excel convertiría «2/4» en una fecha al abrir el CSV
                k, n = int(sel[m].sum()), len(sel[m])
                fila[NOMBRE_MODELO[m]] = _con_intervalo(k, n) if intervalos else f"{k} de {n}"
            for o in otros:
                comunes = sel[modelo].index.intersection(sel[o].index)
                r = mcnemar_exacto(sel[modelo].loc[comunes], sel[o].loc[comunes])
                # «solo modelo / solo el otro (p)», con coma decimal como en la memoria
                fila[f"{NOMBRE_MODELO[modelo]} frente a {NOMBRE_MODELO[o]}"] = (
                    f"{r['solo_a']} / {r['solo_b']} (p = {r['p_mcnemar']:.3f})".replace(".", ","))
            filas.append(fila)
    return pd.DataFrame(filas).set_index(["tipo de ventana", "conjunto"])


def cambios_orientacion(df):
    """Ventanas (todas, también las dudosas) cuyo veredicto cambia al girar la tabla."""
    filas = []
    for (conjunto, modelo, metodo), g in df.groupby(["conjunto", "modelo", "metodo"]):
        p = g.pivot(index="id", columns="orientacion", values="veredicto")
        cambian = (p["senales_en_filas"] != p["reportes_en_filas"]).sum()   # NaN cuenta como distinto
        filas.append({"conjunto": NOMBRE_CONJUNTO[conjunto], "metodo": metodo, "modelo": NOMBRE_MODELO[modelo],
                      "cambian": int(cambian), "ventanas": len(p)})
    t = pd.DataFrame(filas)
    t["celda"] = t["cambian"].astype(str) + " de " + t["ventanas"].astype(str)
    return (t.pivot_table(index="modelo", columns=["metodo", "conjunto"], values="celda", aggfunc="first")
            .reindex([NOMBRE_MODELO[m] for m in MODELOS]))


def sin_anomalia_por_ventana(df, metodo, orientacion):
    """Ventanas mitigadas y normales (filas) y modelos (columnas): 1 = acierta (responde «normal»)."""
    g = df[(df["metodo"] == metodo) & (df["orientacion"] == orientacion) & (df["tipo"].isin(["mitigada", "normal"]))]
    t = g.pivot_table(index=["conjunto", "id"], columns="modelo", values="acierta", aggfunc="first")
    t = t[[m for m in MODELOS if m in t.columns]].astype(int)
    info = g.drop_duplicates(["conjunto", "id"]).set_index(["conjunto", "id"])[["referencia", "grupo"]]
    t = info.join(t)
    t["modelos_que_aciertan"] = t[[m for m in MODELOS if m in t.columns]].sum(axis=1)
    t["_orden"] = t["referencia"].map({"anomalia_mitigada": 0, "normal": 1})   # mitigadas primero
    return t.sort_values(["conjunto", "_orden", "modelos_que_aciertan"], ascending=[True, True, False]).drop(columns="_orden")


# ---------------------------------------------------------------- gráficas
def _estilo(fig, titulo, alto, pie=None):
    fig.update_layout(
        title=dict(text=titulo, x=0, xanchor="left", y=0.98, yref="container", yanchor="top",
                   font=dict(size=16, color=TINTA)),
        height=alto, plot_bgcolor=FONDO, paper_bgcolor=FONDO,
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", size=12, color=TINTA_SUAVE),
        legend=dict(orientation="h", yanchor="bottom", y=1.1, x=0, xanchor="left", title_text=""),
        margin=dict(l=60, r=30, t=120, b=60 + (150 if pie else 0)), hoverlabel=dict(bgcolor="white", font_color=TINTA),
    )
    fig.update_xaxes(showgrid=False, linecolor=REJILLA, ticks="", tickangle=0)
    fig.update_yaxes(gridcolor=REJILLA, zeroline=True, zerolinecolor=REJILLA, linecolor=REJILLA)
    fig.update_annotations(font=dict(color=TINTA, size=13))
    if pie:
        # Pie con el origen de cada conjunto, debajo del gráfico
        fig.add_annotation(x=0, y=0, xref="paper", yref="paper", yshift=-70, xanchor="left", yanchor="top",
                           showarrow=False, align="left", text=pie, font=dict(color=TINTA_SUAVE, size=11))
    return fig


def _pie_conjuntos(df, conjuntos, tipo=None, ancho=150):
    """Texto del pie: origen de cada conjunto y, si se da un tipo, cuántas ventanas de ese tipo tiene."""
    lineas = []
    for c in conjuntos:
        if c not in ORIGEN_CONJUNTO:
            continue
        linea = ORIGEN_CONJUNTO[c]
        if tipo:
            n = _de_tipo(df[df["conjunto"] == c], tipo)["id"].nunique()
            linea += (f" Sin las dudosas: {n}." if tipo == "total" else f" Con la etiqueta «{TIPOS[tipo]}»: {n}.")
        lineas.append(_partir(linea, ancho))
    lineas.append(_partir("Las ventanas dudosas quedan fuera. Sobre cada barra: aciertos / ventanas de ese tipo "
                          "(0/n = el modelo no acierta ninguna).", ancho))
    return "<br>".join(lineas)


def _partir(texto, ancho=150):
    # Plotly no parte las anotaciones: se insertan saltos de línea a mano
    palabras, lineas, actual = texto.split(" "), [], ""
    for p in palabras:
        if len(actual) + len(p) + 1 > ancho:
            lineas.append(actual)
            actual = p
        else:
            actual = f"{actual} {p}".strip()
    return "<br>".join(lineas + [actual])


# Disposición de los 4 paneles (uno por variante): filas, columnas, ancho fijo (None = el del contenedor),
# alto, separaciones y caracteres por línea del pie. "2x2" es más compacta para la memoria.
DISPOSICIONES = {"1x4": dict(filas=1, columnas=4, ancho=None, alto=520, sep_h=0.03, sep_v=0.0, pie=150),
                 "2x2": dict(filas=2, columnas=2, ancho=820, alto=820, sep_h=0.06, sep_v=0.14, pie=120)}


def grafica_aciertos_tipo(df, tipo, conjuntos=("dev", "test"), disposicion="1x4"):
    """
    Aciertos de cada modelo en las ventanas que el autor etiquetó con un tipo («anomalia», «mitigada»
    o «normal»; «total» = todas menos las dudosas), por variante (paneles) y conjunto (eje horizontal). Altura de la barra = % de
    aciertos; encima, «aciertos/ventanas». Al pie, de dónde sale cada conjunto y cómo se eligió.
    disposicion: "1x4" (una fila de 4 paneles) o "2x2" (dos filas de 2, para la memoria).
    """
    dp = DISPOSICIONES[disposicion]
    d = _de_tipo(df, tipo)
    conjuntos = [c for c in conjuntos if c == "juntos" or c in set(d["conjunto"])]
    fig = make_subplots(rows=dp["filas"], cols=dp["columnas"], shared_yaxes=True,
                        horizontal_spacing=dp["sep_h"], vertical_spacing=dp["sep_v"],
                        subplot_titles=[nombre_variante(*v) for v in VARIANTES])
    for k, (metodo, orientacion) in enumerate(VARIANTES, start=1):
        fila, col = (k - 1) // dp["columnas"] + 1, (k - 1) % dp["columnas"] + 1
        v = d[(d["metodo"] == metodo) & (d["orientacion"] == orientacion)]
        for modelo in MODELOS:
            g = v[v["modelo"] == modelo]
            if g.empty:
                continue
            cuentas = [_k(g if c == "juntos" else g[g["conjunto"] == c], tipo) for c in conjuntos]
            fig.add_trace(go.Bar(
                x=[NOMBRE_CONJUNTO[c] for c in conjuntos],
                y=[100 * a / n if n else None for a, n in cuentas],
                text=[f"{a}/{n}" for a, n in cuentas], textposition="outside", cliponaxis=False,
                textfont=dict(size=10, color=TINTA_SUAVE),
                name=NOMBRE_MODELO[modelo], marker_color=COLOR_MODELO[modelo], showlegend=(k == 1),
                legendgroup=modelo, marker_line=dict(color=FONDO, width=1.5),
                hovertemplate=f"{NOMBRE_MODELO[modelo]} · {nombre_variante(metodo, orientacion)}<br>%{{x}}: "
                              f"%{{text}} aciertos<extra></extra>"), row=fila, col=col)
    separador = "<br>" if dp["columnas"] < 4 else " "   # en 2x2 el título va en dos líneas
    titulo = (f"Aciertos en total, según la etiqueta del autor{separador}(anomalías, mitigadas y normales; "
              f"sin las dudosas)" if tipo == "total" else
              f"Aciertos en las ventanas etiquetadas por el autor como {TIPOS[tipo]}{separador}"
              f"(acierto = responder {RESPUESTA_CORRECTA[tipo]})")
    _estilo(fig, titulo, dp["alto"],
            pie=_pie_conjuntos(df, conjuntos, tipo, dp["pie"]))
    fig.update_layout(barmode="group", bargap=0.22, bargroupgap=0.05, uniformtext=dict(minsize=9, mode="show"),
                      width=dp["ancho"], margin=dict(t=140 if dp["filas"] > 1 else 120))
    fig.update_yaxes(range=[0, 112], tickvals=[0, 25, 50, 75, 100])
    for fila in range(1, dp["filas"] + 1):
        fig.update_yaxes(title_text="Aciertos (%)", row=fila, col=1)
    return fig


def grafica_sin_anomalia_por_ventana(df, metodo, orientacion):
    """
    Cada ventana mitigada o normal según el autor (filas, con su etiqueta) y cada modelo (columnas):
    celda oscura = acierta (responde «normal»). Permite ver si un modelo acierta las mismas que otro y alguna más.
    """
    t = sin_anomalia_por_ventana(df, metodo, orientacion)
    modelos = [m for m in MODELOS if m in t.columns]
    conjuntos = [c for c in ["dev", "test"] if c in t.index.get_level_values(0)]
    fig = make_subplots(rows=1, cols=len(conjuntos), horizontal_spacing=0.2,
                        subplot_titles=[f"{NOMBRE_CONJUNTO[c]}: {t.loc[c].shape[0]} ventanas mitigadas o normales"
                                        for c in conjuntos])
    for k, conjunto in enumerate(conjuntos, start=1):
        c = t.loc[conjunto].iloc[::-1]   # mitigadas arriba
        z = c[modelos].values
        filas = [f"{i} · {ETIQUETA_CORTA.get(r, r)}" for i, r in zip(c.index, c["referencia"])]   # etiqueta del autor
        fig.add_trace(go.Heatmap(
            z=z, x=[NOMBRE_MODELO[m] for m in modelos], y=filas,
            colorscale=[[0, REJILLA], [1, TINTA_SUAVE]], zmin=0, zmax=1, showscale=False, xgap=2, ygap=2,
            text=[["✓" if v else "" for v in fila] for fila in z], texttemplate="%{text}",
            textfont=dict(color="white", size=13),
            customdata=[[("acierta: responde «normal»" if v else "falla: responde «anomalía»") for v in fila] for fila in z],
            hovertemplate="%{y} · %{x}<br>%{customdata}<extra></extra>"), row=1, col=k)
    alto = 160 + 26 * max(len(t.loc[c]) for c in conjuntos)
    _estilo(fig, f"Aciertos ventana a ventana en las anomalías mitigadas y el funcionamiento normal · "
                 f"{nombre_variante(metodo, orientacion)}", alto + 150,
            pie=_pie_conjuntos(df, conjuntos).replace("Sobre cada barra: aciertos / ventanas de ese tipo "
                                                      "(0/n = el modelo no acierta ninguna).",
                                                      "✓ celda oscura: acierta (responde «normal») · "
                                                      "celda clara: falla (responde «anomalía»)."))
    fig.update_xaxes(side="top", showline=False)
    fig.update_yaxes(showgrid=False, showline=False, zeroline=False, tickfont=dict(size=11))
    for a in fig.layout.annotations[:len(conjuntos)]:   # títulos de los paneles por encima de los modelos
        a.yshift = 28
    return fig
