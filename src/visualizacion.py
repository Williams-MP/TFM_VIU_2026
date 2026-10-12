"""
Visualización de ventanas de telemetría (12 eventos x N señales).

Gráficas interactivas con Plotly para inspeccionar las ventanas tal cual
se inyectan en los prompts del LLM. Solo sirven para inspeccionar: no
modifican los datos ni lo que recibe el prompt.

El renderizador de Plotly NO se fija aquí, porque depende del entorno
de ejecución (en VS Code con kernel de Colab hace falta
pio.renderers.default = "vscode"); se configura en el notebook.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Colores de la tabla: celdas con valor distinto de 0 resaltadas (en un sistema
# sin incidencias los contadores valen 0, así que lo distinto de 0 es lo que hay que leer)
_COLOR_CELDA_CERO = "white"
_COLOR_CELDA_NO_CERO = "#FAEEDA"
_COLOR_CABECERA = "#E6F1FB"


def _series_por_escala(ventana, n_senales):
    """Valores de cada señal en escala real y en escala signo·log(1+|x|), como listas."""
    y_reales = [ventana[:, i].tolist() for i in range(n_senales)]
    y_log = [(np.sign(ventana[:, i]) * np.log1p(np.abs(ventana[:, i]))).tolist() for i in range(n_senales)]
    return y_reales, y_log


def _botones(y_reales, y_log, indices_trazas, x_botones_visibilidad):
    """
    Botones de la gráfica: cambio de escala (real / log) y mostrar/ocultar todas.
    Se aplican solo a las trazas de líneas (indices_trazas), no a la tabla.
    """
    n = len(indices_trazas)
    return [
        dict(type="buttons", direction="right", x=0, y=1.12, xanchor="left", showactive=True,
             buttons=[
                 dict(label="Escala real", method="update",
                      args=[{"y": y_reales}, {"yaxis.title.text": "Valor real"}, indices_trazas]),
                 dict(label="Escala log(1+x)", method="update",
                      args=[{"y": y_log}, {"yaxis.title.text": "signo · log(1 + |valor|)"}, indices_trazas]),
             ]),
        dict(type="buttons", direction="right", x=x_botones_visibilidad, y=1.12, xanchor="left", showactive=False,
             buttons=[
                 dict(label="Mostrar todas", method="restyle",
                      args=[{"visible": [True] * n}, indices_trazas]),
                 dict(label="Ocultar todas", method="restyle",
                      args=[{"visible": ["legendonly"] * n}, indices_trazas]),
             ]),
    ]


def _trazas_lineas(eventos, nombres, y_reales):
    """Una traza de líneas por señal; el hover muestra siempre el valor real."""
    return [
        go.Scatter(
            x=eventos, y=y_reales[i], mode="lines+markers", name=nombre,
            customdata=y_reales[i],
            hovertemplate=f"{nombre}<br>%{{x}}: %{{customdata:,.0f}}<extra></extra>",
        )
        for i, nombre in enumerate(nombres)
    ]


def graficar_ventana(ventana, nombres, titulo):
    """
    Gráfica interactiva con todas las señales de una ventana.

    - Clic en una señal de la leyenda: la muestra u oculta.
    - Doble clic en una señal: deja solo esa señal.
    - Botones: escala real o log(1+x), y mostrar/ocultar todas las señales.
    - Al pasar el ratón se ve siempre el valor real, sea cual sea la escala.

    Parámetros:
        ventana (np.ndarray): una ventana, forma (12, n_senales).
        nombres (list[str]): nombre de cada señal, en el orden de las columnas.
        titulo (str): título de la gráfica.

    Devuelve:
        plotly.graph_objects.Figure: la figura (ya mostrada).
    """
    eventos = [f"ev{i+1}" for i in range(ventana.shape[0])]  # mismas etiquetas que la tabla del prompt
    n_senales = len(nombres)
    y_reales, y_log = _series_por_escala(ventana, n_senales)

    figura = go.Figure(_trazas_lineas(eventos, nombres, y_reales))
    figura.update_layout(
        title=titulo, height=560, margin=dict(t=120),
        xaxis_title="Evento de la ventana", yaxis_title="Valor real",
        legend=dict(title="Señales (clic: mostrar/ocultar)"),
        updatemenus=_botones(y_reales, y_log, list(range(n_senales)), x_botones_visibilidad=0.45),
    )
    figura.show()
    return figura


def graficar_ventana_y_tabla(ventana, nombres, titulo, mostrar=True):
    """
    Gráfica de la ventana (izquierda) y tabla con sus valores (derecha), en una
    sola figura, para analizar la respuesta del LLM viendo exactamente los datos
    que ha recibido.

    - La gráfica tiene la misma interacción que graficar_ventana() (leyenda,
      escala real/log, mostrar/ocultar todas).
    - En la tabla, las celdas con valor distinto de 0 aparecen resaltadas.

    Parámetros:
        ventana (np.ndarray): una ventana, forma (12, n_senales), con los
            valores que recibe el LLM.
        nombres (list[str]): nombre de cada señal, en el orden de las columnas.
        titulo (str): título de la figura.
        mostrar (bool): si es False, solo devuelve la figura (p. ej. para guardarla en un HTML).

    Devuelve:
        plotly.graph_objects.Figure: la figura.
    """
    n_eventos = ventana.shape[0]
    eventos = [f"ev{i+1}" for i in range(n_eventos)]
    n_senales = len(nombres)
    y_reales, y_log = _series_por_escala(ventana, n_senales)

    figura = make_subplots(
        rows=1, cols=2, column_widths=[0.5, 0.5], horizontal_spacing=0.04,
        specs=[[{"type": "xy"}, {"type": "table"}]],
    )

    # Izquierda: una línea por señal (trazas 0 .. n_senales-1)
    for traza in _trazas_lineas(eventos, nombres, y_reales):
        figura.add_trace(traza, row=1, col=1)

    # Derecha: tabla con una fila por señal y una columna por evento (igual que la tabla del prompt)
    valores_por_columna = [nombres] + [[f"{int(v):,}" for v in ventana[j, :]] for j in range(n_eventos)]
    colores_por_columna = [[_COLOR_CELDA_CERO] * n_senales] + [
        [_COLOR_CELDA_NO_CERO if v != 0 else _COLOR_CELDA_CERO for v in ventana[j, :]] for j in range(n_eventos)
    ]
    figura.add_trace(
        go.Table(
            columnwidth=[3.2] + [1] * n_eventos,
            header=dict(values=["Señal"] + eventos, fill_color=_COLOR_CABECERA, align="center",
                        font=dict(size=11)),
            cells=dict(values=valores_por_columna, fill_color=colores_por_columna,
                       align=["left"] + ["right"] * n_eventos, font=dict(size=11), height=24),
        ),
        row=1, col=2,
    )

    figura.update_layout(
        title=titulo, height=620, margin=dict(t=110, b=40),
        xaxis_title="Evento de la ventana", yaxis_title="Valor real",
        # Leyenda debajo de la gráfica para no tapar la tabla
        legend=dict(title="Señales (clic: mostrar/ocultar)", orientation="h",
                    yanchor="top", y=-0.15, x=0, xanchor="left", font=dict(size=10)),
        updatemenus=_botones(y_reales, y_log, list(range(n_senales)), x_botones_visibilidad=0.22),
    )
    if mostrar:
        figura.show()
    return figura
