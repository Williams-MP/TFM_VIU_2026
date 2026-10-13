"""
Prompts del modelo de lenguaje, guardados como ficheros en la carpeta privada de Drive.

Estructura (fuera del repositorio, por el acuerdo de confidencialidad):

    {RUTA_DRIVE}/privado/
        nombres_senales.json                 nombre corto -> nombre que ve el modelo, y marcadores
        prompts/
            bloques/<tipo>/<nombre>.v<N>.md  texto de un bloque (tipos: rol, instruccion, contexto,
                                             procedimiento, esquema)
            variantes/<nombre>.v<N>.json     qué bloques (y versión) lleva cada variante, qué
                                             representación de la ventana, cuántos tokens de salida
                                             y, si hace falta, el orden de las piezas del mensaje

Los textos se escriben con los nombres cortos de las señales (S01ERR, S03...). Al montar el prompt,
`poner_nombres()` los cambia por los nombres de nombres_senales.json, que es el único sitio donde
están los nombres descriptivos. Ese fichero también puede definir marcadores como {{PREFIJO}} para
fragmentos de texto que no son un nombre completo.

Cada fichero .md es el texto exacto del bloque. Si el editor añade un salto de línea al final, se quita
al leerlo.
"""

import json
import os
import re

import pandas as pd

from config import NOMBRES_CONTADORES, RUTA_DRIVE
from . import representaciones as rep
from .huellas import huella_texto

TIPOS_BLOQUE = ["rol", "instruccion", "contexto", "procedimiento", "esquema"]
OPCIONALES = {"contexto", "procedimiento"}
# Orden de las piezas del mensaje de usuario si la variante no define «orden» (el del Experimento 2)
ORDEN_POR_DEFECTO = ["instruccion", "contexto", "datos", "procedimiento", "esquema"]
RUTA_PRIVADA = f"{RUTA_DRIVE}/privado"


def _leer_texto(ruta):
    with open(ruta, encoding="utf-8") as f:
        texto = f.read()
    return texto[:-1] if texto.endswith("\n") else texto


def _versiones(carpeta, nombre, extension):
    # Versiones guardadas de <nombre>.v<N>.<extension>, ordenadas
    if not os.path.isdir(carpeta):
        return []
    patron = re.compile(rf"^{re.escape(nombre)}\.v(\d+)\.{extension}$")
    return sorted(int(m.group(1)) for f in os.listdir(carpeta) if (m := patron.match(f)))


def _separar_ref(ref):
    # "nombre@v2" -> ("nombre", 2); "nombre" -> ("nombre", None)
    if "@v" in ref:
        nombre, version = ref.split("@v")
        return nombre, int(version)
    return ref, None


# ---------------------------------------------------------------- nombres
def leer_correspondencia(ruta=None):
    """
    Correspondencia nombre corto -> texto que ve el modelo, y marcadores {{CLAVE}} -> texto.
    Sin el fichero privado, cada nombre corto se queda como está (el código funciona igual).
    """
    ruta = ruta or os.environ.get("TFM_NOMBRES_LLM", f"{RUTA_PRIVADA}/nombres_senales.json")
    if not os.path.exists(ruta):
        print("Aviso: no se encuentra el fichero privado de nombres; el modelo verá los nombres cortos.")
        return {n: n for n in NOMBRES_CONTADORES}
    with open(ruta, encoding="utf-8") as f:
        correspondencia = json.load(f)
    faltan = [n for n in NOMBRES_CONTADORES if n not in correspondencia]
    if faltan:
        raise ValueError(f"El fichero de nombres no incluye: {faltan}")
    return correspondencia


def poner_nombres(texto, correspondencia):
    """Cambia, en una sola pasada, los nombres cortos y los marcadores {{CLAVE}} por su texto."""
    if not texto:
        return texto
    marcadores = {f"{{{{{k}}}}}": v for k, v in correspondencia.items() if k not in NOMBRES_CONTADORES}
    for marcador, valor in marcadores.items():
        texto = texto.replace(marcador, valor)
    # Los nombres más largos primero (S03H01 antes que S03), y solo palabras completas
    nombres = sorted(NOMBRES_CONTADORES, key=len, reverse=True)
    patron = re.compile(r"\b(" + "|".join(map(re.escape, nombres)) + r")\b")
    return patron.sub(lambda m: correspondencia[m.group(1)], texto)


def quitar_nombres(texto, correspondencia):
    """
    Lo contrario de poner_nombres(): para pasar a ficheros un texto que ya existe.
    Comprueba que, al volver a poner los nombres, sale exactamente el texto original.
    """
    inverso = {correspondencia[n]: n for n in NOMBRES_CONTADORES}
    patron = re.compile(r"\b(" + "|".join(map(re.escape, sorted(inverso, key=len, reverse=True))) + r")\b")
    resultado = patron.sub(lambda m: inverso[m.group(1)], texto)
    for clave, valor in correspondencia.items():   # marcadores para lo que no es un nombre completo
        if clave not in NOMBRES_CONTADORES and valor:
            resultado = resultado.replace(valor, f"{{{{{clave}}}}}")
    assert poner_nombres(resultado, correspondencia) == texto, "la sustitución de nombres no es reversible"
    return resultado


# ---------------------------------------------------------------- prompts
class Prompts:
    """
    Bloques y variantes guardados en ficheros, y montaje de los mensajes que recibe el modelo.

    Parámetros:
        ruta: carpeta prompts/ (por defecto, {RUTA_DRIVE}/privado/prompts).
        ruta_nombres: nombres_senales.json (por defecto, el de la carpeta privada).
    """

    def __init__(self, ruta=None, ruta_nombres=None):
        self.ruta = ruta or f"{RUTA_PRIVADA}/prompts"
        self.correspondencia = leer_correspondencia(ruta_nombres)
        self.nombres_llm = [self.correspondencia[n] for n in NOMBRES_CONTADORES]   # nombres en las tablas

    # --- lectura
    def bloque(self, tipo, ref):
        """Texto de un bloque tal como está en el fichero (con nombres cortos). ref: "nombre@vN" o "nombre"."""
        nombre, version = _separar_ref(ref)
        carpeta = f"{self.ruta}/bloques/{tipo}"
        versiones = _versiones(carpeta, nombre, "md")
        if not versiones:
            raise KeyError(f"No existe el bloque {tipo}/{nombre}")
        return _leer_texto(f"{carpeta}/{nombre}.v{version or versiones[-1]}.md")

    def variante(self, ref):
        """Definición de una variante. ref: "nombre@vN" o "nombre" (la versión más reciente)."""
        nombre, version = _separar_ref(ref)
        carpeta = f"{self.ruta}/variantes"
        versiones = _versiones(carpeta, nombre, "json")
        if not versiones:
            raise KeyError(f"No existe la variante {nombre}")
        version = version or versiones[-1]
        with open(f"{carpeta}/{nombre}.v{version}.json", encoding="utf-8") as f:
            return {"nombre": nombre, "version": version, **json.load(f)}

    def listar_variantes(self):
        """Tabla con todas las variantes, sus bloques y parámetros."""
        filas = []
        carpeta = f"{self.ruta}/variantes"
        for f in sorted(os.listdir(carpeta)):
            m = re.match(r"^(.+)\.v(\d+)\.json$", f)
            if m:
                d = self.variante(f"{m.group(1)}@v{m.group(2)}")
                filas.append({"variante": f"{d['nombre']}@v{d['version']}",
                              **{t: d["bloques"].get(t) or "—" for t in TIPOS_BLOQUE},
                              "representaciones": "+".join(d["representaciones"]),
                              "max_new_tokens": d["max_new_tokens"], "tipo_respuesta": d["tipo_respuesta"]})
        return pd.DataFrame(filas)

    # --- montaje
    def resolver(self, ref):
        """Textos de todas las piezas de una variante, ya con los nombres que ve el modelo."""
        d = self.variante(ref)
        textos = {}
        for tipo in TIPOS_BLOQUE:
            b = d["bloques"].get(tipo)
            if b is None and tipo not in OPCIONALES:
                raise ValueError(f"{ref}: falta el bloque de tipo {tipo}")
            textos[tipo] = poner_nombres(self.bloque(tipo, b), self.correspondencia) if b else None
        if "{datos}" in textos["instruccion"]:
            datos = " y ".join(rep.REPRESENTACIONES[r].descripcion for r in d["representaciones"])
            textos["instruccion"] = textos["instruccion"].replace("{datos}", datos)
        return {**textos, "id": f"{d['nombre']}@v{d['version']}", "representaciones": d["representaciones"],
                "max_new_tokens": d["max_new_tokens"], "tipo_respuesta": d["tipo_respuesta"],
                "orden": d.get("orden", ORDEN_POR_DEFECTO)}

    def _partes_usuario(self, r, datos):
        # Piezas del mensaje de usuario en el orden de la variante; las opcionales vacías se omiten.
        # datos(nombre de la representación) -> texto que va en el lugar de los datos.
        partes = []
        for pieza in r["orden"]:
            if pieza == "datos":
                partes += [datos(n) for n in r["representaciones"]]
            elif r[pieza]:
                partes.append(r[pieza])
        return partes

    def montar_mensajes(self, ref, ventana):
        """
        Mensajes de chat de una variante para una ventana (12, 11) en enteros.
        Orden del mensaje de usuario: el de la variante («orden»); si no lo define,
        instrucción -> contexto -> datos -> procedimiento -> esquema (Experimento 2).
        """
        r = self.resolver(ref)
        partes = self._partes_usuario(r, lambda n: rep.mostrar(n, ventana, self.nombres_llm))
        return [{"role": "system", "content": r["rol"]}, {"role": "user", "content": "\n\n".join(partes)}]

    def texto_prompt(self, ref):
        """
        Texto completo del prompt sin los datos: «### SISTEMA», el rol, «### USUARIO» y las piezas del mensaje
        de usuario, con un marcador {NOMBRE_DE_LA_REPRESENTACION} en lugar de cada tabla. Es el formato en que
        el Experimento 3 guarda el prompt, y su huella (huella_texto) identifica las ejecuciones.
        """
        r = self.resolver(ref)
        partes = self._partes_usuario(r, lambda n: "{" + n.upper() + "}")
        return f"### SISTEMA\n{r['rol']}\n### USUARIO\n" + "\n\n".join(partes)

    def huella_variante(self, ref):
        """Huella de todo lo que define la variante: textos, representaciones (con versión) y tokens."""
        r = self.resolver(ref)
        reprs = "+".join(f"{n}:{rep.REPRESENTACIONES[n].version}" for n in r["representaciones"])
        extra = [] if r["orden"] == ORDEN_POR_DEFECTO else ["+".join(r["orden"])]   # Exp. 2: misma huella que antes
        return huella_texto(*(r[t] or "" for t in TIPOS_BLOQUE), reprs, str(r["max_new_tokens"]), *extra)
