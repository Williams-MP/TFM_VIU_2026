"""
Genera un conjunto de datos sintético con el mismo formato que las ventanas
reales, para poder ejecutar los notebooks sin los datos originales, que
están protegidos por un acuerdo de confidencialidad.

Los valores son aleatorios: sirven para comprobar que el código funciona de
principio a fin, no para reproducir los resultados de la memoria.

Formato de salida (descrito también en el README):
    <salida>/train/ventanas.parquet
    <salida>/val/ventanas.parquet
    <salida>/test/ventanas.parquet
Una fila por ventana, con las columnas device_pseudo_id, session_id,
window_id, secuencia (12 reportes x 13 variables), inicio_ventana y
fin_ventana.

Uso:
    python generar_datos_sinteticos.py --salida /content/datos_sinteticos
"""

import argparse
import hashlib
import os

import numpy as np
import pandas as pd

N_REPORTES = 12      # reportes por ventana
N_CONTADORES = 11    # S01ERR ... S02REC02
INTERVALO_MIN = 30   # cadencia habitual entre reportes, en minutos


def generar_ventana(rng):
    """
    Genera una ventana de 12 x 13 valores con un comportamiento parecido al
    de los datos reales: la mayoría de contadores a 0 y, de vez en cuando,
    valores pequeños o ráfagas.
    """
    ventana = np.zeros((N_REPORTES, N_CONTADORES + 2))

    # Contadores: inflación de ceros con cola larga.
    activos = rng.random((N_REPORTES, N_CONTADORES)) < 0.15
    valores = rng.geometric(p=0.3, size=(N_REPORTES, N_CONTADORES))
    rafagas = rng.random((N_REPORTES, N_CONTADORES)) < 0.01
    valores = np.where(rafagas, rng.integers(100, 50_000, size=valores.shape), valores)
    ventana[:, :N_CONTADORES] = np.where(activos, valores, 0)

    # V01: minutos desde el reporte anterior (múltiplos de 30, a veces mayores).
    ventana[:, 11] = INTERVALO_MIN * rng.choice([1, 1, 1, 1, 2, 3], size=N_REPORTES)
    ventana[0, 11] = 0 if rng.random() < 0.1 else ventana[0, 11]

    # V02: indicador de salto brusco en S01ERR (casi siempre 0).
    ventana[:, 12] = 0
    return ventana


def anadir_casos_especiales(ventana, rng, tipo):
    """
    Introduce en la ventana uno de los dos comportamientos descritos en la
    memoria, para que el código que los trata tenga casos sobre los que
    actuar.

    tipo = "valor_imposible": S01ERR con un valor de magnitud > 10^6.
    tipo = "meseta": S04L01 fijo en un mismo valor y el resto de contadores a 0.
    """
    if tipo == "valor_imposible":
        reporte = rng.integers(0, N_REPORTES)
        ventana[reporte, 0] = rng.choice([2_128_852_019, -2_074_940_405, 3_500_000])
        ventana[reporte, 12] = 1
    elif tipo == "meseta":
        ventana[:, :N_CONTADORES] = 0
        ventana[:, 5] = 1385  # posición de S04L01
    return ventana


def generar_conjunto(n_ventanas, inicio, rng):
    """Genera las filas de un conjunto (train, val o test). Los dispositivos se repiten entre conjuntos, como en los datos reales."""
    filas = []
    n_dispositivos = max(1, n_ventanas // 20)
    for i in range(n_ventanas):
        ventana = generar_ventana(rng)
        sorteo = rng.random()
        if sorteo < 0.01:
            ventana = anadir_casos_especiales(ventana, rng, "valor_imposible")
        elif sorteo < 0.03:
            ventana = anadir_casos_especiales(ventana, rng, "meseta")

        dispositivo = i % n_dispositivos
        # Seudónimo con el mismo aspecto que los reales (hash de 64 caracteres).
        device_pseudo_id = hashlib.sha256(f"dispositivo-{dispositivo}".encode()).hexdigest()
        inicio_ventana = inicio + pd.Timedelta(minutes=int(rng.integers(0, 60 * 24 * 7)))
        fin_ventana = inicio_ventana + pd.Timedelta(minutes=int(ventana[1:, 11].sum()))

        filas.append({
            "device_pseudo_id": device_pseudo_id,
            "session_id": int(i // n_dispositivos),
            "window_id": 0,
            "secuencia": ventana.tolist(),
            "inicio_ventana": inicio_ventana,
            "fin_ventana": fin_ventana,
        })
    return pd.DataFrame(filas)


def main():
    parser = argparse.ArgumentParser(description="Genera ventanas sintéticas con el formato de entrada.")
    parser.add_argument("--salida", required=True, help="Carpeta donde se crean train/, val/ y test/.")
    parser.add_argument("--n_train", type=int, default=3000, help="Ventanas de entrenamiento.")
    parser.add_argument("--n_val", type=int, default=600, help="Ventanas de validación.")
    parser.add_argument("--n_test", type=int, default=600, help="Ventanas de prueba.")
    parser.add_argument("--semilla", type=int, default=0, help="Semilla para obtener siempre los mismos datos.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.semilla)
    conjuntos = {
        "train": (args.n_train, pd.Timestamp("2017-10-27")),
        "val": (args.n_val, pd.Timestamp("2017-11-10")),
        "test": (args.n_test, pd.Timestamp("2017-11-20")),
    }
    for nombre, (n, inicio) in conjuntos.items():
        df = generar_conjunto(n, inicio, rng)
        carpeta = os.path.join(args.salida, nombre)
        os.makedirs(carpeta, exist_ok=True)
        df.to_parquet(os.path.join(carpeta, "ventanas.parquet"), index=False)
        print(f"{nombre}: {len(df)} ventanas -> {carpeta}")


if __name__ == "__main__":
    main()
