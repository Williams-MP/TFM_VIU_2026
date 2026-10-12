"""
Guardado y carga de checkpoints del entrenamiento.

Cada checkpoint guarda, además del modelo, el optimizador y el planificador
de la tasa de aprendizaje, el estado de todos los generadores aleatorios.
Así un entrenamiento interrumpido y reanudado da exactamente el mismo
modelo que uno sin cortes (apartado 4.3.1 de la memoria).

Los checkpoints no contienen datos: solo pesos, estados del optimizador,
la pérdida de cada época y los estados aleatorios.
"""

import os
import random
import numpy as np
import torch


def _capturar_estado_aleatorio(generador=None):
    """
    Devuelve el estado actual de todos los generadores aleatorios que
    intervienen en el entrenamiento.

    Parámetros:
        generador (torch.Generator o None): generador propio del DataLoader,
            que decide el orden de las ventanas en cada época.
    """
    return {
        "random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        # Azar de la GPU (por ejemplo, el dropout). Solo si hay GPU.
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "generador_dataloader": generador.get_state() if generador is not None else None,
    }


def _restaurar_estado_aleatorio(estado, generador=None):
    """
    Devuelve cada generador aleatorio al estado guardado en un checkpoint.

    torch.load con map_location='cuda' mueve todos los tensores a la GPU,
    pero PyTorch exige que los estados aleatorios estén en la CPU; por eso
    se aplica .cpu() antes de restaurarlos.
    """
    random.setstate(estado["random"])
    np.random.set_state(estado["numpy"])
    torch.set_rng_state(estado["torch_cpu"].cpu())
    if estado.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in estado["torch_cuda"]])
    if generador is not None and estado.get("generador_dataloader") is not None:
        generador.set_state(estado["generador_dataloader"].cpu())


def buscar_ultimo_checkpoint(ruta):
    """
    Busca en una carpeta el checkpoint de la época más avanzada.

    Parámetros:
        ruta (str): carpeta de checkpoints de un modelo.

    Devuelve:
        (ruta_del_fichero, numero_de_epoca), o None si no hay ninguno.
    """
    if not os.path.isdir(ruta):
        return None
    # Los ficheros temporales (.pt.tmp) quedan fuera: un guardado
    # interrumpido nunca se toma como checkpoint válido.
    archivos = [f for f in os.listdir(ruta) if f.startswith("checkpoint_epoca_") and f.endswith(".pt")]
    if not archivos:
        return None
    numeros_epoca = [int(f.replace("checkpoint_epoca_", "").replace(".pt", "")) for f in archivos]
    epoca_maxima = max(numeros_epoca)
    return f"{ruta}/checkpoint_epoca_{epoca_maxima}.pt", epoca_maxima


def cargar_checkpoint_si_existe(ruta, modelo, optimizador, scheduler, dispositivo, generador=None):
    """
    Si hay un checkpoint en la carpeta, carga su estado en el modelo, el
    optimizador, el planificador y los generadores aleatorios (los modifica
    en el sitio). Si no hay ninguno, no hace nada.

    Parámetros:
        ruta (str): carpeta de checkpoints del modelo.
        modelo, optimizador, scheduler: objetos ya creados.
        dispositivo: 'cuda' o 'cpu'.
        generador (torch.Generator o None): generador del DataLoader.

    Devuelve:
        int: época desde la que continuar (0 si no había checkpoint).
    """
    resultado = buscar_ultimo_checkpoint(ruta)
    if resultado is None:
        print("No hay checkpoints previos, entrenamiento desde cero.")
        return 0
    ruta_checkpoint, epoca_guardada = resultado

    # weights_only=False: desde PyTorch 2.6, torch.load rechaza por defecto el
    # estado aleatorio de NumPy. Es seguro porque los checkpoints son propios.
    checkpoint = torch.load(ruta_checkpoint, map_location=dispositivo, weights_only=False)

    modelo.load_state_dict(checkpoint["modelo"])
    optimizador.load_state_dict(checkpoint["optimizador"])
    scheduler.load_state_dict(checkpoint["scheduler"])

    if "estado_aleatorio" in checkpoint:
        _restaurar_estado_aleatorio(checkpoint["estado_aleatorio"], generador)
        print(f"Reanudando desde el checkpoint de la época {epoca_guardada} (estado aleatorio restaurado)")
    else:
        print(f"Reanudando desde el checkpoint de la época {epoca_guardada} (sin estado aleatorio guardado)")
        if generador is not None and epoca_guardada > 0:
            print("Aviso: sin estado aleatorio, este entrenamiento no será exactamente reproducible.")

    return epoca_guardada


def guardar_checkpoint(ruta, epoca, modelo, optimizador, scheduler, perdida, generador=None):
    """
    Guarda el estado del entrenamiento al terminar una época.

    Parámetros:
        ruta (str): carpeta donde guardar el checkpoint (se crea si no existe).
        epoca (int): época que se acaba de completar.
        modelo, optimizador, scheduler: objetos con el estado a guardar.
        perdida (float): pérdida media de la época.
        generador (torch.Generator o None): generador del DataLoader.
    """
    os.makedirs(ruta, exist_ok=True)
    ruta_final = f"{ruta}/checkpoint_epoca_{epoca}.pt"
    ruta_temporal = ruta_final + ".tmp"

    torch.save({
        "modelo": modelo.state_dict(),
        "optimizador": optimizador.state_dict(),
        "scheduler": scheduler.state_dict(),
        "perdida": perdida,
        "estado_aleatorio": _capturar_estado_aleatorio(generador),
    }, ruta_temporal)

    # Se escribe primero en un fichero temporal y después se renombra. El
    # renombrado es instantáneo, así que si Colab se corta durante el guardado
    # solo queda un .tmp incompleto, que buscar_ultimo_checkpoint() ignora.
    os.replace(ruta_temporal, ruta_final)
