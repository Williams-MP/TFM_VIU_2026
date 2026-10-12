"""
Entrenamiento de TranAD y TranAD+ (apartado 4.3.1 de la memoria).

Las dos arquitecturas se entrenan igual: AdamW (tasa 1e-3, weight decay
1e-5), lotes de 128 ventanas, 15 épocas, tasa de aprendizaje reducida a la
mitad cada 5 épocas y sin detención temprana. La pérdida combina los
errores de los dos decodificadores con un peso que cambia con la época n:
L = (1/n)·e1 + (1 - 1/n)·e2 (ecuación 4). No incluye el término adversario
del artículo original.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from checkpoints import cargar_checkpoint_si_existe, guardar_checkpoint
from arquitectura_tranad import TranADNet


def fijar_semilla(semilla):
    """
    Fija las semillas de random, NumPy y PyTorch (CPU y GPU) y pone PyTorch
    en modo determinista, para que dos entrenamientos con la misma semilla
    den el mismo modelo.

    Parámetros:
        semilla (int): valor de la semilla (en la memoria: 0, 7, 42, 123 y 777).
    """
    # Necesario para que las multiplicaciones de matrices en GPU (cuBLAS) sean
    # deterministas. Solo surte efecto si se define antes de la primera
    # operación en GPU de la sesión: conviene reiniciar el entorno de Colab
    # antes de lanzar los entrenamientos.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)           # pesos iniciales y azar en CPU
    torch.cuda.manual_seed_all(semilla)  # azar en GPU

    # cuDNN puede elegir algoritmos distintos en cada ejecución; se fuerza a
    # usar siempre el mismo, a cambio de algo de velocidad.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Algoritmos deterministas. warn_only=True avisa, en lugar de detener el
    # entrenamiento, si alguna operación no tiene versión determinista.
    torch.use_deterministic_algorithms(True, warn_only=True)


def entrenar_modelo(ruta_checkpoints, dim_feedforward, X_entrenamiento, n_feats, n_window,
                    dispositivo, n_epocas=15, tamano_batch=128, tasa_aprendizaje=1e-3,
                    semilla=None):
    """
    Entrena un modelo TranADNet, o reanuda su entrenamiento desde el último
    checkpoint, y guarda un checkpoint al final de cada época.

    Parámetros:
        ruta_checkpoints (str): carpeta de checkpoints del modelo.
        dim_feedforward (int): 16 para TranAD, 64 para TranAD+.
        X_entrenamiento (np.ndarray): ventanas escaladas, forma (n, 12, 13).
        n_feats, n_window (int): dimensiones del modelo (13 y 12).
        dispositivo: 'cuda' o 'cpu'.
        n_epocas, tamano_batch, tasa_aprendizaje: hiperparámetros.
        semilla (int o None): con una semilla, el entrenamiento es
            reproducible; con None, no.

    Devuelve:
        El modelo entrenado, en modo evaluación.
    """
    os.makedirs(ruta_checkpoints, exist_ok=True)

    # La semilla se fija antes de crear el modelo, porque los pesos iniciales
    # se generan en su constructor.
    if semilla is not None:
        fijar_semilla(semilla)

    modelo = TranADNet(feats=n_feats, n_window=n_window, dim_feedforward=dim_feedforward).to(dispositivo)
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=tasa_aprendizaje, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizador, step_size=5, gamma=0.5)
    criterio = nn.MSELoss(reduction="none")

    tensor_train = torch.tensor(X_entrenamiento, dtype=torch.float32)

    # Generador propio para el barajado: el orden de las ventanas en cada
    # época depende solo de la semilla y no de otras operaciones aleatorias
    # ejecutadas antes en el notebook.
    generador = None
    if semilla is not None:
        generador = torch.Generator()
        generador.manual_seed(semilla)

    dataloader_train = DataLoader(TensorDataset(tensor_train), batch_size=tamano_batch, shuffle=True,
                                  pin_memory=True, generator=generador)

    # Si hay un checkpoint, se continúa desde él con el mismo estado aleatorio.
    epoca_inicial = cargar_checkpoint_si_existe(ruta_checkpoints, modelo, optimizador, scheduler, dispositivo,
                                                generador=generador)

    for epoca in range(epoca_inicial, n_epocas):
        modelo.train()
        n = epoca + 1
        perdidas_epoca = []
        for (batch_x,) in dataloader_train:
            # El modelo espera la forma (reportes, lote, variables).
            ventana = batch_x.permute(1, 0, 2).to(dispositivo)
            x1, x2 = modelo(ventana, ventana)
            # Ecuación (4): al principio pesa más el primer decodificador.
            perdida = (1 / n) * criterio(x1, ventana) + (1 - 1 / n) * criterio(x2, ventana)
            perdida_media = torch.mean(perdida)
            optimizador.zero_grad()
            perdida_media.backward()
            optimizador.step()
            perdidas_epoca.append(perdida_media.item())
        scheduler.step()
        perdida_media_epoca = float(np.mean(perdidas_epoca))
        print(f"[{os.path.basename(ruta_checkpoints)} | semilla={semilla}] "
              f"Época {epoca + 1}/{n_epocas}, pérdida media = {perdida_media_epoca:.6f}")
        guardar_checkpoint(ruta_checkpoints, epoca + 1, modelo, optimizador, scheduler, perdida_media_epoca,
                           generador=generador)

    modelo.eval()
    return modelo
