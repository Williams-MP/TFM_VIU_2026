"""
Puntuación de anomalía y umbral POT (apartados 3.2.1 y 4.3.2 de la memoria).
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import genpareto


def calcular_anomaly_scores(modelo, X, dispositivo, alfa=0.0, beta=1.0, tamano_batch=256):
    """
    Calcula la puntuación de anomalía de cada ventana con un modelo ya
    entrenado (ecuación 1): S = alfa·e1 + beta·e2, donde e1 y e2 son el error
    cuadrático medio de las dos reconstrucciones de la ventana.

    Pesos usados en la memoria (Tabla 8): TranAD, alfa=0 y beta=1;
    TranAD+, alfa=beta=0,5.

    Parámetros:
        modelo: TranADNet entrenado, en modo evaluación.
        X (np.ndarray): ventanas escaladas, forma (n, 12, 13).
        dispositivo: 'cuda' o 'cpu'.
        alfa, beta (float): pesos de cada error.
        tamano_batch (int): ventanas por lote (no afecta al resultado).

    Devuelve:
        np.ndarray: una puntuación por ventana, forma (n,).
    """
    tensor_X = torch.tensor(X, dtype=torch.float32)
    dataloader = DataLoader(TensorDataset(tensor_X), batch_size=tamano_batch, shuffle=False)
    scores = []
    with torch.no_grad():
        for (batch_x,) in dataloader:
            ventana = batch_x.permute(1, 0, 2).to(dispositivo)
            x1, x2 = modelo(ventana, ventana)
            error1 = (x1 - ventana) ** 2
            error2 = (x2 - ventana) ** 2
            error_combinado = alfa * error1 + beta * error2
            # Media sobre los 12 reportes y las 13 variables de cada ventana.
            scores.append(error_combinado.mean(dim=(0, 2)).cpu().numpy())
    return np.concatenate(scores)


def umbral_pot(scores_referencia, q=1e-3, nivel_inicial=0.98):
    """
    Umbral de alerta por Peaks-Over-Threshold (Siffer et al., 2017),
    ecuaciones (5) y (6) de la memoria.

    1. Umbral inicial t: percentil (nivel_inicial·100) de las puntuaciones.
    2. Los excesos sobre t se ajustan a una distribución de Pareto
       generalizada (parámetros gamma y beta).
    3. Umbral final: el valor que, según esa distribución, solo superaría
       una fracción q de las ventanas.

    Parámetros:
        scores_referencia (np.ndarray): puntuaciones de validación.
        q (float): fracción objetivo de ventanas por encima del umbral.
        nivel_inicial (float): cuantil del umbral inicial.

    Devuelve:
        float: el umbral.
    """
    n = len(scores_referencia)
    t = np.percentile(scores_referencia, nivel_inicial * 100)
    excedencias = scores_referencia[scores_referencia > t] - t
    n_t = len(excedencias)
    gamma, _, beta = genpareto.fit(excedencias, floc=0)
    if gamma != 0:
        return t + (beta / gamma) * ((q * n / n_t) ** (-gamma) - 1)
    return t - beta * np.log(q * n / n_t)


def calcular_porcentaje_sobre_umbral(scores, umbral):
    """Porcentaje de ventanas cuya puntuación supera el umbral."""
    return (scores > umbral).mean() * 100
