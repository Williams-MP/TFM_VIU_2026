"""
Arquitectura de TranAD (Tuli et al., 2022), usada también para TranAD+
(Yermakov et al., 2025).

Las dos arquitecturas comparten este código. Solo se diferencian en la
dimensión interna de la capa feed-forward (dim_feedforward: 16 en TranAD,
64 en TranAD+) y en los pesos con que se combinan los dos errores de
reconstrucción al calcular la puntuación de anomalía (evaluacion.py).
Ver la Tabla 8 de la memoria.
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Codificación posicional que se suma a cada reporte de la ventana."""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model))
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x, pos=0):
        x = x + self.pe[pos:pos + x.size(0), :]
        return self.dropout(x)


class TransformerEncoderLayerTranAD(nn.Module):
    """Capa del codificador: autoatención y capa feed-forward con conexiones residuales."""

    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(True)

    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return src


class TransformerDecoderLayerTranAD(nn.Module):
    """Capa del decodificador: autoatención, atención sobre el codificador y feed-forward."""

    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(True)

    def forward(self, tgt, memory):
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class TranADNet(nn.Module):
    """
    Modelo completo: un codificador y dos decodificadores.

    Parámetros:
        feats (int): número de variables por reporte (13).
        n_window (int): número de reportes por ventana (12).
        dim_feedforward (int): dimensión interna de la capa feed-forward
            (16 en TranAD, 64 en TranAD+).

    La entrada tiene forma (n_window, lote, feats). La salida son las dos
    reconstrucciones de la ventana, x1 y x2, con la misma forma.
    """

    def __init__(self, feats, n_window, dim_feedforward=16):
        super().__init__()
        self.n_feats = feats
        self.n_window = n_window
        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, n_window)
        self.transformer_encoder = TransformerEncoderLayerTranAD(2 * feats, feats, dim_feedforward=dim_feedforward, dropout=0.1)
        self.transformer_decoder1 = TransformerDecoderLayerTranAD(2 * feats, feats, dim_feedforward=dim_feedforward, dropout=0.1)
        self.transformer_decoder2 = TransformerDecoderLayerTranAD(2 * feats, feats, dim_feedforward=dim_feedforward, dropout=0.1)
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

    def encode(self, src, c, tgt):
        # La ventana se concatena con la puntuación de foco c (ceros en la
        # primera pasada, error de la primera reconstrucción en la segunda).
        src = torch.cat((src, c), dim=2)
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)
        return tgt, memory

    def forward(self, src, tgt):
        # Primera pasada: reconstrucción sin información de foco.
        c = torch.zeros_like(src)
        tgt1, memory1 = self.encode(src, c, tgt)
        x1 = self.fcn(self.transformer_decoder1(tgt1, memory1))
        # Segunda pasada: el error de la primera reconstrucción se usa como foco.
        c = (x1 - src) ** 2
        tgt2, memory2 = self.encode(src, c, tgt)
        x2 = self.fcn(self.transformer_decoder2(tgt2, memory2))
        return x1, x2
