"""Multivariate GRU architecture for indoor environmental forecasting.

Two stacked GRU layers of [128, 64] units with 0.2 dropout, predicting the next
6 steps (30 minutes) from the previous 12 (1 hour).

Tensor flow:

    input     (batch, 12, n_features)   12 steps = 1 hour of context
      GRU 1   (batch, 12, 128)          local patterns across variables
      dropout (batch, 12, 128)
      GRU 2   (batch, 12,  64)          higher-level abstractions
      last    (batch, 64)               hidden state of the final step
      linear  (batch, 6 * n_variables)
    output    (batch, 6, n_variables)   6 steps = 30 minutes

Note on the gate equations. Section 4.2.7.2 of the research document states the
original formulation of Cho et al. (2014):

    r_t = sigma(W_r [h_{t-1}, x_t] + b_r)
    z_t = sigma(W_z [h_{t-1}, x_t] + b_z)
    h~_t = tanh(W_h [r_t (*) h_{t-1}, x_t] + b_h)
    h_t = (1 - z_t) (*) h_{t-1} + z_t (*) h~_t

PyTorch's nn.GRU implements the cuDNN variant, which differs in two ways:

  1. The reset gate is applied AFTER the linear transform of the hidden state,
     not before it:  n_t = tanh(W_in x_t + b_in + r_t (*) (W_hn h_{t-1} + b_hn))
     The document applies r_t to h_{t-1} and then multiplies by W_h. These are
     genuinely different operations, though both are called GRU and perform
     comparably in practice.
  2. The update gate has the opposite sense: PyTorch computes
     h_t = (1 - z_t) (*) n_t + z_t (*) h_{t-1}, so its z_t is "how much of the
     old state to keep" while the document's z_t is "how much of the new state
     to take". This is only a sign convention; the network learns either way.

Both differences are worth a footnote in the thesis, since the equations as
written do not match the library that produces the results.

Reference:
    Section 4.2.6 and 4.2.7.2 of the research document.
    Cho, K., van Merrienboer, B., Gulcehre, C., Bahdanau, D., Bougares, F.,
    Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN
    encoder-decoder for statistical machine translation. arXiv:1406.1078.
"""

from __future__ import annotations

import torch
from torch import nn

import config as cfg


class GRUAmbiental(nn.Module):
    """Two-layer GRU that forecasts several environmental variables at once."""

    def __init__(
        self,
        n_variables: int,
        n_rasgos_continuos: int,
        n_espacios: int,
        unidades: list[int] | None = None,
        dropout: float = cfg.DROPOUT,
        dim_embedding: int = cfg.DIM_EMBEDDING_ESPACIO,
        pasos_salida: int = cfg.PASOS_SALIDA,
    ) -> None:
        """Build the network.

        Args:
            n_variables: How many variables are predicted (VARIABLES_ACTIVAS).
            n_rasgos_continuos: How many continuous input columns there are.
                This can differ from n_variables because the time features are
                fed as input but never predicted.
            n_espacios: Number of distinct spaces, sizing the embedding table.
            unidades: Hidden units per layer. Exactly two values.
            dropout: Dropout probability applied between the two GRU layers.
            dim_embedding: Width of the space-id embedding.
            pasos_salida: How many future steps the head emits per call.

        Raises:
            ValueError: If unidades does not hold exactly two values.
        """
        super().__init__()
        unidades = unidades or cfg.UNIDADES_OCULTAS
        if len(unidades) != 2:
            raise ValueError(f"expected 2 layers, got {len(unidades)}")

        self.n_variables = n_variables
        self.pasos_salida = pasos_salida

        # The space id is categorical, not ordinal: room 3302 is not "more"
        # than room 3301, so it cannot enter as a number. The embedding learns
        # a representation per space and is what allows a single global model
        # for the whole building.
        #
        # Reference: Section 4.2.5 of the research document.
        self.embedding_espacio = nn.Embedding(n_espacios, dim_embedding)

        dim_entrada = n_rasgos_continuos + dim_embedding

        # Two separate nn.GRU modules instead of one with num_layers=2: PyTorch
        # forces every layer inside a single nn.GRU to share hidden_size, and
        # the specification asks for [128, 64].
        self.gru1 = nn.GRU(dim_entrada, unidades[0], batch_first=True)
        self.gru2 = nn.GRU(unidades[0], unidades[1], batch_first=True)

        # Explicit dropout module. The `dropout` argument of nn.GRU only acts
        # between internal layers, so with separate modules it would never fire.
        self.dropout = nn.Dropout(dropout)

        # Output head: emits all 6 steps at once (direct multi-step forecast).
        # No activation function, because this is regression over arbitrary
        # continuous values, which is the standard pairing with MSE loss.
        #
        # Open issue: 'movimiento' is binary, and for a variable bounded in
        # [0, 1] the same table recommends a sigmoid. While it is predicted
        # alongside the others under MSE, its output is an uncalibrated
        # continuous number. The clean alternative is a separate head with
        # BCEWithLogitsLoss. See src/gru/README.md.
        #
        # Reference: Table 1, section 5.3 of the research document.
        self.salida = nn.Linear(unidades[1], pasos_salida * n_variables)

    def forward(self, rasgos: torch.Tensor, espacio: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            rasgos: Normalized inputs, shaped
                (batch, PASOS_ENTRADA, n_rasgos_continuos).
            espacio: Integer space indices, shaped (batch,).

        Returns:
            Predictions in normalized space, shaped
            (batch, pasos_salida, n_variables).
        """
        lote, pasos, _ = rasgos.shape

        # The space embedding is constant along the window, so it is repeated
        # at every step and concatenated to the sensor features.
        emb = self.embedding_espacio(espacio)  # (batch, dim_embedding)
        emb = emb.unsqueeze(1).expand(lote, pasos, -1)
        x = torch.cat([rasgos, emb], dim=-1)

        x, _ = self.gru1(x)
        x = self.dropout(x)
        x, _ = self.gru2(x)

        # Only the final step matters: its hidden state summarizes the window.
        ultimo = x[:, -1, :]
        plano = self.salida(ultimo)
        return plano.view(lote, self.pasos_salida, self.n_variables)

    def n_parametros(self) -> int:
        """Count trainable parameters.

        Returns:
            Number of parameters that require gradients.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def construir(n_espacios: int, n_rasgos_continuos: int) -> GRUAmbiental:
    """Instantiate the network with the project configuration.

    Args:
        n_espacios: Number of distinct spaces in the dataset.
        n_rasgos_continuos: Number of continuous input columns.

    Returns:
        An untrained GRUAmbiental sized from config.py.
    """
    return GRUAmbiental(
        n_variables=len(cfg.VARIABLES_ACTIVAS),
        n_rasgos_continuos=n_rasgos_continuos,
        n_espacios=n_espacios,
    )
