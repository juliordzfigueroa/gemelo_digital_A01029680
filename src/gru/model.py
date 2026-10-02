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

Two interchangeable recurrent implementations live here:

  "cho"     GRUCell, which writes out equations 1 to 4 of section 4.2.7.2
            literally, in the order the research document states them. This is
            the default, so that the equations published in the thesis are the
            equations that produce the reported results.
  "pytorch" nn.GRU, which is the cuDNN variant: roughly 15x faster but not the
            same formulation (see GRUCell for the two differences).

Both reach equivalent accuracy. compare_implementations.py trains them side
by side and prints the comparison.

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-10-01 - Added GRUCell, the literal implementation of equations 1-4.

Reference:
    Sections 4.2.6 and 4.2.7.2 of the research document.
    Cho, K., van Merrienboer, B., Gulcehre, C., Bahdanau, D., Bougares, F.,
    Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN
    encoder-decoder for statistical machine translation. arXiv:1406.1078.
"""

from __future__ import annotations

import math

import torch
from torch import nn

import config as cfg


class GRUCell(nn.Module):
    """One GRU cell written exactly as equations 1 to 4 of section 4.2.7.2.

    The four equations, in the document's own notation:

        (1)  r_t  = sigma(W_r [h_{t-1}, x_t] + b_r)
        (2)  z_t  = sigma(W_z [h_{t-1}, x_t] + b_z)
        (3)  h~_t = tanh(W_h [r_t (*) h_{t-1}, x_t] + b_h)
        (4)  h_t  = (1 - z_t) (*) h_{t-1} + z_t (*) h~_t

    where (*) is the Hadamard product and [a, b] is concatenation.

    This differs from PyTorch's nn.GRU in two ways, which is the reason this
    class exists:

      1. Equation 3 multiplies r_t by h_{t-1} and only then applies W_h.
         nn.GRU computes r_t (*) (W_hn h_{t-1} + b_hn), applying the linear
         transform first and the reset gate afterwards. Those are genuinely
         different operations.
      2. Equation 4 treats z_t as "how much of the new state to take". nn.GRU
         treats its z_t as "how much of the old state to keep". That one is
         only a sign convention.

    The cost of the literal version is speed: a Python loop over timesteps
    cannot use the fused cuDNN kernel, which makes it about 15x slower. On this
    dataset that is roughly 8 minutes of training instead of 33 seconds, which
    is a worthwhile trade for having the documented equations be the ones that
    actually run.

    Reference:
        Section 4.2.7.2 of the research document.
        Cho et al. (2014). arXiv:1406.1078.
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        """Create the three gate transforms.

        Args:
            input_dim: Width of x_t.
            hidden_dim: Width of h_t, which is the number of hidden units.
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # Each gate takes the concatenation [h_{t-1}, x_t], so its input width
        # is the sum of both. One nn.Linear per gate carries both the weight
        # matrix W and the bias vector b of the equation.
        concat_dim = hidden_dim + input_dim
        self.W_r = nn.Linear(concat_dim, hidden_dim)  # equation 1
        self.W_z = nn.Linear(concat_dim, hidden_dim)  # equation 2
        self.W_h = nn.Linear(concat_dim, hidden_dim)  # equation 3

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights the same way nn.GRU does.

        Uniform in [-1/sqrt(hidden), 1/sqrt(hidden)]. Matching PyTorch's scheme
        is what makes the comparison between the two implementations fair: a
        difference in results should come from the formulation, not from a
        different starting point.
        """
        limit = 1.0 / math.sqrt(self.hidden_dim)
        for parameter in self.parameters():
            nn.init.uniform_(parameter, -limit, limit)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """Advance the hidden state by one timestep.

        Args:
            x_t: Input at time t, shaped (batch, input_dim).
            h_prev: Hidden state h_{t-1}, shaped (batch, hidden_dim).

        Returns:
            The new hidden state h_t, shaped (batch, hidden_dim).
        """
        concatenated = torch.cat([h_prev, x_t], dim=-1)

        r = torch.sigmoid(self.W_r(concatenated))  # (1) what of the past to drop
        z = torch.sigmoid(self.W_z(concatenated))  # (2) how much to replace

        # (3) the reset gate filters h_{t-1} BEFORE the linear transform, which
        # is the difference from nn.GRU described in the class docstring.
        candidate = torch.tanh(self.W_h(torch.cat([r * h_prev, x_t], dim=-1)))

        return (1 - z) * h_prev + z * candidate  # (4) blend old and new


class GRULayer(nn.Module):
    """Run a GRUCell across a whole sequence.

    Mirrors the call signature of nn.GRU so the two implementations are
    interchangeable in EnvironmentalGRU without touching anything downstream.
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        """Create the layer.

        Args:
            input_dim: Width of each input step.
            hidden_dim: Number of hidden units.
        """
        super().__init__()
        self.cell = GRUCell(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Process every timestep in order.

        Args:
            x: Input sequence, shaped (batch, pasos, input_dim).

        Returns:
            A tuple of (all hidden states shaped (batch, pasos, hidden_dim),
            final hidden state shaped (1, batch, hidden_dim)). The second
            element carries the extra leading axis so the shape matches what
            nn.GRU returns.
        """
        batch, steps, _ = x.shape
        h = x.new_zeros(batch, self.hidden_dim)

        states = []
        for t in range(steps):
            h = self.cell(x[:, t, :], h)
            states.append(h)

        return torch.stack(states, dim=1), h.unsqueeze(0)


class EnvironmentalGRU(nn.Module):
    """Two-layer GRU that forecasts several environmental variables at once."""

    def __init__(
        self,
        n_variables: int,
        n_continuous_features: int,
        n_spaces: int,
        units: list[int] | None = None,
        dropout: float = cfg.DROPOUT,
        embedding_dim: int = cfg.SPACE_EMBEDDING_DIM,
        output_steps: int = cfg.OUTPUT_STEPS,
        implementation: str = cfg.GRU_IMPLEMENTATION,
    ) -> None:
        """Build the network.

        Args:
            n_variables: How many variables are predicted (ACTIVE_VARIABLES).
            n_continuous_features: How many continuous input columns there are.
                This can differ from n_variables because the time features are
                fed as input but never predicted.
            n_spaces: Number of distinct spaces, sizing the embedding table.
            unidades: Hidden units per layer. Exactly two values.
            dropout: Dropout probability applied between the two GRU layers.
            embedding_dim: Width of the space-id embedding.
            output_steps: How many future steps the head emits per call.
            implementacion: "cho" for the literal equations of section 4.2.7.2,
                "pytorch" for the faster cuDNN variant.

        Raises:
            ValueError: If unidades does not hold exactly two values, or if
                implementacion is not one of the two accepted names.
        """
        super().__init__()
        units = units or cfg.HIDDEN_UNITS
        if len(units) != 2:
            raise ValueError(f"expected 2 layers, got {len(units)}")
        if implementation not in ("cho", "pytorch"):
            raise ValueError(
                f"implementacion must be 'cho' or 'pytorch', got {implementation!r}"
            )

        self.n_variables = n_variables
        self.output_steps = output_steps
        self.implementation = implementation

        # The space id is categorical, not ordinal: room 3302 is not "more"
        # than room 3301, so it cannot enter as a number. The embedding learns
        # a representation per space and is what allows a single global model
        # for the whole building.
        #
        # Reference: Section 4.2.5 of the research document.
        self.space_embedding = nn.Embedding(n_spaces, embedding_dim)

        input_dim = n_continuous_features + embedding_dim

        if implementation == "cho":
            self.gru1 = GRULayer(input_dim, units[0])
            self.gru2 = GRULayer(units[0], units[1])
        else:
            # Two separate nn.GRU modules instead of one with num_layers=2:
            # PyTorch forces every layer inside a single nn.GRU to share
            # hidden_size, and the specification asks for [128, 64].
            self.gru1 = nn.GRU(input_dim, units[0], batch_first=True)
            self.gru2 = nn.GRU(units[0], units[1], batch_first=True)

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
        self.output = nn.Linear(units[1], output_steps * n_variables)

    def forward(self, features: torch.Tensor, space: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            rasgos: Normalized inputs, shaped
                (batch, INPUT_STEPS, n_continuous_features).
            espacio: Integer space indices, shaped (batch,).

        Returns:
            Predictions in normalized space, shaped
            (batch, output_steps, n_variables).
        """
        batch, steps, _ = features.shape

        # The space embedding is constant along the window, so it is repeated
        # at every step and concatenated to the sensor features.
        emb = self.space_embedding(space)  # (batch, embedding_dim)
        emb = emb.unsqueeze(1).expand(batch, steps, -1)
        x = torch.cat([features, emb], dim=-1)

        x, _ = self.gru1(x)
        x = self.dropout(x)
        x, _ = self.gru2(x)

        # Only the final step matters: its hidden state summarizes the window.
        last_step = x[:, -1, :]
        flat = self.output(last_step)
        return flat.view(batch, self.output_steps, self.n_variables)

    def n_parameters(self) -> int:
        """Count trainable parameters.

        Returns:
            Number of parameters that require gradients.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build(
    n_spaces: int,
    n_continuous_features: int,
    implementation: str = cfg.GRU_IMPLEMENTATION,
) -> EnvironmentalGRU:
    """Instantiate the network with the project configuration.

    Args:
        n_spaces: Number of distinct spaces in the dataset.
        n_continuous_features: Number of continuous input columns.
        implementacion: "cho" or "pytorch".

    Returns:
        An untrained EnvironmentalGRU sized from config.py.
    """
    return EnvironmentalGRU(
        n_variables=len(cfg.ACTIVE_VARIABLES),
        n_continuous_features=n_continuous_features,
        n_spaces=n_spaces,
        implementation=implementation,
    )
