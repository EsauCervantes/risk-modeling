"""Masked autoencoder utilities for tabular borrower features.

The network is self-supervised: training targets are created by masking input
features and asking the model to reconstruct the original feature values.
Default labels are not used here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class AutoencoderTrainingConfig:
    latent_dim: int = 8
    mask_fraction: float = 0.25
    learning_rate: float = 1e-3
    batch_size: int = 512
    max_epochs: int = 50
    patience: int = 7
    internal_val_fraction: float = 0.15
    seed: int = 42
    device: str | None = None


class MaskedFeatureAutoencoder(nn.Module):
    """Small masked autoencoder for standardized tabular features."""

    def __init__(self, input_dim: int, latent_dim: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
        )

    def encode(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        encoder_input = torch.cat([x, mask], dim=1)
        return self.encoder(encoder_input)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        latent = self.encode(x, mask)
        return self.decoder(latent)

    def transform(
        self,
        X,
        *,
        device: str | torch.device | None = None,
        batch_size: int = 4096,
    ) -> np.ndarray:
        """Return latent representations for complete unmasked rows."""
        target_device = torch.device(device) if device is not None else next(self.parameters()).device
        values = _as_float32_array(X)
        dataset = TensorDataset(torch.from_numpy(values))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        embeddings = []
        self.eval()
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(target_device)
                mask = torch.zeros_like(batch)
                latent = self.encode(batch, mask)
                embeddings.append(latent.cpu().numpy())

        return np.vstack(embeddings)


def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str | None = None) -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _as_float32_array(X) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        values = X.to_numpy()
    else:
        values = np.asarray(X)
    return values.astype(np.float32, copy=False)


def create_feature_mask(
    x: torch.Tensor,
    mask_fraction: float,
) -> torch.Tensor:
    """Create a binary feature mask with at least one masked value per row."""
    if not 0 < mask_fraction < 1:
        raise ValueError("mask_fraction must be between 0 and 1.")

    mask = (torch.rand(x.shape, device=x.device) < mask_fraction).float()
    rows_without_mask = mask.sum(dim=1) == 0
    if rows_without_mask.any():
        random_columns = torch.randint(
            low=0,
            high=x.shape[1],
            size=(int(rows_without_mask.sum().item()),),
            device=x.device,
        )
        row_indices = torch.where(rows_without_mask)[0]
        mask[row_indices, random_columns] = 1.0
    return mask


def masked_mse_loss(
    reconstruction: torch.Tensor,
    original: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    squared_error = (reconstruction - original).pow(2) * mask
    return squared_error.sum() / mask.sum().clamp_min(1.0)


def _split_internal_train_val(
    X: np.ndarray,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < val_fraction < 1:
        raise ValueError("internal_val_fraction must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    val_size = max(1, int(round(len(X) * val_fraction)))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    if len(train_indices) == 0:
        raise ValueError("Internal train split is empty. Provide more rows.")
    return X[train_indices], X[val_indices]


def train_masked_autoencoder(
    X_train,
    config: AutoencoderTrainingConfig | None = None,
) -> tuple[MaskedFeatureAutoencoder, pd.DataFrame, torch.device]:
    """Train the masked autoencoder using only borrower feature values."""
    config = config or AutoencoderTrainingConfig()
    set_random_seed(config.seed)
    device = resolve_device(config.device)

    values = _as_float32_array(X_train)
    internal_train, internal_val = _split_internal_train_val(
        values,
        config.internal_val_fraction,
        config.seed,
    )

    train_dataset = TensorDataset(torch.from_numpy(internal_train))
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )

    val_tensor = torch.from_numpy(internal_val).to(device)
    val_mask = create_feature_mask(val_tensor, config.mask_fraction)
    val_corrupted = val_tensor * (1.0 - val_mask)

    model = MaskedFeatureAutoencoder(
        input_dim=values.shape[1],
        latent_dim=config.latent_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    best_state = None
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for (batch,) in train_loader:
            batch = batch.to(device)
            mask = create_feature_mask(batch, config.mask_fraction)
            corrupted = batch * (1.0 - mask)

            optimizer.zero_grad()
            reconstruction = model(corrupted, mask)
            loss = masked_mse_loss(reconstruction, batch, mask)
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.detach().cpu())
            train_batches += 1

        model.eval()
        with torch.no_grad():
            val_reconstruction = model(val_corrupted, val_mask)
            val_loss = float(
                masked_mse_loss(val_reconstruction, val_tensor, val_mask)
                .detach()
                .cpu()
            )

        train_loss = train_loss_sum / max(train_batches, 1)
        history.append(
            {
                "epoch": epoch,
                "train_reconstruction_loss": train_loss,
                "internal_val_reconstruction_loss": val_loss,
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)

    return model, pd.DataFrame(history), device
