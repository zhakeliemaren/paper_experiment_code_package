from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


class HistoryContextEncoder:
    """Compress a sliding history window into a context vector.

    The default backend is an MLP autoencoder. It flattens each fixed-length
    history window, learns to reconstruct it, and uses the bottleneck hidden
    layer as the environment context vector.
    """

    def __init__(
        self,
        context_dim: int = 4,
        mode: str = "mlp",
        random_state: int = 7,
        hidden_layer_sizes: tuple[int, ...] = (32,),
        max_iter: int = 250,
    ):
        self.context_dim = int(context_dim)
        self.mode = mode
        self.random_state = int(random_state)
        self.hidden_layer_sizes = tuple(int(v) for v in hidden_layer_sizes)
        self.max_iter = int(max_iter)
        self.scaler: StandardScaler | None = None
        self.pca: PCA | None = None
        self.mlp: MLPRegressor | None = None
        self.bottleneck_layer_index: int | None = None
        self.fitted = False

    def fit(self, history: np.ndarray) -> "HistoryContextEncoder":
        flat = self._flatten(history)
        if self.mode == "stats":
            self.fitted = True
            return self

        self.scaler = StandardScaler()
        scaled = self.scaler.fit_transform(flat)
        if self.mode == "mlp":
            encoder_hidden = self._positive_layers(self.hidden_layer_sizes)
            decoder_hidden = tuple(reversed(encoder_hidden))
            layer_sizes = encoder_hidden + (self.context_dim,) + decoder_hidden
            self.mlp = MLPRegressor(
                hidden_layer_sizes=layer_sizes,
                activation="tanh",
                solver="adam",
                alpha=1e-4,
                batch_size=min(128, max(1, scaled.shape[0])),
                learning_rate_init=1e-3,
                max_iter=self.max_iter,
                random_state=self.random_state,
                n_iter_no_change=20,
                tol=1e-4,
            )
            self.mlp.fit(scaled, scaled)
            self.bottleneck_layer_index = len(encoder_hidden)
            self.fitted = True
            return self

        if self.mode != "pca":
            raise ValueError(f"Unknown encoder mode: {self.mode}")

        n_components = min(self.context_dim, scaled.shape[1], scaled.shape[0])
        self.pca = PCA(n_components=n_components, random_state=self.random_state)
        self.pca.fit(scaled)
        self.fitted = True
        return self

    def transform(self, history: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("HistoryContextEncoder.fit must be called first")
        if self.mode == "stats":
            return self._stats_context(history)
        if self.mode == "mlp":
            return self._mlp_context(history)

        assert self.scaler is not None and self.pca is not None
        flat = self._flatten(history)
        c = self.pca.transform(self.scaler.transform(flat))
        if c.shape[1] < self.context_dim:
            pad = np.zeros((c.shape[0], self.context_dim - c.shape[1]), dtype=float)
            c = np.concatenate([c, pad], axis=1)
        return c

    def fit_transform(self, history: np.ndarray) -> np.ndarray:
        return self.fit(history).transform(history)

    @staticmethod
    def _flatten(history: np.ndarray) -> np.ndarray:
        arr = np.asarray(history, dtype=float)
        return arr.reshape(arr.shape[0], -1)

    @staticmethod
    def _positive_layers(hidden_layer_sizes: tuple[int, ...]) -> tuple[int, ...]:
        layers = tuple(v for v in hidden_layer_sizes if v > 0)
        return layers or (32,)

    @staticmethod
    def _activation(x: np.ndarray, name: str) -> np.ndarray:
        if name == "identity":
            return x
        if name == "logistic":
            return 1.0 / (1.0 + np.exp(-x))
        if name == "tanh":
            return np.tanh(x)
        if name == "relu":
            return np.maximum(x, 0.0)
        raise ValueError(f"Unsupported MLP activation: {name}")

    def _mlp_context(self, history: np.ndarray) -> np.ndarray:
        assert self.scaler is not None and self.mlp is not None and self.bottleneck_layer_index is not None
        h = self.scaler.transform(self._flatten(history))
        for layer_idx in range(self.bottleneck_layer_index + 1):
            h = h @ self.mlp.coefs_[layer_idx] + self.mlp.intercepts_[layer_idx]
            h = self._activation(h, self.mlp.activation)
        if h.shape[1] >= self.context_dim:
            return h[:, : self.context_dim]
        pad = np.zeros((h.shape[0], self.context_dim - h.shape[1]), dtype=float)
        return np.concatenate([h, pad], axis=1)

    def _stats_context(self, history: np.ndarray) -> np.ndarray:
        arr = np.asarray(history, dtype=float)
        mean = arr.mean(axis=1)
        std = arr.std(axis=1)
        delta = arr[:, -1, :] - arr[:, 0, :]
        last = arr[:, -1, :]
        features = np.concatenate([mean, std, delta, last], axis=1)
        if features.shape[1] >= self.context_dim:
            return features[:, : self.context_dim]
        pad = np.zeros((features.shape[0], self.context_dim - features.shape[1]), dtype=float)
        return np.concatenate([features, pad], axis=1)
