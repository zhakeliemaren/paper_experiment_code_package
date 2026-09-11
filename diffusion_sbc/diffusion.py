from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class DiffusionSchedule:
    timesteps: int = 64
    schedule_type: str = "cosine"
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    cosine_s: float = 0.008
    max_beta: float = 0.5
    terminal_alpha_bar_max: float = 1e-3

    def __post_init__(self) -> None:
        if int(self.timesteps) < 2:
            raise ValueError("DiffusionSchedule.timesteps must be at least 2")
        schedule_type = str(self.schedule_type).strip().lower().replace("-", "_")
        if schedule_type == "linear":
            self.betas = np.linspace(self.beta_start, self.beta_end, self.timesteps, dtype=float)
        elif schedule_type == "cosine":
            steps = np.arange(self.timesteps + 1, dtype=float)
            phase = ((steps / float(self.timesteps)) + float(self.cosine_s)) / (1.0 + float(self.cosine_s))
            cumulative = np.cos(phase * np.pi / 2.0) ** 2
            cumulative = cumulative / cumulative[0]
            self.betas = 1.0 - cumulative[1:] / cumulative[:-1]
            self.betas = np.clip(self.betas, 1e-8, float(self.max_beta))
        else:
            raise ValueError("schedule_type must be 'cosine' or 'linear'")
        if not np.all(np.isfinite(self.betas)) or np.any(self.betas <= 0.0) or np.any(self.betas >= 1.0):
            raise ValueError("All diffusion betas must be finite and strictly between 0 and 1")
        self.schedule_type = schedule_type
        self.alphas = 1.0 - self.betas
        self.alpha_bars = np.cumprod(self.alphas)
        if float(self.alpha_bars[-1]) > float(self.terminal_alpha_bar_max):
            raise ValueError(
                "The forward process does not reach the configured Gaussian prior: "
                f"alpha_bar_T={self.alpha_bars[-1]:.6g} exceeds "
                f"terminal_alpha_bar_max={self.terminal_alpha_bar_max:.6g}. "
                "Use the cosine schedule, increase diffusion_steps, or strengthen the linear beta schedule."
            )


class ConditionalResidualDiffusion:
    """Conditional DDPM over standardized one-step residuals."""

    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        context_dim: int,
        residual_dim: int,
        schedule: DiffusionSchedule,
        hidden_layer_sizes: tuple[int, ...] = (64, 64),
        max_iter: int = 150,
        batch_size: int = 256,
        learning_rate_init: float = 5e-4,
        early_stopping: bool = False,
        random_state: int = 7,
    ) -> None:
        self.state_dim = int(state_dim)
        self.control_dim = int(control_dim)
        self.context_dim = int(context_dim)
        self.residual_dim = int(residual_dim)
        self.schedule = schedule
        self.random_state = int(random_state)
        self.max_iter = int(max_iter)
        self.batch_size = int(batch_size)
        self.learning_rate_init = float(learning_rate_init)
        self.early_stopping = bool(early_stopping)
        self.model = Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "mlp",
                    MLPRegressor(
                        hidden_layer_sizes=hidden_layer_sizes,
                        activation="tanh",
                        solver="adam",
                        alpha=1e-4,
                        batch_size=self.batch_size,
                        learning_rate_init=self.learning_rate_init,
                        max_iter=max_iter,
                        early_stopping=self.early_stopping,
                        n_iter_no_change=15,
                        random_state=random_state,
                    ),
                ),
            ]
        )
        self.fitted = False
        self.residual_location: np.ndarray | None = None
        self.residual_scale: np.ndarray | None = None

    def fit(
        self,
        x: np.ndarray,
        u: np.ndarray,
        c: np.ndarray,
        residual: np.ndarray,
        copies: int = 2,
    ) -> dict[str, float]:
        rng = np.random.default_rng(self.random_state)
        residual_arr = np.asarray(residual, dtype=float)
        self.residual_location = residual_arr.mean(axis=0)
        self.residual_scale = np.maximum(residual_arr.std(axis=0), 1e-6)
        normalized_residual = (residual_arr - self.residual_location) / self.residual_scale
        features, target = self._build_training_data(x, u, c, normalized_residual, int(copies), rng)
        model_target = target[:, 0] if self.residual_dim == 1 else target
        self.model.fit(features, model_target)
        pred = np.asarray(self.model.predict(features), dtype=float).reshape(-1, self.residual_dim)
        loss = float(np.mean((pred - target) ** 2))
        self.fitted = True
        mlp: MLPRegressor = self.model.named_steps["mlp"]
        return {
            "noise_mse": loss,
            "n_iter": float(getattr(mlp, "n_iter_", 0)),
            "training_examples": float(features.shape[0]),
            "residual_scale_mean": float(np.mean(self.residual_scale)),
        }

    def predict_noise(
        self,
        x: np.ndarray,
        u: np.ndarray,
        c: np.ndarray,
        y_t: np.ndarray,
        t_index: int,
    ) -> np.ndarray:
        self._check_fitted()
        feature = self._feature(x, u, c, y_t, t_index)
        return np.asarray(self.model.predict(feature), dtype=float).reshape(-1, self.residual_dim)

    def score(
        self,
        x: np.ndarray,
        u: np.ndarray,
        c: np.ndarray,
        y_t: np.ndarray,
        t_index: int,
    ) -> np.ndarray:
        eps = self.predict_noise(x, u, c, y_t, t_index)
        denom = np.sqrt(max(1e-8, 1.0 - self.schedule.alpha_bars[t_index]))
        return -eps / denom

    def sample(
        self,
        x: np.ndarray,
        u: np.ndarray,
        c: np.ndarray,
        n_samples: int = 1,
        deterministic: bool = False,
        seed: int | None = None,
    ) -> np.ndarray:
        self._check_fitted()
        rng = np.random.default_rng(self.random_state if seed is None else seed)
        x_arr = np.asarray(x, dtype=float)
        u_arr = np.asarray(u, dtype=float)
        c_arr = np.asarray(c, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr[None, :]
            u_arr = u_arr[None, :]
            c_arr = c_arr[None, :]

        count = x_arr.shape[0] * int(n_samples)
        x_rep = np.repeat(x_arr, n_samples, axis=0)
        u_rep = np.repeat(u_arr, n_samples, axis=0)
        c_rep = np.repeat(c_arr, n_samples, axis=0)
        y = rng.normal(0.0, 1.0, size=(count, self.residual_dim))

        # DDPM posterior update. The previous implementation reintroduced the
        # full forward variance at every reverse step, inflating terminal
        # residual covariance and consequently the verification abstraction.
        for t in range(self.schedule.timesteps - 1, -1, -1):
            beta_t = float(self.schedule.betas[t])
            alpha_t = float(self.schedule.alphas[t])
            abar_t = float(self.schedule.alpha_bars[t])
            eps = self.predict_noise(x_rep, u_rep, c_rep, y, t)
            posterior_mean = (y - beta_t * eps / np.sqrt(max(1.0 - abar_t, 1e-8))) / np.sqrt(alpha_t)
            if t > 0 and not deterministic:
                abar_prev = float(self.schedule.alpha_bars[t - 1])
                posterior_variance = beta_t * (1.0 - abar_prev) / max(1.0 - abar_t, 1e-8)
                y = posterior_mean + np.sqrt(max(posterior_variance, 0.0)) * rng.normal(0.0, 1.0, size=y.shape)
            else:
                y = posterior_mean

        location, scale = self._residual_normalization()
        return (y.reshape(x_arr.shape[0], n_samples, self.residual_dim) * scale[None, None, :]) + location[None, None, :]

    def sample_mean(self, x: np.ndarray, u: np.ndarray, c: np.ndarray, n_samples: int = 16) -> np.ndarray:
        return self.sample(x, u, c, n_samples=n_samples, deterministic=False).mean(axis=1)

    def _build_training_data(
        self,
        x: np.ndarray,
        u: np.ndarray,
        c: np.ndarray,
        residual: np.ndarray,
        copies: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = residual.shape[0]
        total = n * copies
        x_rep = np.repeat(np.asarray(x, dtype=float), copies, axis=0)
        u_rep = np.repeat(np.asarray(u, dtype=float), copies, axis=0)
        c_rep = np.repeat(np.asarray(c, dtype=float), copies, axis=0)
        r_rep = np.repeat(np.asarray(residual, dtype=float), copies, axis=0)
        t_idx = rng.integers(0, self.schedule.timesteps, size=total)
        eps = rng.normal(0.0, 1.0, size=r_rep.shape)

        abar = self.schedule.alpha_bars[t_idx][:, None]
        y_t = np.sqrt(abar) * r_rep + np.sqrt(1.0 - abar) * eps
        features = self._feature(x_rep, u_rep, c_rep, y_t, t_idx)
        return features, eps

    def _feature(
        self,
        x: np.ndarray,
        u: np.ndarray,
        c: np.ndarray,
        y_t: np.ndarray,
        t_index: int | np.ndarray,
    ) -> np.ndarray:
        x_arr = np.atleast_2d(np.asarray(x, dtype=float))
        u_arr = np.atleast_2d(np.asarray(u, dtype=float))
        c_arr = np.atleast_2d(np.asarray(c, dtype=float))
        y_arr = np.atleast_2d(np.asarray(y_t, dtype=float))
        t_arr = np.asarray(t_index, dtype=float)
        if t_arr.ndim == 0:
            t_arr = np.full((x_arr.shape[0], 1), float(t_arr) / max(1, self.schedule.timesteps - 1))
        else:
            t_arr = t_arr.reshape(-1, 1) / max(1, self.schedule.timesteps - 1)
        return np.concatenate([x_arr, u_arr, c_arr, y_arr, t_arr, np.sin(np.pi * t_arr), np.cos(np.pi * t_arr)], axis=1)

    def _check_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("ConditionalResidualDiffusion.fit must be called first")
        terminal_limit = float(getattr(self.schedule, "terminal_alpha_bar_max", 1e-3))
        terminal_alpha_bar = float(np.asarray(self.schedule.alpha_bars, dtype=float)[-1])
        if terminal_alpha_bar > terminal_limit:
            raise RuntimeError(
                "This diffusion model was trained with a terminal distribution that does not match the Gaussian prior: "
                f"alpha_bar_T={terminal_alpha_bar:.6g} exceeds {terminal_limit:.6g}. Retrain the model with the current schedule."
            )

    def _residual_normalization(self) -> tuple[np.ndarray, np.ndarray]:
        if self.residual_location is None or self.residual_scale is None:
            raise RuntimeError("Residual normalization is unavailable before fit")
        return self.residual_location, self.residual_scale
