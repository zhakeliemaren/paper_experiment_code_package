from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ResidualMoments:
    """Conditional residual moments induced by the trained diffusion model."""

    residual_mean: np.ndarray
    residual_drift: np.ndarray
    diffusion_covariance: np.ndarray
    diagnostics: dict[str, Any] = field(default_factory=dict)


class ConditionalMomentMatchedResidualSDE:
    """Moment-match a terminal residual distribution to a physical-time SDE.

    The diffusion model is sampled by integrating its reverse diffusion
    process.  Its terminal residual distribution is then converted to the
    local physical-time approximation ``dX = mu/dt dt + Sigma^1/2 dW``.
    This is deliberately distinct from the reverse-time diffusion SDE: the
    latter evolves the residual latent variable, not the physical state.
    """

    def __init__(
        self,
        diffusion: Any,
        dt: float,
        n_samples: int = 32,
        random_state: int = 7,
        moment_method: str = "reverse_polynomial",
        polynomial_degree: int = 2,
        fit_radius: float = 2.0,
        validation_radius: float = 3.0,
        validation_margin: float = 1.25,
        physical_state_dim: int | None = None,
        physical_state_indices: tuple[int, ...] | None = None,
    ) -> None:
        self.diffusion = diffusion
        self.dt = float(dt)
        self.n_samples = int(n_samples)
        self.random_state = int(random_state)
        self.moment_method = self._normalize_moment_method(moment_method)
        self.polynomial_degree = int(polynomial_degree)
        self.fit_radius = float(fit_radius)
        self.validation_radius = float(validation_radius)
        self.validation_margin = float(validation_margin)
        residual_dim = int(diffusion.residual_dim)
        self.physical_state_dim = residual_dim if physical_state_dim is None else int(physical_state_dim)
        self.physical_state_indices = tuple(
            range(residual_dim) if physical_state_indices is None else physical_state_indices
        )
        if len(self.physical_state_indices) != residual_dim:
            raise ValueError("physical_state_indices length must match diffusion residual_dim")
        if len(set(self.physical_state_indices)) != residual_dim or any(
            index < 0 or index >= self.physical_state_dim for index in self.physical_state_indices
        ):
            raise ValueError("physical_state_indices must be unique valid physical-state coordinates")
        if self.polynomial_degree != 2:
            raise ValueError("reverse polynomial moment propagation currently requires degree 2")
        if self.fit_radius <= 0.0 or self.validation_radius <= 0.0 or self.validation_margin < 1.0:
            raise ValueError("reverse polynomial radii must be positive and validation_margin must be at least one")

    def score(
        self,
        x: np.ndarray,
        u: np.ndarray,
        c: np.ndarray,
        y_t: np.ndarray,
        t_index: int,
    ) -> np.ndarray:
        return self.diffusion.score(x, u, c, y_t, int(t_index))

    def reverse_sde_drift(
        self,
        x: np.ndarray,
        u: np.ndarray,
        c: np.ndarray,
        y_t: np.ndarray,
        t_index: int,
    ) -> np.ndarray:
        t = int(t_index)
        beta_t = float(self.diffusion.schedule.betas[t])
        score = self.score(x, u, c, y_t, t)
        return -0.5 * beta_t * np.asarray(y_t, dtype=float) - beta_t * score

    def reverse_diffusion_scale(self, t_index: int) -> float:
        beta_t = float(self.diffusion.schedule.betas[int(t_index)])
        return float(np.sqrt(max(beta_t, 0.0)))

    def residual_samples(
        self,
        x: np.ndarray,
        u: np.ndarray,
        c: np.ndarray,
        n_samples: int | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        count = self.n_samples if n_samples is None else int(n_samples)
        return self.diffusion.sample(x, u, c, n_samples=count, deterministic=False, seed=seed)

    def physical_sde_moments(
        self,
        x: np.ndarray,
        u: np.ndarray,
        c: np.ndarray,
        n_samples: int | None = None,
        seed: int | None = None,
    ) -> ResidualMoments:
        # Artifacts serialized before reverse-polynomial propagation was added
        # do not contain ``moment_method`` and must retain their old behavior.
        method = self._normalize_moment_method(getattr(self, "moment_method", "terminal_sampling"))
        if method == "reverse_polynomial":
            return self._embed_physical_moments(self._reverse_polynomial_moments(x, u, c))

        samples = self.residual_samples(x, u, c, n_samples=n_samples, seed=seed)
        mean = samples.mean(axis=1)
        drift = mean / max(self.dt, 1e-12)
        covariances = self._sample_covariances(samples) / max(self.dt, 1e-12)
        return self._embed_physical_moments(ResidualMoments(
            residual_mean=mean,
            residual_drift=drift,
            diffusion_covariance=covariances,
            diagnostics={
                "method": "terminal_sampling",
                "terminal_sampling_used": True,
                "sample_count": int(samples.shape[1]),
                "physical_drift_validation_error": 0.0,
                "validation_error_is_formal_bound": False,
            },
        ))

    def _physical_embedding_spec(self) -> tuple[int, tuple[int, ...]]:
        residual_dim = int(self.diffusion.residual_dim)
        state_dim = int(getattr(self, "physical_state_dim", residual_dim))
        indices = tuple(getattr(self, "physical_state_indices", tuple(range(residual_dim))))
        return state_dim, indices

    def _embed_physical_moments(self, moments: ResidualMoments) -> ResidualMoments:
        state_dim, indices = self._physical_embedding_spec()
        residual_dim = int(self.diffusion.residual_dim)
        if state_dim == residual_dim and indices == tuple(range(residual_dim)):
            return moments

        latent_mean = np.asarray(moments.residual_mean, dtype=float)
        latent_drift = np.asarray(moments.residual_drift, dtype=float)
        latent_covariance = np.asarray(moments.diffusion_covariance, dtype=float)
        full_mean = np.zeros((latent_mean.shape[0], state_dim), dtype=float)
        full_drift = np.zeros((latent_drift.shape[0], state_dim), dtype=float)
        full_covariance = np.zeros((latent_covariance.shape[0], state_dim, state_dim), dtype=float)
        full_mean[:, indices] = latent_mean
        full_drift[:, indices] = latent_drift
        for latent_i, physical_i in enumerate(indices):
            for latent_j, physical_j in enumerate(indices):
                full_covariance[:, physical_i, physical_j] = latent_covariance[:, latent_i, latent_j]
        diagnostics = dict(moments.diagnostics)
        diagnostics.update(
            {
                "latent_diffusion_dimension": residual_dim,
                "physical_state_dimension": state_dim,
                "physical_stochastic_state_indices": list(indices),
            }
        )
        return ResidualMoments(
            residual_mean=full_mean,
            residual_drift=full_drift,
            diffusion_covariance=full_covariance,
            diagnostics=diagnostics,
        )

    def _reverse_polynomial_moments(
        self,
        x: np.ndarray,
        u: np.ndarray,
        c: np.ndarray,
    ) -> ResidualMoments:
        """Propagate reverse-DDPM moments without drawing terminal samples.

        The neural noise predictor is approximated, independently at each
        condition and reverse step, by a quadratic polynomial in a whitened
        latent residual. Mean and covariance are exact for that polynomial
        under the step's Gaussian closure. The next step closes the transformed
        distribution back to a Gaussian, so this is a deterministic surrogate
        of the learned sampler rather than an exact neural-sampler moment.
        """

        self.diffusion._check_fitted()
        x_arr, u_arr, c_arr = self._condition_arrays(x, u, c)
        batch = x_arr.shape[0]
        dim = int(self.diffusion.residual_dim)
        if dim < 1:
            raise ValueError("residual dimension must be positive")

        fit_nodes = self._quadratic_nodes(dim, self.fit_radius)
        validation_nodes = self._quadratic_nodes(dim, self.validation_radius)
        fit_basis, quadratic_pairs = self._quadratic_basis(fit_nodes)
        fit_pinv = np.linalg.pinv(fit_basis)
        validation_basis, _ = self._quadratic_basis(validation_nodes)

        mean = np.zeros((batch, dim), dtype=float)
        covariance = np.repeat(np.eye(dim, dtype=float)[None, :, :], batch, axis=0)
        identity = np.eye(dim, dtype=float)
        step_fit_errors: list[float] = []
        accumulated_update_error = 0.0
        max_quadratic_covariance_fraction = 0.0

        for t in range(int(self.diffusion.schedule.timesteps) - 1, -1, -1):
            roots = self._psd_square_roots(covariance)
            fit_y = mean[:, None, :] + np.einsum("nij,pj->npi", roots, fit_nodes)
            fit_eps = self._predict_noise_at_nodes(x_arr, u_arr, c_arr, fit_y, t)
            coefficients = np.einsum("kp,npd->nkd", fit_pinv, fit_eps)
            constant, linear, quadratic = self._unpack_quadratic_coefficients(
                coefficients,
                dim,
                quadratic_pairs,
            )

            expected_eps = constant + np.trace(quadratic, axis1=2, axis2=3)
            linear_covariance = np.einsum("npi,nqi->npq", linear, linear)
            quadratic_covariance = 2.0 * np.einsum("npij,nqji->npq", quadratic, quadratic)
            eps_covariance = linear_covariance + quadratic_covariance
            cross_covariance = np.einsum("nij,npj->nip", roots, linear)

            validation_y = mean[:, None, :] + np.einsum("nij,pj->npi", roots, validation_nodes)
            validation_eps = self._predict_noise_at_nodes(x_arr, u_arr, c_arr, validation_y, t)
            validation_pred = np.einsum("pk,nkd->npd", validation_basis, coefficients)
            validation_error = np.linalg.norm(validation_eps - validation_pred, axis=2)
            step_error = float(np.max(validation_error)) * self.validation_margin
            step_fit_errors.append(step_error)

            beta = float(self.diffusion.schedule.betas[t])
            alpha = float(self.diffusion.schedule.alphas[t])
            abar = float(self.diffusion.schedule.alpha_bars[t])
            state_gain = 1.0 / np.sqrt(max(alpha, 1e-12))
            predictor_gain = -beta / np.sqrt(max(alpha * (1.0 - abar), 1e-12))
            posterior_variance = 0.0
            if t > 0:
                abar_prev = float(self.diffusion.schedule.alpha_bars[t - 1])
                posterior_variance = beta * (1.0 - abar_prev) / max(1.0 - abar, 1e-12)

            mean = state_gain * mean + predictor_gain * expected_eps
            covariance = (
                state_gain**2 * covariance
                + predictor_gain**2 * eps_covariance
                + state_gain * predictor_gain * (cross_covariance + np.swapaxes(cross_covariance, 1, 2))
                + max(posterior_variance, 0.0) * identity[None, :, :]
            )
            covariance = self._project_psd(covariance)
            accumulated_update_error = (
                abs(state_gain) * accumulated_update_error + abs(predictor_gain) * step_error
            )
            total_eps_trace = np.trace(eps_covariance, axis1=1, axis2=2)
            quadratic_trace = np.trace(quadratic_covariance, axis1=1, axis2=2)
            fractions = quadratic_trace / np.maximum(total_eps_trace, 1e-15)
            max_quadratic_covariance_fraction = max(
                max_quadratic_covariance_fraction,
                float(np.max(fractions)),
            )

        location, scale = self.diffusion._residual_normalization()
        location_arr = np.asarray(location, dtype=float)
        scale_arr = np.asarray(scale, dtype=float)
        residual_mean = mean * scale_arr[None, :] + location_arr[None, :]
        physical_covariance = covariance * scale_arr[None, :, None] * scale_arr[None, None, :]
        dt = max(self.dt, 1e-12)
        drift_error_indicator = accumulated_update_error * float(np.max(np.abs(scale_arr))) / dt
        diagnostics: dict[str, Any] = {
            "method": "reverse_polynomial",
            "terminal_sampling_used": False,
            "polynomial_degree": int(self.polynomial_degree),
            "fit_node_count": int(fit_nodes.shape[0]),
            "validation_node_count": int(validation_nodes.shape[0]),
            "fit_radius": float(self.fit_radius),
            "validation_radius": float(self.validation_radius),
            "validation_margin": float(self.validation_margin),
            "max_step_noise_fit_error": float(max(step_fit_errors, default=0.0)),
            "mean_step_noise_fit_error": float(np.mean(step_fit_errors)) if step_fit_errors else 0.0,
            "normalized_terminal_update_error_indicator": float(accumulated_update_error),
            "physical_drift_validation_error": float(drift_error_indicator),
            "validation_error_is_formal_bound": False,
            "gaussian_moment_closure": True,
            "moment_closure_error_certified": False,
            "max_quadratic_covariance_fraction": float(max_quadratic_covariance_fraction),
            "note": (
                "No terminal residual samples are drawn. The reported validation error is a deterministic-node "
                "indicator for the quadratic reverse surrogate, not a global neural-network or Gaussian-closure bound."
            ),
        }
        return ResidualMoments(
            residual_mean=residual_mean,
            residual_drift=residual_mean / dt,
            diffusion_covariance=physical_covariance / dt,
            diagnostics=diagnostics,
        )

    # Retained for serialized models and downstream callers from older runs.
    def residual_moments(
        self,
        x: np.ndarray,
        u: np.ndarray,
        c: np.ndarray,
        n_samples: int | None = None,
        seed: int | None = None,
    ) -> ResidualMoments:
        return self.physical_sde_moments(x, u, c, n_samples=n_samples, seed=seed)

    def stochastic_dynamics(
        self,
        x: np.ndarray,
        u: np.ndarray,
        c: np.ndarray,
        system: Any,
        n_samples: int | None = None,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        moments = self.physical_sde_moments(x, u, c, n_samples=n_samples, seed=seed)
        nominal = system.nominal_derivative(np.asarray(x, dtype=float), np.asarray(u, dtype=float))
        return nominal + moments.residual_drift, moments.diffusion_covariance

    @staticmethod
    def _sample_covariances(samples: np.ndarray) -> np.ndarray:
        arr = np.asarray(samples, dtype=float)
        n_batch, n_samples, dim = arr.shape
        covs = np.zeros((n_batch, dim, dim), dtype=float)
        if n_samples <= 1:
            return covs
        for i in range(n_batch):
            centered = arr[i] - arr[i].mean(axis=0, keepdims=True)
            covs[i] = centered.T @ centered / float(n_samples - 1)
        return covs

    @staticmethod
    def _normalize_moment_method(method: str) -> str:
        normalized = str(method).strip().lower().replace("-", "_")
        aliases = {
            "sample": "terminal_sampling",
            "sampling": "terminal_sampling",
            "terminal_sample": "terminal_sampling",
            "terminal_sampling": "terminal_sampling",
            "reverse_polynomial": "reverse_polynomial",
            "polynomial": "reverse_polynomial",
        }
        if normalized not in aliases:
            raise ValueError("moment_method must be 'reverse_polynomial' or 'terminal_sampling'")
        return aliases[normalized]

    @staticmethod
    def _condition_arrays(
        x: np.ndarray,
        u: np.ndarray,
        c: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_arr = np.atleast_2d(np.asarray(x, dtype=float))
        u_arr = np.atleast_2d(np.asarray(u, dtype=float))
        c_arr = np.atleast_2d(np.asarray(c, dtype=float))
        if not (x_arr.shape[0] == u_arr.shape[0] == c_arr.shape[0]):
            raise ValueError("x, u, and c must have the same batch dimension")
        return x_arr, u_arr, c_arr

    @staticmethod
    def _quadratic_nodes(dim: int, radius: float) -> np.ndarray:
        nodes = [np.zeros(dim, dtype=float)]
        for axis in range(dim):
            direction = np.zeros(dim, dtype=float)
            direction[axis] = float(radius)
            nodes.extend([direction, -direction])
        pair_radius = float(radius) / np.sqrt(2.0)
        for left in range(dim):
            for right in range(left + 1, dim):
                for left_sign in (-1.0, 1.0):
                    for right_sign in (-1.0, 1.0):
                        direction = np.zeros(dim, dtype=float)
                        direction[left] = left_sign * pair_radius
                        direction[right] = right_sign * pair_radius
                        nodes.append(direction)
        return np.asarray(nodes, dtype=float)

    @staticmethod
    def _quadratic_basis(nodes: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
        arr = np.asarray(nodes, dtype=float)
        dim = arr.shape[1]
        pairs = [(left, right) for left in range(dim) for right in range(left, dim)]
        columns = [np.ones(arr.shape[0], dtype=float)]
        columns.extend(arr[:, axis] for axis in range(dim))
        columns.extend(arr[:, left] * arr[:, right] for left, right in pairs)
        return np.column_stack(columns), pairs

    @staticmethod
    def _unpack_quadratic_coefficients(
        coefficients: np.ndarray,
        dim: int,
        pairs: list[tuple[int, int]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        coeffs = np.asarray(coefficients, dtype=float)
        constant = coeffs[:, 0, :]
        linear = np.transpose(coeffs[:, 1 : 1 + dim, :], (0, 2, 1))
        quadratic = np.zeros((coeffs.shape[0], coeffs.shape[2], dim, dim), dtype=float)
        offset = 1 + dim
        for index, (left, right) in enumerate(pairs):
            value = coeffs[:, offset + index, :]
            if left == right:
                quadratic[:, :, left, right] = value
            else:
                quadratic[:, :, left, right] = 0.5 * value
                quadratic[:, :, right, left] = 0.5 * value
        return constant, linear, quadratic

    def _predict_noise_at_nodes(
        self,
        x: np.ndarray,
        u: np.ndarray,
        c: np.ndarray,
        y: np.ndarray,
        t_index: int,
    ) -> np.ndarray:
        batch, nodes, dim = y.shape
        prediction = self.diffusion.predict_noise(
            np.repeat(x, nodes, axis=0),
            np.repeat(u, nodes, axis=0),
            np.repeat(c, nodes, axis=0),
            y.reshape(batch * nodes, dim),
            int(t_index),
        )
        return np.asarray(prediction, dtype=float).reshape(batch, nodes, dim)

    @staticmethod
    def _psd_square_roots(covariances: np.ndarray) -> np.ndarray:
        symmetric = 0.5 * (np.asarray(covariances, dtype=float) + np.swapaxes(covariances, 1, 2))
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        clipped = np.maximum(eigenvalues, 1e-12)
        return np.einsum("nij,nj->nij", eigenvectors, np.sqrt(clipped))

    @staticmethod
    def _project_psd(covariances: np.ndarray) -> np.ndarray:
        symmetric = 0.5 * (np.asarray(covariances, dtype=float) + np.swapaxes(covariances, 1, 2))
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        clipped = np.maximum(eigenvalues, 1e-12)
        return np.einsum("nij,nj,nkj->nik", eigenvectors, clipped, eigenvectors)


# Compatibility aliases for artifacts and downstream callers from older runs.
DiffusionResidualMomentDynamics = ConditionalMomentMatchedResidualSDE
ScoreInducedResidualDynamics = ConditionalMomentMatchedResidualSDE
