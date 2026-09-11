from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .sos import Poly, add_poly, box_domain_polynomials, constant_poly, monomial_poly


def _clip(value: np.ndarray, bounds: Iterable[tuple[float, float]]) -> np.ndarray:
    arr = np.asarray(value, dtype=float).copy()
    for i, (lo, hi) in enumerate(bounds):
        arr[..., i] = np.clip(arr[..., i], lo, hi)
    return arr


def _sample_box(rng: np.random.Generator, box: tuple[tuple[float, float], ...], n: int) -> np.ndarray:
    lows = np.array([b[0] for b in box], dtype=float)
    highs = np.array([b[1] for b in box], dtype=float)
    return rng.uniform(lows, highs, size=(n, len(box)))


@dataclass
class BenchmarkSystem:
    name: str
    dt: float
    state_dim: int
    control_dim: int
    process_noise: float
    verify_box: tuple[tuple[float, float], ...]
    initial_box: tuple[tuple[float, float], ...]
    unsafe_box: tuple[tuple[float, float], ...]
    control_bounds: tuple[tuple[float, float], ...]
    barrier_box: tuple[tuple[float, float], ...] | None = None

    def sample_initial_states(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return _sample_box(rng, self.initial_box, n)

    def sample_training_initial_states(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Initial states for system-identification trajectories."""
        return self.sample_initial_states(rng, n)

    def accepts_training_transition(self, x: np.ndarray, nominal_next: np.ndarray, next_state: np.ndarray) -> bool:
        """Whether a simulated transition is suitable for residual learning."""
        return True

    def sample_verify_states(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return _sample_box(rng, self.verify_box, n)

    def sample_unsafe_states(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return _sample_box(rng, self.unsafe_box, n)

    def control_policy(self, x: np.ndarray, step: int = 0) -> np.ndarray:
        raise NotImplementedError

    def control_policy_batch(self, x: np.ndarray, step: int = 0) -> np.ndarray:
        """Vectorized policy hook; subclasses should override when possible."""
        rows = np.atleast_2d(np.asarray(x, dtype=float))
        return np.asarray([self.control_policy(row, step=step) for row in rows], dtype=float)

    def exact_derivative(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def nominal_derivative(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self.exact_derivative(x, u)

    def nominal_step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float) + self.dt * self.nominal_derivative(x, u)

    def abstraction_partition_breakpoints(self) -> tuple[tuple[float, ...], ...] | None:
        """Optional state-space breakpoints where the closed-loop drift changes polynomial form."""
        return None

    def stochastic_state_indices(self) -> tuple[int, ...]:
        """Physical-state coordinates directly driven by stochastic increments."""
        return tuple(range(int(self.state_dim)))

    def actual_step(self, x: np.ndarray, u: np.ndarray, rng: np.random.Generator, step: int = 0) -> np.ndarray:
        drift = self.exact_derivative(x, u)
        noise = rng.normal(0.0, self.process_noise * np.sqrt(self.dt), size=np.asarray(x).shape)
        x_next = np.asarray(x, dtype=float) + self.dt * drift + noise
        return _clip(x_next, self.verify_box)

    def clamp_control(self, u: np.ndarray) -> np.ndarray:
        return _clip(np.asarray(u, dtype=float), self.control_bounds)

    def domain_generators(self, box: tuple[tuple[float, float], ...] | None = None) -> list[Poly]:
        return box_domain_polynomials(self.barrier_box if box is None and self.barrier_box is not None else (box or self.verify_box))

    def initial_generators(self) -> list[Poly]:
        return box_domain_polynomials(self.initial_box)

    def unsafe_generators(self) -> list[Poly]:
        return box_domain_polynomials(self.unsafe_box)


def batch_control_policy(system: BenchmarkSystem, x: np.ndarray, step: int = 0) -> np.ndarray:
    x_arr = np.atleast_2d(np.asarray(x, dtype=float))
    if x_arr.size == 0:
        return np.empty((0, system.control_dim), dtype=float)
    actions = system.control_policy_batch(x_arr, step=step)
    return np.asarray(actions, dtype=float).reshape(-1, system.control_dim)


@dataclass
class ToySystem(BenchmarkSystem):
    stiffness: float = 0.9
    damping: float = 0.35

    def control_policy(self, x: np.ndarray, step: int = 0) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        u = -0.65 * x[..., 0] - 0.45 * x[..., 1]
        return self.clamp_control(np.asarray([u], dtype=float))

    def control_policy_batch(self, x: np.ndarray, step: int = 0) -> np.ndarray:
        rows = np.atleast_2d(np.asarray(x, dtype=float))
        u = -0.65 * rows[:, 0] - 0.45 * rows[:, 1]
        return self.clamp_control(u[:, None])

    def exact_derivative(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        u = np.asarray(u, dtype=float)
        out = np.empty_like(x, dtype=float)
        out[..., 0] = x[..., 1]
        out[..., 1] = -self.stiffness * x[..., 0] - self.damping * x[..., 1] + u[..., 0]
        return out


@dataclass
class XPlane11TrajectoryTracking(BenchmarkSystem):
    v: float = 5.0
    wheelbase: float = 5.0

    def control_policy(self, x: np.ndarray, step: int = 0) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        p = x[..., 0]
        theta = x[..., 1]
        phi = -0.9 * p - 1.25 * theta
        return self.clamp_control(np.asarray([phi], dtype=float))

    def control_policy_batch(self, x: np.ndarray, step: int = 0) -> np.ndarray:
        rows = np.atleast_2d(np.asarray(x, dtype=float))
        phi = -0.9 * rows[:, 0] - 1.25 * rows[:, 1]
        return self.clamp_control(phi[:, None])

    def exact_derivative(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        u = np.asarray(u, dtype=float)
        theta = x[..., 1]
        phi = u[..., 0]
        out = np.empty_like(x, dtype=float)
        out[..., 0] = self.v * np.sin(theta)
        out[..., 1] = (self.v / self.wheelbase) * np.tan(phi)
        return out

    def nominal_derivative(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        u = np.asarray(u, dtype=float)
        theta = x[..., 1]
        phi = u[..., 0]
        sin_theta = theta - theta**3 / 6.0 + theta**5 / 120.0
        tan_phi = phi + phi**3 / 3.0 + 2.0 * phi**5 / 15.0
        out = np.empty_like(x, dtype=float)
        out[..., 0] = self.v * sin_theta
        out[..., 1] = (self.v / self.wheelbase) * tan_phi
        return out


@dataclass
class CarlaEmergencyBraking(BenchmarkSystem):
    max_brake: float = 3.0
    friction_enabled: bool = True
    friction_mean: float = 0.82
    friction_std: float = 0.12
    friction_min: float = 0.45
    friction_max: float = 1.12
    friction_speed_gain: float = 0.10
    friction_distance_gain: float = 0.08
    stop_speed: float = 0.5
    absorbing_stop: bool = True

    def abstraction_partition_breakpoints(self) -> tuple[tuple[float, ...], ...]:
        # The max(12-d, 0) controller term is polynomial on either side of d=12.
        return ((12.0,), ())

    def stochastic_state_indices(self) -> tuple[int, ...]:
        # Road/braking uncertainty acts on acceleration and therefore directly
        # perturbs speed. Distance remains random indirectly through d_dot=-v.
        return (1,)

    def control_policy(self, x: np.ndarray, step: int = 0) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        d = x[..., 0]
        v = x[..., 1]
        a = 0.55 * v + 0.12 * np.maximum(12.0 - d, 0.0)
        return self.clamp_control(np.asarray([a], dtype=float))

    def control_policy_batch(self, x: np.ndarray, step: int = 0) -> np.ndarray:
        rows = np.atleast_2d(np.asarray(x, dtype=float))
        a = 0.55 * rows[:, 1] + 0.12 * np.maximum(12.0 - rows[:, 0], 0.0)
        return self.clamp_control(a[:, None])

    def exact_derivative(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        u = np.asarray(u, dtype=float)
        out = np.empty_like(x, dtype=float)
        out[..., 0] = -x[..., 1]
        out[..., 1] = -u[..., 0]
        return out

    def braking_effectiveness(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Sample a bounded non-Gaussian tire-road braking coefficient."""
        rows = np.asarray(x, dtype=float)
        if not self.friction_enabled:
            return np.ones(rows.shape[:-1], dtype=float)
        span = max(float(self.friction_max - self.friction_min), 1e-9)
        mean = np.clip((self.friction_mean - self.friction_min) / span, 1e-3, 1.0 - 1e-3)
        std = max(float(self.friction_std) / span, 1e-4)
        concentration = max(mean * (1.0 - mean) / std**2 - 1.0, 2.0)
        iid_component = self.friction_min + span * rng.beta(mean * concentration, (1.0 - mean) * concentration, size=rows.shape[:-1])
        speed_penalty = self.friction_speed_gain * np.clip(rows[..., 1] / 3.0, 0.0, 1.0)
        close_range_penalty = self.friction_distance_gain * np.clip((12.0 - rows[..., 0]) / 7.0, 0.0, 1.0)
        return np.clip(iid_component - speed_penalty - close_range_penalty, self.friction_min, self.friction_max)

    def actual_step(self, x: np.ndarray, u: np.ndarray, rng: np.random.Generator, step: int = 0) -> np.ndarray:
        rows = np.asarray(x, dtype=float)
        control = np.asarray(u, dtype=float)
        active = rows[..., 1] > float(self.stop_speed)
        drift = self.exact_derivative(rows, control)
        if self.friction_enabled:
            drift[..., 1] = -self.braking_effectiveness(rows, rng) * control[..., 0]
            noise = np.zeros_like(rows)
        else:
            noise = np.zeros_like(rows)
            noise[..., 1] = rng.normal(
                0.0,
                self.process_noise * np.sqrt(self.dt),
                size=rows.shape[:-1],
            )
        candidate = rows + self.dt * drift + noise
        if self.absorbing_stop:
            next_rows = np.where(active[..., None], candidate, rows)
            entered = active & (next_rows[..., 1] <= float(self.stop_speed))
            next_rows[..., 1] = np.where(entered, 0.0, next_rows[..., 1])
        else:
            next_rows = candidate
        # CARLA uncertainty is multiplicative for carla1 and additive Gaussian
        # for carla; neither is injected after the stopped safe state is reached.
        return _clip(next_rows, self.verify_box)

    def accepts_training_transition(self, x: np.ndarray, nominal_next: np.ndarray, next_state: np.ndarray) -> bool:
        # The absorbing stop is a model boundary, not a residual sample. Keep
        # only transitions strictly inside the active verification region.
        points = (x, nominal_next, next_state)
        for point in points:
            values = np.asarray(point, dtype=float)
            if values[1] <= float(self.stop_speed) + 1e-9:
                return False
        return True


@dataclass
class PrajnaCdc2004Continuous(BenchmarkSystem):
    """Example 1 in Prajna, Jadbabaie, and Pappas (CDC 2004)."""

    def control_policy(self, x: np.ndarray, step: int = 0) -> np.ndarray:
        return np.zeros((1,), dtype=float)

    def control_policy_batch(self, x: np.ndarray, step: int = 0) -> np.ndarray:
        return np.zeros((np.atleast_2d(x).shape[0], 1), dtype=float)

    def exact_derivative(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        rows = np.asarray(x, dtype=float)
        out = np.empty_like(rows, dtype=float)
        out[..., 0] = rows[..., 1]
        out[..., 1] = -rows[..., 0] - rows[..., 1] - 0.5 * rows[..., 0] ** 3
        return out

    def actual_step(self, x: np.ndarray, u: np.ndarray, rng: np.random.Generator, step: int = 0) -> np.ndarray:
        rows = np.asarray(x, dtype=float)
        noise = np.zeros_like(rows)
        noise[..., 1] = rng.normal(0.0, self.process_noise * np.sqrt(self.dt), size=rows.shape[:-1])
        candidate = _clip(rows + self.dt * self.exact_derivative(rows, u) + noise, self.verify_box)
        active = self._active_mask(rows)
        return np.where(active[..., None], candidate, rows)

    def accepts_training_transition(self, x: np.ndarray, nominal_next: np.ndarray, next_state: np.ndarray) -> bool:
        # The paper verifies the process stopped on exit from Int(X). Exclude
        # transitions that touch the outer boundary or enter the inner disk;
        # their clipping/absorption is not an SDE residual sample.
        return all(bool(self._active_mask(np.asarray(point, dtype=float))) for point in (x, nominal_next, next_state))

    def _active_mask(self, x: np.ndarray) -> np.ndarray:
        rows = np.asarray(x, dtype=float)
        lower = np.asarray([bound[0] for bound in self.verify_box], dtype=float)
        upper = np.asarray([bound[1] for bound in self.verify_box], dtype=float)
        inside_outer = np.all((rows > lower + 1e-12) & (rows < upper - 1e-12), axis=-1)
        outside_stop_disk = np.sum(rows**2, axis=-1) > 0.25 + 1e-12
        return inside_outer & outside_stop_disk

    def sample_initial_states(self, rng: np.random.Generator, n: int) -> np.ndarray:
        radius = 0.1 * np.sqrt(rng.uniform(0.0, 1.0, size=n))
        angle = rng.uniform(0.0, 2.0 * np.pi, size=n)
        return np.column_stack([-2.0 + radius * np.cos(angle), radius * np.sin(angle)])

    def sample_verify_states(self, rng: np.random.Generator, n: int) -> np.ndarray:
        points: list[np.ndarray] = []
        while sum(len(chunk) for chunk in points) < n:
            candidate = _sample_box(rng, self.verify_box, max(n, 2 * n))
            mask = np.sum(candidate**2, axis=1) >= 0.25
            points.append(candidate[mask])
        return np.vstack(points)[:n]

    def sample_unsafe_states(self, rng: np.random.Generator, n: int) -> np.ndarray:
        points: list[np.ndarray] = []
        while sum(len(chunk) for chunk in points) < n:
            candidate = _sample_box(rng, self.unsafe_box, max(n, 2 * n))
            mask = np.sum(candidate**2, axis=1) >= 0.25
            points.append(candidate[mask])
        return np.vstack(points)[:n]

    def domain_generators(self, box: tuple[tuple[float, float], ...] | None = None) -> list[Poly]:
        target = box or self.verify_box
        circle = add_poly(monomial_poly((2, 0), 1.0), monomial_poly((0, 2), 1.0), constant_poly(2, -0.25))
        return [*box_domain_polynomials(target), circle]

    def initial_generators(self) -> list[Poly]:
        return [
            add_poly(
                monomial_poly((2, 0), -1.0),
                monomial_poly((1, 0), -4.0),
                monomial_poly((0, 2), -1.0),
                constant_poly(2, -3.99),
            )
        ]

    def unsafe_generators(self) -> list[Poly]:
        # Paper definition: Xu = X intersect {x2 >= 2.25}. Preserve that
        # semialgebraic description instead of replacing the x2 constraints by
        # the equivalent interval product; finite-degree Putinar relaxations
        # depend on the selected generators even when the sets are identical.
        unsafe_halfspace = add_poly(monomial_poly((0, 1), 1.0), constant_poly(2, -2.25))
        return [*self.domain_generators(self.verify_box), unsafe_halfspace]


@dataclass
class SynNBCF2Oscillator(BenchmarkSystem):
    """SynNBC F2: a two-state cubic oscillator with rectangular safety sets."""

    def control_policy(self, x: np.ndarray, step: int = 0) -> np.ndarray:
        return np.zeros((1,), dtype=float)

    def control_policy_batch(self, x: np.ndarray, step: int = 0) -> np.ndarray:
        return np.zeros((np.atleast_2d(x).shape[0], 1), dtype=float)

    def exact_derivative(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        rows = np.asarray(x, dtype=float)
        out = np.empty_like(rows, dtype=float)
        out[..., 0] = rows[..., 1]
        out[..., 1] = -rows[..., 0] - rows[..., 1] + rows[..., 0] ** 3 / 3.0
        return out


@dataclass
class SynNBCF21ParametricOscillator(SynNBCF2Oscillator):
    """F2 with bounded non-Gaussian multiplicative restoring-force uncertainty."""

    mu_mean: float = 0.92
    mu_std: float = 0.10
    mu_min: float = 0.65
    mu_max: float = 1.15

    def sample_training_initial_states(self, rng: np.random.Generator, n: int) -> np.ndarray:
        # The certificate domain is larger than the safety initial set. Include
        # interior exploration starts to avoid unconstrained score extrapolation.
        initial_count = max(1, int(np.ceil(0.25 * n)))
        exploration_count = max(0, n - initial_count)
        starts = [self.sample_initial_states(rng, initial_count)]
        if exploration_count:
            interior = tuple(
                (lo + 0.05 * (hi - lo), hi - 0.05 * (hi - lo))
                for lo, hi in self.verify_box
            )
            starts.append(_sample_box(rng, interior, exploration_count))
        return np.vstack(starts)

    def restoring_multiplier(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        rows = np.asarray(x, dtype=float)
        span = max(float(self.mu_max - self.mu_min), 1e-9)
        mean = np.clip((self.mu_mean - self.mu_min) / span, 1e-3, 1.0 - 1e-3)
        std = max(float(self.mu_std) / span, 1e-4)
        concentration = max(mean * (1.0 - mean) / std**2 - 1.0, 2.0)
        return self.mu_min + span * rng.beta(mean * concentration, (1.0 - mean) * concentration, size=rows.shape[:-1])

    def actual_step(self, x: np.ndarray, u: np.ndarray, rng: np.random.Generator, step: int = 0) -> np.ndarray:
        rows = np.asarray(x, dtype=float)
        drift = self.exact_derivative(rows, u)
        restoring_force = -rows[..., 0] + rows[..., 0] ** 3 / 3.0
        drift[..., 1] = -rows[..., 1] + self.restoring_multiplier(rows, rng) * restoring_force
        return _clip(rows + self.dt * drift, self.verify_box)

    def accepts_training_transition(self, x: np.ndarray, nominal_next: np.ndarray, next_state: np.ndarray) -> bool:
        # Clipping at a stopped-domain boundary is not a physical residual.
        # Keep only interior transitions when identifying the latent mu law.
        margin = 0.03
        for point in (x, nominal_next, next_state):
            values = np.asarray(point, dtype=float)
            for coordinate, (lo, hi) in enumerate(self.verify_box):
                if values[coordinate] <= lo + margin or values[coordinate] >= hi - margin:
                    return False
        return True


def build_benchmark_system(
    name: str,
    dt: float,
    process_noise: float,
    *,
    carla_friction_enabled: bool = True,
    carla_friction_mean: float = 0.82,
    carla_friction_std: float = 0.12,
    carla_friction_min: float = 0.45,
    carla_friction_max: float = 1.12,
    carla_friction_speed_gain: float = 0.10,
    carla_friction_distance_gain: float = 0.08,
    carla_absorbing_stop: bool = True,
    carla_stop_speed: float = 0.5,
) -> BenchmarkSystem:
    key = name.strip().lower()
    if key in {"xplane11", "xplane", "trajectory_tracking", "trajectory-tracking"}:
        return XPlane11TrajectoryTracking(
            name="xplane11",
            dt=dt,
            state_dim=2,
            control_dim=1,
            process_noise=process_noise,
            verify_box=((-1.1, 1.1), (-3.0, 3.0)),
            initial_box=((0.88, 1.1), (-3.0, -2.4)),
            unsafe_box=((-1.1, 0.0), (-3.0, -1.0)),
            control_bounds=((-0.5, 0.5),),
        )

    if key in {"carla", "carla_braking", "emergency_braking", "braking", "carla1", "carla_multiplicative", "carla_multiplicative_friction"}:
        use_multiplicative_friction = key in {"carla1", "carla_multiplicative", "carla_multiplicative_friction"}
        return CarlaEmergencyBraking(
            name="carla1_multiplicative_friction" if use_multiplicative_friction else "carla_braking",
            dt=dt,
            state_dim=2,
            control_dim=1,
            process_noise=process_noise,
            verify_box=((5.0, 16.0), (0.0, 3.0)),
            initial_box=((15.0, 16.0), (2.5, 3.0)),
            unsafe_box=((5.0, 6.0), (0.5, 3.0)),
            control_bounds=((0.0, 3.0),),
            barrier_box=((5.0, 16.0), (float(carla_stop_speed), 3.0)),
            max_brake=3.0,
            # Benchmark names define the experimental protocol. `carla` keeps
            # the original additive process-noise setup; `carla1` uses mu*a.
            friction_enabled=use_multiplicative_friction and carla_friction_enabled,
            friction_mean=carla_friction_mean,
            friction_std=carla_friction_std,
            friction_min=carla_friction_min,
            friction_max=carla_friction_max,
            friction_speed_gain=carla_friction_speed_gain,
            friction_distance_gain=carla_friction_distance_gain,
            stop_speed=carla_stop_speed,
            absorbing_stop=carla_absorbing_stop,
        )

    if key in {"prajnacdc2004", "cdc2004", "cdc04", "prajna", "barrier2004"}:
        return PrajnaCdc2004Continuous(
            name="prajna_cdc2004_continuous",
            dt=dt,
            state_dim=2,
            control_dim=1,
            process_noise=process_noise,
            verify_box=((-3.0, 3.0), (-3.0, 3.0)),
            initial_box=((-2.1, -1.9), (-0.1, 0.1)),
            unsafe_box=((-3.0, 3.0), (2.25, 3.0)),
            control_bounds=((0.0, 0.0),),
        )

    if key in {"synncb_f2", "synncb-f2", "f2", "synf2"}:
        return SynNBCF2Oscillator(
            name="synncb_f2_oscillator",
            dt=dt,
            state_dim=2,
            control_dim=1,
            process_noise=process_noise,
            verify_box=((-3.5, 2.0), (-2.0, 1.0)),
            initial_box=((1.0, 2.0), (-0.5, 0.5)),
            unsafe_box=((-1.4, -0.6), (-1.4, -0.6)),
            control_bounds=((0.0, 0.0),),
        )

    if key in {"synncb_f21", "synncb-f21", "f21", "synf21"}:
        return SynNBCF21ParametricOscillator(
            name="synncb_f21_parametric_oscillator",
            dt=dt,
            state_dim=2,
            control_dim=1,
            process_noise=process_noise,
            verify_box=((-3.5, 2.0), (-2.0, 1.0)),
            initial_box=((1.0, 2.0), (-0.5, 0.5)),
            unsafe_box=((-1.4, -0.6), (-1.4, -0.6)),
            control_bounds=((0.0, 0.0),),
        )

    return ToySystem(
        name="toy",
        dt=dt,
        state_dim=2,
        control_dim=1,
        process_noise=process_noise,
        verify_box=((-2.5, 2.5), (-2.5, 2.5)),
        initial_box=((-0.45, 0.45), (-0.45, 0.45)),
        unsafe_box=((1.8, 2.5), (1.8, 2.5)),
        control_bounds=((-1.5, 1.5),),
    )


def build_configured_benchmark_system(config) -> BenchmarkSystem:
    """Construct a benchmark while preserving benchmark-specific settings."""
    return build_benchmark_system(
        config.benchmark,
        config.dt,
        config.process_noise,
        carla_friction_enabled=config.carla_friction_enabled,
        carla_friction_mean=config.carla_friction_mean,
        carla_friction_std=config.carla_friction_std,
        carla_friction_min=config.carla_friction_min,
        carla_friction_max=config.carla_friction_max,
        carla_friction_speed_gain=config.carla_friction_speed_gain,
        carla_friction_distance_gain=config.carla_friction_distance_gain,
        carla_absorbing_stop=config.carla_absorbing_stop,
        carla_stop_speed=config.carla_stop_speed,
    )


def generate_trajectories(
    system: BenchmarkSystem,
    n_trajectories: int,
    horizon: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.zeros((n_trajectories, horizon + 1, system.state_dim), dtype=float)
    u = np.zeros((n_trajectories, horizon, system.control_dim), dtype=float)
    x[:, 0, :] = system.sample_training_initial_states(rng, n_trajectories)

    for t in range(horizon):
        action = batch_control_policy(system, x[:, t], step=t)
        action = system.clamp_control(action + rng.normal(0.0, 0.02, size=action.shape))
        u[:, t] = action
        x[:, t + 1] = system.actual_step(x[:, t], action, rng, step=t)

    return {"x": x, "u": u}


def build_samples(
    trajectories: dict[str, np.ndarray],
    system: BenchmarkSystem,
    history: int,
) -> dict[str, np.ndarray]:
    x_traj = trajectories["x"]
    u_traj = trajectories["u"]
    horizon = u_traj.shape[1]

    histories: list[np.ndarray] = []
    x_now: list[np.ndarray] = []
    u_now: list[np.ndarray] = []
    x_next: list[np.ndarray] = []
    nominal_next: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    trajectory_ids: list[int] = []

    for i in range(x_traj.shape[0]):
        pair = np.concatenate([x_traj[i, :horizon], u_traj[i]], axis=-1)
        for t in range(history - 1, horizon):
            hist = pair[t - history + 1 : t + 1]
            x_t = x_traj[i, t]
            u_t = u_traj[i, t]
            nom = system.nominal_step(x_t, u_t)
            nxt = x_traj[i, t + 1]
            if not system.accepts_training_transition(x_t, nom, nxt):
                continue
            histories.append(hist)
            x_now.append(x_t)
            u_now.append(u_t)
            x_next.append(nxt)
            nominal_next.append(nom)
            residuals.append(nxt - nom)
            trajectory_ids.append(i)

    return {
        "history": np.asarray(histories, dtype=float),
        "x": np.asarray(x_now, dtype=float),
        "u": np.asarray(u_now, dtype=float),
        "x_next": np.asarray(x_next, dtype=float),
        "nominal_next": np.asarray(nominal_next, dtype=float),
        # Euler-Maruyama stochastic increment. This is retained separately
        # from the legacy residual name so diffusion-only experiments do not
        # imply that it is an additional physical drift.
        "stochastic_increment": np.asarray(residuals, dtype=float),
        "residual": np.asarray(residuals, dtype=float),
        "trajectory_id": np.asarray(trajectory_ids, dtype=int),
    }
