from __future__ import annotations

from itertools import product
from math import comb
from numbers import Number
from typing import Any

import numpy as np


Poly = dict[tuple[int, ...], Any]


def clean_poly(poly: Poly) -> Poly:
    return {exp: coeff for exp, coeff in poly.items() if coeff is not None}


def constant_poly(n_vars: int, value: Any) -> Poly:
    return {(0,) * n_vars: value}


def monomial_poly(exp: tuple[int, ...], coeff: Any) -> Poly:
    return {tuple(int(v) for v in exp): coeff}


def add_poly(*polys: Poly) -> Poly:
    out: Poly = {}
    for poly in polys:
        for exp, coeff in poly.items():
            out[exp] = out.get(exp, 0.0) + coeff
    return clean_poly(out)


def scale_poly(poly: Poly, scalar: Any) -> Poly:
    return {exp: scalar * coeff for exp, coeff in poly.items()}


def mul_poly(a: Poly, b: Poly) -> Poly:
    out: Poly = {}
    for exp_a, coeff_a in a.items():
        for exp_b, coeff_b in b.items():
            exp = tuple(x + y for x, y in zip(exp_a, exp_b))
            out[exp] = out.get(exp, 0.0) + coeff_a * coeff_b
    return clean_poly(out)


def affine_substitute_poly(poly: Poly, center: np.ndarray, scale: np.ndarray) -> Poly:
    """Substitute x_i = center_i + scale_i * y_i in a polynomial."""

    center_arr = np.asarray(center, dtype=float)
    scale_arr = np.asarray(scale, dtype=float)
    n_vars = int(center_arr.shape[0])
    out: Poly = {}
    for exp, coeff in poly.items():
        term = constant_poly(n_vars, coeff)
        for var, power in enumerate(exp):
            if power:
                term = mul_poly(term, _affine_power_poly(n_vars, var, power, center_arr[var], scale_arr[var]))
        out = add_poly(out, term)
    return clean_poly(out)


def evaluate_poly(poly: Poly, x: np.ndarray) -> np.ndarray:
    x_arr = np.atleast_2d(np.asarray(x, dtype=float))
    values = np.zeros(x_arr.shape[0], dtype=float)
    for exp, coeff in poly.items():
        term = np.ones(x_arr.shape[0], dtype=float) * float(coeff)
        for var, power in enumerate(exp):
            if power:
                term *= x_arr[:, var] ** power
        values += term
    return values


def _affine_power_poly(n_vars: int, var: int, power: int, center: float, scale: float) -> Poly:
    out: Poly = {}
    for k in range(power + 1):
        exp = [0] * n_vars
        exp[var] = k
        out[tuple(exp)] = float(comb(power, k)) * (float(center) ** (power - k)) * (float(scale) ** k)
    return clean_poly(out)


def derivative_poly(poly: Poly, var: int) -> Poly:
    out: Poly = {}
    for exp, coeff in poly.items():
        power = exp[var]
        if power == 0:
            continue
        reduced = list(exp)
        reduced[var] -= 1
        out[tuple(reduced)] = out.get(tuple(reduced), 0.0) + power * coeff
    return clean_poly(out)


def degree_poly(poly: Poly) -> int:
    if not poly:
        return 0
    return max(sum(exp) for exp in poly)


def all_exponents(n_vars: int, degree: int) -> list[tuple[int, ...]]:
    exps: list[tuple[int, ...]] = []
    for exp in product(range(degree + 1), repeat=n_vars):
        if sum(exp) <= degree:
            exps.append(tuple(int(v) for v in exp))
    exps.sort(key=lambda e: (sum(e), e))
    return exps


def vector_polynomial_from_coeffs(
    exponents: list[tuple[int, ...]],
    coeffs: np.ndarray,
) -> list[Poly]:
    coeff_arr = np.asarray(coeffs, dtype=float)
    dim = coeff_arr.shape[1]
    out: list[Poly] = []
    for j in range(dim):
        poly: Poly = {}
        for exp, value in zip(exponents, coeff_arr[:, j]):
            if abs(float(value)) > 1e-14:
                poly[exp] = float(value)
        out.append(poly)
    return out


def fit_vector_polynomial(
    x: np.ndarray,
    y: np.ndarray,
    degree: int,
) -> tuple[list[tuple[int, ...]], np.ndarray, float]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    exponents = all_exponents(x_arr.shape[1], degree)
    phi = np.ones((x_arr.shape[0], len(exponents)), dtype=float)
    for j, exp in enumerate(exponents):
        for k, power in enumerate(exp):
            if power:
                phi[:, j] *= x_arr[:, k] ** power
    coeffs, *_ = np.linalg.lstsq(phi, y_arr, rcond=None)
    pred = phi @ coeffs
    fit_error = float(np.max(np.linalg.norm(pred - y_arr, axis=1)))
    return exponents, coeffs, fit_error


def box_domain_polynomials(box: tuple[tuple[float, float], ...]) -> list[Poly]:
    n_vars = len(box)
    polys: list[Poly] = []
    for i, (lo, hi) in enumerate(box):
        exp2 = [0] * n_vars
        exp1 = [0] * n_vars
        exp2[i] = 2
        exp1[i] = 1
        polys.append(
            {
                tuple(exp2): -1.0,
                tuple(exp1): float(lo + hi),
                (0,) * n_vars: float(-lo * hi),
            }
        )
    return polys


def gram_sos_poly(z_exponents: list[tuple[int, ...]], gram: Any) -> Poly:
    out: Poly = {}
    for i, exp_i in enumerate(z_exponents):
        for j, exp_j in enumerate(z_exponents):
            exp = tuple(a + b for a, b in zip(exp_i, exp_j))
            out[exp] = out.get(exp, 0.0) + gram[i, j]
    return clean_poly(out)


def putinar_sos_constraints(
    cp: Any,
    poly: Poly,
    box: tuple[tuple[float, float], ...],
    relaxation_degree: int | None = None,
    name: str = "sos",
) -> tuple[list[Any], list[Any]]:
    return putinar_sos_constraints_on_generators(
        cp,
        poly,
        box_domain_polynomials(box),
        relaxation_degree=relaxation_degree,
        name=name,
    )


def putinar_sos_constraints_on_generators(
    cp: Any,
    poly: Poly,
    generators: list[Poly],
    relaxation_degree: int | None = None,
    name: str = "sos",
) -> tuple[list[Any], list[Any]]:
    """Constrain ``poly`` to be nonnegative on a semialgebraic set.

    ``generators`` contains the polynomials g_i with g_i(x) >= 0.
    The existing box helper is retained as a thin wrapper for compatibility.
    """
    if not generators:
        raise ValueError("at least one domain generator is required")
    n_vars = len(next(iter(generators[0])))
    poly_degree = degree_poly(poly)
    degree = max(poly_degree, 0 if relaxation_degree is None else int(relaxation_degree))
    if degree % 2:
        degree += 1

    constraints: list[Any] = []
    gram_matrices: list[Any] = []

    z0 = all_exponents(n_vars, degree // 2)
    q0 = cp.Variable((len(z0), len(z0)), symmetric=True, name=f"{name}_Q0")
    constraints.append(q0 >> 0)
    gram_matrices.append(q0)
    rhs = gram_sos_poly(z0, q0)

    for idx, g_poly in enumerate(generators):
        multiplier_degree = max(0, degree - degree_poly(g_poly))
        z_multiplier = all_exponents(n_vars, multiplier_degree // 2)
        qi = cp.Variable((len(z_multiplier), len(z_multiplier)), symmetric=True, name=f"{name}_Q{idx + 1}")
        constraints.append(qi >> 0)
        gram_matrices.append(qi)
        rhs = add_poly(rhs, mul_poly(g_poly, gram_sos_poly(z_multiplier, qi)))

    for exp in all_exponents(n_vars, degree):
        lhs = poly.get(exp, 0.0)
        rhs_coeff = rhs.get(exp, 0.0)
        if isinstance(lhs, Number) and isinstance(rhs_coeff, Number):
            if abs(float(lhs) - float(rhs_coeff)) > 1e-10:
                constraints.append(cp.Constant(float(lhs)) == cp.Constant(float(rhs_coeff)))
            continue
        constraints.append(lhs == rhs_coeff)

    return constraints, gram_matrices


def barrier_polynomial(coeffs: Any, exponents: list[tuple[int, ...]]) -> Poly:
    return {exp: coeffs[i] for i, exp in enumerate(exponents)}


def generator_polynomial(
    barrier_poly: Poly,
    drift_polys: list[Poly],
    diffusion_covariance: np.ndarray,
) -> Poly:
    n_vars = len(drift_polys)
    first_order = constant_poly(n_vars, 0.0)
    for i, drift_i in enumerate(drift_polys):
        first_order = add_poly(first_order, mul_poly(derivative_poly(barrier_poly, i), drift_i))

    second_order = constant_poly(n_vars, 0.0)
    gbar = np.asarray(diffusion_covariance, dtype=float)
    for i in range(n_vars):
        for j in range(n_vars):
            hess_ij = derivative_poly(derivative_poly(barrier_poly, i), j)
            if hess_ij:
                second_order = add_poly(second_order, scale_poly(hess_ij, 0.5 * float(gbar[i, j])))
    return add_poly(first_order, second_order)
