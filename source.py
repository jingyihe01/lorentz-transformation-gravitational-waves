"""Core functions for Lorentz-boosting gravitational-wave polarizations.

This module contains reusable numerical functions only.  The worked example,
plots, and discussion are kept in ``lorentz_boost_demo.ipynb``.

The implementation follows X. He, X. Liu, and Z. Cao,
"Lorentz transformation of three dimensional gravitational wave tensor",
arXiv:2302.07532 (2023).

All velocities are dimensionless beta = v/c and must satisfy |beta| < 1.
All angles are in radians.
"""

from __future__ import annotations

import numpy as np


# Physical constants (SI)
G = 6.67430e-11
C = 299_792_458.0
M_SUN = 1.98847e30
PC = 3.085677581491367e16
KPC = 1.0e3 * PC


def spherical_triad(theta: float, phi: float):
    """Return the orthonormal spherical triad (r_hat, theta_hat, phi_hat)."""
    st, ct = np.sin(theta), np.cos(theta)
    sp, cp = np.sin(phi), np.cos(phi)

    r_hat = np.array([st * cp, st * sp, ct], dtype=float)
    theta_hat = np.array([ct * cp, ct * sp, -st], dtype=float)
    phi_hat = np.array([-sp, cp, 0.0], dtype=float)
    return r_hat, theta_hat, phi_hat


def polarization_tensors(theta: float, phi: float):
    """Return the propagation direction and the plus/cross polarization tensors.

    The definitions are

        e_plus  = theta_hat (x) theta_hat - phi_hat (x) phi_hat,
        e_cross = theta_hat (x) phi_hat + phi_hat (x) theta_hat.

    Here ``(x)`` denotes an outer product.
    """
    r_hat, theta_hat, phi_hat = spherical_triad(theta, phi)
    e_plus = np.outer(theta_hat, theta_hat) - np.outer(phi_hat, phi_hat)
    e_cross = np.outer(theta_hat, phi_hat) + np.outer(phi_hat, theta_hat)
    return r_hat, e_plus, e_cross


def tensor_from_polarizations(h_plus, h_cross, e_plus, e_cross):
    """Construct h_ij = h_plus e_plus_ij + h_cross e_cross_ij."""
    hp = np.asarray(h_plus)
    hx = np.asarray(h_cross)

    if hp.shape != hx.shape:
        raise ValueError("h_plus and h_cross must have the same shape.")

    if hp.ndim == 0:
        return hp * e_plus + hx * e_cross
    if hp.ndim == 1:
        return (
            hp[:, None, None] * e_plus[None, :, :]
            + hx[:, None, None] * e_cross[None, :, :]
        )
    raise ValueError("h_plus and h_cross must be scalars or one-dimensional arrays.")


def project_polarizations(h_ij, e_plus, e_cross):
    """Project a tensor onto the plus and cross polarization tensors.

    Because e_plus:e_plus = e_cross:e_cross = 2,

        h_plus  = (1/2) h_ij e_plus^ij,
        h_cross = (1/2) h_ij e_cross^ij.
    """
    h = np.asarray(h_ij)
    if h.shape == (3, 3):
        h_plus = 0.5 * np.einsum("ij,ij->", h, e_plus)
        h_cross = 0.5 * np.einsum("ij,ij->", h, e_cross)
        return h_plus, h_cross
    if h.ndim == 3 and h.shape[1:] == (3, 3):
        h_plus = 0.5 * np.einsum("tij,ij->t", h, e_plus)
        h_cross = 0.5 * np.einsum("tij,ij->t", h, e_cross)
        return h_plus, h_cross
    raise ValueError("h_ij must have shape (3, 3) or (N, 3, 3).")


def circular_binary_quadrupole_ddot(t, m1_kg, m2_kg, omega_orb):
    """Return the second time derivative of the circular-binary quadrupole.

    The orbit is in the x-y plane, with orbital angular momentum along +z.
    Kepler's law is used to calculate the binary separation.
    """
    t = np.asarray(t, dtype=float)
    if t.ndim != 1:
        raise ValueError("t must be a one-dimensional array.")

    total_mass = m1_kg + m2_kg
    reduced_mass = m1_kg * m2_kg / total_mass
    separation = (G * total_mass / omega_orb**2) ** (1.0 / 3.0)

    phase_2 = 2.0 * omega_orb * t
    amplitude = 2.0 * reduced_mass * separation**2 * omega_orb**2

    q_ddot = np.zeros((t.size, 3, 3), dtype=float)
    q_ddot[:, 0, 0] = -amplitude * np.cos(phase_2)
    q_ddot[:, 1, 1] = amplitude * np.cos(phase_2)
    q_ddot[:, 0, 1] = -amplitude * np.sin(phase_2)
    q_ddot[:, 1, 0] = q_ddot[:, 0, 1]
    return q_ddot


def source_waveform_from_quadrupole(
    t,
    m1_kg,
    m2_kg,
    omega_orb,
    luminosity_distance_m,
    iota,
):
    """Generate source-frame h_plus and h_cross for a circular binary.

    The source z-axis is aligned with the orbital angular momentum.  The
    propagation direction is chosen as (theta, phi) = (iota, 0).
    """
    q_ddot = circular_binary_quadrupole_ddot(t, m1_kg, m2_kg, omega_orb)
    h_raw = 2.0 * G * q_ddot / (luminosity_distance_m * C**4)

    r_hat, e_plus, e_cross = polarization_tensors(iota, 0.0)
    h_plus, h_cross = project_polarizations(h_raw, e_plus, e_cross)
    h_tt = tensor_from_polarizations(h_plus, h_cross, e_plus, e_cross)
    return h_plus, h_cross, h_tt, r_hat


def gamma_from_beta(beta):
    """Return the Lorentz factor gamma for the dimensionless velocity beta."""
    beta = np.asarray(beta, dtype=float)
    if beta.shape != (3,):
        raise ValueError("beta must be a three-component vector.")

    beta_squared = float(np.dot(beta, beta))
    if beta_squared >= 1.0:
        raise ValueError("The boost must satisfy |beta| < 1.")
    return 1.0 / np.sqrt(1.0 - beta_squared)


def aberrate_direction(r_hat, beta):
    """Apply the exact aberration formula (paper Eq. 79)."""
    r_hat = np.asarray(r_hat, dtype=float)
    beta = np.asarray(beta, dtype=float)
    beta_magnitude = np.linalg.norm(beta)

    if r_hat.shape != (3,) or beta.shape != (3,):
        raise ValueError("r_hat and beta must be three-component vectors.")
    if beta_magnitude < 1.0e-15:
        return r_hat.copy()

    gamma = gamma_from_beta(beta)
    beta_hat = beta / beta_magnitude
    r_dot_beta = float(np.dot(r_hat, beta))
    r_dot_beta_hat = float(np.dot(r_hat, beta_hat))
    denominator = 1.0 - r_dot_beta

    r_prime = (
        ((r_dot_beta_hat - beta_magnitude) / denominator) * beta_hat
        + (r_hat - r_dot_beta_hat * beta_hat) / (gamma * denominator)
    )
    return r_prime / np.linalg.norm(r_prime)


def lorentz_transform_gw_tensor(h_ij, r_hat, beta):
    """Apply the exact 3D GW-tensor Lorentz transformation (paper Eq. 59).

    ``h_ij`` may be a single (3, 3) tensor or an array with shape (N, 3, 3).
    The output has the same shape as the input.
    """
    h = np.asarray(h_ij, dtype=float)
    r = np.asarray(r_hat, dtype=float)
    v = np.asarray(beta, dtype=float)

    gamma = gamma_from_beta(v)
    r_dot_v = float(np.dot(r, v))
    denominator = 1.0 - r_dot_v
    if abs(denominator) < 1.0e-14:
        raise ValueError("1 - r_hat dot beta is too close to zero.")

    coefficient = gamma / (1.0 + gamma)
    b_vector = r - coefficient * v
    bracket = (
        np.outer(r, r)
        - coefficient * (np.outer(r, v) + np.outer(v, r))
        + coefficient**2 * np.outer(v, v)
    )

    if h.shape == (3, 3):
        h_v = h @ v
        v_h_v = float(v @ h @ v)
        return (
            h
            + (v_h_v / denominator**2) * bracket
            + np.outer(b_vector, h_v) / denominator
            + np.outer(h_v, b_vector) / denominator
        )

    if h.ndim == 3 and h.shape[1:] == (3, 3):
        h_v = np.einsum("tij,j->ti", h, v)
        v_h_v = np.einsum("i,tij,j->t", v, h, v)
        return (
            h
            + (v_h_v / denominator**2)[:, None, None] * bracket[None, :, :]
            + np.einsum("i,tj->tij", b_vector, h_v) / denominator
            + np.einsum("ti,j->tij", h_v, b_vector) / denominator
        )

    raise ValueError("h_ij must have shape (3, 3) or (N, 3, 3).")


def beta_XYZ_to_xyz(beta_XYZ, theta, phi, iota):
    """Rotate beta from the detector basis to the source basis (paper Eq. 143)."""
    delta = iota - theta
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)
    cos_delta, sin_delta = np.cos(delta), np.sin(delta)

    rotation = np.array(
        [
            [cos_phi, sin_phi, 0.0],
            [-sin_phi * cos_delta, cos_phi * cos_delta, sin_delta],
            [sin_phi * sin_delta, -cos_phi * sin_delta, cos_delta],
        ]
    )
    return rotation @ np.asarray(beta_XYZ, dtype=float)


def beta_xyz_to_XYZ(beta_xyz, theta, phi, iota):
    """Rotate beta from the source basis to the detector basis (paper Eq. 147)."""
    delta = iota - theta
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)
    cos_delta, sin_delta = np.cos(delta), np.sin(delta)

    inverse_rotation = np.array(
        [
            [cos_phi, -sin_phi * cos_delta, sin_phi * sin_delta],
            [sin_phi, cos_phi * cos_delta, -cos_phi * sin_delta],
            [0.0, sin_delta, cos_delta],
        ]
    )
    return inverse_rotation @ np.asarray(beta_xyz, dtype=float)


def source_h_to_detector_Hprime(
    h_plus,
    h_cross,
    iota,
    theta,
    phi,
    psi,
    beta_XYZ,
):
    """Transform source polarizations into boosted detector polarizations.

    Parameters
    ----------
    h_plus, h_cross : array-like
        Source-frame plus and cross polarization time series.
    iota : float
        Binary inclination angle.  It is retained in the interface because it
        specifies the source geometry used to generate the input waveform.
    theta, phi : float
        Detector-frame sky-location angles of the source.
    psi : float
        Polarization angle.
    beta_XYZ : array-like, shape (3,)
        Source velocity divided by c in the detector Cartesian basis.

    Returns
    -------
    dict
        Polarizations before/after the boost, transformed sky position,
        Doppler factor, tensors, and transformed source direction.
    """
    del iota  # Geometry is already encoded in the supplied source waveform.

    hp = np.asarray(h_plus, dtype=float)
    hx = np.asarray(h_cross, dtype=float)
    beta_XYZ = np.asarray(beta_XYZ, dtype=float)

    if hp.shape != hx.shape:
        raise ValueError("h_plus and h_cross must have the same shape.")

    # Eqs. (129)-(130): rotate into the detector polarization basis.
    cos_2psi = np.cos(2.0 * psi)
    sin_2psi = np.sin(2.0 * psi)
    H_plus = hp * cos_2psi + hx * sin_2psi
    H_cross = hx * cos_2psi - hp * sin_2psi

    # R_hat points from detector to source; the GW propagates along -R_hat.
    R_hat, E_plus, E_cross = polarization_tensors(theta, phi)
    propagation_direction = -R_hat

    H_ij = tensor_from_polarizations(H_plus, H_cross, E_plus, E_cross)
    H_ij_prime = lorentz_transform_gw_tensor(
        H_ij, propagation_direction, beta_XYZ
    )

    propagation_direction_prime = aberrate_direction(
        propagation_direction, beta_XYZ
    )
    R_hat_prime = -propagation_direction_prime

    theta_prime = np.arccos(np.clip(R_hat_prime[2], -1.0, 1.0))
    phi_prime = np.mod(
        np.arctan2(R_hat_prime[1], R_hat_prime[0]), 2.0 * np.pi
    )

    _, E_plus_prime, E_cross_prime = polarization_tensors(
        theta_prime, phi_prime
    )
    H_plus_prime, H_cross_prime = project_polarizations(
        H_ij_prime, E_plus_prime, E_cross_prime
    )

    gamma = gamma_from_beta(beta_XYZ)
    doppler_factor = 1.0 / (
        gamma * (1.0 - np.dot(beta_XYZ, propagation_direction))
    )

    return {
        "Hplus": H_plus,
        "Hcross": H_cross,
        "Hplus_prime": H_plus_prime,
        "Hcross_prime": H_cross_prime,
        "Theta_prime": theta_prime,
        "Phi_prime": phi_prime,
        "k": doppler_factor,
        "Hij": H_ij,
        "Hij_prime": H_ij_prime,
        "Rhat_prime": R_hat_prime,
    }


def lambda_from_tensor_method(theta, phi, beta):
    """Calculate the spin-2 phase lambda using the tensor method.

    A unit plus-polarized tensor is boosted and projected on the aberrated
    basis.  The complex result satisfies exp(-2 i lambda) = h_plus' - i h_cross'.
    """
    r_hat, e_plus, _ = polarization_tensors(theta, phi)
    test_tensor_prime = lorentz_transform_gw_tensor(e_plus, r_hat, beta)
    r_hat_prime = aberrate_direction(r_hat, beta)

    theta_prime = np.arccos(np.clip(r_hat_prime[2], -1.0, 1.0))
    phi_prime = np.arctan2(r_hat_prime[1], r_hat_prime[0])
    _, e_plus_prime, e_cross_prime = polarization_tensors(
        theta_prime, phi_prime
    )

    h_plus_prime, h_cross_prime = project_polarizations(
        test_tensor_prime, e_plus_prime, e_cross_prime
    )
    phase_factor = h_plus_prime - 1j * h_cross_prime
    phase_factor /= abs(phase_factor)
    return -0.5 * np.angle(phase_factor)


__all__ = [
    "G",
    "C",
    "M_SUN",
    "PC",
    "KPC",
    "spherical_triad",
    "polarization_tensors",
    "tensor_from_polarizations",
    "project_polarizations",
    "circular_binary_quadrupole_ddot",
    "source_waveform_from_quadrupole",
    "gamma_from_beta",
    "aberrate_direction",
    "lorentz_transform_gw_tensor",
    "beta_XYZ_to_xyz",
    "beta_xyz_to_XYZ",
    "source_h_to_detector_Hprime",
    "lambda_from_tensor_method",
]
