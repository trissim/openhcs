"""SE(3) refinement with certified RMSD convergence guarantees.

Implements two approaches for theorem-backed optimization budgets:

  Approach A (observed): Run any optimizer, observe energy trajectory,
    certify post-hoc via SE(3) Hessian + Jacobian bridge.

  Approach B (certified_gd): Run standard gradient descent in axis-angle
    parameterization with theorem-derived step size and budget.

Both share the SE(3) spectral certificate computation (Hessian eigenvalues
+ kinematics Jacobian singular values) and the Jacobian bridge theorem.

Lean: EnergyRMSDConvergence.lean — CertifiedQuadraticBasin,
      CertifiedLinearEnergyConvergence,
      rmsd_target_of_canonicalIterationBudgetFromLocalCertificates,
      rmsd_target_of_canonicalIterationBudgetFromGradientDescentDynamics.
New:  SE3JacobianBridge.lean — parameterSpace_quadraticBasin_transfers_to_coordSpace.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from dq_dock_engine.docking.placement import _apply_single_pose
from dq_dock_engine.docking.scoring import _score_certified_lj
from dq_dock_engine.docking_config import RefinementCertificationMode


@dataclass(frozen=True)
class SE3SpectralCertificate:
    """Hessian eigenvalues in SE(3) parameter space + Jacobian bridge.

    Lean: parameterSpace_quadraticBasin_transfers_to_coordSpace.
    """

    lmin_param: float
    lmax_param: float
    sigma_max_sq: float
    mu_coord: float  # lmin_param / sigma_max_sq


@dataclass(frozen=True)
class RefinementCertificate:
    """Combined certificate for theorem-backed n_opt_steps."""

    spectral: SE3SpectralCertificate
    q: float  # contraction rate (observed for A, derived for B)
    initial_gap: float
    target_rmsd: float
    n_steps: int  # certified budget
    mode: RefinementCertificationMode


# ---------------------------------------------------------------------------
# Axis-angle ↔ quaternion conversion
# ---------------------------------------------------------------------------


def _quaternion_to_axis_angle(q: jnp.ndarray) -> jnp.ndarray:
    """Convert quaternion [w, x, y, z] to axis-angle rotation vector (3,).

    Handles the small-angle case (angle ≈ 0) without singularity.
    """
    w = q[0]
    xyz = q[1:]
    sin_half = jnp.linalg.norm(xyz)
    angle = 2.0 * jnp.arctan2(sin_half, jnp.abs(w))
    # When sin_half ≈ 0, axis is arbitrary — return zero rotation vector
    axis = jnp.where(sin_half > 1e-8, xyz / sin_half, jnp.zeros(3))
    # Ensure angle is in [0, π] by flipping if w < 0
    angle = jnp.where(w < 0, 2.0 * jnp.pi - angle, angle)
    return axis * angle


def _axis_angle_to_quaternion(rotvec: jnp.ndarray) -> jnp.ndarray:
    """Convert axis-angle rotation vector (3,) to quaternion [w, x, y, z]."""
    angle_sq = jnp.dot(rotvec, rotvec)

    def _small_angle(rv: jnp.ndarray) -> jnp.ndarray:
        # Taylor expansion around zero with finite derivatives.
        ang_sq = jnp.dot(rv, rv)
        scale = 0.5 - ang_sq / 48.0
        return jnp.array(
            [
                1.0 - ang_sq / 8.0,
                rv[0] * scale,
                rv[1] * scale,
                rv[2] * scale,
            ]
        )

    def _regular_angle(rv: jnp.ndarray) -> jnp.ndarray:
        angle = jnp.sqrt(jnp.dot(rv, rv))
        half = angle / 2.0
        scale = jnp.sin(half) / angle
        return jnp.array(
            [
                jnp.cos(half),
                rv[0] * scale,
                rv[1] * scale,
                rv[2] * scale,
            ]
        )

    return jax.lax.cond(angle_sq < 1e-12, _small_angle, _regular_angle, rotvec)


def _pose_to_se3_params(
    translation: jnp.ndarray, quaternion: jnp.ndarray
) -> jnp.ndarray:
    """Convert (translation, quaternion) to 6D SE(3) parameter vector."""
    q_norm = quaternion / jnp.linalg.norm(quaternion)
    rotvec = _quaternion_to_axis_angle(q_norm)
    return jnp.concatenate([translation, rotvec])


def _se3_params_to_pose(params: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Convert 6D SE(3) parameter vector to (translation, quaternion)."""
    t = params[:3]
    rotvec = params[3:6]
    q = _axis_angle_to_quaternion(rotvec)
    return t, q


# ---------------------------------------------------------------------------
# SE(3) energy and kinematics functions
# ---------------------------------------------------------------------------


def make_se3_energy_fn(
    base_coords: jnp.ndarray,
    receptor_coords: jnp.ndarray,
    receptor_radii: jnp.ndarray,
    ligand_radii: jnp.ndarray,
    cutoff: jnp.ndarray,
    epsilon: float,
):
    """Energy as a function of 6D SE(3) parameter vector.

    The function E: R^6 → R maps [tx, ty, tz, θx, θy, θz] to the
    certified LJ score. This is the optimizer's actual parameter space.
    """

    def energy_fn(params: jnp.ndarray) -> jnp.ndarray:
        t, q = _se3_params_to_pose(params)
        pose_coords = _apply_single_pose(base_coords, t, q)
        energy, _ = _score_certified_lj(
            receptor_coords,
            pose_coords,
            receptor_radii,
            ligand_radii,
            cutoff,
            epsilon,
        )
        return energy

    return energy_fn


def make_se3_kinematics_fn(base_coords: jnp.ndarray):
    """Kinematics K: R^6 → R^(N_atoms×3) mapping SE(3) params to atom coords."""

    def kinematics_fn(params: jnp.ndarray) -> jnp.ndarray:
        t, q = _se3_params_to_pose(params)
        return _apply_single_pose(base_coords, t, q)

    return kinematics_fn


# ---------------------------------------------------------------------------
# Spectral certificate computation
# ---------------------------------------------------------------------------


def compute_se3_spectral_certificate(
    energy_fn,
    kinematics_fn,
    optimized_params: jnp.ndarray,
) -> SE3SpectralCertificate | None:
    """Compute spectral certificate from 6×6 Hessian and kinematics Jacobian.

    Cost: ~6 reverse-mode passes for Hessian + 1 Jacobian.
    For 50ms scoring: ~0.3s per pose.

    Returns None if not in a convex basin (lmin_param ≤ 0).

    Lean: parameterSpace_quadraticBasin_transfers_to_coordSpace.
    """
    H = jax.hessian(energy_fn)(optimized_params)  # (6, 6)
    eigs = jnp.linalg.eigvalsh(H)
    lmin_param = float(eigs[0])
    lmax_param = float(eigs[-1])

    if lmin_param <= 0:
        return None  # not in a convex basin

    J = jax.jacobian(kinematics_fn)(optimized_params)  # (N_atoms, 3, 6)
    J_flat = J.reshape(-1, 6)  # (3*N_atoms, 6)
    sigma_max_sq = float(jnp.max(jnp.linalg.svdvals(J_flat)) ** 2)

    mu_coord = lmin_param / sigma_max_sq
    return SE3SpectralCertificate(
        lmin_param=lmin_param,
        lmax_param=lmax_param,
        sigma_max_sq=sigma_max_sq,
        mu_coord=mu_coord,
    )


# ---------------------------------------------------------------------------
# Iteration budget computation (shared by both approaches)
# ---------------------------------------------------------------------------


def certified_iteration_budget(
    mu_coord: float,
    q: float,
    initial_gap: float,
    target_rmsd: float,
    n_atoms: int,
) -> int:
    """Provably sufficient iteration count for the RMSD target.

    Lean: logarithmicIterationBound applied to q and μ from Jacobian bridge.

    target_gap = μ × n × eps² / 2   (Lean: targetEnergyGap)
    budget = ceil(log(initial_gap / target_gap) / log(1/q))
    """
    target_gap = mu_coord * n_atoms * target_rmsd**2 / 2.0
    if not math.isfinite(target_gap) or target_gap <= 0.0:
        return -1
    if not math.isfinite(initial_gap) or initial_gap <= 0.0:
        return -1
    if initial_gap <= target_gap:
        return 0
    if q <= 0 or q >= 1.0:
        return -1  # cannot certify: non-contractive
    ratio = initial_gap / target_gap
    if not math.isfinite(ratio) or ratio <= 0.0:
        return -1
    denom = math.log(1.0 / q)
    if not math.isfinite(denom) or denom <= 0.0:
        return -1
    numer = math.log(ratio)
    if not math.isfinite(numer):
        return -1
    return math.ceil(numer / denom)


# ---------------------------------------------------------------------------
# Approach A: Observed certification
# ---------------------------------------------------------------------------


def extract_observed_contraction_rate(
    energy_trajectory: list[float],
) -> float | None:
    """Fit q from observed energy gaps: gap(t+1) ≤ q × gap(t).

    Takes the worst-case ratio across all steps. Returns None if any step
    increases the energy gap (not monotonically converging).

    Lean: populates CertifiedOneStepEnergyContraction.step_contract.
    """
    final_energy = energy_trajectory[-1]
    gaps = [e - final_energy for e in energy_trajectory]

    q_max = 0.0
    for t in range(len(gaps) - 1):
        if gaps[t] <= 1e-12:
            continue  # already converged
        ratio = gaps[t + 1] / gaps[t]
        if ratio >= 1.0:
            return None  # energy gap increased — cannot certify
        q_max = max(q_max, ratio)
    return q_max if q_max > 0 else None


def certify_observed(
    spectral: SE3SpectralCertificate,
    energy_trajectory: list[float],
    target_rmsd: float,
    n_atoms: int,
) -> RefinementCertificate | None:
    """Approach A: certify an already-completed optimization run.

    Returns None if the energy trajectory is not monotonically converging
    or the Hessian eigenvalues indicate a non-convex basin.
    """
    q = extract_observed_contraction_rate(energy_trajectory)
    if q is None:
        return None
    initial_gap = energy_trajectory[0] - energy_trajectory[-1]
    if initial_gap <= 0:
        return None
    n_steps = certified_iteration_budget(
        spectral.mu_coord,
        q,
        initial_gap,
        target_rmsd,
        n_atoms,
    )
    if n_steps < 0:
        return None
    return RefinementCertificate(
        spectral=spectral,
        q=q,
        initial_gap=initial_gap,
        target_rmsd=target_rmsd,
        n_steps=n_steps,
        mode=RefinementCertificationMode.OBSERVED,
    )


# ---------------------------------------------------------------------------
# Approach B: Certified gradient descent
# ---------------------------------------------------------------------------


def _certified_gd_step(params: jnp.ndarray, energy_fn, alpha: float) -> jnp.ndarray:
    """One step of standard gradient descent. No clipping, no normalization.

    This IS the optimizer that CertifiedGradientDescentDynamics proves about.
    Lean: step_contract follows from α = 2/(lmin+lmax) and smooth strong convexity.
    """
    grad = jax.grad(energy_fn)(params)
    return params - alpha * grad


def _run_gd_steps(
    params: jnp.ndarray, energy_fn, alpha: float, n_steps: int
) -> jnp.ndarray:
    """Run n_steps of standard gradient descent (no trajectory recorded)."""

    def body_fn(i, p):
        return _certified_gd_step(p, energy_fn, alpha)

    return jax.lax.fori_loop(0, n_steps, body_fn, params)


def _run_gd_steps_with_trajectory(
    params: jnp.ndarray, energy_fn, alpha: float, n_steps: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run n_steps of standard GD, recording per-step energy via jax.lax.scan.

    Returns (final_params, energy_trajectory) where energy_trajectory has
    shape (n_steps + 1,) — the energy at the initial point followed by the
    energy after each step.

    Lean: the full trajectory is required by CertifiedOneStepEnergyContraction
    to extract the worst-case empirical contraction rate q.
    """

    def scan_fn(p, _):
        e = energy_fn(p)
        p_next = _certified_gd_step(p, energy_fn, alpha)
        return p_next, e

    final_params, energies = jax.lax.scan(scan_fn, params, None, length=n_steps)
    # Append the final energy to complete the trajectory
    final_energy = energy_fn(final_params)
    energy_trajectory = jnp.concatenate([energies, final_energy[None]])
    return final_params, energy_trajectory


def observe_gd_trajectory(
    initial_params: jnp.ndarray,
    energy_fn,
    kinematics_fn,
    n_steps: int,
    target_rmsd: float,
    n_atoms: int,
    alpha: float = 0.01,
) -> tuple[jnp.ndarray, RefinementCertificate | None]:
    """Approach A: run GD with fixed step size, record full energy trajectory,
    then certify post-hoc via spectral certificate + observed contraction rate.

    The key difference from Approach B is that q is *empirical* (worst-case
    ratio from the trajectory) rather than *derived* from the Hessian condition
    number. This makes Approach A applicable even when the Hessian-derived q
    is pessimistic.

    Lean: CertifiedOneStepEnergyContraction — each consecutive pair in the
    trajectory must satisfy gap(t+1) ≤ q × gap(t).
    """
    optimized, energy_trajectory = _run_gd_steps_with_trajectory(
        initial_params,
        energy_fn,
        alpha,
        n_steps,
    )

    spectral = compute_se3_spectral_certificate(energy_fn, kinematics_fn, optimized)
    if spectral is None:
        return optimized, None

    trajectory_list = [float(e) for e in energy_trajectory]
    cert = certify_observed(
        spectral=spectral,
        energy_trajectory=trajectory_list,
        target_rmsd=target_rmsd,
        n_atoms=n_atoms,
    )
    return optimized, cert


def optimize_certified_gd(
    initial_params: jnp.ndarray,
    energy_fn,
    kinematics_fn,
    target_rmsd: float,
    n_atoms: int,
    max_probe_steps: int = 10,
) -> tuple[jnp.ndarray, RefinementCertificate | None]:
    """Two-phase certified optimization (Approach B).

    Phase 1: Quick probe with small fixed step size to reach near the basin.
    Phase 2: Compute Hessian, derive α and budget T, run T steps of standard GD.

    Lean: CertifiedGradientDescentDynamics — q = (M-μ)/(M+μ) with
          α = 2/(μ+M) is the optimal fixed step size for smooth strongly
          convex functions.
    """
    # Phase 1: probe
    probed = _run_gd_steps(
        initial_params, energy_fn, alpha=0.01, n_steps=max_probe_steps
    )

    # Compute spectral certificate at probed point
    spectral = compute_se3_spectral_certificate(energy_fn, kinematics_fn, probed)
    if spectral is None:
        return probed, None
    if (
        not math.isfinite(spectral.lmin_param)
        or not math.isfinite(spectral.lmax_param)
        or not math.isfinite(spectral.mu_coord)
        or spectral.lmin_param <= 0.0
        or spectral.lmax_param < spectral.lmin_param
        or spectral.mu_coord <= 0.0
    ):
        return probed, None

    # Phase 2: certified GD
    alpha = 2.0 / (spectral.lmin_param + spectral.lmax_param)
    q = (spectral.lmax_param - spectral.lmin_param) / (
        spectral.lmax_param + spectral.lmin_param
    )
    if not math.isfinite(alpha) or alpha <= 0.0 or not math.isfinite(q):
        return probed, None
    initial_gap = float(energy_fn(initial_params)) - float(energy_fn(probed))
    if initial_gap <= 0:
        return probed, None

    budget = certified_iteration_budget(
        spectral.mu_coord,
        q,
        initial_gap,
        target_rmsd,
        n_atoms,
    )
    if budget < 0:
        return probed, None

    optimized = _run_gd_steps(probed, energy_fn, alpha=alpha, n_steps=budget)
    cert = RefinementCertificate(
        spectral=spectral,
        q=q,
        initial_gap=initial_gap,
        target_rmsd=target_rmsd,
        n_steps=budget,
        mode=RefinementCertificationMode.CERTIFIED_GD,
    )
    return optimized, cert
