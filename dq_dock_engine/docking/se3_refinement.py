"""SE(3) refinement with certified RMSD convergence guarantees.

Implements two approaches for theorem-backed optimization budgets:

  Approach A (observed): Run any optimizer, observe energy trajectory,
    certify post-hoc via SE(3) Hessian + Jacobian bridge.

  Approach B (certified_gd): Run standard gradient descent in axis-angle
    parameterization with theorem-derived step size and budget.

Both share the SE(3) spectral certificate computation (Hessian eigenvalues
and kinematics Jacobian singular values) and the parameter/coordinate
quadratic-window bridge.

Lean: EnergyRMSDConvergence.lean — CertifiedQuadraticBasin,
      CertifiedLinearEnergyConvergence,
      parameter_window_transfers_to_coordinate_window_sq,
      rmsd_target_of_canonicalIterationBudgetFromLocalCertificates,
      rmsd_target_of_canonicalIterationBudgetFromGradientDescentDynamics,
      rmsd_target_of_initial_rmsd_and_linear_energy_convergence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from dq_dock_engine.docking.certified_runtime_plans import (
    CertifiedRefinementBudget,
    CertifiedRefinementBudgetKind,
)
from dq_dock_engine.docking.placement import _apply_single_pose
from dq_dock_engine.docking.scoring import _score_certified_lj
from dq_dock_engine.docking_config import RefinementCertificationMode


@dataclass(frozen=True)
class SE3SpectralCertificate:
    """Hessian eigenvalues in SE(3) parameter space + Jacobian bridge.

    Lean: EnergyRMSDConvergence.parameter_window_transfers_to_coordinate_window_sq.
    """

    lmin_param: float
    lmax_param: float
    sigma_min_sq: float
    sigma_max_sq: float
    mu_coord: float  # lmin_param / sigma_max_sq
    M_coord: float  # lmax_param / sigma_min_sq


@dataclass(frozen=True)
class RefinementCertificate:
    """Combined certificate for theorem-backed n_opt_steps."""

    spectral: SE3SpectralCertificate
    q: float  # contraction rate (observed for A, derived for B)
    initial_gap: float
    target_rmsd: float
    n_steps: int  # certified budget
    mode: RefinementCertificationMode
    iteration_budget_plan: "CertifiedIterationBudgetPlan | None" = None


@dataclass(frozen=True)
class CertifiedIterationBudgetPlan:
    """First-class theorem-backed SE(3) iteration-budget derivation."""

    budget: CertifiedRefinementBudget
    mu_coord: float
    q: float
    initial_gap: float
    target_gap: float
    target_rmsd: float
    n_atoms: int
    theorem_handles: tuple[str, ...]


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
    *,
    scoring_context=None,
    target_error: float | None = None,
):
    """Energy as a function of 6D SE(3) parameter vector.

    The function E: R^6 → R maps [tx, ty, tz, θx, θy, θz] to the
    certified LJ score. This is the optimizer's actual parameter space.
    """

    if scoring_context is None:

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

    else:
        if target_error is None:
            raise ValueError("target_error is required when using a scoring context")

        def energy_fn(params: jnp.ndarray) -> jnp.ndarray:
            t, q = _se3_params_to_pose(params)
            pose_coords = _apply_single_pose(base_coords, t, q)
            batch = scoring_context.score_exact_batch(
                receptor_coords=receptor_coords,
                poses_coords=pose_coords[None, ...],
                receptor_radii=receptor_radii,
                ligand_radii=ligand_radii,
                target_error=target_error,
                epsilon=epsilon,
            )
            return batch.scores[0]

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

    Returns None if not in a convex basin (lmin_param ≤ 0) or if computation fails.

    Lean: EnergyRMSDConvergence.parameter_window_transfers_to_coordinate_window_sq.
    """
    try:
        energy_at_point = energy_fn(optimized_params)
        if not math.isfinite(float(energy_at_point)):
            return None
    except Exception:
        return None

    try:
        H = jax.hessian(energy_fn)(optimized_params)  # (6, 6)
    except Exception:
        return None

    if jnp.any(jnp.isnan(H)):
        return None

    try:
        eigs = jnp.linalg.eigvalsh(H)
    except Exception:
        return None

    lmin_param = float(eigs[0])
    lmax_param = float(eigs[-1])

    if not math.isfinite(lmin_param) or not math.isfinite(lmax_param):
        return None

    if lmin_param <= 0:
        return None  # not in a convex basin

    try:
        J = jax.jacobian(kinematics_fn)(optimized_params)  # (N_atoms, 3, 6)
    except Exception:
        return None

    if jnp.any(jnp.isnan(J)):
        return None

    J_flat = J.reshape(-1, 6)  # (3*N_atoms, 6)
    singular_values = jnp.linalg.svdvals(J_flat)
    sigma_min_sq = float(jnp.min(singular_values) ** 2)
    sigma_max_sq = float(jnp.max(singular_values) ** 2)

    if not math.isfinite(sigma_min_sq) or not math.isfinite(sigma_max_sq):
        return None
    if sigma_max_sq <= 0:
        return None

    mu_coord = lmin_param / sigma_max_sq
    M_coord = (
        float("inf")
        if sigma_min_sq <= 0.0 or not math.isfinite(sigma_min_sq)
        else lmax_param / sigma_min_sq
    )

    if not math.isfinite(mu_coord):
        return None

    return SE3SpectralCertificate(
        lmin_param=lmin_param,
        lmax_param=lmax_param,
        sigma_min_sq=sigma_min_sq,
        sigma_max_sq=sigma_max_sq,
        mu_coord=mu_coord,
        M_coord=M_coord,
    )


# ---------------------------------------------------------------------------
# Iteration budget computation (shared by both approaches)
# ---------------------------------------------------------------------------


def certified_iteration_budget_plan(
    mu_coord: float,
    q: float,
    initial_gap: float,
    target_rmsd: float,
    n_atoms: int,
) -> CertifiedIterationBudgetPlan | None:
    """Provably sufficient iteration count for the RMSD target.

    Lean: logarithmicIterationBound applied to q and μ from Jacobian bridge.

    target_gap = μ × n × eps² / 2   (Lean: targetEnergyGap)
    budget = ceil(log(initial_gap / target_gap) / log(1/q))
    """
    target_gap = mu_coord * n_atoms * target_rmsd**2 / 2.0
    if not math.isfinite(target_gap) or target_gap <= 0.0:
        return None
    if not math.isfinite(initial_gap) or initial_gap <= 0.0:
        return None
    if initial_gap <= target_gap:
        n_steps = 0
    else:
        if q <= 0 or q >= 1.0:
            return None
        ratio = initial_gap / target_gap
        if not math.isfinite(ratio) or ratio <= 0.0:
            return None
        denom = math.log(1.0 / q)
        if not math.isfinite(denom) or denom <= 0.0:
            return None
        numer = math.log(ratio)
        if not math.isfinite(numer):
            return None
        n_steps = math.ceil(numer / denom)
    theorem_handles = ("ERC20", "ERC21", "ERC38", "ERC40")
    return CertifiedIterationBudgetPlan(
        budget=CertifiedRefinementBudget(
            kind=CertifiedRefinementBudgetKind.SE3_ITERATIONS,
            n_steps=n_steps,
            pose_indices=(),
            theorem_handles=theorem_handles,
            note="Certified SE(3) gradient-descent iteration budget",
        ),
        mu_coord=mu_coord,
        q=q,
        initial_gap=initial_gap,
        target_gap=target_gap,
        target_rmsd=target_rmsd,
        n_atoms=n_atoms,
        theorem_handles=theorem_handles,
    )


def certified_iteration_budget(
    mu_coord: float,
    q: float,
    initial_gap: float,
    target_rmsd: float,
    n_atoms: int,
) -> int:
    plan = certified_iteration_budget_plan(
        mu_coord=mu_coord,
        q=q,
        initial_gap=initial_gap,
        target_rmsd=target_rmsd,
        n_atoms=n_atoms,
    )
    return -1 if plan is None else plan.budget.n_steps


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
    iteration_budget_plan = certified_iteration_budget_plan(
        spectral.mu_coord,
        q,
        initial_gap,
        target_rmsd,
        n_atoms,
    )
    if iteration_budget_plan is None:
        return None
    return RefinementCertificate(
        spectral=spectral,
        q=q,
        initial_gap=initial_gap,
        target_rmsd=target_rmsd,
        n_steps=iteration_budget_plan.budget.n_steps,
        mode=RefinementCertificationMode.OBSERVED,
        iteration_budget_plan=iteration_budget_plan,
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


def _stabilized_probe_step_limits(
    ligand_radius: float,
    target_rmsd: float,
) -> tuple[float, float]:
    """Deterministic per-step RMSD budget split for the observed probe.

    Lean: EnergyRMSDConvergence
      - ERC44 rmsd_le_of_pointwiseDist_le
      - ERC45 rmsd_le_of_pointwiseSplitDist_le

    We split the per-step RMSD budget evenly between translation and
    rotation-induced displacement. If every atom moves by at most `tBound` from
    translation and at most `rBound` from rotation with `tBound + rBound ≤
    target_rmsd`, then the full pointwise displacement - and therefore RMSD - is
    bounded by `target_rmsd`.
    """
    translation_limit = target_rmsd / 2.0
    if ligand_radius <= 1e-8:
        rotation_limit = math.pi
    else:
        rotation_limit = min(math.pi, target_rmsd / (2.0 * ligand_radius))
    return translation_limit, rotation_limit


def _initial_probe_backtracking_round(
    translation_norm: float,
    rotation_norm: float,
    ligand_radius: float,
    target_rmsd: float,
) -> int:
    """Canonical first dyadic round whose step budget certifies RMSD safety.

    Lean:
      - ERC45 (`rmsd_le_of_pointwiseSplitDist_le`)
      - SH12 (`leastPositiveJointAdequateDyadicRound_spec`)

    We require each of the two pointwise displacement channels to fit within
    half the RMSD budget so their sum remains within the full target radius.
    """
    from dq_dock_engine.docking.formal_actions import (
        least_positive_joint_adequate_dyadic_round,
    )

    split_budget = target_rmsd / 2.0
    rotation_displacement_norm = ligand_radius * rotation_norm
    return least_positive_joint_adequate_dyadic_round(
        translation_norm,
        rotation_displacement_norm,
        split_budget,
    )


def _run_stabilized_observed_probe(
    initial_params: jnp.ndarray,
    energy_fn,
    *,
    n_steps: int,
    ligand_radius: float,
    target_rmsd: float,
) -> tuple[list[jnp.ndarray], list[float]]:
    """Observed-mode probe with dyadically derived monotone backtracking.

    The observed certificate only depends on the realized energy trajectory, not
    on the specific optimizer family. This probe therefore prioritizes staying in
    a local interacting basin over taking one large unconstrained SE(3) step that
    can eject the ligand into the zero-contact plateau.

    There are no free line-search constants here: we follow the negative gradient
    direction and start at the first dyadic scale whose pointwise translation and
    rotation-induced displacements fit inside the theorem-backed RMSD split
    budget. Further backtracking only halves that already-safe step.
    """
    params = initial_params
    params_trajectory = [params]
    energy_current = float(energy_fn(params))
    energy_trajectory = [energy_current]

    for _ in range(n_steps):
        value, grad = jax.value_and_grad(energy_fn)(params)
        energy_current = float(value)
        step = -grad
        round_index = _initial_probe_backtracking_round(
            float(jnp.linalg.norm(step[:3])),
            float(jnp.linalg.norm(step[3:6])),
            ligand_radius,
            target_rmsd,
        )

        next_params = params
        next_energy = energy_current
        while True:
            candidate_step = step * math.ldexp(1.0, -round_index)
            if float(jnp.max(jnp.abs(candidate_step))) == 0.0:
                break
            candidate = params + candidate_step
            candidate_energy = float(energy_fn(candidate))
            if math.isfinite(candidate_energy) and candidate_energy < energy_current:
                next_params = candidate
                next_energy = candidate_energy
                break
            round_index += 1

        if next_energy >= energy_current:
            break

        params = next_params
        params_trajectory.append(params)
        energy_trajectory.append(next_energy)

    return params_trajectory, energy_trajectory


def _best_observed_certificate_from_trajectory(
    params_trajectory: list[jnp.ndarray],
    energy_trajectory: list[float],
    energy_fn,
    kinematics_fn,
    *,
    target_rmsd: float,
    n_atoms: int,
) -> tuple[jnp.ndarray, RefinementCertificate | None]:
    """Return the latest certifiable prefix of an observed probe trajectory.

    Lean: EnergyRMSDConvergence.rmsd_target_of_maxCertifiedPrefix.
    """
    best_params = params_trajectory[-1]
    for end in range(len(params_trajectory) - 1, 0, -1):
        spectral = compute_se3_spectral_certificate(
            energy_fn,
            kinematics_fn,
            params_trajectory[end],
        )
        if spectral is None:
            continue
        cert = certify_observed(
            spectral=spectral,
            energy_trajectory=energy_trajectory[: end + 1],
            target_rmsd=target_rmsd,
            n_atoms=n_atoms,
        )
        if cert is not None:
            return params_trajectory[end], cert
    return best_params, None


def observe_gd_trajectory(
    initial_params: jnp.ndarray,
    energy_fn,
    kinematics_fn,
    n_steps: int,
    target_rmsd: float,
    n_atoms: int,
    ligand_radius: float,
) -> tuple[jnp.ndarray, RefinementCertificate | None]:
    """Approach A: follow the negative gradient with dyadically derived step
    budgets, record the realized energy trajectory, then certify post-hoc via
    spectral certificate + observed contraction rate.

    The key difference from Approach B is that q is *empirical* (worst-case
    ratio from the trajectory) rather than *derived* from the Hessian condition
    number. This makes Approach A applicable even when the Hessian-derived q
    is pessimistic.

    Lean: CertifiedOneStepEnergyContraction — each consecutive pair in the
    trajectory must satisfy gap(t+1) ≤ q × gap(t).
    """
    params_trajectory, energy_trajectory = _run_stabilized_observed_probe(
        initial_params,
        energy_fn,
        n_steps=n_steps,
        ligand_radius=ligand_radius,
        target_rmsd=target_rmsd,
    )

    optimized, cert = _best_observed_certificate_from_trajectory(
        params_trajectory,
        energy_trajectory,
        energy_fn,
        kinematics_fn,
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
    ligand_radius: float,
    max_probe_steps: int,
) -> tuple[jnp.ndarray, RefinementCertificate | None]:
    """Two-phase certified optimization (Approach B).

    Phase 1: Quick probe with the same dyadically derived observed-step policy.
    Phase 2: Compute Hessian, derive α and budget T, run T steps of standard GD.

    Lean: CertifiedGradientDescentDynamics — q = (M-μ)/(M+μ) with
          α = 2/(μ+M) is the optimal fixed step size for smooth strongly
          convex functions.
    """
    # Phase 1: probe
    probe_params, _ = _run_stabilized_observed_probe(
        initial_params,
        energy_fn,
        n_steps=max_probe_steps,
        ligand_radius=ligand_radius,
        target_rmsd=target_rmsd,
    )
    probed = probe_params[-1]

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
