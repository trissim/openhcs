from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from dq_dock_engine.docking.formal_actions import (
    CertifiedActionFamily,
    apply_action_family,
    apply_local_action,
    create_certified_action_family,
)
from dq_dock_engine.docking.formal_belief import (
    CertifiedBeliefState,
    CertifiedPriorSpec,
    build_prior,
    select_admissible_action,
    update_posterior,
)
from dq_dock_engine.docking.formal_pruning import (
    ambiguity_band_mask,
    certified_survivor_mask,
)
from dq_dock_engine.docking.formal_surrogates import (
    CertifiedCoarseScoreBundle,
    score_exact_and_coarse_round,
)


@dataclass(frozen=True)
class CertifiedOptimizerState:
    coords: jax.Array
    action_family: CertifiedActionFamily
    belief: CertifiedBeliefState
    retained_receptor_indices: jax.Array


def _build_belief_state(
    bundle: CertifiedCoarseScoreBundle,
    prior_spec: CertifiedPriorSpec,
    step_index: int,
) -> CertifiedBeliefState:
    prior = build_prior(prior_spec, len(bundle.exact_scores))
    posterior = update_posterior(prior, bundle.survivor_mask)
    selected_action = select_admissible_action(posterior, bundle.ambiguity_mask)
    return CertifiedBeliefState(
        prior_spec=prior_spec,
        prior=prior,
        posterior=posterior,
        coarse_scores=bundle.coarse_scores,
        exact_scores=bundle.exact_scores,
        exact_error_bound=bundle.exact_error_bound,
        survivor_mask=bundle.survivor_mask,
        ambiguity_mask=bundle.ambiguity_mask,
        selected_action=selected_action,
        step_index=step_index,
    )


def _translation_step_for_round(base_step: float, round_index: int) -> float:
    return base_step / float(2**round_index)


def _rotation_step_for_round(base_step: float, round_index: int) -> float:
    return base_step / float(2**round_index)


def _refine_round(
    coords_batch: jax.Array,
    receptor_coords: jax.Array,
    receptor_radii: jax.Array,
    ligand_radii: jax.Array,
    target_error: float,
    round_index: int,
    base_translation_step: float,
    base_rotation_step_rad: float,
    prior_spec: CertifiedPriorSpec,
    max_coarse_receptor_atoms: int,
) -> tuple[jax.Array, tuple[CertifiedOptimizerState, ...]]:
    action_family = create_certified_action_family(
        translation_step=_translation_step_for_round(
            base_translation_step, round_index
        ),
        rotation_step_rad=_rotation_step_for_round(base_rotation_step_rad, round_index),
        stencil_level=round_index,
    )

    candidate_batches = jax.vmap(apply_action_family, in_axes=(0, None))(
        coords_batch, action_family
    )
    (
        exact_scores_matrix,
        coarse_scores_matrix,
        delta,
        exact_error_bound,
        retained_indices,
    ) = score_exact_and_coarse_round(
        receptor_coords=receptor_coords,
        receptor_radii=receptor_radii,
        ligand_radii=ligand_radii,
        candidate_batches=candidate_batches,
        target_error=target_error,
        max_receptor_atoms=max_coarse_receptor_atoms,
    )
    next_coords = []
    states = []
    for pose_index in range(candidate_batches.shape[0]):
        survivor_mask = certified_survivor_mask(
            exact_scores=exact_scores_matrix[pose_index],
            coarse_scores=coarse_scores_matrix[pose_index],
            k=1,
            delta=delta,
        )
        ambiguity_mask = ambiguity_band_mask(
            exact_scores_matrix[pose_index],
            k=1,
            epsilon=delta,
        )
        bundle = CertifiedCoarseScoreBundle(
            exact_scores=exact_scores_matrix[pose_index],
            coarse_scores=coarse_scores_matrix[pose_index],
            delta=delta,
            exact_error_bound=exact_error_bound,
            survivor_mask=survivor_mask,
            ambiguity_mask=ambiguity_mask,
            retained_receptor_indices=retained_indices,
        )
        belief = _build_belief_state(
            bundle=bundle,
            prior_spec=prior_spec,
            step_index=round_index,
        )
        selected_coords = candidate_batches[pose_index, belief.selected_action]
        next_coords.append(selected_coords)
        states.append(
            CertifiedOptimizerState(
                coords=selected_coords,
                action_family=action_family,
                belief=belief,
                retained_receptor_indices=bundle.retained_receptor_indices,
            )
        )

    return jnp.stack(next_coords, axis=0), tuple(states)


def refine_poses_certified(
    coords_batch: jax.Array,
    receptor_coords: jax.Array,
    receptor_radii: jax.Array,
    ligand_radii: jax.Array,
    n_rounds: int,
    target_error: float,
    base_translation_step: float = 0.5,
    base_rotation_step_rad: float = jnp.pi / 12.0,
    prior_spec: CertifiedPriorSpec | None = None,
    max_coarse_receptor_atoms: int = 64,
) -> tuple[jax.Array, tuple[tuple[CertifiedOptimizerState, ...], ...]]:
    current_coords = coords_batch
    history: list[tuple[CertifiedOptimizerState, ...]] = []
    effective_prior_spec = (
        CertifiedPriorSpec(kind="uniform") if prior_spec is None else prior_spec
    )

    for round_index in range(n_rounds):
        current_coords, states = _refine_round(
            coords_batch=current_coords,
            receptor_coords=receptor_coords,
            receptor_radii=receptor_radii,
            ligand_radii=ligand_radii,
            target_error=target_error,
            round_index=round_index,
            base_translation_step=base_translation_step,
            base_rotation_step_rad=float(base_rotation_step_rad),
            prior_spec=effective_prior_spec,
            max_coarse_receptor_atoms=max_coarse_receptor_atoms,
        )
        history.append(states)

        if all(action_state.belief.selected_action == 0 for action_state in states):
            break

    return current_coords, tuple(history)


@dataclass(frozen=True)
class CertifiedConvergenceCertificate:
    """Certificate that the optimizer has converged to a local minimum."""

    is_local_minimum: bool
    n_consecutive_noop_rounds: int
    final_round_index: int
    energy_gap_to_neighbors: float
    neighbor_energies: tuple[float, ...]


@dataclass(frozen=True)
class CertifiedMultiStartResult:
    """Result from rigorous multi-start optimization with uncertainty quantification."""

    best_coords: jax.Array
    best_energy: float
    best_start_index: int
    all_energies: tuple[float, ...]
    all_coords: tuple[jax.Array, ...]
    convergence_certificates: tuple[CertifiedConvergenceCertificate, ...]
    energy_spread: float
    energy_std: float
    n_starts: int
    n_converged: int


def _check_local_minimum(
    coords: jax.Array,
    receptor_coords: jax.Array,
    receptor_radii: jax.Array,
    ligand_radii: jax.Array,
    action_family: CertifiedActionFamily,
    target_error: float,
    max_coarse_receptor_atoms: int,
) -> tuple[bool, float, tuple[float, ...]]:
    """Check if coords is at a local minimum by evaluating all neighbor actions.

    Returns:
        (is_local_minimum, energy_gap_to_best_neighbor, neighbor_energies)
    """
    candidate_coords = apply_action_family(coords, action_family)

    (
        exact_scores_matrix,
        coarse_scores_matrix,
        delta,
        exact_error_bound,
        retained_indices,
    ) = score_exact_and_coarse_round(
        receptor_coords=receptor_coords,
        receptor_radii=receptor_radii,
        ligand_radii=ligand_radii,
        candidate_batches=candidate_coords[jnp.newaxis, :, :, :],
        target_error=target_error,
        max_receptor_atoms=max_coarse_receptor_atoms,
    )

    exact_scores = exact_scores_matrix[0]
    noop_energy = float(exact_scores[0])

    neighbor_energies = tuple(float(e) for e in exact_scores[1:])
    best_neighbor_energy = min(neighbor_energies)
    energy_gap = noop_energy - best_neighbor_energy

    is_local_minimum = energy_gap < 0.0

    return is_local_minimum, energy_gap, neighbor_energies


def _refine_single_start(
    coords: jax.Array,
    receptor_coords: jax.Array,
    receptor_radii: jax.Array,
    ligand_radii: jax.Array,
    n_rounds: int,
    target_error: float,
    base_translation_step: float,
    base_rotation_step_rad: float,
    prior_spec: CertifiedPriorSpec,
    max_coarse_receptor_atoms: int,
    check_convergence: bool = True,
) -> tuple[
    jax.Array,
    float,
    CertifiedConvergenceCertificate,
    tuple[CertifiedOptimizerState, ...],
]:
    """Refine a single starting pose with convergence certificate.

    Returns:
        (final_coords, final_energy, convergence_certificate, state_history)
    """
    current_coords = coords[jnp.newaxis, :, :]
    history: list[CertifiedOptimizerState] = []
    consecutive_noop = 0

    for round_index in range(n_rounds):
        current_coords_batch, states = _refine_round(
            coords_batch=current_coords,
            receptor_coords=receptor_coords,
            receptor_radii=receptor_radii,
            ligand_radii=ligand_radii,
            target_error=target_error,
            round_index=round_index,
            base_translation_step=base_translation_step,
            base_rotation_step_rad=base_rotation_step_rad,
            prior_spec=prior_spec,
            max_coarse_receptor_atoms=max_coarse_receptor_atoms,
        )

        current_coords = current_coords_batch
        state = states[0]
        history.append(state)

        if state.belief.selected_action == 0:
            consecutive_noop += 1
            if consecutive_noop >= 2:
                break
        else:
            consecutive_noop = 0

    final_coords = current_coords[0]
    final_energy = float(state.belief.exact_scores[state.belief.selected_action])

    convergence_cert = CertifiedConvergenceCertificate(
        is_local_minimum=False,
        n_consecutive_noop_rounds=consecutive_noop,
        final_round_index=len(history) - 1,
        energy_gap_to_neighbors=0.0,
        neighbor_energies=(),
    )

    if check_convergence:
        is_local_min, energy_gap, neighbor_energies = _check_local_minimum(
            coords=final_coords,
            receptor_coords=receptor_coords,
            receptor_radii=receptor_radii,
            ligand_radii=ligand_radii,
            action_family=state.action_family,
            target_error=target_error,
            max_coarse_receptor_atoms=max_coarse_receptor_atoms,
        )
        convergence_cert = CertifiedConvergenceCertificate(
            is_local_minimum=is_local_min,
            n_consecutive_noop_rounds=consecutive_noop,
            final_round_index=len(history) - 1,
            energy_gap_to_neighbors=energy_gap,
            neighbor_energies=neighbor_energies,
        )

    return final_coords, final_energy, convergence_cert, tuple(history)


def refine_poses_certified_rigorous(
    coords_batch: jax.Array,
    receptor_coords: jax.Array,
    receptor_radii: jax.Array,
    ligand_radii: jax.Array,
    n_rounds: int,
    target_error: float,
    n_starts: int = 3,
    base_translation_step: float = 0.5,
    base_rotation_step_rad: float = jnp.pi / 12.0,
    prior_spec: CertifiedPriorSpec | None = None,
    max_coarse_receptor_atoms: int = 64,
    lattice_offsets: tuple[tuple[float, float, float], ...] | None = None,
) -> CertifiedMultiStartResult:
    """Rigorous multi-start optimization with convergence certificates and uncertainty.

    This is the rigorous version of refine_poses_certified that:
    1. Runs multiple starts from different lattice offsets
    2. Checks convergence to local minimum
    3. Quantifies uncertainty via energy spread
    4. Returns the best result with full provenance

    Args:
        coords_batch: Batch of initial poses (N, M, 3)
        receptor_coords: Receptor coordinates
        receptor_radii: Receptor atomic radii
        ligand_radii: Ligand atomic radii
        n_rounds: Number of refinement rounds per start
        target_error: Target error bound for coarse scoring
        n_starts: Number of multi-start attempts (default: 3)
        base_translation_step: Initial translation step size
        base_rotation_step_rad: Initial rotation step size in radians
        prior_spec: Prior specification (default: uniform)
        max_coarse_receptor_atoms: Max atoms in coarse receptor
        lattice_offsets: Custom lattice offsets for multi-start (optional)

    Returns:
        CertifiedMultiStartResult with best pose, energy, and uncertainty
    """
    if n_starts <= 0:
        raise ValueError(f"n_starts must be positive, got {n_starts}")

    effective_prior_spec = (
        CertifiedPriorSpec(kind="uniform") if prior_spec is None else prior_spec
    )

    if lattice_offsets is None:
        base_offset = base_translation_step * 0.5
        lattice_offsets = tuple(
            (
                base_offset * (i % 3 - 1),
                base_offset * ((i // 3) % 3 - 1),
                base_offset * (i // 9 - 1),
            )
            for i in range(n_starts)
        )

    all_energies: list[float] = []
    all_coords: list[jax.Array] = []
    all_certificates: list[CertifiedConvergenceCertificate] = []

    for start_index in range(min(n_starts, len(lattice_offsets))):
        offset = jnp.array(lattice_offsets[start_index])
        offset_coords = coords_batch + offset

        coords, energy, cert, _ = _refine_single_start(
            coords=offset_coords[0],
            receptor_coords=receptor_coords,
            receptor_radii=receptor_radii,
            ligand_radii=ligand_radii,
            n_rounds=n_rounds,
            target_error=target_error,
            base_translation_step=base_translation_step,
            base_rotation_step_rad=float(base_rotation_step_rad),
            prior_spec=effective_prior_spec,
            max_coarse_receptor_atoms=max_coarse_receptor_atoms,
            check_convergence=True,
        )

        all_energies.append(energy)
        all_coords.append(coords)
        all_certificates.append(cert)

    energies_array = jnp.array(all_energies)
    best_index = int(jnp.argmin(energies_array))

    energy_spread = float(jnp.max(energies_array) - jnp.min(energies_array))
    energy_std = float(jnp.std(energies_array))

    n_converged = sum(1 for cert in all_certificates if cert.is_local_minimum)

    return CertifiedMultiStartResult(
        best_coords=all_coords[best_index],
        best_energy=all_energies[best_index],
        best_start_index=best_index,
        all_energies=tuple(all_energies),
        all_coords=tuple(all_coords),
        convergence_certificates=tuple(all_certificates),
        energy_spread=energy_spread,
        energy_std=energy_std,
        n_starts=len(all_energies),
        n_converged=n_converged,
    )
