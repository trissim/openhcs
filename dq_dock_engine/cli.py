#!/usr/bin/env python3
"""
DQ-Dock Engine CLI — command-line interface for molecular docking.
Enum-driven subcommand dispatch (no bare string argparse).
"""
import argparse
import sys
from enum import Enum, auto

import jax
import jax.numpy as jnp

from dq_dock_engine.data.pdb_utils import parse_pdb
from dq_dock_engine.physics.engine import MDState, VelocityVerlet, Langevin, hamiltonian, compute_forces
from dq_dock_engine.physics.potentials import LennardJones, CompositePotential
from dq_dock_engine.physics.srank import compute_srank
from dq_dock_engine.physics.thermodynamics import landauer_constant, thermodynamic_cost
from dq_dock_engine.physics.algorithms import run_standard_md, run_factorized
from dq_dock_engine.pipeline.router import route_by_srank, Tractability

class Command(Enum):
    """Enum-driven CLI dispatch (violation 10 fix)."""
    DOCK = "dock"
    MINIMIZE = "minimize"
    SRANK = "srank"
    ENERGY = "energy"

    def execute(self, args):
        """Dispatch to handler via enum method."""
        dispatch = {
            Command.DOCK: _cmd_dock,
            Command.MINIMIZE: _cmd_minimize,
            Command.SRANK: _cmd_srank,
            Command.ENERGY: _cmd_energy,
        }
        return dispatch[self](args)

def _cmd_dock(args):
    """Run molecular docking simulation."""
    print(f"Loading protein: {args.protein}")
    prot_pos, prot_mass, _ = parse_pdb(args.protein)
    print(f"Loading ligand: {args.ligand}")
    lig_pos, lig_mass, _ = parse_pdb(args.ligand)

    # Compute srank to determine algorithm
    all_pos = jnp.concatenate([prot_pos, lig_pos])
    potential = LennardJones(epsilon=args.epsilon, sigma=args.sigma)
    srank = compute_srank(all_pos, lambda q: potential.energy(q[:len(prot_pos)], q[len(prot_pos):]))

    total_dim = all_pos.shape[0] * 3
    result = route_by_srank(srank, total_dim)
    print(f"srank={srank}, tractability={result.tractability.name}, speedup={result.estimated_speedup:.1f}x")

    # Build initial state
    all_mass = jnp.concatenate([prot_mass, lig_mass])
    key = jax.random.PRNGKey(args.seed)
    state = MDState(all_pos, jnp.zeros_like(all_pos), all_mass, 0.0, key)

    # Run via enum dispatch
    force_fn = lambda q: -jax.grad(lambda q: potential.energy(q[:len(prot_pos)], q[len(prot_pos):]))(q)
    final = result.tractability.run(state, force_fn, args.dt, args.steps, gamma=args.gamma, temperature=args.temperature)

    final_energy = hamiltonian(lambda q: potential.energy(q[:len(prot_pos)], q[len(prot_pos):]), final)
    print(f"Final energy: {final_energy:.6f}")

def _cmd_minimize(args):
    """Energy minimization via steepest descent."""
    print(f"Loading structure: {args.input}")
    pos, mass, _ = parse_pdb(args.input)
    
    potential = LennardJones(epsilon=args.epsilon, sigma=args.sigma)
    energy_fn = lambda q: potential.energy(q[:len(pos)//2], q[len(pos)//2:])
    
    # Simple steepest descent
    q = pos
    for i in range(args.steps):
        grad = jax.grad(energy_fn)(q)
        q = q - args.step_size * grad
        if i % 100 == 0:
            e = energy_fn(q)
            print(f"  step {i}: E = {e:.6f}")
    
    print(f"Final energy: {energy_fn(q):.6f}")

def _cmd_srank(args):
    """Compute structural rank of a molecular system."""
    print(f"Loading protein: {args.protein}")
    prot_pos, _, _ = parse_pdb(args.protein)
    print(f"Loading ligand: {args.ligand}")
    lig_pos, _, _ = parse_pdb(args.ligand)

    all_pos = jnp.concatenate([prot_pos, lig_pos])
    potential = LennardJones()
    srank = compute_srank(all_pos, lambda q: potential.energy(q[:len(prot_pos)], q[len(prot_pos):]))

    total_dim = all_pos.shape[0] * 3
    result = route_by_srank(srank, total_dim)
    cost = thermodynamic_cost(srank, 100, args.temperature)

    print(f"Structural rank: {srank}")
    print(f"Total dimensions: {total_dim}")
    print(f"Ratio: {srank/total_dim:.4f}")
    print(f"Tractability: {result.tractability.name}")
    print(f"Estimated speedup: {result.estimated_speedup:.1f}x")
    print(f"Thermodynamic cost: {cost:.2e} J")
    print(f"Landauer limit: {landauer_constant(args.temperature):.2e} J/bit")

def _cmd_energy(args):
    """Compute system energy."""
    print(f"Loading structure: {args.input}")
    pos, mass, _ = parse_pdb(args.input)
    
    potential = LennardJones(epsilon=args.epsilon, sigma=args.sigma)
    key = jax.random.PRNGKey(0)
    state = MDState(pos, jnp.zeros_like(pos), mass, 0.0, key)
    
    energy_fn = lambda q: potential.energy(q[:len(pos)//2], q[len(pos)//2:])
    H = hamiltonian(energy_fn, state)
    print(f"Total energy: {H:.6f}")

def main():
    parser = argparse.ArgumentParser(description="DQ-Dock Engine — verified molecular docking")
    sub = parser.add_subparsers(dest="command", required=True)

    # dock
    p_dock = sub.add_parser("dock", help="Run docking simulation")
    p_dock.add_argument("--protein", required=True)
    p_dock.add_argument("--ligand", required=True)
    p_dock.add_argument("--dt", type=float, default=0.002)
    p_dock.add_argument("--steps", type=int, default=1000)
    p_dock.add_argument("--gamma", type=float, default=1.0)
    p_dock.add_argument("--temperature", type=float, default=300.0)
    p_dock.add_argument("--epsilon", type=float, default=1.0)
    p_dock.add_argument("--sigma", type=float, default=1.0)
    p_dock.add_argument("--seed", type=int, default=42)

    # minimize
    p_min = sub.add_parser("minimize", help="Energy minimization")
    p_min.add_argument("--input", required=True)
    p_min.add_argument("--steps", type=int, default=1000)
    p_min.add_argument("--step-size", type=float, default=0.001)
    p_min.add_argument("--epsilon", type=float, default=1.0)
    p_min.add_argument("--sigma", type=float, default=1.0)

    # srank
    p_srank = sub.add_parser("srank", help="Compute structural rank")
    p_srank.add_argument("--protein", required=True)
    p_srank.add_argument("--ligand", required=True)
    p_srank.add_argument("--temperature", type=float, default=300.0)

    # energy
    p_energy = sub.add_parser("energy", help="Compute system energy")
    p_energy.add_argument("--input", required=True)
    p_energy.add_argument("--epsilon", type=float, default=1.0)
    p_energy.add_argument("--sigma", type=float, default=1.0)

    args = parser.parse_args()
    cmd = Command(args.command)
    cmd.execute(args)

if __name__ == "__main__":
    main()
