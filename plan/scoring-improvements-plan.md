# Scoring Function Improvements Plan

**Goal**: Implement advanced physics terms for 1-2 Å RMSD improvement

**Status**: 🔄 Planning
**Priority**: 🥈 HIGH - Major impact but significant complexity
**Expected Impact**: 1-2 Å RMSD improvement
**Expected Speed Impact**: Negative (2-5x slower per pose) but worth it for accuracy

---

## Problem Statement

### Current Scoring: Pure Lennard-Jones Only

```python
# Current: dq_dock_engine/docking/scoring.py:20-56
def _score_single_lj(...):
    # Only atom-typed LJ potential
    # No electrostatics
    # No hydrogen bonding
    # No desolvation
    # No hydrophobic effect
```

**Missing Physics**:
1. **Electrostatics**: Critical for polar interactions, salt bridges
2. **Hydrogen bonding**: Directional, highly specific interactions
3. **Desolvation**: Cost of removing water from binding site
4. **Hydrophobic effect**: Entropic driving force for binding

**Impact**: Current scoring can't distinguish between:
- Poses with good steric fit but wrong chemistry
- Correct vs incorrect H-bond patterns
- Favorable vs unfavorable charge interactions

---

## Architecture Design

### Scoring Function Hierarchy

```
Total Score = w_LJ × E_LJ + w_elec × E_elec + w_hb × E_hb + w_desolv × E_desolv

Where:
- E_LJ: Lennard-Jones potential (existing)
- E_elec: Electrostatic energy (NEW)
- E_hb: Hydrogen bonding (NEW)
- E_desolv: Desolvation penalty (NEW)
```

**Weights determined empirically** (see parameter sweep section)

---

## Phase 1: Electrostatics

### 1.1 Charge Assignment

**Goal**: Assign realistic partial charges to all atoms

**Methods** (in order of accuracy):

**Level 1: Element-based rules (current approach)**
```python
ASSIGN_CHARGES_RULES = {
    'C': 0.0,    # Hydrocarbons neutral
    'N': -0.3,   # Amines slightly negative
    'O': -0.4,   # Carbonyls negative
    'S': 0.0,    # Sulfur neutral
    'NA': 1.0,   # Sodium cation
    'K': 1.0,    # Potassium cation
    'CL': -1.0,  # Chloride anion
    'BR': -1.0,  # Bromide anion
}
```

**Level 2: AM1-BCC charges (semi-empirical)**
```python
# File: dq_dock_engine/docking/charges.py

def assign_am1_bcc_charges(molecule: 'RDKitMol') -> jnp.ndarray:
    """
    Assign AM1-BCC partial charges using RDKit.

    AM1-BCC is fast and reasonably accurate for docking.
    """
    from rdkit.Chem import AllChem

    # Compute AM1 charges
    charges = AllChem.ComputeGasteigerCharges(molecule)

    # Apply BCC (Bond Charge Corrections)
    # (Simplified - full BCC requires correction rules)

    return jnp.array(charges)
```

**Level 3: AMBER/OPLS force field charges**
```python
def assign_forcefield_charges(
    coords: jnp.ndarray,
    elements: list[str],
    atom_names: list[str],
    forcefield: str = "AMBER"
) -> jnp.ndarray:
    """
    Assign charges using force field parameters.

    Requires atom typing and parameter lookup.
    """
    # Parse force field parameters
    # Assign atom types
    # Look up charges
    # This is complex - requires full force field implementation

    raise NotImplementedError("Use AM1-BCC instead")
```

**Recommendation**: Start with Level 1, upgrade to Level 2 if needed

### 1.2 Coulomb Potential

**Basic implementation**:
```python
@jax.jit
def coulomb_energy(
    pose_coords: jnp.ndarray,
    pose_charges: jnp.ndarray,
    receptor_coords: jnp.ndarray,
    receptor_charges: jnp.ndarray,
    dielectric: float = 4.0
) -> float:
    """
    Compute Coulomb electrostatic energy.

    E = sum(q_i * q_j / (4 * pi * epsilon_0 * epsilon_r * r_ij))

    Simplified: E = sum(q_i * q_j / (dielectric * r_ij))

    Args:
        pose_coords: (N_lig, 3)
        pose_charges: (N_lig,)
        receptor_coords: (N_rec, 3)
        receptor_charges: (N_rec,)
        dielectric: Dielectric constant (4 = protein interior)
    """
    # Compute all pairwise distances
    diffs = pose_coords[:, None, :] - receptor_coords[None, :, :]
    distances = jnp.sqrt(jnp.sum(diffs ** 2, axis=-1))  # (N_lig, N_rec)

    # Avoid division by zero
    distances = jnp.maximum(distances, 1.0)  # 1Å minimum

    # Compute charge products
    charge_products = pose_charges[:, None] * receptor_charges[None, :]

    # Coulomb energy
    energies = charge_products / (dielectric * distances)

    return jnp.sum(energies)
```

**Performance optimization with cutoff**:
```python
@jax.jit
def coulomb_energy_cutoff(
    pose_coords: jnp.ndarray,
    pose_charges: jnp.ndarray,
    receptor_coords: jnp.ndarray,
    receptor_charges: jnp.ndarray,
    dielectric: float = 4.0,
    cutoff: float = 10.0  # Angstroms
) -> float:
    """
    Coulomb energy with distance cutoff.

    Interactions beyond cutoff are ignored.
    """
    # Use spatial hashing for neighbor search
    # (See multi-stage plan for implementation)

    # Only compute for nearby pairs
    nearby_pairs = get_neighbors_within_cutoff(
        pose_coords, receptor_coords, cutoff
    )

    total_energy = 0.0
    for i, j in nearby_pairs:
        r = jnp.linalg.norm(pose_coords[i] - receptor_coords[j])
        r = jnp.maximum(r, 1.0)
        energy = pose_charges[i] * receptor_charges[j] / (dielectric * r)
        total_energy += energy

    return total_energy
```

**Distance-dependent dielectric** (more realistic):
```python
@jax.jit
def coulomb_energy_distance_dependent(
    pose_coords: jnp.ndarray,
    pose_charges: jnp.ndarray,
    receptor_coords: jnp.ndarray,
    receptor_charges: jnp.ndarray,
    dielectric_base: float = 4.0
) -> float:
    """
    Coulomb energy with distance-dependent dielectric.

    epsilon(r) = dielectric_base * r

    This mimics screening by polarizable medium without
    expensive Poisson-Boltzmann calculations.
    """
    diffs = pose_coords[:, None, :] - receptor_coords[None, :, :]
    distances = jnp.sqrt(jnp.sum(diffs ** 2, axis=-1))
    distances = jnp.maximum(distances, 1.0)

    charge_products = pose_charges[:, None] * receptor_charges[None, :]

    # Distance-dependent dielectric
    energies = charge_products / (dielectric_base * distances ** 2)

    return jnp.sum(energies)
```

### 1.3 Solvation Screening

**Goal**: Account for water screening of electrostatics

**Simple model**:
```python
@jax.jit
def solvation_screened_coulomb(
    pose_coords: jnp.ndarray,
    pose_charges: jnp.ndarray,
    receptor_coords: jnp.ndarray,
    receptor_charges: jnp.ndarray,
    epsilon_protein: float = 4.0,
    epsilon_water: float = 80.0,
    debye_length: float = 10.0  # Angstroms
) -> float:
    """
    Coulomb energy with solvation screening.

    Uses Debye-Hückel screening for ionic strength:
    E = sum(q_i * q_j * exp(-r_ij/lambda) / (epsilon * r_ij))

    where lambda is Debye length.
    """
    diffs = pose_coords[:, None, :] - receptor_coords[None, :, :]
    distances = jnp.sqrt(jnp.sum(diffs ** 2, axis=-1))
    distances = jnp.maximum(distances, 1.0)

    charge_products = pose_charges[:, None] * receptor_charges[None, :]

    # Debye-Hückel screening
    screening = jnp.exp(-distances / debye_length)

    energies = charge_products * screening / (epsilon_protein * distances)

    return jnp.sum(energies)
```

---

## Phase 2: Hydrogen Bonding

### 2.1 Donor/Acceptor Detection

```python
# File: dq_dock_engine/docking/hbonds.py

@dataclass
class HydrogenBondDonor:
    """A hydrogen bond donor."""
    heavy_atom_idx: int      # Index of N or O
    h_atom_idx: int          # Index of attached H
    position: jnp.ndarray    # (3,) Position
    direction: jnp.ndarray   # (3,) Direction (heavy -> H)

@dataclass
class HydrogenBondAcceptor:
    """A hydrogen bond acceptor."""
    atom_idx: int            # Index of O or N
    position: jnp.ndarray    # (3,) Position
    lone_pair_direction: jnp.ndarray  # (3,) Approximate direction

def detect_hbond_donors(
    coords: jnp.ndarray,
    elements: list[str],
    connectivity: list[list[int]]  # Adjacency list
) -> list[HydrogenBondDonor]:
    """
    Detect hydrogen bond donors.

    Rules:
    - N-H, O-H groups are donors
    - Must have attached hydrogen
    """
    donors = []

    for i, (coord, elem) in enumerate(zip(coords, elements)):
        if elem in ['N', 'O']:
            # Find attached hydrogens
            for j in connectivity[i]:
                if elements[j] == 'H':
                    donors.append(HydrogenBondDonor(
                        heavy_atom_idx=i,
                        h_atom_idx=j,
                        position=coord,
                        direction=coords[j] - coord  # Direction of H
                    ))

    return donors

def detect_hbond_acceptors(
    coords: jnp.ndarray,
    elements: list[str]
) -> list[HydrogenBondAcceptor]:
    """
    Detect hydrogen bond acceptors.

    Rules:
    - O, N with lone pairs are acceptors
    - Direction estimated from geometry
    """
    acceptors = []

    for i, (coord, elem) in enumerate(zip(coords, elements)):
        if elem in ['O', 'N']:
            # Estimate lone pair direction (simplified)
            # In reality, need to look at hybridization
            acceptors.append(HydrogenBondAcceptor(
                atom_idx=i,
                position=coord,
                lone_pair_direction=jnp.array([0.0, 0.0, 0.0])  # Unknown
            ))

    return acceptors
```

### 2.2 Hydrogen Bond Scoring

**Geometric criteria**:
- Distance: 1.5 - 2.5 Å (H...acceptor)
- Angle: > 120° (donor-H...acceptor)

```python
@jax.jit
def hydrogen_bond_energy(
    donor: HydrogenBondDonor,
    acceptor: HydrogenBondAcceptor,
    optimal_distance: float = 2.0,
    optimal_angle: float = 180.0
) -> float:
    """
    Compute hydrogen bond energy using geometric criteria.

    E = E_distance * E_angle

    where both terms are 0-1 (1 = optimal)
    """
    # Distance component (Gaussian)
    h_pos = donor.position + donor.direction  # H position (approximate)
    r = jnp.linalg.norm(h_pos - acceptor.position)

    E_distance = jnp.exp(-((r - optimal_distance) ** 2) / (2 * 0.5 ** 2))

    # Angle component
    # Angle between donor->H and H->acceptor vectors
    vec_dh = donor.direction
    vec_ha = acceptor.position - h_pos
    vec_ha = vec_ha / jnp.linalg.norm(vec_ha)

    cos_angle = jnp.dot(vec_dh, vec_ha)
    angle = jnp.arccos(jnp.clip(cos_angle, -1.0, 1.0))
    angle_degrees = jnp.degrees(angle)

    E_angle = jnp.exp(-((angle_degrees - optimal_angle) ** 2) / (2 * 30.0 ** 2))

    # Total energy (negative = favorable)
    # Typical H-bond strength: -1 to -5 kcal/mol
    E_max = -5.0  # Maximum strength
    return E_max * E_distance * E_angle

@jax.jit
def total_hbond_energy(
    pose_coords: jnp.ndarray,
    pose_elements: list[str],
    pose_connectivity: list[list[int]],
    receptor_coords: jnp.ndarray,
    receptor_elements: list[str],
    receptor_connectivity: list[list[int]]
) -> float:
    """
    Compute total hydrogen bond energy.

    Sums over all donor-acceptor pairs within cutoff.
    """
    # Detect donors and acceptors
    pose_donors = detect_hbond_donors(pose_coords, pose_elements, pose_connectivity)
    pose_acceptors = detect_hbond_acceptors(pose_coords, pose_elements)
    receptor_donors = detect_hbond_donors(receptor_coords, receptor_elements, receptor_connectivity)
    receptor_acceptors = detect_hbond_acceptors(receptor_coords, receptor_elements)

    total_energy = 0.0

    # Pose donor -> receptor acceptor
    for donor in pose_donors:
        for acceptor in receptor_acceptors:
            # Check distance cutoff first
            if jnp.linalg.norm(donor.position - acceptor.position) < 4.0:
                energy = hydrogen_bond_energy(donor, acceptor)
                total_energy += energy

    # Receptor donor -> pose acceptor
    for donor in receptor_donors:
        for acceptor in pose_acceptors:
            if jnp.linalg.norm(donor.position - acceptor.position) < 4.0:
                energy = hydrogen_bond_energy(donor, acceptor)
                total_energy += energy

    return total_energy
```

**Simplified version** (no connectivity required):
```python
@jax.jit
def hbond_energy_simple(
    pose_coords: jnp.ndarray,
    pose_elements: list[str],
    receptor_coords: jnp.ndarray,
    receptor_elements: list[str]
) -> float:
    """
    Simplified H-bond scoring without connectivity.

    Assumes all N/O can participate in H-bonds.
    Less accurate but doesn't require bond information.
    """
    # Find all N and O atoms
    pose_no_indices = [i for i, e in enumerate(pose_elements) if e in ['N', 'O']]
    rec_no_indices = [i for i, e in enumerate(receptor_elements) if e in ['N', 'O']]

    total_energy = 0.0

    for i in pose_no_indices:
        for j in rec_no_indices:
            r = jnp.linalg.norm(pose_coords[i] - receptor_coords[j])

            # Simple distance-based scoring
            # Optimal: 2.8 - 3.2 Å
            if 2.5 < r < 3.5:
                energy = -1.0 * jnp.exp(-((r - 3.0) ** 2) / (2 * 0.3 ** 2))
                total_energy += energy

    return total_energy
```

---

## Phase 3: Desolvation

### 3.1 Atomic Solvation Parameters

**Goal**: Penalize removal of water from polar/charged surfaces

```python
# Atomic solvation parameters (cal/mol/Å²)
# From: Eisenberg & McLachlan (1986)
SOLVATION_PARAMETERS = {
    'C': 16.0,   # Hydrophobic (positive = unfavorable to desolvate)
    'N': -11.0,  # Polar (negative = favorable)
    'O': -11.0,  # Polar
    'S': 21.0,   # Hydrophobic
    'NA': -20.0, # Charged
    'K': -20.0,
    'CL': -20.0,
    'BR': -20.0,
}

@jax.jit
def desolvation_energy(
    pose_coords: jnp.ndarray,
    pose_elements: list[str],
    receptor_coords: jnp.ndarray,
    receptor_elements: list[str],
    probe_radius: float = 1.4
) -> float:
    """
    Compute desolvation energy using atomic solvation parameters.

    E_desolv = sum(ASA_i * sigma_i)

    where ASA_i is accessible surface area and sigma_i is solvation parameter.

    Simplified: count buried atoms
    """
    # Count buried atoms (neighbors within probe_radius)
    pose_buried = jnp.zeros(len(pose_coords))
    rec_buried = jnp.zeros(len(receptor_coords))

    for i, coord in enumerate(pose_coords):
        distances = jnp.linalg.norm(receptor_coords - coord, axis=1)
        n_contacts = jnp.sum(distances < (probe_radius + 2.0))  # 2Å buffer
        if n_contacts > 0:
            pose_buried = pose_buried.at[i].set(1.0)

    for i, coord in enumerate(receptor_coords):
        distances = jnp.linalg.norm(pose_coords - coord, axis=1)
        n_contacts = jnp.sum(distances < (probe_radius + 2.0))
        if n_contacts > 0:
            rec_buried = rec_buried.at[i].set(1.0)

    # Compute desolvation energy
    pose_energy = 0.0
    for i, (elem, buried) in enumerate(zip(pose_elements, pose_buried)):
        if buried:
            sigma = SOLVATION_PARAMETERS.get(elem.upper(), 0.0)
            # Assume ~10 Å² buried area per atom
            pose_energy += 10.0 * sigma

    rec_energy = 0.0
    for i, (elem, buried) in enumerate(zip(receptor_elements, rec_buried)):
        if buried:
            sigma = SOLVATION_PARAMETERS.get(elem.upper(), 0.0)
            rec_energy += 10.0 * sigma

    # Total desolvation (convert cal to kcal)
    return (pose_energy + rec_energy) / 1000.0
```

### 3.2 Hydrophobic Effect

**Goal**: Reward burial of hydrophobic surface

```python
@jax.jit
def hydrophobic_energy(
    pose_coords: jnp.ndarray,
    pose_elements: list[str],
    receptor_coords: jnp.ndarray,
    receptor_elements: list[str],
    contact_distance: float = 4.5
) -> float:
    """
    Compute hydrophobic interaction energy.

    Rewards contacts between hydrophobic atoms.
    """
    # Identify hydrophobic atoms
    pose_hydrophobic = [i for i, e in enumerate(pose_elements) if e in ['C', 'S']]
    rec_hydrophobic = [i for i, e in enumerate(receptor_elements) if e in ['C', 'S']]

    n_contacts = 0
    for i in pose_hydrophobic:
        for j in rec_hydrophobic:
            r = jnp.linalg.norm(pose_coords[i] - receptor_coords[j])
            if r < contact_distance:
                n_contacts += 1

    # Each contact contributes ~-0.5 kcal/mol
    return -0.5 * n_contacts
```

---

## Integration: Combined Scoring Function

```python
# File: dq_dock_engine/docking/scoring_advanced.py

@jax.jit
def score_advanced(
    pose_coords: jnp.ndarray,
    receptor_coords: jnp.ndarray,
    receptor_radii: jnp.ndarray,
    ligand_radii: jnp.ndarray,
    pose_charges: jnp.ndarray,
    receptor_charges: jnp.ndarray,
    pose_elements: list[str],
    receptor_elements: list[str],
    # Weights
    w_lj: float = 1.0,
    w_elec: float = 0.1,
    w_hb: float = 1.0,
    w_desolv: float = 0.5,
    w_hydro: float = 0.5,
    # LJ parameters
    repulsion_weight: float = 3.0,
    attraction_weight: float = 1.5,
) -> float:
    """
    Combined scoring function with multiple physics terms.

    E_total = w_lj * E_LJ + w_elec * E_elec + w_hb * E_hb +
              w_desolv * E_desolv + w_hydro * E_hydro
    """
    # Lennard-Jones
    E_LJ = _score_single_lj(
        pose_coords, receptor_coords,
        receptor_radii, ligand_radii,
        repulsion_weight, attraction_weight
    )

    # Electrostatics
    E_elec = coulomb_energy_distance_dependent(
        pose_coords, pose_charges,
        receptor_coords, receptor_charges
    )

    # Hydrogen bonding
    E_hb = hbond_energy_simple(
        pose_coords, pose_elements,
        receptor_coords, receptor_elements
    )

    # Desolvation
    E_desolv = desolvation_energy(
        pose_coords, pose_elements,
        receptor_coords, receptor_elements
    )

    # Hydrophobic
    E_hydro = hydrophobic_energy(
        pose_coords, pose_elements,
        receptor_coords, receptor_elements
    )

    # Combined score
    E_total = (
        w_lj * E_LJ +
        w_elec * E_elec +
        w_hb * E_hb +
        w_desolv * E_desolv +
        w_hydro * E_hydro
    )

    return E_total
```

---

## Parameter Optimization

### Weight Optimization Strategy

**Goal**: Find optimal weights for each scoring term

```python
# File: dq_dock_engine/benchmark/optimize_weights.py

def optimize_scoring_weights():
    """
    Grid search over weight combinations.

    Test on validation set of complexes with known binding modes.
    """
    # Weight grid
    W_ELEC = [0.0, 0.05, 0.1, 0.2, 0.5]
    W_HB = [0.0, 0.5, 1.0, 2.0, 5.0]
    W_DESOLV = [0.0, 0.25, 0.5, 1.0]
    W_HYDRO = [0.0, 0.25, 0.5, 1.0]

    best_config = None
    best_rmsd = float('inf')

    for w_elec, w_hb, w_desolv, w_hydro in itertools.product(
        W_ELEC, W_HB, W_DESOLV, W_HYDRO
    ):
        # Run docking on validation set
        rmsds = []
        for complex_id in VALIDATION_SET:
            result = run_docking_with_weights(
                complex_id,
                w_elec=w_elec, w_hb=w_hb,
                w_desolv=w_desolv, w_hydro=w_hydro
            )
            rmsds.append(result['rmsd'])

        avg_rmsd = jnp.mean(jnp.array(rmsds))

        if avg_rmsd < best_rmsd:
            best_rmsd = avg_rmsd
            best_config = {
                'w_elec': w_elec,
                'w_hb': w_hb,
                'w_desolv': w_desolv,
                'w_hydro': w_hydro
            }

    print(f"Best config: {best_config}")
    print(f"Best RMSD: {best_rmsd:.2f} Å")

    return best_config
```

---

## Performance Optimization

### Caching Strategy

```python
# Precompute per-receptor quantities
@dataclass
class ReceptorCache:
    """Precomputed receptor data for fast scoring."""
    coords: jnp.ndarray
    radii: jnp.ndarray
    charges: jnp.ndarray
    elements: list[str]
    spatial_hash: dict  # For cutoffs
    hbond_donors: list
    hbond_acceptors: list

def build_receptor_cache(
    receptor_coords: jnp.ndarray,
    receptor_elements: list[str]
) -> ReceptorCache:
    """
    Build cache for repeated scoring.
    """
    return ReceptorCache(
        coords=receptor_coords,
        radii=compute_vdw_radii(receptor_elements),
        charges=assign_charges_simple(receptor_elements),
        elements=receptor_elements,
        spatial_hash=build_spatial_hash(receptor_coords),
        hbond_donors=detect_hbond_donors(receptor_coords, receptor_elements, []),
        hbond_acceptors=detect_hbond_acceptors(receptor_coords, receptor_elements)
    )
```

---

## Implementation Plan

### Phase 1: Electrostatics (2 days)

**Tasks**:
1. Implement charge assignment (simple rules)
2. Implement basic Coulomb potential
3. Implement distance-dependent dielectric
4. Add cutoff for performance
5. Unit tests

**Validation**:
- Test on salt bridge (known geometry)
- Verify energy trends make sense
- Benchmark performance

### Phase 2: Hydrogen Bonding (2 days)

**Tasks**:
1. Implement donor/acceptor detection (simple)
2. Implement geometric H-bond scoring
3. Implement simplified version
4. Unit tests

**Validation**:
- Test on known H-bond patterns
- Verify angular dependence
- Check energy magnitudes

### Phase 3: Desolvation (1 day)

**Tasks**:
1. Implement atomic solvation parameters
2. Implement burial counting
3. Implement hydrophobic term
4. Unit tests

**Validation**:
- Verify hydrophobic burial is rewarded
- Check polar desolvation penalty

### Phase 4: Integration and Optimization (2 days)

**Tasks**:
1. Implement combined scoring function
2. Add weight configuration
3. Implement weight optimization
4. Run parameter sweep
5. Performance profiling

**Validation**:
- RMSD improvement ≥ 0.5 Å
- Correlation with binding affinity (if data available)
- Acceptable performance (< 5ms per pose)

---

## Testing Strategy

### Unit Tests

```python
def test_coulomb_energy():
    """Test Coulomb energy on simple system."""
    # Opposite charges should attract
    pass

def test_hbond_geometry():
    """Test H-bond angular dependence."""
    # 180° should be strongest
    # 90° should be weak
    pass

def test_desolvation_sign():
    """Test desolvation has correct sign."""
    # Desolvating hydrophobics should be favorable (negative)
    # Desolvating polars should be unfavorable (positive)
    pass
```

### Integration Tests

```python
def test_combined_scoring():
    """Test combined scoring function."""
    # Should reproduce known trends
    pass

def test_weight_optimization():
    """Test weight optimization converges."""
    pass
```

---

## Risk Assessment

**Risk**: Performance degradation too severe
- **Probability**: High
- **Impact**: Medium
- **Mitigation**:
  - Use cutoffs aggressively
  - Cache precomputed quantities
  - Profile and optimize hotspots

**Risk**: Overfitting to training set
- **Probability**: Medium
- **Impact**: High
- **Mitigation**:
  - Use separate validation and test sets
  - Cross-validation
  - Conservative weight ranges

**Risk**: Charge assignment errors
- **Probability**: Medium
- **Impact**: High
- **Mitigation**:
  - Start with simple rules
  - Upgrade to AM1-BCC if needed
  - Visualize charges for sanity check

---

## Success Criteria

- ✅ RMSD improvement ≥ 0.5 Å on validation set
- ✅ Correlation with experimental affinity (if available)
- ✅ Performance < 5ms per pose
- ✅ All tests pass

---

## Hand-wavy gaps to fill:

1. **Optimal weight values?**
   - Need empirical optimization
   - Likely target-dependent
   - Should test on diverse set

2. **Charge assignment method?**
   - Simple rules vs AM1-BCC
   - Trade-off: accuracy vs speed
   - Should compare both

3. **H-bond directionality?**
   - Need connectivity info
   - Can approximate from geometry
   - Simplified version may be sufficient

4. **Performance bottlenecks?**
   - Electrostatics is O(N²)
   - Need spatial hashing
   - Profile to identify hotspots

**Action**: Implement and profile to answer these questions.

---

## Next Steps

1. **Implement electrostatics** (2 days)
2. **Implement H-bonds** (2 days)
3. **Implement desolvation** (1 day)
4. **Integrate and optimize** (2 days)

**Total timeline**: 7 days to fully validated advanced scoring
