# Pocket-Guided Pose Sampling Plan

**Goal**: Implement shape-aware pose sampling for 0.5-1.0 Å RMSD improvement

**Status**: 🔄 Planning
**Priority**: 🥉 MEDIUM - Significant impact but more complex
**Expected Impact**: 0.5-1.0 Å RMSD improvement
**Expected Speed Impact**: Neutral (better poses with same sampling)

---

## Problem Statement

### Current Limitation: Blind Random Sampling
```python
# Current: dq_dock_engine/docking/placement.py:35-47
def sample_random_poses(key: jax.Array, box: DockingBox, n_poses: int) -> PoseVector:
    """Pure geometric sampling within the constraints of the DockingBox."""
    translations = jax.random.uniform(
        key_t, shape=(n_poses, 3),
        minval=box.center - half_size,
        maxval=box.center + half_size
    )
    quaternions = _uniform_quaternions(key_r, n_poses)
```

**Inefficiency**:
- Uniform sampling wastes effort on irrelevant regions
- No consideration of pocket shape or geometry
- All rotations equally likely (should be biased toward pocket)

### Solution: Intelligence-Guided Sampling

**Key insight**: Binding poses are not uniformly distributed. They cluster around:
1. Pocket sub-sites (concave regions)
2. Shape-complementary orientations
3. Pharmacophore features (H-bonds, hydrophobic patches)

---

## Architecture Design

### Overview Pipeline

```
Input: Protein coords + Pocket definition
  ↓
1. Pocket Analysis (precompute once)
   - Detect sub-pockets
   - Compute shape descriptors
   - Identify pharmacophore features
  ↓
2. Sampling Strategy Generation
   - Weight sub-pockets by volume/accessibility
   - Generate pose templates for each region
  ↓
3. Intelligent Pose Sampling
   - Sample translations biased toward sub-pockets
   - Sample rotations biased toward shape complementarity
  ↓
Output: Concentrated sampling in relevant regions
```

---

## Phase 1: Pocket Analysis

### 1.1 Pocket Subdivision

**Goal**: Identify distinct binding sub-sites within pocket

**Algorithm: Grid-based clustering**

```python
# File: dq_dock_engine/docking/pocket_analysis.py

from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform

def identify_subpockets(
    pocket_coords: jnp.ndarray,
    pocket_center: jnp.ndarray,
    min_subpocket_size: int = 10,
    distance_cutoff: float = 4.0
) -> list[jnp.ndarray]:
    """
    Identify sub-pockets using hierarchical clustering.

    Args:
        pocket_coords: (N, 3) pocket atom coordinates
        pocket_center: (3,) center of binding pocket
        min_subpocket_size: Minimum atoms per sub-pocket
        distance_cutoff: Distance for clustering (Å)

    Returns:
        List of (M_i, 3) arrays, one per sub-pocket
    """
    # Compute pairwise distances
    distances = pdist(pocket_coords)

    # Hierarchical clustering
    Z = hierarchy.linkage(distances, method='ward')

    # Cut tree to get clusters
    clusters = hierarchy.fcluster(Z, t=distance_cutoff, criterion='distance')

    # Filter small clusters
    unique_clusters = set(clusters)
    subpockets = []

    for cluster_id in unique_clusters:
        mask = clusters == cluster_id
        if jnp.sum(mask) >= min_subpocket_size:
            subpockets.append(pocket_coords[mask])

    return subpockets
```

**Validation**:
```python
def test_subpocket_detection():
    """Test sub-pocket detection on known pockets."""
    # HIV protease: Should detect 2 sub-sites (flaps)
    # Kinase: Should detect ATP-binding region + selectivity pocket
    pass
```

### 1.2 Shape Descriptors

**Goal**: Quantify pocket shape for sampling bias

**Descriptors to compute**:

```python
@dataclass
class PocketShape:
    """Geometric descriptors of pocket shape."""
    center_of_mass: jnp.ndarray        # (3,) COM
    principal_axes: jnp.ndarray        # (3, 3) PCA axes
    extents: jnp.ndarray               # (3,) Extent along each axis
    volume: float                      # Estimated volume
    concavity: float                   # 0=flat, 1=spherical
    openness: tuple[float, float, float]  # Openness along each axis

def compute_pocket_shape(pocket_coords: jnp.ndarray) -> PocketShape:
    """
    Compute geometric shape descriptors of pocket.

    Uses PCA to identify principal axes of pocket.
    """
    # Center of mass
    com = jnp.mean(pocket_coords, axis=0)

    # Principal component analysis
    centered = pocket_coords - com
    cov = jnp.cov(centered.T)
    eigenvalues, eigenvectors = jnp.linalg.eigh(cov)

    # Sort by eigenvalue (largest first)
    sort_idx = jnp.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]

    # Extents (3 std devs along each axis)
    extents = 3 * jnp.sqrt(eigenvalues)

    # Volume (approximate as ellipsoid)
    volume = (4/3) * jnp.pi * jnp.prod(extents / 3)

    # Concavity: ratio of smallest to largest eigenvalue
    concavity = eigenvalues[2] / eigenvalues[0]

    # Openness: Distance to farthest atom along each axis
    projections = centered @ eigenvectors
    openness = (
        jnp.max(projections[:, 0]) - jnp.min(projections[:, 0]),
        jnp.max(projections[:, 1]) - jnp.min(projections[:, 1]),
        jnp.max(projections[:, 2]) - jnp.min(projections[:, 2]),
    )

    return PocketShape(
        center_of_mass=com,
        principal_axes=eigenvectors,
        extents=extents,
        volume=volume,
        concavity=concavity,
        openness=openness
    )
```

### 1.3 Surface Accessibility

**Goal**: Identify accessible surface regions for ligand placement

**Algorithm: Ray casting or neighbor counting**

```python
def compute_accessibility(
    pocket_coords: jnp.ndarray,
    probe_radius: float = 1.4  # Water molecule radius
) -> jnp.ndarray:
    """
    Compute solvent accessibility for each pocket atom.

    Returns: (N,) array of accessibility scores (0-1)
    """
    n_atoms = len(pocket_coords)
    accessibility = jnp.zeros(n_atoms)

    for i in range(n_atoms):
        # Count neighbors within probe radius
        distances = jnp.linalg.norm(pocket_coords - pocket_coords[i], axis=1)

        # Atom is accessible if few neighbors around it
        n_neighbors = jnp.sum(distances < probe_radius)

        # Normalize (0 = buried, 1 = fully exposed)
        accessibility = jnp.exp(-n_neighbors / 10.0)

    return accessibility
```

### 1.4 Pharmacophore Feature Detection

**Goal**: Identify key interaction points (H-bonds, charges, hydrophobics)

```python
@dataclass
class PharmacophoreFeature:
    """A pharmacophore feature (interaction point)."""
    position: jnp.ndarray           # (3,) Location
    feature_type: str              # 'HBA', 'HBD', 'HYD', 'POS', 'NEG'
    direction: jnp.ndarray         # (3,) Direction (for H-bonds)
    strength: float                # Interaction strength

def detect_pharmacophore_features(
    pocket_coords: jnp.ndarray,
    pocket_elements: list[str],
    accessibility: jnp.ndarray
) -> list[PharmacophoreFeature]:
    """
    Detect pharmacophore features in pocket.

    Rules:
    - H-bond acceptor: O, N with high accessibility
    - H-bond donor: N-H, O-H with high accessibility
    - Hydrophobic: C, S with low accessibility
    - Positive: NH3+, metal ions
    - Negative: COO-, phosphate
    """
    features = []

    for i, (coord, elem, acc) in enumerate(zip(pocket_coords, pocket_elements, accessibility)):
        # Skip buried atoms
        if acc < 0.3:
            continue

        if elem == 'O':
            # Could be H-bond acceptor
            features.append(PharmacophoreFeature(
                position=coord,
                feature_type='HBA',
                direction=jnp.array([0.0, 0.0, 0.0]),  # Unknown direction
                strength=acc
            ))

        elif elem == 'N':
            # Could be H-bond donor or acceptor
            features.append(PharmacophoreFeature(
                position=coord,
                feature_type='HBD',
                direction=jnp.array([0.0, 0.0, 0.0]),
                strength=acc
            ))

        elif elem in ['C', 'S']:
            # Hydrophobic
            features.append(PharmacophoreFeature(
                position=coord,
                feature_type='HYD',
                direction=jnp.array([0.0, 0.0, 0.0]),
                strength=1.0 - acc  # Strength = burial
            ))

    return features
```

---

## Phase 2: Sampling Strategy Generation

### 2.1 Sub-pocket Weighting

**Goal**: Assign sampling weights to sub-pockets based on properties

```python
def compute_subpocket_weights(
    subpockets: list[jnp.ndarray],
    pharmacophore_features: list[PharmacophoreFeature],
    base_weight: float = 1.0
) -> list[float]:
    """
    Compute sampling weights for each sub-pocket.

    Factors:
    - Volume (larger = more poses)
    - Feature density (more features = higher weight)
    - Accessibility (more accessible = higher weight)
    """
    weights = []

    for subpocket in subpockets:
        # Base weight by volume (number of atoms)
        weight = len(subpocket) * base_weight

        # Bonus for pharmacophore features
        n_features = count_features_in_region(
            pharmacophore_features, subpocket, radius=5.0
        )
        weight *= (1 + 0.5 * n_features)

        # Normalize
        weights.append(weight)

    # Normalize to sum to 1
    total = sum(weights)
    weights = [w / total for w in weights]

    return weights
```

### 2.2 Pose Template Generation

**Goal**: Generate promising initial orientations for each sub-pocket

```python
@dataclass
class PoseTemplate:
    """A template pose for sampling."""
    translation: jnp.ndarray      # (3,) Preferred location
    rotation_bias: jnp.ndarray    # (4,) Preferred quaternion [w,x,y,z]
    uncertainty: tuple[float, float]  # (trans_var, rot_var) in radians

def generate_pose_templates(
    subpocket: jnp.ndarray,
    pocket_shape: PocketShape,
    n_templates: int = 10
) -> list[PoseTemplate]:
    """
    Generate pose templates for a sub-pocket.

    Strategy:
    1. Place ligand at sub-pocket COM
    2. Orient along principal axes
    3. Add randomness for coverage
    """
    templates = []

    for i in range(n_templates):
        # Translation: Sub-pocket COM with small variation
        translation = pocket_shape.center_of_mass + jax.random.normal(
            jax.random.PRNGKey(i), shape=(3,)
        ) * 1.0  # 1Å standard deviation

        # Rotation: Align with principal axes
        # For each template, rotate around one principal axis
        axis_idx = i % 3
        axis = pocket_shape.principal_axes[:, axis_idx]

        # Create quaternion for rotation around axis
        angle = (2 * jnp.pi / n_templates) * (i // 3)
        quaternion = axis_angle_to_quaternion(axis, angle)

        templates.append(PoseTemplate(
            translation=translation,
            rotation_bias=quaternion,
            uncertainty=(1.0, 0.5)  # 1Å translation, 0.5 rad rotation
        ))

    return templates

def axis_angle_to_quaternion(axis: jnp.ndarray, angle: float) -> jnp.ndarray:
    """Convert axis-angle representation to quaternion."""
    half_angle = angle / 2.0
    sin_half = jnp.sin(half_angle)

    return jnp.array([
        jnp.cos(half_angle),      # w
        axis[0] * sin_half,       # x
        axis[1] * sin_half,       # y
        axis[2] * sin_half        # z
    ])
```

---

## Phase 3: Intelligent Pose Sampling

### 3.1 Biased Translation Sampling

**Goal**: Sample translations concentrated around sub-pockets

```python
def sample_biased_translations(
    key: jax.Array,
    pose_templates: list[PoseTemplate],
    subpocket_weights: list[float],
    n_poses: int
) -> jnp.ndarray:
    """
    Sample translations biased toward sub-pockets.

    Uses Gaussian mixture model:
    - Each template is a Gaussian component
    - Weights determine sampling frequency
    """
    # Select templates for each pose (weighted by subpocket_weights)
    template_indices = jax.random.categorical(
        key, jnp.array(subpocket_weights), shape=(n_poses,)
    )

    translations = []
    for i, template_idx in enumerate(template_indices):
        template = pose_templates[template_idx]

        # Sample from Gaussian around template
        key_i = jax.random.fold_in(key, i)
        translation = jax.random.normal(
            key_i, shape=(3,)
        ) * template.uncertainty[0] + template.translation

        translations.append(translation)

    return jnp.stack(translations)
```

### 3.2 Biased Rotation Sampling

**Goal**: Sample rotations biased toward shape complementarity

```python
def sample_biased_rotations(
    key: jax.Array,
    pose_templates: list[PoseTemplate],
    template_indices: jnp.ndarray,
    n_poses: int
) -> jnp.ndarray:
    """
    Sample rotations biased toward template orientations.

    Uses von Mises-Fisher distribution for quaternions.
    """
    quaternions = []

    for i, template_idx in enumerate(template_indices):
        template = pose_templates[template_idx]

        # Sample around template quaternion
        key_i = jax.random.fold_in(key, i)

        # Use Shoemake's algorithm with bias
        # (Simplified: just add small noise to template)
        noise = jax.random.normal(key_i, shape=(4,)) * template.uncertainty[1]
        q_noisy = template.rotation_bias + noise

        # Normalize
        q_noisy = q_noisy / jnp.linalg.norm(q_noisy)

        quaternions.append(q_noisy)

    return jnp.stack(quaternions)
```

### 3.3 Pharmacophore-Guided Sampling

**Goal**: Bias poses to satisfy pharmacophore constraints

```python
def sample_pharmacophore_guided(
    key: jax.Array,
    ligand_ctx: LigandContext,
    pharmacophore_features: list[PharmacophoreFeature],
    n_poses: int
) -> PoseVector:
    """
    Sample poses guided by pharmacophore features.

    Strategy:
    1. Select random feature as "anchor"
    2. Place ligand near feature
    3. Orient to maximize feature complementarity
    """
    poses = []

    for i in range(n_poses):
        # Select anchor feature
        key_i = jax.random.fold_in(key, i)
        anchor = jax.random.choice(
            key_i,
            len(pharmacophore_features)
        )
        feature = pharmacophore_features[anchor]

        # Place ligand near feature
        ligand_com = ligand_ctx.center_of_mass

        # Translation: Feature position + small offset
        offset = jax.random.normal(key_i, shape=(3,)) * 2.0  # 2Å variation
        translation = feature.position + offset - ligand_com

        # Rotation: Align toward feature
        if feature.feature_type in ['HBA', 'HBD']:
            # Orient donor/acceptor groups toward feature
            # (Simplified: random rotation with bias)
            rotation = _uniform_quaternions(key_i, 1)[0]
        else:
            rotation = _uniform_quaternions(key_i, 1)[0]

        poses.append((translation, rotation))

    translations = jnp.stack([p[0] for p in poses])
    quaternions = jnp.stack([p[1] for p in poses])

    return PoseVector(translation=translations, quaternion=quaternions)
```

---

## Integration with Existing Pipeline

### Modified Sampling Function

**File**: `dq_dock_engine/docking/placement.py`

```python
def sample_intelligent_poses(
    key: jax.Array,
    box: DockingBox,
    n_poses: int,
    protein_coords: jnp.ndarray,
    ligand_ctx: LigandContext,
    sampling_strategy: str = "guided"  # "random", "guided", "hybrid"
) -> PoseVector:
    """
    Intelligent pose sampling using pocket analysis.

    Args:
        key: JAX random key
        box: Docking box constraints
        n_poses: Number of poses to sample
        protein_coords: Protein coordinates for pocket analysis
        ligand_ctx: Ligand context
        sampling_strategy:
            - "random": Pure random (baseline)
            - "guided": Fully guided by pockets
            - "hybrid": 50% guided, 50% random

    Returns:
        PoseVector with biased translations and rotations
    """
    if sampling_strategy == "random":
        return sample_random_poses(key, box, n_poses)

    # Analyze pocket
    pocket_center = box.center
    pocket_radius = box.size[0] / 2.0

    # Extract pocket atoms
    distances = jnp.linalg.norm(protein_coords - pocket_center, axis=1)
    pocket_mask = distances < pocket_radius
    pocket_coords = protein_coords[pocket_mask]

    # Sub-pocket detection
    subpockets = identify_subpockets(pocket_coords, pocket_center)

    # Shape analysis
    pocket_shape = compute_pocket_shape(pocket_coords)

    # Pharmacophore features
    accessibility = compute_accessibility(pocket_coords)
    pocket_elements = ['C'] * len(pocket_coords)  # TODO: get actual elements
    features = detect_pharmacophore_features(pocket_coords, pocket_elements, accessibility)

    # Generate templates
    all_templates = []
    subpocket_weights = compute_subpocket_weights(subpockets, features)

    for subpocket, weight in zip(subpockets, subpocket_weights):
        subpocket_shape = compute_pocket_shape(subpocket)
        templates = generate_pose_templates(subpocket, subpocket_shape, n_templates=10)
        all_templates.extend(templates)

    # Sample poses
    n_guided = n_poses if sampling_strategy == "guided" else n_poses // 2
    n_random = n_poses - n_guided

    # Guided sampling
    key_guided, key_random = jax.random.split(key)
    guided_translations = sample_biased_translations(
        key_guided, all_templates, subpocket_weights, n_guided
    )
    guided_quaternions = sample_biased_rotations(
        key_guided, all_templates, jax.random.categorical(key_guided, jnp.array(subpocket_weights), shape=(n_guided,))
    )

    # Random sampling (for hybrid mode)
    if n_random > 0:
        random_poses = sample_random_poses(key_random, box, n_random)
        translations = jnp.concatenate([guided_translations, random_poses.translation])
        quaternions = jnp.concatenate([guided_quaternions, random_poses.quaternion])
    else:
        translations = guided_translations
        quaternions = guided_quaternions

    return PoseVector(translation=translations, quaternion=quaternions)
```

---

## Performance Analysis

### Expected Sampling Efficiency

**Metric**: Hit rate (poses with RMSD < 2Å)

| Method | Hit Rate | Sampling Required |
|--------|----------|-------------------|
| Random uniform | 0.1% | 1000 poses per hit |
| Pocket-guided | 1.0% | 100 poses per hit |
| Pharmacophore-guided | 2.0% | 50 poses per hit |

**Implication**: With pocket-guided sampling, we can achieve same coverage with 10x fewer poses.

### Timing Analysis

| Component | Time | Frequency |
|-----------|------|-----------|
| Pocket analysis | 0.1s | Once per receptor |
| Template generation | 0.05s | Once per receptor |
| Guided sampling | 0.0001s/pose | Per pose |
| **Total overhead** | **0.15s** | Amortized over all poses |

For 10,000 poses: 0.15s + 10,000 × 0.0001s = 1.15s (vs 0.03s for random)

**Trade-off**: 1.12s overhead for 10x better hit rate

---

## Implementation Plan

### Phase 1: Pocket Analysis (2 days)

**Tasks**:
1. Implement `identify_subpockets()` with hierarchical clustering
2. Implement `compute_pocket_shape()` with PCA
3. Implement `compute_accessibility()` with neighbor counting
4. Implement `detect_pharmacophore_features()` with simple rules
5. Unit tests for each function

**Validation**:
- Test subpocket detection on known pockets (HIV protease, kinases)
- Verify shape descriptors make sense
- Check feature detection finds expected interactions

### Phase 2: Template Generation (1 day)

**Tasks**:
1. Implement `compute_subpocket_weights()`
2. Implement `generate_pose_templates()`
3. Implement `axis_angle_to_quaternion()`
4. Unit tests

**Validation**:
- Verify templates are within pocket
- Check orientation diversity
- Test coverage of pocket volume

### Phase 3: Biased Sampling (2 days)

**Tasks**:
1. Implement `sample_biased_translations()` with Gaussian mixture
2. Implement `sample_biased_rotations()` with von Mises-Fisher
3. Implement `sample_pharmacophore_guided()`
4. Implement `sample_intelligent_poses()` integration
5. Unit tests

**Validation**:
- Verify samples cluster around templates
- Check diversity not too low (avoid collapse)
- Test hit rate improvement

### Phase 4: Integration and Testing (1 day)

**Tasks**:
1. Update `pipeline.py` to use intelligent sampling
2. Add configuration options (strategy, weights)
3. Run benchmarks comparing random vs guided
4. Optimize performance bottlenecks

**Validation**:
- RMSD improvement ≥ 0.5 Å
- Acceptable overhead (< 2s)
- No regression in scoring quality

---

## Testing Strategy

### Unit Tests

**File**: `tests/docking/test_pocket_analysis.py`

```python
def test_subpocket_detection():
    """Test sub-pocket detection on synthetic pocket."""
    # Create pocket with 2 clear clusters
    coords = jnp.concatenate([
        jnp.zeros((10, 3)),  # Cluster 1
        jnp.ones((10, 3)) * 10.0  # Cluster 2
    ])

    subpockets = identify_subpockets(coords, center=jnp.zeros(3))

    assert len(subpockets) == 2
    assert len(subpockets[0]) == 10
    assert len(subpockets[1]) == 10

def test_pocket_shape():
    """Test pocket shape computation."""
    # Spherical pocket
    theta = jnp.linspace(0, 2*jnp.pi, 100)
    phi = jnp.linspace(0, jnp.pi, 100)
    coords = jnp.stack([
        jnp.cos(theta) * jnp.sin(phi),
        jnp.sin(theta) * jnp.sin(phi),
        jnp.cos(phi)
    ], axis=1) * 5.0  # Radius 5

    shape = compute_pocket_shape(coords)

    # Should be roughly spherical
    assert shape.concavity > 0.8
    assert jnp.allclose(shape.extents, shape.extents[0], atol=1.0)

def test_pharmacophore_detection():
    """Test pharmacophore feature detection."""
    # Pocket with O and N atoms
    coords = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    elements = ['O', 'N']
    accessibility = jnp.array([1.0, 1.0])

    features = detect_pharmacophore_features(coords, elements, accessibility)

    assert len(features) == 2
    assert features[0].feature_type == 'HBA'
    assert features[1].feature_type == 'HBD'
```

### Integration Tests

**File**: `tests/docking/test_guided_sampling.py`

```python
def test_guided_vs_random_sampling():
    """Compare guided vs random sampling hit rate."""
    # Use known complex
    # Generate 1000 poses with random
    # Generate 1000 poses with guided
    # Compute RMSD to native
    # Verify guided has better hit rate
    pass

def test_hybrid_sampling():
    """Test hybrid sampling combines both strategies."""
    pass

def test_sampling_diversity():
    """Verify guided sampling doesn't collapse to single pose."""
    # Generate 1000 guided poses
    # Compute pairwise RMSDs
    # Verify sufficient diversity
    pass
```

---

## Risk Assessment

### Technical Risks

**Risk**: Sub-pocket detection fails on complex pockets
- **Probability**: Medium
- **Impact**: Medium (sampling no better than random)
- **Mitigation**:
  - Test on diverse pocket types
  - Fallback to uniform sampling if detection fails
  - Use conservative clustering parameters

**Risk**: Templates too similar (low diversity)
- **Probability**: Medium
- **Impact**: High (poor coverage of binding modes)
- **Mitigation**:
  - Add randomness to templates
  - Ensure templates span pocket volume
  - Use hybrid mode (guided + random)

**Risk**: Overhead too high
- **Probability**: Low
- **Impact**: Low (still faster than optimization)
- **Mitigation**:
  - Profile and optimize hotspots
  - Cache pocket analysis results
  - Parallelize sub-pocket processing

### Engineering Risks

**Risk**: Code complexity increases significantly
- **Probability**: High
- **Impact**: Medium (harder to maintain)
- **Mitigation**:
  - Clear module boundaries
  - Extensive documentation
  - Unit tests for each component

**Risk**: Many parameters to tune
- **Probability**: High
- **Impact**: Medium (complex configuration)
- **Mitigation**:
  - Provide sensible defaults
  - Auto-tune where possible
  - Document parameter effects

---

## Success Criteria

### Primary Metrics
- ✅ Hit rate (RMSD < 2Å) improved by 5x vs random
- ✅ RMSD improvement ≥ 0.5 Å on test set
- ✅ Overhead < 2s for pocket analysis
- ✅ No regression in scoring quality

### Secondary Metrics
- ✅ Code quality maintained
- ✅ All tests pass
- ✅ Documentation complete
- ✅ Configuration manageable

### Stretch Goals
- 🎯 Hit rate improved by 10x
- 🎯 RMSD improvement ≥ 1.0 Å
- 🎯 Real-time pocket analysis (< 0.5s)

---

## Hand-wavy gaps to fill:

1. **Optimal number of sub-pockets?**
   - Need to experiment with clustering cutoff
   - Test distance_cutoff = [3Å, 4Å, 5Å, 6Å]
   - Validate on diverse pocket types

2. **Optimal template count?**
   - Test n_templates = [5, 10, 20, 50]
   - Balance coverage vs redundancy

3. **Gaussian vs uniform mixture?**
   - Currently using Gaussian
   - Could try von Mises-Fisher for rotations
   - Test different distributions

4. **Feature detection accuracy?**
   - Current rules are very simple
   - Could use more sophisticated methods
   - Trade-off: accuracy vs speed

5. **How to handle failed pocket analysis?**
   - Need robust fallback to random sampling
   - Should detect and report failures
   - Graceful degradation

**Action**: Create experimental framework to test these questions systematically.

---

## Next Steps

1. **Implement pocket analysis** (2 days)
   - Sub-pocket detection
   - Shape analysis
   - Feature detection

2. **Implement template generation** (1 day)
   - Weight computation
   - Template creation

3. **Implement biased sampling** (2 days)
   - Translation sampling
   - Rotation sampling
   - Integration

4. **Validate and optimize** (1 day)
   - Hit rate comparison
   - Performance profiling
   - Parameter tuning

**Total timeline**: 6 days to fully validated guided sampling

---

## References and Prior Art

1. **Fpocket**: Pocket detection using Voronoi tessellation and alpha spheres
2. **SiteMap (Schrödinger)**: Pocket mapping with hydrophobic/hydrophilic mapping
3. **DOCK 6**: Sphere-based pocket sampling
4. **GLIDE**: Grid-based sampling with hierarchical filters

Key insight: Pocket-guided sampling is standard in major docking programs. We're implementing proven technology with modern JAX acceleration.
