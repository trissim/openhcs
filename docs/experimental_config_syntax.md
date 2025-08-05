# Experimental Configuration Syntax (config.xlsx)

This document describes the syntax for defining complex experimental designs in Excel format for use with OpenHCS experimental analysis.

## File Structure

The configuration uses an Excel file with multiple sheets:
- `drug_curve_map`: Main experimental design definition
- `plate_groups`: Mapping of replicates to physical plates
- Additional sheets as needed

## Sheet 1: drug_curve_map

### Global Parameters (Required)

These parameters must appear at the top of the sheet:

```
N                    3                    # Number of biological replicates
Scope               EDDU_metaxpress       # Microscope format
```

**Supported Scopes:**
- `EDDU_CX5`: ThermoFisher CX5 format
- `EDDU_metaxpress`: MetaXpress format

### Control Definition Block (Optional)

Define control wells for normalization:

```
Controls            A01  B01  E01  F01  A05  B05  E05  F05  A09  B09  E09  F09
Plate Group         1    1    1    1    1    1    1    1    1    1    1    1
Group N             1    1    1    1    2    2    2    2    3    3    3    3
```

- **Controls**: Well positions for control conditions
- **Plate Group**: Physical plate identifier for each control well
- **Group N**: Biological replicate assignment (1=N1, 2=N2, 3=N3, etc.)

### Experimental Condition Blocks (Required)

Each experimental condition follows this pattern:

```
Condition           [Condition Name]      # Name of the experimental condition
Dose                [dose1] [dose2] ...   # Dose series (concentrations, timepoints, etc.)
Wells1              [well1] [well2] ...   # Wells for biological replicate 1
Plate Group         [plate] [plate] ...   # Plate assignment for Wells1
Wells1              [well1] [well2] ...   # Additional rows = technical replicates
Plate Group         [plate] [plate] ...   # Plate assignment for additional Wells1
Wells2              [well1] [well2] ...   # Wells for biological replicate 2
Plate Group         [plate] [plate] ...   # Plate assignment for Wells2
Wells3              [well1] [well2] ...   # Wells for biological replicate 3
Plate Group         [plate] [plate] ...   # Plate assignment for Wells3
```

**Key Rules:**
1. **WellsN** (N=1,2,3...): Each number corresponds to a biological replicate
2. **Wells** (no number): Same wells applied to ALL biological replicates
3. **Multiple rows per WellsN**: Creates technical replicates (averaged together)
4. **Dose-to-well mapping**: First dose maps to first well, second dose to second well, etc.
5. **Plate Group**: Must follow each Wells row, maps wells to physical plates
6. **Empty rows**: Used to separate different conditions

### Example Complete Block

```
Condition           Drug_A + Inhibitor_B
Dose                0    10   50   100
Wells1              A01  A02  A03  A04    # N1: Control, 10μM, 50μM, 100μM
Plate Group         1    1    1    1
Wells1              B01  B02  B03  B04    # N1: Technical replicates
Plate Group         1    1    1    1
Wells2              A05  A06  A07  A08    # N2: Same doses
Plate Group         1    1    1    1
Wells2              B05  B06  B07  B08    # N2: Technical replicates
Plate Group         1    1    1    1
Wells3              A09  A10  A11  A12    # N3: Same doses
Plate Group         1    1    1    1
Wells3              B09  B10  B11  B12    # N3: Technical replicates
Plate Group         1    1    1    1
```

## Sheet 2: plate_groups

Maps biological replicates to physical plate identifiers:

```
     0         1
0  NaN         1
1   N1  20220818
2   N2  20220818  
3   N3  20220818
```

- Column 0: Replicate names (N1, N2, N3, etc.)
- Column 1: Physical plate identifier/barcode

## Data Processing Flow

1. **Parse global parameters** (N, Scope)
2. **Extract control definitions** for normalization
3. **Process each condition block**:
   - Map doses to wells for each biological replicate
   - Group technical replicates (multiple rows per WellsN)
   - Assign plate groups
4. **Load plate group mappings**
5. **Create data structure**: `experiment_dict[condition][replicate][dose] = [(well, plate_group), ...]`

## Advanced Features

### Multi-Plate Experiments
```
Wells1              A01  A02  A03  A04
Plate Group         1    1    2    2      # Wells A01,A02 on plate 1; A03,A04 on plate 2
```

### Same Wells Across All Replicates
```
Wells               A01  A02  A03  A04    # Applied to ALL biological replicates (N1, N2, N3...)
Plate Group         1    1    1    1      # Plate mapping for all replicates
```

### Complex Replication
```
Wells1              A01  A02  A03  A04    # First technical replicate
Plate Group         1    1    1    1
Wells1              B01  B02  B03  B04    # Second technical replicate
Plate Group         1    1    1    1
Wells1              C01  C02  C03  C04    # Third technical replicate
Plate Group         1    1    1    1
```

### Variable Replicate Numbers
```
N                   4                     # Can have any number of replicates
...
Wells1              ...                   # N1
Wells2              ...                   # N2  
Wells3              ...                   # N3
Wells4              ...                   # N4
```

## Error Handling

- **Missing Plate Group**: Each Wells row must be followed by Plate Group
- **Dose-Well Mismatch**: Number of doses must match number of wells
- **Invalid Scope**: Only EDDU_CX5 and EDDU_metaxpress supported
- **Missing N**: Number of replicates must be specified

## Best Practices

1. **Consistent naming**: Use clear, descriptive condition names
2. **Logical well layout**: Group related conditions in adjacent plate regions
3. **Control placement**: Distribute controls across the plate to account for edge effects
4. **Documentation**: Include description in first row for complex experiments
5. **Validation**: Check that all WellsN (1 to N) are defined for each condition
