"""Declaration-configured standalone experimental analysis."""

from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from openhcs.core.config import ExperimentalAnalysisConfig
from openhcs.formats.experimental_layout_rows import ExperimentalAnalysisScope
from openhcs.formats.experimental_result_formats import (
    ExperimentalResultFormatStrategy,
)


class DataProcessingError(RuntimeError):
    """Raised when standalone experimental analysis cannot complete."""


class ExperimentalAnalysisEngine:
    """Coordinate analysis through the result scope's nominal strategy."""

    def __init__(self, config: ExperimentalAnalysisConfig):
        """
        Initialize analysis engine with configuration.

        Args:
            config: Experimental analysis configuration
        """
        self.config = config

    def run_analysis(
        self,
        results_path: str,
        config_file: str,
        compiled_results_path: str,
        heatmap_path: Optional[str] = None,
        raw_results_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run complete experimental analysis for the workbook-declared format.

        Args:
            results_path: Path to microscope results file
            config_file: Path to experimental configuration Excel file
            compiled_results_path: Output path for compiled results
            heatmap_path: Optional output path for heatmap visualization

        Returns:
            Dictionary containing analysis results and metadata

        Raises:
            DataProcessingError: If data processing fails
        """
        try:
            # Parse the workbook declaration that owns result format.
            experiment_config = self._parse_experiment_config(config_file)
            scope = ExperimentalAnalysisScope.coerce(experiment_config["scope"])

            # Process microscope data through the scope-keyed leaf.
            processed_data = ExperimentalResultFormatStrategy.for_enum_member(
                scope
            ).process(results_path)
            format_name = scope.value

            # Create the experiment data structure.
            experiment_dict_locations = self._make_experiment_dict_locations(
                experiment_config["plate_groups"],
                experiment_config["plate_layout"],
                experiment_config["conditions"],
            )

            # Apply the workbook's replicate-scoped exclusions before values,
            # controls, or visual projections consume those locations.
            self._apply_exclusions(
                experiment_dict_locations,
                experiment_config["ctrl_positions"],
                experiment_config["excluded_positions"],
            )

            # Map experimental design to measured values.
            experiment_dict_values = self._make_experiment_dict_values(
                processed_data["plates_dict"],
                experiment_dict_locations,
                processed_data["features"],
                experiment_config["plate_groups"],
                experiment_config["per_well_datapoints"],
            )

            # Apply normalization if controls are defined.
            if experiment_config["ctrl_positions"] is not None:
                experiment_dict_values_normalized = self._normalize_experiment(
                    experiment_dict_values,
                    experiment_config["ctrl_positions"],
                    processed_data["features"],
                    processed_data["plates_dict"],
                    experiment_config["plate_groups"],
                )
            else:
                experiment_dict_values_normalized = experiment_dict_values

            # Generate results tables.
            feature_tables = self._create_all_feature_tables(
                experiment_dict_values_normalized,
                processed_data["features"],
                experiment_config["per_well_datapoints"],
            )

            # Export normalized results.
            self._export_results(feature_tables, compiled_results_path)

            # Export raw results if configured.
            if self.config.export_raw_results:
                if raw_results_path is None:
                    compiled_path = Path(compiled_results_path)
                    raw_results_path = str(
                        compiled_path.with_name(
                            f"{compiled_path.stem}_raw{compiled_path.suffix}"
                        )
                    )
                feature_tables_raw = self._create_all_feature_tables(
                    experiment_dict_values,
                    processed_data["features"],
                    experiment_config["per_well_datapoints"],
                )
                self._export_results(feature_tables_raw, raw_results_path)

            # Generate heatmaps if configured.
            if self.config.export_heatmaps and heatmap_path:
                self._export_heatmaps(
                    processed_data["plates_dict"],
                    processed_data["features"],
                    experiment_config["excluded_positions"],
                    experiment_config["plate_groups"],
                    heatmap_path,
                )

            return {
                "format_name": format_name,
                "features": processed_data["features"],
                "conditions": experiment_config["conditions"],
                "feature_tables": feature_tables,
                "experiment_values": experiment_dict_values_normalized,
                "experiment_config": experiment_config,
                "processed_data": processed_data,
            }

        except Exception as e:
            raise DataProcessingError(f"Analysis failed: {e}") from e

    def run_directory(self, analysis_directory: str | Path) -> Dict[str, Any]:
        """Run the declaration-configured directory workflow."""
        analysis_path = Path(analysis_directory)
        return self.run_analysis(
            results_path=str(analysis_path / self.config.results_file_name),
            config_file=str(analysis_path / self.config.config_file_name),
            compiled_results_path=str(
                analysis_path / self.config.compiled_results_file_name
            ),
            heatmap_path=(
                str(analysis_path / self.config.heatmap_file_name)
                if self.config.export_heatmaps
                else None
            ),
            raw_results_path=(
                str(analysis_path / self.config.raw_results_file_name)
                if self.config.export_raw_results
                else None
            ),
        )

    def _parse_experiment_config(self, config_file: str) -> Dict[str, Any]:
        """
        Parse experimental configuration from Excel file.

        Args:
            config_file: Path to configuration file

        Returns:
            Parsed configuration dictionary

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config parsing fails
        """
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

        try:
            # Parse experimental design
            (
                scope,
                plate_layout,
                conditions,
                ctrl_positions,
                excluded_positions,
                per_well_datapoints,
            ) = self._read_plate_layout(config_file)

            # Parse plate groups
            plate_groups = self._load_plate_groups(config_file)

            return {
                "scope": scope,
                "plate_layout": plate_layout,
                "conditions": conditions,
                "ctrl_positions": ctrl_positions,
                "excluded_positions": excluded_positions,
                "per_well_datapoints": per_well_datapoints,
                "plate_groups": plate_groups,
            }

        except Exception as e:
            raise ValueError(f"Failed to parse configuration file {config_file}: {e}")

    def _read_plate_layout(
        self, config_path: str
    ) -> Tuple[str, Dict, List, Optional[Dict], Optional[Dict], bool]:
        """
        Read plate layout from configuration file.

        Args:
            config_path: Path to configuration file

        Returns:
            Tuple of (scope, plate_layout, conditions, ctrl_positions, excluded_positions, per_well_datapoints)
        """
        from openhcs.formats.experimental_analysis import read_plate_layout

        return read_plate_layout(
            config_path,
            sheet_name=self.config.design_sheet_name,
        )

    def _load_plate_groups(self, config_path: str) -> Dict:
        """
        Load plate groups from configuration file.

        Args:
            config_path: Path to configuration file

        Returns:
            Plate groups dictionary
        """
        from openhcs.formats.experimental_analysis import load_plate_groups

        return load_plate_groups(
            config_path,
            sheet_name=self.config.plate_groups_sheet_name,
        )

    def _make_experiment_dict_locations(
        self, plate_groups: Dict, plate_layout: Dict, conditions: List
    ) -> Dict:
        """Create experiment location mapping."""
        from openhcs.formats.experimental_analysis import make_experiment_dict_locations

        return make_experiment_dict_locations(plate_groups, plate_layout, conditions)

    def _make_experiment_dict_values(
        self,
        plates_dict: Dict,
        experiment_dict_locations: Dict,
        features: List,
        plate_groups: Dict,
        per_well_datapoints: bool = False,
    ) -> Dict:
        """Map experimental design to measured values."""
        from openhcs.formats.experimental_analysis import make_experiment_dict_values

        return make_experiment_dict_values(
            plates_dict,
            experiment_dict_locations,
            features,
            plate_groups,
            per_well_datapoints,
        )

    def _apply_exclusions(
        self,
        experiment_dict_locations: Dict,
        ctrl_positions: Optional[Dict],
        excluded_positions: Optional[Dict],
    ) -> None:
        """Apply the parser-owned exclusion projection to analysis locations."""
        from openhcs.formats.experimental_analysis import (
            apply_excluded_positions_to_control_positions,
            apply_excluded_positions_to_experiment_locations,
        )

        apply_excluded_positions_to_experiment_locations(
            experiment_dict_locations,
            excluded_positions,
        )
        apply_excluded_positions_to_control_positions(
            ctrl_positions,
            excluded_positions,
        )

    def _normalize_experiment(
        self,
        experiment_dict_values: Dict,
        ctrl_positions: Dict,
        features: List,
        plates_dict: Dict,
        plate_groups: Dict,
    ) -> Dict:
        """Apply normalization using control wells."""
        from openhcs.formats.experimental_analysis import normalize_experiment

        return normalize_experiment(
            experiment_dict_values,
            ctrl_positions,
            features,
            plates_dict,
            plate_groups,
            method=self.config.normalization_method,
        )

    def _create_all_feature_tables(
        self,
        experiment_dict_values: Dict,
        features: List,
        per_well_datapoints: bool = False,
    ) -> Dict:
        """Create feature tables for export."""
        from openhcs.formats.experimental_analysis import create_all_feature_tables

        return create_all_feature_tables(
            experiment_dict_values, features, per_well_datapoints
        )

    def _export_results(self, feature_tables: Dict, output_path: str):
        """Export results to Excel file."""
        from openhcs.formats.experimental_analysis import feature_tables_to_excel

        feature_tables_to_excel(feature_tables, output_path)

    def _export_heatmaps(
        self,
        plates_dict: Dict,
        features: List,
        excluded_positions: Optional[Dict],
        plate_groups: Dict,
        output_path: str,
    ) -> None:
        """Export conditionally formatted plate grids after exclusions."""
        from openhcs.formats.experimental_analysis import (
            project_plates_without_excluded_positions,
            write_values_heat_map,
        )

        projected_plates = project_plates_without_excluded_positions(
            plates_dict,
            excluded_positions,
            plate_groups,
        )
        write_values_heat_map(projected_plates, features, output_path)
