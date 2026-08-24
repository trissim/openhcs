"""Declaration-owned policy for native OpenHCS desktop installations."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from urllib.parse import urlparse

from packaging.requirements import Requirement
from packaging.version import Version


class DesktopPackageExtra(str, Enum):
    """Closed package extras selected by the native desktop surface."""

    BIOFORMATS = "bioformats"
    CELLPROFILER_COMPAT = "cellprofiler-compat"
    GUI = "gui"
    MCP = "mcp"
    VIZ = "viz"


class DesktopBinaryOnlyPackage(str, Enum):
    """Packages that native installs require from a binary distribution."""

    LLVMLITE = "llvmlite"
    NUMBA = "numba"
    OPENCV = "opencv-python"
    OPENCV_HEADLESS = "opencv-python-headless"


class DesktopInstallerSchemaVersion(str, Enum):
    """Supported native installer contract schemas."""

    V2 = "openhcs.installer.v2"


@dataclass(frozen=True, slots=True)
class DesktopUvRelease:
    """Reviewed standalone uv release used by native bootstrap installers."""

    version: str
    base_url: str

    def __post_init__(self) -> None:
        if not re.fullmatch(r"\d+\.\d+\.\d+", self.version):
            raise ValueError("Desktop install uv version must be stable SemVer.")
        parsed_base_url = urlparse(self.base_url)
        if (
            parsed_base_url.scheme != "https"
            or parsed_base_url.hostname != "astral.sh"
            or parsed_base_url.path != "/uv"
            or parsed_base_url.params
            or parsed_base_url.query
            or parsed_base_url.fragment
        ):
            raise ValueError(
                "Desktop install uv base URL must be the official "
                "https://astral.sh/uv endpoint."
            )


@dataclass(frozen=True, slots=True)
class DesktopInstallProfile:
    """Installer-specific policy independent of package and brand declarations."""

    python_version: str
    package_extras: tuple[DesktopPackageExtra, ...]
    binary_only_packages: tuple[DesktopBinaryOnlyPackage, ...]
    uv_release: DesktopUvRelease

    def __post_init__(self) -> None:
        if not re.fullmatch(r"3\.\d+", self.python_version):
            raise ValueError(
                "Desktop install profile Python version must be a Python 3 minor."
            )
        self._validate_members(
            self.package_extras,
            member_type=DesktopPackageExtra,
            description="package extras",
        )
        self._validate_members(
            self.binary_only_packages,
            member_type=DesktopBinaryOnlyPackage,
            description="binary-only packages",
        )

    @staticmethod
    def _validate_members(
        values: tuple[Enum, ...],
        *,
        member_type: type[Enum],
        description: str,
    ) -> None:
        if (
            not values
            or not all(type(value) is member_type for value in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError(
                f"Desktop install profile {description} must be unique declared "
                "members."
            )

    def select(
        self,
        package_name: str,
        version: Version | str,
    ) -> DesktopPackageInstallSelection:
        """Select one package release without copying this profile's policy."""

        requirement = Requirement(package_name)
        if (
            requirement.url is not None
            or requirement.marker is not None
            or requirement.specifier
            or requirement.extras
        ):
            raise ValueError("Desktop install package name must be unversioned.")
        return DesktopPackageInstallSelection(
            profile=self,
            package_name=requirement.name,
            version=Version(str(version)),
        )

    def project_contract(
        self,
        *,
        product_name: str,
        package_name: str,
        version: Version | str,
        entry_point: str,
        gui_entry_point: str,
    ) -> DesktopInstallerContract:
        """Project all installer-owned policy into one native contract."""

        selection = self.select(package_name, version)
        return DesktopInstallerContract(
            product_name=product_name,
            python_version=self.python_version,
            package_requirement=selection.package_requirement,
            binary_only_packages=selection.binary_only_argument,
            entry_point=entry_point,
            gui_entry_point=gui_entry_point,
            uv_release=self.uv_release,
        )


@dataclass(frozen=True, slots=True)
class DesktopPackageInstallSelection:
    """One release identity that retains its owning desktop profile."""

    profile: DesktopInstallProfile
    package_name: str
    version: Version

    @property
    def package_requirement(self) -> str:
        """Return the exact requirement derived from profile extras."""

        extras = ",".join(sorted(extra.value for extra in self.profile.package_extras))
        return f"{self.package_name}[{extras}]=={self.version}"

    @property
    def binary_only_argument(self) -> str:
        """Return pip's comma-separated binary-only projection."""

        return ",".join(package.value for package in self.profile.binary_only_packages)


@dataclass(frozen=True, slots=True)
class DesktopInstallerContract:
    """Nominal cross-language document embedded in one native installer."""

    schema_version: DesktopInstallerSchemaVersion = field(
        init=False,
        default=DesktopInstallerSchemaVersion.V2,
    )
    product_name: str
    python_version: str
    package_requirement: str
    binary_only_packages: str
    entry_point: str
    gui_entry_point: str
    uv_release: DesktopUvRelease

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 ._-]*", self.product_name):
            raise ValueError("Installer contract product name has an invalid format.")
        for value, description in (
            (self.entry_point, "command entry point"),
            (self.gui_entry_point, "GUI entry point"),
        ):
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
                raise ValueError(
                    f"Installer contract {description} has an invalid format."
                )

    def write(self, path: Path) -> None:
        """Serialize this validated projection for native consumers."""

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )


DESKTOP_INSTALL_PROFILE = DesktopInstallProfile(
    python_version="3.12",
    package_extras=(
        DesktopPackageExtra.BIOFORMATS,
        DesktopPackageExtra.CELLPROFILER_COMPAT,
        DesktopPackageExtra.GUI,
        DesktopPackageExtra.MCP,
        DesktopPackageExtra.VIZ,
    ),
    binary_only_packages=(
        DesktopBinaryOnlyPackage.LLVMLITE,
        DesktopBinaryOnlyPackage.NUMBA,
        DesktopBinaryOnlyPackage.OPENCV,
        DesktopBinaryOnlyPackage.OPENCV_HEADLESS,
    ),
    uv_release=DesktopUvRelease(
        version="0.11.28",
        base_url="https://astral.sh/uv",
    ),
)
