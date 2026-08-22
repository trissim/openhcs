#!/usr/bin/env python3
"""
Verify that OpenHCS is ready for PyPI release.

This script checks:
- Version is valid
- Package can be built
- Metadata is correct
- Dependencies are available
"""

import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path


def check_version():
    """Check that version is valid and follows semantic versioning."""
    print("Checking version...")
    init_file = Path("openhcs/__init__.py")
    if not init_file.exists():
        print("  ❌ openhcs/__init__.py not found")
        return False

    content = init_file.read_text()
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
    if not match:
        print("  ❌ __version__ not found in openhcs/__init__.py")
        return False

    version = match.group(1)
    # Basic semantic versioning check
    if not re.match(r"^\d+\.\d+\.\d+", version):
        print(
            f"  ❌ Version '{version}' doesn't follow semantic versioning (MAJOR.MINOR.PATCH)"
        )
        return False

    print(f"  ✅ Version: {version}")
    return True


def check_pyproject_toml():
    """Check declared project and build metadata through TOML authority."""
    print("\nChecking pyproject.toml...")
    pyproject_file = Path("pyproject.toml")
    if not pyproject_file.exists():
        print("  ❌ pyproject.toml not found")
        return False

    try:
        metadata = tomllib.loads(pyproject_file.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        print(f"  ❌ Could not parse pyproject.toml: {exc}")
        return False

    project = metadata.get("project", {})
    build_system = metadata.get("build-system", {})
    required_fields = {
        "name": project.get("name") == "openhcs",
        "version authority": bool(project.get("version"))
        or "version" in project.get("dynamic", ()),
        "description": bool(project.get("description")),
        "authors": bool(project.get("authors")),
        "build-backend": build_system.get("build-backend") == "setuptools.build_meta",
    }

    all_found = all(required_fields.values())
    for field, valid in required_fields.items():
        if not valid:
            print(f"  ❌ Missing or invalid field: {field}")

    if all_found:
        print("  ✅ All required fields present")
    return all_found


def check_readme():
    """Check that README.md exists and is not empty."""
    print("\nChecking README.md...")
    readme_file = Path("README.md")
    if not readme_file.exists():
        print("  ❌ README.md not found")
        return False

    content = readme_file.read_text()
    if len(content.strip()) < 100:
        print("  ⚠️  README.md seems very short")
        return False

    print(f"  ✅ README.md exists ({len(content)} chars)")
    return True


def check_build_dependencies():
    """Check that build dependencies are installed."""
    print("\nChecking build dependencies...")
    required = ["build", "twine", "packaging", "requests"]
    missing = []

    for package in required:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} not installed")
            missing.append(package)

    if missing:
        print(f"\n  Install missing packages: pip install {' '.join(missing)}")
        return False
    return True


def check_git_status():
    """Require the release checkout to be clean and on main."""
    print("\nChecking git status...")
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
        clean = not status.stdout.strip()
        if clean:
            print("  ✅ Working directory clean")
        else:
            print("  ❌ Working directory has staged, unstaged, or untracked changes")

        result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            check=True,
        )
        branch = result.stdout.strip()
        on_main = branch == "main"
        if on_main:
            print("  ✅ On main branch")
        else:
            print(f"  ❌ Current branch is '{branch}', not 'main'")

        return clean and on_main
    except subprocess.CalledProcessError:
        print("  ❌ Not a git repository or git not available")
        return False


def try_build():
    """Build and validate distributions through the active interpreter."""
    print("\nTrying to build package...")
    try:
        with tempfile.TemporaryDirectory(prefix="openhcs-release-") as output_dir:
            subprocess.run(
                [sys.executable, "-m", "build", "--outdir", output_dir],
                capture_output=True,
                text=True,
                check=True,
            )

            dist_files = sorted(Path(output_dir).iterdir())
            if not dist_files:
                print("  ❌ Build succeeded but produced no distributions")
                return False

            print("  ✅ Build successful!")
            print(f"     Created {len(dist_files)} files:")
            for dist_file in dist_files:
                print(f"       - {dist_file.name}")

            wheel_paths = tuple(
                dist_file for dist_file in dist_files if dist_file.suffix == ".whl"
            )
            if not wheel_paths:
                print("  ❌ Build succeeded but produced no wheel")
                return False
            for wheel_path in wheel_paths:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "scripts.validate_wheel_deployment",
                        str(wheel_path),
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
            print("  ✅ Wheel deployment boundaries valid")

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "twine",
                    "check",
                    *(str(dist_file) for dist_file in dist_files),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            print("  ✅ Package metadata valid (twine check passed)")
        return True

    except subprocess.CalledProcessError as exc:
        print("  ❌ Build or metadata validation failed:")
        print(f"     {exc.stderr}")
        return False
    except Exception as exc:
        print(f"  ❌ Error during build: {exc}")
        return False


def check_github_workflow():
    """Check that GitHub Actions workflow exists."""
    print("\nChecking GitHub Actions workflow...")
    workflow_file = Path(".github/workflows/publish.yml")
    if not workflow_file.exists():
        print("  ❌ .github/workflows/publish.yml not found")
        return False

    content = workflow_file.read_text()
    if "PYPI_API_TOKEN" not in content:
        print("  ❌ PYPI_API_TOKEN not referenced in workflow")
        return False

    print("  ✅ GitHub Actions workflow configured")
    print("     Remember to set PYPI_API_TOKEN secret in GitHub!")
    return True


def main():
    """Run all checks."""
    print("=" * 60, flush=True)
    print("OpenHCS PyPI Release Readiness Check", flush=True)
    print("=" * 60, flush=True)

    checks = [
        ("Version", check_version),
        ("pyproject.toml", check_pyproject_toml),
        ("README.md", check_readme),
        ("Build dependencies", check_build_dependencies),
        ("Git status", check_git_status),
        ("GitHub workflow", check_github_workflow),
        ("Package build", try_build),
    ]

    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n❌ Error checking {name}: {e}")
            results[name] = False

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    passed = sum(results.values())
    total = len(results)

    for name, result in results.items():
        status = "✅" if result else "❌"
        print(f"{status} {name}")

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\n🎉 All checks passed! Ready for release!")
        print("\nNext steps:")
        print("  1. Set PYPI_API_TOKEN in GitHub secrets")
        print("  2. Run: python scripts/update_and_release.py")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please fix issues before releasing.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
