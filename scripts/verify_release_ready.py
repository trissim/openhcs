#!/usr/bin/env python3
"""
Verify that OpenHCS is ready for PyPI release.

This script checks:
- Version is valid
- Package can be built
- Metadata is correct
- Dependencies are available
"""

import os
import re
import subprocess
import sys
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
    if not re.match(r'^\d+\.\d+\.\d+', version):
        print(f"  ❌ Version '{version}' doesn't follow semantic versioning (MAJOR.MINOR.PATCH)")
        return False
    
    print(f"  ✅ Version: {version}")
    return True


def check_pyproject_toml():
    """Check that pyproject.toml exists and has required fields."""
    print("\nChecking pyproject.toml...")
    pyproject_file = Path("pyproject.toml")
    if not pyproject_file.exists():
        print("  ❌ pyproject.toml not found")
        return False

    content = pyproject_file.read_text()
    required_fields = {
        'name': r'name\s*=\s*["\']openhcs["\']',
        'version': r'version\s*=',
        'description': r'description\s*=',
        'authors': r'authors\s*=',
        'build-backend': r'build-backend\s*=\s*["\']setuptools\.build_meta["\']',
    }

    all_found = True
    for field, pattern in required_fields.items():
        if not re.search(pattern, content):
            print(f"  ❌ Missing or invalid field: {field}")
            all_found = False

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
    required = ['build', 'twine', 'packaging', 'requests']
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
    """Check git status."""
    print("\nChecking git status...")
    try:
        # Check if we're in a git repo
        subprocess.run(['git', 'status'], capture_output=True, check=True)
        
        # Check for uncommitted changes
        staged = subprocess.run(['git', 'diff', '--staged', '--quiet'], capture_output=True)
        unstaged = subprocess.run(['git', 'diff', '--quiet'], capture_output=True)
        
        if staged.returncode != 0 or unstaged.returncode != 0:
            print("  ⚠️  You have uncommitted changes")
            print("     (This is OK if you plan to commit before release)")
        else:
            print("  ✅ Working directory clean")
        
        # Check current branch
        result = subprocess.run(['git', 'branch', '--show-current'], 
                              capture_output=True, text=True, check=True)
        branch = result.stdout.strip()
        if branch != 'main':
            print(f"  ⚠️  Current branch is '{branch}', not 'main'")
        else:
            print(f"  ✅ On main branch")
        
        return True
    except subprocess.CalledProcessError:
        print("  ❌ Not a git repository or git not available")
        return False


def try_build():
    """Try to build the package."""
    print("\nTrying to build package...")
    try:
        # Clean old builds
        import shutil
        for dir_name in ['dist', 'build', 'openhcs.egg-info']:
            if Path(dir_name).exists():
                shutil.rmtree(dir_name)
                print(f"  🧹 Cleaned {dir_name}/")
        
        # Build
        result = subprocess.run(
            ['python', '-m', 'build'],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Check dist directory
        dist_files = list(Path('dist').glob('*'))
        if not dist_files:
            print("  ❌ Build succeeded but no files in dist/")
            return False
        
        print(f"  ✅ Build successful!")
        print(f"     Created {len(dist_files)} files:")
        for f in dist_files:
            print(f"       - {f.name}")
        
        # Try to check with twine
        try:
            import glob
            import shutil

            # Check if twine is available
            if not shutil.which('twine'):
                print("  ⚠️  twine not found, skipping metadata check")
                print("     Install with: pip install twine")
                print("     (Build succeeded, but metadata not validated)")
                return True  # Don't fail the build check just because twine is missing

            dist_files = glob.glob('dist/*')
            result = subprocess.run(
                ['twine', 'check'] + dist_files,
                capture_output=True,
                text=True,
                check=True
            )
            print("  ✅ Package metadata valid (twine check passed)")
        except subprocess.CalledProcessError as e:
            print("  ❌ Package metadata invalid:")
            print(f"     {e.stderr}")
            return False
        
        return True
        
    except subprocess.CalledProcessError as e:
        print("  ❌ Build failed:")
        print(f"     {e.stderr}")
        return False
    except Exception as e:
        print(f"  ❌ Error during build: {e}")
        return False


def check_github_workflow():
    """Check that GitHub Actions workflow exists."""
    print("\nChecking GitHub Actions workflow...")
    workflow_file = Path(".github/workflows/publish.yml")
    if not workflow_file.exists():
        print("  ❌ .github/workflows/publish.yml not found")
        return False
    
    content = workflow_file.read_text()
    if 'PYPI_API_TOKEN' not in content:
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

