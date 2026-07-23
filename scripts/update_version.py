#!/usr/bin/env python3
import argparse
import re
import subprocess
import sys
from pathlib import Path

from packaging import version


def get_current_version():
    """Get the current version from __init__.py"""
    init_file = Path("openhcs/__init__.py")
    content = init_file.read_text()
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
    return match.group(1) if match else None

def validate_version(new_version):
    """Validate the new version string"""
    try:
        v = version.parse(new_version)
        if not isinstance(v, version.Version):
            raise ValueError("Invalid version format")
        return True
    except version.InvalidVersion:
        raise ValueError("Invalid version format. Use semantic versioning (e.g., 1.2.3)")

def check_for_uncommitted_changes():
    """Check if there are any uncommitted changes in tracked files"""
    # Check for staged and unstaged changes in tracked files
    staged = subprocess.run(['git', 'diff', '--staged', '--quiet'])
    unstaged = subprocess.run(['git', 'diff', '--quiet'])
    return staged.returncode != 0 or unstaged.returncode != 0

def update_version(new_version=None):
    """Update version in files and create git commit"""
    try:
        # Get current version first
        current_version = get_current_version()
        if not current_version:
            print("Error: Could not find current version in __init__.py")
            sys.exit(1)

        if new_version is None:
            # Auto-increment patch version if no version specified
            v = version.parse(current_version)
            new_version = f"{v.major}.{v.minor}.{v.micro + 1}"
        
        # Validate new version
        validate_version(new_version)
        if version.parse(new_version) <= version.parse(current_version):
            print(f"Error: New version ({new_version}) must be greater than current version ({current_version})")
            sys.exit(1)

        # Check for uncommitted changes
        if check_for_uncommitted_changes():
            print("Error: You have uncommitted changes. Please commit or stash them first.")
            sys.exit(1)

        # Pull latest changes
        try:
            subprocess.run(['git', 'pull', 'origin', 'main'], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: Failed to pull latest changes: {e}")
            sys.exit(1)
        
        # Update version in __init__.py
        init_file = Path("openhcs/__init__.py")
        content = init_file.read_text()
        new_content = re.sub(
            r'(__version__\s*=\s*[\'"])[^\'"]+([\'"])',
            rf'\g<1>{new_version}\2',
            content
        )
        init_file.write_text(new_content)

        # Commit and push changes
        subprocess.run(['git', 'add', 'openhcs/__init__.py'], check=True)
        subprocess.run(['git', 'commit', '-m', f'bump version to {new_version}'], check=True)
        subprocess.run(['git', 'push', 'origin', 'main'], check=True)
        
        print(f"Successfully updated version to {new_version}")
        print("Now run: python scripts/release.py")
        
    except subprocess.CalledProcessError as e:
        print(f"Error during version update: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Update OpenHCS version number')
    parser.add_argument('--version', '-v',
                      help='New version number (e.g., 1.2.3). If not provided, will increment patch version.')
    args = parser.parse_args()

    update_version(args.version)

if __name__ == "__main__":
    main()
