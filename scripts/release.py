#!/usr/bin/env python3
import re
import subprocess
import sys
from pathlib import Path

import requests
from packaging import version


def get_current_version():
    with open("ezstitcher/__init__.py", "r") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"\'')
    return None

def get_pypi_version():
    try:
        response = requests.get("https://pypi.org/pypi/ezstitcher/json")
        if response.status_code == 200:
            return response.json()["info"]["version"]
    except:
        pass
    return None

def main():
    # Get current version
    current_version = get_current_version()
    if not current_version:
        print("Error: Could not find version in __init__.py")
        sys.exit(1)
    
    # Get PyPI version
    pypi_version = get_pypi_version()
    print(f"Current package version: {current_version}")
    print(f"Current PyPI version: {pypi_version}")
    
    if pypi_version and version.parse(current_version) <= version.parse(pypi_version):
        print(f"Error: Current version ({current_version}) must be greater than PyPI version ({pypi_version})")
        sys.exit(1)
    
    # Confirm with user
    response = input(f"Create release for v{current_version}? [y/N] ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    try:
        # Create and push tag
        subprocess.run(['git', 'tag', '-a', f'v{current_version}', '-m', f'Release version {current_version}'], check=True)
        subprocess.run(['git', 'push', 'origin', f'v{current_version}'], check=True)
        
        print(f"\nSuccessfully created and pushed tag v{current_version}")
        print("GitHub Actions workflow should start automatically.")
        print("Monitor progress at: https://github.com/YOUR_USERNAME/ezstitcher/actions")
    
    except subprocess.CalledProcessError as e:
        print(f"Error during release process: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
