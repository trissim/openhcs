#!/usr/bin/env python3
import re
import subprocess
import sys
from pathlib import Path

from packaging import version


def run_command(command, check=True):
    """Run a command and return its output"""
    try:
        result = subprocess.run(command, check=check, capture_output=True, text=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command {' '.join(command)}: {e}")
        print(f"Error output: {e.stderr}")
        sys.exit(1)

def get_modified_files():
    """Get list of modified files"""
    # Get both staged and unstaged changes
    staged = run_command(['git', 'diff', '--staged', '--name-only']).split('\n')
    unstaged = run_command(['git', 'diff', '--name-only']).split('\n')
    # Combine and remove duplicates while preserving order
    all_files = []
    for f in staged + unstaged:
        if f and f not in all_files:
            all_files.append(f)
    return all_files

def check_unstaged_changes():
    """Check for unstaged changes"""
    result = subprocess.run(['git', 'diff', '--quiet'], capture_output=True)
    return result.returncode != 0

def commit_changes(files, message):
    """Commit specified files with a message"""
    try:
        # Add files explicitly first
        for file in files:
            print(f"Adding file: {file}")
            subprocess.run(['git', 'add', file], check=True)
        
        # Show what's being committed
        print("\nFiles staged for commit:")
        subprocess.run(['git', 'status', '--short'], check=True)
        
        # Commit
        print(f"\nCommitting with message: {message}")
        subprocess.run(['git', 'commit', '-m', message], check=True)
        
        # Now pull with rebase (after changes are committed)
        print("\nPulling latest changes...")
        subprocess.run(['git', 'pull', '--rebase', 'origin', 'main'], check=True)
        
        # Push
        print("\nPushing to origin main...")
        subprocess.run(['git', 'push', 'origin', 'main'], check=True)
        
        print("Changes committed and pushed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"Git operation failed: {e}")
        print(f"Error output: {e.stderr if e.stderr else 'None'}")
        sys.exit(1)

def update_version():
    """Run the version update script"""
    try:
        subprocess.run([sys.executable, 'scripts/update_version.py'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error updating version: {e}")
        sys.exit(1)

def create_release():
    """Run the release script"""
    try:
        subprocess.run([sys.executable, 'scripts/release.py'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error creating release: {e}")
        sys.exit(1)

def main():
    # 1. Get modified files
    modified_files = [f for f in get_modified_files() if f]
    
    # Handle existing changes if any
    if modified_files:
        print("\nModified files:")
        for f in modified_files:
            print(f"- {f}")
        
        response = input("\nCommit these changes before release? [y/N] ")
        if response.lower() == 'y':
            commit_message = input("\nEnter commit message: ")
            if not commit_message:
                print("Commit message cannot be empty.")
                sys.exit(1)
            
            print("\nCommitting changes...")
            commit_changes(modified_files, commit_message)
    
    # Always proceed with version update and release
    print("\nUpdating version...")
    update_version()
    
    print("\nCreating release...")
    create_release()
    
    print("\nProcess completed successfully!")

if __name__ == "__main__":
    main()
