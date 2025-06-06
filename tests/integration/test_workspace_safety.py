"""
Test file to verify workspace mirroring safety guarantees.

This test verifies that the workspace system:
1. Only works on symlinks - Will refuse to delete real files
2. Workspace boundary checks - Will not delete files outside workspace  
3. Symlink resolution - Copies real files, not broken symlinks
4. Error on real files in workspace - Fails loudly if workspace contains real files instead of symlinks
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch
import os

from openhcs.microscopes.opera_phenix import OperaPhenixHandler
from openhcs.io.filemanager import FileManager
from openhcs.io.base import storage_registry
from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator


class TestWorkspaceSafety:
    """Test workspace safety guarantees."""

    def setup_method(self):
        """Set up test environment with temporary directories."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.original_dir = self.temp_dir / "original_data"
        self.workspace_dir = self.temp_dir / "test_workspace"
        
        # Create original data structure
        self.original_dir.mkdir(parents=True)
        self.images_dir = self.original_dir / "Images"
        self.images_dir.mkdir()
        
        # Create test files
        self.test_files = [
            "r01c01f001p001-ch1sk1fk1fl1.tiff",
            "r01c01f001p001-ch2sk1fk1fl1.tiff",
            "Index.xml"
        ]
        
        for filename in self.test_files:
            test_file = self.images_dir / filename
            test_file.write_text(f"test content for {filename}")
        
        # Create workspace structure
        self.workspace_dir.mkdir(parents=True)
        self.workspace_images_dir = self.workspace_dir / "Images"
        self.workspace_images_dir.mkdir()

    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_safety_claim_1_only_deletes_symlinks(self):
        """
        SAFETY CLAIM 1: Only works on symlinks - Will refuse to delete real files
        """
        # Create symlinks in workspace
        for filename in self.test_files[:2]:  # Skip Index.xml for this test
            original_file = self.images_dir / filename
            workspace_file = self.workspace_images_dir / filename
            workspace_file.symlink_to(original_file)
        
        # Create Index.xml in workspace (copy, not symlink)
        workspace_index = self.workspace_images_dir / "Index.xml"
        workspace_index.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<OperaLX xmlns="http://www.perkinelmer.com/PEHH/HarmonyV6">
    <Field id="1" x="0" y="0"/>
</OperaLX>""")
        
        # Create handler and filemanager
        filemanager = FileManager(storage_registry)
        handler = OperaPhenixHandler(filemanager)
        
        # This should work - deletes symlinks only
        try:
            handler._prepare_workspace(self.workspace_dir, filemanager)
            # Should succeed without error
        except Exception as e:
            pytest.fail(f"Should not fail when deleting symlinks: {e}")
        
        # Verify original files are untouched
        for filename in self.test_files:
            original_file = self.images_dir / filename
            assert original_file.exists(), f"Original file {filename} should still exist"
            assert original_file.read_text() == f"test content for {filename}"

    def test_safety_claim_2_workspace_boundary_checks(self):
        """
        SAFETY CLAIM 2: Workspace boundary checks - Will not delete files outside workspace
        """
        # Create a file outside workspace
        outside_file = self.temp_dir / "outside_workspace.tiff"
        outside_file.write_text("outside content")
        
        # Create handler
        filemanager = FileManager(storage_registry)
        handler = OperaPhenixHandler(filemanager)
        
        # Mock list_image_files to return file outside workspace
        with patch.object(filemanager, 'list_image_files') as mock_list:
            mock_list.return_value = [str(outside_file)]
            
            # This should raise an error
            with pytest.raises(RuntimeError, match="Workspace preparation tried to delete file outside workspace"):
                handler._prepare_workspace(self.workspace_dir, filemanager)
        
        # Verify outside file is untouched
        assert outside_file.exists(), "File outside workspace should not be deleted"
        assert outside_file.read_text() == "outside content"

    def test_safety_claim_3_symlink_resolution(self):
        """
        SAFETY CLAIM 3: Symlink resolution - Copies real files, not broken symlinks
        """
        # Create symlinks in workspace pointing to real files
        for filename in self.test_files[:2]:  # Skip Index.xml
            original_file = self.images_dir / filename
            workspace_file = self.workspace_images_dir / filename
            workspace_file.symlink_to(original_file)
        
        # Create Index.xml with field mapping
        workspace_index = self.workspace_images_dir / "Index.xml"
        workspace_index.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<OperaLX xmlns="http://www.perkinelmer.com/PEHH/HarmonyV6">
    <Field id="1" x="0" y="0"/>
</OperaLX>""")
        
        # Create handler and run workspace preparation
        filemanager = FileManager(storage_registry)
        handler = OperaPhenixHandler(filemanager)
        
        # This should resolve symlinks and copy real files
        handler._prepare_workspace(self.workspace_dir, filemanager)
        
        # Verify that new files were created (not symlinks)
        renamed_files = list(self.workspace_images_dir.glob("r01c01f001p001-ch*sk1fk1fl1.tiff"))
        assert len(renamed_files) >= 2, "Should have created renamed files"
        
        for renamed_file in renamed_files:
            assert not renamed_file.is_symlink(), f"Renamed file {renamed_file} should not be a symlink"
            assert renamed_file.is_file(), f"Renamed file {renamed_file} should be a real file"

    def test_safety_claim_4_error_on_real_files_in_workspace(self):
        """
        SAFETY CLAIM 4: Error on real files in workspace - Fails loudly if workspace contains real files instead of symlinks
        """
        # Create real files in workspace (not symlinks)
        for filename in self.test_files[:2]:  # Skip Index.xml
            workspace_file = self.workspace_images_dir / filename
            workspace_file.write_text(f"real file content for {filename}")
        
        # Create Index.xml
        workspace_index = self.workspace_images_dir / "Index.xml"
        workspace_index.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<OperaLX xmlns="http://www.perkinelmer.com/PEHH/HarmonyV6">
    <Field id="1" x="0" y="0"/>
</OperaLX>""")
        
        # Create handler
        filemanager = FileManager(storage_registry)
        handler = OperaPhenixHandler(filemanager)
        
        # This should raise an error when it finds real files
        with pytest.raises(RuntimeError, match="Workspace contains real file instead of symlink"):
            handler._prepare_workspace(self.workspace_dir, filemanager)
        
        # Verify real files are untouched (not deleted)
        for filename in self.test_files[:2]:
            workspace_file = self.workspace_images_dir / filename
            assert workspace_file.exists(), f"Real file {filename} should not be deleted"
            assert workspace_file.read_text() == f"real file content for {filename}"

    def test_broken_symlink_handling(self):
        """
        Test that broken symlinks are handled gracefully.
        """
        # Create broken symlinks in workspace
        for filename in self.test_files[:2]:
            workspace_file = self.workspace_images_dir / filename
            # Create symlink to non-existent file
            workspace_file.symlink_to("/non/existent/path")
        
        # Create Index.xml
        workspace_index = self.workspace_images_dir / "Index.xml"
        workspace_index.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<OperaLX xmlns="http://www.perkinelmer.com/PEHH/HarmonyV6">
    <Field id="1" x="0" y="0"/>
</OperaLX>""")
        
        # Create handler
        filemanager = FileManager(storage_registry)
        handler = OperaPhenixHandler(filemanager)
        
        # This should handle broken symlinks gracefully (skip them)
        try:
            handler._prepare_workspace(self.workspace_dir, filemanager)
            # Should not crash, just skip broken symlinks
        except Exception as e:
            # Should not raise an exception for broken symlinks
            pytest.fail(f"Should handle broken symlinks gracefully: {e}")

    def test_comprehensive_safety_integration(self):
        """
        Integration test that verifies all safety features work together.
        """
        # Create symlinks in workspace
        for filename in self.test_files[:2]:  # Skip Index.xml
            original_file = self.images_dir / filename
            workspace_file = self.workspace_images_dir / filename
            workspace_file.symlink_to(original_file)

        # Create Index.xml with field mapping
        workspace_index = self.workspace_images_dir / "Index.xml"
        workspace_index.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<OperaLX xmlns="http://www.perkinelmer.com/PEHH/HarmonyV6">
    <Field id="1" x="0" y="0"/>
    <Field id="2" x="100" y="0"/>
</OperaLX>""")

        # Create handler
        filemanager = FileManager(storage_registry)
        handler = OperaPhenixHandler(filemanager)

        # This should work completely safely
        result = handler._prepare_workspace(self.workspace_dir, filemanager)

        # Verify all original files are untouched
        for filename in self.test_files:
            original_file = self.images_dir / filename
            assert original_file.exists(), f"Original file {filename} should still exist"
            assert original_file.read_text() == f"test content for {filename}"

        # Verify workspace contains processed files (not symlinks)
        processed_files = list(self.workspace_images_dir.glob("*.tiff"))
        assert len(processed_files) >= 2, "Should have processed files in workspace"

        for processed_file in processed_files:
            assert not processed_file.is_symlink(), f"Processed file {processed_file} should not be a symlink"
            assert processed_file.is_file(), f"Processed file {processed_file} should be a real file"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
