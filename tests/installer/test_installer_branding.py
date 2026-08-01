"""Installer branding contracts backed by the official OpenHCS asset family."""

from __future__ import annotations

from pathlib import Path
import struct


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = REPOSITORY_ROOT / "openhcs" / "resources" / "assets"
RENDER_SCRIPT = REPOSITORY_ROOT / "scripts" / "render_brand_assets.sh"
WINDOWS_ROOT = REPOSITORY_ROOT / "packaging" / "installers" / "windows"
MACOS_ROOT = REPOSITORY_ROOT / "packaging" / "installers" / "macos"
PUBLISH_WORKFLOW = REPOSITORY_ROOT / ".github" / "workflows" / "publish.yml"
INTEGRATION_WORKFLOW = (
    REPOSITORY_ROOT / ".github" / "workflows" / "integration-tests.yml"
)


def test_native_installer_icons_are_derivatives_of_the_official_square_icon() -> None:
    renderer = RENDER_SCRIPT.read_text(encoding="utf-8")
    source = ASSET_ROOT / "openhcs-icon-square.svg"
    windows_icon = ASSET_ROOT / "openhcs.ico"
    macos_icon = ASSET_ROOT / "openhcs.icns"

    assert source.is_file()
    assert windows_icon.is_file()
    assert macos_icon.is_file()
    assert 'source_svg="$asset_directory/openhcs-icon-square.svg"' in renderer
    assert 'windows_icon="$asset_directory/openhcs.ico"' in renderer
    assert 'macos_icon="$asset_directory/openhcs.icns"' in renderer
    assert 'rsvg-convert --width 1024 --height 1024 "$source_svg"' in renderer
    assert 'image.save(\n        windows_destination,\n        format="ICO"' in renderer
    assert 'image.save(macos_destination, format="ICNS")' in renderer

    reserved, image_type, image_count = struct.unpack(
        "<HHH", windows_icon.read_bytes()[:6]
    )
    assert (reserved, image_type) == (0, 1)
    assert image_count >= 7

    macos_bytes = macos_icon.read_bytes()
    assert macos_bytes[:4] == b"icns"
    assert int.from_bytes(macos_bytes[4:8], "big") == len(macos_bytes)


def test_windows_installer_executable_and_window_share_the_packaged_icon() -> None:
    build = (WINDOWS_ROOT / "Build-InstallerLauncher.ps1").read_text(
        encoding="utf-8"
    )
    project = (WINDOWS_ROOT / "InstallerLauncher.csproj").read_text(
        encoding="utf-8"
    )
    launcher = (WINDOWS_ROOT / "InstallerLauncher.cs").read_text(encoding="utf-8")
    wizard = (WINDOWS_ROOT / "Install-OpenHCS.ps1").read_text(encoding="utf-8")

    assert '"openhcs",\n        "resources",\n        "assets",\n        "openhcs.ico"' in build
    assert (
        '"openhcs",\n        "resources",\n        "assets",\n'
        '        "openhcs-icon-square.png"'
    ) in build
    assert "<ApplicationIcon>OpenHCS.ico</ApplicationIcon>" in project
    assert '<EmbeddedResource Include="OpenHCS.ico">' in project
    assert "<LogicalName>OpenHCS.Installer.OpenHCS.ico</LogicalName>" in project
    assert '<EmbeddedResource Include="OpenHCS.png">' in project
    assert "<LogicalName>OpenHCS.Installer.OpenHCS.png</LogicalName>" in project
    assert '"OpenHCS.Installer.OpenHCS.ico"' in launcher
    assert '"OpenHCS.Installer.OpenHCS.png"' in launcher
    assert 'Path.Combine(\n                temporaryDirectory,\n                "OpenHCS.ico"' in launcher
    assert "ExtractEmbeddedFile(BrandIconResourceName, installerBrandIcon)" in launcher
    assert "ExtractEmbeddedFile(BrandLogoResourceName, installerBrandLogo)" in launcher
    assert "$resolvedBrandIconPath = [IO.Path]::GetFullPath($BrandIconPath)" in wizard
    assert "$resolvedBrandLogoPath = [IO.Path]::GetFullPath($BrandLogoPath)" in wizard
    assert "$installerLogo = [Drawing.Image]::FromFile($resolvedBrandLogoPath)" in wizard
    assert "$installerIcon.ToBitmap()" not in wizard
    assert "$form.Icon = $installerIcon" in wizard
    assert "$brandPicture.Image = $installerLogo" in wizard
    assert "$brandPicture.SizeMode = [Windows.Forms.PictureBoxSizeMode]::Zoom" in wizard


def test_macos_installer_bundle_and_window_share_the_packaged_icon() -> None:
    build = (MACOS_ROOT / "build-installer.sh").read_text(encoding="utf-8")
    window = (MACOS_ROOT / "OpenHCSInstaller.swift").read_text(encoding="utf-8")
    publish = PUBLISH_WORKFLOW.read_text(encoding="utf-8")
    integration = INTEGRATION_WORKFLOW.read_text(encoding="utf-8")

    assert 'openhcs/resources/assets/openhcs.icns"' in build
    assert '<key>CFBundleIconFile</key><string>OpenHCS.icns</string>' in build
    assert '"$temporary_app/Contents/Resources/OpenHCS.icns"' in build
    assert "iconView.image = NSImage(named: NSImage.applicationIconName)" in window
    assert "iconView.imageScaling = .scaleProportionallyUpOrDown" in window
    assert '"$DMG_SOURCE/.VolumeIcon.icns"' in publish
    assert 'xcrun SetFile -a C "$DMG_SOURCE"' in publish
    assert '"$dmg_source/.VolumeIcon.icns"' in integration
    assert 'xcrun SetFile -a C "$dmg_source"' in integration
    assert 'test -f "$mount_point/.VolumeIcon.icns"' in integration
    assert 'xcrun GetFileInfo -a "$mount_point" | grep -q C' in integration
