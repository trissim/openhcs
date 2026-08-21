"""Environment declarations shared by UI bridge producers and clients."""

from __future__ import annotations


class UiBridgeDescriptorEnvironment:
    """Own the environment selectors for one live UI bridge descriptor."""

    descriptor_file_path_key = "OPENHCS_UI_BRIDGE_DESCRIPTOR"
    descriptor_directory_path_key = "OPENHCS_UI_BRIDGE_DESCRIPTOR_DIR"

    @classmethod
    def child_process_environment_keys(cls) -> tuple[str, str]:
        """Return every selector required by a descriptor-consuming child."""

        return (
            cls.descriptor_file_path_key,
            cls.descriptor_directory_path_key,
        )
