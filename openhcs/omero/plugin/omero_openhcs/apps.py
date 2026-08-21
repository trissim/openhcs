"""
Django app configuration for OMERO-OpenHCS plugin.
"""

from django.apps import AppConfig


class OpenHCSConfig(AppConfig):
    """
    App configuration for OpenHCS plugin.

    This registers the plugin with OMERO.web.
    OMERO.web automatically includes URLs from apps in omero.web.apps
    at /<app_label>/ (e.g., /omero_openhcs/).
    """
    name = 'omero_openhcs'
    label = 'omero_openhcs'
    verbose_name = "OMERO OpenHCS"
