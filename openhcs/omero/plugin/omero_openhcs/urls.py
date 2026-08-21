"""
URL routing for OMERO-OpenHCS plugin.
"""

from django.urls import re_path
from . import views

# URL patterns for the plugin
# OMERO.web will include these under /webclient/
urlpatterns = [
    # Panel HTML
    re_path(r'^panel/(?P<plate_id>[0-9]+)/$',
            views.panel,
            name='openhcs_panel'),

    # Submit pipeline for processing
    re_path(r'^submit/(?P<plate_id>[0-9]+)/$',
            views.submit_pipeline,
            name='openhcs_submit'),

    # Get job status
    re_path(r'^status/(?P<execution_id>[a-f0-9-]+)/$',
            views.job_status,
            name='openhcs_status'),

    # List all jobs
    re_path(r'^jobs/$',
            views.list_jobs,
            name='openhcs_jobs'),

    # Get server status with workers
    re_path(r'^server-status/$',
            views.server_status,
            name='openhcs_server_status'),

    # Cancel job
    re_path(r'^cancel/(?P<execution_id>[a-f0-9-]+)/$',
            views.cancel_job,
            name='openhcs_cancel'),

    # Start execution server
    re_path(r'^start-server/$',
            views.start_server,
            name='openhcs_start_server'),
]
