"""
Django views for OMERO-OpenHCS plugin.

These views handle communication between OMERO.web and the OpenHCS execution server.
"""

import json
import logging
import pickle
import zmq
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_http_methods
from omeroweb.webclient.decorators import login_required

logger = logging.getLogger(__name__)


class SimpleZMQClient:
    """
    Lightweight ZMQ client for communicating with OpenHCS execution server.

    This avoids needing to install the full OpenHCS package in OMERO.web.
    """

    def __init__(self, host='localhost', port=7777):
        self.host = host
        self.data_port = port
        self.control_port = port + 1000  # Control port is data port + 1000

    def _send_request(self, request):
        """Send request to server and get response."""
        try:
            context = zmq.Context()
            socket = context.socket(zmq.REQ)
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout
            socket.connect(f"tcp://{self.host}:{self.control_port}")

            socket.send(pickle.dumps(request))
            response = socket.recv()
            response_data = pickle.loads(response)

            socket.close()
            context.term()

            return response_data
        except Exception as e:
            logger.error(f"ZMQ request failed: {e}")
            return {'status': 'error', 'message': str(e)}

    def ping(self):
        """Ping server to check if alive."""
        response = self._send_request({'type': 'ping'})
        return response.get('type') == 'pong' and response.get('ready')

    def get_server_info(self):
        """Get detailed server info including workers."""
        return self._send_request({'type': 'ping'})

    def get_status(self, execution_id=None):
        """Get execution status."""
        request = {'type': 'status'}
        if execution_id:
            request['execution_id'] = execution_id
        return self._send_request(request)

    def execute_pipeline(self, plate_id, pipeline_code, config_code):
        """Execute pipeline on server."""
        request = {
            'type': 'execute',
            'plate_id': str(plate_id),
            'pipeline_code': pipeline_code,
            'config_code': config_code
        }
        return self._send_request(request)

    def cancel(self, execution_id):
        """Cancel execution."""
        request = {
            'type': 'cancel',
            'execution_id': execution_id
        }
        return self._send_request(request)


def _get_zmq_client():
    """
    Get ZMQ client instance.

    Reads server config from environment variables:
    - OPENHCS_EXECUTION_HOST: Execution server host (default: host.docker.internal for Docker)
    - OPENHCS_EXECUTION_PORT: Execution server port (default: 7777)
    """
    import os

    # Use host.docker.internal by default for Docker deployments
    # This allows the container to reach services on the host machine
    server_host = os.getenv('OPENHCS_EXECUTION_HOST', 'host.docker.internal')
    server_port = int(os.getenv('OPENHCS_EXECUTION_PORT', '7777'))

    return SimpleZMQClient(host=server_host, port=server_port)


@login_required()
@require_http_methods(["GET"])
def panel(request, plate_id, conn=None, **kwargs):
    """
    Render the OpenHCS panel HTML for a plate.

    Args:
        request: Django request object
        plate_id: OMERO plate ID
        conn: OMERO connection (provided by @login_required decorator)

    Returns:
        Rendered HTML template
    """
    return render(request, 'omero_openhcs/panel.html', {
        'plate_id': plate_id
    })


@login_required()
@require_http_methods(["POST"])
def submit_pipeline(request, plate_id, conn=None, **kwargs):
    """
    Submit OpenHCS pipeline for a plate.

    Args:
        request: Django request object
        plate_id: OMERO plate ID
        conn: OMERO connection (provided by @login_required decorator)

    Returns:
        JSON response with execution_id
    """
    try:
        # Parse request body
        data = json.loads(request.body)
        pipeline_code = data.get('pipeline_code')
        config_code = data.get('config_code')

        if not pipeline_code:
            return JsonResponse({
                'status': 'error',
                'message': 'Missing pipeline_code'
            }, status=400)

        # If no config provided, use default (auto-detection will handle OMERO-specific settings)
        if not config_code:
            config_code = 'from openhcs.core.config import GlobalPipelineConfig\nconfig = GlobalPipelineConfig()'

        # Get ZMQ client
        client = _get_zmq_client()

        # Submit to execution server
        response = client.execute_pipeline(
            plate_id=int(plate_id),
            pipeline_code=pipeline_code,
            config_code=config_code
        )

        logger.info(f"Submitted pipeline for plate {plate_id}: {response}")

        return JsonResponse(response)

    except Exception as e:
        logger.error(f"Error submitting pipeline: {e}", exc_info=True)
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)


@login_required()
@require_http_methods(["GET"])
def job_status(request, execution_id, conn=None, **kwargs):
    """
    Get status of a specific job.

    Args:
        request: Django request object
        execution_id: Execution ID from submit_pipeline
        conn: OMERO connection (provided by @login_required decorator)

    Returns:
        JSON response with job status
    """
    try:
        client = _get_zmq_client()
        status = client.get_status(execution_id)

        return JsonResponse(status)

    except Exception as e:
        logger.error(f"Error getting job status: {e}", exc_info=True)
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)


@login_required()
@require_http_methods(["GET"])
def list_jobs(request, conn=None, **kwargs):
    """
    List all jobs on the execution server.

    Args:
        conn: OMERO connection (provided by @login_required decorator)

    Returns:
        JSON response with list of jobs
    """
    try:
        client = _get_zmq_client()
        status = client.get_status()  # No execution_id = server status

        return JsonResponse(status)

    except Exception as e:
        logger.error(f"Error listing jobs: {e}", exc_info=True)
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)


@login_required()
@require_http_methods(["GET"])
def server_status(request, conn=None, **kwargs):
    """
    Get detailed server status including worker processes.

    Args:
        conn: OMERO connection (provided by @login_required decorator)

    Returns:
        JSON response with server info, workers, active executions
    """
    try:
        client = _get_zmq_client()
        server_info = client.get_server_info()

        return JsonResponse(server_info)

    except Exception as e:
        logger.error(f"Error getting server status: {e}", exc_info=True)
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)


@login_required()
@require_http_methods(["POST"])
def cancel_job(request, execution_id, conn=None, **kwargs):
    """
    Cancel a running job.

    Args:
        request: Django request object
        execution_id: Execution ID to cancel
        conn: OMERO connection (provided by @login_required decorator)

    Returns:
        JSON response with cancellation status
    """
    try:
        client = _get_zmq_client()
        response = client.cancel(execution_id)

        return JsonResponse(response)

    except Exception as e:
        logger.error(f"Error cancelling job: {e}", exc_info=True)
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)


@login_required()
@require_http_methods(["POST"])
def start_server(request, conn=None, **kwargs):
    """
    Start the ZMQ execution server.

    Args:
        request: Django request object
        conn: OMERO connection (provided by @login_required decorator)

    Returns:
        JSON response with server start status
    """
    try:
        import subprocess
        import sys
        from pathlib import Path
        import time

        # Check if server is already running
        client = _get_zmq_client()
        if client.ping():
            return JsonResponse({
                'status': 'success',
                'message': 'Server is already running'
            })

        # Start the server as a background process
        log_dir = Path.home() / ".local" / "share" / "openhcs" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = log_dir / f"openhcs_zmq_server_port_7777_{int(time.time() * 1000000)}.log"

        cmd = [
            sys.executable, '-m',
            'openhcs.runtime.zmq_execution_server_launcher',
            '--port', '7777',
            '--persistent',
            '--log-file-path', str(log_file_path)
        ]

        # Start the server process detached
        subprocess.Popen(
            cmd,
            stdout=open(log_file_path, 'w'),
            stderr=subprocess.STDOUT,
            start_new_session=True
        )

        logger.info(f"Started ZMQ execution server on port 7777, log: {log_file_path}")

        return JsonResponse({
            'status': 'success',
            'message': 'Server started successfully',
            'log_file': str(log_file_path)
        })

    except Exception as e:
        logger.error(f"Error starting server: {e}", exc_info=True)
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)
