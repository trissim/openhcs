from textual.app import App, ComposeResult
from textual.widgets import Static, Header, Footer
from textual.containers import Container, Horizontal, Vertical, Grid
from typing import Optional
from textual import events, work
from textual.reactive import reactive
from textual.worker import get_current_worker
import asyncio
import psutil
from collections import deque
import time

# Use textual-plotext for proper Textual integration
try:
    from textual_plotext import PlotextPlot
    PLOTEXT_AVAILABLE = True
except ImportError:
    PLOTEXT_AVAILABLE = False

# Import the SystemMonitor from services
from openhcs.textual_tui.services.system_monitor import SystemMonitor

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

class SystemMonitorTextual(Container):
    """A Textual widget for system monitoring using proper Textual components"""

    DEFAULT_CSS = """
    SystemMonitorTextual {
        background: $background;
        color: $text;
    }

    #top_row {
        height: 9;
        background: $surface;
    }

    #header {
        width: 1fr;
        text-align: left;
        color: $accent;
        padding: 1;
    }

    #system_info {
        width: 1fr;
        color: $text-muted;
        text-align: right;
        padding: 1;
        content-align: center middle;
    }

    #plots_grid {
        height: 1fr;
        grid-size: 2 2;
        grid-gutter: 1;
    }

    #fallback_display {
        height: 1fr;
        text-align: center;
        color: $primary;
    }


    """

    update_interval = reactive(1.0)
    is_monitoring = reactive(True)

    def __init__(self):
        super().__init__()
        self.monitor = SystemMonitor()
        self.monitor_worker = None

    def compose(self) -> ComposeResult:
        """Compose the system monitor layout"""
        # Top row with title on left and system info on right
        with Horizontal(id="top_row"):
            yield Static(self._get_ascii_header(), id="header")
            yield Static("", id="system_info")

        if PLOTEXT_AVAILABLE:
            # Use Grid layout for plots
            with Grid(id="plots_grid"):
                yield PlotextPlot(id="cpu_plot")
                yield PlotextPlot(id="ram_plot")
                yield PlotextPlot(id="gpu_plot")
                yield PlotextPlot(id="vram_plot")
        else:
            # Fallback to text-based display
            yield Static("", id="fallback_display")

    async def on_mount(self) -> None:
        """Start the monitoring when widget is mounted"""
        # Start the worker using the @work decorated method
        self.monitor_worker = self._monitor_worker()
        self.is_monitoring = True

    def on_unmount(self) -> None:
        """Stop monitoring and cleanup background thread when widget is unmounted"""
        self._stop_monitoring()

    def _start_monitoring(self) -> None:
        """Start the monitoring Textual worker"""
        # Stop any existing monitoring
        self._stop_monitoring()

        # Start Textual worker using @work decorated method
        self.monitor_worker = self._monitor_worker()
        self.is_monitoring = True

    def _stop_monitoring(self) -> None:
        """Stop the monitoring worker"""
        if self.monitor_worker and not self.monitor_worker.is_finished:
            self.monitor_worker.cancel()
        self.monitor_worker = None
        self.is_monitoring = False

    def toggle_monitoring(self) -> None:
        """Toggle monitoring on/off - called from menu"""
        if self.is_monitoring:
            self._stop_monitoring()
        else:
            self._start_monitoring()

    async def manual_refresh(self) -> None:
        """Manual refresh - called from menu"""
        await self.update_display()

    @work(thread=True, exclusive=True)
    def _monitor_worker(self) -> None:
        """Textual worker that collects system metrics and updates UI."""
        worker = get_current_worker()
        import logging
        logger = logging.getLogger(__name__)
        logger.info("SystemMonitor worker started")

        while not worker.is_cancelled:
            try:
                # Collect system metrics
                self.monitor.update_metrics()

                # Update UI via call_from_thread every cycle (1 second updates)
                if not worker.is_cancelled:
                    self.app.call_from_thread(self._update_ui_from_thread)

                # Sleep for the update interval (default 1 second)
                time.sleep(self.update_interval)

            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"Error in system monitor worker: {e}")
                time.sleep(2.0)  # Longer sleep on error

    def _update_ui_from_thread(self) -> None:
        """Update UI components - called from background thread via call_from_thread."""
        if PLOTEXT_AVAILABLE:
            self._update_plots()
        else:
            self._update_fallback_display()

        self._update_system_info()

    def _get_ascii_header(self) -> str:
        """Get the OpenHCS ASCII art header"""
        return """
╔══════════════════════════════════════════════════════════════╗
║  ██████╗ ██████╗ ███████╗███╗   ██╗██╗  ██╗ ██████╗███████╗  ║
║ ██╔═══██╗██╔══██╗██╔════╝████╗  ██║██║  ██║██╔════╝██╔════╝  ║
║ ██║   ██║██████╔╝█████╗  ██╔██╗ ██║███████║██║     ███████╗  ║
║ ██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║██╔══██║██║     ╚════██║  ║
║ ╚██████╔╝██║     ███████╗██║ ╚████║██║  ██║╚██████╗███████║  ║
║  ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝╚═╝  ╚═╝ ╚═════╝╚══════╝  ║
╚══════════════════════════════════════════════════════════════╝"""

    async def update_display(self) -> None:
        """Update the display with new system stats - for manual refresh only"""
        # Update metrics using the SystemMonitor from services - wrap in executor to avoid blocking
        import asyncio
        await asyncio.get_event_loop().run_in_executor(None, self.monitor.update_metrics)

        # Update UI directly (this is for manual refresh)
        self._update_ui_from_thread()

    def _update_plots(self) -> None:
        """Update the plotext plots"""
        try:
            # Get current metrics
            current_cpu = self.monitor.cpu_history[-1] if self.monitor.cpu_history else 0
            current_ram = self.monitor.ram_history[-1] if self.monitor.ram_history else 0
            current_gpu = self.monitor.gpu_history[-1] if self.monitor.gpu_history else 0
            current_vram = self.monitor.vram_history[-1] if self.monitor.vram_history else 0

            x_range = list(range(len(self.monitor.cpu_history)))

            # Update CPU plot with Braille patterns for higher resolution
            cpu_plot = self.query_one("#cpu_plot", PlotextPlot)
            cpu_plot.plt.clear_data()
            cpu_plot.plt.plot(x_range, list(self.monitor.cpu_history), marker="braille")
            cpu_plot.plt.title(f"CPU Usage: {current_cpu:.1f}%")
            cpu_plot.plt.ylim(0, 100)
            cpu_plot.refresh()  # Force visual refresh

            # Update RAM plot with Braille patterns for higher resolution
            ram_plot = self.query_one("#ram_plot", PlotextPlot)
            ram_plot.plt.clear_data()
            ram_plot.plt.plot(x_range, list(self.monitor.ram_history), marker="braille")
            ram_plot.plt.title(f"RAM Usage: {current_ram:.1f}%")
            ram_plot.plt.ylim(0, 100)
            ram_plot.refresh()  # Force visual refresh

            # Update GPU plot with Braille patterns for higher resolution
            gpu_plot = self.query_one("#gpu_plot", PlotextPlot)
            gpu_plot.plt.clear_data()
            if GPU_AVAILABLE and any(self.monitor.gpu_history):
                gpu_plot.plt.plot(x_range, list(self.monitor.gpu_history), marker="braille")
                gpu_plot.plt.title(f"GPU Usage: {current_gpu:.1f}%")
            else:
                gpu_plot.plt.title("GPU: Not Available")
            gpu_plot.plt.ylim(0, 100)
            gpu_plot.refresh()  # Force visual refresh

            # Update VRAM plot with Braille patterns for higher resolution
            vram_plot = self.query_one("#vram_plot", PlotextPlot)
            vram_plot.plt.clear_data()
            if GPU_AVAILABLE and any(self.monitor.vram_history):
                vram_plot.plt.plot(x_range, list(self.monitor.vram_history), marker="braille")
                vram_plot.plt.title(f"VRAM Usage: {current_vram:.1f}%")
            else:
                vram_plot.plt.title("VRAM: Not Available")
            vram_plot.plt.ylim(0, 100)
            vram_plot.refresh()  # Force visual refresh

        except Exception as e:
            # If plots fail, fall back to text display
            self._update_fallback_display()

    def _update_fallback_display(self) -> None:
        """Update with text-based display if plots aren't available"""
        try:
            metrics = self.monitor.get_metrics_dict()

            display_text = f"""
┌─────────────────────────────────────────────────────────────────┐
│ CPU:  {self._create_bar(metrics.get('cpu_percent', 0))} {metrics.get('cpu_percent', 0):5.1f}%
│ RAM:  {self._create_bar(metrics.get('ram_percent', 0))} {metrics.get('ram_percent', 0):5.1f}% ({metrics.get('ram_used_gb', 0):.1f}/{metrics.get('ram_total_gb', 0):.1f}GB)
│ GPU:  {self._create_bar(metrics.get('gpu_percent', 0))} {metrics.get('gpu_percent', 0):5.1f}%
│ VRAM: {self._create_bar(metrics.get('vram_percent', 0))} {metrics.get('vram_percent', 0):5.1f}%
└─────────────────────────────────────────────────────────────────┘
"""

            fallback_widget = self.query_one("#fallback_display", Static)
            fallback_widget.update(display_text)
        except Exception:
            pass

    def _update_system_info(self) -> None:
        """Update system information in top-right"""
        try:
            from datetime import datetime
            # Now get_metrics_dict() just returns cached data - no blocking calls!
            metrics = self.monitor.get_metrics_dict()

            info_text = f"""
═══════════════════════════════════════════════════════════════════════
System Information | Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
═══════════════════════════════════════════════════════════════════════
CPU Cores: {metrics.get('cpu_cores', 'N/A')} | CPU Frequency: {metrics.get('cpu_freq_mhz', 0):.0f} MHz
Total RAM: {metrics.get('ram_total_gb', 0):.1f} GB | Used RAM: {metrics.get('ram_used_gb', 0):.1f} GB"""

            if GPU_AVAILABLE and 'gpu_name' in metrics:
                info_text += f"\nGPU: {metrics.get('gpu_name', 'N/A')} | Temperature: {metrics.get('gpu_temp', 'N/A')}°C"
                info_text += f"\nVRAM: {metrics.get('vram_used_mb', 0):.0f}/{metrics.get('vram_total_mb', 0):.0f} MB"

            info_text += "\n═══════════════════════════════════════════════════════════════════════"

            info_widget = self.query_one("#system_info", Static)
            info_widget.update(info_text)
        except Exception:
            pass

    def _create_bar(self, percent):
        """Create a text-based progress bar"""
        bar_length = 20
        filled = int(bar_length * percent / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        return f"[{bar}]"

