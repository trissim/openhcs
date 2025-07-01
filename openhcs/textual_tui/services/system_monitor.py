"""
System Monitor Core Module
Real-time system monitoring with plotext visualization
"""

import plotext as plt
import platform
import psutil
import subprocess
import time
from datetime import datetime
from collections import deque
import io
import sys

# Try to import GPU monitoring libraries
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


def is_wsl():
    """Check if running in Windows Subsystem for Linux."""
    return 'microsoft' in platform.uname().release.lower()


def get_cpu_freq_mhz():
    """Get CPU frequency in MHz, with WSL compatibility."""
    if is_wsl():
        try:
            output = subprocess.check_output(
                ['powershell.exe', '-Command',
                 'Get-CimInstance -ClassName Win32_Processor | Select-Object -ExpandProperty CurrentClockSpeed'],
                stderr=subprocess.DEVNULL
            )
            return int(output.strip())
        except Exception:
            return 0
    try:
        freq = psutil.cpu_freq()
        return int(freq.current) if freq else 0
    except Exception:
        return 0


class SystemMonitor:
    """Core system monitoring class with plotext visualization"""
    
    def __init__(self, history_length=60):
        self.history_length = history_length

        # Initialize data storage
        self.cpu_history = deque(maxlen=history_length)
        self.ram_history = deque(maxlen=history_length)
        self.gpu_history = deque(maxlen=history_length)
        self.vram_history = deque(maxlen=history_length)
        self.time_stamps = deque(maxlen=history_length)

        # Cache current metrics to avoid duplicate system calls
        self._current_metrics = {}
        
        # Initialize with zeros
        for _ in range(history_length):
            self.cpu_history.append(0)
            self.ram_history.append(0)
            self.gpu_history.append(0)
            self.vram_history.append(0)
            self.time_stamps.append(0)
    
    def update_metrics(self):
        """Update system metrics and cache current values"""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=None)
        self.cpu_history.append(cpu_percent)

        # RAM usage
        ram = psutil.virtual_memory()
        ram_percent = ram.percent
        self.ram_history.append(ram_percent)

        # Cache current metrics to avoid duplicate calls in get_metrics_dict()
        self._current_metrics = {
            'cpu_percent': cpu_percent,
            'ram_percent': ram_percent,
            'ram_used_gb': ram.used / (1024**3),
            'ram_total_gb': ram.total / (1024**3),
            'cpu_cores': psutil.cpu_count(),
            'cpu_freq_mhz': get_cpu_freq_mhz(),
        }

        # GPU usage (if available)
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    gpu_load = gpu.load * 100
                    vram_util = gpu.memoryUtil * 100
                    self.gpu_history.append(gpu_load)
                    self.vram_history.append(vram_util)

                    # Cache GPU metrics
                    self._current_metrics.update({
                        'gpu_percent': gpu_load,
                        'vram_percent': vram_util,
                        'gpu_name': gpu.name,
                        'gpu_temp': gpu.temperature,
                        'vram_used_mb': gpu.memoryUsed,
                        'vram_total_mb': gpu.memoryTotal,
                    })
                else:
                    self.gpu_history.append(0)
                    self.vram_history.append(0)
            except:
                self.gpu_history.append(0)
                self.vram_history.append(0)
        else:
            self.gpu_history.append(0)
            self.vram_history.append(0)

        # Update timestamps
        self.time_stamps.append(time.time())
    
    def get_ascii_title(self):
        """Return the OpenHCS ASCII art title"""
        return """
╔═══════════════════════════════════════════════════════════════════════╗
║  ██████╗ ██████╗ ███████╗███╗   ██╗██╗  ██╗ ██████╗███████╗         ║
║ ██╔═══██╗██╔══██╗██╔════╝████╗  ██║██║  ██║██╔════╝██╔════╝         ║
║ ██║   ██║██████╔╝█████╗  ██╔██╗ ██║███████║██║     ███████╗         ║
║ ██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║██╔══██║██║     ╚════██║         ║
║ ╚██████╔╝██║     ███████╗██║ ╚████║██║  ██║╚██████╗███████║         ║
║  ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝╚═╝  ╚═╝ ╚═════╝╚══════╝         ║
╚═══════════════════════════════════════════════════════════════════════╝"""
    
    def create_monitor_view(self, width=80, height=30):
        """Create the system monitor visualization"""
        plt.clear_figure()
        
        # Create subplots
        plt.subplots(2, 2)
        plt.plot_size(width=width, height=height)
        
        # Get current metrics for display
        current_cpu = self.cpu_history[-1] if self.cpu_history else 0
        current_ram = self.ram_history[-1] if self.ram_history else 0
        current_gpu = self.gpu_history[-1] if self.gpu_history else 0
        current_vram = self.vram_history[-1] if self.vram_history else 0
        
        # Get system info
        ram_info = psutil.virtual_memory()
        ram_used_gb = ram_info.used / (1024**3)
        ram_total_gb = ram_info.total / (1024**3)
        
        x_range = list(range(len(self.cpu_history)))
        
        # CPU Usage Plot (Top Left)
        plt.subplot(1, 1)
        plt.theme('dark')
        plt.plot(x_range, list(self.cpu_history), color='cyan')
        plt.ylim(0, 100)
        plt.title(f"CPU Usage: {current_cpu:.1f}%")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Usage %")
        
        # RAM Usage Plot (Top Right)
        plt.subplot(1, 2)
        plt.theme('dark')
        plt.plot(x_range, list(self.ram_history), color='green')
        plt.ylim(0, 100)
        plt.title(f"RAM: {ram_used_gb:.1f}/{ram_total_gb:.1f} GB ({current_ram:.1f}%)")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Usage %")
        
        # GPU Usage Plot (Bottom Left)
        plt.subplot(2, 1)
        plt.theme('dark')
        if GPU_AVAILABLE and any(self.gpu_history):
            plt.plot(x_range, list(self.gpu_history), color='yellow')
            plt.ylim(0, 100)
            plt.title(f"GPU Usage: {current_gpu:.1f}%")
        else:
            plt.plot([0, 1], [0, 0], color='red')
            plt.title("GPU: Not Available")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Usage %")
        
        # VRAM Usage Plot (Bottom Right)
        plt.subplot(2, 2)
        plt.theme('dark')
        if GPU_AVAILABLE and any(self.vram_history):
            plt.plot(x_range, list(self.vram_history), color='magenta')
            plt.ylim(0, 100)
            plt.title(f"VRAM Usage: {current_vram:.1f}%")
        else:
            plt.plot([0, 1], [0, 0], color='red')
            plt.title("VRAM: Not Available")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Usage %")
        
        # Capture the plot output
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        try:
            # Print title
            print(self.get_ascii_title())
            
            # Show plots
            plt.show()
            
            # Additional system info
            print("\n" + "═" * 75)
            print(f"System Information | Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("═" * 75)
            print(f"CPU Cores: {psutil.cpu_count()} | CPU Frequency: {get_cpu_freq_mhz()} MHz")
            print(f"Total RAM: {ram_total_gb:.1f} GB | Available RAM: {ram_info.available/(1024**3):.1f} GB")
            
            if GPU_AVAILABLE:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]
                        print(f"GPU: {gpu.name} | VRAM: {gpu.memoryUsed:.0f}/{gpu.memoryTotal:.0f} MB")
                        print(f"GPU Temperature: {gpu.temperature}°C")
                except:
                    pass
            
            # Disk usage
            disk = psutil.disk_usage('/')
            print(f"Disk Usage: {disk.used/(1024**3):.1f}/{disk.total/(1024**3):.1f} GB ({disk.percent}%)")
            
            # Network info
            net = psutil.net_io_counters()
            print(f"Network - Sent: {net.bytes_sent/(1024**2):.1f} MB | Recv: {net.bytes_recv/(1024**2):.1f} MB")
            print("═" * 75)
            
            output = buffer.getvalue()
        finally:
            sys.stdout = old_stdout
        
        return output
    
    def get_metrics_dict(self):
        """Get current metrics as a dictionary - uses cached data from update_metrics()"""
        # Return cached metrics to avoid duplicate system calls
        # If no cached data exists (first call), return defaults
        if not self._current_metrics:
            return {
                'cpu_percent': 0,
                'ram_percent': 0,
                'ram_used_gb': 0,
                'ram_total_gb': 0,
                'cpu_cores': 0,
                'cpu_freq_mhz': 0,
            }

        return self._current_metrics.copy()


# Standalone CLI usage
def main():
    """Run system monitor in standalone mode"""
    monitor = SystemMonitor()
    
    print("Starting System Monitor...")
    print("Press Ctrl+C to exit")
    
    try:
        while True:
            monitor.update_metrics()
            print("\033[2J\033[H")  # Clear screen
            print(monitor.create_monitor_view())
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting System Monitor...")


if __name__ == "__main__":
    main()