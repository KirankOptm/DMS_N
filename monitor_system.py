#!/usr/bin/env python3
"""
System Resource Monitor for IMX Board DMS
Monitors CPU, NPU, Memory, and Temperature usage in real-time
"""

import os
import time
import psutil
from collections import deque

class SystemMonitor:
    def __init__(self):
        self.cpu_history = deque(maxlen=30)
        self.mem_history = deque(maxlen=30)
        self.npu_history = deque(maxlen=30)
        
        # NPU sysfs paths for Ethos-U on IMX
        self.npu_paths = [
            '/sys/class/misc/ethosu0/device/power_state',
            '/sys/kernel/debug/ethosu0/status',
            '/sys/devices/platform/ethos-u/npu_usage',
            '/sys/class/npu/usage',
        ]
        
        self.npu_available = self._check_npu_available()
    
    def _check_npu_available(self):
        """Check if NPU monitoring is available"""
        for path in self.npu_paths:
            if os.path.exists(path):
                print(f"[NPU] Found NPU interface at: {path}")
                return True
        
        # Try alternative: check if vela delegate is loaded
        try:
            import tflite_runtime.interpreter as tflite
            if os.path.exists('/usr/lib/libethosu_delegate.so'):
                print("[NPU] Ethos-U delegate found, NPU available")
                return True
        except:
            pass
        
        print("[NPU] NPU monitoring not available, will show CPU only")
        return False
    
    def get_npu_usage(self):
        """Get NPU usage percentage (approximate)"""
        # Method 1: Try reading from sysfs
        for path in self.npu_paths:
            try:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        content = f.read().strip()
                        # Parse content (format varies by driver)
                        if 'usage' in content.lower() or '%' in content:
                            # Extract percentage
                            import re
                            match = re.search(r'(\d+\.?\d*)%?', content)
                            if match:
                                return float(match.group(1))
            except (IOError, PermissionError):
                continue
        
        # Method 2: Estimate from process CPU time of NPU-heavy processes
        # This is approximate - NPU usage doesn't show in standard CPU metrics
        try:
            # Check if DMS process is using TFLite delegate
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
                try:
                    if 'python' in proc.info['name'].lower() and 'main_board' in ' '.join(proc.cmdline()):
                        # NPU load correlates with low CPU but active inference
                        cpu_usage = proc.cpu_percent(interval=0.1)
                        # Heuristic: if CPU is moderate, NPU is likely active
                        if 10 < cpu_usage < 50:
                            return min(cpu_usage * 1.5, 100.0)  # Estimate
                        return 0.0
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except:
            pass
        
        return 0.0  # Unknown
    
    def get_cpu_usage(self):
        """Get overall CPU usage percentage"""
        return psutil.cpu_percent(interval=0.1)
    
    def get_cpu_per_core(self):
        """Get per-core CPU usage"""
        return psutil.cpu_percent(interval=0.1, percpu=True)
    
    def get_memory_usage(self):
        """Get memory usage percentage"""
        mem = psutil.virtual_memory()
        return mem.percent
    
    def get_memory_details(self):
        """Get detailed memory info"""
        mem = psutil.virtual_memory()
        return {
            'total_mb': mem.total / (1024**2),
            'used_mb': mem.used / (1024**2),
            'available_mb': mem.available / (1024**2),
            'percent': mem.percent
        }
    
    def get_temperature(self):
        """Get CPU/NPU temperature"""
        try:
            # Try common thermal zones on IMX
            thermal_paths = [
                '/sys/class/thermal/thermal_zone0/temp',
                '/sys/class/thermal/thermal_zone1/temp',
                '/sys/devices/virtual/thermal/thermal_zone0/temp',
            ]
            
            temps = []
            for path in thermal_paths:
                try:
                    if os.path.exists(path):
                        with open(path, 'r') as f:
                            # Temperature is usually in millidegrees
                            temp = float(f.read().strip()) / 1000.0
                            temps.append(temp)
                except:
                    continue
            
            if temps:
                return max(temps)  # Return hottest zone
        except:
            pass
        
        return None
    
    def get_process_stats(self, process_name='python'):
        """Get stats for specific process"""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    if process_name.lower() in proc.info['name'].lower():
                        cmdline = proc.cmdline()
                        if any('main_board' in cmd for cmd in cmdline):
                            return {
                                'pid': proc.info['pid'],
                                'cpu_percent': proc.cpu_percent(interval=0.1),
                                'memory_percent': proc.info['memory_percent'],
                                'num_threads': proc.num_threads()
                            }
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except:
            pass
        
        return None
    
    def display_stats(self):
        """Display current system statistics"""
        # Clear screen
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("=" * 70)
        print(" IMX BOARD - DMS SYSTEM MONITOR")
        print("=" * 70)
        print()
        
        # CPU Usage
        cpu_overall = self.get_cpu_usage()
        cpu_cores = self.get_cpu_per_core()
        self.cpu_history.append(cpu_overall)
        
        print(f"CPU Usage:      {cpu_overall:5.1f}%  ", end='')
        print(self._make_bar(cpu_overall))
        
        # Show per-core
        print("CPU Cores:      ", end='')
        for i, core_usage in enumerate(cpu_cores):
            print(f"[{i}:{core_usage:4.1f}%] ", end='')
        print()
        
        # NPU Usage
        if self.npu_available:
            npu_usage = self.get_npu_usage()
            self.npu_history.append(npu_usage)
            print(f"NPU Usage:      {npu_usage:5.1f}%  ", end='')
            print(self._make_bar(npu_usage))
        else:
            print("NPU Usage:      [Not Available]")
        
        print()
        
        # Memory Usage
        mem_usage = self.get_memory_usage()
        mem_details = self.get_memory_details()
        self.mem_history.append(mem_usage)
        
        print(f"Memory Usage:   {mem_usage:5.1f}%  ", end='')
        print(self._make_bar(mem_usage))
        print(f"                {mem_details['used_mb']:.0f} MB / {mem_details['total_mb']:.0f} MB")
        print()
        
        # Temperature
        temp = self.get_temperature()
        if temp:
            print(f"Temperature:    {temp:5.1f}°C")
        else:
            print("Temperature:    [Not Available]")
        
        print()
        
        # DMS Process Stats
        dms_stats = self.get_process_stats('python')
        if dms_stats:
            print("DMS Process:")
            print(f"  PID:          {dms_stats['pid']}")
            print(f"  CPU:          {dms_stats['cpu_percent']:5.1f}%")
            print(f"  Memory:       {dms_stats['memory_percent']:5.1f}%")
            print(f"  Threads:      {dms_stats['num_threads']}")
        else:
            print("DMS Process:    [Not Running]")
        
        print()
        print("=" * 70)
        
        # Show averages
        if len(self.cpu_history) > 0:
            avg_cpu = sum(self.cpu_history) / len(self.cpu_history)
            print(f"CPU Average (30s):   {avg_cpu:5.1f}%")
        
        if len(self.mem_history) > 0:
            avg_mem = sum(self.mem_history) / len(self.mem_history)
            print(f"Memory Average (30s): {avg_mem:5.1f}%")
        
        if self.npu_available and len(self.npu_history) > 0:
            avg_npu = sum(self.npu_history) / len(self.npu_history)
            print(f"NPU Average (30s):    {avg_npu:5.1f}%")
        
        print()
        print("Press Ctrl+C to exit")
    
    def _make_bar(self, percent, width=40):
        """Create a visual bar for percentage"""
        filled = int(width * percent / 100)
        bar = '█' * filled + '░' * (width - filled)
        return f"[{bar}]"
    
    def run(self, interval=1.0):
        """Run monitoring loop"""
        print("Starting system monitor...")
        print("Monitoring DMS system resources...")
        print()
        
        try:
            while True:
                self.display_stats()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")


def main():
    monitor = SystemMonitor()
    monitor.run(interval=2.0)  # Update every 2 seconds


if __name__ == "__main__":
    main()
