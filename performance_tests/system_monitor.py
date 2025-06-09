
"""
System monitoring script for performance testing
"""
import psutil
import time
import json
import csv
import threading
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import statistics
import numpy as np


class SystemMonitor:
    """Monitor system resources during performance tests"""
    
    def __init__(self, output_dir="performance_results", interval=1.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.interval = interval
        self.monitoring = False
        self.data = []
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start system monitoring in background thread"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.data = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("🔍 System monitoring started")
    
    def stop_monitoring(self):
        """Stop system monitoring and save data"""
        if not self.monitoring:
            return
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        self._save_data()
        print("⏹️ System monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        start_time = time.time()
        
        while self.monitoring:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # Network I/O
                net_io = psutil.net_io_counters()
                
                # Process-specific metrics (for the service)
                service_processes = []
                for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                    if 'python' in proc.info['name'].lower() or 'uvicorn' in proc.info['name'].lower():
                        service_processes.append(proc.info)
                
                data_point = {
                    'timestamp': datetime.now().isoformat(),
                    'elapsed_time': time.time() - start_time,
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_used_gb': memory.used / (1024**3),
                    'memory_available_gb': memory.available / (1024**3),
                    'disk_percent': disk.percent,
                    'disk_used_gb': disk.used / (1024**3),
                    'network_sent_mb': net_io.bytes_sent / (1024**2),
                    'network_recv_mb': net_io.bytes_recv / (1024**2),
                    'service_processes': len(service_processes),
                    'service_cpu': sum(p.get('cpu_percent', 0) for p in service_processes),
                    'service_memory': sum(p.get('memory_percent', 0) for p in service_processes)
                }
                
                self.data.append(data_point)
                time.sleep(self.interval)
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(self.interval)
    
    def _save_data(self):
        """Save monitoring data to files"""
        if not self.data:
            return
        
        # Save as JSON
        json_file = self.output_dir / "system_monitoring.json"
        with open(json_file, 'w') as f:
            json.dump(self.data, f, indent=2)
        
        # Save as CSV
        csv_file = self.output_dir / "system_monitoring.csv"
        with open(csv_file, 'w', newline='') as f:
            if self.data:
                writer = csv.DictWriter(f, fieldnames=self.data[0].keys())
                writer.writeheader()
                writer.writerows(self.data)
        
        # Generate monitoring charts
        self._generate_monitoring_charts()
        
        print(f"💾 Monitoring data saved to {json_file} and {csv_file}")
    
    def _generate_monitoring_charts(self):
        """Generate system monitoring charts"""
        if not self.data:
            return
        
        # Extract time series data
        timestamps = [datetime.fromisoformat(d['timestamp']) for d in self.data]
        elapsed_times = [d['elapsed_time'] for d in self.data]
        cpu_usage = [d['cpu_percent'] for d in self.data]
        memory_usage = [d['memory_percent'] for d in self.data]
        
        # Create monitoring chart
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('System Monitoring During Performance Tests', fontsize=16)
        
        # CPU Usage
        axes[0, 0].plot(elapsed_times, cpu_usage, 'b-', linewidth=2)
        axes[0, 0].set_title('CPU Usage Over Time')
        axes[0, 0].set_xlabel('Time (seconds)')
        axes[0, 0].set_ylabel('CPU Usage (%)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=80, color='r', linestyle='--', alpha=0.7, label='80% threshold')
        axes[0, 0].legend()
        
        # Memory Usage
        axes[0, 1].plot(elapsed_times, memory_usage, 'g-', linewidth=2)
        axes[0, 1].set_title('Memory Usage Over Time')
        axes[0, 1].set_xlabel('Time (seconds)')
        axes[0, 1].set_ylabel('Memory Usage (%)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=75, color='r', linestyle='--', alpha=0.7, label='75% threshold')
        axes[0, 1].legend()
        
        # Service-specific metrics
        service_cpu = [d['service_cpu'] for d in self.data]
        service_memory = [d['service_memory'] for d in self.data]
        
        axes[1, 0].plot(elapsed_times, service_cpu, 'purple', linewidth=2)
        axes[1, 0].set_title('Service CPU Usage')
        axes[1, 0].set_xlabel('Time (seconds)')
        axes[1, 0].set_ylabel('Service CPU (%)')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(elapsed_times, service_memory, 'orange', linewidth=2)
        axes[1, 1].set_title('Service Memory Usage')
        axes[1, 1].set_xlabel('Time (seconds)')
        axes[1, 1].set_ylabel('Service Memory (%)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "system_monitoring.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def get_summary_stats(self):
        """Get summary statistics from monitoring data"""
        if not self.data:
            return {}
        
        cpu_values = [d['cpu_percent'] for d in self.data]
        memory_values = [d['memory_percent'] for d in self.data]
        service_cpu_values = [d['service_cpu'] for d in self.data]
        service_memory_values = [d['service_memory'] for d in self.data]
        
        return {
            'duration_seconds': self.data[-1]['elapsed_time'],
            'cpu': {
                'avg': sum(cpu_values) / len(cpu_values),
                'max': max(cpu_values),
                'min': min(cpu_values)
            },
            'memory': {
                'avg': sum(memory_values) / len(memory_values),
                'max': max(memory_values),
                'min': min(memory_values)
            },
            'service_cpu': {
                'avg': sum(service_cpu_values) / len(service_cpu_values),
                'max': max(service_cpu_values)
            },
            'service_memory': {
                'avg': sum(service_memory_values) / len(service_memory_values),
                'max': max(service_memory_values)
            }
        }


class PerformanceTestWithMonitoring:
    """Enhanced performance test runner with system monitoring"""
    
    def __init__(self):
        self.monitor = SystemMonitor()
    
    def run_test_with_monitoring(self, test_function):
        """Run a test function with system monitoring"""
        print("🚀 Starting performance test with monitoring...")
        
        # Start monitoring
        self.monitor.start_monitoring()
        
        try:
            # Run the actual test
            result = test_function()
            return result
        finally:
            # Stop monitoring and get results
            self.monitor.stop_monitoring()
            
            # Print summary
            stats = self.monitor.get_summary_stats()
            if stats:
                print("\n📊 System Resource Summary:")
                print(f"   Test Duration: {stats['duration_seconds']:.1f}s")
                print(f"   CPU Usage: {stats['cpu']['avg']:.1f}% avg, {stats['cpu']['max']:.1f}% max")
                print(f"   Memory Usage: {stats['memory']['avg']:.1f}% avg, {stats['memory']['max']:.1f}% max")
                print(f"   Service CPU: {stats['service_cpu']['avg']:.1f}% avg, {stats['service_cpu']['max']:.1f}% max")
                print(f"   Service Memory: {stats['service_memory']['avg']:.1f}% avg, {stats['service_memory']['max']:.1f}% max")


# Integration example with pytest
def test_with_monitoring():
    """Example of how to integrate monitoring with tests"""
    def sample_test():
        import requests
        import time
        
        # Simulate some load
        for _ in range(10):
            requests.get("http://localhost:8000/health")
            time.sleep(0.1)
        
        return True
    
    test_runner = PerformanceTestWithMonitoring()
    return test_runner.run_test_with_monitoring(sample_test)


if __name__ == "__main__":
    # Example usage
    monitor = SystemMonitor()
    
    print("Starting system monitoring for 30 seconds...")
    monitor.start_monitoring()
    time.sleep(30)
    monitor.stop_monitoring()
    
    stats = monitor.get_summary_stats()
    print(f"Monitoring completed. Stats: {stats}")

