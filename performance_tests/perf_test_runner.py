
#!/usr/bin/env python3
"""
Performance Test Runner and Reporter for Face Recognition Service

This script runs various performance tests and generates comprehensive reports.
"""
import os
import sys
import json
import time
import subprocess
import argparse
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any


class PerformanceTestRunner:
    """Main class for running performance tests and generating reports"""
    
    def __init__(self, service_url: str = "http://localhost:8000", output_dir: str = "performance_results"):
        self.service_url = service_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Cấu hình logging với mã hóa UTF-8
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'test_run.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Test results storage
        self.results = {}
        self.start_time = datetime.now()
    
    def check_service_availability(self) -> bool:
        """Check if the service is running and accessible"""
        try:
            import requests
            response = requests.get(f"{self.service_url}/health", timeout=10)
            if response.status_code == 200:
                self.logger.info("✅ Service is available and healthy")
                return True
            else:
                self.logger.error(f"❌ Service returned status code: {response.status_code}")
                return False
        except Exception as e:
            self.logger.error(f"❌ Service is not accessible: {e}")
            return False
    
    def run_pytest_tests(self) -> Dict[str, Any]:
        """Run pytest-based performance tests"""
        self.logger.info("🧪 Running pytest performance tests...")
        
        # Create pytest command
        pytest_cmd = [
            "python", "-m", "pytest", 
            "test_performance.py",
            "-v", 
            "--tb=short",
            f"--html={self.output_dir}/pytest_report.html",
            "--self-contained-html",
            "--benchmark-json={}/benchmark_results.json".format(self.output_dir)
        ]
        
        try:
            # Run pytest
            start_time = time.time()
            result = subprocess.run(
                pytest_cmd,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes timeout
            )
            end_time = time.time()
            
            pytest_results = {
                'duration': end_time - start_time,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'success': result.returncode == 0
            }
            
            self.logger.info(f"✅ Pytest tests completed in {pytest_results['duration']:.2f}s")
            return pytest_results
            
        except subprocess.TimeoutExpired:
            self.logger.error("❌ Pytest tests timed out")
            return {'success': False, 'error': 'Timeout'}
        except Exception as e:
            self.logger.error(f"❌ Error running pytest: {e}")
            return {'success': False, 'error': str(e)}
    
    def run_locust_test(self, users: int = 10, spawn_rate: int = 2, run_time: str = "60s") -> Dict[str, Any]:
        """Run Locust load test"""
        self.logger.info(f"🔥 Running Locust load test (users: {users}, spawn-rate: {spawn_rate}, time: {run_time})...")
        
        # Create locust command
        locust_cmd = [
            "locust",
            "-f", "locust_load_test.py",
            "--host", self.service_url,
            "--users", str(users),
            "--spawn-rate", str(spawn_rate),
            "--run-time", run_time,
            "--headless",
            "--csv", str(self.output_dir / "locust_results"),
            "--html", str(self.output_dir / "locust_report.html")
        ]
        
        try:
            start_time = time.time()
            result = subprocess.run(
                locust_cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            end_time = time.time()
            
            locust_results = {
                'duration': end_time - start_time,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'success': result.returncode == 0
            }
            
            self.logger.info(f"✅ Locust test completed in {locust_results['duration']:.2f}s")
            return locust_results
            
        except subprocess.TimeoutExpired:
            self.logger.error("❌ Locust test timed out")
            return {'success': False, 'error': 'Timeout'}
        except Exception as e:
            self.logger.error(f"❌ Error running Locust: {e}")
            return {'success': False, 'error': str(e)}
    
    def generate_performance_charts(self):
        """Generate performance analysis charts"""
        self.logger.info("📊 Generating performance charts...")
        
        try:
            # Set up plotting style
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
            # Try to load and analyze Locust results
            self._generate_locust_charts()
            
            # Try to load and analyze benchmark results
            self._generate_benchmark_charts()
            
            self.logger.info("✅ Performance charts generated successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Error generating charts: {e}")
    
    def _generate_locust_charts(self):
        """Generate charts from Locust CSV results"""
        stats_file = self.output_dir / "locust_results_stats.csv"
        history_file = self.output_dir / "locust_results_stats_history.csv"
        
        if stats_file.exists():
            # Load Locust stats
            stats_df = pd.read_csv(stats_file)
            
            # Response time distribution
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Locust Load Test Results', fontsize=16)
            
            # Request statistics
            axes[0, 0].bar(stats_df['Name'], stats_df['Request Count'])
            axes[0, 0].set_title('Request Count by Endpoint')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Response times
            axes[0, 1].bar(stats_df['Name'], stats_df['Average Response Time'])
            axes[0, 1].set_title('Average Response Time (ms)')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Failure rate
            if 'Failure Count' in stats_df.columns:
                failure_rate = (stats_df['Failure Count'] / stats_df['Request Count'] * 100).fillna(0)
                axes[1, 0].bar(stats_df['Name'], failure_rate)
                axes[1, 0].set_title('Failure Rate (%)')
                axes[1, 0].tick_params(axis='x', rotation=45)
            
            # RPS
            if 'Requests/s' in stats_df.columns:
                axes[1, 1].bar(stats_df['Name'], stats_df['Requests/s'])
                axes[1, 1].set_title('Requests per Second')
                axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "locust_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        if history_file.exists():
            history_df = pd.read_csv(history_file)
            self.logger.info(f"Các cột trong history_df: {history_df.columns.tolist()}")  # Logging debug tùy chọn
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            fig.suptitle('Locust Performance Timeline', fontsize=16)
            axes[0].plot(history_df['Timestamp'], history_df['Total Average Response Time'])
            axes[0].set_title('Thời gian phản hồi trung bình theo thời gian')
            axes[0].set_ylabel('Thời gian phản hồi (ms)')
            axes[1].plot(history_df['Timestamp'], history_df['Requests/s'])
            axes[1].set_title('Yêu cầu mỗi giây theo thời gian')
            axes[1].set_ylabel('RPS')
            plt.tight_layout()
            plt.savefig(self.output_dir / "locust_timeline.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def _generate_benchmark_charts(self):
        """Generate charts from benchmark JSON results"""
        benchmark_file = self.output_dir / "benchmark_results.json"
        
        if benchmark_file.exists():
            with open(benchmark_file, 'r') as f:
                benchmark_data = json.load(f)
            
            # Extract benchmark results
            benchmarks = benchmark_data.get('benchmarks', [])
            
            if benchmarks:
                # Create DataFrame for analysis
                df_data = []
                for bench in benchmarks:
                    df_data.append({
                        'test': bench['name'],
                        'min': bench['stats']['min'],
                        'max': bench['stats']['max'],
                        'mean': bench['stats']['mean'],
                        'median': bench['stats']['median']
                    })
                
                df = pd.DataFrame(df_data)
                
                # Create benchmark visualization
                fig, axes = plt.subplots(2, 1, figsize=(12, 10))
                fig.suptitle('Benchmark Results', fontsize=16)
                
                # Response time comparison
                x_pos = range(len(df))
                axes[0].bar([i-0.2 for i in x_pos], df['min'], 0.2, label='Min', alpha=0.7)
                axes[0].bar(x_pos, df['mean'], 0.2, label='Mean', alpha=0.7)
                axes[0].bar([i+0.2 for i in x_pos], df['max'], 0.2, label='Max', alpha=0.7)
                axes[0].set_xlabel('Test Cases')
                axes[0].set_ylabel('Time (seconds)')
                axes[0].set_title('Response Time Distribution')
                axes[0].set_xticks(x_pos)
                axes[0].set_xticklabels(df['test'], rotation=45, ha='right')
                axes[0].legend()
                
                # Box plot for distribution
                axes[1].boxplot([df['min'], df['mean'], df['median'], df['max']], 
                               labels=['Min', 'Mean', 'Median', 'Max'])
                axes[1].set_ylabel('Time (seconds)')
                axes[1].set_title('Overall Performance Distribution')
                
                plt.tight_layout()
                plt.savefig(self.output_dir / "benchmark_analysis.png", dpi=300, bbox_inches='tight')
                plt.close()
    
    def generate_report(self):
        """Generate comprehensive HTML report"""
        self.logger.info("📄 Generating comprehensive report...")
        
        report_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Face Recognition Service - Performance Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .metric {{ background-color: #f9f9f9; padding: 10px; margin: 5px 0; border-left: 4px solid #007cba; }}
        .success {{ border-left-color: #28a745; }}
        .warning {{ border-left-color: #ffc107; }}
        .error {{ border-left-color: #dc3545; }}
        .chart {{ text-align: center; margin: 20px 0; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 Face Recognition Service - Performance Test Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Service URL:</strong> {self.service_url}</p>
        <p><strong>Test Duration:</strong> {(datetime.now() - self.start_time).total_seconds():.1f} seconds</p>
    </div>
    
    <div class="section">
        <h2>📊 Test Results Summary</h2>
        {self._generate_results_summary()}
    </div>
    
    <div class="section">
        <h2>📈 Performance Charts</h2>
        {self._generate_charts_section()}
    </div>
    
    <div class="section">
        <h2>🔧 Recommendations</h2>
        {self._generate_recommendations()}
    </div>
    
    <div class="section">
        <h2>📋 Detailed Results</h2>
        {self._generate_detailed_results()}
    </div>
</body>
</html>
        """
        
        with open(self.output_dir / "performance_report.html", 'w', encoding='utf-8') as f:
            f.write(report_html)
    
        self.logger.info(f"✅ Báo cáo đã tạo: {self.output_dir / 'performance_report.html'}")
    
    def _generate_results_summary(self) -> str:
        """Generate HTML summary of results"""
        summary = ""
        
        for test_type, results in self.results.items():
            status_class = "success" if results.get('success', False) else "error"
            summary += f"""
            <div class="metric {status_class}">
                <strong>{test_type.title()}</strong>: 
                {'✅ Passed' if results.get('success', False) else '❌ Failed'}
                <br>Duration: {results.get('duration', 0):.2f}s
            </div>
            """
        
        return summary
    
    def _generate_charts_section(self) -> str:
        """Generate HTML section for charts"""
        charts_html = ""
        
        chart_files = [
            ("locust_analysis.png", "Locust Load Test Analysis"),
            ("locust_timeline.png", "Locust Performance Timeline"),
            ("benchmark_analysis.png", "Benchmark Results")
        ]
        
        for filename, title in chart_files:
            chart_path = self.output_dir / filename
            if chart_path.exists():
                charts_html += f"""
                <div class="chart">
                    <h3>{title}</h3>
                    <img src="{filename}" alt="{title}" style="max-width: 100%; height: auto;">
                </div>
                """
        
        return charts_html if charts_html else "<p>No charts generated.</p>"
    
    def _generate_recommendations(self) -> str:
        """Generate performance recommendations"""
        recommendations = [
            "🚀 <strong>Caching:</strong> Implement Redis caching for frequently accessed data",
            "⚡ <strong>Async Processing:</strong> Use background tasks for heavy computations",
            "🔄 <strong>Load Balancing:</strong> Consider horizontal scaling with multiple instances",
            "📊 <strong>Monitoring:</strong> Set up continuous monitoring with Prometheus/Grafana",
            "🗃️ <strong>Database:</strong> Optimize database queries and consider indexing",
            "🖼️ <strong>Image Processing:</strong> Implement image compression and optimization"
        ]
        
        return "<ul>" + "".join(f"<li>{rec}</li>" for rec in recommendations) + "</ul>"
    
    def _generate_detailed_results(self) -> str:
        """Generate detailed results section"""
        return "<pre>" + json.dumps(self.results, indent=2) + "</pre>"
    
    def run_full_test_suite(self, **kwargs):
        """Run complete performance test suite"""
        self.logger.info("🚀 Starting full performance test suite...")
        
        # Check service availability
        if not self.check_service_availability():
            self.logger.error("❌ Service not available. Aborting tests.")
            return False
        
        # Run pytest tests
        self.results['pytest'] = self.run_pytest_tests()
        
        # Run Locust tests
        locust_config = {
            'users': kwargs.get('users', 10),
            'spawn_rate': kwargs.get('spawn_rate', 2),
            'run_time': kwargs.get('run_time', '60s')
        }
        self.results['locust'] = self.run_locust_test(**locust_config)
        
        # Generate charts and reports
        self.generate_performance_charts()
        self.generate_report()
        
        # Summary
        total_duration = (datetime.now() - self.start_time).total_seconds()
        self.logger.info(f"🎉 Performance test suite completed in {total_duration:.2f}s")
        
        # Print summary
        print("\n" + "="*60)
        print("PERFORMANCE TEST SUMMARY")
        print("="*60)
        for test_type, results in self.results.items():
            status = "✅ PASSED" if results.get('success', False) else "❌ FAILED"
            print(f"{test_type.upper()}: {status} ({results.get('duration', 0):.2f}s)")
        print("="*60)
        print(f"📊 Full report: {self.output_dir / 'performance_report.html'}")
        print("="*60)
        
        return all(r.get('success', False) for r in self.results.values())


def main():
    """Main entry point for the performance test runner"""
    parser = argparse.ArgumentParser(description="Face Recognition Service Performance Test Runner")
    parser.add_argument('--url', default='http://localhost:8000', help='Service URL')
    parser.add_argument('--output', default='performance_results', help='Output directory')
    parser.add_argument('--users', type=int, default=10, help='Number of concurrent users for load test')
    parser.add_argument('--spawn-rate', type=int, default=2, help='User spawn rate for load test')
    parser.add_argument('--run-time', default='60s', help='Load test duration')
    parser.add_argument('--pytest-only', action='store_true', help='Run only pytest tests')
    parser.add_argument('--locust-only', action='store_true', help='Run only Locust tests')
    
    args = parser.parse_args()
    
    # Create test runner
    runner = PerformanceTestRunner(
        service_url=args.url,
        output_dir=args.output
    )
    
    # Run tests based on arguments
    if args.pytest_only:
        runner.results['pytest'] = runner.run_pytest_tests()
    elif args.locust_only:
        runner.results['locust'] = runner.run_locust_test(
            users=args.users,
            spawn_rate=args.spawn_rate,
            run_time=args.run_time
        )
    else:
        # Run full suite
        success = runner.run_full_test_suite(
            users=args.users,
            spawn_rate=args.spawn_rate,
            run_time=args.run_time
        )
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
