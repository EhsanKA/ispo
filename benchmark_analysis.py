#!/usr/bin/env python3
"""
Benchmark Analysis and Visualization for In-Silico Perturbation Optimization

This script analyzes the results from various optimization methods and creates
visualizations comparing performance metrics.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import warnings
warnings.filterwarnings("ignore")

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")


class ISPOBenchmarkAnalyzer:
    """Analyzer for In-Silico Perturbation Optimization benchmarks."""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        self.benchmark_df = None

    def load_benchmark_data(self) -> pd.DataFrame:
        """Load benchmark results from CSV file."""
        benchmark_file = os.path.join(self.results_dir, "comprehensive_benchmark.csv")
        if not os.path.exists(benchmark_file):
            raise FileNotFoundError(f"Benchmark file not found: {benchmark_file}")

        self.benchmark_df = pd.read_csv(benchmark_file)
        return self.benchmark_df

    def create_performance_summary(self) -> Dict:
        """Create a performance summary with improvements."""
        if self.benchmark_df is None:
            self.load_benchmark_data()

        baseline = self.benchmark_df[self.benchmark_df['method'] == 'baseline'].iloc[0]

        summary = {
            'baseline_time': baseline['total_time_seconds'],
            'baseline_throughput': baseline['throughput_samples_per_sec'],
            'methods': []
        }

        for _, row in self.benchmark_df.iterrows():
            if row['method'] != 'baseline':
                improvement_time = (baseline['total_time_seconds'] - row['total_time_seconds']) / baseline['total_time_seconds'] * 100
                improvement_throughput = (row['throughput_samples_per_sec'] - baseline['throughput_samples_per_sec']) / baseline['throughput_samples_per_sec'] * 100

                summary['methods'].append({
                    'method': row['method'],
                    'time_seconds': row['total_time_seconds'],
                    'throughput': row['throughput_samples_per_sec'],
                    'time_improvement_percent': improvement_time,
                    'throughput_improvement_percent': improvement_throughput,
                    'speedup_factor': baseline['total_time_seconds'] / row['total_time_seconds']
                })

        return summary

    def plot_performance_comparison(self, save_path: str = None):
        """Create performance comparison plots."""
        if self.benchmark_df is None:
            self.load_benchmark_data()

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Color mapping for methods
        method_colors = {
            'baseline': '#1f77b4',
            'batching_bs32': '#ff7f0e',
            'mixed_precision_fp16': '#2ca02c',
            'quantization_8bit': '#d62728'
        }

        colors = [method_colors.get(method, '#7f7f7f') for method in self.benchmark_df['method']]

        # Plot 1: Total Time Comparison
        bars1 = ax1.bar(self.benchmark_df['method'], self.benchmark_df['total_time_seconds'], color=colors, alpha=0.7)
        ax1.set_title('Total Inference Time Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Time (seconds)', fontsize=12)
        ax1.set_xlabel('Optimization Method', fontsize=12)
        ax1.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    '.2f', ha='center', va='bottom', fontsize=10)

        # Plot 2: Throughput Comparison
        bars2 = ax2.bar(self.benchmark_df['method'], self.benchmark_df['throughput_samples_per_sec'], color=colors, alpha=0.7)
        ax2.set_title('Throughput Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Samples per Second', fontsize=12)
        ax2.set_xlabel('Optimization Method', fontsize=12)
        ax2.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    '.1f', ha='center', va='bottom', fontsize=10)

        # Plot 3: CPU Usage
        cpu_data = self.benchmark_df[['method', 'avg_cpu_percent', 'max_cpu_percent']].set_index('method')
        cpu_data.plot(kind='bar', ax=ax3, color=['#87CEEB', '#4682B4'], alpha=0.7)
        ax3.set_title('CPU Usage Comparison', fontsize=14, fontweight='bold')
        ax3.set_ylabel('CPU Usage (%)', fontsize=12)
        ax3.set_xlabel('Optimization Method', fontsize=12)
        ax3.legend(['Average', 'Maximum'], loc='upper right')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Memory Usage
        memory_data = self.benchmark_df[['method', 'avg_memory_percent', 'max_memory_percent']].set_index('method')
        memory_data.plot(kind='bar', ax=ax4, color=['#98FB98', '#32CD32'], alpha=0.7)
        ax4.set_title('Memory Usage Comparison', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Memory Usage (%)', fontsize=12)
        ax4.set_xlabel('Optimization Method', fontsize=12)
        ax4.legend(['Average', 'Maximum'], loc='upper right')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance comparison plot saved to {save_path}")

        plt.show()

    def plot_improvement_analysis(self, save_path: str = None):
        """Create improvement analysis plot."""
        summary = self.create_performance_summary()

        if not summary['methods']:
            print("No optimization methods found to compare.")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        methods = [m['method'] for m in summary['methods']]
        time_improvements = [m['time_improvement_percent'] for m in summary['methods']]
        throughput_improvements = [m['throughput_improvement_percent'] for m in summary['methods']]
        speedup_factors = [m['speedup_factor'] for m in summary['methods']]

        # Plot improvement percentages
        x = np.arange(len(methods))
        width = 0.35

        bars1 = ax1.bar(x - width/2, time_improvements, width, label='Time Reduction', color='#ff7f0e', alpha=0.7)
        bars2 = ax1.bar(x + width/2, throughput_improvements, width, label='Throughput Increase', color='#2ca02c', alpha=0.7)

        ax1.set_title('Performance Improvements vs Baseline', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Improvement (%)', fontsize=12)
        ax1.set_xlabel('Optimization Method', fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(methods, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    '.1f', ha='center', va='bottom', fontsize=9)

        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    '.1f', ha='center', va='bottom', fontsize=9)

        # Plot speedup factors
        bars3 = ax2.bar(methods, speedup_factors, color='#1f77b4', alpha=0.7)
        ax2.set_title('Speedup Factors vs Baseline', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Speedup Factor (x)', fontsize=12)
        ax2.set_xlabel('Optimization Method', fontsize=12)
        ax2.set_xticklabels(methods, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)

        # Add value labels
        for bar in bars3:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    '.2f', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Improvement analysis plot saved to {save_path}")

        plt.show()

    def generate_report(self) -> str:
        """Generate a comprehensive text report."""
        if self.benchmark_df is None:
            self.load_benchmark_data()

        summary = self.create_performance_summary()

        report = []
        report.append("=" * 80)
        report.append("IN-SILICO PERTURBATION OPTIMIZATION BENCHMARK REPORT")
        report.append("=" * 80)
        report.append("")

        report.append("EXPERIMENTAL SETUP:")
        report.append("- Model: Geneformer (gf-6L-10M-i2048)")
        report.append("- Dataset: SciPlex2 perturbation data (subset)")
        report.append(f"- Number of perturbations: {self.benchmark_df['num_perturbations'].iloc[0]}")
        report.append("- Device: CUDA GPU")
        report.append("")

        report.append("BASELINE PERFORMANCE:")
        baseline = self.benchmark_df[self.benchmark_df['method'] == 'baseline'].iloc[0]
        report.append(".2f")
        report.append(".2f")
        report.append(".1f")
        report.append(".1f")
        report.append("")

        report.append("OPTIMIZATION RESULTS:")
        report.append("-" * 40)

        for method_info in summary['methods']:
            report.append(f"\nMethod: {method_info['method'].replace('_', ' ').title()}")
            report.append(".2f")
            report.append(".2f")
            report.append("+.1f")
            report.append("+.1f")
            report.append(".2f")

        report.append("")
        report.append("TECHNICAL DETAILS:")
        report.append("- Batching: Increased batch size from 10 to 32 samples")
        report.append("- Mixed Precision: Automatic mixed precision (FP16) using torch.cuda.amp")
        report.append("- Quantization: Attempted but not supported by current Geneformer implementation")
        report.append("")

        report.append("RECOMMENDATIONS:")
        best_method = max(summary['methods'], key=lambda x: x['speedup_factor'])
        report.append(f"- Best performing method: {best_method['method'].replace('_', ' ').title()}")
        report.append(".2f")
        report.append("- For large-scale ISP experiments, use combined optimizations")
        report.append("- Monitor GPU memory usage when scaling batch sizes")
        report.append("")

        return "\n".join(report)

    def save_embeddings_comparison(self):
        """Compare embeddings from different methods for consistency."""
        import numpy as np

        embeddings_files = {}
        for method in self.benchmark_df['method'].unique():
            emb_file = os.path.join(self.results_dir, method.replace('batching_', '').replace('_', '_'), 'embeddings.npz')
            if os.path.exists(emb_file):
                embeddings_files[method] = np.load(emb_file)['embeddings']

        if len(embeddings_files) < 2:
            print("Need at least 2 embedding sets for comparison")
            return

        # Compare embeddings between methods
        methods_list = list(embeddings_files.keys())
        baseline_emb = embeddings_files.get('baseline', embeddings_files[methods_list[0]])

        print("\nEMBEDDINGS CONSISTENCY ANALYSIS:")
        print("-" * 40)

        for method, embeddings in embeddings_files.items():
            if method == 'baseline':
                continue

            # Calculate cosine similarity between embeddings
            similarities = []
            for i in range(min(len(baseline_emb), len(embeddings))):
                vec1 = baseline_emb[i] / np.linalg.norm(baseline_emb[i])
                vec2 = embeddings[i] / np.linalg.norm(embeddings[i])
                similarity = np.dot(vec1, vec2)
                similarities.append(similarity)

            avg_similarity = np.mean(similarities)
            std_similarity = np.std(similarities)

            print(f"{method} vs baseline:")
            print(".4f")
            print(".4f")


def main():
    """Main function to run benchmark analysis."""

    analyzer = ISPOBenchmarkAnalyzer()

    try:
        # Load data
        benchmark_df = analyzer.load_benchmark_data()
        print("Loaded benchmark data:")
        print(benchmark_df.to_string(index=False))
        print()

        # Create performance plots
        print("Generating performance comparison plots...")
        analyzer.plot_performance_comparison(save_path="results/performance_comparison.png")

        print("Generating improvement analysis plots...")
        analyzer.plot_improvement_analysis(save_path="results/improvement_analysis.png")

        # Generate and save report
        report = analyzer.generate_report()
        with open("results/benchmark_report.txt", "w") as f:
            f.write(report)

        print("Benchmark report saved to results/benchmark_report.txt")
        print("\n" + report)

        # Analyze embeddings consistency
        analyzer.save_embeddings_comparison()

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure to run the optimization benchmarks first.")


if __name__ == "__main__":
    main()




