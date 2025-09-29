#!/usr/bin/env python3
"""
Analysis script for IDTO parameter tuning results.
This script loads tuning results and provides statistical analysis and visualization.
"""

import pathlib
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import argparse


def load_tuning_results(results_file: str) -> Dict[str, Any]:
    """Load tuning results from pickle file."""
    with open(results_file, 'rb') as f:
        return pickle.load(f)


def results_to_dataframe(results: Dict[str, Any]) -> pd.DataFrame:
    """Convert results to pandas DataFrame for analysis."""
    all_results = results['all_results']
    
    rows = []
    for result in all_results:
        if not result.get('success', False):
            continue
            
        row = {
            'trial_id': result['trial_id'],
            'clockwise_rotation': result['clockwise_rotation'],
            'successful_cycles': result['successful_cycles'],
            'failed_cycles': result['failed_cycles'],
            'total_cycles': result['total_cycles'],
        }
        
        # Add parameter values
        param_config = result['param_config']
        for param_name, param_value in param_config.items():
            row[param_name] = param_value
            
        rows.append(row)
    
    return pd.DataFrame(rows)


def analyze_parameter_importance(df: pd.DataFrame, target_col: str = 'clockwise_rotation') -> Dict[str, float]:
    """Analyze the importance of each parameter using correlation."""
    param_cols = ['Qq_hand', 'Qq', 'Qv_hand', 'Qv', 'hand_R', 'screw_r', 'Qf_q_hand', 'Qf_q', 'Qf_v']
    
    correlations = {}
    for param in param_cols:
        if param in df.columns:
            # Use log scale for parameters since they span many orders of magnitude
            log_param = np.log10(df[param])
            corr = log_param.corr(df[target_col])
            correlations[param] = corr
    
    return correlations


def find_best_configurations(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Find the best parameter configurations based on performance."""
    param_cols = ['Qq_hand', 'Qq', 'Qv_hand', 'Qv', 'hand_R', 'screw_r', 'Qf_q_hand', 'Qf_q', 'Qf_v']
    
    # Group by parameter configuration and calculate statistics
    grouped = df.groupby(param_cols).agg({
        'clockwise_rotation': ['mean', 'std', 'count'],
        'successful_cycles': 'mean',
        'total_cycles': 'mean'
    }).reset_index()
    
    # Flatten column names
    grouped.columns = [
        '_'.join(col).strip() if col[1] else col[0] 
        for col in grouped.columns.values
    ]
    
    # Sort by mean performance
    best_configs = grouped.sort_values('clockwise_rotation_mean', ascending=False).head(top_n)
    
    return best_configs


def create_parameter_heatmap(df: pd.DataFrame, output_path: Optional[str] = None):
    """Create heatmap showing parameter correlations with performance."""
    param_cols = ['Qq_hand', 'Qq', 'Qv_hand', 'Qv', 'hand_R', 'screw_r', 'Qf_q_hand', 'Qf_q', 'Qf_v']
    
    # Create correlation matrix with log-scaled parameters
    corr_data = {}
    for param in param_cols:
        if param in df.columns:
            corr_data[param] = np.log10(df[param])
    corr_data['Performance'] = df['clockwise_rotation']
    
    corr_df = pd.DataFrame(corr_data)
    correlation_matrix = corr_df.corr()
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0,
                square=True, fmt='.3f', cbar_kws={'label': 'Correlation'})
    plt.title('Parameter Correlation Matrix (Log Scale)')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()


def create_performance_distribution(df: pd.DataFrame, output_path: Optional[str] = None):
    """Create histogram of performance distribution."""
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(df['clockwise_rotation'], bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Clockwise Rotation (radians)')
    plt.ylabel('Frequency')
    plt.title('Performance Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(df['successful_cycles'], bins=20, alpha=0.7, edgecolor='black', color='orange')
    plt.xlabel('Successful MPC Cycles')
    plt.ylabel('Frequency')
    plt.title('Successful Cycles Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()


def create_parameter_vs_performance_plots(df: pd.DataFrame, output_path: Optional[str] = None):
    """Create scatter plots of each parameter vs performance."""
    param_cols = ['Qq_hand', 'Qq', 'Qv_hand', 'Qv', 'hand_R', 'screw_r', 'Qf_q_hand', 'Qf_q', 'Qf_v']
    available_params = [p for p in param_cols if p in df.columns]
    
    n_params = len(available_params)
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols
    
    plt.figure(figsize=(15, 5 * n_rows))
    
    for i, param in enumerate(available_params, 1):
        plt.subplot(n_rows, n_cols, i)
        plt.scatter(np.log10(df[param]), df['clockwise_rotation'], alpha=0.6)
        plt.xlabel(f'log10({param})')
        plt.ylabel('Clockwise Rotation')
        plt.title(f'{param} vs Performance')
        plt.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        corr = np.corrcoef(np.log10(df[param]), df['clockwise_rotation'])[0, 1]
        plt.text(0.05, 0.95, f'r = {corr:.3f}', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()


def print_summary_statistics(df: pd.DataFrame, results: Dict[str, Any]):
    """Print summary statistics of the tuning results."""
    print("="*60)
    print("IDTO PARAMETER TUNING RESULTS SUMMARY")
    print("="*60)
    
    print(f"\nExperiment Details:")
    print(f"  Total configurations tested: {results['total_configs']}")
    print(f"  Trials per configuration: {results['num_trials_per_config']}")  
    print(f"  Total trials attempted: {results['total_trials']}")
    print(f"  Successful trials: {len(df)}")
    print(f"  Success rate: {len(df)/results['total_trials']*100:.1f}%")
    
    print(f"\nPerformance Statistics:")
    print(f"  Mean clockwise rotation: {df['clockwise_rotation'].mean():.4f} ± {df['clockwise_rotation'].std():.4f}")
    print(f"  Best performance: {df['clockwise_rotation'].max():.4f}")
    print(f"  Worst performance: {df['clockwise_rotation'].min():.4f}")
    print(f"  Median performance: {df['clockwise_rotation'].median():.4f}")
    
    print(f"\nExecution Statistics:")
    print(f"  Mean successful cycles: {df['successful_cycles'].mean():.1f} ± {df['successful_cycles'].std():.1f}")
    print(f"  Mean total cycles: {df['total_cycles'].mean():.1f} ± {df['total_cycles'].std():.1f}")
    
    # Parameter importance
    correlations = analyze_parameter_importance(df)
    print(f"\nParameter Importance (correlation with performance):")
    sorted_params = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for param, corr in sorted_params:
        print(f"  {param:12s}: {corr:6.3f}")


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description='Analyze IDTO parameter tuning results')
    parser.add_argument('results_file', help='Path to the tuning results pickle file')
    parser.add_argument('--output-dir', default='analysis_output', help='Directory to save plots')
    parser.add_argument('--no-plots', action='store_true', help='Skip generating plots')
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from: {args.results_file}")
    results = load_tuning_results(args.results_file)
    
    # Convert to DataFrame
    df = results_to_dataframe(results)
    
    if df.empty:
        print("No successful trials found in the results!")
        return
    
    # Print summary statistics
    print_summary_statistics(df, results)
    
    # Find best configurations
    print(f"\nTop 5 Parameter Configurations:")
    print("-"*60)
    best_configs = find_best_configurations(df, top_n=5)
    print(best_configs.to_string(index=False))
    
    # Generate plots if requested
    if not args.no_plots:
        output_dir = pathlib.Path(args.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"\nGenerating plots in: {output_dir}")
        
        # Performance distribution
        create_performance_distribution(df, output_dir / 'performance_distribution.png')
        
        # Parameter correlations heatmap
        create_parameter_heatmap(df, output_dir / 'parameter_correlations.png')
        
        # Parameter vs performance plots
        create_parameter_vs_performance_plots(df, output_dir / 'parameter_vs_performance.png')
        
        print("Plots saved successfully!")
    
    # Save processed data
    output_csv = pathlib.Path(args.output_dir) / 'processed_results.csv'
    df.to_csv(output_csv, index=False)
    print(f"\nProcessed data saved to: {output_csv}")


if __name__ == "__main__":
    main()


