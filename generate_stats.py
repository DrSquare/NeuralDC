"""
Descriptive Statistics Generator
Generates comprehensive descriptive statistics for CSV datasets
"""

import pandas as pd
import numpy as np

def generate_descriptive_stats(file_path, output_path=None):
    """
    Generate comprehensive descriptive statistics for a CSV file
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    output_path : str, optional
        Path to save the summary CSV file
    
    Returns:
    --------
    dict : Dictionary containing all statistics
    """
    
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Initialize results dictionary
    results = {
        'dataset_summary': {},
        'column_stats': [],
        'numeric_stats': {},
        'categorical_stats': {}
    }
    
    # Overall dataset statistics
    results['dataset_summary'] = {
        'total_observations': len(df),
        'total_variables': len(df.columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
    }
    
    print("="*80)
    print("DESCRIPTIVE STATISTICS")
    print("="*80)
    print(f"Observations: {results['dataset_summary']['total_observations']:,}")
    print(f"Variables: {results['dataset_summary']['total_variables']}")
    print(f"Memory: {results['dataset_summary']['memory_usage_mb']:.2f} MB")
    print()
    
    # Column-level statistics
    for col in df.columns:
        col_stats = {
            'column': col,
            'observations': len(df),
            'non_null': df[col].notna().sum(),
            'null': df[col].isna().sum(),
            'null_pct': (df[col].isna().sum() / len(df)) * 100,
            'unique': df[col].nunique(),
            'dtype': str(df[col].dtype)
        }
        
        # Numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            col_stats.update({
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'q25': df[col].quantile(0.25),
                'median': df[col].median(),
                'q75': df[col].quantile(0.75),
                'max': df[col].max()
            })
            results['numeric_stats'][col] = col_stats
        else:
            # Categorical columns
            top_5 = df[col].value_counts().head(5)
            col_stats['top_values'] = {
                val: {'count': count, 'pct': (count / df[col].notna().sum()) * 100}
                for val, count in top_5.items()
            }
            results['categorical_stats'][col] = col_stats
        
        results['column_stats'].append(col_stats)
    
    # Save summary to CSV if requested
    if output_path:
        summary_df = pd.DataFrame(results['column_stats'])
        summary_df.to_csv(output_path, index=False)
        print(f"Summary saved to: {output_path}")
    
    return results


def print_numeric_summary(results):
    """Print summary of numeric variables"""
    print("\nNUMERIC VARIABLES SUMMARY")
    print("-"*80)
    
    if results['numeric_stats']:
        for col, stats in results['numeric_stats'].items():
            print(f"\n{col}:")
            print(f"  Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
            print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            print(f"  Quartiles: Q1={stats['q25']:.4f}, Median={stats['median']:.4f}, Q3={stats['q75']:.4f}")
    else:
        print("No numeric variables found.")


def print_categorical_summary(results, top_n=5):
    """Print summary of categorical variables"""
    print("\n\nCATEGORICAL VARIABLES SUMMARY (Top 5 values)")
    print("-"*80)
    
    for col, stats in results['categorical_stats'].items():
        if stats['unique'] > 1:  # Skip constant columns
            print(f"\n{col} ({stats['unique']} unique values):")
            for i, (val, info) in enumerate(stats['top_values'].items(), 1):
                print(f"  {i}. '{val}': {info['count']} ({info['pct']:.2f}%)")


# Example usage
if __name__ == "__main__":
    # Generate statistics
    results = generate_descriptive_stats(
        '/mnt/user-data/uploads/prod_saltsnck.csv',
        '/mnt/user-data/outputs/summary_stats.csv'
    )
    
    # Print summaries
    print_numeric_summary(results)
    print_categorical_summary(results)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)
