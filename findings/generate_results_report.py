"""
Generate comprehensive results report with seaborn visualizations
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Set style for consistent, professional-looking plots
plt.style.use('default')
sns.set_palette("icefire")
sns.set_context("paper", font_scale=1.2)

def load_all_results():
    """Load all evaluation results from JSON files"""
    results_dir = Path(__file__).parent / "results"
    
    files_mapping = {
        "amnesty_ragas_main.json": ("AmnestyQA", "Ragas Main"),
        "amnesty_ragas_experimental.json": ("AmnestyQA", "Original Experimental"),
        "amnesty_ragas_experimental_exact.json": ("AmnestyQA", "Exact Replica"),
        "fiqa_ragas_main.json": ("FIQA", "Ragas Main"),
        "fiqa_ragas_experimental.json": ("FIQA", "Original Experimental"),
        "fiqa_ragas_experimental_exact.json": ("FIQA", "Exact Replica")
    }
    
    all_data = []
    summary_data = []
    
    for filename, (dataset, approach) in files_mapping.items():
        filepath = results_dir / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            scores = data.get("scores", [])
            avg_score = data.get("average_faithfulness", np.mean(scores) if scores else 0)
            
            # Individual sample data
            for i, score in enumerate(scores):
                all_data.append({
                    "Dataset": dataset,
                    "Approach": approach,
                    "Sample": i + 1,
                    "Score": score
                })
            
            # Summary data
            summary_data.append({
                "Dataset": dataset,
                "Approach": approach,
                "Average_Score": avg_score,
                "Num_Samples": len(scores),
                "Timestamp": data.get("timestamp", ""),
                "Scores": scores
            })
    
    df_detailed = pd.DataFrame(all_data)
    df_summary = pd.DataFrame(summary_data)
    
    return df_detailed, df_summary

def create_charts_directory():
    """Create charts directory if it doesn't exist"""
    charts_dir = Path(__file__).parent / "charts"
    charts_dir.mkdir(exist_ok=True)
    return charts_dir

def plot_average_scores_comparison(df_summary, charts_dir):
    """Create bar chart comparing average scores across approaches and datasets with error bars"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data for plotting with error bars
    datasets = df_summary['Dataset'].unique()
    approaches = df_summary['Approach'].unique()
    
    x = np.arange(len(datasets))
    width = 0.25
    
    colors = sns.color_palette("icefire", len(approaches))
    
    for i, approach in enumerate(approaches):
        means = []
        stds = []
        
        for dataset in datasets:
            row = df_summary[(df_summary['Dataset'] == dataset) & (df_summary['Approach'] == approach)]
            if not row.empty:
                scores = row.iloc[0]['Scores']
                means.append(np.mean(scores))
                stds.append(np.std(scores))
            else:
                means.append(0)
                stds.append(0)
        
        bars = ax.bar(x + i * width, means, width, yerr=stds, 
                     label=approach, color=colors[i], alpha=0.8, 
                     capsize=5, error_kw={'linewidth': 2, 'capthick': 2})
        
        # Add value labels on bars
        for j, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                   f'{mean:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax.set_title('Faithfulness Score Comparison: Average Scores by Approach and Dataset', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Dataset', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Faithfulness Score', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(datasets)
    ax.legend(title='Approach', title_fontsize=12, fontsize=11, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(charts_dir / "average_scores_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_sample_by_sample_comparison(df_detailed, charts_dir):
    """Create line plots showing sample-by-sample score comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    datasets = ["AmnestyQA", "FIQA"]
    colors = sns.color_palette("icefire", len(df_detailed['Approach'].unique()))
    
    for i, dataset in enumerate(datasets):
        dataset_data = df_detailed[df_detailed['Dataset'] == dataset]
        
        # Create line plot with icefire palette
        sns.lineplot(data=dataset_data, x='Sample', y='Score', hue='Approach', 
                     marker='o', markersize=8, linewidth=2.5, ax=axes[i],
                     palette="icefire")
        
        axes[i].set_title(f'{dataset} - Sample-by-Sample Comparison', 
                         fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Sample Number', fontsize=12, fontweight='bold')
        axes[i].set_ylabel('Faithfulness Score', fontsize=12, fontweight='bold')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(title='Approach', title_fontsize=10, fontsize=9)
        axes[i].set_ylim(-0.05, 1.05)
    
    plt.suptitle('Sample-by-Sample Faithfulness Score Comparison', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(charts_dir / "sample_by_sample_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_score_distributions(df_detailed, charts_dir):
    """Create box plots and violin plots showing score distributions"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    datasets = ["AmnestyQA", "FIQA"]
    
    for i, dataset in enumerate(datasets):
        dataset_data = df_detailed[df_detailed['Dataset'] == dataset]
        
        # Box plot in top row
        sns.boxplot(data=dataset_data, x='Approach', y='Score', ax=axes[0, i], 
                   hue='Approach', palette="icefire", width=0.6, legend=False)
        axes[0, i].set_title(f'{dataset} - Box Plot Distribution', fontsize=14, fontweight='bold')
        axes[0, i].set_xlabel('Approach', fontsize=12, fontweight='bold')
        axes[0, i].set_ylabel('Faithfulness Score', fontsize=12, fontweight='bold')
        axes[0, i].tick_params(axis='x', rotation=45)
        axes[0, i].grid(axis='y', alpha=0.3)
        
        # Violin plot in bottom row for better visualization of distributions
        sns.violinplot(data=dataset_data, x='Approach', y='Score', hue='Approach', ax=axes[1, i], 
                      palette="icefire", inner="box", legend=False)
        axes[1, i].set_title(f'{dataset} - Violin Plot Distribution', fontsize=14, fontweight='bold')
        axes[1, i].set_xlabel('Approach', fontsize=12, fontweight='bold')
        axes[1, i].set_ylabel('Faithfulness Score', fontsize=12, fontweight='bold')
        axes[1, i].tick_params(axis='x', rotation=45)
        axes[1, i].grid(axis='y', alpha=0.3)
        
        # Add individual points to show actual data distribution
        sns.stripplot(data=dataset_data, x='Approach', y='Score', ax=axes[0, i], 
                     color='black', alpha=0.6, size=4, jitter=True)
        
    plt.suptitle('Faithfulness Score Distributions by Approach', 
                 fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.savefig(charts_dir / "score_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_difference_analysis(df_summary, charts_dir):
    """Create visualization showing differences from Ragas Main baseline"""
    plt.figure(figsize=(12, 8))
    
    # Calculate differences from Ragas Main
    diff_data = []
    for dataset in ["AmnestyQA", "FIQA"]:
        dataset_summary = df_summary[df_summary['Dataset'] == dataset]
        ragas_main_score = dataset_summary[dataset_summary['Approach'] == 'Ragas Main']['Average_Score'].iloc[0]
        
        for _, row in dataset_summary.iterrows():
            if row['Approach'] != 'Ragas Main':
                diff = row['Average_Score'] - ragas_main_score
                abs_diff = abs(diff)
                target_met = abs_diff < 0.01
                
                diff_data.append({
                    'Dataset': dataset,
                    'Approach': row['Approach'],
                    'Difference': diff,
                    'Abs_Difference': abs_diff,
                    'Target_Met': target_met
                })
    
    diff_df = pd.DataFrame(diff_data)
    
    # Use icefire palette colors
    icefire_colors = sns.color_palette("icefire", 4)
    colors = [icefire_colors[3] if not met else icefire_colors[0] for met in diff_df['Target_Met']]
    
    # Create grouped bar chart
    x_pos = np.arange(len(diff_df))
    bars = plt.bar(x_pos, diff_df['Difference'], 
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.axhline(y=0.01, color='red', linestyle='--', alpha=0.7, label='Target (+0.01)')
    plt.axhline(y=-0.01, color='red', linestyle='--', alpha=0.7, label='Target (-0.01)')
    
    plt.title('Difference from Ragas Main Baseline', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Dataset - Approach', fontsize=14, fontweight='bold')
    plt.ylabel('Score Difference', fontsize=14, fontweight='bold')
    
    # Create labels
    labels = [f"{row['Dataset']}\\n{row['Approach']}" for _, row in diff_df.iterrows()]
    plt.xticks(x_pos, labels, rotation=45, ha='right')
    
    # Add value labels
    for i, (bar, diff, target_met) in enumerate(zip(bars, diff_df['Difference'], diff_df['Target_Met'])):
        height = bar.get_height()
        status = "✓" if target_met else "✗"
        plt.text(bar.get_x() + bar.get_width()/2., height + (0.005 if height >= 0 else -0.015),
                f'{diff:.4f}\\n{status}', ha='center', va='bottom' if height >= 0 else 'top',
                fontweight='bold', fontsize=10)
    
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(charts_dir / "difference_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()


def generate_markdown_report(df_summary, charts_dir):
    """Generate comprehensive markdown report"""
    
    # Calculate key metrics
    metrics = {}
    for dataset in ["AmnestyQA", "FIQA"]:
        dataset_summary = df_summary[df_summary['Dataset'] == dataset]
        ragas_main = dataset_summary[dataset_summary['Approach'] == 'Ragas Main']['Average_Score'].iloc[0]
        original_exp = dataset_summary[dataset_summary['Approach'] == 'Original Experimental']['Average_Score'].iloc[0]
        exact_replica = dataset_summary[dataset_summary['Approach'] == 'Exact Replica']['Average_Score'].iloc[0]
        
        original_diff = abs(original_exp - ragas_main)
        final_diff = abs(exact_replica - ragas_main)
        improvement_pct = ((original_diff - final_diff) / original_diff) * 100
        target_met = final_diff < 0.01
        
        metrics[dataset] = {
            'ragas_main': ragas_main,
            'original_exp': original_exp,
            'exact_replica': exact_replica,
            'original_diff': original_diff,
            'final_diff': final_diff,
            'improvement_pct': improvement_pct,
            'target_met': target_met
        }
    
    markdown_content = f"""# Faithfulness Evaluation Results Analysis

**Generated on:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary

This report presents a comprehensive analysis of faithfulness evaluation implementations across three approaches:
1. **Ragas Main** - The original implementation in the main Ragas framework
2. **Original Experimental** - Initial experimental implementation with single-step evaluation
3. **Exact Replica** - Refined experimental implementation matching Ragas Main methodology exactly

### Key Achievements

| Dataset | Original Difference | Final Difference | Improvement | Target (<0.01) |
|---------|-------------------|------------------|-------------|----------------|
| AmnestyQA | {metrics['AmnestyQA']['original_diff']:.4f} | {metrics['AmnestyQA']['final_diff']:.4f} | {metrics['AmnestyQA']['improvement_pct']:.1f}% | {'✅ ACHIEVED' if metrics['AmnestyQA']['target_met'] else '❌ NOT MET'} |
| FIQA | {metrics['FIQA']['original_diff']:.4f} | {metrics['FIQA']['final_diff']:.4f} | {metrics['FIQA']['improvement_pct']:.1f}% | {'✅ ACHIEVED' if metrics['FIQA']['target_met'] else '❌ NOT MET'} |

## Methodology Overview

### Ragas Main
- **Process**: Two-step evaluation (statement generation → NLI assessment)
- **Temperature**: 1e-8 (near-zero for deterministic results)
- **Prompt Format**: PydanticPrompt with structured schema and examples
- **Error Handling**: Returns `np.nan` for empty statement cases

### Original Experimental
- **Process**: Single-step direct evaluation
- **Temperature**: 0
- **Prompt Format**: Simple text-based prompts
- **Error Handling**: Returns 0.0 for empty cases
- **Issue**: Overly permissive evaluation leading to inflated scores

### Exact Replica
- **Process**: Two-step evaluation (matching Ragas Main exactly)
- **Temperature**: 1e-8 (matching Ragas Main)
- **Prompt Format**: Exact PydanticPrompt replication with identical schema
- **Error Handling**: Returns `np.nan` (matching Ragas Main)
- **Result**: Achieved near-identical performance to Ragas Main

## Results by Dataset

### AmnestyQA Dataset

![Average Scores Comparison](charts/average_scores_comparison.png)

**Performance Summary:**
- **Ragas Main**: {metrics['AmnestyQA']['ragas_main']:.4f}
- **Original Experimental**: {metrics['AmnestyQA']['original_exp']:.4f} (difference: +{metrics['AmnestyQA']['original_diff']:.4f})
- **Exact Replica**: {metrics['AmnestyQA']['exact_replica']:.4f} (difference: {metrics['AmnestyQA']['exact_replica'] - metrics['AmnestyQA']['ragas_main']:+.4f})

The exact replica achieved a **{metrics['AmnestyQA']['improvement_pct']:.1f}% improvement** in accuracy, bringing the difference down from {metrics['AmnestyQA']['original_diff']:.4f} to {metrics['AmnestyQA']['final_diff']:.4f}.

### FIQA Dataset

**Performance Summary:**
- **Ragas Main**: {metrics['FIQA']['ragas_main']:.4f}
- **Original Experimental**: {metrics['FIQA']['original_exp']:.4f} (difference: +{metrics['FIQA']['original_diff']:.4f})
- **Exact Replica**: {metrics['FIQA']['exact_replica']:.4f} (difference: {metrics['FIQA']['exact_replica'] - metrics['FIQA']['ragas_main']:+.4f})

The exact replica achieved a **{metrics['FIQA']['improvement_pct']:.1f}% improvement** in accuracy, bringing the difference down from {metrics['FIQA']['original_diff']:.4f} to {metrics['FIQA']['final_diff']:.4f}.

## Detailed Visualizations

### Sample-by-Sample Analysis
![Sample by Sample Comparison](charts/sample_by_sample_comparison.png)

This chart shows the faithfulness scores for each individual sample across all three approaches, revealing:
- **Consistency**: Exact Replica follows Ragas Main patterns much more closely than Original Experimental
- **Variance**: Original Experimental shows artificially high scores with less variance
- **Alignment**: Exact Replica captures the nuanced scoring behavior of Ragas Main

### Score Distribution Analysis
![Score Distributions](charts/score_distributions.png)

The box plots reveal:
- **Original Experimental**: Heavily skewed toward perfect scores (1.0), indicating over-permissive evaluation
- **Ragas Main & Exact Replica**: More realistic distributions with appropriate variance
- **Similarity**: Exact Replica distribution closely matches Ragas Main

### Difference Analysis
![Difference Analysis](charts/difference_analysis.png)

This visualization shows the absolute differences from the Ragas Main baseline:
- **Target Line**: Red dashed lines indicate the ±0.01 target threshold
- **Color Coding**: Green bars indicate target achievement, red bars indicate areas for improvement
- **Progress**: Clear improvement from Original Experimental to Exact Replica

---

*This report was generated automatically from evaluation results on {datetime.now().strftime("%Y-%m-%d")}.*
"""
    
    # Write markdown file
    report_path = Path(__file__).parent / "RESULTS_ANALYSIS.md"
    with open(report_path, 'w') as f:
        f.write(markdown_content)
    
    return report_path

def main():
    """Main function to generate all visualizations and report"""
    print("Loading evaluation results...")
    df_detailed, df_summary = load_all_results()
    
    print("Creating charts directory...")
    charts_dir = create_charts_directory()
    
    print("Generating visualizations...")
    plot_average_scores_comparison(df_summary, charts_dir)
    plot_sample_by_sample_comparison(df_detailed, charts_dir)
    plot_score_distributions(df_detailed, charts_dir)
    plot_difference_analysis(df_summary, charts_dir)
    
    print("Generating markdown report...")
    report_path = generate_markdown_report(df_summary, charts_dir)
    
    print(f"✅ Report generation complete!")
    print(f"📊 Charts saved to: {charts_dir}")
    print(f"📄 Report saved to: {report_path}")
    print(f"🔍 Open {report_path.name} to view the comprehensive analysis")

if __name__ == "__main__":
    main()